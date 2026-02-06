#!/usr/bin/env swift
import Foundation
import CoreML

struct Half {
    static func floatToHalfBits(_ value: Float) -> UInt16 {
        let f = value.bitPattern
        let sign = UInt16((f >> 16) & 0x8000)
        let exp = Int((f >> 23) & 0xFF) - 127
        let mant = f & 0x7FFFFF

        if exp > 15 { return sign | 0x7C00 }
        if exp < -14 {
            if exp < -24 { return sign }
            let m = mant | 0x800000
            let shift = UInt32(-exp - 14 + 13)
            let halfMant = UInt16(m >> shift)
            return sign | halfMant
        }
        let halfExp = UInt16((exp + 15) << 10)
        let halfMant = UInt16(mant >> 13)
        return sign | halfExp | halfMant
    }
}

func makeInput(shape: [Int], dtype: MLMultiArrayDataType) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: dtype)
    let count = arr.count
    switch dtype {
    case .float16:
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        var seed: UInt64 = 0x12345678
        for i in 0..<count {
            seed = 1664525 &* seed &+ 1013904223
            let f = Float(Int(seed & 0xFFFF) - 0x8000) / 32768.0
            ptr[i] = Half.floatToHalfBits(f)
        }
    case .float32:
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
        var seed: UInt64 = 0x12345678
        for i in 0..<count {
            seed = 1664525 &* seed &+ 1013904223
            let f = Float(Int(seed & 0xFFFF) - 0x8000) / 32768.0
            ptr[i] = f
        }
    default:
        fatalError("Unsupported dtype: \(dtype)")
    }
    return arr
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    let sorted = values.sorted()
    if sorted.isEmpty { return 0 }
    let idx = Int(Double(sorted.count - 1) * p)
    return sorted[idx]
}

func inferredShape(from input: MLFeatureDescription, defaultShape: [Int]) -> [Int] {
    guard let constraint = input.multiArrayConstraint, !constraint.shape.isEmpty else {
        return defaultShape
    }
    return constraint.shape.map { Int(truncating: $0) }
}

func run(modelURL: URL,
         computeUnits: MLComputeUnits,
         inputHW: Int,
         latentChannels: Int,
         dtype: MLMultiArrayDataType,
         warmup: Int,
         iters: Int,
         useConstraint: Bool,
         layout: String) throws {
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits
    let model = try MLModel(contentsOf: modelURL, configuration: config)

    guard let input = model.modelDescription.inputDescriptionsByName.first else {
        fatalError("No inputs found")
    }
    let inputName = input.key
    let latHW = inputHW / 8
    let defaultShape: [Int]
    if layout == "nhwc" {
        defaultShape = [1, latHW, latHW, latentChannels]
    } else {
        defaultShape = [1, latentChannels, latHW, latHW]
    }
    let shape = useConstraint ? inferredShape(from: input.value, defaultShape: defaultShape) : defaultShape
    let inputArray = try makeInput(shape: shape, dtype: dtype)
    let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])

    print("input: name=\(inputName) shape=\(shape) dtype=\(dtype)")

    for _ in 0..<warmup { _ = try model.prediction(from: provider) }

    var times: [Double] = []
    times.reserveCapacity(iters)
    for _ in 0..<iters {
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try model.prediction(from: provider)
        let t1 = CFAbsoluteTimeGetCurrent()
        times.append((t1 - t0) * 1000.0)
    }

    let avg = times.reduce(0, +) / Double(times.count)
    let med = percentile(times, 0.5)
    let p10 = percentile(times, 0.1)
    let p90 = percentile(times, 0.9)
    let minv = times.min() ?? 0
    let maxv = times.max() ?? 0

    print(String(format: "avg ms: %.3f", avg))
    print(String(format: "median ms: %.3f", med))
    print(String(format: "p10 ms: %.3f", p10))
    print(String(format: "p90 ms: %.3f", p90))
    print(String(format: "min ms: %.3f", minv))
    print(String(format: "max ms: %.3f", maxv))
}

func usage() {
    print("Usage: run_coreml_instruments.swift <model.mlmodelc> [--compute-units cpu|cpu_ne|all] [--input-hw 768] [--latent-channels 16] [--dtype float16|float32] [--warmup 3] [--iters 200] [--use-constraint] [--layout nchw|nhwc]")
}

let args = CommandLine.arguments
if args.count < 2 {
    usage()
    exit(1)
}

let modelURL = URL(fileURLWithPath: args[1])
var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
var inputHW = 768
var latentChannels = 16
var dtype: MLMultiArrayDataType = .float16
var warmup = 3
var iters = 200
var useConstraint = false
var layout = "nchw"

var i = 2
while i < args.count {
    switch args[i] {
    case "--compute-units":
        i += 1
        if i >= args.count { break }
        switch args[i] {
        case "cpu": computeUnits = .cpuOnly
        case "cpu_ne": computeUnits = .cpuAndNeuralEngine
        case "all": computeUnits = .all
        default: break
        }
    case "--input-hw":
        i += 1
        if i < args.count { inputHW = Int(args[i]) ?? inputHW }
    case "--latent-channels":
        i += 1
        if i < args.count { latentChannels = Int(args[i]) ?? latentChannels }
    case "--dtype":
        i += 1
        if i < args.count { dtype = (args[i] == "float32") ? .float32 : .float16 }
    case "--warmup":
        i += 1
        if i < args.count { warmup = Int(args[i]) ?? warmup }
    case "--iters":
        i += 1
        if i < args.count { iters = Int(args[i]) ?? iters }
    case "--use-constraint":
        useConstraint = true
    case "--layout":
        i += 1
        if i < args.count { layout = args[i] }
    default:
        break
    }
    i += 1
}

do {
    try run(modelURL: modelURL,
            computeUnits: computeUnits,
            inputHW: inputHW,
            latentChannels: latentChannels,
            dtype: dtype,
            warmup: warmup,
            iters: iters,
            useConstraint: useConstraint,
            layout: layout)
} catch {
    print("Error: \(error)")
    exit(1)
}
