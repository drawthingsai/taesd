import Foundation
import CoreML

struct BinTensor {
    let data: Data
    let count: Int
}

func loadBin(_ path: String) throws -> BinTensor {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    let count = data.count / MemoryLayout<Float>.size
    return BinTensor(data: data, count: count)
}

func makeMLMultiArray(from tensor: BinTensor, shape: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    tensor.data.withUnsafeBytes { buf in
        if let base = buf.baseAddress {
            array.dataPointer.copyMemory(from: base, byteCount: tensor.data.count)
        }
    }
    return array
}

func readOutput(from array: MLMultiArray) -> [Float] {
    let shape = array.shape.map { $0.intValue }
    let strides = array.strides.map { $0.intValue }
    precondition(shape.count == 4, "Expected 4D output")
    let count = array.count
    var out = [Float](repeating: 0, count: count)

    func valueAt(_ offset: Int) -> Float {
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            return ptr[offset]
        case .float16:
            let ptr = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            return Float(Float16(bitPattern: ptr[offset]))
        default:
            fatalError("Unsupported output data type: \(array.dataType)")
        }
    }

    let n = shape[0], h = shape[1], w = shape[2], c = shape[3]
    var idx = 0
    for i0 in 0..<n {
        let o0 = i0 * strides[0]
        for i1 in 0..<h {
            let o1 = o0 + i1 * strides[1]
            for i2 in 0..<w {
                let o2 = o1 + i2 * strides[2]
                for i3 in 0..<c {
                    let offset = o2 + i3 * strides[3]
                    out[idx] = valueAt(offset)
                    idx += 1
                }
            }
        }
    }
    return out
}

func compare(_ a: [Float], _ b: [Float]) -> (maxAbs: Float, mae: Float, rmse: Float) {
    precondition(a.count == b.count)
    var maxAbs: Float = 0
    var sumAbs: Float = 0
    var sumSq: Float = 0
    for i in 0..<a.count {
        let diff = a[i] - b[i]
        let absd = abs(diff)
        if absd > maxAbs { maxAbs = absd }
        sumAbs += absd
        sumSq += diff * diff
    }
    let mae = sumAbs / Float(a.count)
    let rmse = sqrt(sumSq / Float(a.count))
    return (maxAbs, mae, rmse)
}

func usage() {
    let msg = "Usage: validate_coreml.swift <mlmodelc_path> <input_bin> <output_bin> <latent_channels>"
    print(msg)
}

guard CommandLine.arguments.count == 5 else {
    usage()
    exit(2)
}

let modelPath = CommandLine.arguments[1]
let inputPath = CommandLine.arguments[2]
let outputPath = CommandLine.arguments[3]
let latentChannels = Int(CommandLine.arguments[4]) ?? 0

if latentChannels <= 0 {
    usage()
    exit(2)
}

let inputShape = [1, 96, 96, latentChannels]
let outputShape = [1, 768, 768, 3]

let inputTensor = try loadBin(inputPath)
let expectedTensor = try loadBin(outputPath)

let inputArray = try makeMLMultiArray(from: inputTensor, shape: inputShape)

let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath))
let inputName = model.modelDescription.inputDescriptionsByName.keys.first!
let outputName = model.modelDescription.outputDescriptionsByName.keys.first!

let provider = try MLDictionaryFeatureProvider(dictionary: [
    inputName: MLFeatureValue(multiArray: inputArray)
])

let outFeatures = try model.prediction(from: provider)
let outputArray = outFeatures.featureValue(for: outputName)!.multiArrayValue!

let got = readOutput(from: outputArray)
let expected = expectedTensor.data.withUnsafeBytes {
    Array(UnsafeBufferPointer<Float>(start: $0.bindMemory(to: Float.self).baseAddress!,
                                     count: expectedTensor.count))
}

let metrics = compare(got, expected)
print("max_abs=\(metrics.maxAbs), mae=\(metrics.mae), rmse=\(metrics.rmse)")
