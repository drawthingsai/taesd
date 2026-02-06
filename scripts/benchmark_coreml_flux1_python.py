#!/usr/bin/env python3
"""Benchmark Core ML FLUX.1 decoder via coremltools CompiledMLModel."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import coremltools as ct


def latent_hw(input_hw: int) -> int:
    if input_hw % 8 != 0:
        raise ValueError(f"input_hw must be divisible by 8 (got {input_hw})")
    return input_hw // 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelc", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--input-hw", type=int, default=768)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compute-units", choices=["CPU_ONLY", "CPU_AND_NE", "ALL"], default="CPU_AND_NE")
    args = parser.parse_args()

    spec_obj = ct.models.utils.load_spec(str(args.spec))
    input_name = spec_obj.description.input[0].name

    lat_hw = latent_hw(args.input_hw)
    rng = np.random.default_rng(0)
    latent = rng.uniform(-1.0, 1.0, size=(1, 16, lat_hw, lat_hw)).astype(
        np.float16 if args.dtype == "float16" else np.float32
    )

    cu_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    model = ct.models.CompiledMLModel(str(args.modelc), compute_units=cu_map[args.compute_units])

    for _ in range(args.warmup):
        _ = model.predict({input_name: latent})

    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _ = model.predict({input_name: latent})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    avg = sum(times) / len(times)
    med = times[len(times) // 2]
    p10 = times[int(0.1 * (args.iters - 1))]
    p90 = times[int(0.9 * (args.iters - 1))]

    print(f"avg ms: {avg:.3f}")
    print(f"median ms: {med:.3f}")
    print(f"p10 ms: {p10:.3f}")
    print(f"p90 ms: {p90:.3f}")
    print(f"min ms: {times[0]:.3f}")
    print(f"max ms: {times[-1]:.3f}")


if __name__ == "__main__":
    main()
