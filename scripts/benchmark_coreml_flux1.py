#!/usr/bin/env python3
"""Benchmark Core ML models for FLUX.1 decoder with different compute units."""
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


def get_input_name(model_path: Path) -> str:
    spec = ct.models.utils.load_spec(str(model_path))
    return spec.description.input[0].name


def bench(mlmodelc_path: Path, model_path_for_spec: Path, input_np: np.ndarray, compute_units, warmup: int, iters: int) -> float:
    model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units=compute_units)
    input_name = get_input_name(model_path_for_spec)
    for _ in range(warmup):
        _ = model.predict({input_name: input_np})
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.predict({input_name: input_np})
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelc", type=Path, required=True, help="Path to .mlmodelc directory.")
    parser.add_argument("--spec", type=Path, required=True, help="Path to .mlpackage or .mlmodel for spec.")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--input-hw", type=int, default=768)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--compute-units", nargs="+", default=["CPU_AND_NE", "ALL"])
    args = parser.parse_args()

    lat_hw = latent_hw(args.input_hw)
    rng = np.random.default_rng(0)
    latent = rng.uniform(-1.0, 1.0, size=(1, args.latent_channels, lat_hw, lat_hw)).astype(
        np.float16 if args.dtype == "float16" else np.float32
    )

    cu_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }

    for cu_name in args.compute_units:
        if cu_name not in cu_map:
            raise SystemExit(f"Unknown compute unit: {cu_name}")
        t = bench(args.modelc, args.spec, latent, cu_map[cu_name], args.warmup, args.iters)
        print(f"{cu_name}: {t*1000:.3f} ms")


if __name__ == "__main__":
    main()
