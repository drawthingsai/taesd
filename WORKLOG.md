# WORKLOG

Date: 2026-01-28
Project: /Users/liu/workspace/taesd

## User Requests (Chronological)
- Read `README.md` to understand the repo.
- Write reproducible scripts using coremltools to convert FLUX.1 VAE and FLUX.2 VAE to Core ML (`.mlmodelc`), fixed input size 768x768.
- Validate Core ML results match PyTorch.
- Start with NCHW, then validate NHWC (`[1, 768, 768, 16]`/`[1, 768, 768, 32]`); later corrected to 96x96 latent (since VAE downsamples by 8).
- Focus only on decoder (remove encoder support).
- Run the scripts and fix errors.
- Provide validation code for review.
- Write Swift code to load NHWC `.mlmodelc` and validate it matches Python output.
- Run Swift validation for FLUX.1/2 (NHWC).
- Switch conversion to Float16, re-run Python CoreML validation and Swift validation.
- Verify ANE-only execution (not possible to prove in this environment).
- Write a replay log; then asked to write this WORKLOG.

## Files Added/Modified
- Added `scripts/convert_flux_vae_coreml.py`
  - Decoder-only conversion for FLUX.1/2
  - Supports NCHW/NHWC
  - TorchScript tracing before conversion
  - ML Program output (`.mlpackage`) + compiled `.mlmodelc`
  - coremlc fallback compile when MLModelProxy fails
  - Validation against PyTorch with error metrics
  - Added precision switch (`float32`/`float16`) and tolerance handling
- Added `scripts/export_flux_decoder_io.py`
  - Exports deterministic NHWC input/output tensors to `.bin` for Swift validation
- Added `scripts/validate_coreml.swift`
  - Loads `.mlmodelc`, runs inference, compares output to reference
  - Handles float16 outputs and non‑contiguous strides
- Updated `scripts/convert_flux_vae_coreml.py` multiple times to:
  - Remove encoder support
  - Fix repo import
  - Force `source="pytorch"` and TorchScript tracing
  - Save ML Program as `.mlpackage`
  - Compile via `coremlc` when needed
  - Avoid `/var/folders` temp permission issues
  - Add precision selection and default tolerances

## Key Corrections
- Latent input is 96x96 for 768x768 image (1/8 downsample), not 768x768.
- Decoder-only conversion required setting `encoder_path=None`.
- coremltools compile failed in sandbox; used `xcrun coremlc compile`.
- Swift validation initially incorrect due to float16 output and strides; fixed.

## Commands Run (Repro Steps)
Install deps in venv:
- `pip install coremltools numpy torch`

Convert + validate with coremltools (float32):
- `python scripts/convert_flux_vae_coreml.py --variant flux1 --layout nchw --precision float32 --validate`
- `python scripts/convert_flux_vae_coreml.py --variant flux2 --layout nchw --precision float32 --validate`
- `python scripts/convert_flux_vae_coreml.py --variant flux1 --layout nhwc --precision float32 --validate`
- `python scripts/convert_flux_vae_coreml.py --variant flux2 --layout nhwc --precision float32 --validate`

Export deterministic NHWC IO for Swift:
- `python scripts/export_flux_decoder_io.py --variant flux1`
- `python scripts/export_flux_decoder_io.py --variant flux2`

Convert + validate with coremltools (float16, adjusted tolerances):
- `python scripts/convert_flux_vae_coreml.py --variant flux1 --layout nchw --precision float16 --validate --atol 0.02`
- `python scripts/convert_flux_vae_coreml.py --variant flux2 --layout nchw --precision float16 --validate --atol 0.03`
- `python scripts/convert_flux_vae_coreml.py --variant flux1 --layout nhwc --precision float16 --validate --atol 0.02`
- `python scripts/convert_flux_vae_coreml.py --variant flux2 --layout nhwc --precision float16 --validate --atol 0.03`

Swift validation (NHWC, float16):
- `swift -Xfrontend -module-cache-path -Xfrontend /tmp/swift-module-cache scripts/validate_coreml.swift coreml_out/flux1_decoder_nhwc_float16.mlmodelc coreml_io/flux1/input.bin coreml_io/flux1/output.bin 16`
- `swift -Xfrontend -module-cache-path -Xfrontend /tmp/swift-module-cache scripts/validate_coreml.swift coreml_out/flux2_decoder_nhwc_float16.mlmodelc coreml_io/flux2/input.bin coreml_io/flux2/output.bin 32`

Optional Python validation using CompiledMLModel with CPU+NE:
- See the Python snippet in the previous replay log message.

## Validation Results (Recorded)
Float32 (Python coremltools validation):
- FLUX.1 decoder NCHW: max_abs ~2.03e-06
- FLUX.2 decoder NCHW: max_abs ~2.06e-04
- FLUX.1 decoder NHWC: max_abs ~2.44e-06
- FLUX.2 decoder NHWC: max_abs ~1.67e-04

Float16 (Python coremltools validation):
- FLUX.1 decoder NCHW: max_abs 0.01793 (atol 0.02)
- FLUX.2 decoder NCHW: max_abs 0.02299 (atol 0.03)
- FLUX.1 decoder NHWC: max_abs 0.01973 (atol 0.02)
- FLUX.2 decoder NHWC: max_abs 0.02479 (atol 0.03)

Swift (NHWC, float16):
- FLUX.1 decoder NHWC: max_abs 0.004456, mae 0.000260, rmse 0.000347
- FLUX.2 decoder NHWC: max_abs 0.005844, mae 0.000406, rmse 0.000538

## ANE-only Verification
- coremltools exposes `ComputeUnit.CPU_AND_NE` but no `NEURAL_ENGINE_ONLY`.
- No runtime reporting to prove ANE-only execution in this environment.
- Result correctness validated with CPU+NE (not exclusive to ANE).

Date: 2026-01-28 (continued)

## User Requests (Additional)
- Clarify NHWC vs NCHW for Core ML / ANE; user confirmed ANE prefers channels-first and asked to keep NHWC work as reference.
- Run NCHW benchmarks for float32/float16 and clean up NHWC artifacts.
- Explore ANE performance optimization: ML Program FP16, float16 I/O, neuralnetwork format, and compute units.
- Verify performance discrepancies between Python vs Swift; determine if sandbox overhead affected timings.
- Switch focus to FLUX.2; generate Core ML models and enable Swift benchmarking.
- Generate 1024x1024 models for FLUX.1/2 and compare performance; then add flexible shape models supporting 768 and 1024.

## Work Performed
- Added NHWC reference implementation in `taesd_nhwc.py` (NHWC wrappers for Conv/GroupNorm/Upsample) and `--nhwc-model` option in `scripts/convert_flux_vae_coreml.py`.
- Rebuilt NHWC float32/float16 models, inspected MIL op counts (extra transposes only at input/output), and validated correctness.
- Cleaned NHWC and bench artifacts when requested.
- Added conversion options:
  - `--convert-to {mlprogram, neuralnetwork}`
  - `--io-precision {auto,float16,float32}`
  - lowered min deployment target for neuralnetwork
  - float16 I/O validation handling
- Generated ANE-optimized ML Program (float16 compute + float16 I/O) for FLUX.1 and FLUX.2; compiled `.mlmodelc` using `coremlc`.
- Added benchmarking scripts:
  - `scripts/benchmark_coreml_flux1.py` (CoreML compute units)
  - `scripts/benchmark_coreml_flux1_python.py` (Python timing script)
- Added/updated Swift harness for Instruments/benchmarking: `scripts/run_coreml_instruments.swift`
  - supports compute units, dtype, input-hw, latent-channels
  - prints input shape/dtype
  - reports avg/median/p10/p90/min/max over N iters
  - fixed behavior to use `--input-hw` for enumerated shapes by default
- Demonstrated that sandboxed Python timing was ~10x slower than local runs; user confirmed ~20 ms in local Swift/Python.

## Benchmark Results (Local Swift / User)
- FLUX.1 fixed 768: ~20 ms (CPU+NE, float16) in local environment.
- FLUX.2 fixed 768: ~21 ms (CPU+NE, float16) in local environment.
- FLUX.1 fixed 1024: ~36.2 ms (CPU+NE, float16).
- FLUX.2 fixed 1024: ~38.3 ms (CPU+NE, float16).

## Flexible Shape Models (768 & 1024)
- Implemented enumerated shape support via `--input-hw-list 768,1024` in converter.
- Generated ML Program float16 models with float16 I/O:
  - `coreml_out/ane_opt_flex/flux1_decoder_nchw_float16_hw768_1024_iofloat16.mlmodelc`
  - `coreml_out/ane_opt_flex/flux2_decoder_nchw_float16_hw768_1024_iofloat16.mlmodelc`
- User benchmarks for flex models:
  - FLUX.1 flex: 768 ~21.84 ms, 1024 ~38.38 ms
  - FLUX.2 flex: 768 ~22.94 ms, 1024 ~40.57 ms
  (CPU+NE, float16, 200 iters)

## Cleanup Actions
- Removed: `coreml_out/ane_opt/bench`, `coreml_out/ane_opt_256`, `coreml_out/ane_opt_512`, `coreml_out/ane_opt_flux2/bench`
- Removed all NHWC artifacts when requested.

## Notable Decisions / Findings
- Core ML/ANE prefers channels-first; NHWC graph not beneficial beyond input/output.
- coremltools MLModel predict in sandbox showed large overhead vs Swift/Python locally; confirmed local results are fast.
- Flexible (enumerated) shapes do not significantly degrade performance vs fixed shapes.

## Quantization / QAT Follow-ups (since last entry)
- Discussed ANE performance optimization ideas for FLUX.1/2; user asked to try 4-bit palettization, 8-bit palettization, int8 weights, and int8 activations (QAT), focusing on 1024x1024 first.
- Implemented additional converter options and scripts to support quantization/benching:
  - `scripts/convert_flux_vae_coreml.py`: added `--input-hw-list`, `--palettize-nbits`, `--quantize-weight-int8`, `--convert-to`, `--io-precision`; supports enumerated shapes and naming suffixes.
  - `scripts/benchmark_coreml_flux1.py` and `scripts/benchmark_coreml_flux1_python.py` for CoreML timing.
  - `scripts/qat_flux_decoder_int8.py` for QAT int8 weights + activations using coremltools LinearQuantizer.
  - `scripts/download_soa_subset.py` to fetch ~20 images from `madebyollin/soa-aesthetic` for calibration.
  - `scripts/roundtrip_compare.py` to save round-trip outputs for visual inspection.
- Generated CoreML artifacts for 1024:
  - Palettized 4-bit and 8-bit (`coreml_out/ane_opt_pal4_1024`, `coreml_out/ane_opt_pal8_1024`).
  - Int8 weight-only (`coreml_out/ane_opt_int8w_1024`).
  - QAT int8 weights+activations (`coreml_out/ane_opt_qat_int8_full`).
- User-reported Swift benchmarks (1024):
  - Palettization 4/8-bit: ~40 ms (no speedup).
  - Int8 weights: ~40 ms (no speedup).
  - QAT int8 weights+activations: ~36.3 ms (faster than fp16 baseline ~38.4 ms).
- QAT quality check (CoreML CPU compare): max_abs ~0.34, MAE ~0.015, RMSE ~0.020 across small dataset; accuracy did not improve vs earlier runs.
- Saved round-trip images for inspection to `coreml_io/roundtrip_compare/` with subdirs: `original`, `fp16_baseline`, `pal4`, `pal8`, `int8w`, `qat_int8`.
- Installed Python deps in `_env`: `requests`, `datasets`, `pillow` (and dependencies).
