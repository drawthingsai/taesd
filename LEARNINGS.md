# TAESD CoreML Learnings (Full Handoff)

This file is a complete handoff of work done in this repo for FLUX.1/FLUX.2 TAEs, including architecture experiments, benchmarking, conversion scripts, quantization attempts, artifact generation, and final packaging strategy.

## 1) Scope and Outcome
- Objective: optimize and package CoreML decoder models for FLUX.1 and FLUX.2.
- Final direction:
  - Keep runtime in FP16.
  - Use fixed-shape models (not flexible shape) due to memory behavior.
  - Use NHWC input/output wrapper models where requested.
  - Deduplicate per-variant `weight.bin` across resolutions.

## 2) Major Conclusions
- Flexible shape (`EnumeratedShapes`) is functional but caused higher RAM usage in practice.
- Fixed-shape models per resolution behaved better for memory.
- NHWC I/O is feasible by wrapping decoder with internal transposes.
- Weight tensors are identical across fixed-shape resolutions for a given variant, enabling single `weight.bin` reuse.
- Quantization attempts (palettization, int8 weight-only) did not improve latency versus FP16 baseline in tested runs.
- QAT int8 gave some speed gain in one setup but showed quality tradeoffs; FP16 remained preferred baseline for this project.

## 3) What Was Implemented

### Conversion pipeline
- Updated `scripts/convert_flux_vae_coreml.py` to support:
  - `--convert-to {mlprogram,neuralnetwork}`
  - `--io-precision {auto,float16,float32}`
  - `--input-hw-list` for enumerated shapes
  - `--palettize-nbits`
  - `--quantize-weight-int8`
- Added NHWC wrapper path with internal transpose (NHWC in/out around standard conv decoder).
- Extended validation logic so list-based shapes validate all sizes.

### Benchmarking
- Added/used `scripts/run_coreml_instruments.swift` as canonical benchmark harness.
- Extended Swift harness to support `--layout nhwc|nchw` so input shape is correct for NHWC models.
- Added Python benchmark scripts for convenience, with note that sandboxed timing can be misleading.

### Roundtrip and data tools
- Added `scripts/download_soa_subset.py` for small dataset from `madebyollin/soa-aesthetic`.
- Added/updated `scripts/roundtrip_compare.py` for per-method output images and metrics (MAE/RMSE/PSNR).

### Quantization experiments
- Ran post-training palettization (4-bit, 8-bit), int8 weight-only, and QAT int8 (weights+activations).
- Observed no practical speed win for palettization/weight-only int8 in this context.

## 4) Benchmark Findings (from user-local Swift runs)
- Baseline FP16 NCHW:
  - FLUX.1: ~20-21 ms @768, ~36-38 ms @1024.
  - FLUX.2: ~21-23 ms @768, ~38-41 ms @1024.
- Pal8 vs FP16 was consistently slower (roughly ~11% in sampled runs).
- These findings drove final preference for FP16 fixed-shape models.

## 5) Flexible vs Fixed Shape Decision
- Flexible-shape models were generated and tested for 768/1024 and later larger sets.
- User observed excessive RAM usage with flexible shape.
- Switched to fixed-shape models per resolution: 768, 1024, 1280, 1536, 1792, 2048.

## 6) iOS18/macOS15 Redo (Latest)
A full redo was performed with minimum deployment aligned to:
- iOS / iPadOS: 18+
- macOS: 15+

### Deployment target mapping detail
- In current `coremltools` enum, `ct.target.iOS18` maps to `macOS15` compatibility.
- Converter was updated to use `ct.target.iOS18` for MLProgram outputs.

### Compatibility note for previous artifacts
- Earlier artifacts targeted lower deployment (e.g., macOS13 equivalent).
- Those are generally expected to run on iOS18/macOS15 due to backward compatibility.
- New artifacts are now explicitly targeted to iOS18/macOS15 baseline.

## 7) Final Artifacts Generated

### Per-resolution iOS18/macOS15 NHWC FP16 outputs
Generated for each resolution in:
- `coreml_out/ane_opt_fp16_nhwc_ios18_768/`
- `coreml_out/ane_opt_fp16_nhwc_ios18_1024/`
- `coreml_out/ane_opt_fp16_nhwc_ios18_1280/`
- `coreml_out/ane_opt_fp16_nhwc_ios18_1536/`
- `coreml_out/ane_opt_fp16_nhwc_ios18_1792/`
- `coreml_out/ane_opt_fp16_nhwc_ios18_2048/`

Each directory contains:
- `flux1_decoder_nhwc_float16_iofloat16.mlmodelc`
- `flux2_decoder_nhwc_float16_iofloat16.mlmodelc`
- plus corresponding `.mlpackage` during conversion flow

### Validation status
- FLUX.1 validated at each size with `--atol 0.03`.
- FLUX.2 validated at each size with `--atol 0.05`.
- All requested size/variant combinations passed.

## 8) Weight Equality Verification (Critical for dedup)
For iOS18 fixed-shape artifacts, SHA256 on `weights/weight.bin`:

- FLUX.1 (all sizes identical):
  - `3f0dc80e4e4f52cc36e10a9a6d0604ba80f87dc5f1d73034ea811f7690d49b9f`
- FLUX.2 (all sizes identical):
  - `d9d30f8c992b265b3a08c19e2bcadff8da06fb3f2f96bb2dd7b8ee01c1519058`

Result: one `weight.bin` per variant is valid across all six resolutions.

## 9) Final Packaging Layout Built
Per request, created two directories with per-size `.mlmodelc` bundles and a single shared weight file per variant:

```text
flux_1_tae/
  768.mlmodelc/
  1024.mlmodelc/
  1280.mlmodelc/
  1536.mlmodelc/
  1792.mlmodelc/
  2048.mlmodelc/
  weight.bin

flux_2_tae/
  768.mlmodelc/
  1024.mlmodelc/
  1280.mlmodelc/
  1536.mlmodelc/
  1792.mlmodelc/
  2048.mlmodelc/
  weight.bin
```

Packaging behavior implemented:
- Copied `.mlmodelc` bundles from the iOS18 fixed-shape outputs.
- Removed embedded `weights/weight.bin` from each per-size bundle.
- Copied one canonical `weight.bin` to each variant root (`flux_1_tae/weight.bin`, `flux_2_tae/weight.bin`).

## 10) Files Modified / Added During This Project
- `scripts/convert_flux_vae_coreml.py`
- `scripts/run_coreml_instruments.swift`
- `scripts/benchmark_coreml_flux1.py`
- `scripts/benchmark_coreml_flux1_python.py`
- `scripts/download_soa_subset.py`
- `scripts/qat_flux_decoder_int8.py`
- `scripts/roundtrip_compare.py`
- `taesd_nhwc.py`
- `WORKLOG.md`
- `LEARNINGS.md`

## 11) Operational Notes
- `coremltools` warns that Torch 2.10.0 is untested (informational in this setup).
- In sandboxed runs, `predict()` compile paths can warn; fallback compile via `xcrun coremlc` was already part of conversion script behavior.
- Swift benchmarking remains the reliable latency measurement path.

## 12) Recommended Next Use in `taehv`
- Reuse this exact fixed-shape + shared-weight packaging pattern.
- Keep deployment target consistent with product requirement (`iOS18/macOS15`).
- Validate per-size models, then verify `weight.bin` hash equality before packaging.
- If packaging as zip with no compression, preserve directory structure and root-level shared weights.

## 13) Chronological Summary (Compact)
1. Read prior worklog and resumed NCHW vs NHWC discussions.
2. Built/iterated benchmark harness and validated local performance methodology.
3. Explored NHWC wrapper conversion and ML package inspection.
4. Concluded ANE preference and shifted to pragmatic performance focus.
5. Tested quantization variants; kept FP16 baseline preference.
6. Built flexible-shape models; identified RAM concerns.
7. Switched to fixed-shape model generation across target resolutions.
8. Verified per-resolution `weight.bin` identity.
9. Reorganized deliverables to `flux_1_tae` / `flux_2_tae` with shared variant weights.
10. Redid full model generation for iOS18/macOS15 baseline and repeated dedup packaging.
