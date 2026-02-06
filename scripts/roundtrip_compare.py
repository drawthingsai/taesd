#!/usr/bin/env python3
"""Save round-trip reconstructions for multiple decoder variants."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import coremltools as ct

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from taesd import TAESD


def load_images(image_dir: Path, input_hw: int, max_images: int) -> list[tuple[str, torch.Tensor]]:
    items: list[tuple[str, torch.Tensor]] = []
    for path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        im = Image.open(path).convert("RGB").resize((input_hw, input_hw), Image.BICUBIC)
        arr = np.asarray(im).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        items.append((path.stem, tensor))
        if len(items) >= max_images:
            break
    return items


def save_image(t: torch.Tensor, path: Path) -> None:
    t = t.clamp(0, 1)
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def compute_metrics(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    ref = ref.to(dtype=torch.float32)
    out = out.to(dtype=torch.float32)
    diff = out - ref
    mae = diff.abs().mean().item()
    mse = diff.pow(2).mean().item()
    rmse = mse**0.5
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20.0 * np.log10(1.0) - 10.0 * np.log10(mse)
    return {"mae": mae, "rmse": rmse, "psnr": psnr}


def run_coreml(modelc: Path, spec_path: Path, latent: np.ndarray) -> np.ndarray:
    spec = ct.models.utils.load_spec(str(spec_path))
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    model = ct.models.CompiledMLModel(str(modelc), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = model.predict({input_name: latent})[output_name]
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["flux1", "flux2"], default="flux1")
    parser.add_argument("--image-dir", type=Path, default=Path("coreml_io/soa_subset"))
    parser.add_argument("--out-dir", type=Path, default=Path("coreml_io/roundtrip_compare"))
    parser.add_argument("--input-hw", type=int, default=1024)
    parser.add_argument("--max-images", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    if args.variant == "flux1":
        encoder_path = "taef1_encoder.pth"
        decoder_path = "taef1_decoder.pth"
        latent_channels = 16
        arch_variant = None
        fp16_modelc = "coreml_out/ane_opt_flex_fp16/flux1_decoder_nchw_float16_hw768_1024_iofloat16.mlmodelc"
        fp16_spec = "coreml_out/ane_opt_flex_fp16/flux1_decoder_nchw_float16_hw768_1024_iofloat16.mlpackage"
        pal8_modelc = "coreml_out/ane_opt_flex_pal8/flux1_decoder_nchw_float16_hw768_1024_pal8_iofloat16.mlmodelc"
        pal8_spec = "coreml_out/ane_opt_flex_pal8/flux1_decoder_nchw_float16_hw768_1024_pal8_iofloat16.mlpackage"
    else:
        encoder_path = "taef2_encoder.pth"
        decoder_path = "taef2_decoder.pth"
        latent_channels = 32
        arch_variant = "flux_2"
        fp16_modelc = "coreml_out/ane_opt_flex_fp16/flux2_decoder_nchw_float16_hw768_1024_iofloat16.mlmodelc"
        fp16_spec = "coreml_out/ane_opt_flex_fp16/flux2_decoder_nchw_float16_hw768_1024_iofloat16.mlpackage"
        pal8_modelc = "coreml_out/ane_opt_flex_pal8/flux2_decoder_nchw_float16_hw768_1024_pal8_iofloat16.mlmodelc"
        pal8_spec = "coreml_out/ane_opt_flex_pal8/flux2_decoder_nchw_float16_hw768_1024_pal8_iofloat16.mlpackage"

    taesd = TAESD(encoder_path=encoder_path, decoder_path=None, latent_channels=latent_channels, arch_variant=arch_variant)
    encoder = taesd.encoder.eval()

    images = load_images(args.image_dir, args.input_hw, args.max_images)
    if not images:
        raise SystemExit("No images found.")

    # Save originals
    orig_dir = args.out_dir / "original"
    orig_dir.mkdir(exist_ok=True)
    for name, img in images:
        save_image(img, orig_dir / f"{name}.png")

    # Decode via PyTorch baseline
    dec = TAESD(encoder_path=None, decoder_path=decoder_path, latent_channels=latent_channels, arch_variant=arch_variant).decoder.eval()

    methods = {
        "fp16_baseline": {
            "modelc": Path(fp16_modelc),
            "spec": Path(fp16_spec),
        },
        "pal8": {
            "modelc": Path(pal8_modelc),
            "spec": Path(pal8_spec),
        },
    }

    metrics_rows: list[dict[str, float | str]] = []

    # Precompute latents + PyTorch baseline once
    precomputed: list[dict[str, object]] = []
    for name, img in images:
        with torch.no_grad():
            latent = encoder(img).cpu().numpy().astype(np.float32)
            out_pt = dec(torch.from_numpy(latent)).cpu()
        precomputed.append({"name": name, "img": img, "latent": latent, "out_pt": out_pt})

    # Save PyTorch baseline
    pytorch_dir = args.out_dir / "pytorch_fp32"
    pytorch_dir.mkdir(exist_ok=True)
    for item in precomputed:
        name = item["name"]
        img = item["img"]
        out_pt = item["out_pt"]
        save_image(out_pt, pytorch_dir / f"{name}.png")
        m = compute_metrics(img, out_pt)
        metrics_rows.append({"method": "pytorch_fp32", "image": name, **m})

    for method, paths in methods.items():
        out_dir = args.out_dir / method
        out_dir.mkdir(exist_ok=True)
        if not paths["modelc"].exists():
            continue
        for item in precomputed:
            name = item["name"]
            img = item["img"]
            latent = item["latent"]
            out = run_coreml(paths["modelc"], paths["spec"], latent)
            out_t = torch.from_numpy(out)
            save_image(out_t, out_dir / f"{name}.png")
            m = compute_metrics(img, out_t)
            metrics_rows.append({"method": method, "image": name, **m})

    if metrics_rows:
        import csv

        metrics_path = args.out_dir / "metrics.csv"
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "image", "mae", "rmse", "psnr"])
            writer.writeheader()
            writer.writerows(metrics_rows)

    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
