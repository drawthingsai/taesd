#!/usr/bin/env python3
"""QAT-style int8 weight+activation quantization for FLUX decoder using a tiny calibration set."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import coremltools as ct
from coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from taesd import TAESD

FLUX_VARIANTS = {
    "flux1": {
        "latent_channels": 16,
        "arch_variant": None,
        "encoder_path": "taef1_encoder.pth",
        "decoder_path": "taef1_decoder.pth",
    },
    "flux2": {
        "latent_channels": 32,
        "arch_variant": "flux_2",
        "encoder_path": "taef2_encoder.pth",
        "decoder_path": "taef2_decoder.pth",
    },
}


def load_images(image_dir: Path, input_hw: int, max_images: int) -> list[torch.Tensor]:
    images = []
    for path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        im = Image.open(path).convert("RGB").resize((input_hw, input_hw), Image.BICUBIC)
        arr = np.asarray(im).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        images.append(tensor)
        if len(images) >= max_images:
            break
    return images


def encode_latents(encoder: torch.nn.Module, images: list[torch.Tensor]) -> list[torch.Tensor]:
    latents = []
    encoder.eval()
    with torch.no_grad():
        for im in images:
            latents.append(encoder(im))
    return latents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=FLUX_VARIANTS.keys(), default="flux1")
    parser.add_argument("--input-hw", type=int, default=1024)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("coreml_out/ane_opt_qat_int8"))
    parser.add_argument("--iters", type=int, default=1, help="Calibration passes over dataset")
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    cfg = FLUX_VARIANTS[args.variant]
    taesd = TAESD(
        encoder_path=cfg["encoder_path"],
        decoder_path=cfg["decoder_path"],
        latent_channels=cfg["latent_channels"],
        arch_variant=cfg["arch_variant"],
    )

    encoder = taesd.encoder.eval()
    decoder = taesd.decoder.eval()

    images = load_images(args.image_dir, args.input_hw, args.max_images)
    if not images:
        raise SystemExit("No images found for calibration.")

    latents = encode_latents(encoder, images)

    # QAT configuration: int8 weights + int8 activations (quint8)
    qconfig = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": "symmetric",
                "weight_dtype": "qint8",
                "activation_dtype": "quint8",
                "weight_per_channel": False,
                "milestones": [0, 100, 200, 300],
            }
        }
    )
    quantizer = LinearQuantizer(decoder, qconfig)
    example = (latents[0],)
    qat_model = quantizer.prepare(example_inputs=example, inplace=False)

    qat_model.train()
    with torch.no_grad():
        for _ in range(args.iters):
            for latent in latents:
                _ = qat_model(latent)
                quantizer.step()

    finalized = quantizer.finalize(qat_model, inplace=False)
    finalized.eval()

    # Convert to Core ML
    io_dtype = np.float16 if args.precision == "float16" else np.float32
    input_type = ct.TensorType(name="latent", shape=latents[0].shape, dtype=io_dtype)
    example_input = latents[0].cpu().numpy().astype(np.float32)
    traced = torch.jit.trace(finalized, torch.from_numpy(example_input), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[input_type],
        source="pytorch",
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )

    stem = f"{args.variant}_decoder_qat_int8_{args.input_hw}"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{stem}.mlpackage"
    mlmodel.save(model_path)
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
