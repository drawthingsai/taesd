#!/usr/bin/env python3
"""
Export deterministic NHWC input/output tensors for FLUX decoders.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from taesd import TAESD


FLUX_VARIANTS = {
    "flux1": {"latent_channels": 16, "arch_variant": None, "decoder_path": "taef1_decoder.pth"},
    "flux2": {"latent_channels": 32, "arch_variant": "flux_2", "decoder_path": "taef2_decoder.pth"},
}


def main():
    parser = argparse.ArgumentParser(description="Export NHWC tensors for FLUX decoder validation.")
    parser.add_argument("--variant", choices=FLUX_VARIANTS.keys(), required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("coreml_io"))
    parser.add_argument("--input-hw", type=int, default=768)
    args = parser.parse_args()

    cfg = FLUX_VARIANTS[args.variant]
    lat_hw = args.input_hw // 8

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    taesd = TAESD(
        encoder_path=None,
        decoder_path=cfg["decoder_path"],
        latent_channels=cfg["latent_channels"],
        arch_variant=cfg["arch_variant"],
    ).eval()

    lat_nhwc = rng.uniform(-1, 1, size=(1, lat_hw, lat_hw, cfg["latent_channels"])).astype(np.float32)
    lat_nchw = torch.from_numpy(lat_nhwc).permute(0, 3, 1, 2)
    with torch.no_grad():
        out_nhwc = taesd.decoder(lat_nchw).permute(0, 2, 3, 1).cpu().numpy()

    out_dir = args.out_dir / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input.bin").write_bytes(lat_nhwc.tobytes())
    (out_dir / "output.bin").write_bytes(out_nhwc.tobytes())


if __name__ == "__main__":
    main()
