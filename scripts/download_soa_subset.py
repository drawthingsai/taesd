#!/usr/bin/env python3
"""Download a tiny subset of images from madebyollin/soa-aesthetic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
from datasets import load_dataset


def pick_url(row: dict) -> tuple[str | None, str | None]:
    for key in ("url", "image_url", "img_url", "image"):
        if key in row and row[key]:
            return row[key], row.get("text") or row.get("caption") or row.get("title")
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("coreml_io/soa_subset"))
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="madebyollin/soa-aesthetic")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "metadata.jsonl"

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    downloaded = 0
    with meta_path.open("w", encoding="utf-8") as meta_f:
        for row in ds:
            url, text = pick_url(row)
            if not url:
                continue
            h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            out_path = args.out_dir / f"{h}.jpg"
            if out_path.exists():
                continue
            try:
                resp = requests.get(url, timeout=args.timeout)
                if resp.status_code != 200:
                    continue
                out_path.write_bytes(resp.content)
            except Exception:
                continue
            meta_f.write(json.dumps({"url": url, "text": text, "file": out_path.name}) + "\n")
            downloaded += 1
            if downloaded >= args.max_images:
                break

    print(f"Downloaded {downloaded} images to {args.out_dir}")


if __name__ == "__main__":
    main()
