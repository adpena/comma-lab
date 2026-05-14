#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a trained szabolcs renderer to the SZv1 archive format.

Pipeline
--------
1. Load a torch pickle produced by ``experiments/train_szabolcs.py``.
2. Rebuild a ``SzabolcsRenderer`` from the embedded config.
3. Block-FP pack the weights + tar.xz the payload via
   ``tac.szabolcs_archive.pack_szabolcs_archive``.
4. Write the SZv1 binary to disk.

The output binary is the ONLY artifact the contest archive needs from the
szabolcs lane — there is no masks.mkv (the LUT is reconstructed in code at
inflate time) and no optimized_poses.pt (the per-frame affine embedding
travels inside the renderer state).

Usage
-----
    python experiments/export_szabolcs_archive.py \\
        --checkpoint results/lane_sz_phase2/szabolcs_best.pt \\
        --output    results/lane_sz_phase2/renderer.bin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

from tac.contrib.szabolcs_renderer import SzabolcsRenderer  # noqa: E402
from tac.szabolcs_archive import pack_szabolcs_archive  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack a trained szabolcs renderer to SZv1.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a torch pickle from train_szabolcs.py.")
    p.add_argument("--output", type=str, default="szabolcs_renderer.bin",
                   help="Output SZv1 binary path.")
    p.add_argument("--block-size", type=int, default=16,
                   help="Block-FP partition size along axis 0 (default 16).")
    p.add_argument("--clip-threshold", type=float, default=0.5,
                   help="Ternary rounding threshold (default 0.5).")
    p.add_argument(
        "--predicted-band-low",
        type=float, default=0.30,
        help="Advisory predicted contest score lower bound (provenance only).",
    )
    p.add_argument(
        "--predicted-band-high",
        type=float, default=0.50,
        help="Advisory predicted contest score upper bound (provenance only).",
    )
    return p.parse_args()


def _rebuild_model(ckpt: dict) -> SzabolcsRenderer:
    cfg = ckpt.get("config") or {}
    state = ckpt.get("model_state_dict")
    if state is None:
        raise SystemExit("checkpoint missing 'model_state_dict' — incompatible format")

    model = SzabolcsRenderer(
        hidden=int(cfg.get("hidden", 32)),
        block_hidden=int(cfg["block_hidden"]) if cfg.get("block_hidden") is not None else None,
        num_blocks=int(cfg.get("num_blocks", 4)),
        max_frame_index=int(cfg.get("max_frame_index", 1200)),
        affine_max_zoom_delta=float(cfg.get("affine_max_zoom_delta", 0.12)),
        affine_max_aspect_delta=float(cfg.get("affine_max_aspect_delta", 0.03)),
        affine_max_shear=float(cfg.get("affine_max_shear", 0.03)),
        affine_max_translation=float(cfg.get("affine_max_translation", 0.08)),
        latent_input_scale=float(cfg.get("latent_input_scale", 1.0)),
        shared_latent_channels=int(cfg.get("shared_latent_channels", 3)),
        shared_latent_height=int(cfg.get("shared_latent_height", 30)),
        shared_latent_width=int(cfg.get("shared_latent_width", 40)),
        latent_canvas_scale=float(cfg.get("latent_canvas_scale", 1.25)),
        num_classes=int(cfg.get("num_classes", 5)),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(f"checkpoint missing state keys: {sorted(missing)[:8]}")
    if unexpected:
        print(f"[export_szabolcs] WARNING: ignoring unexpected keys: "
              f"{sorted(unexpected)[:8]}")
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    print(f"[export_szabolcs] loading {ckpt_path} …", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _rebuild_model(ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export_szabolcs] rebuilt SzabolcsRenderer ({n_params:,} params)",
          flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob, stats = pack_szabolcs_archive(
        model,
        output_path=out_path,
        block_size=args.block_size,
        clip_threshold=args.clip_threshold,
        predicted_band=(args.predicted_band_low, args.predicted_band_high),
    )

    summary = {
        "binary": str(out_path),
        "magic": "SZv1",
        "raw_param_count": stats.raw_param_count,
        "raw_param_bytes_fp32": stats.raw_param_bytes,
        "packed_bytes": stats.packed_bytes,
        "tarxz_compressed_bytes": stats.tarxz_compressed_bytes,
        "bits_per_weight": round(stats.bits_per_weight, 3),
        "compression_ratio_vs_fp32": round(
            stats.raw_param_bytes / max(stats.packed_bytes, 1), 2
        ),
    }
    print(json.dumps(summary, indent=2))
    print(f"[export_szabolcs] wrote {out_path} "
          f"({stats.packed_bytes:,} bytes, "
          f"{stats.bits_per_weight:.3f} bits/weight)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
