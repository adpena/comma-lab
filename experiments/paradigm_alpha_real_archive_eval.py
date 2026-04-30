#!/usr/bin/env python3
"""Paradigm α — Mask payload overhaul: real-archive empirical evaluator.

Decodes the Lane G v3 archive's masks.mkv into a (T, H, W) int64 mask
tensor, then runs each candidate codec end-to-end:

    α2 wavelet codec — Haar 2-level + uniform quantize + arithmetic code
    α3 VQ-VAE codec — top-K patch codebook + arithmetic code
    α4 grayscale-LUT — AV1 monochrome encode of class→gray mapping

Reports per-candidate:
    bytes encoded
    bytes saved vs 421KB AV1 baseline
    argmax disagreement vs decoded source
    % byte savings
    [empirical:reports/paradigm_alpha_<candidate>_real_archive.json] tag.

Usage:
    .venv/bin/python experiments/paradigm_alpha_real_archive_eval.py \\
        --archive experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \\
        --output-dir reports/

Tagging:
    All results stamped with [empirical:reports/...] tag.
    NO scorer load (CLAUDE.md compliant for compress-time-only path).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src",):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tac.wavelet_mask_codec import (  # noqa: E402
    WaveletConfig,
    decode_wavelet_codec,
    encode_wavelet_codec,
)
from tac.vqvae_mask_codec import (  # noqa: E402
    VQVAEConfig,
    build_codebook_top_k,
    decode_vqvae_codec,
    encode_vqvae_codec,
)
from tac.mask_grayscale_lut import CLASS_TO_GRAY  # noqa: E402


def _decode_masks_mkv(mkv: Path) -> torch.Tensor:
    """Decode masks.mkv → (T, H, W) int64 class IDs.

    Uses the Lane A encoding pixel = class * (255 // 4) (so 5 classes get
    {0, 63, 127, 191, 255}).
    """
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames",
            "-of", "csv=p=0", str(mkv),
        ],
        capture_output=True, text=True, check=True, timeout=60,
    )
    parts = probe.stdout.strip().split(",")
    w, h, t = int(parts[0]), int(parts[1]), int(parts[2])

    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(mkv),
            "-f", "rawvideo", "-pix_fmt", "gray", "-v", "error", "pipe:1",
        ],
        capture_output=True, timeout=300, check=True,
    )
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    pixels = raw.reshape(t, h, w)
    scale = 255 // 4  # 63
    classes = np.round(pixels.astype(np.float32) / scale).astype(np.int64)
    classes = np.clip(classes, 0, 4)
    return torch.from_numpy(classes)


def _encode_grayscale_av1(masks: torch.Tensor, output_path: Path, *, crf: int = 50) -> int:
    """Encode masks → grayscale.mkv via Selfcomp class targets, return byte size."""
    t, h, w = masks.shape
    targets = torch.tensor([CLASS_TO_GRAY[c] for c in range(5)], dtype=torch.uint8)
    pixels = targets[masks].numpy()
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "gray",
        "-r", "20", "-i", "pipe:0",
        "-c:v", "libsvtav1",
        "-crf", str(crf), "-preset", "6",
        "-svtav1-params", "enable-restoration=0:enable-cdef=0",
        "-pix_fmt", "gray", "-an",
        str(output_path),
    ]
    proc = subprocess.run(cmd, input=pixels.tobytes(), capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='replace')}")
    return output_path.stat().st_size


def _decode_grayscale_av1(grayscale_mkv: Path, t: int, h: int, w: int) -> torch.Tensor:
    """Decode grayscale.mkv → (T,H,W) int64 class IDs via Selfcomp Gaussian-LUT."""
    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(grayscale_mkv),
            "-f", "rawvideo", "-pix_fmt", "gray", "-v", "error", "pipe:1",
        ],
        capture_output=True, timeout=300, check=True,
    )
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    pixels = raw.reshape(t, h, w)
    # Nearest-neighbour over Selfcomp targets [0, 255, 64, 192, 128]
    targets = np.array([0, 255, 64, 192, 128], dtype=np.float32)
    p = pixels.astype(np.float32)[..., None]  # (T,H,W,1)
    dists = np.abs(p - targets[None, None, None, :])
    classes = dists.argmin(axis=-1).astype(np.int64)
    return torch.from_numpy(classes)


def _eval_alpha2_wavelet(masks: torch.Tensor, out_dir: Path) -> dict:
    """Evaluate Wavelet codec on real masks."""
    print("[α2 wavelet] encoding...", flush=True)
    config = WaveletConfig(levels=2, step_ll=0.5, step_detail=1.0)
    blob = encode_wavelet_codec(masks, config=config)
    masks_recovered = decode_wavelet_codec(blob)
    agreement = (masks_recovered == masks).float().mean().item()
    return {
        "candidate": "alpha2_wavelet",
        "config": {
            "levels": config.levels,
            "step_ll": config.step_ll,
            "step_detail": config.step_detail,
        },
        "encoded_bytes": len(blob),
        "argmax_agreement_vs_source": agreement,
        "argmax_disagreement_vs_source": 1.0 - agreement,
    }


def _eval_alpha3_vqvae(masks: torch.Tensor, out_dir: Path) -> dict:
    """Evaluate VQ-VAE codec on real masks."""
    print("[α3 vqvae] building codebook...", flush=True)
    cb = build_codebook_top_k(masks, patch_size=4, k=256)
    config = VQVAEConfig(patch_size=4, codebook_size=256, num_classes=5)
    print("[α3 vqvae] encoding...", flush=True)
    blob = encode_vqvae_codec(masks, codebook=cb, config=config)
    print("[α3 vqvae] decoding...", flush=True)
    masks_recovered = decode_vqvae_codec(blob)
    agreement = (masks_recovered == masks).float().mean().item()
    return {
        "candidate": "alpha3_vqvae",
        "config": {
            "patch_size": config.patch_size,
            "codebook_size": config.codebook_size,
            "num_classes": config.num_classes,
        },
        "encoded_bytes": len(blob),
        "argmax_agreement_vs_source": agreement,
        "argmax_disagreement_vs_source": 1.0 - agreement,
    }


def _eval_alpha4_grayscale_lut(masks: torch.Tensor, out_dir: Path) -> dict:
    """Evaluate Selfcomp grayscale-LUT on real masks via AV1 monochrome."""
    out_dir.mkdir(parents=True, exist_ok=True)
    grayscale_mkv = out_dir / "alpha4_grayscale.mkv"
    print("[α4 grayscale-LUT] AV1 monochrome encoding...", flush=True)
    encoded_bytes = _encode_grayscale_av1(masks, grayscale_mkv, crf=50)
    print("[α4 grayscale-LUT] decoding...", flush=True)
    t, h, w = masks.shape
    masks_recovered = _decode_grayscale_av1(grayscale_mkv, t, h, w)
    agreement = (masks_recovered == masks).float().mean().item()
    return {
        "candidate": "alpha4_grayscale_lut",
        "config": {"crf": 50, "class_targets": [0, 255, 64, 192, 128]},
        "encoded_bytes": encoded_bytes,
        "argmax_agreement_vs_source": agreement,
        "argmax_disagreement_vs_source": 1.0 - agreement,
        "grayscale_mkv": str(grayscale_mkv),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive",
        type=Path,
        default=_REPO / "experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "reports",
    )
    p.add_argument(
        "--candidates",
        type=str,
        default="alpha2,alpha3,alpha4",
        help="Comma-separated list of candidates to run (alpha2,alpha3,alpha4).",
    )
    args = p.parse_args()

    # Load real masks from Lane G v3 archive
    print(f"[load] decoding masks from {args.archive}...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(args.archive) as z:
            z.extractall(td_path)
        masks_mkv = td_path / "masks.mkv"
        if not masks_mkv.exists():
            raise FileNotFoundError(f"archive missing masks.mkv: {args.archive}")
        baseline_av1_bytes = masks_mkv.stat().st_size
        masks = _decode_masks_mkv(masks_mkv)

    print(f"[load] masks shape: {tuple(masks.shape)}, dtype: {masks.dtype}", flush=True)
    print(f"[load] baseline AV1 masks.mkv = {baseline_av1_bytes:,} bytes", flush=True)
    print(f"[load] class distribution: {[int((masks == c).sum().item()) for c in range(5)]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = args.candidates.split(",")
    results = {
        "archive": str(args.archive),
        "baseline_av1_bytes": baseline_av1_bytes,
        "masks_shape": list(masks.shape),
        "candidates": {},
    }

    if "alpha2" in candidates:
        r = _eval_alpha2_wavelet(masks, args.output_dir / "alpha2_wavelet_artifacts")
        r["bytes_saved_vs_av1"] = baseline_av1_bytes - r["encoded_bytes"]
        r["pct_savings_vs_av1"] = round(100 * (1 - r["encoded_bytes"] / baseline_av1_bytes), 2)
        results["candidates"]["alpha2_wavelet"] = r
        print(f"[α2 wavelet] {r['encoded_bytes']:,} bytes → {r['pct_savings_vs_av1']}% saved, agreement {r['argmax_agreement_vs_source']:.4f}", flush=True)

    if "alpha3" in candidates:
        r = _eval_alpha3_vqvae(masks, args.output_dir / "alpha3_vqvae_artifacts")
        r["bytes_saved_vs_av1"] = baseline_av1_bytes - r["encoded_bytes"]
        r["pct_savings_vs_av1"] = round(100 * (1 - r["encoded_bytes"] / baseline_av1_bytes), 2)
        results["candidates"]["alpha3_vqvae"] = r
        print(f"[α3 vqvae] {r['encoded_bytes']:,} bytes → {r['pct_savings_vs_av1']}% saved, agreement {r['argmax_agreement_vs_source']:.4f}", flush=True)

    if "alpha4" in candidates:
        r = _eval_alpha4_grayscale_lut(masks, args.output_dir / "alpha4_grayscale_artifacts")
        r["bytes_saved_vs_av1"] = baseline_av1_bytes - r["encoded_bytes"]
        r["pct_savings_vs_av1"] = round(100 * (1 - r["encoded_bytes"] / baseline_av1_bytes), 2)
        results["candidates"]["alpha4_grayscale_lut"] = r
        print(f"[α4 grayscale-LUT] {r['encoded_bytes']:,} bytes → {r['pct_savings_vs_av1']}% saved, agreement {r['argmax_agreement_vs_source']:.4f}", flush=True)

    out_json = args.output_dir / "paradigm_alpha_real_archive.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[empirical:{out_json}] paradigm-α real-archive eval complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
