"""Lane H sweep: 8 mask-encoding measurements.

Variants on full (1200 frames) and half (600 odd frames):
  1. AV1 CRF 50 (canonical)        — full + half
  2. AV1 CRF 56                    — full only
  3. AV1 CRF 63                    — full only
  4. Lossless entropy coder (LZMA) — full + half
  5. Entropy raw + Brotli q=11      — full + half

Total: 8 measurements. Reports rate score (bytes / 37545489 * 25).
"""
from __future__ import annotations

import io
import json
import lzma
import sys
import time
from pathlib import Path

PROJECT = Path("/Users/adpena/Projects/pact")
for p in (PROJECT, PROJECT / "src", PROJECT / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import brotli
import numpy as np
import torch

from tac.mask_codec import decode_masks, encode_masks_monochrome
from tac.mask_entropy_coder import (
    _build_fulldelta_payload,
    _build_sparse_payload,
    decode_masks_entropy,
    encode_masks_entropy,
)

RATE_DENOM = 37_545_489
OUT = PROJECT / "experiments" / "results" / "lane_h_mask_sweep"
OUT.mkdir(parents=True, exist_ok=True)


def disagreement(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a != b).float().mean().item())


def half_frame(masks: torch.Tensor) -> torch.Tensor:
    return masks[1::2].contiguous()


def reconstruct_full_from_half(half: torch.Tensor) -> torch.Tensor:
    n_half = half.shape[0]
    full = torch.empty(2 * n_half, *half.shape[1:], dtype=half.dtype)
    full[1::2] = half
    full[0::2] = half
    return full


def run_av1(masks_src: torch.Tensor, gt: torch.Tensor, label: str, crf: int, half: bool) -> dict:
    path = OUT / f"av1_{label}_crf{crf}.mkv"
    t0 = time.monotonic()
    size = encode_masks_monochrome(masks_src, path, crf=crf)
    enc_s = time.monotonic() - t0
    decoded = decode_masks(str(path))
    if half:
        decoded_full = reconstruct_full_from_half(decoded)
    else:
        decoded_full = decoded
    decoded_full = decoded_full[: gt.shape[0]]
    return {
        "codec": "av1_monochrome",
        "mode": label,
        "crf": crf,
        "size_bytes": int(size),
        "rate_term": 25 * size / RATE_DENOM,
        "disagreement": disagreement(gt, decoded_full),
        "encode_s": round(enc_s, 1),
        "frames_encoded": int(masks_src.shape[0]),
    }


def run_entropy(masks_src: torch.Tensor, gt: torch.Tensor, label: str, half: bool) -> dict:
    path = OUT / f"entropy_{label}.bin"
    t0 = time.monotonic()
    size = encode_masks_entropy(masks_src, path)
    enc_s = time.monotonic() - t0
    decoded = decode_masks_entropy(path)
    if half:
        decoded_full = reconstruct_full_from_half(decoded)
    else:
        decoded_full = decoded
    decoded_full = decoded_full[: gt.shape[0]]
    return {
        "codec": "entropy_lossless_lzma",
        "mode": label,
        "crf": None,
        "size_bytes": int(size),
        "rate_term": 25 * size / RATE_DENOM,
        "disagreement": disagreement(gt, decoded_full),
        "encode_s": round(enc_s, 1),
        "frames_encoded": int(masks_src.shape[0]),
    }


def run_entropy_brotli(masks_src: torch.Tensor, gt: torch.Tensor, label: str, half: bool) -> dict:
    """Build the same delta payload as entropy coder, but compress with brotli q=11.

    Picks min(full-delta, sparse) like the LZMA path does, but uses brotli.
    """
    masks_np = masks_src.cpu().numpy().astype(np.uint8)
    payload_full = _build_fulldelta_payload(masks_np)
    payload_sparse = _build_sparse_payload(masks_np)

    t0 = time.monotonic()
    comp_full = brotli.compress(payload_full, quality=11)
    comp_sparse = brotli.compress(payload_sparse, quality=11)
    enc_s = time.monotonic() - t0

    if len(comp_full) <= len(comp_sparse):
        method = "full-delta"
        compressed = comp_full
    else:
        method = "sparse"
        compressed = comp_sparse

    # Persist for transparency (header is delta-payload metadata not roundtrip-needed
    # since lossless brotli vs lossless lzma share same payload)
    out_path = OUT / f"entropy_brotli_{label}.bin"
    out_path.write_bytes(compressed)

    # Lossless on src: roundtrip is exact. But for HALF mode the reconstruction
    # to full-frame uses naive even=copy-of-next-odd, which produces real
    # disagreement against the GT full-frame masks.
    decompressed = brotli.decompress(compressed)
    expected_len = len(payload_full) if method == "full-delta" else len(payload_sparse)
    assert len(decompressed) == expected_len, "brotli roundtrip length mismatch"

    if half:
        # Disagreement comes from half->full reconstruction error (lossless src)
        decoded_full = reconstruct_full_from_half(masks_src)
        decoded_full = decoded_full[: gt.shape[0]]
        disagree = disagreement(gt, decoded_full)
    else:
        disagree = 0.0

    size = out_path.stat().st_size
    print(f"[brotli] {label}: {size:,} B  rate={25*size/RATE_DENOM:.4f}  ({enc_s:.1f}s, method={method}, disagree={disagree:.4f})")
    return {
        "codec": "entropy_lossless_brotli11",
        "mode": label,
        "crf": None,
        "size_bytes": int(size),
        "rate_term": 25 * size / RATE_DENOM,
        "disagreement": disagree,
        "encode_s": round(enc_s, 1),
        "frames_encoded": int(masks_src.shape[0]),
        "compression_method": method,
    }


def main() -> int:
    print("[lane_h] loading masks…")
    masks = torch.load(str(PROJECT / "precomputed_local" / "masks.pt"),
                       map_location="cpu", weights_only=True).long()
    n, h, w = masks.shape
    print(f"[lane_h] masks: {n} frames @ {h}x{w}")
    print()

    masks_full = masks
    masks_half_src = half_frame(masks)
    print(f"[lane_h] half-frame source: {masks_half_src.shape[0]} frames")
    print()

    results: list[dict] = []

    # AV1 sweep
    print("=== AV1 monochrome ===")
    print("[full] CRF 50…")
    results.append(run_av1(masks_full, masks, "full", 50, half=False))
    print("[full] CRF 56…")
    results.append(run_av1(masks_full, masks, "full", 56, half=False))
    print("[full] CRF 63…")
    results.append(run_av1(masks_full, masks, "full", 63, half=False))
    print("[half] CRF 50…")
    results.append(run_av1(masks_half_src, masks, "half", 50, half=True))
    print()

    # Entropy LZMA
    print("=== Entropy LZMA ===")
    print("[full] entropy-lzma…")
    results.append(run_entropy(masks_full, masks, "full", half=False))
    print("[half] entropy-lzma…")
    results.append(run_entropy(masks_half_src, masks, "half", half=True))
    print()

    # Entropy brotli q=11
    print("=== Entropy brotli q=11 ===")
    print("[full] entropy-brotli…")
    results.append(run_entropy_brotli(masks_full, masks, "full", half=False))
    print("[half] entropy-brotli…")
    results.append(run_entropy_brotli(masks_half_src, masks, "half", half=True))
    print()

    # Sort by rate
    results.sort(key=lambda r: r["rate_term"])

    # Write JSON
    summary = {
        "n_frames": n,
        "resolution": [h, w],
        "rate_denominator_bytes": RATE_DENOM,
        "results": results,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))

    # Write markdown table
    lines = [
        "# Lane H — Mask Encoder Sweep",
        "",
        f"- Source: precomputed_local/masks.pt ({n} frames @ {h}x{w})",
        f"- Rate denominator: {RATE_DENOM:,} bytes",
        f"- Rate score = 25 * bytes / denom",
        "",
        "| Codec | Mode | CRF | Size (B) | Rate | Disagreement | Enc (s) |",
        "|-------|------|-----|---------:|-----:|-------------:|--------:|",
    ]
    for r in results:
        crf_s = str(r["crf"]) if r["crf"] is not None else "—"
        lines.append(
            f"| {r['codec']} | {r['mode']} | {crf_s} | {r['size_bytes']:,} | "
            f"{r['rate_term']:.4f} | {r['disagreement']:.4f} | {r['encode_s']} |"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")

    print()
    print("=== SORTED BY RATE (lowest first) ===")
    for r in results:
        crf_s = str(r["crf"]) if r["crf"] is not None else "—"
        print(
            f"  {r['codec']:28s} {r['mode']:5s} CRF={crf_s:>3s}  "
            f"{r['size_bytes']:>8,} B  rate={r['rate_term']:.4f}  "
            f"disagree={r['disagreement']:.4f}  ({r['encode_s']:.1f}s)"
        )

    print()
    print(f"Results: {OUT / 'results.json'}")
    print(f"Report:  {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
