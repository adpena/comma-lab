#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""NUCLEAR #1: Mask encoding sweep across codecs + CRFs.

Quantizr ships a 293KB archive at score 0.33. Our 2.01 score is dominated by
rate (1.53 = 76% of total). The masks alone are ~250KB at CRF 50; a better
codec/CRF combo could cut that significantly.

This sweep produces an honest tradeoff curve:
  - X: archive bytes (rate-term contribution = 25 * bytes / 37545489)
  - Y: mask disagreement vs the GT extracted by SegNet (proxy for
       SegNet score impact at inflate time)

Sweeps:
  - Monochrome AV1 at CRF 30, 40, 50, 56, 60, 63 (Quantizr uses higher CRF)
  - Lossless entropy coder (frequency_coder) for the 600 odd-frame subset
  - Half-frame mode (600 odd frames) vs full-frame (1200) for each codec

Output: experiments/results/mask_sweep_<timestamp>/
  - results.json (every config + size + disagreement)
  - frontier.md (best size/quality tradeoffs)

Usage:
    PYTHONPATH=src:upstream python experiments/mask_encoding_sweep.py \\
        --masks precomputed_local/masks.pt \\
        --output experiments/results/mask_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Bootstrap project + src + upstream onto sys.path so this script works under
# any cwd (consistent with the engineered_quant_noise R41 fix).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src", _PROJECT_ROOT / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

ORIGINAL_VIDEO_BYTES = 37_545_489  # for rate-term computation
RATE_DENOM = ORIGINAL_VIDEO_BYTES


def disagreement_rate(masks_a: torch.Tensor, masks_b: torch.Tensor) -> float:
    """Fraction of pixels where masks disagree (proxy for SegNet score impact)."""
    return float((masks_a != masks_b).float().mean().item())


def half_frame(masks: torch.Tensor) -> torch.Tensor:
    """Quantizr paradigm: keep only the odd-indexed frames (frame2 of each pair)."""
    return masks[1::2].contiguous()


def reconstruct_full_from_half(half_masks: torch.Tensor) -> torch.Tensor:
    """Naive frame doubling: copy each odd-frame mask to the prior even frame.

    This is the simplest "even-from-odd" warp — identity-warp from the
    immediately-following odd mask. Real warp would use scene flow; this
    upper-bound tells us how lossy the half-frame paradigm is BEFORE we
    add proper warping.
    """
    n_half = half_masks.shape[0]
    full = torch.empty(2 * n_half, *half_masks.shape[1:], dtype=half_masks.dtype)
    full[1::2] = half_masks  # odd frames (the originals)
    full[0::2] = half_masks  # even frames (copy from next odd)
    return full


def sweep_av1_monochrome(
    masks: torch.Tensor,
    out_dir: Path,
    crfs: list[int],
    half: bool,
) -> list[dict]:
    """Sweep monochrome AV1 at multiple CRFs."""
    from tac.mask_codec import decode_masks, encode_masks_monochrome

    results: list[dict] = []
    label = "half" if half else "full"
    src = half_frame(masks) if half else masks

    for crf in crfs:
        path = out_dir / f"masks_av1mono_{label}_crf{crf}.mkv"
        t0 = time.monotonic()
        try:
            size = encode_masks_monochrome(src, path, crf=crf)
        except Exception as exc:  # noqa: BLE001
            results.append({
                "codec": "av1_monochrome",
                "mode": label,
                "crf": crf,
                "error": str(exc),
                "size_bytes": None,
                "rate_term": None,
                "disagreement": None,
            })
            continue
        encode_s = time.monotonic() - t0

        decoded = decode_masks(str(path))
        if half:
            decoded_full = reconstruct_full_from_half(decoded)
        else:
            decoded_full = decoded
        # Trim to match source length (decoder may emit extra frame)
        decoded_full = decoded_full[: masks.shape[0]]
        disagree = disagreement_rate(masks, decoded_full)

        results.append({
            "codec": "av1_monochrome",
            "mode": label,
            "crf": crf,
            "size_bytes": size,
            "rate_term": 25 * size / RATE_DENOM,
            "disagreement": disagree,
            "encode_s": round(encode_s, 1),
            "frames_encoded": int(src.shape[0]),
            "path": str(path),
        })
        print(f"  AV1mono {label} CRF={crf}: {size:>8,} B  rate={25*size/RATE_DENOM:.4f}  "
              f"disagree={disagree:.4f}  ({encode_s:.1f}s)")
    return results


def sweep_entropy_coder(masks: torch.Tensor, out_dir: Path, half: bool) -> list[dict]:
    """Lossless entropy coder via mask_entropy_coder.encode_masks_entropy."""
    from tac.mask_entropy_coder import decode_masks_entropy, encode_masks_entropy

    label = "half" if half else "full"
    src = half_frame(masks) if half else masks
    path = out_dir / f"masks_entropy_{label}.bin"
    t0 = time.monotonic()
    try:
        size = encode_masks_entropy(src, path)
    except Exception as exc:  # noqa: BLE001
        return [{
            "codec": "entropy_lossless",
            "mode": label,
            "size_bytes": None,
            "rate_term": None,
            "disagreement": None,
            "error": str(exc),
        }]
    encode_s = time.monotonic() - t0

    decoded = decode_masks_entropy(path)
    if half:
        decoded_full = reconstruct_full_from_half(decoded)
    else:
        decoded_full = decoded
    decoded_full = decoded_full[: masks.shape[0]]
    disagree = disagreement_rate(masks, decoded_full)

    print(f"  entropy {label}:           {size:>8,} B  rate={25*size/RATE_DENOM:.4f}  "
          f"disagree={disagree:.4f}  ({encode_s:.1f}s)")
    return [{
        "codec": "entropy_lossless",
        "mode": label,
        "size_bytes": size,
        "rate_term": 25 * size / RATE_DENOM,
        "disagreement": disagree,
        "encode_s": round(encode_s, 1),
        "frames_encoded": int(src.shape[0]),
        "path": str(path),
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mask encoding sweep — find smallest mask archive")
    parser.add_argument("--masks", default=str(_PROJECT_ROOT / "precomputed_local" / "masks.pt"),
                        help="Path to cached SegNet GT masks .pt")
    parser.add_argument("--output", default=None,
                        help="Output dir (default: experiments/results/mask_sweep_<ts>)")
    parser.add_argument("--crfs", nargs="+", type=int,
                        default=[30, 40, 50, 56, 60, 63],
                        help="AV1 CRF values to sweep")
    parser.add_argument("--skip-entropy", action="store_true",
                        help="Skip entropy-coder sweep (slow on full 1200 frames)")
    args = parser.parse_args()

    masks_path = Path(args.masks)
    if not masks_path.exists():
        print(f"FATAL: masks not found at {masks_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output) if args.output else (
        _PROJECT_ROOT / "experiments" / "results" /
        f"mask_sweep_{time.strftime('%Y%m%dT%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mask-sweep] Loading masks from {masks_path}")
    masks = torch.load(str(masks_path), map_location="cpu", weights_only=True)
    if masks.dtype != torch.long:
        masks = masks.long()
    n, h, w = masks.shape
    print(f"[mask-sweep] {n} masks at {h}x{w}, classes={masks.max().item() + 1}")
    print(f"[mask-sweep] Output: {out_dir}")
    print()

    all_results: list[dict] = []

    # ── AV1 monochrome ─────────────────────────────────────────────────
    print("[mask-sweep] AV1 monochrome — full frames (1200)")
    all_results.extend(sweep_av1_monochrome(masks, out_dir, args.crfs, half=False))
    print()
    print("[mask-sweep] AV1 monochrome — half frames (600 odd, frame2 of each pair)")
    all_results.extend(sweep_av1_monochrome(masks, out_dir, args.crfs, half=True))
    print()

    # ── Entropy coder ──────────────────────────────────────────────────
    if not args.skip_entropy:
        print("[mask-sweep] Entropy coder — full + half")
        all_results.extend(sweep_entropy_coder(masks, out_dir, half=False))
        all_results.extend(sweep_entropy_coder(masks, out_dir, half=True))
        print()

    # ── Persist + frontier report ──────────────────────────────────────
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps({
        "n_frames": int(n),
        "resolution": [int(h), int(w)],
        "rate_denominator_bytes": RATE_DENOM,
        "results": all_results,
    }, indent=2))

    # Pareto frontier: smallest size at each disagreement bucket
    valid = [r for r in all_results if r.get("size_bytes") is not None]
    valid.sort(key=lambda r: r["size_bytes"])

    lines = [
        "# Mask Encoding Sweep — Pareto Frontier",
        "",
        f"- Frames: {n}, Resolution: {h}x{w}, Classes: {int(masks.max().item()) + 1}",
        f"- Rate denominator: {RATE_DENOM:,} bytes (original video)",
        "",
        "| Codec | Mode | CRF | Size (B) | Rate term | Disagreement |",
        "|-------|------|-----|---------:|----------:|-------------:|",
    ]
    for r in valid:
        lines.append(
            f"| {r['codec']} | {r['mode']} | {r.get('crf', '—')} | "
            f"{r['size_bytes']:,} | {r['rate_term']:.4f} | {r['disagreement']:.4f} |"
        )
    (out_dir / "frontier.md").write_text("\n".join(lines) + "\n")

    print(f"[mask-sweep] Wrote {results_path}")
    print(f"[mask-sweep] Wrote {out_dir / 'frontier.md'}")
    print()
    print("[mask-sweep] Top 5 by smallest size:")
    for r in valid[:5]:
        print(f"  {r['codec']:18s} {r['mode']:5s} CRF={str(r.get('crf', '—')):3s}  "
              f"{r['size_bytes']:>8,} B  rate={r['rate_term']:.4f}  disagree={r['disagreement']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
