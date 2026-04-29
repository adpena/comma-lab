#!/usr/bin/env python3
"""Lane STC-AV1-RESIDUAL smoke (codex top-1 STC redesign at 78% endorsement).

The current Lane STC implementation tries to be a full lossless mask codec
and produces 21MB on 1200 frames (50× regression vs AV1's 421KB) due to a
structural one-majority-class-plus-exceptions encoding flaw.

This redesign uses AV1 as the PREDICTOR and STC encodes only the
CORRECTIONS where AV1 misclassifies. Math:

    base_av1_bytes = AV1 monochrome at chosen CRF (typically 250-340KB)
    residual = clean_argmax XOR av1_decoded_argmax           # sparse
    residual_bytes = arithmetic-coded sparse positions + class deltas

If AV1 misclassifies <5% of pixels at moderate CRF, residual is ~10-30KB.
Total target: 280-370KB, beating raw AV1 at 421KB → -0.020 to -0.038 score.

Compress time is UNLIMITED — at compress time we can:
  - Sweep CRF densely
  - Run SegNet on the full GT video to get the OPTIMAL clean argmax
  - Train a tiny entropy prior on the residual class distribution
  - Use PoseNet-aware loss to weight residual encoding by pose-importance

Inflate time is bound by 30 min on T4 with no scorer load. Both AV1 decode
(CPU ffmpeg) and residual decode (pure integer arithmetic on a class-id
correction stream) are strict-scorer-rule compliant.

This file is the LOCAL CPU SMOKE — produces actual byte counts on Lane A
masks without any CUDA/MPS dependence. The byte count is deterministic
(ffmpeg decode + python integer XOR are bit-identical regardless of host).

Usage:
    python experiments/build_lane_stc_av1_residual_smoke.py \\
        --anchor-archive experiments/results/lane_a_landed/archive_lane_a.zip \\
        --gt-video upstream/videos/0.mkv \\
        --crf-sweep 35 40 45 50 55 \\
        --output-dir experiments/results/lane_stc_av1_residual_smoke

Per the codex max-rigor verdict: hard abandon if no CRF point yields total
mask layer < 380KB. Otherwise scope full lane integration.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT / "src",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _decode_av1_mkv_to_argmax(mkv_path: Path) -> torch.Tensor:
    """Decode an AV1 monochrome masks.mkv back to (N, H, W) class-id tensor."""
    from tac.mask_codec import decode_masks

    return decode_masks(str(mkv_path)).long()


def _encode_clean_to_av1(clean_argmax: torch.Tensor, output_mkv: Path, *, crf: int) -> int:
    """Encode (N, H, W) class-id tensor to AV1 monochrome at given CRF.

    Returns the byte count of the output MKV.
    """
    from tac.mask_codec import encode_masks_monochrome

    encode_masks_monochrome(clean_argmax, str(output_mkv), crf=crf)
    return output_mkv.stat().st_size


def _compute_residual(clean: torch.Tensor, av1_decoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the residual mask: where does AV1 differ from clean?

    Returns:
        diff_mask: (N, H, W) bool — True where av1 != clean
        diff_classes: (N_diff,) long — the CORRECT class at each diff position
    """
    diff_mask = (av1_decoded != clean)
    diff_classes = clean[diff_mask]
    return diff_mask, diff_classes


def _encode_residual_bytes(diff_mask: torch.Tensor, diff_classes: torch.Tensor) -> int:
    """Encode the residual via:
    1. RLE on the diff_mask boolean stream (sparse → tight)
    2. Arithmetic coding on the diff_classes sequence

    Returns the total byte count. Both decoders are pure integer arithmetic,
    strict-scorer-rule compliant.
    """
    from tac.arithmetic_qint_codec import encode_qints_arithmetic

    # Stage 1: RLE-encode the boolean diff_mask stream.
    # For a sparse mask, RLE produces alternating run-lengths of (False, True, False, ...)
    # We store run-lengths as varints. Worst case for dense diff is O(N_pixels) bytes,
    # best case for very sparse diff is O(num_runs * log(run_length)).
    flat = diff_mask.flatten().to(torch.uint8).numpy()
    # Encode as run-lengths via numpy diff.
    transitions = np.where(np.diff(flat.astype(np.int8)) != 0)[0] + 1
    run_starts = np.concatenate([[0], transitions])
    run_lengths = np.diff(np.concatenate([run_starts, [len(flat)]]))
    # Store starting-state (1 byte) + varint-encoded run lengths.
    rle_bytes = bytearray([int(flat[0])])
    for rl in run_lengths:
        # Store as 4-byte little-endian uint32 (simple; later optimize to varint).
        rle_bytes.extend(int(rl).to_bytes(4, "little"))

    # Stage 2: arithmetic-code the diff_classes (one int per diff position).
    # Class IDs are in [0, NUM_CLASSES); encode as int8 array.
    class_array = diff_classes.numpy().astype(np.int8)
    if class_array.size == 0:
        class_blob = b""
    else:
        class_blob = encode_qints_arithmetic(class_array, max_abs=4)

    return len(rle_bytes) + len(class_blob)


def smoke_one_crf(
    clean_argmax: torch.Tensor,
    crf: int,
    work_dir: Path,
) -> dict:
    """Run one full encode-residual-decode cycle at a single CRF."""
    work_dir.mkdir(parents=True, exist_ok=True)
    av1_path = work_dir / f"masks_crf{crf}.mkv"
    av1_bytes = _encode_clean_to_av1(clean_argmax, av1_path, crf=crf)
    av1_decoded = _decode_av1_mkv_to_argmax(av1_path)
    if av1_decoded.shape != clean_argmax.shape:
        raise ValueError(
            f"AV1 decoded shape {av1_decoded.shape} != clean shape {clean_argmax.shape}"
        )
    diff_mask, diff_classes = _compute_residual(clean_argmax, av1_decoded)
    n_diff = int(diff_mask.sum().item())
    n_total = int(diff_mask.numel())
    diff_frac = n_diff / max(n_total, 1)
    residual_bytes = _encode_residual_bytes(diff_mask, diff_classes)
    total_bytes = av1_bytes + residual_bytes
    return {
        "crf": crf,
        "av1_bytes": av1_bytes,
        "residual_bytes": residual_bytes,
        "total_bytes": total_bytes,
        "n_diff_pixels": n_diff,
        "diff_fraction": diff_frac,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lane STC-AV1-residual SMOKE — measure if Hybrid beats AV1"
    )
    parser.add_argument(
        "--clean-masks",
        type=Path,
        help=(
            "Path to a precomputed clean-argmax tensor (.pt) of shape (N, H, W). "
            "If omitted, decode the anchor archive's masks.mkv and treat it as "
            "the 'clean' reference (ONLY for round-trip verification — proper "
            "smoke needs SegNet on GT video at compress time)."
        ),
    )
    parser.add_argument(
        "--anchor-archive",
        type=Path,
        default=_REPO_ROOT / "experiments/results/lane_a_landed/archive_lane_a.zip",
        help="Lane A archive for AV1 byte-count baseline reference",
    )
    parser.add_argument(
        "--crf-sweep",
        type=int,
        nargs="+",
        default=[35, 40, 45, 50, 55],
        help="CRF values to sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "experiments/results/lane_stc_av1_residual_smoke",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Anchor reference for sanity.
    with zipfile.ZipFile(args.anchor_archive) as zf:
        anchor_masks_bytes = zf.getinfo("masks.mkv").file_size
    print(f"[smoke] anchor archive masks.mkv = {anchor_masks_bytes:,} B")

    # Load clean argmax masks.
    if args.clean_masks is not None and args.clean_masks.exists():
        clean = torch.load(args.clean_masks)
    else:
        # FALLBACK: decode the anchor masks.mkv. Note: this is the AV1-decoded
        # argmax (NOT the SegNet output), so the residual will appear small
        # because we're comparing AV1 output to AV1 output. PROPER smoke
        # requires SegNet on GT video — but that step needs CUDA per CLAUDE.md.
        # This fallback is ONLY for verifying the codec scaffolding works.
        print("[smoke] WARNING: using AV1-decoded anchor masks as 'clean' "
              "fallback. The residual will appear small because both sides "
              "are AV1-decoded. Proper smoke requires CUDA SegNet output.")
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(args.anchor_archive) as zf:
                zf.extract("masks.mkv", td)
            clean = _decode_av1_mkv_to_argmax(Path(td) / "masks.mkv")

    print(f"[smoke] clean masks shape: {tuple(clean.shape)}, "
          f"unique classes: {clean.unique().tolist()}")

    results = []
    for crf in args.crf_sweep:
        print(f"[smoke] === CRF {crf} ===")
        with tempfile.TemporaryDirectory() as td:
            r = smoke_one_crf(clean, crf, Path(td))
        results.append(r)
        print(
            f"  av1={r['av1_bytes']:,} B  "
            f"residual={r['residual_bytes']:,} B  "
            f"total={r['total_bytes']:,} B  "
            f"diff_frac={r['diff_fraction']:.4%}"
        )

    # Manifest.
    manifest = {
        "anchor_archive": str(args.anchor_archive),
        "anchor_masks_bytes": anchor_masks_bytes,
        "clean_masks_source": str(args.clean_masks) if args.clean_masks else "av1_anchor_fallback",
        "crf_sweep": args.crf_sweep,
        "results": results,
        "best_total_bytes": min(r["total_bytes"] for r in results),
        "best_crf": min(results, key=lambda r: r["total_bytes"])["crf"],
        "savings_vs_anchor_bytes": anchor_masks_bytes - min(r["total_bytes"] for r in results),
        "council_kill_threshold_bytes": 380_000,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[smoke] wrote {manifest_path}")
    print(f"[smoke] best total = {manifest['best_total_bytes']:,} B at CRF "
          f"{manifest['best_crf']} → savings = "
          f"{manifest['savings_vs_anchor_bytes']:,} B vs anchor {anchor_masks_bytes:,} B")
    if manifest["best_total_bytes"] < 380_000:
        print("[smoke] PASS: best total < 380KB — proceed to CUDA validation")
    else:
        print(f"[smoke] FAIL: best total {manifest['best_total_bytes']:,} B "
              f">= 380KB threshold — codex council says abandon")


if __name__ == "__main__":
    main()
