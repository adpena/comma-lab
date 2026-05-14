#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PARADIGM-α mask payload empirical: Filler STC vs AV1/entropy-coder baseline.

Background (CLAUDE.md "Tomáš Filler"):
    > "Syndrome-trellis coding (STC); parity-check codes for per-frame mask
    >  payload"

REVIEW-SEC-COMP Filler finding: STC opportunity unexploited for PARADIGM-α
mask payload (per-frame mask deltas are ternary — canonical STC target per
Filler-Fridrich-Pevný 2011).

Methodology
-----------
1. Decode the canonical mask stream (``masks.mkv`` from
   ``submissions/robust_current``, the working PARADIGM-α frontier).
2. Compute consecutive frame-deltas projected to ternary {-1, 0, +1}.
3. Encode the ternary stream three ways:
   a. Direct LZMA on ``int8`` raw bytes (the trivial post-encoder baseline).
   b. Existing ``encode_masks_entropy`` LZMA path on the original masks
      (i.e. NO ternary projection — full 5-class delta semantics).
   c. Filler-style ternary STC — embed an all-zero message into each
      block, leaving the cheapest valid-syndrome stego stream (this is the
      "minimum-distortion realization" baseline; the message-bearing variant
      requires a sender/receiver protocol we do not yet specify).
4. Report bytes-per-stream + bytes-per-mask-frame for each encoding.

Falsification scope
-------------------
``filler_stc_ternary_mask_delta_only``: only the ternary STC of consecutive
mask deltas with all-zero message and uniform costs is tested. Score-aware
detector-cost weighting (Yousfi-Fridrich detector-in-loop), ternary coverage
of the message stream, double-layer STC (Filler-Pevný 2010 dual variant), and
the GF(q>2) variant remain in ``reactivation_criteria_remaining``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import lzma
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codec.syndrome_trellis_codec import (  # noqa: E402
    STCParams,
    extract_mask_deltas_ternary,
    ternary_stc_encode_stream,
)
from tac.mask_codec import decode_masks_auto  # noqa: E402
from tac.mask_entropy_coder import encode_masks_entropy  # noqa: E402

TOOL_NAME = "tools/pr_alpha_mask_stc_empirical.py"
SCHEMA_VERSION = "pr_alpha_mask_stc_empirical.v1"
EVIDENCE_GRADE = "[CPU-prep faithful Filler-STC PARADIGM-α test]"


# ---------------------------------------------------------------------------
# Encoding paths
# ---------------------------------------------------------------------------


def encode_lzma_raw_int8(deltas: np.ndarray) -> int:
    """LZMA over the raw int8 ternary stream (trivial baseline)."""
    return len(lzma.compress(deltas.astype(np.int8).tobytes(), preset=6))


def encode_existing_entropy_coder(masks: np.ndarray, tmp_dir: Path) -> int:
    """Existing ``encode_masks_entropy`` path — full 5-class delta semantics."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / "masks_entropy.bin"
    return int(encode_masks_entropy(masks, out, backend="lzma"))


def encode_stc_zero_message(
    deltas: np.ndarray,
    *,
    constraint_height: int,
    block_size: int,
) -> dict:
    """Encode ternary deltas with Filler STC (all-zero message per block).

    Returns the LZMA-compressed stego stream size + raw STC stats.
    """
    costs = np.ones_like(deltas, dtype=np.float64)
    params = STCParams(constraint_height=constraint_height, submatrix_seed=0)
    res = ternary_stc_encode_stream(
        deltas, costs, block_size=block_size, params=params
    )
    stego = res["stego"]
    lzma_bytes = len(lzma.compress(stego.astype(np.int8).tobytes(), preset=6))
    return {
        "lzma_compressed_bytes": lzma_bytes,
        "raw_stego_bytes": int(stego.size),
        "total_flips_soz": int(res["flips_soz"]),
        "total_flips_sign": int(res["flips_sign"]),
        "n_blocks": int(res["n_blocks"]),
        "block_size": int(res["block_size"]),
        "embedding_distortion": float(res["total_cost"]),
    }


# ---------------------------------------------------------------------------
# Top-level experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    masks_path: Path,
    tmp_dir: Path,
    *,
    constraint_height: int,
    block_size: int,
    max_symbols: int | None = None,
) -> dict:
    masks_t = decode_masks_auto(str(masks_path))
    masks_np = (
        masks_t.numpy() if hasattr(masks_t, "numpy") else np.asarray(masks_t)
    ).astype(np.int64)
    n_frames, h, w = masks_np.shape
    deltas_full = extract_mask_deltas_ternary(masks_np)
    # The reference Viterbi STC is O(2^h · n) per block; for the canonical
    # 1199-frame 48×64 mask stream that is ~9×10⁸ Python ops. We allow
    # callers to truncate to a tractable sample for the CPU-prep anchor —
    # the runtime is independent of the upstream production tile and the
    # bytes/symbol ratio is what we care about. Production STC requires a
    # native-Viterbi extension (in ``reactivation_criteria_remaining``).
    if max_symbols is not None and deltas_full.size > max_symbols:
        deltas = deltas_full[:max_symbols].copy()
        sampled = True
    else:
        deltas = deltas_full
        sampled = False

    # Baseline 1: LZMA over the ternary projection
    lzma_ternary_bytes = encode_lzma_raw_int8(deltas)

    # Baseline 2: existing 5-class entropy coder (note: this preserves the full
    # delta semantics; STC on ternary projects out the magnitude bits, so this
    # is an apples-vs-oranges comparison and we tag it as such.)
    av1_baseline_bytes = masks_path.stat().st_size
    entropy_baseline_bytes = encode_existing_entropy_coder(masks_np, tmp_dir)

    # Filler-style STC — minimum-distortion realization of the syndrome
    stc_result = encode_stc_zero_message(
        deltas,
        constraint_height=constraint_height,
        block_size=block_size,
    )

    stc_bytes = stc_result["lzma_compressed_bytes"]
    savings_vs_lzma_ternary = lzma_ternary_bytes - stc_bytes
    savings_vs_av1 = av1_baseline_bytes - stc_bytes

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "family_falsified": False,
        "falsification_scope": "filler_stc_ternary_mask_delta_only",
        "input_masks": str(masks_path),
        "n_frames": int(n_frames),
        "frame_shape": [int(h), int(w)],
        "ternary_delta_count": int(deltas.size),
        "ternary_delta_count_full_stream": int(deltas_full.size),
        "sampled": bool(sampled),
        "constraint_height": int(constraint_height),
        "block_size": int(block_size),
        "av1_baseline_bytes": int(av1_baseline_bytes),
        "entropy_coder_5class_baseline_bytes": int(entropy_baseline_bytes),
        "lzma_raw_ternary_baseline_bytes": int(lzma_ternary_bytes),
        "stc_compressed_bytes": int(stc_bytes),
        "stc_savings_vs_lzma_ternary_bytes": int(savings_vs_lzma_ternary),
        "stc_savings_vs_av1_bytes": int(savings_vs_av1),
        "stc_raw_stego_bytes": stc_result["raw_stego_bytes"],
        "stc_total_flips_soz": stc_result["total_flips_soz"],
        "stc_total_flips_sign": stc_result["total_flips_sign"],
        "stc_n_blocks": stc_result["n_blocks"],
        "stc_embedding_distortion_uniform_cost": stc_result[
            "embedding_distortion"
        ],
        "headline": (
            f"Filler STC ternary mask-delta encoded: {stc_bytes:,} B "
            f"(vs LZMA-ternary {lzma_ternary_bytes:,}, "
            f"vs AV1 baseline {av1_baseline_bytes:,}). "
            f"Δ_vs_LZMA_ternary={savings_vs_lzma_ternary:+,} B"
        ),
        "dispatch_blockers": [
            "stc_zero_message_minimum_distortion_realization_only",
            "no_score_aware_detector_in_loop_costs",
            "no_dual_layer_filler_pevny_stc_variant",
            "missing_exact_cuda_auth_eval",
            "ternary_projection_loses_full_5class_delta_magnitude",
        ],
        "reactivation_criteria_remaining": [
            "score_aware_detector_in_loop_embedding_costs",
            "double_layer_filler_pevny_2010_dual_stc",
            "GF_q_greater_than_2_variant",
            "ternary_stc_with_real_payload_bearing_message_protocol",
            "stc_with_av1_residual_for_full_delta_magnitude",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Filler STC empirical on PARADIGM-α mask payload"
    )
    p.add_argument(
        "--masks",
        type=Path,
        default=REPO_ROOT / "submissions/robust_current/masks.mkv",
        help="Path to masks.mkv (decoded via tac.mask_codec.decode_masks_auto).",
    )
    p.add_argument("--constraint-height", type=int, default=8)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help=(
            "Cap the ternary stream length (the reference Viterbi STC is O(2^h n) "
            "per block; full 1199-frame 48×64 input ⇒ ~3.6M symbols. Default None "
            "= no cap (slow); pass e.g. 32768 for a fast CPU-prep anchor."
        ),
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    p.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Workdir for the entropy-coder baseline (defaults to alongside output).",
    )
    args = p.parse_args(argv)

    if not args.masks.is_file():
        raise SystemExit(f"masks file not found: {args.masks}")

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr_alpha_mask_stc_empirical_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir is None:
        args.tmp_dir = args.output_json.parent / "scratch"

    print(f"PARADIGM-α Filler STC empirical (h={args.constraint_height}, block={args.block_size}, max_symbols={args.max_symbols})")
    manifest = run_experiment(
        args.masks,
        args.tmp_dir,
        constraint_height=args.constraint_height,
        block_size=args.block_size,
        max_symbols=args.max_symbols,
    )
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {args.output_json}\n")

    print(f"  Input masks: {manifest['input_masks']}")
    print(
        f"  N frames: {manifest['n_frames']}, "
        f"shape: {manifest['frame_shape']}, "
        f"ternary deltas: {manifest['ternary_delta_count']:,}"
    )
    print(f"  AV1 baseline:           {manifest['av1_baseline_bytes']:>9,} B")
    print(f"  5-class entropy coder:  {manifest['entropy_coder_5class_baseline_bytes']:>9,} B")
    print(f"  LZMA raw ternary:       {manifest['lzma_raw_ternary_baseline_bytes']:>9,} B")
    print(f"  Filler STC compressed:  {manifest['stc_compressed_bytes']:>9,} B")
    print(f"  Δ vs LZMA-ternary:      {manifest['stc_savings_vs_lzma_ternary_bytes']:>+9,} B")
    print(f"  Δ vs AV1:               {manifest['stc_savings_vs_av1_bytes']:>+9,} B")
    print(f"\n  {manifest['headline']}")

    if args.output_evidence:
        evidence_row = {
            "technique": "filler_stc_ternary_mask_delta_paradigm_alpha",
            "empirical_archive_bytes": manifest["stc_compressed_bytes"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": "filler_stc_ternary_mask_byte_anchor_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": False,
            "family_falsified": False,
            "falsification_scope": "filler_stc_ternary_mask_delta_only",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(stc={manifest['stc_compressed_bytes']:,}; "
                f"av1={manifest['av1_baseline_bytes']:,}; "
                f"lzma_ternary={manifest['lzma_raw_ternary_baseline_bytes']:,}; "
                f"5class_entropy={manifest['entropy_coder_5class_baseline_bytes']:,})"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": [
                "filler_stc_zero_message_uniform_cost"
            ],
            "reactivation_criteria_remaining": manifest[
                "reactivation_criteria_remaining"
            ],
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
