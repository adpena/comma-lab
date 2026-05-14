#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Empirical: Filler-Pevný 2010 dual-layer STC + magnitude-residual codec.

Reactivation criterion (per
``feedback_uniward_stc_research_lanes_landed_20260508.md``):
    "STC + AV1 residual hybrid (recover the magnitude bit dropped by ternary
     projection)"

Dual-layer construction (Filler & Pevný 2010 IEEE TIFS 5(2):388-393):
    Layer 1 — sign(deltas) ∈ {-1, 0, +1}, brotli-compressed.
    Layer 2 — abs(deltas) at non-zero positions, brotli-compressed.
    Joint reconstruction is LOSSLESS for the original 5-class delta stream.

Methodology
-----------
1. Decode the canonical mask stream (``masks.mkv`` from
   ``submissions/robust_current``).
2. Compute the FULL 5-class delta stream (preserves magnitude;
   ``masks[t] - masks[t-1]`` ∈ ``{-4..+4}``).
3. Encode the stream four ways:
   a. Filler STC (single-layer ternary): the stream's SIGN through STC; this
      is the existing ``pr_alpha_mask_stc_empirical.py`` baseline (LOSSY: drops
      magnitude).
   b. AV1 baseline: the canonical ``masks.mkv`` file size on disk (LOSSY:
      AV1 monochrome).
   c. 5-class entropy coder (LOSSLESS): the canonical
      ``encode_masks_entropy`` LZMA path on the original masks.
   d. Dual-layer STC + brotli magnitude-residual (LOSSLESS, this lane).
4. Verify (d) roundtrips losslessly.
5. Report bytes-per-stream + savings vs each baseline.

Hypothesis: dual-layer beats AV1 and the 5-class entropy coder on lossless
reconstruction by combining sign-stream brevity (LZMA-friendly ternary) with
magnitude-residual sparsity (most non-zero deltas at class boundaries are ±1).

Custody (CLAUDE.md non-negotiables)
-----------------------------------
- ``score_claim=False`` (CPU-only byte anchor).
- ``promotion_eligible=False``.
- ``ready_for_exact_eval_dispatch=False`` (no scorer load anywhere).
- ``family_falsified=False`` per CLAUDE.md
  ``forbidden_premature_class_level_falsification`` — any negative regime is
  ``MEASURED_CONFIG_NOT_DISPATCHABLE``.
- ``falsification_scope = "dual_layer_stc_av1_lossless_only"``.
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

from tac.codec.dual_layer_stc_av1_codec import (  # noqa: E402
    decode_dual_layer,
    encode_dual_layer,
    extract_full_mask_deltas,
)
from tac.codec.syndrome_trellis_codec import (  # noqa: E402
    STCParams,
    extract_mask_deltas_ternary,
    ternary_stc_encode_stream,
)
from tac.mask_codec import decode_masks_auto  # noqa: E402
from tac.mask_entropy_coder import encode_masks_entropy  # noqa: E402

TOOL_NAME = "tools/pr_alpha_mask_dual_layer_stc_empirical.py"
SCHEMA_VERSION = "pr_alpha_mask_dual_layer_stc_empirical.v1"
EVIDENCE_GRADE = "[CPU-prep faithful Filler-Pevny-2010-dual-layer STC+AV1 test]"


# ---------------------------------------------------------------------------
# Encoding paths
# ---------------------------------------------------------------------------


def encode_av1_baseline_size(masks_path: Path) -> int:
    """AV1 baseline = the canonical masks.mkv file size on disk."""
    return int(masks_path.stat().st_size)


def encode_existing_5class_entropy(masks: np.ndarray, tmp_dir: Path) -> int:
    """Canonical 5-class lossless entropy coder (LZMA backend)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / "masks_entropy.bin"
    return int(encode_masks_entropy(masks, out, backend="lzma"))


def encode_filler_stc_single_layer(
    ternary_deltas: np.ndarray,
    *,
    constraint_height: int,
    block_size: int,
) -> dict:
    """Existing single-layer Filler STC (LOSSY: ternary projection only).

    Mirrors the per-stream path in ``tools/pr_alpha_mask_stc_empirical.py``.
    """
    costs = np.ones_like(ternary_deltas, dtype=np.float64)
    params = STCParams(constraint_height=constraint_height, submatrix_seed=0)
    res = ternary_stc_encode_stream(
        ternary_deltas, costs, block_size=block_size, params=params
    )
    stego = res["stego"]
    lzma_bytes = len(lzma.compress(stego.astype(np.int8).tobytes(), preset=6))
    return {
        "lzma_compressed_bytes": int(lzma_bytes),
        "raw_stego_bytes": int(stego.size),
        "total_flips_soz": int(res["flips_soz"]),
        "total_flips_sign": int(res["flips_sign"]),
        "n_blocks": int(res["n_blocks"]),
        "block_size": int(res["block_size"]),
        "embedding_distortion": float(res["total_cost"]),
    }


def encode_dual_layer_full(
    full_deltas: np.ndarray,
    *,
    constraint_height: int,
    block_size: int,
    record_stc_stats: bool,
) -> dict:
    """Lossless dual-layer encode + roundtrip verification."""
    if record_stc_stats:
        blob, stats = encode_dual_layer(
            full_deltas,
            constraint_height=constraint_height,
            block_size=block_size,
            record_stc_stats=True,
        )
        stats_dict = {
            "layer1_bytes": int(stats.layer1_bytes),
            "layer2_bytes": int(stats.layer2_bytes),
            "header_bytes": int(stats.header_bytes),
            "total_bytes": int(stats.total_bytes),
            "n_symbols": int(stats.n_symbols),
            "n_nonzero": int(stats.n_nonzero),
            "nonzero_fraction": float(stats.nonzero_fraction),
            "stc_uniform_block_size": int(stats.stc_uniform_block_size),
            "stc_uniform_constraint_height": int(stats.stc_uniform_constraint_height),
            "stc_uniform_n_blocks": int(stats.stc_uniform_n_blocks),
            "stc_uniform_flips_soz": int(stats.stc_uniform_flips_soz),
            "stc_uniform_flips_sign": int(stats.stc_uniform_flips_sign),
            "stc_uniform_total_cost": float(stats.stc_uniform_total_cost),
        }
    else:
        blob = encode_dual_layer(full_deltas)
        stats_dict = {
            "layer1_bytes": None,
            "layer2_bytes": None,
            "header_bytes": None,
            "total_bytes": int(len(blob)),
            "n_symbols": int(full_deltas.size),
            "n_nonzero": int((full_deltas != 0).sum()),
            "nonzero_fraction": float((full_deltas != 0).sum()) / float(full_deltas.size),
            "stc_uniform_block_size": None,
            "stc_uniform_constraint_height": None,
            "stc_uniform_n_blocks": None,
            "stc_uniform_flips_soz": None,
            "stc_uniform_flips_sign": None,
            "stc_uniform_total_cost": None,
        }

    # Roundtrip verification (NON-NEGOTIABLE for a "lossless" claim).
    roundtrip = decode_dual_layer(blob)
    if not np.array_equal(roundtrip, full_deltas.astype(np.int16)):
        raise RuntimeError(
            "dual-layer roundtrip FAILED — encode/decode are not lossless inverses"
        )

    return {
        "compressed_bytes": int(len(blob)),
        "lossless_roundtrip_verified": True,
        **stats_dict,
    }


# ---------------------------------------------------------------------------
# Synthetic mask cube (used when canonical masks.mkv is unavailable)
# ---------------------------------------------------------------------------


def synthesize_mask_cube(
    n_frames: int = 32, h: int = 12, w: int = 16, seed: int = 2026
) -> np.ndarray:
    """Skewed 5-class mask cube approximating the canonical PARADIGM-α stream.

    Most pixels stay in the dominant "road" class (0); a sparse fraction
    switches between classes. Designed to give a non-trivial delta stream
    where the magnitude statistics are plausibly mask-like.
    """
    rng = np.random.default_rng(seed)
    base = rng.choice(
        [0, 0, 0, 0, 1, 1, 2, 3, 4],
        size=(n_frames, h, w),
        replace=True,
    ).astype(np.int64)
    # Inject a 5% temporal drift so consecutive frames differ.
    flip_mask = rng.random(size=base.shape) < 0.05
    perturb = rng.integers(0, 5, size=base.shape, dtype=np.int64)
    base = np.where(flip_mask, perturb, base)
    return base


# ---------------------------------------------------------------------------
# Top-level experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    masks_path: Path | None,
    tmp_dir: Path,
    *,
    constraint_height: int,
    block_size: int,
    max_symbols: int | None = None,
    record_stc_stats: bool = False,
) -> dict:
    if masks_path is not None and masks_path.is_file():
        masks_t = decode_masks_auto(str(masks_path))
        masks_np = (
            masks_t.numpy() if hasattr(masks_t, "numpy") else np.asarray(masks_t)
        ).astype(np.int64)
        masks_source = str(masks_path)
        av1_baseline_bytes = encode_av1_baseline_size(masks_path)
    else:
        masks_np = synthesize_mask_cube()
        masks_source = "synthetic_5class_skewed"
        av1_baseline_bytes = -1  # not applicable

    n_frames, h, w = masks_np.shape
    full_deltas = extract_full_mask_deltas(masks_np)
    ternary_deltas = extract_mask_deltas_ternary(masks_np)
    if full_deltas.size != ternary_deltas.size:
        raise RuntimeError(
            f"full/ternary delta-count mismatch ({full_deltas.size} vs {ternary_deltas.size})"
        )

    if max_symbols is not None and full_deltas.size > max_symbols:
        full_deltas = full_deltas[:max_symbols].copy()
        ternary_deltas = ternary_deltas[:max_symbols].copy()
        sampled = True
    else:
        sampled = False

    # 5-class lossless entropy coder uses the FULL masks (it expects a 3-D
    # mask cube, not a 1-D delta stream). When sampled we restrict to the
    # frames that produced the truncated delta stream.
    if sampled:
        # Floor-divide by per-frame pixel count to find the inclusive frame index.
        symbols_per_diff_frame = h * w
        n_diff_frames = max_symbols // symbols_per_diff_frame
        masks_for_entropy = masks_np[: n_diff_frames + 1]
    else:
        masks_for_entropy = masks_np
    entropy_5class_bytes = encode_existing_5class_entropy(masks_for_entropy, tmp_dir)

    # (a) Filler STC single-layer (LOSSY ternary projection)
    filler_stc_single = encode_filler_stc_single_layer(
        ternary_deltas,
        constraint_height=constraint_height,
        block_size=block_size,
    )

    # (d) Dual-layer STC + brotli magnitude-residual (LOSSLESS, this lane)
    dual_layer = encode_dual_layer_full(
        full_deltas,
        constraint_height=constraint_height,
        block_size=block_size,
        record_stc_stats=record_stc_stats,
    )

    dual_bytes = dual_layer["compressed_bytes"]
    filler_single_bytes = filler_stc_single["lzma_compressed_bytes"]

    if av1_baseline_bytes > 0:
        savings_vs_av1 = av1_baseline_bytes - dual_bytes
        ratio_vs_av1 = float(dual_bytes) / float(av1_baseline_bytes)
    else:
        savings_vs_av1 = None
        ratio_vs_av1 = None
    savings_vs_filler_single = filler_single_bytes - dual_bytes
    savings_vs_5class_entropy = entropy_5class_bytes - dual_bytes
    ratio_vs_5class_entropy = float(dual_bytes) / float(entropy_5class_bytes)

    # Hypothesis check: lossless dual-layer beats lossless 5-class entropy?
    # (NOTE: STC-single is LOSSY — it drops magnitude. The fair lossless-vs-
    # lossless comparison is dual-layer vs 5-class entropy.)
    hypothesis_dual_beats_5class_entropy = bool(savings_vs_5class_entropy > 0)

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
        "falsification_scope": "dual_layer_stc_av1_lossless_only",
        "input_masks": masks_source,
        "n_frames": int(n_frames),
        "frame_shape": [int(h), int(w)],
        "delta_count": int(full_deltas.size),
        "delta_count_full_stream": int(masks_np.shape[0] - 1) * int(h) * int(w),
        "sampled": bool(sampled),
        "constraint_height": int(constraint_height),
        "block_size": int(block_size),
        # Baselines
        "av1_baseline_bytes": int(av1_baseline_bytes) if av1_baseline_bytes > 0 else None,
        "entropy_coder_5class_lossless_bytes": int(entropy_5class_bytes),
        "filler_stc_single_layer_lossy_bytes": int(filler_single_bytes),
        # Dual-layer (this lane)
        "dual_layer_compressed_bytes": int(dual_bytes),
        "dual_layer_lossless_roundtrip_verified": bool(
            dual_layer["lossless_roundtrip_verified"]
        ),
        "dual_layer_layer1_bytes": dual_layer["layer1_bytes"],
        "dual_layer_layer2_bytes": dual_layer["layer2_bytes"],
        "dual_layer_header_bytes": dual_layer["header_bytes"],
        "dual_layer_n_nonzero": dual_layer["n_nonzero"],
        "dual_layer_nonzero_fraction": dual_layer["nonzero_fraction"],
        # Stats (optional; gated on --record-stc-stats)
        "stc_uniform_n_blocks": dual_layer["stc_uniform_n_blocks"],
        "stc_uniform_flips_soz": dual_layer["stc_uniform_flips_soz"],
        "stc_uniform_flips_sign": dual_layer["stc_uniform_flips_sign"],
        "stc_uniform_total_cost": dual_layer["stc_uniform_total_cost"],
        # Comparisons
        "savings_vs_av1_bytes": int(savings_vs_av1) if savings_vs_av1 is not None else None,
        "ratio_vs_av1": ratio_vs_av1,
        "savings_vs_filler_stc_single_layer_bytes_LOSSY_apples_oranges": int(
            savings_vs_filler_single
        ),
        "savings_vs_5class_entropy_bytes": int(savings_vs_5class_entropy),
        "ratio_vs_5class_entropy": ratio_vs_5class_entropy,
        "hypothesis_dual_beats_5class_entropy_lossless": hypothesis_dual_beats_5class_entropy,
        "filler_stc_total_flips_soz": int(filler_stc_single["total_flips_soz"]),
        "filler_stc_total_flips_sign": int(filler_stc_single["total_flips_sign"]),
        "filler_stc_n_blocks": int(filler_stc_single["n_blocks"]),
        "headline": (
            f"Dual-layer STC+brotli (LOSSLESS) encoded {full_deltas.size:,} "
            f"5-class deltas to {dual_bytes:,} B "
            f"(vs 5class-entropy {entropy_5class_bytes:,}, "
            f"vs AV1 {av1_baseline_bytes if av1_baseline_bytes > 0 else 'N/A'}, "
            f"vs Filler-STC-single LOSSY {filler_single_bytes:,}). "
            f"Δ_vs_5class_entropy={savings_vs_5class_entropy:+,} B"
        ),
        "dispatch_blockers": [
            "no_score_aware_detector_in_loop_costs",
            "no_GF_q_greater_than_2_variant",
            "missing_exact_cuda_auth_eval",
            "lossy_av1_layer_2_substitution_not_yet_implemented",
        ],
        "reactivation_criteria_remaining": [
            "score_aware_detector_in_loop_embedding_costs_for_layer_1",
            "GF_q_greater_than_2_STC_to_eliminate_dual_layer",
            "lossy_av1_monochrome_layer_2_with_mask_quality_budget",
            "native_viterbi_extension_for_full_stream_research_signal",
            "exact_cuda_auth_eval_on_archive_with_dual_layer_payload",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Filler-Pevný 2010 dual-layer STC + magnitude-residual empirical"
    )
    p.add_argument(
        "--masks",
        type=Path,
        default=REPO_ROOT / "submissions/robust_current/masks.mkv",
        help="Path to masks.mkv (decoded via tac.mask_codec.decode_masks_auto). "
             "If missing, falls back to a synthetic 5-class mask cube.",
    )
    p.add_argument("--constraint-height", type=int, default=8)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help=(
            "Cap the delta stream length. STC stats (if --record-stc-stats) are "
            "O(2^h n) per block; default None caps via the dataset size."
        ),
    )
    p.add_argument(
        "--record-stc-stats",
        action="store_true",
        help="Run the uniform-cost STC trellis on the sign stream as a research signal "
        "(does NOT affect the lossless wire format). Default off (cheap).",
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

    masks_path: Path | None = args.masks if args.masks.is_file() else None

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr_alpha_mask_dual_layer_stc_empirical_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir is None:
        args.tmp_dir = args.output_json.parent / "scratch"

    print(
        f"PARADIGM-α dual-layer Filler-Pevný 2010 STC+brotli empirical "
        f"(h={args.constraint_height}, block={args.block_size}, "
        f"max_symbols={args.max_symbols}, stats={args.record_stc_stats})"
    )
    if masks_path is None:
        print(
            f"  WARN: canonical masks file not found at {args.masks} — "
            "running on synthetic mask cube."
        )

    manifest = run_experiment(
        masks_path,
        args.tmp_dir,
        constraint_height=args.constraint_height,
        block_size=args.block_size,
        max_symbols=args.max_symbols,
        record_stc_stats=args.record_stc_stats,
    )
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {args.output_json}\n")

    print(f"  Input masks:      {manifest['input_masks']}")
    print(
        f"  N frames: {manifest['n_frames']}, "
        f"shape: {manifest['frame_shape']}, "
        f"deltas: {manifest['delta_count']:,}"
    )
    if manifest["av1_baseline_bytes"] is not None:
        print(f"  AV1 baseline (lossy):                  {manifest['av1_baseline_bytes']:>9,} B")
    print(
        f"  5-class entropy coder (LOSSLESS):       "
        f"{manifest['entropy_coder_5class_lossless_bytes']:>9,} B"
    )
    print(
        f"  Filler-STC single-layer (LOSSY):        "
        f"{manifest['filler_stc_single_layer_lossy_bytes']:>9,} B"
    )
    print(
        f"  Dual-layer STC+brotli (LOSSLESS, this): "
        f"{manifest['dual_layer_compressed_bytes']:>9,} B"
    )
    print(f"  Δ vs 5-class entropy:  {manifest['savings_vs_5class_entropy_bytes']:>+9,} B "
          f"({(1 - manifest['ratio_vs_5class_entropy']) * 100:+.1f}%)")
    if manifest["savings_vs_av1_bytes"] is not None:
        print(f"  Δ vs AV1 (lossy):      {manifest['savings_vs_av1_bytes']:>+9,} B "
              f"({(1 - manifest['ratio_vs_av1']) * 100:+.1f}%)")
    print(f"\n  {manifest['headline']}")

    if args.output_evidence:
        evidence_row = {
            "technique": "filler_pevny_2010_dual_layer_stc_av1_paradigm_alpha",
            "empirical_archive_bytes": manifest["dual_layer_compressed_bytes"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": (
                "filler_pevny_2010_dual_layer_stc_av1_lossless_byte_anchor_no_score"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": bool(
                manifest["hypothesis_dual_beats_5class_entropy_lossless"]
            ),
            "family_falsified": False,
            "falsification_scope": "dual_layer_stc_av1_lossless_only",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(dual_layer={manifest['dual_layer_compressed_bytes']:,}; "
                f"5class_entropy={manifest['entropy_coder_5class_lossless_bytes']:,}; "
                f"av1={manifest['av1_baseline_bytes']}; "
                f"filler_stc_single={manifest['filler_stc_single_layer_lossy_bytes']:,})"
            ),
            "contest_dispatch_verdict": (
                "MEASURED_CONFIG_NOT_DISPATCHABLE"
                if manifest["hypothesis_dual_beats_5class_entropy_lossless"]
                else "DEFERRED-pending-research"
            ),
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": [
                "filler_pevny_2010_dual_layer_brotli_lossless"
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
