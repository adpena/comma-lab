#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR110 frame-0 optimization bundle sweep (TaskCreate #1313 + #1324 + #1325).

This is a non-promotional sister tool to ``tools/frame_exploit_segnet_posenet_sweep.py``
that adds three new catalogs / analysis variants without mutating the canonical
existing tool (sister-territory per CLAUDE.md "Subagent coherence-by-default" +
Catalog #230 ownership-map discipline):

1. ``frame0_widened`` (PR110-OPT-1): 50-100 frame-0-only candidates from the
   axes documented in the existing sweep tool docstring -- integer luma biases
   (-4..+4), per-channel chroma biases (Cb/Cr separately, +/-1..3), single-pixel
   rolls (vertical / horizontal / diagonal, +/-1 px), 8x8 signed chroma tile
   patterns from a Hadamard / DCT-2 basis, per-frame Gaussian noise at small
   sigma, per-frame quantization-step biases (round-to-nearest-N for N in
   {2,3,4,5,6,7,8}).

2. ``frame0_pose_null`` (PR110-OPT-12): the same widened catalog re-screened
   to rank candidates by |d_pose| ascending. Selects the bottom decile -- the
   "PoseNet-null frame-0" set that pays near-zero pose cost on top of
   frame-0's structural zero-seg cost.

3. ``frame0_tier_split`` (PR110-OPT-13): per-tier K=16 menu split. Reads the
   pair_component_rows.jsonl emitted by the widened sweep, partitions the
   600 pairs (or N-pair smoke subset) into tiers by baseline component score
   (proxy for "hardness" -- harder pairs have larger baseline component), then
   computes the per-tier best-K menu and the predicted unified-K=16 baseline.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192 every row
is tagged ``[macOS-CPU advisory]`` non-promotable. NO ``[contest-CPU]`` or
``[contest-CUDA]`` claims. This tool is INTENDED for offline $0 macOS CPU smoke
discovery to seed paid CUDA dispatch ranking.

Catalog #287: every numeric claim in the emitted JSON manifest carries
``axis_tag="[macOS-CPU advisory]"`` + ``archive_sha256`` + the artifact path.
Catalog #323 canonical Provenance umbrella is honored via the
``promotion_blockers`` list pinned to every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

# Reuse canonical helpers from the sister tool. We intentionally import the
# functions rather than reimplementing them so any future maintenance of the
# canonical sweep tool propagates to this sister.
from frame_exploit_segnet_posenet_sweep import (  # type: ignore[import-not-found] # noqa: E402
    CAMERA_H,
    CAMERA_W,
    CHANNELS,
    DEFAULT_BASELINE_JSON,
    DEFAULT_GT,
    DEFAULT_RAW,
    FRAMES_PER_PAIR,
    PROMOTION_BLOCKERS,
    Mode,
    _apply_mode,
    _blue_tile,
    _candidate_row,
    _decode_gt_pair_indices,
    _decode_gt_pairs,
    _git_head,
    _json_default,
    _load_archive_bytes,
    _load_raw_pair_indices,
    _load_raw_pairs,
    _mode_id_for_rgb,
    _score_pairs,
    _select_device,
    _sha256_file_prefix,
    _write_json,
)

from tac.score_geometry import CONTEST_REFERENCE_BYTES, contest_score  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "experiments/results/pr110_frame0_optimization_bundle_sweep_20260526"


# ---------------------------------------------------------------------------
# Catalog 1: widened frame-0 catalog (PR110-OPT-1, TaskCreate #1313)
# ---------------------------------------------------------------------------


def _hadamard_tile_8() -> np.ndarray:
    """Return the 8x8 Sylvester-construction Hadamard tile with entries +/-1.

    Used as a structured signed chroma tile pattern -- distinct from the
    existing 8x8 ``_blue_tile`` which is a specific signed pattern not based on
    Hadamard / DCT structure. Hadamard rows form an orthogonal basis on the
    8x8 grid; we use selected rows-outer-products as candidate tiles.
    """
    h2 = np.array([[1, 1], [1, -1]], dtype=np.int8)
    h4 = np.kron(h2, h2)
    h8 = np.kron(h4, h2)
    return h8


def _dct_basis_tile_8(u: int, v: int) -> np.ndarray:
    """Return a quantized 8x8 DCT-II basis tile (sign of the basis function).

    The full DCT basis would carry fractional values; signing it produces a
    deterministic 8x8 +/-1 pattern that lives on the frequency-domain axes of
    the tile while remaining integer-valued for our integer-bias arithmetic.
    """
    tile = np.zeros((8, 8), dtype=np.float32)
    for y in range(8):
        for x in range(8):
            tile[y, x] = math.cos(math.pi * (2 * x + 1) * u / 16.0) * math.cos(
                math.pi * (2 * y + 1) * v / 16.0
            )
    return np.sign(tile).astype(np.int8)


def _frame0_widened_catalog() -> list[Mode]:
    """Return ~80 frame-0-only candidates covering 6 axes.

    Axes (per the existing sweep tool's frame-0 inline docs + the operator's
    op-routable expansion):

    - integer luma biases (-4, -3, -2, -1, +1, +2, +3, +4) -> 8 modes
    - integer chroma biases per channel: (R, -R/2, -R/2), (-R, R/2, R/2),
      (G, ...), (B, ...) for R/G/B in {1, 2, 3} -> 18 modes (luma-neutral
      single-channel chroma swings)
    - existing chroma vector lattice (luma-neutral 6-vector lattice with
      max-abs <= 4 and sum-abs <= 6): 8 modes (operator-set; preserved from
      existing catalog for cross-reference)
    - single-pixel rolls vert/horiz/diag (8 directions for (dx,dy) in {-1,0,1}
      excluding (0,0)) -> 8 modes
    - 8x8 chroma tile patterns: existing blue_tile amp{1,2,3} (preserved
      from existing catalog) + Hadamard tile amp{1,2,3} + DCT basis (u,v) in
      {(1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2)} amp{1,2}
      -> 3 + 3 + 16 = 22 modes
    - per-frame Gaussian noise at small sigma in {0.5, 1.0, 1.5, 2.0} (4 seeds
      each, deterministic per seed) -> 16 modes
    - per-frame quantization round-to-nearest-N for N in {2,3,4,5,6,7,8} -> 7
      modes

    Total: 8 + 18 + 8 + 8 + 22 + 16 + 7 = 87 candidates plus the identity
    baseline.
    """
    modes: list[Mode] = [Mode("none", "identity", {}, "No transform control")]

    # Axis 1: integer luma biases (frame-0 only)
    for bias in (-4, -3, -2, -1, 1, 2, 3, 4):
        modes.append(
            Mode(
                f"frame0_widened_luma_bias_{bias:+d}",
                "frame0_luma_bias",
                {"rgb_delta": [bias, bias, bias]},
                f"Widened: integer luma RGB bias {bias:+d} frame-0 only",
            )
        )

    # Axis 2: integer per-channel chroma biases (luma-neutral)
    # For R/G/B each, push the channel by +/-R while pulling the other two by
    # -/+R/2 so the unweighted luma stays approximately constant.
    for amp in (1, 2, 3):
        for axis_idx, name in enumerate(("r", "g", "b")):
            for sign in (+1, -1):
                vec = [-(sign * amp) // 2, -(sign * amp) // 2, -(sign * amp) // 2]
                vec[axis_idx] = sign * amp
                # Repair the divided sign for odd amp so the rounding lands
                # symmetric: when amp=1 -> [+1, 0, 0] or [+1, -1, 0] both
                # are valid; we use the symmetric integer rounding.
                modes.append(
                    Mode(
                        f"frame0_widened_chroma_{name}{sign:+d}_amp{amp}",
                        "frame0_rgb_bias",
                        {"rgb_delta": list(vec)},
                        f"Widened: chroma-{name} {sign:+d}*{amp} luma-neutral frame-0 only",
                    )
                )

    # Axis 3: existing 8-vector chroma lattice (preserved from canonical
    # catalog so widened results stay back-comparable)
    for vec in (
        (0, -1, 1),
        (0, 1, -1),
        (2, -1, -1),
        (-2, 1, 1),
        (0, -2, 2),
        (0, 2, -2),
        (4, -2, -2),
        (-4, 2, 2),
    ):
        modes.append(
            Mode(
                _mode_id_for_rgb("frame0_widened_lattice", vec),
                "frame0_rgb_bias",
                {"rgb_delta": list(vec)},
                "Widened: luma-neutral 6-vector chroma lattice frame-0 only",
            )
        )

    # Axis 4: single-pixel rolls (8 directions)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            modes.append(
                Mode(
                    f"frame0_widened_roll_dx{dx:+d}_dy{dy:+d}",
                    "frame0_roll",
                    {"dx": dx, "dy": dy},
                    f"Widened: 1-pixel roll dx={dx:+d} dy={dy:+d} frame-0 only",
                )
            )

    # Axis 5a: existing blue-chroma tile (preserved from canonical)
    for amp in (1, 2, 3):
        modes.append(
            Mode(
                f"frame0_widened_blue_chroma_amp_{amp}",
                "frame0_blue_chroma",
                {"amp": amp},
                "Widened: existing blue chroma tile (sign pattern) frame-0 only",
            )
        )

    # Axis 5b: Hadamard tile family (signed chroma R<->B swap)
    for amp in (1, 2, 3):
        modes.append(
            Mode(
                f"frame0_widened_hadamard_chroma_amp_{amp}",
                "frame0_hadamard_chroma",
                {"amp": amp},
                f"Widened: 8x8 Hadamard tile chroma R-B swap amp {amp} frame-0 only",
            )
        )

    # Axis 5c: DCT-II basis tile family (8 frequency bins x 2 amplitudes)
    for u, v in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)):
        for amp in (1, 2):
            modes.append(
                Mode(
                    f"frame0_widened_dct_u{u}_v{v}_amp_{amp}",
                    "frame0_dct_chroma",
                    {"u": u, "v": v, "amp": amp},
                    f"Widened: 8x8 DCT-II sign-basis (u={u},v={v}) amp {amp} frame-0 only",
                )
            )

    # Axis 6: per-frame Gaussian noise at small sigma (deterministic seed)
    for sigma in (0.5, 1.0, 1.5, 2.0):
        for seed in (1, 2, 3, 4):
            modes.append(
                Mode(
                    f"frame0_widened_noise_sigma_{sigma:.1f}_seed_{seed}",
                    "frame0_gaussian_noise",
                    {"sigma": sigma, "seed": seed},
                    f"Widened: Gaussian noise sigma={sigma} seed={seed} frame-0 only",
                )
            )

    # Axis 7: per-frame quantization round-to-nearest-N
    for n_step in (2, 3, 4, 5, 6, 7, 8):
        modes.append(
            Mode(
                f"frame0_widened_quant_step_{n_step}",
                "frame0_quant_step",
                {"n_step": n_step},
                f"Widened: round-to-nearest-{n_step} per-channel quantization frame-0 only",
            )
        )

    return modes


# ---------------------------------------------------------------------------
# Mode application extensions for new families
# ---------------------------------------------------------------------------


def _apply_widened_mode(pairs: torch.Tensor, mode: Mode) -> torch.Tensor:
    """Apply a widened-catalog mode. Delegates to canonical ``_apply_mode``
    for families it already handles; implements the 3 new families
    (hadamard_chroma, dct_chroma, gaussian_noise, quant_step) locally."""
    family = mode.family
    if family in {
        "identity",
        "frame0_luma_bias",
        "frame0_rgb_bias",
        "frame0_blue_chroma",
        "frame0_roll",
        "frame1_luma_bias",
        "frame1_rgb_bias",
        "frame1_blue_chroma",
    }:
        return _apply_mode(pairs, mode)

    out = pairs.to(dtype=torch.float32).clone()
    f0 = out[:, 0]

    if family == "frame0_hadamard_chroma":
        amp = float(mode.params["amp"])
        h8 = torch.from_numpy(_hadamard_tile_8()).to(out.device).to(out.dtype)
        reps_h = (CAMERA_H + 7) // 8
        reps_w = (CAMERA_W + 7) // 8
        tiled = h8.repeat(reps_h, reps_w)[:CAMERA_H, :CAMERA_W].view(
            1, CAMERA_H, CAMERA_W, 1
        )
        f0[..., 0:1].add_(tiled * amp)
        f0[..., 2:3].sub_(tiled * amp)
    elif family == "frame0_dct_chroma":
        amp = float(mode.params["amp"])
        u = int(mode.params["u"])
        v = int(mode.params["v"])
        tile_np = _dct_basis_tile_8(u, v)
        tile = torch.from_numpy(tile_np).to(out.device).to(out.dtype)
        reps_h = (CAMERA_H + 7) // 8
        reps_w = (CAMERA_W + 7) // 8
        tiled = tile.repeat(reps_h, reps_w)[:CAMERA_H, :CAMERA_W].view(
            1, CAMERA_H, CAMERA_W, 1
        )
        f0[..., 0:1].add_(tiled * amp)
        f0[..., 2:3].sub_(tiled * amp)
    elif family == "frame0_gaussian_noise":
        sigma = float(mode.params["sigma"])
        seed = int(mode.params["seed"])
        # Deterministic per-pair Gaussian noise. We use np.random for portability
        # across torch devices (MPS / CPU); generate once per pair shape.
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(size=(f0.shape[0], CAMERA_H, CAMERA_W, CHANNELS)).astype(
            np.float32
        ) * sigma
        f0.add_(torch.from_numpy(noise).to(out.device).to(out.dtype))
    elif family == "frame0_quant_step":
        n_step = int(mode.params["n_step"])
        # Round each frame-0 pixel to the nearest multiple of n_step.
        f0.copy_((f0 / n_step).round_().mul_(n_step))
    else:
        raise AssertionError(f"unhandled widened mode family: {family}")
    return out.clamp_(0.0, 255.0).round_()


# ---------------------------------------------------------------------------
# Selector + tier-split analysis (PR110-OPT-13, TaskCreate #1325)
# ---------------------------------------------------------------------------


def _per_pair_baseline_component_scores(rows_by_mode: dict[str, dict[str, Any]]) -> list[tuple[int, float]]:
    """Extract per-pair component-score-no-rate for the identity baseline,
    sorted ascending (easy -> hard). Returns list of (pair_id, score)."""
    baseline = rows_by_mode["none"]["metrics"]
    return [
        (int(p["pair"]), float(p["component_score_no_rate"]))
        for p in baseline["per_pair"]
    ]


def _tier_split_analysis(
    rows_by_mode: dict[str, dict[str, Any]],
    *,
    k_unified: int,
    tier_definitions: list[tuple[str, float, float]],
    seg_guard_delta: float = 0.0,
) -> dict[str, Any]:
    """Compute per-tier K-best menus and compare aggregate vs unified-K.

    tier_definitions: list of (tier_name, lo_quantile, hi_quantile). The
    function partitions pairs by baseline component-score-no-rate (proxy
    for "hardness") and reports (a) per-tier best menu of size up to
    ``k_unified // n_tiers + 1`` and (b) the unified K=``k_unified`` menu
    computed globally.

    All numbers are macOS-CPU advisory proxies. Per-tier menu mean
    component delta vs global unified menu mean component delta is the
    headline signal.
    """
    baseline_per_pair = _per_pair_baseline_component_scores(rows_by_mode)
    if not baseline_per_pair:
        return {"tier_analysis": [], "unified_menu": None}

    scores_sorted = sorted(baseline_per_pair, key=lambda x: x[1])
    n_pairs = len(scores_sorted)
    quantile_values = [score for _, score in scores_sorted]

    def _quantile_threshold(q: float) -> float:
        if q <= 0.0:
            return float("-inf")
        if q >= 1.0:
            return float("inf")
        idx = int(q * (n_pairs - 1))
        return quantile_values[idx]

    mode_ids = [mid for mid in rows_by_mode if mid != "none"]

    def _pair_component_delta(mode_id: str, pair_index: int) -> tuple[float, float, float]:
        """Return (component_score_delta, pose_delta, seg_delta) for one pair."""
        baseline_per = rows_by_mode["none"]["metrics"]["per_pair"]
        cand_per = rows_by_mode[mode_id]["metrics"]["per_pair"]
        # per_pair lists share order with pair_ids; pair_index here is the
        # index into that ordered list, not the underlying pair_id
        b = baseline_per[pair_index]
        c = cand_per[pair_index]
        return (
            float(c["component_score_no_rate"] - b["component_score_no_rate"]),
            float(c["posenet_dist"] - b["posenet_dist"]),
            float(c["segnet_dist"] - b["segnet_dist"]),
        )

    def _best_menu_for_pair_set(pair_indices: list[int], k_size: int) -> tuple[list[str], float]:
        """Greedy + exhaustive K-best menu: rank modes by aggregate component
        delta over the pair set, keep top-K, return menu mode_ids + mean per-pair
        component delta when selector chooses the in-menu mode minimizing per-pair
        delta with seg-guard."""
        if not pair_indices or k_size <= 0:
            return ([], 0.0)
        candidate_aggregate = []
        for mid in mode_ids:
            total = 0.0
            valid_pairs = 0
            for pi in pair_indices:
                delta, _, seg_delta = _pair_component_delta(mid, pi)
                if seg_delta > seg_guard_delta:
                    continue
                total += delta
                valid_pairs += 1
            if valid_pairs == 0:
                continue
            candidate_aggregate.append((mid, total / valid_pairs))
        candidate_aggregate.sort(key=lambda x: x[1])
        menu = [mid for mid, _ in candidate_aggregate[:k_size]]
        # Now compute the per-pair best-in-menu mean delta
        per_pair_deltas: list[float] = []
        for pi in pair_indices:
            best_delta = 0.0  # "none" is always a fallback (delta=0)
            for mid in menu:
                delta, _, seg_delta = _pair_component_delta(mid, pi)
                if seg_delta > seg_guard_delta:
                    continue
                if delta < best_delta:
                    best_delta = delta
            per_pair_deltas.append(best_delta)
        mean_delta = float(sum(per_pair_deltas) / max(len(per_pair_deltas), 1)) if per_pair_deltas else 0.0
        return (menu, mean_delta)

    # Compute the unified-K menu over ALL pairs first.
    all_pair_indices = list(range(n_pairs))
    unified_menu, unified_mean_delta = _best_menu_for_pair_set(all_pair_indices, k_unified)

    tier_results = []
    for tier_name, lo_q, hi_q in tier_definitions:
        lo_threshold = _quantile_threshold(lo_q)
        hi_threshold = _quantile_threshold(hi_q)
        tier_pair_indices = [
            idx for idx, (_, score) in enumerate(scores_sorted) if lo_threshold <= score < hi_threshold
        ]
        if hi_q >= 1.0:
            tier_pair_indices = [
                idx for idx, (_, score) in enumerate(scores_sorted) if lo_threshold <= score <= hi_threshold
            ]
        # Per-tier K = unified K split proportionally
        n_tiers = len(tier_definitions)
        tier_k = max(1, k_unified // n_tiers + 1)
        tier_menu, tier_mean_delta = _best_menu_for_pair_set(tier_pair_indices, tier_k)

        # Also compute the unified menu's mean delta on this tier (apples-to-apples)
        unified_per_pair_deltas: list[float] = []
        for pi in tier_pair_indices:
            best_delta = 0.0
            for mid in unified_menu:
                delta, _, seg_delta = _pair_component_delta(mid, pi)
                if seg_delta > seg_guard_delta:
                    continue
                if delta < best_delta:
                    best_delta = delta
            unified_per_pair_deltas.append(best_delta)
        unified_on_tier_mean_delta = (
            float(sum(unified_per_pair_deltas) / max(len(unified_per_pair_deltas), 1))
            if unified_per_pair_deltas
            else 0.0
        )

        tier_results.append(
            {
                "tier_name": tier_name,
                "quantile_range": [lo_q, hi_q],
                "n_pairs": len(tier_pair_indices),
                "tier_menu": tier_menu,
                "tier_menu_size_k": tier_k,
                "tier_menu_mean_component_delta": tier_mean_delta,
                "unified_menu_on_tier_mean_component_delta": unified_on_tier_mean_delta,
                "tier_vs_unified_advantage": unified_on_tier_mean_delta - tier_mean_delta,
                "axis_tag": "[macOS-CPU advisory]",
            }
        )

    # Aggregate tier-split mean across the whole sample weighted by pair count
    n_total = sum(t["n_pairs"] for t in tier_results)
    aggregate_tier_split_mean = (
        sum(t["tier_menu_mean_component_delta"] * t["n_pairs"] for t in tier_results) / max(n_total, 1)
        if n_total
        else 0.0
    )

    return {
        "k_unified": k_unified,
        "unified_menu": unified_menu,
        "unified_menu_mean_component_delta": unified_mean_delta,
        "tier_definitions": [list(t) for t in tier_definitions],
        "tier_results": tier_results,
        "aggregate_tier_split_mean_component_delta": aggregate_tier_split_mean,
        "aggregate_advantage_vs_unified": unified_mean_delta - aggregate_tier_split_mean,
        "seg_guard_delta": seg_guard_delta,
        "axis_tag": "[macOS-CPU advisory]",
        "promotion_blockers": list(PROMOTION_BLOCKERS),
    }


# ---------------------------------------------------------------------------
# PoseNet-null analysis (PR110-OPT-12, TaskCreate #1324)
# ---------------------------------------------------------------------------


def _posenet_null_analysis(
    candidate_rows: list[dict[str, Any]],
    *,
    top_n: int = 30,
    pose_null_quantile: float = 0.10,
) -> dict[str, Any]:
    """Rank frame-0 candidates by absolute PoseNet delta ascending and return
    the bottom decile -- the PoseNet-null frame-0 set."""
    non_identity = [r for r in candidate_rows if r["mode_id"] != "none"]
    if not non_identity:
        return {
            "top_n": top_n,
            "pose_null_decile": [],
            "pose_null_quantile": pose_null_quantile,
            "axis_tag": "[macOS-CPU advisory]",
            "promotion_blockers": list(PROMOTION_BLOCKERS),
        }
    ranked_by_abs_pose = sorted(non_identity, key=lambda r: abs(r["posenet_delta_vs_none"]))
    quantile_count = max(1, int(len(ranked_by_abs_pose) * pose_null_quantile))
    decile = ranked_by_abs_pose[:quantile_count]

    return {
        "top_n": top_n,
        "pose_null_decile_count": quantile_count,
        "pose_null_quantile": pose_null_quantile,
        "ranked_top_n_by_abs_pose": [
            {
                "mode_id": r["mode_id"],
                "family": r["family"],
                "abs_pose_delta": abs(r["posenet_delta_vs_none"]),
                "pose_delta": r["posenet_delta_vs_none"],
                "seg_delta": r["segnet_delta_vs_none"],
                "score_delta_proxy": r["score_delta_vs_none_proxy"],
                "axis_tag": "[macOS-CPU advisory]",
            }
            for r in ranked_by_abs_pose[:top_n]
        ],
        "pose_null_decile": [
            {
                "mode_id": r["mode_id"],
                "family": r["family"],
                "abs_pose_delta": abs(r["posenet_delta_vs_none"]),
                "pose_delta": r["posenet_delta_vs_none"],
                "seg_delta": r["segnet_delta_vs_none"],
                "score_delta_proxy": r["score_delta_vs_none_proxy"],
                "axis_tag": "[macOS-CPU advisory]",
            }
            for r in decile
        ],
        "axis_tag": "[macOS-CPU advisory]",
        "promotion_blockers": list(PROMOTION_BLOCKERS),
    }


# ---------------------------------------------------------------------------
# Top-10 widened analysis (PR110-OPT-1)
# ---------------------------------------------------------------------------


def _widened_top_n_analysis(candidate_rows: list[dict[str, Any]], *, top_n: int = 10) -> dict[str, Any]:
    """Rank widened frame-0 candidates by predicted score delta ascending."""
    non_identity = [r for r in candidate_rows if r["mode_id"] != "none"]
    ranked_by_score = sorted(non_identity, key=lambda r: r["score_delta_vs_none_proxy"])
    safe_ranked = [
        r
        for r in ranked_by_score
        if r["score_delta_vs_none_proxy"] < 0.0
        and r["segnet_delta_vs_none"] <= 0.0
    ]
    return {
        "top_n": top_n,
        "ranked_top_n_by_score_delta": [
            {
                "mode_id": r["mode_id"],
                "family": r["family"],
                "params": r["params"],
                "score_delta_proxy": r["score_delta_vs_none_proxy"],
                "pose_delta": r["posenet_delta_vs_none"],
                "seg_delta": r["segnet_delta_vs_none"],
                "axis_tag": "[macOS-CPU advisory]",
            }
            for r in ranked_by_score[:top_n]
        ],
        "safe_top_n_seg_zero_or_negative": [
            {
                "mode_id": r["mode_id"],
                "family": r["family"],
                "params": r["params"],
                "score_delta_proxy": r["score_delta_vs_none_proxy"],
                "pose_delta": r["posenet_delta_vs_none"],
                "seg_delta": r["segnet_delta_vs_none"],
                "axis_tag": "[macOS-CPU advisory]",
            }
            for r in safe_ranked[:top_n]
        ],
        "axis_tag": "[macOS-CPU advisory]",
        "promotion_blockers": list(PROMOTION_BLOCKERS),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-raw", type=Path, default=DEFAULT_RAW, help="Inflated RGB24 0.raw candidate stream")
    parser.add_argument("--gt-video", type=Path, default=DEFAULT_GT, help="Ground-truth contest video")
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON, help="Optional auth-eval JSON for archive bytes")
    parser.add_argument("--archive-bytes", type=int, help="Override archive bytes used in formula proxy (default 178517 for fec6 6bae0201 sha)", default=178517)
    parser.add_argument("--archive-sha256", type=str, default="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf", help="Target archive sha256 for provenance stamp")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Artifact directory")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="Explicit local device; no auto fallback")
    parser.add_argument("--start-pair", type=int, default=0, help="First non-overlapping pair index")
    parser.add_argument("--n-pairs", type=int, default=8, help="Number of non-overlapping pairs to screen (smoke default; use 600 for full)")
    parser.add_argument("--pair-indices", default="", help="Optional comma-separated non-contiguous pair ids; overrides --start-pair/--n-pairs")
    parser.add_argument("--batch-size", type=int, default=2, help="Scorer batch size in pairs")
    parser.add_argument("--seg-guard-delta", type=float, default=0.0, help="Per-pair selector max allowed SegNet delta")
    parser.add_argument("--k-unified", type=int, default=16, help="Unified-menu size K for tier-split analysis")
    parser.add_argument("--pose-null-quantile", type=float, default=0.10, help="Quantile of |d_pose| ascending to count as PoseNet-null (default bottom 10%)")
    return parser.parse_args(argv)


def _parse_pair_indices(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    indices: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        pair = int(item)
        if pair < 0:
            raise SystemExit("--pair-indices values must be non-negative")
        if pair in seen:
            raise SystemExit(f"--pair-indices contains duplicate pair {pair}")
        seen.add(pair)
        indices.append(pair)
    return indices or None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start_pair < 0 or args.n_pairs <= 0 or args.batch_size <= 0:
        raise SystemExit("start-pair must be >=0 and n-pairs/batch-size must be >0")

    started = time.time()
    device = _select_device(args.device)
    archive_bytes = _load_archive_bytes(
        args.baseline_json if args.baseline_json and args.baseline_json.is_file() else None,
        args.archive_bytes,
    )
    if archive_bytes <= 0:
        raise SystemExit("archive bytes unavailable; pass --archive-bytes or --baseline-json")

    pair_indices = _parse_pair_indices(args.pair_indices)
    if pair_indices is not None:
        gt_pairs = _decode_gt_pair_indices(args.gt_video, pair_indices=pair_indices)
        raw_pairs = _load_raw_pair_indices(args.candidate_raw, pair_indices=pair_indices)
        pair_ids = pair_indices
    else:
        gt_pairs = _decode_gt_pairs(args.gt_video, start_pair=args.start_pair, n_pairs=args.n_pairs)
        raw_pairs = _load_raw_pairs(args.candidate_raw, start_pair=args.start_pair, n_pairs=args.n_pairs)
        pair_ids = list(range(args.start_pair, args.start_pair + args.n_pairs))

    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore[import-not-found]

    distortion_net = DistortionNet().eval().to(device=device)
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    # Catalog 1: Widened frame-0 sweep
    widened_modes = _frame0_widened_catalog()
    rows_by_mode: dict[str, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None
    for mode in widened_modes:
        transformed = _apply_widened_mode(raw_pairs, mode)
        metrics = _score_pairs(
            distortion_net,
            gt_pairs,
            transformed,
            device=device,
            batch_size=args.batch_size,
            archive_bytes=archive_bytes,
            pair_ids=pair_ids,
        )
        if mode.mode_id == "none":
            baseline_metrics = metrics
        if baseline_metrics is None:
            raise AssertionError("identity mode must be first")
        row = _candidate_row(mode, metrics, baseline_metrics, device=device)
        row["axis_tag"] = "[macOS-CPU advisory]"
        candidate_rows.append(row)
        rows_by_mode[mode.mode_id] = {"mode": mode, "metrics": metrics, "row": row}

    widened_top_n = _widened_top_n_analysis(candidate_rows, top_n=10)
    pose_null_analysis = _posenet_null_analysis(
        candidate_rows, top_n=30, pose_null_quantile=args.pose_null_quantile
    )
    tier_split_analysis = _tier_split_analysis(
        rows_by_mode,
        k_unified=args.k_unified,
        tier_definitions=[
            ("easy_lower_third", 0.0, 1.0 / 3.0),
            ("middle_third", 1.0 / 3.0, 2.0 / 3.0),
            ("hard_upper_third", 2.0 / 3.0, 1.0),
        ],
        seg_guard_delta=args.seg_guard_delta,
    )

    command = [sys.executable, Path(__file__).as_posix(), *sys.argv[1:]]
    common_provenance = {
        "archive_sha256": args.archive_sha256,
        "archive_bytes": archive_bytes,
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "macOS-CPU advisory only" if device.type == "cpu" else "MPS-research-signal",
        "device": device.type,
        "measurement_hardware": (
            "darwin_arm64_macos_cpu_advisory" if device.type == "cpu" else "darwin_arm64_macos_mps_research_signal"
        ),
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "command": command,
        "repo_root": REPO_ROOT.as_posix(),
        "git_head": _git_head(),
        "n_pairs_sampled": len(pair_ids),
        "pair_ids": list(pair_ids),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Bundle manifest (composite)
    bundle_manifest = {
        "schema": "pr110_frame0_optimization_bundle_sweep.v1",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "lane_id": "lane_pr110_frame0_optimization_bundle_sweep_20260526",
        "provenance": common_provenance,
        "bundles": {
            "pr110_opt_1_widened_frame0": {
                "task_create_id": 1313,
                "n_widened_modes": len(widened_modes) - 1,
                "top_n_analysis": widened_top_n,
                "output_file": "pr110_opt1_widened_frame0_top10.json",
            },
            "pr110_opt_12_posenet_null_frame0": {
                "task_create_id": 1324,
                "pose_null_analysis": pose_null_analysis,
                "output_file": "pr110_opt12_posenet_null_frame0.json",
            },
            "pr110_opt_13_tier_split_k16": {
                "task_create_id": 1325,
                "tier_split_analysis": tier_split_analysis,
                "output_file": "pr110_opt13_tier_split_k16.json",
            },
        },
        "baseline_none": rows_by_mode["none"]["row"],
        "elapsed_seconds": time.time() - started,
    }

    bundle_manifest_path = args.output_dir / "pr110_frame0_bundle_sweep_manifest.json"
    _write_json(bundle_manifest_path, bundle_manifest)

    # Per-bundle artifact JSONs for canonical state ingestion
    timestamp_compact = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    opt1_path = REPO_ROOT / f".omx/state/pr110_opt1_widened_frame0_top10_{timestamp_compact}.json"
    opt12_path = REPO_ROOT / f".omx/state/pr110_opt12_posenet_null_frame0_{timestamp_compact}.json"
    opt13_path = REPO_ROOT / f".omx/state/pr110_opt13_tier_split_k16_{timestamp_compact}.json"
    _write_json(opt1_path, {"provenance": common_provenance, "analysis": widened_top_n})
    _write_json(opt12_path, {"provenance": common_provenance, "analysis": pose_null_analysis})
    _write_json(opt13_path, {"provenance": common_provenance, "analysis": tier_split_analysis})

    print(f"wrote {bundle_manifest_path}")
    print(f"wrote {opt1_path}")
    print(f"wrote {opt12_path}")
    print(f"wrote {opt13_path}")
    if widened_top_n["safe_top_n_seg_zero_or_negative"]:
        top = widened_top_n["safe_top_n_seg_zero_or_negative"][0]
        print(
            f"opt1 best_safe {top['mode_id']} score_delta={top['score_delta_proxy']:+.9f} [macOS-CPU advisory]"
        )
    else:
        print("opt1 best_safe none [macOS-CPU advisory]")
    if pose_null_analysis["pose_null_decile"]:
        top_null = pose_null_analysis["pose_null_decile"][0]
        print(
            f"opt12 pose_null_top {top_null['mode_id']} |pose_delta|={top_null['abs_pose_delta']:.9g} seg_delta={top_null['seg_delta']:+.9g} [macOS-CPU advisory]"
        )
    print(
        f"opt13 unified_K={args.k_unified} mean_delta={tier_split_analysis['unified_menu_mean_component_delta']:+.9f} "
        f"tier_split_mean={tier_split_analysis['aggregate_tier_split_mean_component_delta']:+.9f} "
        f"advantage={tier_split_analysis['aggregate_advantage_vs_unified']:+.9f} [macOS-CPU advisory]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
