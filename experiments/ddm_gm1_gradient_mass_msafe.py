#!/usr/bin/env python3
"""ddm_gm1 -- where does the seg gradient MASS land, split at the n600 m_safe?

WHAT THIS MEASURES AND WHY IT IS NOT sd1
----------------------------------------
sd1 (``experiments/ddm_sd1_surrogate_exact_map.py``) measured that 67-85% of the expected-flip
gradient lands on pixels that are ALREADY CORRECT (85.10% at tau=0.15, 67.18% at tau=0.05).  That
single number fuses two populations with opposite meanings:

  * a correct pixel whose margin is INSIDE the R-noise band is genuinely at risk -- the
    render->camera->uint8 roundtrip can still flip it, so gradient spent there defends a real flip;
  * a correct pixel whose margin is OUTSIDE the band cannot be un-done by R, so gradient spent
    there buys nothing the score can see.  That is the WASTE the margin-band satisficing law
    (``margin_band_satisficing_threshold_v1``) exists to cap.

This instrument splits sd1's "already-correct" share at ``m_safe`` and reports the three-way
partition per GT class, per milestone, per cell, over a tau grid that reaches down to the physical
scale of the noise floor itself (0.5 delta_R).  The split is the input the third race needs:
"85% wasted" and "X% wasted" are different charters.

THE PARTITION (exact, MEASURED, never a proxy)
----------------------------------------------
With ``margin = logits[GT] - max_{c != GT} logits[c]`` (``ddm_qbt1_qbflow_trainer.py:523-548``) and
the flip indicator taken from the RETAINED argmax (sd1's float16-tie cure, ``:1907-1936``):

  group 0  CORRECT_OUTSIDE : not flipped and margin  >  m_safe   -- R cannot undo it; WASTE
  group 1  CORRECT_INSIDE  : not flipped and margin <=  m_safe   -- R can still undo it; DEFENCE
  group 2  WRONG           : flipped                             -- REPAIR

``m_safe`` is RESOLVED THROUGH THE LAW (``resolve_margin_band_threshold``), never carried as a
literal -- dr1's addendum repointed the artifact to n600, so a literal here would be a split bank
([[m107]]).  Per-class ``m_safe_c = headroom * delta_R_c`` uses dr1's per-class p95 read from its
receipts artifact, so the class-vs-global over-push is measured in the units the cap uses.

THE TWO MASSES
--------------
``surrogate`` is ``sigmoid(-m/tau)`` -- what the loss COUNTS.
``grad`` is ``|d sigmoid(-m/tau)/dm| = sigmoid(z)sigmoid(-z)/tau`` -- what the loss can MOVE.
Only the second steers descent, so every share in this instrument is a GRADIENT share and says so.

ROW 1 (vr1 ``_live_margin_weight``)
-----------------------------------
``experiments/train_witness_realized_through_R_mlx.py:1086`` weights each pixel by an allocator over
the LIVE **top1-top2** gap -- which is UNSIGNED, and therefore is NOT ``|signed margin|`` on the
~2% of flips where GT is not the runner-up.  This instrument computes the gap the way that code
does (``sort(logits)[-1] - sort(logits)[-2]``), mean-1 normalizes per pair as the code does, and
MEASURES the divergence from ``|signed margin|`` rather than assuming it away.

AXIS
----
``[macOS-CPU advisory; retained EMA-shadow scorer logits; frozen CPU-torch SegNet argmax;
non-promotable]``.  No score claim; this arm measures inputs and cannot move a pointer.

ALWAYS KEEP THE PAYLOAD: every per-(milestone, lineage) bin table is written to JSON with sha256s
and the per-pair rows are appended to a JSONL as they are produced.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_qbt1_qbflow_trainer as qbt
from experiments import ddm_sd1_surrogate_exact_map as sd1  # instrument REUSED, never rebuilt
from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
    resolve_margin_band_threshold,
)

INSTRUMENT = "ddm_gm1_gradient_mass_msafe"
SCHEMA = "ddm_gm1_gradient_mass_msafe.v1"
AXIS = (
    "[macOS-CPU advisory; retained EMA-shadow scorer logits; frozen CPU-torch SegNet argmax; "
    "non-promotable]"
)

CLASS_NAMES = sd1.CLASS_NAMES
N_CLASSES = sd1.N_CLASSES
LINEAGES = sd1.LINEAGES

GROUP_NAMES = ("correct_outside", "correct_inside", "wrong")
N_GROUPS = len(GROUP_NAMES)

# dr1 per-class annulus p95, read from its receipts artifact (never a literal in this file).
DR1_RECEIPTS = Path(
    "/Volumes/APDataStore/pact/ddm_dr1_delta_R_n600/delta_R_receipts_n600.json"
)

# vr1 row 1 allocator grid.  The SHIPPED defaults are inverse / temp 1.0 with a documented anneal
# target near 0.3 (``train_witness_realized_through_R_mlx.py:3293,3299,3314-3316``); bottom-k 0.05 is
# the measured concentration (FEED-bp: 89.2% of d_seg in the bottom-5%-margin pixels).
ROW1_CONFIGS: tuple[tuple[str, float], ...] = (
    ("inverse", 1.0),
    ("inverse", 0.3),
    ("inverse", 0.1),
    ("exp", 1.0),
    ("exp", 0.3),
    ("exp", 0.1),
    ("bottom-k", 0.05),
    ("bottom-k", 0.01),
)


class GM1Error(RuntimeError):
    """Fail-closed error for the gm1 instrument."""


# ---------------------------------------------------------------------------
# thresholds, resolved through the law
# ---------------------------------------------------------------------------
def resolve_thresholds(receipts_path: Path = DR1_RECEIPTS) -> dict[str, Any]:
    """Resolve the global ``m_safe`` through the canonical law and per-class ``m_safe_c``.

    The global value comes from ``resolve_margin_band_threshold`` so this instrument cannot drift
    from whatever the live consumers resolve.  The per-class values apply the SAME DERIVED headroom
    to each class's own MEASURED ``delta_R_c`` (dr1's construction, and dr1's caveat travels with
    it: headroom is a policy factor, not a per-class measurement).
    """

    resolution = resolve_margin_band_threshold()
    headroom = float(resolution.headroom)
    per_class_delta: dict[str, float] = {}
    per_class_m_safe: dict[str, float] = {}
    receipts_fact: dict[str, Any] | None = None
    if Path(receipts_path).is_file():
        blob = Path(receipts_path).read_bytes()
        receipts_fact = {
            "path": str(receipts_path),
            "bytes": len(blob),
            "sha256": sd1.sha256_bytes(blob),
        }
        pooled = json.loads(blob)["per_class_annulus_pooled"]
        for name in CLASS_NAMES:
            if name not in pooled:
                raise GM1Error(f"dr1 receipts lack per-class delta_R for {name}")
            per_class_delta[name] = float(pooled[name]["p95"])
            per_class_m_safe[name] = headroom * per_class_delta[name]
    return {
        "delta_r": float(resolution.delta_r),
        "headroom": headroom,
        "m_safe": float(resolution.m_safe),
        "n_frames": int(resolution.n_frames),
        "artifact_path": str(resolution.artifact_path),
        "artifact_fallback_used": bool(resolution.artifact_fallback_used),
        "lawref_fallback_used": bool(resolution.lawref_fallback_used),
        "full_r_annulus_p95": float(resolution.full_r_annulus_p95),
        "per_class_delta_r": per_class_delta,
        "per_class_m_safe": per_class_m_safe,
        "dr1_receipts": receipts_fact,
    }


def tau_grid(delta_r: float, extra: Sequence[float] = ()) -> list[float]:
    """The charter's six taus, a log scan for the crossing solve, and any milestone's own tau."""

    charter = [0.15, 0.10, 0.05, 2.0 * delta_r, delta_r, 0.5 * delta_r]
    scan = list(np.exp(np.linspace(np.log(0.004), np.log(0.20), 31)))
    merged = sorted({round(float(t), 12) for t in (*charter, *scan, *extra)})
    return merged


CHARTER_TAU_LABELS = ("0.15", "0.10", "0.05", "2dR", "1dR", "0.5dR")


def charter_taus(delta_r: float) -> dict[str, float]:
    values = (0.15, 0.10, 0.05, 2.0 * delta_r, delta_r, 0.5 * delta_r)
    return {
        label: round(float(value), 12)
        for label, value in zip(CHARTER_TAU_LABELS, values, strict=True)
    }


# ---------------------------------------------------------------------------
# the partition
# ---------------------------------------------------------------------------
def group_of(margin: np.ndarray, flip: np.ndarray, m_safe: float) -> np.ndarray:
    """Return the three-way group index per pixel (0 outside, 1 inside, 2 wrong)."""

    wrong = np.asarray(flip, dtype=bool)
    inside = np.abs(np.asarray(margin, dtype=np.float64)) <= float(m_safe)
    group = np.where(wrong, 2, np.where(inside, 1, 0)).astype(np.int64)
    return group


def per_class_group(
    margin: np.ndarray, flip: np.ndarray, gt: np.ndarray, m_safe_by_class: Sequence[float]
) -> np.ndarray:
    """Three-way group using each pixel's OWN class threshold ``m_safe_c``."""

    thresholds = np.asarray(m_safe_by_class, dtype=np.float64)
    if thresholds.shape != (N_CLASSES,):
        raise GM1Error(f"per-class m_safe geometry differs: {thresholds.shape}")
    per_pixel = thresholds[np.asarray(gt, dtype=np.int64)]
    wrong = np.asarray(flip, dtype=bool)
    inside = np.abs(np.asarray(margin, dtype=np.float64)) <= per_pixel
    return np.where(wrong, 2, np.where(inside, 1, 0)).astype(np.int64)


# bin layout: (gt * N_GROUPS + global_group) * N_GROUPS + class_group
N_BINS = N_CLASSES * N_GROUPS * N_GROUPS


def bin_index(gt: np.ndarray, group_global: np.ndarray, group_class: np.ndarray) -> np.ndarray:
    return (np.asarray(gt, dtype=np.int64) * N_GROUPS + group_global) * N_GROUPS + group_class


def decode_bins(vector: np.ndarray) -> np.ndarray:
    """Reshape a length-``N_BINS`` vector to ``(gt, global_group, class_group)``."""

    return np.asarray(vector, dtype=np.float64).reshape(N_CLASSES, N_GROUPS, N_GROUPS)


def grad_magnitude(margin: np.ndarray, tau: float) -> np.ndarray:
    """``|d sigmoid(-m/tau)/dm| = sigmoid(z)sigmoid(-z)/tau`` with ``z = m/tau``.

    Computed as ``a/(1+a)^2`` with ``a = exp(-|z|)`` rather than as ``p*(1-p)``.  MEASURED reason,
    not style: the loss forms ``p = sigmoid(-m/tau)``, so on a confidently-WRONG pixel (``m << 0``)
    ``p -> 1`` and ``p*(1-p)`` computes ``1 - (1 - e^-|z|)`` -- catastrophic cancellation against
    one float64 ulp of 1.0.  MEASURED relative error of the naive form: 3.6e-08 at ``z = 20``,
    4.2e-06 at ``z = 25``, 1.0e-03 at ``z = 30``.  (The already-CORRECT tail is exact under the same
    branch split; a first reading of this had the sign backwards and the trace corrected it.)  The
    ``a/(1+a)^2`` form has no subtraction of nearly-equal quantities, underflows gracefully to 0,
    and is EXACTLY even in ``m`` by construction -- which is the mathematical truth: the quantity
    that steers descent depends only on the distance to the boundary, not on which side of it the
    pixel sits.  Aggregate significance here is below 1e-12 of the total mass (those pixels carry
    ~e^-z of it): hygiene on an exact group share, not a finding.
    """

    if not tau > 0:
        raise GM1Error("tau must be positive")
    a = np.exp(-np.abs(np.asarray(margin, dtype=np.float64)) / float(tau))
    return a / np.square(1.0 + a) / float(tau)


def top1_top2_gap(logits: np.ndarray) -> np.ndarray:
    """The UNSIGNED top1-top2 logit gap -- exactly what ``_live_margin_weight`` consumes.

    ``train_witness_realized_through_R_mlx.py:1107-1108`` sorts the class axis and subtracts the two
    largest.  This is NOT ``|signed margin|`` wherever GT is not the runner-up on a flipped pixel;
    the divergence is measured, never assumed.
    """

    work = np.asarray(logits, dtype=np.float32)
    if work.ndim != 3 or work.shape[0] != N_CLASSES:
        raise GM1Error(f"logits geometry differs: {work.shape}")
    ordered = np.sort(work, axis=0)
    return (ordered[-1] - ordered[-2]).astype(np.float64)


def row1_weight(gap: np.ndarray, fn: str, temp: float) -> np.ndarray:
    """``_live_margin_weight`` in numpy: allocator over the gap, mean-1, stop-grad.

    Mean-1 is applied over the pixels handed in -- one pair -- matching the per-forward
    normalization of the MLX original (``:1104``).  stop-grad is a no-op here because this
    instrument never differentiates the weight; it is recorded so the reading stays honest.
    """

    values = np.asarray(gap, dtype=np.float64)
    scale = max(float(temp), 1e-6)
    if fn == "exp":
        weight = np.exp(-values / scale)
    elif fn == "bottom-k":
        flat = values.reshape(-1)
        count = flat.size
        # int(round(...)) mirrors the MLX original verbatim
        # (train_witness_realized_through_R_mlx.py:1115); faithfulness to the ported
        # source outranks the redundant-cast lint.
        k = min(max(int(round(scale * count)), 1), count)  # noqa: RUF046
        threshold = np.partition(flat, k - 1)[k - 1]
        weight = (values <= threshold).astype(np.float64)
    elif fn == "inverse":
        weight = 1.0 / (1.0 + values / scale)
    else:
        raise GM1Error(f"unknown live-margin allocator: {fn}")
    return weight / (weight.mean() + 1e-8)


# ---------------------------------------------------------------------------
# per-pair accumulation
# ---------------------------------------------------------------------------
def accumulate_pair(
    margin: np.ndarray,
    gt: np.ndarray,
    flip: np.ndarray,
    gap: np.ndarray,
    taus: Sequence[float],
    row1_taus: Sequence[float],
    m_safe: float,
    m_safe_by_class: Sequence[float],
) -> dict[str, Any]:
    """Bin one pair and return pixel / surrogate / gradient / row1-weighted-gradient mass."""

    flat_m = np.asarray(margin, dtype=np.float64).reshape(-1)
    flat_gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    flat_flip = np.asarray(flip, dtype=bool).reshape(-1)
    flat_gap = np.asarray(gap, dtype=np.float64).reshape(-1)

    group_global = group_of(flat_m, flat_flip, m_safe)
    group_class = per_class_group(flat_m, flat_flip, flat_gt, m_safe_by_class)
    idx = bin_index(flat_gt, group_global, group_class)

    out: dict[str, Any] = {
        "pixels": np.bincount(idx, minlength=N_BINS).astype(np.float64),
        "surrogate": {},
        "grad": {},
        "row1_grad": {},
        "total_pixels": int(flat_m.size),
        "flips": int(flat_flip.sum()),
        # the gap-vs-|signed margin| divergence that row 1 inherits, MEASURED per pair
        "gap_ne_abs_margin_sites": int(
            (np.abs(flat_gap - np.abs(flat_m)) > 1e-6).sum()
        ),
        "gap_ne_abs_margin_sites_on_flips": int(
            (np.abs(flat_gap - np.abs(flat_m)) > 1e-6)[flat_flip].sum()
        ),
    }
    weights = {
        (fn, temp): row1_weight(flat_gap, fn, temp) for fn, temp in ROW1_CONFIGS
    }
    row1_keys = {round(float(t), 12) for t in row1_taus}
    for tau in taus:
        key = f"{float(tau):.9f}"
        # the surrogate keeps the TRAINER's own form (sd1's oracle test pins it); the gradient
        # uses the cancellation-free even form -- see grad_magnitude.
        probability = sd1.stable_sigmoid(-flat_m / float(tau))
        gradient = grad_magnitude(flat_m, tau)
        out["surrogate"][key] = np.bincount(idx, weights=probability, minlength=N_BINS)
        out["grad"][key] = np.bincount(idx, weights=gradient, minlength=N_BINS)
        if round(float(tau), 12) in row1_keys:
            for (fn, temp), weight in weights.items():
                config = f"{fn}@{temp:g}"
                out["row1_grad"].setdefault(config, {})[key] = np.bincount(
                    idx, weights=weight * gradient, minlength=N_BINS
                )
    return out


# ---------------------------------------------------------------------------
# milestone measurement
# ---------------------------------------------------------------------------
def measure_milestone(
    run_root: Path,
    step: int,
    pair_ids: Sequence[int],
    ground_truth: Mapping[str, Any],
    taus: Sequence[float],
    row1_taus: Sequence[float],
    thresholds: Mapping[str, Any],
    *,
    rows_handle: Any = None,
) -> dict[str, Any]:
    """Measure one milestone across ``pair_ids`` for both GT lineages, HT-weighted and raw."""

    milestone = sd1.read_milestone_json(run_root, step)
    sample_weights = sd1.sample_weight_lookup()
    m_safe = float(thresholds["m_safe"])
    per_class = thresholds["per_class_m_safe"]
    m_safe_by_class = [float(per_class[name]) for name in CLASS_NAMES] if per_class else [m_safe] * N_CLASSES

    def empty() -> dict[str, Any]:
        return {
            "pixels": np.zeros(N_BINS),
            "surrogate": {f"{float(t):.9f}": np.zeros(N_BINS) for t in taus},
            "grad": {f"{float(t):.9f}": np.zeros(N_BINS) for t in taus},
            "row1_grad": {
                f"{fn}@{temp:g}": {f"{float(t):.9f}": np.zeros(N_BINS) for t in row1_taus}
                for fn, temp in ROW1_CONFIGS
            },
        }

    totals = {lineage: {"ht": empty(), "raw": empty()} for lineage in LINEAGES}
    recorded = {row["pair_id"]: row for row in milestone["pair_rows"]}
    exact_flip_totals = dict.fromkeys(LINEAGES, 0)
    d_seg_recomputed_ht = dict.fromkeys(LINEAGES, 0.0)
    weight_sum = 0.0
    gap_divergence_sites = 0
    gap_divergence_on_flips = 0
    recorded_minus_recomputed_max = 0.0
    pair_rows: list[dict[str, Any]] = []

    for pair_id in pair_ids:
        arrays = sd1.read_pair_arrays(run_root, step, pair_id)
        logits = arrays["segnet_logits_f16"].astype(np.float32)
        retained_argmax = arrays["segnet_argmax_u8"]
        gap = top1_top2_gap(logits)
        weight = float(sample_weights[int(pair_id)])
        weight_sum += weight
        targets = {
            "vehicle_pyav": arrays["target_argmax_u8"],
            "authority_dali": np.asarray(
                ground_truth["dali_seg"][int(pair_id)], dtype=np.uint8
            ),
        }
        row: dict[str, Any] = {
            "schema": SCHEMA,
            "step": int(step),
            "pair_id": int(pair_id),
            "sample_weight": weight,
            "lineages": {},
        }
        for lineage, target in targets.items():
            margin, _competitor, _argmax = sd1.margin_and_competitor(logits, target)
            flip = retained_argmax != target
            binned = accumulate_pair(
                margin, target, flip, gap, taus, row1_taus, m_safe, m_safe_by_class
            )
            for mode, multiplier in (("ht", weight), ("raw", 1.0)):
                bucket = totals[lineage][mode]
                bucket["pixels"] += multiplier * binned["pixels"]
                for key, vector in binned["surrogate"].items():
                    bucket["surrogate"][key] += multiplier * vector
                for key, vector in binned["grad"].items():
                    bucket["grad"][key] += multiplier * vector
                for config, per_tau in binned["row1_grad"].items():
                    for key, vector in per_tau.items():
                        bucket["row1_grad"][config][key] += multiplier * vector
            exact_flips = int(flip.sum())
            exact_flip_totals[lineage] += exact_flips
            d_seg_recomputed_ht[lineage] += weight * exact_flips / float(flip.size)
            if lineage == "vehicle_pyav":
                gap_divergence_sites += binned["gap_ne_abs_margin_sites"]
                gap_divergence_on_flips += binned["gap_ne_abs_margin_sites_on_flips"]
            row["lineages"][lineage] = {
                "exact_flips": exact_flips,
                "d_seg_exact": exact_flips / float(flip.size),
                "group_pixels": decode_bins(binned["pixels"]).sum(axis=2).sum(axis=0).tolist(),
            }
        recorded_row = recorded.get(int(pair_id))
        if recorded_row is not None:
            delta = abs(
                float(recorded_row["d_seg"]) - row["lineages"]["vehicle_pyav"]["d_seg_exact"]
            )
            recorded_minus_recomputed_max = max(recorded_minus_recomputed_max, delta)
            row["recorded_d_seg"] = float(recorded_row["d_seg"])
        pair_rows.append(row)
        if rows_handle is not None:
            rows_handle.write(json.dumps(row, sort_keys=True) + "\n")
            rows_handle.flush()

    result: dict[str, Any] = {
        "step": int(step),
        "tau_eval": sd1.tau_for_milestone(step),
        "n_pairs": len(pair_rows),
        "weight_sum": weight_sum,
        "milestone_recorded": {
            "d_seg_hat": milestone["d_seg_hat"],
            "d_pose_hat": milestone["d_pose_hat"],
            "S_hat": milestone["S_hat"],
        },
        "calibration": {
            # sd1 receipt 1, re-asserted on this arm's own read: HT-weighted recompute == recorded
            "d_seg_ht_recomputed_vehicle": d_seg_recomputed_ht["vehicle_pyav"] / weight_sum,
            "d_seg_hat_recorded": float(milestone["d_seg_hat"]),
            "abs_max_pair_recorded_minus_recomputed": recorded_minus_recomputed_max,
            "gap_ne_abs_margin_sites": gap_divergence_sites,
            "gap_ne_abs_margin_sites_on_flips": gap_divergence_on_flips,
            "exact_flips_vehicle": exact_flip_totals["vehicle_pyav"],
            "exact_flips_dali": exact_flip_totals["authority_dali"],
        },
        "lineages": {},
    }
    for lineage in LINEAGES:
        result["lineages"][lineage] = {
            mode: {
                "pixels": totals[lineage][mode]["pixels"].tolist(),
                "surrogate": {
                    key: vector.tolist()
                    for key, vector in totals[lineage][mode]["surrogate"].items()
                },
                "grad": {
                    key: vector.tolist() for key, vector in totals[lineage][mode]["grad"].items()
                },
                "row1_grad": {
                    config: {key: vector.tolist() for key, vector in per_tau.items()}
                    for config, per_tau in totals[lineage][mode]["row1_grad"].items()
                },
            }
            for mode in ("ht", "raw")
        }
    return result


# ---------------------------------------------------------------------------
# analysis over the measured bins
# ---------------------------------------------------------------------------
def group_shares(grad_vector: Sequence[float]) -> dict[str, float]:
    """Global-cap group shares of the total gradient mass."""

    table = decode_bins(np.asarray(grad_vector)).sum(axis=2)  # (gt, global_group)
    total = float(table.sum())
    if total <= 0.0:
        return {name: float("nan") for name in GROUP_NAMES}
    return {
        name: float(table[:, index].sum() / total)
        for index, name in enumerate(GROUP_NAMES)
    }


def per_class_group_shares(grad_vector: Sequence[float]) -> dict[str, dict[str, float]]:
    """Per GT class: the three group shares of THAT class's own gradient mass, plus its share."""

    table = decode_bins(np.asarray(grad_vector)).sum(axis=2)
    total = float(table.sum())
    out: dict[str, dict[str, float]] = {}
    for index, name in enumerate(CLASS_NAMES):
        class_total = float(table[index].sum())
        out[name] = {
            "class_share_of_total_grad": class_total / total if total > 0 else float("nan"),
            **{
                group: (float(table[index, gi]) / class_total if class_total > 0 else float("nan"))
                for gi, group in enumerate(GROUP_NAMES)
            },
        }
    return out


def over_push_share(grad_vector: Sequence[float], class_name: str) -> dict[str, float]:
    """The mass a GLOBAL cap still pushes that the class's OWN cap would already release.

    Cell ``(global_group == correct_inside, class_group == correct_outside)``: the pixel is inside
    the global band (a global cap keeps pushing it) but outside its own class band (its own
    ``m_safe_c`` already declares it R-safe).  For a class whose ``delta_R_c`` exceeds the global
    ``delta_R`` the mirror cell (global outside, class inside) is the UNDER-protection.
    """

    table = decode_bins(np.asarray(grad_vector))
    index = CLASS_NAMES.index(class_name)
    class_total = float(table[index].sum())
    over = float(table[index, 1, 0])
    under = float(table[index, 0, 1])
    return {
        "class_grad_mass": class_total,
        "over_push_grad_mass": over,
        "over_push_share_of_class": over / class_total if class_total > 0 else float("nan"),
        "under_protect_grad_mass": under,
        "under_protect_share_of_class": under / class_total if class_total > 0 else float("nan"),
    }


def first_crossing(taus: Sequence[float], shares: Sequence[float], level: float) -> float | None:
    """Largest tau at which the (monotone-decreasing-in-tau-downward) share first falls below ``level``.

    ``taus`` ascending.  The share of gradient on correct pixels increases with tau, so scanning
    DOWNWARD in tau the share falls; the crossing is reported by linear interpolation in log-tau.
    Returns ``None`` when the level is not crossed inside the measured grid, which is a finding, not
    an error.
    """

    order = np.argsort(np.asarray(taus, dtype=np.float64))
    t = np.asarray(taus, dtype=np.float64)[order]
    s = np.asarray(shares, dtype=np.float64)[order]
    for index in range(len(t) - 1, 0, -1):
        upper, lower = s[index], s[index - 1]
        if upper >= level > lower:
            # upper > lower is implied by the guard, so the denominator cannot vanish.
            span = np.log(t[index]) - np.log(t[index - 1])
            fraction = (level - lower) / (upper - lower)
            return float(np.exp(np.log(t[index - 1]) + fraction * span))
    return None


def analyse(
    milestones: Sequence[Mapping[str, Any]],
    taus: Sequence[float],
    row1_taus: Sequence[float],
    thresholds: Mapping[str, Any],
    *,
    lineage: str = "authority_dali",
    mode: str = "ht",
) -> dict[str, Any]:
    """Assemble the charter's four readings from the measured bins."""

    tau_keys = [f"{float(t):.9f}" for t in taus]
    labelled = charter_taus(float(thresholds["delta_r"]))
    out: dict[str, Any] = {
        "lineage": lineage,
        "mode": mode,
        "charter_taus": labelled,
        "m_safe": float(thresholds["m_safe"]),
        "per_class_m_safe": thresholds["per_class_m_safe"],
        "milestones": {},
        "crossings": {},
        "row1": {},
        "over_push": {},
    }
    for record in milestones:
        step = str(record["step"])
        bins = record["lineages"][lineage][mode]
        own_tau = float(record["tau_eval"])
        own_key = f"{own_tau:.9f}"
        curve = {key: group_shares(bins["grad"][key]) for key in tau_keys}
        out["milestones"][step] = {
            "tau_eval": own_tau,
            "grad_group_share": curve,
            "grad_group_share_own_tau": group_shares(bins["grad"][own_key])
            if own_key in bins["grad"]
            else None,
            "surrogate_group_share": {
                key: group_shares(bins["surrogate"][key]) for key in tau_keys
            },
            "pixel_group_share": group_shares(bins["pixels"]),
            "per_class": {
                label: per_class_group_shares(bins["grad"][f"{value:.9f}"])
                for label, value in labelled.items()
                if f"{value:.9f}" in bins["grad"]
            },
        }
        correct = [curve[key]["correct_outside"] + curve[key]["correct_inside"] for key in tau_keys]
        wasted = [curve[key]["correct_outside"] for key in tau_keys]
        out["crossings"][step] = {
            "correct_total_below_0.50": first_crossing(taus, correct, 0.50),
            "correct_total_below_0.25": first_crossing(taus, correct, 0.25),
            "correct_outside_below_0.50": first_crossing(taus, wasted, 0.50),
            "correct_outside_below_0.25": first_crossing(taus, wasted, 0.25),
        }
        out["over_push"][step] = {
            label: {
                name: over_push_share(bins["grad"][f"{value:.9f}"], name)
                for name in CLASS_NAMES
            }
            for label, value in labelled.items()
            if f"{value:.9f}" in bins["grad"]
        }
        row1_records: dict[str, Any] = {}
        for config, per_tau in bins["row1_grad"].items():
            entry: dict[str, Any] = {}
            for label, value in labelled.items():
                key = f"{value:.9f}"
                if key in per_tau:
                    entry[label] = {
                        "grad_group_share": group_shares(per_tau[key]),
                        "per_class": per_class_group_shares(per_tau[key]),
                    }
            if own_key in per_tau:
                entry["own_tau"] = {
                    "grad_group_share": group_shares(per_tau[own_key]),
                    "per_class": per_class_group_shares(per_tau[own_key]),
                }
            row1_records[config] = entry
        out["row1"][step] = row1_records
    return out


def reanalyse(report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every ``analysis`` view from a stored report's MEASURED bins.

    The bins ARE the measurement; ``analysis`` is a pure function of them.  Re-deriving here means
    every cell's analysis is produced by one version of this code even when the cells were measured
    by separate processes, and it means the memo's tables replay from the committed payload without
    re-reading the QBR1 run directory.
    """

    thresholds = report["thresholds"]
    taus = [float(t) for t in report["taus"]]
    row1_taus = [float(t) for t in report["row1_taus"]]
    return {
        f"{lineage}:{mode}": analyse(
            report["milestones"], taus, row1_taus, thresholds, lineage=lineage, mode=mode
        )
        for lineage in LINEAGES
        for mode in ("ht", "raw")
    }


def cross_cell_agreement(
    reports: Mapping[str, Mapping[str, Any]], *, lineage: str = "authority_dali", mode: str = "ht"
) -> dict[str, Any]:
    """Max absolute disagreement between cells on each group share, per milestone and tau.

    Two cells of ONE seed are a same-seed repeat, never a seed replication -- this quantity says how
    far apart the repeat lands, so the memo can state that limit with a number instead of a hope.
    """

    views = {cell: reanalyse(report)[f"{lineage}:{mode}"] for cell, report in reports.items()}
    names = sorted(views)
    if len(names) < 2:
        return {"cells": names, "note": "fewer than two cells; no agreement computable"}
    if len(names) > 2:
        # Fail closed rather than silently comparing only the first two and reporting the result
        # as if it covered the rest -- a silently-dropped cell is a silently-wrong agreement bound.
        raise GM1Error(
            f"cross-cell agreement is defined for exactly two cells; got {len(names)}: {names}"
        )
    left, right = views[names[0]], views[names[1]]
    worst: dict[str, Any] = {"cells": names, "per_group_max_abs_delta": {}, "argmax": {}}
    for group in GROUP_NAMES:
        biggest = 0.0
        where: tuple[str, str] | None = None
        for step, record in left["milestones"].items():
            other = right["milestones"].get(step)
            if other is None:
                continue
            for key, shares in record["grad_group_share"].items():
                mirror = other["grad_group_share"].get(key)
                if mirror is None:
                    continue
                delta = abs(float(shares[group]) - float(mirror[group]))
                if delta > biggest:
                    biggest, where = delta, (step, key)
        worst["per_group_max_abs_delta"][group] = biggest
        worst["argmax"][group] = {"step": where[0], "tau": where[1]} if where else None
    return worst


def _configure_runtime(threads: int) -> dict[str, Any]:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(int(threads))
    import torch

    torch.set_num_threads(int(threads))
    return {
        "threads": int(threads),
        "torch_num_threads": torch.get_num_threads(),
        "nice": os.nice(0),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    runtime = _configure_runtime(args.threads)
    store = Path(args.store)
    cell_store = store / args.cell_id
    cell_store.mkdir(parents=True, exist_ok=True)

    thresholds = resolve_thresholds()
    delta_r = float(thresholds["delta_r"])
    own_taus = [sd1.tau_for_milestone(int(step)) for step in args.steps]
    taus = tau_grid(delta_r, extra=own_taus)
    row1_taus = sorted({*charter_taus(delta_r).values(), *[round(float(t), 12) for t in own_taus]})

    ground_truth = sd1.ar1.load_ground_truth()
    pair_ids = list(qbt.SELECTION_IDS)
    if args.limit_pairs:
        pair_ids = pair_ids[: int(args.limit_pairs)]

    milestones: list[dict[str, Any]] = []
    rows_path = cell_store / "pair_rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as rows_handle:
        for step in args.steps:
            milestones.append(
                measure_milestone(
                    Path(args.run_root),
                    int(step),
                    pair_ids,
                    ground_truth,
                    taus,
                    row1_taus,
                    thresholds,
                    rows_handle=rows_handle,
                )
            )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "instrument": INSTRUMENT,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "cell_id": args.cell_id,
        "run_root": str(Path(args.run_root).resolve()),
        "git_head": sd1.ar1.git_head(),
        "runtime": runtime,
        "thresholds": thresholds,
        "taus": taus,
        "row1_taus": row1_taus,
        "row1_configs": [f"{fn}@{temp:g}" for fn, temp in ROW1_CONFIGS],
        "class_names": list(CLASS_NAMES),
        "group_names": list(GROUP_NAMES),
        "pair_ids": pair_ids,
        "gt_lineage": ground_truth["lineage"],
        "milestones": milestones,
        "analysis": {
            f"{lineage}:{mode}": analyse(
                milestones, taus, row1_taus, thresholds, lineage=lineage, mode=mode
            )
            for lineage in LINEAGES
            for mode in ("ht", "raw")
        },
        "elapsed_seconds": None,
        "peak_rss_gib": None,
        "rows_jsonl": None,
    }
    report["elapsed_seconds"] = time.time() - started
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    report["rows_jsonl"] = sd1.ar1.file_fact(rows_path)
    fact = sd1.atomic_json(cell_store / "GM1_REPORT.json", report)
    sd1.atomic_json(cell_store / "GM1_REPORT.fact.json", fact)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--run-root", help="QBR1 cell run directory (READ-ONLY)")
    parser.add_argument("--cell-id", help="cell label for the store subdirectory")
    parser.add_argument("--store", required=True, help="payload store root")
    parser.add_argument(
        "--reanalyse", action="store_true",
        help="skip measurement; re-derive every analysis view from the stored per-cell reports "
        "under --store and write GM1_COMBINED.json beside them",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[0, 1000, 2000, 3000, 4000, 5000],
        help="milestone steps to measure",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--limit-pairs", type=int, default=0, help="pilot only: cap the pair count"
    )
    return parser


def run_reanalysis(store: Path) -> dict[str, Any]:
    """Re-derive the combined analysis from every ``<cell>/GM1_REPORT.json`` under ``store``."""

    reports: dict[str, Any] = {}
    for path in sorted(Path(store).glob("*/GM1_REPORT.json")):
        with open(path, "rb") as handle:
            reports[path.parent.name] = json.load(handle)
    if not reports:
        raise GM1Error(f"no per-cell reports under {store}")
    combined = {
        "schema": SCHEMA,
        "instrument": INSTRUMENT,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "git_head": sd1.ar1.git_head(),
        "cells": sorted(reports),
        "thresholds": next(iter(reports.values()))["thresholds"],
        "analysis": {cell: reanalyse(report) for cell, report in reports.items()},
        "cross_cell_agreement": cross_cell_agreement(reports),
        "source_reports": {
            cell: {"d_seg_hat": [m["milestone_recorded"]["d_seg_hat"] for m in report["milestones"]],
                   "calibration": [m["calibration"] for m in report["milestones"]]}
            for cell, report in reports.items()
        },
    }
    fact = sd1.atomic_json(Path(store) / "GM1_COMBINED.json", combined)
    sd1.atomic_json(Path(store) / "GM1_COMBINED.fact.json", fact)
    return combined


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reanalyse:
        combined = run_reanalysis(Path(args.store))
        print(f"[{INSTRUMENT}] reanalysed cells: {combined['cells']}")
        print(f"  cross-cell max |delta| share: {combined['cross_cell_agreement']['per_group_max_abs_delta']}")
        return 0
    if not args.run_root or not args.cell_id:
        raise GM1Error("--run-root and --cell-id are required unless --reanalyse is given")
    report = run(args)
    analysis = report["analysis"]["authority_dali:ht"]
    print(f"[{INSTRUMENT}] cell={args.cell_id} elapsed={report['elapsed_seconds']:.1f}s")
    print(f"  m_safe={report['thresholds']['m_safe']!r} (n={report['thresholds']['n_frames']})")
    for step, record in analysis["milestones"].items():
        own = record["grad_group_share_own_tau"]
        if own is None:
            continue
        print(
            f"  step {step:>5} tau={record['tau_eval']:.5f}  "
            f"outside={own['correct_outside']:.4f} inside={own['correct_inside']:.4f} "
            f"wrong={own['wrong']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
