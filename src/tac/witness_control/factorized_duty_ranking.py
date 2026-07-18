# SPDX-License-Identifier: MIT
"""Exact-factorized duty-to-measure ranking (organ upgrade A, 2026-07-17).

Ranks candidate levers by a CLOSED-FORM first-order d_seg marginal instead of the
statistical "% of remaining descent" heuristic the digest already carries.  The two
rankings are surfaced SIDE BY SIDE (this module never replaces the statistical order —
comparing them is the point; see ``tools/costate_digest.section_factorized_sense``).

The marginal (DERIVED, first-order, from canonical measured laws — no new constants):

  * ``segnet_head_rank4_linear_flipdist_v1`` (MEASURED): the frozen head is EXACTLY
    rank-4 linear, so a flip pixel with pairwise logit margin ``m = z_wrong - z_gt`` is
    corrected exactly when the induced margin change crosses ``m``; the maximal margin
    change per unit feature-space (penultimate-patch) move on pair (c,c') is
    ``||w_c - w_c'||`` (the canonical pair norms).
  * a lever's class-direction ``u`` is its DSL trunk coordinate (``lambda_net.
    lever_features`` — the SAME feature vector the #516 exact factorized adjoint consumes).
    Its aimable share of the (wrong->gt) pair axis is ``align = max(u_gt - u_wrong, 0) /
    (sqrt(2) * ||u||)`` in [0, 1] (equality when u = e_gt - e_wrong, i.e. perfectly aimed).
  * ker(A) (``realization_necessity_preimage_per_stratum_v1`` + the EXACT closed-form tap
    table in ``factorized_features``): a lever whose actuation is expressed as a
    CAMERA-space map only acts through ``range(A)``; the surviving amplitude scale is the
    root visible-energy fraction, and a map supported ENTIRELY inside ker(A) has EXACTLY
    zero scorer-input effect — hence a PROVABLY zero marginal (theorem, since A's taps
    never read those pixels; verified live in tests via torch one-hot probes).

  marginal_d_seg(lever, eps) =
      (1/N_px) * sum over remaining flip pixels p of
          1[ m(p) <= eps * kappa(lever) * ||w_pair(p)|| * align(lever, p) ]

with ``eps`` a feature-space actuation budget (default: SELF-CALIBRATED to the snapshot's
median feature-space flip distance — a measured scale, not a hardcoded constant) and
``kappa`` the ker(A) survival scale (1 for logit/class-space levers; the root visible
energy fraction for camera-map levers; 0 for pure-ker maps).

HONESTY / limits (stated, per the operating manual §5): the crossing test is pairwise
first-order — after a move a THIRD class can intercept the argmax, so the marginal is an
upper-bound-flavored first-order estimate on the sampled surface; the snapshot is a
labeled stride subset.  All outputs advisory ``[macOS-CPU advisory] NON-PROMOTABLE``,
``score_claim=False``.  This module is read-only and dynamic: rankings recompute from the
latest margin snapshot as the live margin field evolves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tac.witness_control.factorized_adjoint import (
    exact_response_direction,
)
from tac.witness_control.factorized_features import (
    AXIS_TAG,
    MARGIN_HIST_EDGES,
    MarginSnapshot,
    ker_a_zero_weight_mask,
    pair_norm_for_oriented,
    parse_oriented_key,
    visible_energy_split,
)
from tac.witness_control.lambda_net import LEVER_FEATURE_MAP, N_CLASSES, lever_features


def lever_class_direction(lever: str) -> np.ndarray:
    """The lever's class-space direction u (its DSL trunk class-target coordinates)."""
    return np.asarray(lever_features(lever)[:N_CLASSES], dtype=np.float64)


def alignment(u: np.ndarray, wrong: int, gt: int) -> float:
    """Aimable share of the (wrong->gt) pair axis in [0,1].

    ``align = max(u_gt - u_wrong, 0) / (sqrt(2) * ||u||)``; 1.0 iff u ∝ e_gt - e_wrong.
    A lever with no class direction (||u|| == 0, e.g. pure rate/pose levers) cannot aim
    any pair: 0.0 (honest zero, not an error)."""
    nu = float(np.linalg.norm(u))
    if nu <= 0.0:
        return 0.0
    val = (float(u[gt]) - float(u[wrong])) / (math.sqrt(2.0) * nu)
    return max(0.0, min(1.0, val))


def camera_map_survival_scale(camera_map: np.ndarray) -> float:
    """ker(A)-projection survival scale for a CAMERA-space actuation map: the root of the
    range(A)-visible energy fraction.  EXACTLY 0.0 for a map supported inside ker(A)
    (the zero-marginal theorem — A's taps never read those pixels)."""
    split = visible_energy_split(camera_map, ker_a_zero_weight_mask())
    vf = split["visible_frac"]
    if vf is None:  # zero-energy map: no actuation at all
        return 0.0
    return math.sqrt(max(0.0, min(1.0, float(vf))))


def _crossed_count_from_hist(hist: list[int], underflow: int, edges: np.ndarray, thr: float) -> float:
    """Crossed flip count for threshold ``thr`` from a persisted margin histogram
    (linear-within-bin interpolation; labeled histogram-resolution approximation)."""
    if thr <= float(edges[0]):
        return float(underflow) * (1.0 if thr > 0 else 0.0)
    total = float(underflow)
    for b, cnt in enumerate(hist):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if thr >= hi:
            total += float(cnt)
        elif thr > lo:
            total += float(cnt) * (thr - lo) / (hi - lo)
            break
        else:
            break
    return total


@dataclass(frozen=True)
class LeverMarginal:
    lever: str
    marginal_d_seg: float          # first-order d_seg crossed at the budget (sample surface)
    crossed_flips: float
    eps_feat: float                # the feature-space budget used (self-calibrated default)
    kappa: float                   # ker(A) survival scale applied (1.0 = logit-space lever)
    align_mass_weighted: float     # flip-mass-weighted mean alignment (0 => cannot aim)
    adjoint_response_l1: float     # |exact_response_direction(phi)[:5]|_1 (the #516 magnitude)
    axis_tag: str = AXIS_TAG
    score_claim: bool = False

    def to_dict(self) -> dict:
        return {
            "lever": self.lever,
            "marginal_d_seg": self.marginal_d_seg,
            "crossed_flips": self.crossed_flips,
            "eps_feat": self.eps_feat,
            "kappa": self.kappa,
            "align_mass_weighted": self.align_mass_weighted,
            "adjoint_response_l1": self.adjoint_response_l1,
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
        }


def self_calibrated_eps(snapshot: MarginSnapshot) -> float:
    """Default actuation budget: the snapshot's OWN median feature-space flip distance
    (m/||w_pair||) over remaining flips — a measured scale from the live margin field
    (value-provenance: DERIVED from the snapshot; no hardcoded constant).  Falls back to
    0 flips -> 0.0 (nothing to cross)."""
    fd = snapshot.flipdist_feature_space_by_oriented_pair()
    if not fd:
        return 0.0
    allv = np.concatenate(list(fd.values()))
    return float(np.median(allv))


def lever_marginal_from_snapshot(
    lever: str,
    snapshot: MarginSnapshot,
    *,
    eps_feat: float | None = None,
    camera_map: np.ndarray | None = None,
) -> LeverMarginal:
    """Closed-form first-order marginal for one lever on the LIVE margin snapshot.

    Computes, per remaining flip pixel, whether the lever's aimable margin push at budget
    ``eps_feat`` crosses the EXACT measured margin.  ``camera_map`` (optional): the lever's
    camera-space actuation support — projected through the EXACT ker(A) mask; pure-ker
    maps yield a provably zero marginal."""
    u = lever_class_direction(lever)
    eps = self_calibrated_eps(snapshot) if eps_feat is None else float(eps_feat)
    kappa = 1.0 if camera_map is None else camera_map_survival_scale(camera_map)
    phi = lever_features(lever)
    resp = exact_response_direction(phi)
    resp_l1 = float(np.abs(resp[:N_CLASSES]).sum())

    crossed = 0.0
    align_wsum = 0.0
    mass = 0
    for key, margins in snapshot.margins_by_oriented_pair().items():
        w, g = parse_oriented_key(key)
        al = alignment(u, w, g)
        mass += margins.size
        align_wsum += al * margins.size
        if al <= 0.0 or kappa <= 0.0 or eps <= 0.0:
            continue
        thr = eps * kappa * pair_norm_for_oriented(key) * al
        crossed += float(np.count_nonzero(margins <= thr))
    return LeverMarginal(
        lever=lever,
        marginal_d_seg=crossed / float(snapshot.total_px),
        crossed_flips=crossed,
        eps_feat=eps,
        kappa=kappa,
        align_mass_weighted=(align_wsum / mass) if mass else 0.0,
        adjoint_response_l1=resp_l1,
    )


def rank_levers_from_snapshot(
    snapshot: MarginSnapshot,
    lever_names: list[str] | None = None,
    *,
    eps_feat: float | None = None,
) -> list[LeverMarginal]:
    """Rank the candidate lever set by closed-form marginal (descending).  Default lever
    set = the DSL trunk coordinates the adjoint itself uses (LEVER_FEATURE_MAP keys)."""
    names = sorted(LEVER_FEATURE_MAP) if lever_names is None else list(lever_names)
    eps = self_calibrated_eps(snapshot) if eps_feat is None else float(eps_feat)
    rows = [lever_marginal_from_snapshot(nm, snapshot, eps_feat=eps) for nm in names]
    rows.sort(key=lambda r: (-r.marginal_d_seg, r.lever))
    return rows


# ---------------------------------------------------------------------------
# Persisted-row path (the digest recomputes rankings from the snapshot JSONL row —
# cheap, no decode/SegNet at SessionStart)
# ---------------------------------------------------------------------------
def rank_levers_from_summary_row(
    row: dict,
    lever_names: list[str] | None = None,
    *,
    eps_feat: float | None = None,
) -> list[dict]:
    """Same ranking recomputed from a persisted ``MarginSnapshot.summary_row()`` dict.

    Uses the stored per-oriented-pair margin histograms (histogram-resolution
    approximation of the exact crossing count; labeled).  Default eps: the stored median
    feature-space flip distance across pairs (mass-weighted median of stored medians)."""
    by_pair: dict = row.get("by_oriented_pair") or {}
    edges = np.asarray(row.get("margin_hist_edges") or MARGIN_HIST_EDGES, dtype=np.float64)
    total_px = float(row.get("total_px") or 0)
    if total_px <= 0 or not by_pair:
        return []
    if eps_feat is None:
        meds, wts = [], []
        for _k, d in by_pair.items():
            q = (d.get("flipdist_feat_q") or {}).get(0.5) or (d.get("flipdist_feat_q") or {}).get("0.5")
            if isinstance(q, (int, float)):
                meds.append(float(q))
                wts.append(float(d.get("n", 0)))
        if not meds:
            return []
        order = np.argsort(meds)
        cum = np.cumsum(np.asarray(wts)[order])
        eps = float(np.asarray(meds)[order][int(np.searchsorted(cum, cum[-1] / 2.0))])
    else:
        eps = float(eps_feat)

    names = sorted(LEVER_FEATURE_MAP) if lever_names is None else list(lever_names)
    out: list[dict] = []
    for nm in names:
        u = lever_class_direction(nm)
        crossed = 0.0
        for key, d in by_pair.items():
            w, g = parse_oriented_key(key)
            al = alignment(u, w, g)
            if al <= 0.0 or eps <= 0.0:
                continue
            thr = eps * pair_norm_for_oriented(key) * al
            crossed += _crossed_count_from_hist(
                list(d.get("margin_hist") or []), int(d.get("margin_underflow") or 0), edges, thr
            )
        out.append({
            "lever": nm,
            "marginal_d_seg": crossed / total_px,
            "crossed_flips": crossed,
            "eps_feat": eps,
            "axis_tag": AXIS_TAG,
            "score_claim": False,
        })
    out.sort(key=lambda r: (-r["marginal_d_seg"], r["lever"]))
    return out


def format_factorized_duty_line(rows: list[dict], *, ema_epoch=None, age_s: float | None = None,
                                top_n: int = 5) -> str:
    """Digest line for the exact-factorized ranking (ALTERNATIVE alongside the statistical
    duty line; never a replacement).  Pure formatter — unit-testable."""
    if not rows:
        return "factorized-duty: no snapshot rows yet (run tools/costate_live_ingest.py)"
    cells = ", ".join(
        f"{r['lever']} {r['marginal_d_seg']:.2e}" for r in rows[:top_n] if r.get("marginal_d_seg", 0) > 0
    ) or "(no lever crosses flip mass at this budget)"
    head = "factorized-duty (exact rank-4/ker(A) first-order marginal, ALTERNATIVE ranking"
    if ema_epoch is not None:
        head += f", ema ep{ema_epoch}"
    if age_s is not None:
        head += f", {age_s / 60:.0f}m old"
    return f"{head}): {cells} [advisory NON-PROMOTABLE]"


__all__ = [
    "LeverMarginal",
    "alignment",
    "camera_map_survival_scale",
    "format_factorized_duty_line",
    "lever_class_direction",
    "lever_marginal_from_snapshot",
    "rank_levers_from_snapshot",
    "rank_levers_from_summary_row",
    "self_calibrated_eps",
]
