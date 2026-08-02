#!/usr/bin/env python
"""ddm_ti1 -- TEACHER NON-REDUNDANCY PROBE: does a candidate teacher carry information
the loss cannot already express, and does that information predict where the student
is actually wrong?

WHY THIS EXISTS (the derivation, ddm_ti1 2026-08-02)
----------------------------------------------------
``tools/ddm_lr1_teacher_information_probe.py`` (ddm_lr1) killed the ms2r/QA75 solve teacher
by showing its SegNet margin field is a 16.6%-of-sigma noisy copy of the free GT margin
field the trainer already memmaps. Its criterion -- *a teacher pays only to the extent it
is NOT redundant with what the loss already sees* -- is right, but its implementation tests
one specific comparison (teacher margin vs GT margin, raw correlation) and only accepts
(P,C,H,W) SegNet logit caches.

This probe GENERALIZES that criterion along two axes, and subsumes it:

**Axis 1 -- project onto the loss's whole information sigma-algebra, not one field.**
The live TR1 per-pixel seg loss is *per-pair separable*: for pair ``t`` it reads only
``lstars[t]`` (the hard target), ``margins[t]`` (via the boundary-annulus / lane-guard
margin-floor weighting), and the student's own live logits. Therefore EVERY per-pixel
reweighting the current loss can express from its own GT inputs is a measurable function
of the pair ``(class, GT margin)``. So the right null model is not "correlated with GT
margin" -- it is "measurable w.r.t. sigma(class, GT margin)". This probe projects the
candidate onto that sigma-algebra NONPARAMETRICALLY (conditional means over a
class x margin-bin grid, i.e. an exact one-way ANOVA decomposition) and reports the
residual variance fraction. A candidate whose residual fraction is ~0 cannot be
distinguished from a reweighting the loss could already apply, no matter how it is
plumbed in.

**Axis 2 -- non-redundancy is necessary but NOT sufficient; test predictiveness.**
A residual can be non-zero and still be worthless if it is noise (which is exactly what
the lr1 teacher's 16.6% residual was: the solve's own realization error). So the second
leg joins the candidate against the student's REALIZED FLIP SET and asks whether the
candidate raises flip risk *within* each (class, margin) stratum -- a Mantel-Haenszel
pooled risk ratio. Crude lift is not evidence: a field that merely tracks low margin will
show a large crude lift and a stratified lift of ~1, because the loss is already weighting
by margin. The STRATIFIED lift is the number that matters.

**The structural consequence (why this is not just a bigger correlation):** because the
loss is per-pair separable, any candidate that is a function of a single pair's own
(class, margin) is at structural risk of redundancy, while a CROSS-PAIR candidate
(temporal structure computed from lstars[t-1], lstars[t+1]) is outside the loss's
information set BY CONSTRUCTION. This probe measures how much that structural
non-redundancy is actually worth.

CONTROLS (an "everything is redundant" readout is exactly what a dead instrument prints)
---------------------------------------------------------------------------------------
* ``margin_bin_CALIBRATION_exact``: the stratum's own margin-bin index. Measurable w.r.t.
  the conditioning grid BY CONSTRUCTION => residual variance fraction must be EXACTLY 0.
  If it is not, the projection is broken and no readout is admissible.
* ``margin_CALIBRATION_redundant``: the continuous GT margin, split MID-BIN (the median is
  itself a bin edge, which would leave no stratum populated on both sides). The loss already
  weights by this field, so its crude lift must be large while its STRATIFIED lift collapses
  to ~1. This calibration is what demonstrates that the crude-vs-stratified distinction is
  real and not an artifact of the estimator.
* ``flip_CALIBRATION_oracle``: the realized flip indicator itself => stratified lift must be
  enormous. If it is not, the lift machinery cannot see signal and no negative is
  admissible.
* ``hashnoise_CALIBRATION_null``: a deterministic hash field, non-redundant by construction
  but carrying no information about student error, thresholded to the same sparsity as
  ``spike`` => stratified lift must be ~1. This is the calibration that guards a POSITIVE:
  it proves the pooled estimator does not manufacture lift out of sparse strata.
* shuffled-pair control on the lift leg: candidate field from pair ``(t+stride)`` against
  flips from pair ``t``. Preserves every marginal rate and destroys pair-specific
  alignment => lift must collapse toward 1.
* VACUOUS scope (zero pairs, zero flips, globally degenerate split, constant field) RAISES.
  An empty scope is never reported as a pass. NOTE the deliberate exception: zero USABLE
  strata is NOT vacuity -- it is the signature of a field the strata already determine, and
  it is reported as such.

KNOWN RESOLUTION LIMIT (stated because it sets the direction a wrong answer can err in)
---------------------------------------------------------------------------------------
The margin axis is DISCRETISED into quantile bins, so the projection is onto a COARSENING of
sigma(class, margin). A coarser projection explains LESS variance, so the reported residual
fraction is an UPPER bound on the true residual and the reported stratified lift is an UPPER
bound on the true conditional lift. Consequence: a REDUNDANT verdict from this probe is
conservative (the true redundancy is at least as strong), while a NON-REDUNDANT verdict must
be checked for bin-count sensitivity before it is believed. Always run at two bin counts.

Authority: ``[macOS-CPU advisory]``, ``score_claim=False``, ``research_only=True``.
ZERO scorer forwards -- consumes the cached frozen-authority GT argmax/margin and a
precomputed realized-flip atlas. This moves no pointer; it is MEANS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

#: Shuffle offset for the pair-shuffled lift control. Coprime with typical interior-pair
#: counts so the control pairing is a derangement (no pair joined against itself).
CONTROL_STRIDE: int = 137

#: LIVENESS FLOOR (a guard, not a measurement, and deliberately NOT on the value-provenance
#: ladder for scientific values): the shuffled-pair control's stratified lift must fall at
#: least this far toward 1 relative to the real lift, or we refuse to trust our own readout.
#: Its only job is to separate "this field really tracks THIS pair's flips" from "the
#: comparison is degenerate and would report a lift against anything".
CONTROL_LIVENESS_MIN_LIFT_DROP_FRAC: float = 0.25

#: Minimum pixels a (class, margin-bin) stratum needs on BOTH sides of the candidate split
#: before it may contribute to the pooled estimate. Guards against strata whose risk ratio
#: is an artifact of a handful of pixels. Reported alongside the usable-stratum count so the
#: denominator is always visible.
MIN_STRATUM_SIDE_PIXELS: int = 1000


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# Candidate teacher fields. Each takes (lstars, margins, flip_mask, t) and returns a
# (H,W) float array for the redundancy leg plus a bool array for the lift leg.
# ---------------------------------------------------------------------------------------


def field_spike(lstars: Any, t: int) -> np.ndarray:
    """GT temporal SPIKE (ddm_fl1 definition): scored-frame argmax differs from BOTH
    stride-2 neighbours. The pixel set a temporally-smooth witness provably cannot score.
    CROSS-PAIR: reads lstars[t-1] and lstars[t+1], neither of which the per-pair loss sees.
    """
    prev = np.asarray(lstars[t - 1])
    cur = np.asarray(lstars[t])
    nxt = np.asarray(lstars[t + 1])
    return ((cur != prev) & (cur != nxt))


def field_change(lstars: Any, t: int) -> np.ndarray:
    """GT temporal CHANGE (ddm_ru1 definition): argmax differs from the NEXT scored frame.
    The broader temporal-boundary set. CROSS-PAIR.
    """
    cur = np.asarray(lstars[t])
    nxt = np.asarray(lstars[t + 1])
    return cur != nxt


def field_coherent(lstars: Any, t: int) -> np.ndarray:
    """GT temporally-unstable but NOT a spike: differs from exactly ONE stride-2 neighbour.

    This is the complement of ``spike`` inside the unstable set, and it is the OTHER knob of
    the already-built #274 lever (``coh = (dp | dn) & ~sp`` at
    ``experiments/train_levelset_witness_realized_through_R_mlx.py:9487``, up-weighted by
    ``--seg-coherent-upweight``). Naming it separately lets the two scalars be priced in
    their own coordinates instead of through the union. CROSS-PAIR.
    """
    prev = np.asarray(lstars[t - 1])
    cur = np.asarray(lstars[t])
    nxt = np.asarray(lstars[t + 1])
    dp, dn = (cur != prev), (cur != nxt)
    return (dp | dn) & ~(dp & dn)


def field_flickcount(lstars: Any, t: int) -> np.ndarray:
    """Graded temporal instability: how many of the two stride-2 neighbours disagree (0/1/2).
    CROSS-PAIR. Returned as float for the redundancy leg; binarised at >=1 for the lift leg.
    """
    prev = np.asarray(lstars[t - 1])
    cur = np.asarray(lstars[t])
    nxt = np.asarray(lstars[t + 1])
    return (cur != prev).astype(np.float32) + (cur != nxt).astype(np.float32)


#: Prevalence the pure-noise NULL field is thresholded at, so that its rarity matches the
#: ``spike`` candidate's and the two are compared at equal sparsity. This is a control's
#: nuisance parameter, NOT a scientific value: it is the measured GT spike prevalence
#: (625,297 px / 598 interior pairs, ddm_fl1 + reproduced independently by this probe).
#: Nothing is derived from it; changing it changes only how sparse the null is.
NULL_FIELD_PREVALENCE: float = 0.005318


def field_hashnoise(lstars: Any, t: int) -> np.ndarray:
    """PURE-NOISE NULL: a deterministic per-(pair,y,x) hash, independent of everything.

    Non-redundant with (class, margin) BY CONSTRUCTION but carrying ZERO information about
    student error. The estimator must therefore return a stratified lift of ~1 on it. This
    is the calibration that proves a large stratified lift is not something the Mantel-
    Haenszel pooling manufactures out of sparse strata. Deterministic (no RNG state), so the
    whole probe stays reproducible.
    """
    h, w = int(lstars.shape[1]), int(lstars.shape[2])
    idx = np.arange(h * w, dtype=np.uint64) + np.uint64(t) * np.uint64(h * w)
    # splitmix64 finaliser: cheap, well-distributed, and fully deterministic.
    z = idx + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return (u < NULL_FIELD_PREVALENCE).reshape(h, w)


#: name -> (builder, is_binary, binarise_threshold_or_None)
CANDIDATE_FIELDS: dict[str, tuple[Callable[[Any, int], np.ndarray], bool, float | None]] = {
    "spike": (field_spike, True, None),
    "change": (field_change, True, None),
    "coherent": (field_coherent, True, None),
    "flickcount": (field_flickcount, False, 1.0),
    "hashnoise_CALIBRATION_null": (field_hashnoise, True, None),
}


class _Strata:
    """Streaming (class x margin-bin) accumulator for BOTH probe legs.

    Redundancy leg: per-cell sum/sumsq/count of the candidate value -> exact ANOVA split of
    Var(T) into between-cell (expressible by the loss) and within-cell (residual) parts.

    Lift leg: per-cell 2x2 table of (candidate high/low) x (flip/no-flip).
    """

    def __init__(self, n_classes: int, n_bins: int) -> None:
        shape = (n_classes, n_bins)
        self.count = np.zeros(shape, dtype=np.float64)
        self.tsum = np.zeros(shape, dtype=np.float64)
        self.tsq = np.zeros(shape, dtype=np.float64)
        # [class, bin, hi(0/1)] pixel counts and flip counts
        self.n_side = np.zeros((*shape, 2), dtype=np.float64)
        self.flip_side = np.zeros((*shape, 2), dtype=np.float64)

    def add(
        self,
        cls: np.ndarray,
        mbin: np.ndarray,
        tval: np.ndarray,
        thi: np.ndarray,
        flip: np.ndarray,
    ) -> None:
        cls = cls.ravel()
        mbin = mbin.ravel()
        tval = tval.ravel().astype(np.float64)
        thi = thi.ravel().astype(np.int64)
        flip = flip.ravel()
        n_c, n_b = self.count.shape
        cell = cls * n_b + mbin
        flat_n = np.bincount(cell, minlength=n_c * n_b)
        flat_s = np.bincount(cell, weights=tval, minlength=n_c * n_b)
        flat_q = np.bincount(cell, weights=tval * tval, minlength=n_c * n_b)
        self.count += flat_n.reshape(n_c, n_b)
        self.tsum += flat_s.reshape(n_c, n_b)
        self.tsq += flat_q.reshape(n_c, n_b)
        side_cell = cell * 2 + thi
        flat_side = np.bincount(side_cell, minlength=n_c * n_b * 2)
        flat_side_flip = np.bincount(
            side_cell, weights=flip.astype(np.float64), minlength=n_c * n_b * 2
        )
        self.n_side += flat_side.reshape(n_c, n_b, 2)
        self.flip_side += flat_side_flip.reshape(n_c, n_b, 2)

    # -- leg 1 -------------------------------------------------------------------------
    def residual_variance_fraction(self) -> dict[str, float]:
        """Fraction of Var(candidate) NOT expressible as a function of (class, margin-bin).

        Exact one-way ANOVA: Var(T) = E[Var(T|cell)] + Var(E[T|cell]).
        Returned value is E[Var(T|cell)] / Var(T). ~0 => the loss could already express it.
        """
        n_tot = float(self.count.sum())
        if n_tot <= 1.0:
            raise ValueError("VACUOUS: fewer than 2 pixels accumulated -- this is not a pass")
        grand_sum = float(self.tsum.sum())
        grand_sq = float(self.tsq.sum())
        total_ss = grand_sq - grand_sum * grand_sum / n_tot
        nz = self.count > 0
        between_ss = float((self.tsum[nz] ** 2 / self.count[nz]).sum()) - grand_sum**2 / n_tot
        within_ss = total_ss - between_ss
        if total_ss <= 0.0:
            raise ValueError(
                "VACUOUS: candidate field is globally constant (zero variance) -- "
                "there is nothing to project and this is not a pass"
            )
        return {
            "total_variance": total_ss / n_tot,
            "between_cell_variance": between_ss / n_tot,
            "within_cell_residual_variance": within_ss / n_tot,
            "residual_variance_fraction": within_ss / total_ss,
            "n_pixels": n_tot,
            "n_nonempty_cells": int(nz.sum()),
        }

    # -- leg 2 -------------------------------------------------------------------------
    def lift(self, min_side: int = MIN_STRATUM_SIDE_PIXELS) -> dict[str, float | int]:
        """Crude and Mantel-Haenszel-pooled stratified flip risk ratio, hi vs lo candidate.

        Crude RR answers "do flips concentrate where the candidate is high?" -- which a field
        that merely tracks low margin passes trivially. The MH RR conditions on the
        (class, margin-bin) stratum, i.e. on everything the loss can already weight by, and
        is the number that decides whether the candidate adds anything.
        """
        n_lo = self.n_side[..., 0]
        n_hi = self.n_side[..., 1]
        f_lo = self.flip_side[..., 0]
        f_hi = self.flip_side[..., 1]
        tot_lo, tot_hi = float(n_lo.sum()), float(n_hi.sum())
        tot_flo, tot_fhi = float(f_lo.sum()), float(f_hi.sum())
        if tot_hi <= 0.0 or tot_lo <= 0.0:
            raise ValueError(
                "VACUOUS: candidate split is degenerate (one side has zero pixels) -- not a pass"
            )
        if tot_flo + tot_fhi <= 0.0:
            raise ValueError("VACUOUS: zero flips in scope -- this is not a pass")
        crude = (tot_fhi / tot_hi) / (tot_flo / tot_lo) if tot_flo > 0 else float("inf")

        usable = (n_hi >= min_side) & (n_lo >= min_side)
        n_usable = int(usable.sum())
        if n_usable == 0:
            # NOT vacuity: a field with no stratum containing both sides is one the strata
            # already DETERMINE. That is the strongest possible redundancy readout, and
            # collapsing it into an exception would hide the answer.
            return {
                "crude_risk_ratio": crude,
                "stratified_mh_risk_ratio": None,
                "strata_verdict": "NO_USABLE_STRATA_FIELD_IS_STRATUM_DETERMINED",
                "n_usable_strata": 0,
                "n_strata_total": int(n_hi.size),
                "min_stratum_side_pixels": min_side,
                "pixels_hi": tot_hi,
                "pixels_lo": tot_lo,
                "flips_hi": tot_fhi,
                "flips_lo": tot_flo,
                "flip_rate_hi": tot_fhi / tot_hi,
                "flip_rate_lo": tot_flo / tot_lo,
            }
        nk = n_hi[usable] + n_lo[usable]
        num = float((f_hi[usable] * n_lo[usable] / nk).sum())
        den = float((f_lo[usable] * n_hi[usable] / nk).sum())
        mh = num / den if den > 0 else float("inf")
        return {
            "crude_risk_ratio": crude,
            "stratified_mh_risk_ratio": mh,
            "strata_verdict": "OK",
            "n_usable_strata": n_usable,
            "n_strata_total": int(n_hi.size),
            "min_stratum_side_pixels": min_side,
            "pixels_hi": tot_hi,
            "pixels_lo": tot_lo,
            "flips_hi": tot_fhi,
            "flips_lo": tot_flo,
            "flip_rate_hi": tot_fhi / tot_hi,
            "flip_rate_lo": tot_flo / tot_lo,
            "mh_numerator": num,
            "mh_denominator": den,
        }


def margin_bin_edges(
    margins: Any, n_bins: int, pair_ids: np.ndarray, pixel_stride: int
) -> tuple[np.ndarray, float]:
    """Global quantile edges for the margin conditioning axis, plus the global median.

    Deterministic strided subsample (no RNG). Returns ``(edges, median)``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    sample = []
    for t in pair_ids[:: max(1, len(pair_ids) // 40)]:
        sample.append(np.asarray(margins[int(t)], dtype=np.float32).ravel()[::pixel_stride])
    flat = np.concatenate(sample)
    if flat.size == 0:
        raise ValueError("VACUOUS: margin subsample is empty -- this is not a pass")
    qs = np.linspace(0.0, 100.0, n_bins + 1)[1:-1]
    # The redundant-margin calibration is split MID-BIN, not at the median: the median IS a
    # bin edge, so a median split leaves no stratum with pixels on both sides and the
    # calibration would return UNDEF instead of the ~1 stratified lift it exists to show.
    split_q = 50.0 + 50.0 / n_bins
    return np.percentile(flat, qs).astype(np.float64), float(np.percentile(flat, split_q))


def _flip_masks_by_pair(
    atlas: Path, shape_hw: tuple[int, int]
) -> tuple[dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]]]:
    """Per-pair realized-flip masks, plus per-pair (y, x, realized_class) for error typing.

    ``realized_class`` is None when the atlas does not carry it; the typing block is then
    skipped and says so rather than silently reporting nothing.
    """
    with np.load(atlas) as data:
        for key in ("pair", "y", "x"):
            if key not in data.files:
                raise ValueError(f"flip atlas {atlas} missing '{key}' (has {sorted(data.files)})")
        pair = np.asarray(data["pair"], dtype=np.int64)
        yy = np.asarray(data["y"], dtype=np.int64)
        xx = np.asarray(data["x"], dtype=np.int64)
        rc = (
            np.asarray(data["realized_class"], dtype=np.int64)
            if "realized_class" in data.files
            else None
        )
    if pair.size == 0:
        raise ValueError(f"VACUOUS: flip atlas {atlas} has zero flips -- this is not a pass")
    h, w = shape_hw
    masks: dict[int, np.ndarray] = {}
    coords: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    order = np.argsort(pair, kind="stable")
    pair, yy, xx = pair[order], yy[order], xx[order]
    if rc is not None:
        rc = rc[order]
    bounds = np.searchsorted(pair, np.arange(pair[0], pair[-1] + 2))
    for i, p in enumerate(range(int(pair[0]), int(pair[-1]) + 1)):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        m = np.zeros((h, w), dtype=bool)
        m[yy[lo:hi], xx[lo:hi]] = True
        masks[p] = m
        coords[p] = (yy[lo:hi], xx[lo:hi], None if rc is None else rc[lo:hi])
    return masks, coords


def type_flicker_errors(
    lstars: Any,
    coords: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    pair_ids: np.ndarray,
    control_stride: int,
) -> dict[str, Any]:
    """DIRECTION leg: on temporally-unstable pixels, is the student emitting the label the
    TEMPORAL NEIGHBOURS carry?

    This is what separates the two opposite actuations of a temporal-instability teacher.
    If the student's wrong label is the neighbour-frame label, the student is behaving like a
    temporally-SMOOTH witness and GT is spiking away from it -- a phase-faithfulness debt, and
    the mass is conceded only if it is genuinely unreachable. If the wrong label is unrelated
    to the neighbours, the flicker pixels are simply hard, and the teacher's natural direction
    is to attack rather than concede.

    A raw match rate proves nothing on its own (a wrong label could hit a neighbour's label by
    chance, and the class priors are extremely skewed), so every rate is paired with the same
    shuffled-pair control used elsewhere: the SAME realized labels scored against a DIFFERENT
    pair's temporal neighbours.
    """
    # ``realized_class`` is a property of the atlas as a whole, so decide availability ONCE
    # up front rather than bailing out mid-accumulation (a mid-loop return would silently
    # discard partial counts).
    if any(coords[int(t)][2] is None for t in pair_ids):
        return {
            "available": False,
            "reason": "flip atlas carries no 'realized_class' -- typing SKIPPED, not passed",
        }
    tot = matched = matched_ctrl = 0
    per_class_tot = np.zeros(5, dtype=np.int64)
    per_class_matched = np.zeros(5, dtype=np.int64)
    for i, t in enumerate(pair_ids):
        t = int(t)
        yy, xx, rc = coords[t]
        prev = np.asarray(lstars[t - 1])[yy, xx]
        nxt = np.asarray(lstars[t + 1])[yy, xx]
        cur = np.asarray(lstars[t])[yy, xx]
        unstable = (cur != prev) | (cur != nxt)
        if not unstable.any():
            continue
        rcu = rc[unstable]
        hit = (rcu == prev[unstable]) | (rcu == nxt[unstable])
        tot += int(unstable.sum())
        matched += int(hit.sum())
        for c in range(5):
            sel = cur[unstable] == c
            per_class_tot[c] += int(sel.sum())
            per_class_matched[c] += int((hit & sel).sum())
        tc = int(pair_ids[(i + control_stride) % len(pair_ids)])
        p_c = np.asarray(lstars[tc - 1])[yy, xx][unstable]
        n_c = np.asarray(lstars[tc + 1])[yy, xx][unstable]
        matched_ctrl += int(((rcu == p_c) | (rcu == n_c)).sum())
    if tot == 0:
        raise ValueError("VACUOUS: zero flips on temporally-unstable pixels -- not a pass")
    return {
        "available": True,
        "flips_on_unstable_pixels": tot,
        "realized_label_equals_a_temporal_neighbour_label": matched,
        "match_rate": matched / tot,
        "match_rate_pair_shuffled_control": matched_ctrl / tot,
        "control_fired": bool(matched_ctrl < matched),
        "per_class_match_rate": {
            str(c): (float(per_class_matched[c] / per_class_tot[c]) if per_class_tot[c] else None)
            for c in range(5)
        },
        "per_class_flips_on_unstable": {str(c): int(per_class_tot[c]) for c in range(5)},
    }


def run_probe(
    lstars: Any,
    margins: Any,
    flip_masks: dict[int, np.ndarray],
    flip_coords: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] | None = None,
    *,
    candidates: list[str],
    n_bins: int,
    control_stride: int,
    n_classes: int = 5,
    min_side: int = MIN_STRATUM_SIDE_PIXELS,
    edge_pixel_stride: int = 97,
) -> dict[str, Any]:
    n_pairs_total = int(lstars.shape[0])
    if n_pairs_total < 3:
        raise ValueError("VACUOUS: fewer than 3 pairs -- no interior pair has both neighbours")
    # Interior pairs only: the cross-pair candidates need both stride-2 neighbours, and a
    # pair with no flip mask contributes no lift information.
    interior = [t for t in range(1, n_pairs_total - 1) if t in flip_masks]
    if not interior:
        raise ValueError(
            "VACUOUS: no interior pair has BOTH temporal neighbours and a flip mask -- "
            "this is not a pass"
        )
    pair_ids = np.asarray(interior, dtype=np.int64)
    if control_stride % len(pair_ids) == 0:
        raise ValueError(
            f"control_stride {control_stride} is degenerate for {len(pair_ids)} pairs "
            "(the control would join each pair against itself)"
        )
    edges, margin_median = margin_bin_edges(margins, n_bins, pair_ids, edge_pixel_stride)
    h, w = int(lstars.shape[1]), int(lstars.shape[2])

    names = [
        *candidates,
        "margin_bin_CALIBRATION_exact",
        "margin_CALIBRATION_redundant",
        "flip_CALIBRATION_oracle",
    ]
    real = {n: _Strata(n_classes, n_bins) for n in names}
    ctrl = {n: _Strata(n_classes, n_bins) for n in names}

    for i, t in enumerate(pair_ids):
        t = int(t)
        cls = np.asarray(lstars[t]).astype(np.int64)
        mg = np.asarray(margins[t], dtype=np.float32)
        mbin = np.searchsorted(edges, mg.ravel()).reshape(mg.shape).astype(np.int64)
        flip = flip_masks[t]
        t_ctrl = int(pair_ids[(i + control_stride) % len(pair_ids)])
        mg_c = np.asarray(margins[t_ctrl], dtype=np.float32)
        mbin_c = np.searchsorted(edges, mg_c.ravel()).reshape(mg_c.shape).astype(np.int64)

        for name in names:
            if name == "margin_bin_CALIBRATION_exact":
                # Constant within every (class, margin-bin) cell BY CONSTRUCTION =>
                # residual variance must be exactly 0. This validates the projection.
                val_r = mbin.astype(np.float64)
                hi_r = mbin >= (n_bins // 2)
                val_c = mbin_c.astype(np.float64)
                hi_c = mbin_c >= (n_bins // 2)
            elif name == "margin_CALIBRATION_redundant":
                val_r = mg.astype(np.float64)
                hi_r = mg < margin_median  # "hi" = HIGH FLIP RISK side = LOW margin (mid-bin split)
                val_c = mg_c.astype(np.float64)
                hi_c = mg_c < margin_median
            elif name == "flip_CALIBRATION_oracle":
                val_r = flip.astype(np.float64)
                hi_r = flip
                cf = flip_masks[t_ctrl]
                val_c = cf.astype(np.float64)
                hi_c = cf
            else:
                builder, is_binary, thr = CANDIDATE_FIELDS[name]
                raw_r = builder(lstars, t)
                raw_c = builder(lstars, t_ctrl)
                val_r = raw_r.astype(np.float64)
                val_c = raw_c.astype(np.float64)
                hi_r = raw_r if is_binary else (raw_r >= thr)
                hi_c = raw_c if is_binary else (raw_c >= thr)
            real[name].add(cls, mbin, val_r, hi_r, flip)
            ctrl[name].add(cls, mbin, val_c, hi_c, flip)

    results: dict[str, Any] = {}
    for name in names:
        red = real[name].residual_variance_fraction()
        lift_real = real[name].lift(min_side)
        lift_ctrl = ctrl[name].lift(min_side)
        r_mh = lift_real["stratified_mh_risk_ratio"]
        c_mh = lift_ctrl["stratified_mh_risk_ratio"]
        # The control must move the stratified lift substantially back toward 1. When the
        # stratified estimate is undefined (stratum-determined field) there is nothing for
        # the control to falsify, and we say so rather than claiming a fired control.
        if r_mh is None or c_mh is None:
            fired = None
        elif not np.isfinite(r_mh):
            # Real lift is unbounded (zero flips on the low side): the control fires iff it
            # is NOT also unbounded. Guarding this explicitly because inf - inf -> nan would
            # silently report a dead control as "not fired" and look like a real negative.
            fired = bool(np.isfinite(c_mh))
        else:
            real_excess = abs(r_mh - 1.0)
            ctrl_excess = abs(c_mh - 1.0) if np.isfinite(c_mh) else float("inf")
            fired = bool(
                real_excess > 0.0
                and (real_excess - ctrl_excess) / real_excess
                >= CONTROL_LIVENESS_MIN_LIFT_DROP_FRAC
            )
        results[name] = {
            "redundancy_leg": red,
            "lift_leg": lift_real,
            "lift_leg_pair_shuffled_control": lift_ctrl,
            "control_fired": fired,
            "control_liveness_min_lift_drop_frac": CONTROL_LIVENESS_MIN_LIFT_DROP_FRAC,
        }

    return {
        "schema": "ddm_ti1_teacher_nonredundancy_probe.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "research_only": True,
        "conditioning": {
            "sigma_algebra": "sigma(GT class, GT margin bin) -- everything the per-pair "
            "separable seg loss can express as a per-pixel weight from its own GT inputs",
            "n_classes": n_classes,
            "n_margin_bins": n_bins,
            "margin_bin_edges": [float(e) for e in edges],
            "margin_midbin_calibration_split_value": margin_median,
        },
        "scope": {
            "n_pairs_total": n_pairs_total,
            "n_interior_pairs_used": len(pair_ids),
            "first_pair": int(pair_ids[0].item()),
            "last_pair": int(pair_ids[-1].item()),
            "pixels_per_pair": h * w,
            "n_pixels": len(pair_ids) * h * w,
        },
        "control_stride": control_stride,
        "candidates": results,
        "error_typing_direction_leg": (
            type_flicker_errors(lstars, flip_coords, pair_ids, control_stride)
            if flip_coords is not None
            else {"available": False, "reason": "no flip coords supplied -- typing SKIPPED"}
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
        help="frozen-authority GT cache carrying lstars + margins",
    )
    ap.add_argument(
        "--flip-atlas",
        required=True,
        type=Path,
        help="flat realized-flip atlas .npz carrying pair/y/x (the student's error set)",
    )
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--margin-bins", type=int, default=10)
    ap.add_argument("--control-stride", type=int, default=CONTROL_STRIDE)
    ap.add_argument("--min-stratum-side", type=int, default=MIN_STRATUM_SIDE_PIXELS)
    ap.add_argument(
        "--candidates",
        default=",".join(CANDIDATE_FIELDS),
        help=f"comma-separated subset of {sorted(CANDIDATE_FIELDS)}",
    )
    ap.add_argument(
        "--limit-pairs",
        type=int,
        default=0,
        help="0 = all pairs. Any non-zero value is a SUBSET and is labelled as such.",
    )
    args = ap.parse_args(argv)

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap

    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    margins = open_stored_npy_memmap(args.gt_cache, "margins")
    if args.limit_pairs:
        k = min(args.limit_pairs, int(lstars.shape[0]))
        lstars, margins = lstars[:k], margins[:k]

    cand = [c.strip() for c in args.candidates.split(",") if c.strip()]
    unknown = [c for c in cand if c not in CANDIDATE_FIELDS]
    if unknown:
        raise SystemExit(f"unknown candidate field(s) {unknown}; known: {sorted(CANDIDATE_FIELDS)}")
    if not cand:
        raise SystemExit("VACUOUS: zero candidate fields requested -- this is not a pass")

    flip_masks, flip_coords = _flip_masks_by_pair(
        args.flip_atlas, (int(lstars.shape[1]), int(lstars.shape[2]))
    )
    report = run_probe(
        lstars,
        margins,
        flip_masks,
        flip_coords,
        candidates=cand,
        n_bins=args.margin_bins,
        control_stride=args.control_stride,
        min_side=args.min_stratum_side,
    )
    report["gt_cache_path"] = str(args.gt_cache)
    report["flip_atlas_path"] = str(args.flip_atlas)
    report["flip_atlas_sha256"] = _sha256(args.flip_atlas)
    report["scope"]["label"] = "FULL" if not args.limit_pairs else f"SUBSET_{args.limit_pairs}"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    sc = report["scope"]
    print(f"scope={sc['label']} interior_pairs={sc['n_interior_pairs_used']} px={sc['n_pixels']}")
    rc = 0
    ty = report["error_typing_direction_leg"]
    if ty.get("available"):
        print(
            f"  DIRECTION: on {ty['flips_on_unstable_pixels']} flips at temporally-unstable "
            f"pixels, realized label == a temporal-neighbour label "
            f"{ty['match_rate']:.4f} (pair-shuffled control {ty['match_rate_pair_shuffled_control']:.4f}, "
            f"control_fired={ty['control_fired']})"
        )
    else:
        print(f"  DIRECTION: SKIPPED -- {ty.get('reason')}")

    def _f(v: float | None) -> str:
        return "  UNDEF" if v is None else f"{v:7.4f}"

    for name, res in report["candidates"].items():
        red = res["redundancy_leg"]["residual_variance_fraction"]
        lf = res["lift_leg"]
        cf = res["lift_leg_pair_shuffled_control"]
        print(
            f"  {name:<34} residual_var_frac={red:.6f}  "
            f"crude_RR={_f(lf['crude_risk_ratio'])}  MH_RR={_f(lf['stratified_mh_risk_ratio'])}  "
            f"ctrl_MH_RR={_f(cf['stratified_mh_risk_ratio'])}  "
            f"strata={lf['n_usable_strata']}/{lf['n_strata_total']}  "
            f"control_fired={res['control_fired']}"
        )
        mh = lf["stratified_mh_risk_ratio"]
        if name == "flip_CALIBRATION_oracle" and (mh is None or mh < 10.0):
            print("WARNING: oracle calibration lift < 10x -- the lift leg cannot see signal")
            rc = 3
        if name == "margin_bin_CALIBRATION_exact" and red > 1e-9:
            print("WARNING: exact calibration has non-zero residual -- projection is BROKEN")
            rc = 3
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
