# SPDX-License-Identifier: MIT
"""Horizon-weighted margin (#169 lever) — the $0 numpy REFERENCE TWIN of the trainer loss term.

WHAT IT IS (mechanically). The frontier residual d_seg flips split by the GT top-2 SegNet argmax MARGIN
(``|margin| = top1 − top2`` logit gap): the ``<0.05``-margin flips are IRREDUCIBLE frozen-SegNet
label-noise (a near coin-flip, ~193× concentrated — chasing them is FITTING NOISE), while the flips at GT
margin ∈ ``[0.3, 0.5]`` are the ONLY ones both REDUCIBLE and STABLY-DECIDED (oracle ceiling ΔS≈0.024 at
margin≥0.3 / 0.012 at margin≥0.5; derivation ``dseg_reducibility_gt_margin_verdict_20260623.md``). 97.8%
of the frontier d_seg lives in the horizon band (SEG rows ~96-288, where sky/far meets the ground classes).

So the #169 lever is a ONE-SIDED HINGE ``L_hz = w_h · mean_{mask} relu(m_target − m_wit)`` on the SHARED
realized through-R witness GT-class margin ``m_wit`` (``= gt_class_logit − top_competitor_logit``; the
``_signed`` field, #141 — reuses the SAME SegNet forward the seg loss already ran, NO 2nd forward, 0
archive bytes), STRATIFIED to the θ-INDEPENDENT mask ``(row ∈ [row_lo, row_hi)) AND (GT margin ∈
[lo, hi])``. It pushes ONLY the reducible confident-GT band toward the ``m_target`` ceiling and EXCLUDES
the ``<lo`` irreducible label-noise BY CONSTRUCTION. Zero contribution/gradient where ``m_wit ≥ m_target``
(satisficing — do not over-push into the noise regime).

WHY it is byte-free AND rule-118 FREE. The term adds ZERO trainable params (it is a TRAIN-TIME loss
reweighting, ships nothing) → the archive is unchanged. The stratified mask is a deterministic function of
the FIXED GT margin field + the row band — θ-independent, precomputed ONCE per pair (numpy, no SegNet), a
geometry PRIOR not learned/video-derived weights.

THE BYTE-IDENTITY CONTRACT (the NO-FAKE degeneracies, all tested). ``weight == 0`` ⇒ the trainer NEVER
constructs the branch (default-OFF path) ⇒ the loss graph is byte-identical; this reference returns EXACTLY
``0.0`` there as the runtime twin. An EMPTY stratified mask (no pixel in the band) ⇒ ``sum(mask)==0`` ⇒ the
``+1e-6`` denominator guard makes the term ``0.0`` (never a /0). ``m_wit ≥ m_target`` everywhere in the band
⇒ ``relu`` is 0 ⇒ the term is ``0.0`` (satisficing).

VERDICT STATUS (do NOT strip this caveat). #169 lever ledger row: est ΔS 0.012–0.024 (MEASURED oracle
CEILING on the reducible band), **A/B arm NOT a claim** — the owed exit criterion is a CONVERGED n600
byte-close A/B (re-run ``tools/measure_dseg_reducibility_gt_margin.py --n-pairs 600`` on the ON vs OFF
ckpts; require the surviving flips shift to HIGHER GT margin, else terminal-finding). This module is the
REFERENCE TWIN + is unit-testable at $0; it makes NO score claim. means != ends: pointer 0.19110 UNMOVED.

BIT-FAITHFUL to the trainer. The formulas here mirror
``experiments/train_levelset_witness_realized_through_R_mlx.py`` EXACTLY: the mask build (rows clamped to
the real grid, ``margin ∈ [lo, hi)`` half-open, AND the row band) is the numpy twin of the ``_hz_mask_prov``
precompute (L5615-5628), and the hinge term is the numpy twin of the loss block (L4953-4962). Pure numpy
(no MLX / torch / GPU) so the mechanism is verifiable at $0.

DSL leg: ``tac.witness_dsl.curriculum_dsl.HorizonWeightedMargin``. Equations leg:
``tac.canonical_equations.horizon_weighted_margin_20260709``.
"""
from __future__ import annotations

import numpy as np

# Denominator guard mirroring the trainer's ``mx.sum(_hz_mask) + 1e-6`` (never a /0 on an empty band).
_MASK_DENOM_EPS = 1e-6


def horizon_stratified_mask(
    gt_margin: np.ndarray,
    *,
    row_lo: int = 96,
    row_hi: int = 288,
    margin_lo: float = 0.3,
    margin_hi: float = 0.5,
) -> np.ndarray:
    """θ-independent stratified mask ``(H, W)`` in ``{0.0, 1.0}`` = (row ∈ [row_lo, row_hi)) AND (GT margin
    ∈ [margin_lo, margin_hi)). The numpy twin of the trainer's ``_hz_mask_prov`` precompute.

    ``gt_margin``: the per-pair GT top1-top2 SegNet argmax margin field ``(H, W)`` (``>= 0`` by
    construction). The band is HALF-OPEN ``[lo, hi)`` — ``margin == lo`` is IN (reducible edge), ``margin
    == hi`` is OUT (confidently-decided) — matching the trainer's ``(_mg >= hz_lo) & (_mg < hz_hi)``.

    Rows are CLAMPED to the real grid ``H`` exactly like the trainer (``_rlo = max(0, min(row_lo, H-1))``;
    ``_rhi = max(_rlo+1, min(row_hi, H))``) so an over-wide default degrades to the full column and NEVER
    crashes/empties by row alone. Fail-closed on a non-2-D input and on ``margin_lo >= margin_hi``."""
    m = np.asarray(gt_margin, np.float64)
    if m.ndim != 2:
        raise ValueError(f"horizon_stratified_mask: gt_margin must be 2-D (H,W); got shape {m.shape}")
    if not (float(margin_lo) < float(margin_hi)):
        raise ValueError(
            f"horizon_stratified_mask: margin_lo ({margin_lo!r}) must be < margin_hi ({margin_hi!r}) — "
            "the reducible GT-margin band [lo,hi) must be non-empty (#169 measured band [0.3,0.5]).")
    if not (int(row_lo) < int(row_hi)):
        raise ValueError(
            f"horizon_stratified_mask: row_lo ({row_lo!r}) must be < row_hi ({row_hi!r}) — the horizon "
            "band must be a non-empty SEG-row range (#169 measured band rows ~96-288).")
    h, w = m.shape
    rlo = max(0, min(int(row_lo), h - 1))
    rhi = max(rlo + 1, min(int(row_hi), h))
    band = ((m >= float(margin_lo)) & (m < float(margin_hi))).astype(np.float64)  # reducible GT-margin band
    row = np.zeros((h, w), np.float64)
    row[rlo:rhi, :] = 1.0                                                          # horizon row band
    return (band * row).astype(np.float32)


def horizon_margin_hinge_term(
    m_wit: np.ndarray,
    mask: np.ndarray,
    m_target: float = 0.5,
) -> float:
    """One-sided satisficing hinge ``mean_{mask} relu(m_target − m_wit)`` (the numpy twin of the trainer
    loss block, WITHOUT the outer ``w_h`` weight — see :func:`horizon_weighted_margin_loss`).

    ``m_wit``: the realized through-R witness GT-class margin ``(H, W)`` (``_signed`` = gt-class logit −
    top-competitor logit). ``mask``: the ``(H, W)`` stratified mask from :func:`horizon_stratified_mask`.

    ``hinge(px) = relu(m_target − m_wit(px)) * mask(px)``; term ``= sum(hinge) / (sum(mask) + 1e-6)``. Zero
    where ``m_wit ≥ m_target`` (satisficing — the relu saturates) and where ``mask == 0`` (off-band). An
    all-zero mask ⇒ ``sum(mask) == 0`` ⇒ the ``+1e-6`` guard returns ``0.0`` (never a /0). Fail-closed on a
    shape mismatch (NO-FAKE: never silently broadcast the wrong grid)."""
    mw = np.asarray(m_wit, np.float64)
    mk = np.asarray(mask, np.float64)
    if mw.shape != mk.shape:
        raise ValueError(
            f"horizon_margin_hinge_term: m_wit {mw.shape} and mask {mk.shape} must share the pixel grid")
    hinge = np.maximum(float(m_target) - mw, 0.0) * mk        # relu deficit on the band px
    return float(hinge.sum() / (mk.sum() + _MASK_DENOM_EPS))  # mean over the stratified band px


def horizon_weighted_margin_loss(
    m_wit: np.ndarray,
    gt_margin: np.ndarray,
    *,
    weight: float = 0.0,
    m_target: float = 0.5,
    margin_lo: float = 0.3,
    margin_hi: float = 0.5,
    row_lo: int = 96,
    row_hi: int = 288,
) -> float:
    """The full #169 loss contribution ``w_h · mean_{mask} relu(m_target − m_wit)`` (the exact numpy twin
    of ``L += hz_w * hz_term`` / ``terms_out["horizon_margin"]`` in the trainer).

    ``weight == 0`` (DEFAULT) ⇒ returns EXACTLY ``0.0`` (the trainer skips the branch entirely — byte-
    identity), computing NO mask. Otherwise builds the stratified mask and returns
    ``weight * horizon_margin_hinge_term(m_wit, mask, m_target)``. Fail-closed on ``weight < 0``."""
    w = float(weight)
    if w < 0.0:
        raise ValueError(f"horizon_weighted_margin_loss: weight must be >= 0, got {weight!r}")
    if w == 0.0:
        return 0.0  # default-OFF branch: byte-identical (the trainer never constructs the term)
    mask = horizon_stratified_mask(
        gt_margin, row_lo=row_lo, row_hi=row_hi, margin_lo=margin_lo, margin_hi=margin_hi)
    return w * horizon_margin_hinge_term(m_wit, mask, m_target)


__all__ = [
    "horizon_margin_hinge_term",
    "horizon_stratified_mask",
    "horizon_weighted_margin_loss",
]
