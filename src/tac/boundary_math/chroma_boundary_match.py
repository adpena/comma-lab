# SPDX-License-Identifier: MIT
"""Annulus-directed chroma-boundary match (LEVER-4c) — the $0 numpy REFERENCE TWIN of the trainer loss term.

WHAT IT IS (mechanically). SegNet reads RGB, so its per-pixel argmax depends on CHROMA (rgb − BT.601-luma),
not just luma. The chroma-DOF probe a3e9f0bd (MEASURED n96, 100% L*-match to the frozen SegNet, #276 chroma
lever) found removing chroma (constant-luma) FLIPS ``7.54%`` of Lane→Road and ``4.38%`` of Movable→Undrivable,
with ``93.4%`` of those chroma-flips living in the ``margin < 1`` fragile ANNULUS (→ ``33.7%`` at margin<0.25);
the flips are LUMA-INDEPENDENT (constant-luma DESAT still flips ``3.1%`` of the annulus) and the SegNet
margin-gradient energy is ``78.8%`` luma / ``21.2%`` chroma. So chroma is a PROVEN INDEPENDENT d_seg BOUNDARY
SHARPENER (power at the knife-edge large-signal flips, NOT bulk), ORTHOGONAL to the geometry levers.

The witness UNDER-exploits it: its rendered chroma converges to a near per-class CONSTANT palette (the seg CE
only rewards argmax; nothing supervises per-pixel chroma) whose inter-class separation is SMALLER than the
intra-class chroma std → it cannot PAINT the per-pixel boundary chroma SegNet keys on. So LEVER-4c is an
additive chroma-MATCH loss over the fragile annulus:

    chroma(x)   = rgb(x) − BT.601_luma(x)·[1,1,1]          (LUMA-INVARIANT: rgb + c·[1,1,1] leaves it fixed)
    ann(x)      = 1[ m_gt(x) < band ]                       (θ-independent fragile-annulus mask, {0,1})
    L_chroma    = w · ( Σ_x ‖chroma(f1)(x) − chroma(GT)(x)‖² · ann(x) ) / ( Σ_x ann(x) + 1e-6 )

on the SHARED realized-through-R render ``f1`` (the SAME render the SegNet forward / ``_signed`` margin come
from — NO 2nd render, NO 2nd SegNet forward). The GT chroma target + annulus weight are θ-INDEPENDENT
constants precomputed ONCE per pair (stop-grad by construction); the witness chroma is the differentiable
path that pulls the per-pixel RGB head (``self.out = Linear(hidden, 3)``, which HAS per-pixel chroma capacity
— the constant palette is a convergence habit, not a structural ceiling) to paint the boundary chroma.

WHY it is byte-free AND rule-118 FREE. The term adds ZERO trainable params (a TRAIN-TIME loss reweighting,
ships nothing) → the archive is unchanged. The chroma target + annulus mask are a deterministic function of
the FIXED GT frame + GT margin field — a geometry PRIOR, not learned/video-derived weights. Because chroma is
LUMA-INVARIANT by construction (subtracting BT.601 luma leaves ``rgb + c·[1,1,1]`` unchanged) the term is
ORTHOGONAL to every luma lever and is a per-pixel chroma MATCH at the boundary, NOT a full-RGB reconstruction.

THE BYTE-IDENTITY CONTRACT (the NO-FAKE degeneracies, all tested). ``weight == 0`` ⇒ the trainer NEVER
constructs the branch (default-OFF path, ``chroma_bnd_w > 0.0`` gate) AND the providers stay ``None`` ⇒ the
loss graph is byte-identical; this reference returns EXACTLY ``0.0`` there as the runtime twin. An EMPTY
annulus (no pixel with ``m_gt < band``) ⇒ ``Σ ann == 0`` ⇒ the ``+1e-6`` denominator guard makes the term
``0.0`` (never a /0). A PERFECT chroma match on the annulus ⇒ the squared error is 0 ⇒ the term is ``0.0``.

VERDICT STATUS (do NOT strip this caveat). LEVER-4c ledger row: the 7.54% / 4.38% / 93.4% numbers are a
MEASURED chroma-REMOVAL ABLATION (the DOF EXISTENCE proof, eq ``chroma_decides_lane_and_movable_at_annulus_v1``)
— NOT the ADD-BACK score ΔS of THIS chroma-MATCH loss term through the witness. That score ΔS is
**UNMEASURED**; the owed exit criterion is a CONVERGED n600 byte-close A/B (ON vs OFF ckpts; the surviving
annulus flips must shift toward the GT chroma, else terminal-finding). This module is the REFERENCE TWIN + is
unit-testable at $0; it makes NO score claim. means != ends: pointer 0.19110 UNMOVED.

BIT-FAITHFUL to the trainer. The formulas here mirror
``experiments/train_levelset_witness_realized_through_R_mlx.py`` EXACTLY: :func:`bt601_chroma` is the numpy
twin of the witness/GT chroma builds (``0.299 R + 0.587 G + 0.114 B``; L4938-4940 loss + L5749 provider),
:func:`annulus_mask` is the numpy twin of the ``(_mg < band)`` precompute (L5750), and
:func:`chroma_boundary_term` is the numpy twin of the loss block (``sum(cdiff2·cw)/(sum(cw)+1e-6)``;
L4941-4942). Pure numpy (no MLX / torch / GPU) so the mechanism is verifiable at $0.

DSL leg: ``tac.witness_dsl.curriculum_dsl.SegChromaBoundary``. Equations leg:
``tac.canonical_equations.chroma_boundary_match_20260709`` (chroma_boundary_annulus_match_hinge_v1).
"""
from __future__ import annotations

import numpy as np

# BT.601 luma coefficients — the SAME the witness ``_apply_chroma`` and the trainer chroma builds use.
_BT601 = (0.299, 0.587, 0.114)
# Denominator guard mirroring the trainer's ``mx.sum(_cw) + 1e-6`` (never a /0 on an empty annulus).
_MASK_DENOM_EPS = 1e-6


def bt601_chroma(rgb: np.ndarray) -> np.ndarray:
    """LUMA-INVARIANT chroma ``rgb − BT.601_luma·[1,1,1]`` of an ``(..., 3)`` RGB array (last axis = RGB).

    The numpy twin of the trainer's witness-chroma (``_cwit = _f1 - _lum_w``, L4938-4940) and GT-chroma
    (``_chr = _rs - _lum[..., None]``, L5749) builds: ``luma = 0.299 R + 0.587 G + 0.114 B`` broadcast over
    the 3 channels. Adding a constant luma ``c`` to all three channels (``rgb + c·[1,1,1]``) leaves the
    result UNCHANGED — that is the luma-invariance that makes this ORTHOGONAL to every luma lever. Fail-closed
    on a non-3-channel last axis (NO-FAKE: never silently compute the wrong basis)."""
    x = np.asarray(rgb, np.float64)
    if x.ndim < 1 or x.shape[-1] != 3:
        raise ValueError(f"bt601_chroma: rgb last axis must be 3 (RGB); got shape {x.shape}")
    luma = _BT601[0] * x[..., 0:1] + _BT601[1] * x[..., 1:2] + _BT601[2] * x[..., 2:3]  # (..., 1)
    return (x - luma).astype(np.float32)                                                 # (..., 3)


def annulus_mask(gt_margin: np.ndarray, band: float = 1.0) -> np.ndarray:
    """θ-independent fragile-annulus mask ``(H, W)`` in ``{0.0, 1.0}`` = ``1[ GT top1-top2 margin < band ]``.

    The numpy twin of the trainer's ``_ann = (_mg < chroma_bnd_band)`` precompute (L5750). A pixel is
    supervised only where the GT margin is BELOW ``band`` (chroma's d_seg power is at the knife-edge:
    MEASURED gt_n96 band 1.0 captures 93.4% of chroma-flips, → 33.7% at 0.25). Note the STRICT ``<`` (NOT
    half-open like the horizon lever) — this mirrors the trainer exactly. Fail-closed on a non-2-D input and
    on ``band <= 0`` (an empty band that would silently do nothing)."""
    m = np.asarray(gt_margin, np.float64)
    if m.ndim != 2:
        raise ValueError(f"annulus_mask: gt_margin must be 2-D (H,W); got shape {m.shape}")
    if not (float(band) > 0.0):
        raise ValueError(
            f"annulus_mask: band ({band!r}) must be > 0 — a non-positive band selects NO pixel (the "
            "chroma-match term would silently do nothing; LEVER-4c measured band 1.0).")
    return (m < float(band)).astype(np.float32)


def chroma_boundary_term(
    f1_rgb: np.ndarray,
    gt_chroma: np.ndarray,
    annulus_w: np.ndarray,
) -> float:
    """Annulus-mean squared chroma error ``Σ ‖chroma(f1) − chroma(GT)‖² · ann / (Σ ann + 1e-6)`` (the numpy
    twin of the trainer loss block, WITHOUT the outer ``w`` weight — see :func:`chroma_boundary_loss`).

    ``f1_rgb``: the realized through-R witness RGB frame ``(H, W, 3)`` (its chroma is computed here via
    :func:`bt601_chroma`, matching the trainer's differentiable ``_cwit``). ``gt_chroma``: the θ-independent
    GT chroma target ``(H, W, 3)`` (already luma-subtracted, matching the precomputed ``_chroma_gt_prov``).
    ``annulus_w``: the ``(H, W)`` annulus mask from :func:`annulus_mask`.

    ``err(x) = Σ_c (chroma(f1)(x,c) − gt_chroma(x,c))²``; term ``= Σ_x err·ann / (Σ_x ann + 1e-6)``. Zero
    where the annulus mask is 0 (off-band) and where the chroma matches exactly. An all-zero annulus ⇒ ``Σ
    ann == 0`` ⇒ the ``+1e-6`` guard returns ``0.0`` (never a /0). Fail-closed on a grid mismatch (NO-FAKE:
    never silently broadcast the wrong grid)."""
    cwit = bt601_chroma(f1_rgb)                       # (H,W,3) witness chroma (luma-invariant)
    cgt = np.asarray(gt_chroma, np.float64)           # (H,W,3) GT chroma target (already luma-subtracted)
    aw = np.asarray(annulus_w, np.float64)            # (H,W) annulus weight
    if cwit.shape != cgt.shape:
        raise ValueError(
            f"chroma_boundary_term: witness chroma {cwit.shape} and gt_chroma {cgt.shape} must share the grid")
    if cwit.shape[:-1] != aw.shape:
        raise ValueError(
            f"chroma_boundary_term: chroma grid {cwit.shape[:-1]} and annulus_w {aw.shape} must match")
    cdiff2 = np.sum(np.square(cwit.astype(np.float64) - cgt), axis=-1)  # (H,W) 3-chan sq chroma error
    return float(np.sum(cdiff2 * aw) / (aw.sum() + _MASK_DENOM_EPS))    # mean over the annulus px


def chroma_boundary_loss(
    f1_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    gt_margin: np.ndarray,
    *,
    weight: float = 0.0,
    band: float = 1.0,
) -> float:
    """The full LEVER-4c loss contribution ``w · annulus-mean ‖chroma(f1) − chroma(GT)‖²`` (the exact numpy
    twin of ``L += chroma_bnd_w * chroma_bnd_term`` / ``terms_out["chroma_boundary"]`` in the trainer).

    ``weight == 0`` (DEFAULT) ⇒ returns EXACTLY ``0.0`` (the trainer skips the branch entirely AND the
    providers stay None — byte-identity), computing NO chroma target and NO mask. Otherwise builds the GT
    chroma target (:func:`bt601_chroma` of ``gt_rgb``) + the annulus mask (:func:`annulus_mask` of
    ``gt_margin``) and returns ``weight * chroma_boundary_term(...)``. ``gt_rgb`` is the GT frame at the SAME
    (SegNet-input) resolution as ``f1_rgb`` (in the trainer the camera GT is bilinear-resized to SegNet res
    BEFORE luma subtraction; that resize is a provider concern, so this twin takes the already-resized GT).
    Fail-closed on ``weight < 0``."""
    w = float(weight)
    if w < 0.0:
        raise ValueError(f"chroma_boundary_loss: weight must be >= 0, got {weight!r}")
    if w == 0.0:
        return 0.0  # default-OFF branch: byte-identical (the trainer never constructs the term)
    gt_chroma = bt601_chroma(gt_rgb)                      # (H,W,3) GT chroma target (luma-invariant)
    ann = annulus_mask(gt_margin, band=band)             # (H,W) fragile-annulus mask
    return w * chroma_boundary_term(f1_rgb, gt_chroma, ann)


__all__ = [
    "annulus_mask",
    "bt601_chroma",
    "chroma_boundary_loss",
    "chroma_boundary_term",
]
