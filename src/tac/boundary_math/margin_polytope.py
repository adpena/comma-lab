# SPDX-License-Identifier: MIT
"""Per-pixel margin-polytope free-budget ``b(p) = m(p)/||g_p||`` (the §4 system).

Source spec §4: the evaluator-equivalence CELL for frame1 is the polytope

    ( J_{L*(p),p} - J_{c,p} ) . delta  >=  - m_{p,c}     for every wrong class c != L*(p)

A first-order (Taylor about GT) system of linear inequalities.  The per-pixel
**free budget** is how far ``delta`` can move before the nearest argmax constraint
binds:

    b(p) = m(p) / ||g_p||

where ``m(p)`` is the SegNet top1-top2 logit margin (distance-to-flip in logit
space) and ``g_p = J_{L*(p),p} - J_{c2,p}`` is the Jacobian-difference toward the
runner-up class ``c2`` (the steepest flip direction).  Large ``b(p)`` = interior =
slack = free to coarsen/quantize/region-fill.  Small ``b(p)`` = boundary = tight =
the only bits we pay.

Reuse (NO FAKE, only what the code honors): the margin field ``m(p)`` comes from the
real SegNet forward via ``tac.optimization.frame1_seg_repair_atoms.measure_segnet_argmax``
(top1-top2 margin); the Jacobian-difference norm ``||g_p||`` is the polytope
coefficient magnitude.  When a measured Jacobian atlas is supplied (per-pixel
``||g_p||`` from ``tac.optimization.evaluator_response_atlas``), we use it; otherwise
we expose the margin-only budget ``b(p) = m(p)`` (a monotone proxy: with ``||g_p||``
unknown the ranking by margin is preserved up to the per-pixel gradient scale) and
flag ``jacobian_supplied=False`` so callers never mistake it for the full polytope
radius.  We do NOT fabricate a Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerPixelFreeBudget:
    """The per-pixel polytope free-budget field + its provenance.

    ``budget`` is ``(H, W)`` float: how far the pixel may move before its nearest
    argmax constraint binds (in input-pixel units when ``jacobian_supplied``, else in
    logit-margin units as a monotone ranking proxy).  ``margin`` is the raw top1-top2
    logit margin.  ``free_mask`` marks interior (large-budget) pixels at the supplied
    quantile threshold.  ``jacobian_supplied`` records whether the full polytope
    radius (margin / ||g_p||) or the margin-only proxy was computed.
    """

    budget: np.ndarray
    margin: np.ndarray
    free_mask: np.ndarray
    jacobian_supplied: bool
    free_quantile: float


def free_budget_from_margin_jacobian(
    margin_hw: np.ndarray,
    jacobian_norm_hw: np.ndarray | None = None,
    *,
    free_quantile: float = 0.5,
    eps: float = 1e-8,
) -> PerPixelFreeBudget:
    """Compute ``b(p) = m(p)/||g_p||`` (or the margin-only proxy if no Jacobian).

    Parameters
    ----------
    margin_hw : (H, W) float
        SegNet top1-top2 logit margin (>= 0); 0 == on the argmax boundary.
    jacobian_norm_hw : (H, W) float or None
        Per-pixel ``||g_p|| = ||J_{L*,p} - J_{c2,p}||``.  When given, the budget is the
        true polytope radius; when None, the margin-only proxy is returned.
    free_quantile : float
        Pixels with budget >= this quantile are flagged interior/free.
    """

    m = np.asarray(margin_hw, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"margin must be (H, W); got {m.shape}")
    if (m < -1e-9).any():
        raise ValueError("margin must be non-negative (top1-top2 logit gap)")

    if jacobian_norm_hw is None:
        budget = m.copy()
        jac_supplied = False
    else:
        g = np.asarray(jacobian_norm_hw, dtype=np.float64)
        if g.shape != m.shape:
            raise ValueError(f"jacobian_norm shape {g.shape} != margin shape {m.shape}")
        if (g < 0).any():
            raise ValueError("jacobian norm must be non-negative")
        budget = m / np.maximum(g, eps)
        jac_supplied = True

    if not 0.0 <= free_quantile <= 1.0:
        raise ValueError("free_quantile must be in [0, 1]")
    threshold = float(np.quantile(budget, free_quantile))
    free_mask = budget >= threshold

    return PerPixelFreeBudget(
        budget=budget,
        margin=m,
        free_mask=free_mask,
        jacobian_supplied=jac_supplied,
        free_quantile=free_quantile,
    )


def boundary_pixel_mask(margin_hw: np.ndarray, margin_threshold: float = 0.5) -> np.ndarray:
    """Pixels with margin below ``margin_threshold`` — the tight (paid) boundary set.

    Spec §4 anchor: ~91% of boundary pixels have margin < 0.5; those are the only
    bits the carrier pays.
    """

    m = np.asarray(margin_hw, dtype=np.float64)
    return m < margin_threshold
