# SPDX-License-Identifier: MIT
"""Trainer-side wire-in SCAFFOLD for score-weighted reconstruction loss.

Per RESPAWN-MG-7-BUNDLE exploit #2 (sister consumer at
``tac.cathedral_consumers.score_weighted_reconstruction_error_consumer``).
Drop-in replacement for canonical ``tac.losses.score_pair_components``
augmented with per-pixel M_contest weighting.

**SCAFFOLD STATUS**: per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY" non-negotiable + Catalog #220 + Catalog #240, this module is
DESIGN-TIME SCAFFOLD ONLY. Operator routing is required before any production
trainer adopts this loss as primary objective. Adoption requires:

1. Empirical paired contest-CUDA dispatch comparing M_contest-weighted loss
   vs raw L2 on the same substrate; the score-weighted loss MUST measurably
   beat raw L2 (per CLAUDE.md "Apples-to-apples evidence discipline").
2. The trainer MUST carry research_only=True OR pass Catalog #325 per-
   substrate symposium gate.
3. The trainer MUST route M_contest through the canonical producer
   ``tac.master_gradient_comparison.multi_granularity.extract_M_contest``
   (Catalog #318 chain-rule discipline).

## Canonical-vs-unique decision per layer

- L2 component: ADOPT canonical (same per-pixel reconstruction error).
- Weighting layer: FORK from canonical scorer-loss helper because the
  canonical helper does not accept a per-pixel M_contest weight tensor.
- Provenance contract: ADOPT canonical Provenance per Catalog #323; this
  module's outputs are PREDICTED until paired-axis empirical evidence
  promotes them.

## Observability surface

Per Catalog #305:

1. Inspectable per layer: returns scalar loss + per-pair-axis breakdown.
2. Decomposable per signal: per-pair contribution is queryable.
3. Diff-able across runs: deterministic numpy compute.
4. Queryable post-hoc: helper is operator-runnable.
5. Cite-able: docstring cites M_contest provenance.
6. Counterfactual-able: per-pair structure allows ablation.

## 9-dimension success checklist evidence

(See sister consumer at
``tac.cathedral_consumers.score_weighted_reconstruction_error_consumer``
for the full 9-dim checklist.)

## Cargo-cult audit per assumption

(See sister consumer.)
"""
from __future__ import annotations

from collections.abc import Sequence

# This module is SCAFFOLD; we do NOT import torch / pytorch_lightning at module
# load time so the scaffold can be inspected on CPU-only review boxes. Trainers
# that adopt the loss as primary objective MUST guard torch import at their
# own loading point.


def compute_score_weighted_reconstruction_loss_scaffold(
    I_inflated,
    I_contest,
    M_contest,
    *,
    per_pair_axis_breakdown: bool = False,
):
    """Compute scalar score-weighted reconstruction loss (numpy SCAFFOLD).

    SCAFFOLD reference implementation in pure numpy. Production trainers
    MUST adapt this to PyTorch (with autograd) per the canonical trainer
    pattern in ``src/tac/substrates/_shared/score_aware_common.py``.

    Formula: ``loss = mean over (pair, pixel) of (I_inflated - I_contest)^2 *
    ||M_contest[:, axis, :, :]||^2``.

    Per Catalog #287 evidence-tag discipline: this scaffold's output is
    [predicted] axis tag; promotion to contest-axis requires paired
    empirical evidence.

    Args:
        I_inflated: array of shape (N_pairs, 3, H, W) - encoder reconstruction.
        I_contest: array of shape (N_pairs, 3, H, W) - contest ground truth.
        M_contest: array of shape (N_pairs, 3, H, W) - per-pixel scorer
            gradient (loaded from ContestGradientTensor.load() per producer
            surface).
        per_pair_axis_breakdown: if True, also returns per-pair contribution
            (for observability per Catalog #305).

    Returns:
        If per_pair_axis_breakdown=False: scalar loss as float.
        Else: tuple of (scalar loss, per-pair contribution ndarray of shape
        (N_pairs,)).

    Raises:
        ValueError: on shape mismatch.
        RuntimeError: if numpy unavailable.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for score-weighted recon scaffold") from exc

    inflated = np.asarray(I_inflated, dtype=np.float64)
    contest = np.asarray(I_contest, dtype=np.float64)
    m = np.asarray(M_contest, dtype=np.float64)
    if inflated.shape != contest.shape:
        raise ValueError(
            f"I_inflated shape {inflated.shape} != I_contest shape {contest.shape}"
        )
    if inflated.ndim != 4 or inflated.shape[1] != 3:
        raise ValueError(
            f"I_inflated must have shape (N_pairs, 3, H, W); got {inflated.shape}"
        )
    n_pairs, _, h, w = inflated.shape
    if m.shape != (n_pairs, 3, h, w):
        raise ValueError(
            f"M_contest shape {m.shape} != expected ({n_pairs}, 3, {h}, {w})"
        )
    m_l2_sq = np.sum(np.square(m), axis=1)  # (N_pairs, H, W)
    diff = inflated - contest
    err_per_pixel = np.sum(np.square(diff), axis=1)  # (N_pairs, H, W)
    weighted = err_per_pixel * m_l2_sq
    total_loss = float(weighted.mean())
    if per_pair_axis_breakdown:
        # Per-pair contribution = mean over (H, W) for that pair.
        per_pair = weighted.mean(axis=(1, 2)).astype(float)
        return total_loss, per_pair
    return total_loss


def get_scaffold_status() -> dict:
    """Return scaffold status metadata per Catalog #220 / #240 contract.

    The trainer scaffold is RESEARCH_ONLY by construction; production
    adoption requires paired empirical evidence + Catalog #325 per-
    substrate symposium gate per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM
    via adversarial grand council symposium" non-negotiable.
    """
    return {
        "scaffold_kind": "trainer_loss_drop_in_replacement",
        "exploit_id": 2,
        "exploit_name": "score_weighted_reconstruction_error",
        "research_only": True,
        "dispatch_enabled": False,
        "production_adoption_blockers": (
            "paired_contest_cuda_dispatch_required",
            "per_substrate_symposium_required",
            "canonical_producer_surface_routing_required_per_catalog_318",
        ),
        "canonical_helper_invocation": (
            "tac.training.score_weighted_reconstruction_loss."
            "compute_score_weighted_reconstruction_loss_scaffold"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
    }


__all__ = [
    "compute_score_weighted_reconstruction_loss_scaffold",
    "get_scaffold_status",
]
