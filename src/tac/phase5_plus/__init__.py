# SPDX-License-Identifier: MIT
"""``tac.phase5_plus`` — Phase 5+ theoretical floor saturation (CONJECTURE-ONLY).

This package is the **publication moonshot** scaffold per the EXTREME-OBSESSION
Fields-medal grand council §"Phase 5+" deliberation
(memo: ``fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md``).

Phase 5+ targets the lower-tail of the Phase 2/3 council Bayesian posterior:

  S_floor = 0.131 ± 0.013 (lower bound 0.116)

via four conjectural mechanisms:

  1. **Score-aware substrate co-evolution** — joint optimization of substrate
     + bit allocation + renderer + codec via Phase 3 IB Lagrangian extended
     with substrate-evolution dual.
  2. **Closed-form Fisher-Rao geodesic on the score manifold** — T7 evolved
     to Phase 5; closed-form geodesic instead of numerical surrogate.
  3. **Cross-paradigm composition** — all winning Phase 1+2+3 components
     end-to-end re-coordinated under one Lagrangian.
  4. **Joint-source coding refinement** — Berger 1971 finite-block correction
     + (N-1)/N Gaussian-Markov bound.

CONJECTURE-ONLY scope
---------------------

This package is **CONJECTURE-ONLY**. It carries:
  - ``DISPATCH_READY = False``
  - ``REQUIRES_OPERATOR_APPROVAL = True``
  - ``GATED_ON_PHASE3_ANCHOR = True`` (Phase 3 must land 0.115-0.130 first)
  - ``GPU_BUDGET_USD = 0.0`` (no GPU spend; SKETCH-ONLY for next 6+ months)
  - ``CATHEDRAL_AUTOPILOT_DISPATCH_QUEUE_INCLUDED = False``

The conjectural-band is tagged ``[conjecture; Phase 5+ council]``, NOT
``[predicted; ...]`` per CLAUDE.md ``forbidden_score_claims``. The distinction:
  - ``[predicted]`` = derived from a verifiable mechanism with explicit
    theorem citations and empirical sub-anchors
  - ``[conjecture]`` = mechanism named but not yet decomposed into individual
    verifiable predictions; reserved for moonshot research

Phase 5+ is the publication-grade research target, not the contest target.

CLAUDE.md compliance
--------------------

  - No code beyond design module + lane reg L0 SKETCH
  - No GPU dispatch path
  - All conjectural-band claims tagged ``[conjecture; Phase 5+ council]``
  - Substrate-engineering exception applies as in Phase 3 (anchored on
    Hinton-distilled aux scorer)
  - All theorems cited explicitly; no hand-waving
"""
from __future__ import annotations

PHASE5_PLUS_VERSION = "0.0.1-phase5plus-sketch-conjecture-only"
"""Semantic version. SKETCH-ONLY at L0; promotes to L1 only after Phase 3
empirical anchor lands at 0.115-0.130 [contest-CUDA verified].
"""

PHASE5_PLUS_CONJECTURAL_BAND_TAG = "[conjecture; Phase 5+ council; multi-source aggregated lower-tail; conditional on Phase 3 landing 0.115-0.130]"
"""Tag every Phase 5+ band reference with this string. Per CLAUDE.md
``forbidden_score_claims``, no Phase 5+ score number may appear in any
artifact without this tag.

Conjectural band per coherence council §1 lower-tail:
  0.118-0.131
  median ~0.124

These are MOONSHOT bands. They become predicted bands ONLY when Phase 3
empirical anchor lands AND the substrate-evolution dual converges.
"""

PHASE5_PLUS_PROVENANCE = {
    "schema_version": 1,
    "version": PHASE5_PLUS_VERSION,
    "phase": "5plus_sketch_conjecture_only",
    "conjectural_score_band": PHASE5_PLUS_CONJECTURAL_BAND_TAG,
    "council_memo_refs": [
        "fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md",
        "feedback_grand_council_portfolio_coherence_journal_grade_20260509.md",
        "feedback_grand_council_fields_medal_phase2_floor_REBASELINE_with_integration_discipline_20260509.md",
    ],
    "compliance_tags": [
        "sketch_conjecture_only",
        "no_gpu_dispatch",
        "operator_approval_required_for_promotion",
        "substrate_engineering_exception_principled",
        "score_tag_conjecture_only",
        "no_cathedral_autopilot_dispatch_queue",
    ],
    "dispatch_readiness": {
        "DISPATCH_READY": False,
        "REQUIRES_OPERATOR_APPROVAL": True,
        "GATED_ON_PHASE3_ANCHOR": True,
        "GPU_BUDGET_USD": 0.0,
        "CATHEDRAL_AUTOPILOT_DISPATCH_QUEUE_INCLUDED": False,
        "PROMOTION_BLOCKER_TO_L1": "Phase 3 empirical anchor at 0.115-0.130 [contest-CUDA verified]",
    },
    "conjectural_mechanisms": {
        "score_aware_substrate_co_evolution": (
            "Phase 3 IB Lagrangian extended with substrate-evolution dual; "
            "5-axis ADMM (R + d_seg + d_pose + substrate + codec). Convergence "
            "requires T19 adaptive-ρ on 5 axes simultaneously."
        ),
        "closed_form_fisher_rao_geodesic": (
            "T7 evolved to Phase 5; closed-form geodesic on score manifold "
            "instead of numerical surrogate. Tao consult: harmonic analysis "
            "on score manifold required."
        ),
        "cross_paradigm_composition_phase_1_plus_2_plus_3": (
            "All winning Phase 1+2+3 components end-to-end re-coordinated under "
            "one Lagrangian. Sub-additivity caveat: T7+T8+T11 may collapse "
            "to one effective surrogate post-coordination."
        ),
        "joint_source_coding_refinement_finite_block": (
            "Berger 1971 §4.5 finite-block correction + (N-1)/N Gaussian-Markov "
            "bound. Applies to per-pair latent stream when N (pairs) is small."
        ),
    },
}

from tac.phase5_plus.theoretical_floor_saturation import (  # noqa: E402
    TheoreticalFloorSaturationConjecture,
    TheoreticalFloorSaturationConfig,
    phase5_plus_conjectural_band,
    phase5_plus_lagrangian_form,
)

__all__ = [
    "PHASE5_PLUS_VERSION",
    "PHASE5_PLUS_PROVENANCE",
    "PHASE5_PLUS_CONJECTURAL_BAND_TAG",
    "TheoreticalFloorSaturationConjecture",
    "TheoreticalFloorSaturationConfig",
    "phase5_plus_conjectural_band",
    "phase5_plus_lagrangian_form",
]
