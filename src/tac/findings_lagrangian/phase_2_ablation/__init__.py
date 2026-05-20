# SPDX-License-Identifier: MIT
"""tac.findings_lagrangian.phase_2_ablation — per-adjuster ablation framework.

Per WAVE-3-DIM-1-PHASE-2-START (operator blanket approval 2026-05-20 per the
CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 1 Phase 2 blueprint) +
Carmack's T3 dissent verbatim *"the apparatus has been producing infrastructure
that protects the frontier; it has not been producing frontier"*.

This package is THE Phase 2 implementation surface that replaces the
hand-derived ``adjust_predicted_delta_for_*`` adjusters in
``tools/cathedral_autopilot_autonomous_loop.py::main()`` with solver-derived
dual variables from ``tac.findings_lagrangian``.

Phase 2 full scope = 4-7 weeks across the 10 adjusters. THIS package lands
the framework + the first 3 highest-leverage adjusters:

  1. ``adjust_predicted_delta_for_mdl_density`` (Tier A MDL density)
  2. ``adjust_predicted_delta_for_predicted_dispatch_risk`` (SLIM preflight)
  3. ``adjust_predicted_delta_for_composition_alpha_v2`` (composition additivity)

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: each existing
adjuster IS a per-track Lagrangian term L_i; dual variables alpha_i are
derived via the closed-form Gaussian posterior + 4-term findings
Lagrangian, not hand-set.

Quick start::

    from tac.findings_lagrangian.phase_2_ablation import (
        AdjusterAblationContext,
        AdjusterAblationVerdict,
        AblationMode,
        PHASE_2_ABLATION_POSTERIOR_PATH,
        compute_solver_dual_variable_for_adjuster,
        paired_comparison_against_hand_derived,
        append_paired_comparison_row,
        load_paired_comparison_rows_strict,
        SUPPORTED_ADJUSTERS,
        PROMOTION_THRESHOLD_MIN_ANCHORS,
        PROMOTION_THRESHOLD_MAX_REGRESSION_USD,
    )

    # Build context for one adjuster + candidate set.
    ctx = AdjusterAblationContext(
        adjuster_name="mdl_density",
        mode=AblationMode.PAIRED_COMPARISON,
        ...
    )

    # Run paired comparison.
    verdict = paired_comparison_against_hand_derived(
        adjuster_name="mdl_density",
        candidates=candidate_list,
        posterior=lagrangian_posterior,
    )

    # Persist observability row for the operator-facing posterior.
    append_paired_comparison_row(verdict)

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
Phase 2 ABLATION is paired-comparison-research, NOT score-mutation.
Every row carries canonical Provenance with ``score_claim=False`` +
``promotable=False`` + ``axis_tag="[predicted]"`` + Tier A non-promotable
markers per Catalog #341 sister discipline.

Cross-references:
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "Mission alignment" Consequence 4 (frontier-breaking dominates)
- Catalog #355 META-LAGRANGIAN-WIRE-1 Phase 1 (the invocation point this
  package extends; Phase 2 cannot remove the Phase 1 wire-in)
- Catalog #131 fcntl-locked JSONL bare-write discipline
- Catalog #138 strict-load discipline
- Catalog #245 canonical 4-layer ledger pattern this posterior follows
- Catalog #287 placeholder-rationale rejection (NO ``<rationale>`` literals)
- Catalog #306 paired-comparison empirical discipline
- Catalog #323 canonical Provenance umbrella
- Catalog #341 cathedral consumer non-promotable canonical markers
- Phase 1 landing memo: ``feedback_slot_meta_lagrangian_wire_1_phase_1_canonical_invocation_landed_20260520.md``
- Master design memo: ``.omx/research/cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md`` Dimension 1
"""
from __future__ import annotations

from tac.findings_lagrangian.phase_2_ablation.ablation_framework import (
    AblationMode,
    AdjusterAblationContext,
    AdjusterAblationVerdict,
    DEFAULT_ABLATION_MODE,
    DEFAULT_DIVERGENCE_SIGMA_BOUND,
    PHASE_2_ABLATION_POSTERIOR_PATH,
    PHASE_2_ABLATION_POSTERIOR_LOCK_PATH,
    PHASE_2_ABLATION_SCHEMA,
    PROMOTION_THRESHOLD_MAX_REGRESSION_USD,
    PROMOTION_THRESHOLD_MIN_ANCHORS,
    SUPPORTED_ADJUSTERS,
    AblationError,
    append_paired_comparison_row,
    compute_solver_dual_variable_for_adjuster,
    load_paired_comparison_rows_lenient,
    load_paired_comparison_rows_strict,
    paired_comparison_against_hand_derived,
)


__all__ = [
    "AblationMode",
    "AdjusterAblationContext",
    "AdjusterAblationVerdict",
    "AblationError",
    "DEFAULT_ABLATION_MODE",
    "DEFAULT_DIVERGENCE_SIGMA_BOUND",
    "PHASE_2_ABLATION_POSTERIOR_PATH",
    "PHASE_2_ABLATION_POSTERIOR_LOCK_PATH",
    "PHASE_2_ABLATION_SCHEMA",
    "PROMOTION_THRESHOLD_MAX_REGRESSION_USD",
    "PROMOTION_THRESHOLD_MIN_ANCHORS",
    "SUPPORTED_ADJUSTERS",
    "append_paired_comparison_row",
    "compute_solver_dual_variable_for_adjuster",
    "load_paired_comparison_rows_lenient",
    "load_paired_comparison_rows_strict",
    "paired_comparison_against_hand_derived",
]
