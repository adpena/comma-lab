# SPDX-License-Identifier: MIT
"""Cathedral autopilot extension helpers (Wave N+46 META-orchestrator extension).

Per operator binding correction 2026-05-28 ~23:55Z verbatim: *"isn't the
meta orchestrator ou descirbed what the cathedral autopilot was supopsed
to be? proceed with all 7"*.

The cathedral autopilot at ``tools/cathedral_autopilot_autonomous_loop.py``
IS the canonical META-orchestrator (Catalog #335 + #341 + #344 + #354 +
#355 + #356 + #357 + #371 + #372 surface). This package EXTENDS that
canonical surface per 4 missing gaps + 5 anti-patterns + 5 equations
from the operator triple-message + correction sequence.

THIS IS NOT a parallel package. The canonical META-orchestrator IS the
cathedral autopilot; this package exposes helpers it consumes.

4 missing gaps closed:

  1. **3-metric trichotomy ranking** — :func:`rank_candidates_via_three_metric_trichotomy`
     per :mod:`tac.cathedral_autopilot.three_metric_trichotomy`.
  2. **Operator-correction META-pattern formalization** —
     :func:`register_operator_binding_correction` per
     :mod:`tac.cathedral_autopilot.operator_correction_meta_pattern`.
  3. **Per-turn main-thread spawn-decision helper** —
     :func:`select_canonical_next_spawn_for_main_thread` per
     :mod:`tac.cathedral_autopilot.per_turn_spawn_decision`.
  4. **No-ad-hoc/signal-loss/rediscovery/duplicate/drift invariants** —
     :func:`validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants`
     per :mod:`tac.cathedral_autopilot.canonical_invariants`.

Cross-references:

- ``feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md``
- ``feedback_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_canonicalize_and_harden_for_automation_standing_directive_20260528.md``
- ``feedback_canonical_ev_metric_trichotomy_hygiene_vs_frontier_vs_highest_ev_shortest_wall_clock_20260528.md``
- ``feedback_prioritization_metric_hygiene_vs_frontier_breaking_orthogonal_plus_13_lessons_incomplete_20260528.md``
- ``feedback_memos_must_be_acted_upon_canonical_apparatus_mutation_enforcement_standing_directive_20260528.md``
- Catalog #335 (canonical cathedral consumer auto-discovery)
- Catalog #341 (Tier A canonical-routing markers)
- Catalog #344 (canonical equations + anti-patterns registry)
- Catalog #354 (master-gradient exploit consumer bundle)
- Catalog #355 (META-LAGRANGIAN-WIRE-1 Phase 1 invoker callsite)
- Catalog #372 (DYKSTRA-PARETO-SOLVER WIRE-IN invoker callsite)
- Catalog #379 (this package's STRICT preflight gate — sister of
  #336/#337/#355/#372 invoker-callsite pattern)
"""
from __future__ import annotations

from tac.cathedral_autopilot.three_metric_trichotomy import (
    CandidateWithThreeMetric,
    ThreeMetricTrichotomyRankingResult,
    rank_candidates_via_three_metric_trichotomy,
)
from tac.cathedral_autopilot.operator_correction_meta_pattern import (
    OperatorBindingCorrectionRegistration,
    register_operator_binding_correction,
)
from tac.cathedral_autopilot.per_turn_spawn_decision import (
    CandidateSelection,
    select_canonical_next_spawn_for_main_thread,
)
from tac.cathedral_autopilot.canonical_invariants import (
    InvariantValidationVerdict,
    InvariantValidationStatus,
    validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants,
)


__all__ = [
    "CandidateSelection",
    "CandidateWithThreeMetric",
    "InvariantValidationStatus",
    "InvariantValidationVerdict",
    "OperatorBindingCorrectionRegistration",
    "ThreeMetricTrichotomyRankingResult",
    "rank_candidates_via_three_metric_trichotomy",
    "register_operator_binding_correction",
    "select_canonical_next_spawn_for_main_thread",
    "validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants",
]
