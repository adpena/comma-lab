# SPDX-License-Identifier: MIT
"""Canonical solver-surface wire-in for Cable D consumers 7-14 (Slot HH).

Per Catalog #125 6-hook wire-in non-negotiable + Slot FF cathedral autopilot
cascade wire-in landing memo (commit `d7c28c737`) highest-EV op-routable:
*"Wire the 6 Cable D consumer sidecars BACK into canonical sensitivity_map +
pareto + bit_allocator solver surfaces per the producer headers' 'sister
subagent owns wiring' declarations."*

The CONSUMER side is now landed at the ranker (FF); this package lands the
SOLVER side, closing the canonical producer → sidecar → ranker (FF) → solver
(HH) loop per Catalog #125 hooks #1 (sensitivity-map) + #2 (Pareto constraint)
+ #3 (bit-allocator).

Per Catalog #335 + tac.cathedral.consumer_contract: a canonical contract
exposes the wire-in surface so future sister subagents can extend WITHOUT
touching the canonical solver-surface modules (sensitivity_map / pareto /
bit_allocator). This package is the canonical entry point for that extension.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323/#341:
every solver-surface contribution is OBSERVABILITY-ONLY:
- `score_claim_valid=False`
- `promotion_eligible=False`
- canonical `[predicted]` axis tag
- predicted_delta_adjustment=0.0 (does NOT mutate scores; observability only)

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" the contributions
are derived from canonical-sidecar SCHEMA-validated presence (per FF helper)
NOT from empirical anchors. Per Catalog #318 raw-byte master-gradient guard:
the contributions NEVER expose raw archive-byte tensors — only typed
per-pair/per-axis structural-signal markers.

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305: every
contribution is inspectable per layer (per-consumer × per-hook accessor) +
decomposable per signal (per-hook return type) + diff-able across runs
(stateless functions of canonical sidecar JSON) + queryable post-hoc (the
SolverHookContribution dataclass exposes all fields) + cite-able (canonical
sidecar path + sha256) + counterfactual-able (removing a sidecar zeros that
consumer × hook contribution).
"""
from __future__ import annotations

from tac.cathedral_solver_wire_in.consumers_7_14_contributions import (
    CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY,
    CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS,
    SOLVER_WIRE_IN_OBSERVABILITY_AXIS,
    SolverHookContribution,
    bit_allocator_contribution_for_consumer,
    collect_all_solver_contributions_for_archive,
    consumer_owns_hook,
    is_solver_contribution_promotable,
    pareto_constraint_contribution_for_consumer,
    sensitivity_map_contribution_for_consumer,
)

__all__ = [
    "CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY",
    "CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS",
    "SOLVER_WIRE_IN_OBSERVABILITY_AXIS",
    "SolverHookContribution",
    "bit_allocator_contribution_for_consumer",
    "collect_all_solver_contributions_for_archive",
    "consumer_owns_hook",
    "is_solver_contribution_promotable",
    "pareto_constraint_contribution_for_consumer",
    "sensitivity_map_contribution_for_consumer",
]
