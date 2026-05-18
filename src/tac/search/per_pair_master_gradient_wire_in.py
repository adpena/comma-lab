# SPDX-License-Identifier: MIT
"""Per-pair master gradient wire-in shim for ``tac.search``.

LOW gap closure widened wave 2026-05-17 — `lane_low_gap_closure_widened_8_modules_
plus_autopilot_wire_in_20260517` BUCKET A. Thin namespace shim around the
canonical helper at ``tac.optimization.per_pair_namespace_wire_in.
compose_namespace_per_pair_wire_in`` with ``namespace_id="search"`` baked in.

Per CLAUDE.md "Subagent coherence-by-default" Hook 1 + Hook 3 + Hook 5 mapping
for ``tac.search``:
  - Hook 1: search-over-treatments callers consume the canonical Wyner-Ziv
    reweight via the shared helper to bias the search distribution per byte
    sensitivity (high-sensitivity bytes get denser sampling).
  - Hook 3: per-treatment byte budgets consume ``allocate_per_pair_bits`` via
    the shared helper (search-cascade's Cascade 1 OptimalPlan path is the
    canonical search-of-treatments answer).
  - Hook 5: PRESERVED via the canonical sister
    ``tac.search.persistence.<sister-append-locked>`` (NOT replaced).
"""

from __future__ import annotations

from collections.abc import Mapping

from tac.optimization.per_pair_namespace_wire_in import (
    NamespacePerPairWireInOutcome,
    compose_namespace_per_pair_wire_in,
)

__all__ = [
    "SearchPerPairWireInOutcome",
    "compose_search_per_pair_wire_in",
]


SearchPerPairWireInOutcome = NamespacePerPairWireInOutcome


def compose_search_per_pair_wire_in(
    *,
    archive_sha256: str,
    total_bit_budget: int,
    sensitivity_reweight: Mapping[int, float] | None = None,
    auto_load: bool = True,
) -> NamespacePerPairWireInOutcome:
    """Compose Hook 1 + Hook 3 wire-in for ``tac.search``.

    Thin shim around
    :func:`tac.optimization.per_pair_namespace_wire_in.
    compose_namespace_per_pair_wire_in` with ``namespace_id="search"`` baked
    in.

    Per CLAUDE.md "Apples-to-apples evidence discipline" the outcome is
    `[predicted; namespace per-pair wire-in v1]` — NO score claim.
    """
    return compose_namespace_per_pair_wire_in(
        namespace_id="search",
        archive_sha256=archive_sha256,
        total_bit_budget=total_bit_budget,
        sensitivity_reweight=sensitivity_reweight,
        auto_load=auto_load,
    )
