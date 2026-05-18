# SPDX-License-Identifier: MIT
"""Per-pair master gradient wire-in shim for ``tac.side_information``.

LOW gap closure widened wave 2026-05-17 — `lane_low_gap_closure_widened_8_modules_
plus_autopilot_wire_in_20260517` BUCKET A. Thin namespace shim around the
canonical helper at ``tac.optimization.per_pair_namespace_wire_in.
compose_namespace_per_pair_wire_in`` with ``namespace_id="side_information"``
baked in.

Per CLAUDE.md "Subagent coherence-by-default" Hook 1 + Hook 3 + Hook 5 mapping
for ``tac.side_information``:
  - Hook 1: side-information channel selectors consume the canonical
    Wyner-Ziv reweight via the shared helper (the SHARED-PRIOR byte class
    IS the canonical side-info channel; Hook 1 reweight surfaces those bytes
    structurally).
  - Hook 3: side-info channel byte budgets consume ``allocate_per_pair_bits``
    via the shared helper (the cascade's Wyner-Ziv composition path is the
    canonical side-info-bit allocation surface).
  - Hook 5: PRESERVED via the canonical sister
    ``tac.side_information.persistence.<sister-append-locked>`` (NOT replaced).
"""

from __future__ import annotations

from collections.abc import Mapping

from tac.optimization.per_pair_namespace_wire_in import (
    NamespacePerPairWireInOutcome,
    compose_namespace_per_pair_wire_in,
)

__all__ = [
    "SideInformationPerPairWireInOutcome",
    "compose_side_information_per_pair_wire_in",
]


SideInformationPerPairWireInOutcome = NamespacePerPairWireInOutcome


def compose_side_information_per_pair_wire_in(
    *,
    archive_sha256: str,
    total_bit_budget: int,
    sensitivity_reweight: Mapping[int, float] | None = None,
    auto_load: bool = True,
) -> NamespacePerPairWireInOutcome:
    """Compose Hook 1 + Hook 3 wire-in for ``tac.side_information``.

    Thin shim around
    :func:`tac.optimization.per_pair_namespace_wire_in.
    compose_namespace_per_pair_wire_in` with
    ``namespace_id="side_information"`` baked in.

    Per CLAUDE.md "Apples-to-apples evidence discipline" the outcome is
    `[predicted; namespace per-pair wire-in v1]` — NO score claim.
    """
    return compose_namespace_per_pair_wire_in(
        namespace_id="side_information",
        archive_sha256=archive_sha256,
        total_bit_budget=total_bit_budget,
        sensitivity_reweight=sensitivity_reweight,
        auto_load=auto_load,
    )
