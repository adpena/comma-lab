# SPDX-License-Identifier: MIT
"""Per-pair master gradient wire-in shim for ``tac.boosting``.

LOW gap closure widened wave 2026-05-17 — `lane_low_gap_closure_widened_8_modules_
plus_autopilot_wire_in_20260517` BUCKET A. Thin namespace shim around the
canonical helper at ``tac.optimization.per_pair_namespace_wire_in.
compose_namespace_per_pair_wire_in`` with ``namespace_id="boosting"`` baked in.

Per CLAUDE.md "Subagent coherence-by-default" Hook 1 + Hook 3 + Hook 5 mapping
for ``tac.boosting``:
  - Hook 1: stage outputs that ARE byte-axis (e.g. per-pair decoder ensemble
    selector_index_stream) consume the canonical Wyner-Ziv reweight via the
    shared helper.
  - Hook 3: stage byte budgets consume ``allocate_per_pair_bits`` via the
    shared helper.
  - Hook 5: PRESERVED via the canonical sister
    ``tac.boosting.persistence.append_stage_outcome_locked`` (NOT replaced by
    ``tac.continual_learning.posterior_update_locked``).
"""

from __future__ import annotations

from collections.abc import Mapping

from tac.optimization.per_pair_namespace_wire_in import (
    NamespacePerPairWireInOutcome,
    compose_namespace_per_pair_wire_in,
)

__all__ = [
    "BoostingPerPairWireInOutcome",
    "compose_boosting_per_pair_wire_in",
]


# Per CLAUDE.md "Beauty, simplicity, and developer experience": the namespace
# re-exports the canonical typed outcome under its own name for trace clarity.
BoostingPerPairWireInOutcome = NamespacePerPairWireInOutcome


def compose_boosting_per_pair_wire_in(
    *,
    archive_sha256: str,
    total_bit_budget: int,
    sensitivity_reweight: Mapping[int, float] | None = None,
    auto_load: bool = True,
) -> NamespacePerPairWireInOutcome:
    """Compose Hook 1 + Hook 3 wire-in for the ``tac.boosting`` namespace.

    Thin shim around
    :func:`tac.optimization.per_pair_namespace_wire_in.
    compose_namespace_per_pair_wire_in` with ``namespace_id="boosting"`` baked
    in.

    Per CLAUDE.md "Apples-to-apples evidence discipline" the outcome is
    `[predicted; namespace per-pair wire-in v1]` — NO score claim.

    Parameters
    ----------
    archive_sha256
        64-char hex sha of the target archive bytes.
    total_bit_budget
        Hard global byte cap for the per-pair allocation cascade.
    sensitivity_reweight
        Optional pre-resolved per-byte reweight map.
    auto_load
        When True (default), auto-loads missing inputs from canonical anchors.

    Returns
    -------
    NamespacePerPairWireInOutcome with ``namespace_id="boosting"``.
    """
    return compose_namespace_per_pair_wire_in(
        namespace_id="boosting",
        archive_sha256=archive_sha256,
        total_bit_budget=total_bit_budget,
        sensitivity_reweight=sensitivity_reweight,
        auto_load=auto_load,
    )
