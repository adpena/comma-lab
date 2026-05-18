# SPDX-License-Identifier: MIT
"""Per-pair master gradient wire-in shim for ``tac.training_curriculum``.

Training is where per-pair score geometry first becomes actionable: curriculum
stages, pause/swap transitions, distillation schedules, and quantization
fine-tunes can all bias scarce training attention toward pair-specific bytes
instead of treating the archive as an aggregate blob.

This module is deliberately a thin namespace shim around the canonical helper
at ``tac.optimization.per_pair_namespace_wire_in.
compose_namespace_per_pair_wire_in`` with
``namespace_id="training_curriculum"`` baked in. It does not load scorers,
dispatch jobs, claim scores, or mutate training artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping

from tac.optimization.per_pair_namespace_wire_in import (
    NamespacePerPairWireInOutcome,
    compose_namespace_per_pair_wire_in,
)

__all__ = [
    "TrainingCurriculumPerPairWireInOutcome",
    "compose_training_curriculum_per_pair_wire_in",
]


TrainingCurriculumPerPairWireInOutcome = NamespacePerPairWireInOutcome


def compose_training_curriculum_per_pair_wire_in(
    *,
    archive_sha256: str,
    total_bit_budget: int,
    sensitivity_reweight: Mapping[int, float] | None = None,
    auto_load: bool = True,
) -> NamespacePerPairWireInOutcome:
    """Compose Hook 1 + Hook 3 wire-in for ``tac.training_curriculum``.

    The returned envelope is intended for trainers to thread into curriculum
    stage selection, pause-to-swap-loss triggers, distillation focus, or
    quantization fine-tune budgets. It remains a planning artifact tagged
    ``[predicted; namespace per-pair wire-in v1]`` and carries no promotion
    authority.
    """
    return compose_namespace_per_pair_wire_in(
        namespace_id="training_curriculum",
        archive_sha256=archive_sha256,
        total_bit_budget=total_bit_budget,
        sensitivity_reweight=sensitivity_reweight,
        auto_load=auto_load,
    )
