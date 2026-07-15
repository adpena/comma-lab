# SPDX-License-Identifier: MIT
"""Argv-inert V9 policy binding for the categorical-Fisher ``H^-1`` solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tac.information_geometry.fisher_natural_solver import EQUATION_ID, METRIC_ID
from tac.witness_dsl.lever_registry import resolve_canonical_metric


@dataclass(frozen=True, slots=True)
class FisherNaturalSolverPolicy:
    """Typed, fail-closed solver selection; activation awaits a measured A/B."""

    metric_id: str = METRIC_ID
    equation_id: str = EQUATION_ID
    numpy_solver: str = (
        "tac.information_geometry.fisher_natural_solver:"
        "solve_categorical_fisher_natural_step_numpy_fp32"
    )
    mlx_solver: str = (
        "tac.information_geometry.fisher_natural_solver_mlx:"
        "solve_categorical_fisher_natural_step_mlx"
    )
    delta_convention: str = "delta_kl"
    damping: float = 0.0
    activation: str = "built_not_activated_measurement_owed"
    research_only: bool = True

    def __post_init__(self) -> None:
        descriptor = resolve_canonical_metric(self.metric_id)
        if descriptor.metric_id != METRIC_ID:
            raise ValueError(f"Fisher solver requires canonical metric {METRIC_ID!r}")
        if self.equation_id != EQUATION_ID:
            raise ValueError(f"Fisher solver requires canonical equation {EQUATION_ID!r}")
        if self.delta_convention not in {"delta_kl", "delta_quad"}:
            raise ValueError("delta_convention must be delta_kl or delta_quad")
        if not math.isfinite(float(self.damping)) or float(self.damping) < 0.0:
            raise ValueError("damping must be finite and non-negative")
        if self.activation != "built_not_activated_measurement_owed":
            raise ValueError("solver activation requires a future measured A/B receipt")
        if self.research_only is not True:
            raise ValueError("unmeasured solver policy must remain research_only")

    def flags(self) -> dict[str, object]:
        """Emit no trainer argv; this is a solver-stack policy, not a fake flag."""

        return {}

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "equation_id": self.equation_id,
            "numpy_solver": self.numpy_solver,
            "mlx_solver": self.mlx_solver,
            "delta_convention": self.delta_convention,
            "damping": self.damping,
            "activation": self.activation,
            "research_only": self.research_only,
            "trainer_argv": [],
        }


def canonical_fisher_natural_solver_policy() -> FisherNaturalSolverPolicy:
    return FisherNaturalSolverPolicy()


__all__ = ["FisherNaturalSolverPolicy", "canonical_fisher_natural_solver_policy"]
