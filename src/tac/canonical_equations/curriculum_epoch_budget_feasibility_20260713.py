# SPDX-License-Identifier: MIT
"""Canonical compile-time law for curriculum epoch-budget feasibility.

For an enabled curriculum with run budget ``E`` and active/effective stage
starts ``S_active``, the configuration is boot-runnable only when

    m_sched = E - max(S_active) >= 0.

Curriculum disabled is an explicit vacuous pass, not a measurement.  The law's
verdict scope is config/boot-runnability only; it says nothing about optimizer
quality, evaluator score, archive bytes, or promotion.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_PARAMETER_REFIT,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "curriculum_epoch_budget_feasibility_v1"
_MODULE = "tac.canonical_equations.curriculum_epoch_budget_feasibility_20260713"
_MEMO = ".omx/research/timer_curriculum_complete_20260713.md"


@dataclass(frozen=True)
class ScheduleBudgetResult:
    curriculum_enabled: bool
    epochs: int
    max_active_start_epoch: int | None
    margin_epochs: int | None
    feasible: bool
    status: str
    verdict_scope: str = "config/boot-runnability-only"


def curriculum_epoch_budget_feasibility(
    epochs: int,
    active_start_epochs: Iterable[int],
    *,
    curriculum_enabled: bool,
) -> ScheduleBudgetResult:
    """Evaluate ``m_sched = E - max(S_active)`` with fail-closed inputs."""

    budget = int(epochs)
    if budget < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")
    starts = tuple(int(value) for value in active_start_epochs)
    if any(value < 0 for value in starts):
        raise ValueError(f"active start epochs must be >= 0, got {starts!r}")
    if not curriculum_enabled:
        return ScheduleBudgetResult(
            curriculum_enabled=False,
            epochs=budget,
            max_active_start_epoch=max(starts) if starts else None,
            margin_epochs=None,
            feasible=True,
            status="VACUOUS_CURRICULUM_DISABLED",
        )
    maximum = max(starts, default=0)
    margin = budget - maximum
    return ScheduleBudgetResult(
        curriculum_enabled=True,
        epochs=budget,
        max_active_start_epoch=maximum,
        margin_epochs=margin,
        feasible=margin >= 0,
        status="PASS" if margin >= 0 else "REFUSE_OUT_OF_BUDGET_STAGE",
    )


def build_curriculum_epoch_budget_feasibility_v1() -> CanonicalEquation:
    """Build the machine-readable triality equation entity."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "Re-run the compile audit whenever trainer stage-start argparse/defaults "
            "or named-config schedule ownership changes."
        ),
        measurement_axis="[source-inspection/config-compile]",
        hardware_substrate="hardware-independent",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Curriculum epoch-budget feasibility margin",
        one_line_summary=(
            "An enabled curriculum is boot-feasible iff E-max(S_active)>=0; disabled "
            "curriculum is an explicit vacuous pass."
        ),
        latex_form=(
            r"m_{sched}=E-\max S_{active},\quad "
            r"\mathrm{feasible}\iff m_{sched}\ge 0"
        ),
        python_callable_module_path=f"{_MODULE}:curriculum_epoch_budget_feasibility",
        domain_of_validity={
            "included": ["witness trainer configs with an epoch budget and stage-start caps"],
            "excluded": ["training-quality, score, archive-byte, or promotion verdicts"],
            "verdict_scope": "config/boot-runnability-only",
            "req_R": (
                "trainer stage-start argparse/default change or named-config schedule ownership change"
            ),
        },
        units_in={"epochs": "epochs", "active_start_epochs": "epochs"},
        units_out={"margin_epochs": "epochs", "feasible": "bool"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-14T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_PARAMETER_REFIT,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl.schedule_epoch_budget_violations",
            "tools.launch_witness_run.derive_named_config",
        ),
        canonical_producers=("tac.witness_dsl.curriculum_dsl.WitnessProgram.validate",),
        provenance=provenance,
    )


def populate_curriculum_epoch_budget_feasibility_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append the law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_curriculum_epoch_budget_feasibility_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="timer-curriculum-complete; compile-time schedule feasibility; MEANS-only",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "ScheduleBudgetResult",
    "build_curriculum_epoch_budget_feasibility_v1",
    "curriculum_epoch_budget_feasibility",
    "populate_curriculum_epoch_budget_feasibility_v1",
]
