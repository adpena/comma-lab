# SPDX-License-Identifier: MIT
"""n600 epochs/update accounting tickets for the 2026-07-13 95-kill audit.

This module deliberately contains no trainer invocation or measurement.  It
turns three source-audited dispositions into typed n600 tickets and refuses to
turn a right-censored crossing or missing timing into a zero saving.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Literal

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_predicted

EQUATION_ID = "steps_dimension_epochs_to_target_v1"
N_PAIRS = 600
FRESH_ABSOLUTE_TARGET_DSEG = 0.040763
TicketStatus = Literal["AB_TICKET_ONLY", "WIRING_NEEDED"]
MeasurementStatus = Literal["UNMEASURED", "MEASURED", "MEASURED_CENSORED"]

_LAW = (
    "E_saved=E_control-E_treatment; "
    "f_step_saved=1-U_treatment/U_control; "
    "wall=U*t_update+E*t_recurring_nonupdate+one_time+terminal_critical_path; "
    "f_wall=elapsed_treatment/elapsed_control or fully_allocated_wall_treatment/fully_allocated_wall_control; "
    "f_wall_saved=1-f_wall"
)
_UTC = "2026-07-13T00:00:00Z"


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = _nonnegative_finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


@dataclass(frozen=True)
class EpochsToTargetTicket:
    """Frozen A/B ticket whose measured fields stay ``None`` until a receipt exists."""

    lever_id: str
    existing_dsl_surface: str
    status: TicketStatus
    start_custody: str
    target_rule: str
    maximum_nominal_epochs: int
    censoring_rule: str
    control_definition: str
    treatment_definition: str
    verdict_scope: str
    reformulation_reactivation_queue: tuple[str, ...]
    source_artifacts: tuple[str, ...]
    speed_configuration_rule: str
    measurement_authority_rule: str
    receipt_custody: tuple[str, ...] = ()
    speed_configuration_custody: str | None = None
    measurement_authority_custody: str | None = None
    wiring_closure_evidence: str | None = None
    n_pairs: int = N_PAIRS
    measurement_status: MeasurementStatus = "UNMEASURED"
    control_epochs_to_target: int | None = None
    treatment_epochs_to_target: int | None = None
    control_optimizer_updates: int | None = None
    treatment_optimizer_updates: int | None = None
    control_solver_hvp_steps: int | None = None
    treatment_solver_hvp_steps: int | None = None
    control_one_time_overhead_seconds: float | None = None
    treatment_one_time_overhead_seconds: float | None = None
    control_seconds_per_update: float | None = None
    treatment_seconds_per_update: float | None = None
    control_recurring_nonupdate_seconds_per_epoch: float | None = None
    treatment_recurring_nonupdate_seconds_per_epoch: float | None = None
    control_terminal_critical_path_seconds: float | None = None
    treatment_terminal_critical_path_seconds: float | None = None
    control_async_service_seconds: float | None = None
    treatment_async_service_seconds: float | None = None
    control_direct_elapsed_seconds_to_crossing: float | None = None
    treatment_direct_elapsed_seconds_to_crossing: float | None = None
    wall_composition_admissible: bool = False
    wall_composition_refusal_reason: str | None = "UNMEASURED: no matched wall receipt"
    authority_axis: str = "[n600 ticket; no execution; non-promotable]"
    score_claim: bool = False
    pointer_moved: bool = False

    def __post_init__(self) -> None:
        if self.n_pairs != N_PAIRS:
            raise ValueError("n_pairs must be exactly 600 for this accounting law")
        if self.status not in {"AB_TICKET_ONLY", "WIRING_NEEDED"}:
            raise ValueError("status must be AB_TICKET_ONLY or WIRING_NEEDED")
        if self.measurement_status not in {"UNMEASURED", "MEASURED", "MEASURED_CENSORED"}:
            raise ValueError("measurement_status must be UNMEASURED, MEASURED, or MEASURED_CENSORED")
        for name in (
            "lever_id", "existing_dsl_surface", "start_custody", "target_rule", "censoring_rule",
            "control_definition", "treatment_definition", "verdict_scope", "speed_configuration_rule",
            "measurement_authority_rule", "authority_axis",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        _positive_int(self.maximum_nominal_epochs, "maximum_nominal_epochs")
        if not self.reformulation_reactivation_queue:
            raise ValueError("reformulation_reactivation_queue must be non-empty")
        if not self.source_artifacts or any(not path.strip() for path in self.source_artifacts):
            raise ValueError("source_artifacts must contain non-empty custody paths")
        if self.score_claim or self.pointer_moved:
            raise ValueError("ticket rows cannot claim score movement or a pointer move")

        measured = (
            self.control_epochs_to_target,
            self.treatment_epochs_to_target,
            self.control_optimizer_updates,
            self.treatment_optimizer_updates,
            self.control_solver_hvp_steps,
            self.treatment_solver_hvp_steps,
            self.control_one_time_overhead_seconds,
            self.treatment_one_time_overhead_seconds,
            self.control_seconds_per_update,
            self.treatment_seconds_per_update,
            self.control_recurring_nonupdate_seconds_per_epoch,
            self.treatment_recurring_nonupdate_seconds_per_epoch,
            self.control_terminal_critical_path_seconds,
            self.treatment_terminal_critical_path_seconds,
            self.control_async_service_seconds,
            self.treatment_async_service_seconds,
            self.control_direct_elapsed_seconds_to_crossing,
            self.treatment_direct_elapsed_seconds_to_crossing,
        )
        if self.measurement_status == "UNMEASURED":
            if any(value is not None for value in measured):
                raise ValueError("UNMEASURED ticket fields must remain None; crossings are right-censored")
            if self.wall_composition_admissible:
                raise ValueError("UNMEASURED ticket cannot admit wall composition")
            if not (isinstance(self.wall_composition_refusal_reason, str) and self.wall_composition_refusal_reason.strip()):
                raise ValueError("UNMEASURED ticket requires a wall-composition refusal reason")
            if (
                self.receipt_custody
                or self.speed_configuration_custody is not None
                or self.measurement_authority_custody is not None
                or self.wiring_closure_evidence is not None
            ):
                raise ValueError("UNMEASURED ticket cannot claim receipt or wiring-closure evidence")
            return
        if not self.receipt_custody or any(not path.strip() for path in self.receipt_custody):
            raise ValueError("receipt-backed ticket requires non-empty receipt_custody paths")
        if not (isinstance(self.speed_configuration_custody, str) and self.speed_configuration_custody.strip()):
            raise ValueError("receipt-backed ticket requires speed_configuration_custody")
        if not (isinstance(self.measurement_authority_custody, str) and self.measurement_authority_custody.strip()):
            raise ValueError("receipt-backed ticket requires measurement_authority_custody")
        if self.status == "WIRING_NEEDED" and not (
            isinstance(self.wiring_closure_evidence, str) and self.wiring_closure_evidence.strip()
        ):
            raise ValueError("WIRING_NEEDED ticket cannot be MEASURED before wiring_closure_evidence")
        non_crossing = (
            self.control_optimizer_updates,
            self.treatment_optimizer_updates,
            self.control_solver_hvp_steps,
            self.treatment_solver_hvp_steps,
            self.control_one_time_overhead_seconds,
            self.treatment_one_time_overhead_seconds,
            self.control_recurring_nonupdate_seconds_per_epoch,
            self.treatment_recurring_nonupdate_seconds_per_epoch,
            self.control_terminal_critical_path_seconds,
            self.treatment_terminal_critical_path_seconds,
            self.control_async_service_seconds,
            self.treatment_async_service_seconds,
        )
        if any(value is None for value in non_crossing):
            raise ValueError("receipt-backed ticket requires update counts and critical-path timings")
        crossings = (self.control_epochs_to_target, self.treatment_epochs_to_target)
        if self.measurement_status == "MEASURED" and any(value is None for value in crossings):
            raise ValueError("MEASURED ticket requires both first crossings; use MEASURED_CENSORED otherwise")
        if self.measurement_status == "MEASURED_CENSORED" and all(value is not None for value in crossings):
            raise ValueError("MEASURED_CENSORED ticket requires at least one missing first crossing")
        if (
            self.control_epochs_to_target is not None
            and _nonnegative_int(self.control_epochs_to_target, "control_epochs_to_target") > self.maximum_nominal_epochs
        ):
            raise ValueError("control_epochs_to_target cannot exceed maximum_nominal_epochs")
        if (
            self.treatment_epochs_to_target is not None
            and _nonnegative_int(self.treatment_epochs_to_target, "treatment_epochs_to_target") > self.maximum_nominal_epochs
        ):
            raise ValueError("treatment_epochs_to_target cannot exceed maximum_nominal_epochs")
        assert self.control_optimizer_updates is not None
        assert self.treatment_optimizer_updates is not None
        assert self.control_solver_hvp_steps is not None
        assert self.treatment_solver_hvp_steps is not None
        assert self.control_one_time_overhead_seconds is not None
        assert self.treatment_one_time_overhead_seconds is not None
        assert self.control_recurring_nonupdate_seconds_per_epoch is not None
        assert self.treatment_recurring_nonupdate_seconds_per_epoch is not None
        assert self.control_terminal_critical_path_seconds is not None
        assert self.treatment_terminal_critical_path_seconds is not None
        assert self.control_async_service_seconds is not None
        assert self.treatment_async_service_seconds is not None
        control_updates = _nonnegative_int(self.control_optimizer_updates, "control_optimizer_updates")
        treatment_updates = _nonnegative_int(self.treatment_optimizer_updates, "treatment_optimizer_updates")
        _nonnegative_int(self.control_solver_hvp_steps, "control_solver_hvp_steps")
        _nonnegative_int(self.treatment_solver_hvp_steps, "treatment_solver_hvp_steps")
        _nonnegative_finite(self.control_one_time_overhead_seconds, "control_one_time_overhead_seconds")
        _nonnegative_finite(self.treatment_one_time_overhead_seconds, "treatment_one_time_overhead_seconds")
        self._validate_update_seconds(control_updates, self.control_seconds_per_update, "control_seconds_per_update")
        self._validate_update_seconds(treatment_updates, self.treatment_seconds_per_update, "treatment_seconds_per_update")
        _nonnegative_finite(self.control_recurring_nonupdate_seconds_per_epoch, "control_recurring_nonupdate_seconds_per_epoch")
        _nonnegative_finite(self.treatment_recurring_nonupdate_seconds_per_epoch, "treatment_recurring_nonupdate_seconds_per_epoch")
        _nonnegative_finite(self.control_terminal_critical_path_seconds, "control_terminal_critical_path_seconds")
        _nonnegative_finite(self.treatment_terminal_critical_path_seconds, "treatment_terminal_critical_path_seconds")
        _nonnegative_finite(self.control_async_service_seconds, "control_async_service_seconds")
        _nonnegative_finite(self.treatment_async_service_seconds, "treatment_async_service_seconds")
        direct_elapsed = (
            self.control_direct_elapsed_seconds_to_crossing,
            self.treatment_direct_elapsed_seconds_to_crossing,
        )
        if any(value is None for value in direct_elapsed) and any(value is not None for value in direct_elapsed):
            raise ValueError("direct elapsed timing must be present for both arms or neither")
        if self.wall_composition_admissible:
            if any(value is not None for value in direct_elapsed):
                _positive_finite(self.control_direct_elapsed_seconds_to_crossing, "control_direct_elapsed_seconds_to_crossing")
                _positive_finite(self.treatment_direct_elapsed_seconds_to_crossing, "treatment_direct_elapsed_seconds_to_crossing")
        elif not (isinstance(self.wall_composition_refusal_reason, str) and self.wall_composition_refusal_reason.strip()):
            raise ValueError("non-composable wall receipt requires a refusal reason")

    @staticmethod
    def _validate_update_seconds(update_count: int, seconds_per_update: float | None, name: str) -> None:
        """Require timing only when updates occurred; zero-update arms never invent one."""

        if update_count == 0:
            if seconds_per_update is not None:
                raise ValueError(f"{name} must be None for a zero-update arm")
            return
        if seconds_per_update is None:
            raise ValueError(f"{name} is required when optimizer updates occurred")
        _positive_finite(seconds_per_update, name)

    def with_measured_receipt(
        self,
        *,
        control_epochs_to_target: int,
        treatment_epochs_to_target: int,
        control_optimizer_updates: int,
        treatment_optimizer_updates: int,
        control_solver_hvp_steps: int,
        treatment_solver_hvp_steps: int,
        control_one_time_overhead_seconds: float,
        treatment_one_time_overhead_seconds: float,
        control_seconds_per_update: float | None,
        treatment_seconds_per_update: float | None,
        control_recurring_nonupdate_seconds_per_epoch: float,
        treatment_recurring_nonupdate_seconds_per_epoch: float,
        control_terminal_critical_path_seconds: float,
        treatment_terminal_critical_path_seconds: float,
        control_async_service_seconds: float,
        treatment_async_service_seconds: float,
        control_direct_elapsed_seconds_to_crossing: float | None = None,
        treatment_direct_elapsed_seconds_to_crossing: float | None = None,
        wall_composition_admissible: bool = False,
        wall_composition_refusal_reason: str | None = "receipt composition has not been explicitly admitted",
        receipt_custody: tuple[str, ...],
        speed_configuration_custody: str,
        measurement_authority_custody: str,
        wiring_closure_evidence: str | None = None,
    ) -> EpochsToTargetTicket:
        """Return a receipt-backed copy; missing/censored values fail closed at construction."""

        return replace(
            self,
            measurement_status="MEASURED",
            control_epochs_to_target=control_epochs_to_target,
            treatment_epochs_to_target=treatment_epochs_to_target,
            control_optimizer_updates=control_optimizer_updates,
            treatment_optimizer_updates=treatment_optimizer_updates,
            control_solver_hvp_steps=control_solver_hvp_steps,
            treatment_solver_hvp_steps=treatment_solver_hvp_steps,
            control_one_time_overhead_seconds=control_one_time_overhead_seconds,
            treatment_one_time_overhead_seconds=treatment_one_time_overhead_seconds,
            control_seconds_per_update=control_seconds_per_update,
            treatment_seconds_per_update=treatment_seconds_per_update,
            control_recurring_nonupdate_seconds_per_epoch=control_recurring_nonupdate_seconds_per_epoch,
            treatment_recurring_nonupdate_seconds_per_epoch=treatment_recurring_nonupdate_seconds_per_epoch,
            control_terminal_critical_path_seconds=control_terminal_critical_path_seconds,
            treatment_terminal_critical_path_seconds=treatment_terminal_critical_path_seconds,
            control_async_service_seconds=control_async_service_seconds,
            treatment_async_service_seconds=treatment_async_service_seconds,
            control_direct_elapsed_seconds_to_crossing=control_direct_elapsed_seconds_to_crossing,
            treatment_direct_elapsed_seconds_to_crossing=treatment_direct_elapsed_seconds_to_crossing,
            wall_composition_admissible=wall_composition_admissible,
            wall_composition_refusal_reason=wall_composition_refusal_reason,
            receipt_custody=receipt_custody,
            speed_configuration_custody=speed_configuration_custody,
            measurement_authority_custody=measurement_authority_custody,
            wiring_closure_evidence=wiring_closure_evidence,
        )

    def with_censored_receipt(
        self,
        *,
        control_epochs_to_target: int | None,
        treatment_epochs_to_target: int | None,
        control_optimizer_updates: int,
        treatment_optimizer_updates: int,
        control_solver_hvp_steps: int,
        treatment_solver_hvp_steps: int,
        control_one_time_overhead_seconds: float,
        treatment_one_time_overhead_seconds: float,
        control_seconds_per_update: float | None,
        treatment_seconds_per_update: float | None,
        control_recurring_nonupdate_seconds_per_epoch: float,
        treatment_recurring_nonupdate_seconds_per_epoch: float,
        control_terminal_critical_path_seconds: float,
        treatment_terminal_critical_path_seconds: float,
        control_async_service_seconds: float,
        treatment_async_service_seconds: float,
        control_direct_elapsed_seconds_to_crossing: float | None = None,
        treatment_direct_elapsed_seconds_to_crossing: float | None = None,
        wall_composition_admissible: bool = False,
        wall_composition_refusal_reason: str | None = "right-censored receipt has no admitted wall composition",
        receipt_custody: tuple[str, ...],
        speed_configuration_custody: str,
        measurement_authority_custody: str,
        wiring_closure_evidence: str | None = None,
    ) -> EpochsToTargetTicket:
        """Record a complete timed window while preserving an absent first crossing as ``None``."""

        return replace(
            self,
            measurement_status="MEASURED_CENSORED",
            control_epochs_to_target=control_epochs_to_target,
            treatment_epochs_to_target=treatment_epochs_to_target,
            control_optimizer_updates=control_optimizer_updates,
            treatment_optimizer_updates=treatment_optimizer_updates,
            control_solver_hvp_steps=control_solver_hvp_steps,
            treatment_solver_hvp_steps=treatment_solver_hvp_steps,
            control_one_time_overhead_seconds=control_one_time_overhead_seconds,
            treatment_one_time_overhead_seconds=treatment_one_time_overhead_seconds,
            control_seconds_per_update=control_seconds_per_update,
            treatment_seconds_per_update=treatment_seconds_per_update,
            control_recurring_nonupdate_seconds_per_epoch=control_recurring_nonupdate_seconds_per_epoch,
            treatment_recurring_nonupdate_seconds_per_epoch=treatment_recurring_nonupdate_seconds_per_epoch,
            control_terminal_critical_path_seconds=control_terminal_critical_path_seconds,
            treatment_terminal_critical_path_seconds=treatment_terminal_critical_path_seconds,
            control_async_service_seconds=control_async_service_seconds,
            treatment_async_service_seconds=treatment_async_service_seconds,
            control_direct_elapsed_seconds_to_crossing=control_direct_elapsed_seconds_to_crossing,
            treatment_direct_elapsed_seconds_to_crossing=treatment_direct_elapsed_seconds_to_crossing,
            wall_composition_admissible=wall_composition_admissible,
            wall_composition_refusal_reason=wall_composition_refusal_reason,
            receipt_custody=receipt_custody,
            speed_configuration_custody=speed_configuration_custody,
            measurement_authority_custody=measurement_authority_custody,
            wiring_closure_evidence=wiring_closure_evidence,
        )


def epochs_saved(control_epochs: int, treatment_epochs: int) -> int:
    """Return the nominal-epoch difference; epoch-zero crossings are valid."""

    return _nonnegative_int(control_epochs, "control_epochs") - _nonnegative_int(treatment_epochs, "treatment_epochs")


def step_fraction_saved(control_updates: int, treatment_updates: int) -> float | None:
    """Return update saving or ``None`` when a zero-update control has no denominator."""

    control = _nonnegative_int(control_updates, "control_updates")
    treatment = _nonnegative_int(treatment_updates, "treatment_updates")
    if control == 0:
        return None
    return 1.0 - treatment / control


def fixed_quality_threshold_factor(*, control_epoch0_d_seg: float, absolute_target_d_seg: float) -> float:
    """Derive the existing harness's factor from a deterministic epoch-zero verdict.

    The fixed-quality receipt accepts ``threshold_factor`` rather than an
    absolute CLI threshold.  This conversion is admissible only when both
    values are measured and the target is strictly below the control's epoch-0
    value, yielding the harness-required factor in ``(0, 1)``.
    """

    control = _positive_finite(control_epoch0_d_seg, "control_epoch0_d_seg")
    target = _positive_finite(absolute_target_d_seg, "absolute_target_d_seg")
    factor = target / control
    if not 0.0 < factor < 1.0:
        raise ValueError("absolute target must yield fixed-quality threshold_factor strictly in (0, 1)")
    return factor


def wall_seconds(
    optimizer_updates: int,
    nominal_epochs: int,
    seconds_per_update: float | None,
    recurring_nonupdate_seconds_per_epoch: float,
    one_time_overhead_seconds: float,
    terminal_critical_path_seconds: float,
) -> float:
    """Return allocated critical-path seconds without laundering residual time into updates."""

    updates = _nonnegative_int(optimizer_updates, "optimizer_updates")
    epochs = _nonnegative_int(nominal_epochs, "nominal_epochs")
    if updates == 0:
        if seconds_per_update is not None:
            raise ValueError("seconds_per_update must be None for a zero-update arm")
        update_seconds = 0.0
    else:
        if seconds_per_update is None:
            raise ValueError("seconds_per_update is required when optimizer updates occurred")
        update_seconds = _positive_finite(seconds_per_update, "seconds_per_update")
    recurring_seconds = _nonnegative_finite(recurring_nonupdate_seconds_per_epoch, "recurring_nonupdate_seconds_per_epoch")
    overhead = _nonnegative_finite(one_time_overhead_seconds, "one_time_overhead_seconds")
    terminal = _nonnegative_finite(terminal_critical_path_seconds, "terminal_critical_path_seconds")
    return updates * update_seconds + epochs * recurring_seconds + overhead + terminal


def wall_fraction(
    *,
    composition_admissible: bool,
    composition_refusal_reason: str | None,
    control_direct_elapsed_seconds_to_crossing: float | None,
    treatment_direct_elapsed_seconds_to_crossing: float | None,
    control_updates: int,
    treatment_updates: int,
    control_epochs: int,
    treatment_epochs: int,
    control_seconds_per_update: float | None,
    treatment_seconds_per_update: float | None,
    control_recurring_nonupdate_seconds_per_epoch: float,
    treatment_recurring_nonupdate_seconds_per_epoch: float,
    control_one_time_overhead_seconds: float,
    treatment_one_time_overhead_seconds: float,
    control_terminal_critical_path_seconds: float,
    treatment_terminal_critical_path_seconds: float,
) -> float | None:
    """Return direct elapsed ratio, or a fully allocated critical-path ratio when admitted.

    Async service seconds are intentionally not an argument: they are recorded
    in the ticket but excluded unless a measured wait is allocated as a
    critical-path term. A refusal is data, returning ``None`` rather than a
    fabricated multiplier.
    """

    if not composition_admissible:
        if not (isinstance(composition_refusal_reason, str) and composition_refusal_reason.strip()):
            raise ValueError("non-composable wall input requires a refusal reason")
        return None
    # Even a direct elapsed receipt must prove that its count/cost fields are
    # physically admissible; direct timing cannot mask negative accounting.
    control_seconds = wall_seconds(
        control_updates,
        control_epochs,
        control_seconds_per_update,
        control_recurring_nonupdate_seconds_per_epoch,
        control_one_time_overhead_seconds,
        control_terminal_critical_path_seconds,
    )
    treatment_seconds = wall_seconds(
        treatment_updates,
        treatment_epochs,
        treatment_seconds_per_update,
        treatment_recurring_nonupdate_seconds_per_epoch,
        treatment_one_time_overhead_seconds,
        treatment_terminal_critical_path_seconds,
    )
    direct = (control_direct_elapsed_seconds_to_crossing, treatment_direct_elapsed_seconds_to_crossing)
    if any(value is None for value in direct) and any(value is not None for value in direct):
        raise ValueError("direct elapsed timing must be present for both arms or neither")
    if all(value is not None for value in direct):
        assert control_direct_elapsed_seconds_to_crossing is not None
        assert treatment_direct_elapsed_seconds_to_crossing is not None
        return _positive_finite(
            treatment_direct_elapsed_seconds_to_crossing, "treatment_direct_elapsed_seconds_to_crossing"
        ) / _positive_finite(control_direct_elapsed_seconds_to_crossing, "control_direct_elapsed_seconds_to_crossing")
    if control_seconds <= 0.0:
        raise ValueError("fallback wall fraction requires positive control total wall")
    return treatment_seconds / control_seconds


def wall_fraction_saved(**kwargs: object) -> float | None:
    """Return ``1 - wall_fraction`` with the same fail-closed input checks."""

    fraction = wall_fraction(**kwargs)  # type: ignore[arg-type]
    return None if fraction is None else 1.0 - fraction


def measured_accounting(ticket: EpochsToTargetTicket) -> dict[str, float | int | None]:
    """Calculate all accounting outputs only for a fully receipt-backed n600 ticket."""

    if ticket.measurement_status != "MEASURED":
        raise ValueError("UNMEASURED or MEASURED_CENSORED ticket has no accounting delta")
    assert ticket.control_epochs_to_target is not None
    assert ticket.treatment_epochs_to_target is not None
    assert ticket.control_optimizer_updates is not None
    assert ticket.treatment_optimizer_updates is not None
    assert ticket.control_solver_hvp_steps is not None
    assert ticket.treatment_solver_hvp_steps is not None
    assert ticket.control_one_time_overhead_seconds is not None
    assert ticket.treatment_one_time_overhead_seconds is not None
    args = {
        "composition_admissible": ticket.wall_composition_admissible,
        "composition_refusal_reason": ticket.wall_composition_refusal_reason,
        "control_direct_elapsed_seconds_to_crossing": ticket.control_direct_elapsed_seconds_to_crossing,
        "treatment_direct_elapsed_seconds_to_crossing": ticket.treatment_direct_elapsed_seconds_to_crossing,
        "control_updates": ticket.control_optimizer_updates,
        "treatment_updates": ticket.treatment_optimizer_updates,
        "control_epochs": ticket.control_epochs_to_target,
        "treatment_epochs": ticket.treatment_epochs_to_target,
        "control_seconds_per_update": ticket.control_seconds_per_update,
        "treatment_seconds_per_update": ticket.treatment_seconds_per_update,
        "control_recurring_nonupdate_seconds_per_epoch": ticket.control_recurring_nonupdate_seconds_per_epoch,
        "treatment_recurring_nonupdate_seconds_per_epoch": ticket.treatment_recurring_nonupdate_seconds_per_epoch,
        "control_one_time_overhead_seconds": ticket.control_one_time_overhead_seconds,
        "treatment_one_time_overhead_seconds": ticket.treatment_one_time_overhead_seconds,
        "control_terminal_critical_path_seconds": ticket.control_terminal_critical_path_seconds,
        "treatment_terminal_critical_path_seconds": ticket.treatment_terminal_critical_path_seconds,
    }
    fraction = wall_fraction(**args)
    return {
        "epochs_saved": epochs_saved(ticket.control_epochs_to_target, ticket.treatment_epochs_to_target),
        "step_fraction_saved": step_fraction_saved(ticket.control_optimizer_updates, ticket.treatment_optimizer_updates),
        "control_solver_hvp_steps": ticket.control_solver_hvp_steps,
        "treatment_solver_hvp_steps": ticket.treatment_solver_hvp_steps,
        "wall_fraction": fraction,
        "wall_fraction_saved": None if fraction is None else 1.0 - fraction,
    }


def symbolic_independent_step_fraction_saved(step_fractions_saved: tuple[float, ...]) -> dict[str, object]:
    """Label an independent product as an assumption, never a measured composition."""

    remaining = 1.0
    for fraction_saved in step_fractions_saved:
        value = float(fraction_saved)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("each step fraction saved must be finite and in [0, 1]")
        remaining *= 1.0 - value
    return {
        "composition_status": "ASSUMED_INDEPENDENT_SYMBOLIC_SCENARIO",
        "empirical_verification_status": ASSUMED_AWAITING_VERIFICATION,
        "step_fraction_saved": 1.0 - remaining,
    }


FRESH_TICKET = EpochsToTargetTicket(
    lever_id="fresh_frequency_shift_init",
    existing_dsl_surface="tac.witness_dsl.curriculum_dsl.FreShInitControl + FreshFrequencyShift + FreShFixedQualitySlice",
    status="AB_TICKET_ONLY",
    start_custody=(
        "cold seed/config only for the initialization A/B; a non-FreSh checkpoint is inadmissible, "
        "while same-arm FreSh checkpoints may continue bit-faithfully with selected frequency/bias and state restored"
    ),
    target_rule=(
        "first emitted d_seg <= 0.040763; no interpolation; MEASURED advisory n600 epoch-50 reference only; "
        "derive existing fixed-quality harness threshold_factor=0.040763/control_epoch0_d_seg after deterministic epoch-0 verdict; "
        "source custody: coherent run.log SHA-256 3860bcf20a341f562e1dd402e281a3298a347f60fa94928cb592ee5dcee480e8, "
        "launch.sh SHA-256 bd760505c445d51dc51d0b31eadd5a4d2628261220ffa46e2474ca83f358c601, "
        "ep251 checkpoint SHA-256 c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758"
    ),
    maximum_nominal_epochs=50,  # ASSUMED ticket ceiling anchored to that measured epoch-50 reference.
    censoring_rule="right-censor at epoch 50; missing crossing remains None, never zero",
    control_definition="existing FreShInitControl plus FreShFixedQualitySlice(eval_every=1, ckpt_every=1)",
    treatment_definition="existing FreshFrequencyShift plus the same FreShFixedQualitySlice",
    verdict_scope="cold-init FreSh A/B readiness only; no n600 matched result, wall claim, score claim, or pointer move",
    reformulation_reactivation_queue=(
        "governed cold n600 matched A/B with receipt-custodied candidate-sweep and training overhead",
        "reformulate only after a right-censored or adverse cold-init receipt",
    ),
    source_artifacts=(
        "src/tac/witness_dsl/curriculum_dsl.py",
        "experiments/train_levelset_witness_realized_through_R_mlx.py",
        "src/tac/witness_init/fixed_quality.py",
        "src/tac/witness_init/fresh_trainer_contract.py",
        "tools/measure_witness_fixed_quality.py",
        "experiments/results/fresh_init_n8_fixed_quality_20260712/measurement_blocker.json",
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/run.log",
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/launch.sh",
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_ckpt_stageOctave1_ep251.npz",
        ".omx/research/steps_dimension_95kill_20260713_SPEC.md",
    ),
    speed_configuration_rule=(
        "both arms require identical exact machine-readable speed configuration with every currently admitted/requested "
        "neutral fleet speed lever ON and all_requested_speed_levers_on=true; otherwise receipt is a blocker"
    ),
    measurement_authority_rule=(
        "future crossing receipt requires deterministic NumPy-fp32 realization through actual R plus the frozen CPU-torch "
        "scorer on all 600 states; MLX training is advisory and has no score authority"
    ),
    authority_axis="[macOS-CPU advisory verdict from macOS-MLX training; NON-PROMOTABLE]",
)

HARDNESS_TICKET = EpochsToTargetTicket(
    lever_id="hardness_oversample_lever5",
    existing_dsl_surface="tac.witness_dsl.curriculum_dsl.HardnessOversample",
    status="WIRING_NEEDED",
    start_custody="stage-Octave1 epoch-251 weights may seed a separately labelled weights-only re-treatment; not bit-faithful resume",
    target_rule="first emitted d_seg <= 0.040915; no interpolation; MEASURED advisory n600 epoch-275 reference only",
    maximum_nominal_epochs=25,  # ASSUMED bounded-ticket ceiling.
    censoring_rule="right-censor at the 25 nominal-epoch window; report exact updates; missing crossing remains None",
    control_definition=(
        "oversample=0.5 (existing DSL default / ASSUMED policy, not measured optimum), source=realized, "
        "weighted=False uniform extras, same seed and exact update count"
    ),
    treatment_definition=(
        "oversample=0.5 (existing DSL default / ASSUMED policy, not measured optimum), source=realized, "
        "weighted=True hardness extras, same seed and exact update count"
    ),
    verdict_scope=(
        "current additive/full-base-coverage HardnessOversample semantics only: order has P+n_extra but "
        "the loop consumes P shuffled draws, so it can omit base pairs and cannot support promised extra-update accounting"
    ),
    reformulation_reactivation_queue=(
        "trainer owner consumes all len(order) visits and asserts P base visits plus exactly round(P*oversample) extras",
        "preserve hardness RNG/resume state then run equal-update n600 weighted-vs-uniform A/B",
        "separately pre-register and measure a fixed-budget replacement-resampling formulation if desired",
    ),
    source_artifacts=(
        "src/tac/witness_dsl/curriculum_dsl.py",
        "experiments/train_levelset_witness_realized_through_R_mlx.py",
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/run.log",
        "experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_ckpt_stageOctave1_ep251.npz",
        "experiments/results/cheapen_real95_tilehalo_fp16_20260713/current_wall_receipt.json",
        ".omx/research/steps_dimension_95kill_20260713_SPEC.md",
    ),
    speed_configuration_rule=(
        "both arms require identical exact machine-readable speed configuration with every currently admitted/requested "
        "neutral fleet speed lever ON and all_requested_speed_levers_on=true; otherwise receipt is a blocker"
    ),
    measurement_authority_rule=(
        "future crossing receipt requires deterministic NumPy-fp32 realization through actual R plus the frozen CPU-torch "
        "scorer on all 600 states; MLX training is advisory and has no score authority"
    ),
)

TERMINAL_SOLVE_TICKET = EpochsToTargetTicket(
    lever_id="terminal_solve_full_p",
    existing_dsl_surface="tac.witness_dsl.curriculum_dsl.TerminalSolve (ScheduleDisplay; flags() returns no argv)",
    status="WIRING_NEEDED",
    start_custody=(
        "frozen A/B premise start: experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/"
        "levelset_witness_ema_BEST.npz SHA-256 6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca; "
        "#341 premise record: reports/basin_finisher_probe_20260707.json "
        "SHA-256 7515cfe7495526e0dcae656477dc2718180d71f77447e69c23159250ca1afbb2; "
        "still build-gated"
    ),
    target_rule=(
        "first emitted d_seg <= 0.98 * d_seg_start from a common n600 realized-through-R replay; "
        "0.98 is an ASSUMED preregistration policy and the resulting numeric threshold is DERIVED only after d_seg_start"
    ),
    maximum_nominal_epochs=250,  # ASSUMED ticket ceiling.
    censoring_rule="right-censor at 250 control epochs / one treatment LM proposal; missing crossing remains None",
    control_definition="continue unchanged terminal schedule for at most 250 epochs",
    treatment_definition=(
        "one registered full-P damped-GN/CG stage, at most one LM proposal with 16 CG steps "
        "(ceiling inherited from measured #341 probe, not promised optimum), exact n600 accept/rollback"
    ),
    verdict_scope="K=8 post-run subset solve formulation is NO-GO; full-P in-trainer HVP/CG family remains open and is not built",
    reformulation_reactivation_queue=(
        "build typed default-OFF compiler connection and in-trainer full-P HVP/CG stage",
        "add atomic pre/post-solve checkpoints, complete solver/resume state, and accept/rollback mutation ledger",
        "run cloned full-P matched A/B only after the build gate closes",
    ),
    source_artifacts=(
        "src/tac/witness_dsl/curriculum_dsl.py",
        "src/tac/canonical_equations/quadratic_head_chart_subset_solve_gap_20260707.py",
        "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz",
        "reports/basin_finisher_probe_20260707.json",
        "experiments/results/cheapen_real95_tilehalo_fp16_20260713/current_wall_receipt.json",
        ".omx/research/steps_dimension_95kill_20260713_SPEC.md",
    ),
    speed_configuration_rule=(
        "both arms require identical exact machine-readable speed configuration with every currently admitted/requested "
        "neutral fleet speed lever ON and all_requested_speed_levers_on=true; otherwise receipt is a blocker"
    ),
    measurement_authority_rule=(
        "future crossing receipt requires deterministic NumPy-fp32 realization through actual R plus the frozen CPU-torch "
        "scorer on all 600 states; MLX training is advisory and has no score authority"
    ),
)

TICKETS = (FRESH_TICKET, HARDNESS_TICKET, TERMINAL_SOLVE_TICKET)


def build_steps_dimension_epochs_to_target_v1() -> CanonicalEquation:
    """Build the unanchored accounting law; tickets are not empirical measurements."""

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="n600 epochs-to-target, update-count, and overhead-aware wall accounting",
        one_line_summary="Ticket-only n600 accounting keeps censored crossings and whole-step deltas unknown until matched receipts land.",
        latex_form=(
            r"E_{saved}=E_c-E_t,\quad f_{step,saved}=1-U_t/U_c,\quad "
            r"W=Ut_{update}+Et_{recurring}+o+t_{terminal},\quad "
            r"f_{wall}=T_t/T_c\ \mathrm{or}\ W_t/W_c,\quad f_{wall,saved}=1-f_{wall}"
        ),
        python_callable_module_path="tac.canonical_equations.steps_dimension_epochs_to_target_20260713:measured_accounting",
        domain_of_validity={
            "n_pairs": N_PAIRS,
            "empirical_verification_status": ASSUMED_AWAITING_VERIFICATION,
            "ticket_semantics": "no empirical anchors; only receipt-backed matched n600 crossings may populate deltas",
            "registry_integration": (
                "BLOCKED until a future importer verifies durable receipt schema, n600 cohort, epoch-0 history, "
                "matched config/source/checkpoint hashes, target/censor/init hashes, and WIRING_NEEDED closure"
            ),
            "right_censoring": "MEASURED_CENSORED retains completed window timings but missing first crossing is None, never zero",
            "hardness_accounting": "nominal epochs never substitute for exact optimizer update counts",
            "wall_composition": "prefer direct elapsed-to-crossing; fallback requires every recurring critical-path term allocated; composition_admissible=false returns None",
            "composition": "sequential composition requires measured sequential inputs; independent products are ASSUMED scenarios",
            "authority_axis": "no score authority; local MLX/CPU remains advisory",
        },
        units_in={
            "epochs": "nominal epochs",
            "optimizer_updates": "optimizer updates",
            "solver_hvp_steps": "full-P solver HVP/CG operations; separate from optimizer updates",
            "seconds_per_update": "seconds per optimizer update",
            "recurring_nonupdate_seconds_per_epoch": "seconds per nominal epoch",
            "one_time_overhead": "seconds",
            "terminal_critical_path": "seconds",
            "async_service": "seconds outside critical path unless wait is measured",
        },
        units_out={
            "epochs_saved": "nominal epochs",
            "step_fraction_saved": "unitless fraction",
            "wall_fraction": "unitless fraction",
            "wall_fraction_saved": "unitless fraction",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.canonical_equations.steps_dimension_epochs_to_target_20260713.measured_accounting",
            ".omx.research.sub015_DAG_steps_dimension_95kill_20260713",
        ),
        canonical_producers=(),
        provenance=build_provenance_for_predicted(
            model_id=EQUATION_ID,
            inputs_sha256=hashlib.sha256(_LAW.encode("utf-8")).hexdigest(),
            measurement_axis="[n600 ticket accounting; no empirical anchor]",
            hardware_substrate="numpy-portable",
            captured_at_utc=_UTC,
        ),
    )


__all__ = [
    "EQUATION_ID",
    "FRESH_ABSOLUTE_TARGET_DSEG",
    "FRESH_TICKET",
    "HARDNESS_TICKET",
    "N_PAIRS",
    "TERMINAL_SOLVE_TICKET",
    "TICKETS",
    "EpochsToTargetTicket",
    "build_steps_dimension_epochs_to_target_v1",
    "epochs_saved",
    "fixed_quality_threshold_factor",
    "measured_accounting",
    "step_fraction_saved",
    "symbolic_independent_step_fraction_saved",
    "wall_fraction",
    "wall_fraction_saved",
    "wall_seconds",
]
