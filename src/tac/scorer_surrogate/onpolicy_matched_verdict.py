# SPDX-License-Identifier: MIT
"""Fail-closed matched-window verdict math for on-policy costate surrogates.

This module is deliberately pure: it neither runs a scorer nor trains a model.
It adjudicates already-measured exact-control and surrogate-driven traces.  A
surrogate trajectory passes only when its exact through-R ``d_seg``, teacher CE,
and exact through-R ``d_pose`` traces remain within a deterministic-repeat noise
floor of the exact-teacher trajectory at every identical optimizer step.

Endpoint descent is not trajectory parity.  A target may descend independently
and still fail this gate.  Likewise, these training-window observations are not
an archive score and can never move the canonical frontier pointer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_METRICS: Final[tuple[str, ...]] = ("d_seg", "ce", "d_pose")
SCHEMA: Final[str] = "onpolicy_matched_verdict.v1"


class EvidenceStatus(StrEnum):
    """Operational state of one arm in a matched window."""

    MEASURED = "MEASURED"
    VALID_TERMINAL_FLOOR = "VALID_TERMINAL_FLOOR"
    BLOCKED = "BLOCKED"
    MISSING = "MISSING"


class VerdictKind(StrEnum):
    """Admission verdict for the requested formulation and regimes."""

    GO = "GO"
    NO_GO = "NO-GO"
    NEEDS_MORE = "NEEDS-MORE"


def _finite_nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CommonStepSchedule:
    """Declared optimizer-step schedule shared by both trajectory arms.

    ``control_values`` are the predeclared scalar control applied at each
    recorded step (for example, a learning rate or normalized step norm).  The
    semantic name and derivation are explicit so equal indices cannot hide
    different line-search or controller policies.
    """

    step_indices: tuple[int, ...]
    control_values: tuple[float, ...]
    control_name: str
    derivation: str

    def __post_init__(self) -> None:
        if not self.step_indices:
            raise ValueError("step_indices must be non-empty")
        if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in self.step_indices):
            raise ValueError("step_indices must contain nonnegative integers")
        if any(right <= left for left, right in pairwise(self.step_indices)):
            raise ValueError("step_indices must be strictly increasing")
        if len(self.control_values) != len(self.step_indices):
            raise ValueError("control_values must have one entry per recorded step")
        values = tuple(
            _finite_nonnegative(value, name=f"control_values[{index}]")
            for index, value in enumerate(self.control_values)
        )
        object.__setattr__(self, "control_values", values)
        _nonempty(self.control_name, name="control_name")
        _nonempty(self.derivation, name="derivation")

    @property
    def sha256(self) -> str:
        """Content identity for receipt and trace custody."""

        return _canonical_sha256(
            {
                "control_name": self.control_name,
                "control_values": self.control_values,
                "derivation": self.derivation,
                "step_indices": self.step_indices,
            }
        )


@dataclass(frozen=True)
class MetricTrace:
    """Exact observed metrics for one optimizer trajectory."""

    step_indices: tuple[int, ...]
    d_seg: tuple[float, ...]
    ce: tuple[float, ...]
    d_pose: tuple[float, ...]
    common_step_schedule_sha256: str

    def __post_init__(self) -> None:
        if not self.step_indices:
            raise ValueError("trace step_indices must be non-empty")
        if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in self.step_indices):
            raise ValueError("trace step_indices must contain nonnegative integers")
        if any(right <= left for left, right in pairwise(self.step_indices)):
            raise ValueError("trace step_indices must be strictly increasing")
        for metric in _METRICS:
            raw = tuple(getattr(self, metric))
            if len(raw) != len(self.step_indices):
                raise ValueError(f"{metric} must have one value per trace step")
            values = tuple(
                _finite_nonnegative(value, name=f"{metric}[{index}]")
                for index, value in enumerate(raw)
            )
            if metric == "d_seg" and any(value > 1.0 for value in values):
                raise ValueError("d_seg values must be fractions in [0, 1]")
            object.__setattr__(self, metric, values)
        _sha256(self.common_step_schedule_sha256, name="common_step_schedule_sha256")


@dataclass(frozen=True)
class ExactMetricAuthority:
    """Authority custody for metrics observed on either trajectory arm."""

    ce_exact_teacher_through_r: bool
    d_seg_exact_argmax_through_r: bool
    d_pose_exact_frozen_posenet_through_r: bool
    axis: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.axis, name="axis")
        _sha256(self.evidence_sha256, name="evidence_sha256")

    @property
    def complete(self) -> bool:
        return bool(
            self.ce_exact_teacher_through_r
            and self.d_seg_exact_argmax_through_r
            and self.d_pose_exact_frozen_posenet_through_r
        )


@dataclass(frozen=True)
class MetricObservation:
    """A metric trace coupled to its scorer custody."""

    trace: MetricTrace
    authority: ExactMetricAuthority


@dataclass(frozen=True)
class ArmEvidence:
    """Measured, floor-complete, blocked, or missing evidence for one arm."""

    arm_id: str
    status: EvidenceStatus
    observation: MetricObservation | None
    status_reason: str | None = None
    status_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.arm_id, name="arm_id")
        if self.status in {EvidenceStatus.VALID_TERMINAL_FLOOR, EvidenceStatus.BLOCKED}:
            _nonempty(self.status_reason or "", name="status_reason")
            _sha256(self.status_evidence_sha256 or "", name="status_evidence_sha256")
        elif self.status_evidence_sha256 is not None:
            _sha256(self.status_evidence_sha256, name="status_evidence_sha256")
        if (
            self.status in {EvidenceStatus.MEASURED, EvidenceStatus.VALID_TERMINAL_FLOOR}
            and self.observation is None
        ):
            raise ValueError(f"{self.status.value} evidence requires an observation")
        if self.status is EvidenceStatus.MISSING and self.observation is not None:
            raise ValueError("MISSING evidence cannot carry an observation")


@dataclass(frozen=True)
class RegimeEvidence:
    """Exact control and surrogate target for one named saved regime."""

    regime: str
    exact_control: ArmEvidence
    surrogate_target: ArmEvidence

    def __post_init__(self) -> None:
        _nonempty(self.regime, name="regime")


@dataclass(frozen=True)
class DeterministicRepeatNoiseFloor:
    """Per-metric tolerance backed by repeated deterministic exact evaluations."""

    d_seg: float
    ce: float
    d_pose: float
    repeat_count: int
    common_step_schedule_sha256: str
    source_evidence_sha256: str
    derivation: str

    def __post_init__(self) -> None:
        for metric in _METRICS:
            object.__setattr__(
                self,
                metric,
                _finite_nonnegative(getattr(self, metric), name=f"noise_floor.{metric}"),
            )
        if isinstance(self.repeat_count, bool) or self.repeat_count < 2:
            raise ValueError("repeat_count must be >= 2")
        _sha256(self.common_step_schedule_sha256, name="common_step_schedule_sha256")
        _sha256(self.source_evidence_sha256, name="source_evidence_sha256")
        if self.derivation != "maximum_pairwise_deterministic_repeat_delta_at_identical_steps":
            raise ValueError("noise floor must use the registered deterministic-repeat derivation")

    def for_metric(self, metric: str) -> float:
        if metric not in _METRICS:
            raise ValueError(f"unknown metric {metric!r}")
        return float(getattr(self, metric))

    @classmethod
    def from_repeat_receipt(
        cls,
        *,
        d_seg: float,
        ce: float,
        d_pose: float,
        repeat_count: int,
        common_step_schedule_sha256: str,
        repeat_receipt_sha256: str,
    ) -> DeterministicRepeatNoiseFloor:
        """Load already-derived repeat deltas with content-addressed custody."""

        return cls(
            d_seg=d_seg,
            ce=ce,
            d_pose=d_pose,
            repeat_count=repeat_count,
            common_step_schedule_sha256=common_step_schedule_sha256,
            source_evidence_sha256=repeat_receipt_sha256,
            derivation="maximum_pairwise_deterministic_repeat_delta_at_identical_steps",
        )


def derive_deterministic_repeat_noise_floor(
    repeated_observations: Sequence[MetricObservation],
    *,
    common_step_schedule: CommonStepSchedule,
) -> DeterministicRepeatNoiseFloor:
    """Derive the maximum pairwise repeat delta at every identical step.

    The repeats must all carry complete exact through-R metric authority and
    bind to the full declared schedule.  Identical deterministic repeats
    correctly derive a zero tolerance; no epsilon is injected.
    """

    repeats = tuple(repeated_observations)
    if len(repeats) < 2:
        raise ValueError("at least two deterministic repeat observations are required")
    axes = {observation.authority.axis for observation in repeats}
    if len(axes) != 1:
        raise ValueError("deterministic repeats must use one identical authority axis")
    for observation in repeats:
        if not observation.authority.complete:
            raise ValueError("deterministic repeat lacks complete exact through-R authority")
        if observation.trace.common_step_schedule_sha256 != common_step_schedule.sha256:
            raise ValueError("deterministic repeat is not bound to the common step schedule")
        if observation.trace.step_indices != common_step_schedule.step_indices:
            raise ValueError("deterministic repeat does not cover every common schedule step")

    floors: dict[str, float] = {}
    for metric in _METRICS:
        traces = [getattr(observation.trace, metric) for observation in repeats]
        floors[metric] = max(
            max(values) - min(values)
            for values in zip(*traces, strict=True)
        )
    source_evidence_sha256 = _canonical_sha256(
        {
            "authority_evidence_sha256": [
                observation.authority.evidence_sha256 for observation in repeats
            ],
            "metric_traces": [
                {
                    metric: getattr(observation.trace, metric) for metric in _METRICS
                }
                for observation in repeats
            ],
            "schedule_sha256": common_step_schedule.sha256,
        }
    )
    return DeterministicRepeatNoiseFloor(
        **floors,
        repeat_count=len(repeats),
        common_step_schedule_sha256=common_step_schedule.sha256,
        source_evidence_sha256=source_evidence_sha256,
        derivation="maximum_pairwise_deterministic_repeat_delta_at_identical_steps",
    )


@dataclass(frozen=True)
class RegimeMatchedVerdict:
    regime: str
    verdict: VerdictKind
    reason: str
    exact_control_status: EvidenceStatus
    target_status: EvidenceStatus
    metric_comparisons: Mapping[str, Mapping[str, Any]]
    exact_control_valid_terminal_floor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "exact_control_status": self.exact_control_status.value,
            "target_status": self.target_status.value,
            "metric_comparisons": {
                key: dict(value) for key, value in self.metric_comparisons.items()
            },
            "exact_control_valid_terminal_floor": self.exact_control_valid_terminal_floor,
        }


@dataclass(frozen=True)
class MatchedWindowVerdict:
    verdict: VerdictKind
    reason: str
    verdict_scope: str
    requested_regimes: tuple[str, ...]
    regime_verdicts: tuple[RegimeMatchedVerdict, ...]
    common_step_schedule_sha256: str
    deterministic_repeat_noise_floor: DeterministicRepeatNoiseFloor
    score_claim: bool = False
    promotion_eligible: bool = False
    pointer_expected_unmoved: bool = True
    authority_statement: str = (
        "training-window exact-metric comparison only; not archive/evaluator score authority"
    )
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        floor = self.deterministic_repeat_noise_floor
        return {
            "schema": self.schema,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "verdict_scope": self.verdict_scope,
            "requested_regimes": list(self.requested_regimes),
            "regime_verdicts": [row.to_dict() for row in self.regime_verdicts],
            "common_step_schedule_sha256": self.common_step_schedule_sha256,
            "deterministic_repeat_noise_floor": {
                "d_seg": floor.d_seg,
                "ce": floor.ce,
                "d_pose": floor.d_pose,
                "repeat_count": floor.repeat_count,
                "common_step_schedule_sha256": floor.common_step_schedule_sha256,
                "source_evidence_sha256": floor.source_evidence_sha256,
                "derivation": floor.derivation,
            },
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "pointer_expected_unmoved": self.pointer_expected_unmoved,
            "authority_statement": self.authority_statement,
        }


def _trace_schedule_problem(
    arm: ArmEvidence,
    *,
    common_step_schedule: CommonStepSchedule,
    exact_floor_prefix_allowed: bool,
) -> str | None:
    if arm.observation is None:
        return "metric observation is missing"
    trace = arm.observation.trace
    if not arm.observation.authority.complete:
        return "complete exact through-R CE/d_seg/d_pose authority is missing"
    if trace.common_step_schedule_sha256 != common_step_schedule.sha256:
        return "trace common-step-schedule custody does not match"
    if exact_floor_prefix_allowed:
        prefix = common_step_schedule.step_indices[: len(trace.step_indices)]
        if trace.step_indices != prefix:
            return "terminal-floor trace is not a prefix of the common step schedule"
    elif trace.step_indices != common_step_schedule.step_indices:
        return "trace does not cover the complete common step schedule"
    return None


def _is_nonworsening(values: tuple[float, ...]) -> bool:
    return all(right <= left for left, right in pairwise(values))


def _adjudicate_regime(
    row: RegimeEvidence,
    *,
    common_step_schedule: CommonStepSchedule,
    noise_floor: DeterministicRepeatNoiseFloor,
) -> RegimeMatchedVerdict:
    exact = row.exact_control
    target = row.surrogate_target
    if exact.status not in {EvidenceStatus.MEASURED, EvidenceStatus.VALID_TERMINAL_FLOOR}:
        return RegimeMatchedVerdict(
            regime=row.regime,
            verdict=VerdictKind.NEEDS_MORE,
            reason="exact-teacher control is neither measured nor a proven terminal floor",
            exact_control_status=exact.status,
            target_status=target.status,
            metric_comparisons={},
            exact_control_valid_terminal_floor=False,
        )
    exact_problem = _trace_schedule_problem(
        exact,
        common_step_schedule=common_step_schedule,
        exact_floor_prefix_allowed=exact.status is EvidenceStatus.VALID_TERMINAL_FLOOR,
    )
    if exact_problem is not None:
        return RegimeMatchedVerdict(
            regime=row.regime,
            verdict=VerdictKind.NEEDS_MORE,
            reason=f"exact-control authority invalid: {exact_problem}",
            exact_control_status=exact.status,
            target_status=target.status,
            metric_comparisons={},
            exact_control_valid_terminal_floor=False,
        )
    if target.status is EvidenceStatus.BLOCKED:
        return RegimeMatchedVerdict(
            regime=row.regime,
            verdict=VerdictKind.NO_GO,
            reason=(
                "surrogate target failed operationally after a valid exact control: "
                f"{target.status_reason}"
            ),
            exact_control_status=exact.status,
            target_status=target.status,
            metric_comparisons={},
            exact_control_valid_terminal_floor=exact.status is EvidenceStatus.VALID_TERMINAL_FLOOR,
        )
    if target.status is not EvidenceStatus.MEASURED:
        return RegimeMatchedVerdict(
            regime=row.regime,
            verdict=VerdictKind.NEEDS_MORE,
            reason="surrogate target measurement is missing",
            exact_control_status=exact.status,
            target_status=target.status,
            metric_comparisons={},
            exact_control_valid_terminal_floor=exact.status is EvidenceStatus.VALID_TERMINAL_FLOOR,
        )
    # A terminal-floor exact control defines a valid shorter matched prefix.
    # The target must cover exactly that prefix, neither fewer nor more steps.
    assert exact.observation is not None
    assert target.observation is not None
    target_problem = _trace_schedule_problem(
        target,
        common_step_schedule=common_step_schedule,
        exact_floor_prefix_allowed=exact.status is EvidenceStatus.VALID_TERMINAL_FLOOR,
    )
    if target_problem is not None:
        return RegimeMatchedVerdict(
            regime=row.regime,
            verdict=VerdictKind.NEEDS_MORE,
            reason=f"surrogate-target authority invalid: {target_problem}",
            exact_control_status=exact.status,
            target_status=target.status,
            metric_comparisons={},
            exact_control_valid_terminal_floor=exact.status is EvidenceStatus.VALID_TERMINAL_FLOOR,
        )
    exact_trace = exact.observation.trace
    target_trace = target.observation.trace
    if target_trace.step_indices != exact_trace.step_indices:
        return RegimeMatchedVerdict(
            regime=row.regime,
            verdict=VerdictKind.NEEDS_MORE,
            reason="exact and surrogate traces are not observed at identical step indices",
            exact_control_status=exact.status,
            target_status=target.status,
            metric_comparisons={},
            exact_control_valid_terminal_floor=exact.status is EvidenceStatus.VALID_TERMINAL_FLOOR,
        )

    comparisons: dict[str, dict[str, Any]] = {}
    for metric in _METRICS:
        exact_values = getattr(exact_trace, metric)
        target_values = getattr(target_trace, metric)
        deltas = tuple(abs(target_value - exact_value) for exact_value, target_value in zip(
            exact_values, target_values, strict=True
        ))
        tolerance = noise_floor.for_metric(metric)
        comparisons[metric] = {
            "max_abs_delta": max(deltas),
            "tolerance": tolerance,
            "within_repeat_noise_floor_at_every_step": all(delta <= tolerance for delta in deltas),
            "first_failing_step": next(
                (
                    step
                    for step, delta in zip(exact_trace.step_indices, deltas, strict=True)
                    if delta > tolerance
                ),
                None,
            ),
            "exact_trace_nonworsening": _is_nonworsening(exact_values),
            "target_trace_nonworsening": _is_nonworsening(target_values),
        }
    if not all(
        comparison["within_repeat_noise_floor_at_every_step"]
        for comparison in comparisons.values()
    ):
        return RegimeMatchedVerdict(
            regime=row.regime,
            verdict=VerdictKind.NO_GO,
            reason=(
                "surrogate-driven exact metric trace drift exceeds the deterministic-repeat "
                "noise floor at one or more matched steps"
            ),
            exact_control_status=exact.status,
            target_status=target.status,
            metric_comparisons=comparisons,
            exact_control_valid_terminal_floor=exact.status is EvidenceStatus.VALID_TERMINAL_FLOOR,
        )
    return RegimeMatchedVerdict(
        regime=row.regime,
        verdict=VerdictKind.GO,
        reason="all exact d_seg/CE/d_pose trace deltas stay within deterministic-repeat floors",
        exact_control_status=exact.status,
        target_status=target.status,
        metric_comparisons=comparisons,
        exact_control_valid_terminal_floor=exact.status is EvidenceStatus.VALID_TERMINAL_FLOOR,
    )


def adjudicate_matched_windows(
    *,
    requested_regimes: Sequence[str],
    regime_evidence: Sequence[RegimeEvidence],
    common_step_schedule: CommonStepSchedule,
    deterministic_repeat_noise_floor: DeterministicRepeatNoiseFloor,
) -> MatchedWindowVerdict:
    """Adjudicate every requested regime with epistemic insufficiency first.

    ``NEEDS-MORE`` has global precedence when any requested regime or exact
    authority is missing.  This prevents a local formulation failure from
    being inflated into an all-regime verdict before the declared evidence
    contract is complete.
    """

    requested = tuple(requested_regimes)
    if not requested or any(not isinstance(regime, str) or not regime.strip() for regime in requested):
        raise ValueError("requested_regimes must contain non-empty names")
    if len(set(requested)) != len(requested):
        raise ValueError("requested_regimes must be unique")
    if deterministic_repeat_noise_floor.common_step_schedule_sha256 != common_step_schedule.sha256:
        reason = "deterministic-repeat noise floor is not bound to the declared common schedule"
        return MatchedWindowVerdict(
            verdict=VerdictKind.NEEDS_MORE,
            reason=reason,
            verdict_scope=(
                "formulation — on-policy costate surrogate matched-window gate on requested regimes; "
                "not surrogate family/paradigm and not score authority"
            ),
            requested_regimes=requested,
            regime_verdicts=(),
            common_step_schedule_sha256=common_step_schedule.sha256,
            deterministic_repeat_noise_floor=deterministic_repeat_noise_floor,
        )
    by_regime: dict[str, RegimeEvidence] = {}
    for row in regime_evidence:
        if row.regime in by_regime:
            raise ValueError(f"duplicate regime evidence for {row.regime!r}")
        by_regime[row.regime] = row

    verdict_rows: list[RegimeMatchedVerdict] = []
    for regime in requested:
        row = by_regime.get(regime)
        if row is None:
            verdict_rows.append(
                RegimeMatchedVerdict(
                    regime=regime,
                    verdict=VerdictKind.NEEDS_MORE,
                    reason="requested regime evidence is missing",
                    exact_control_status=EvidenceStatus.MISSING,
                    target_status=EvidenceStatus.MISSING,
                    metric_comparisons={},
                    exact_control_valid_terminal_floor=False,
                )
            )
            continue
        verdict_rows.append(
            _adjudicate_regime(
                row,
                common_step_schedule=common_step_schedule,
                noise_floor=deterministic_repeat_noise_floor,
            )
        )

    if any(row.verdict is VerdictKind.NEEDS_MORE for row in verdict_rows):
        verdict = VerdictKind.NEEDS_MORE
        reason = "one or more requested regimes lack valid matched exact authority"
    elif any(row.verdict is VerdictKind.NO_GO for row in verdict_rows):
        verdict = VerdictKind.NO_GO
        reason = "the tested surrogate formulation failed in one or more requested regimes"
    else:
        verdict = VerdictKind.GO
        reason = "the tested surrogate formulation matched exact-teacher traces in every requested regime"
    regimes_label = ",".join(requested)
    return MatchedWindowVerdict(
        verdict=verdict,
        reason=reason,
        verdict_scope=(
            "formulation — on-policy costate surrogate matched-window gate on requested regimes "
            f"[{regimes_label}]; not surrogate family/paradigm and not score authority"
        ),
        requested_regimes=requested,
        regime_verdicts=tuple(verdict_rows),
        common_step_schedule_sha256=common_step_schedule.sha256,
        deterministic_repeat_noise_floor=deterministic_repeat_noise_floor,
    )


def _timing_stats(samples: Sequence[float], *, name: str) -> dict[str, float | int]:
    values = tuple(
        _finite_nonnegative(value, name=f"{name}[{index}]")
        for index, value in enumerate(samples)
    )
    if not values:
        raise ValueError(f"{name} timing samples must be non-empty")
    return {
        "count": len(values),
        "total_seconds": math.fsum(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def aggregate_isolated_timings(
    *,
    common_step_schedule: CommonStepSchedule,
    exact_schedule_sha256: str,
    surrogate_schedule_sha256: str,
    exact_forward_only: Sequence[float],
    exact_costate_forward_backward: Sequence[float],
    anchor_fit: Sequence[float],
    surrogate_inference: Sequence[float],
    renderer_vjp_exact_control: Sequence[float],
    renderer_vjp_surrogate_target: Sequence[float],
    whole_matched_window_exact_control: Sequence[float],
    whole_matched_window_surrogate_target: Sequence[float],
) -> dict[str, Any]:
    """Aggregate matched sums of complete per-step operational timers.

    Each input window must already sum the same complete step boundary for its
    arm: render, provider, renderer VJP, and candidate update. Isolated
    component timers remain diagnostics and are not used to manufacture it.
    """

    expected_sha256 = common_step_schedule.sha256
    if _sha256(exact_schedule_sha256, name="exact_schedule_sha256") != expected_sha256:
        raise ValueError("exact timing window is not bound to the common step schedule")
    if _sha256(surrogate_schedule_sha256, name="surrogate_schedule_sha256") != expected_sha256:
        raise ValueError("surrogate timing window is not bound to the common step schedule")
    exact_window = _timing_stats(
        whole_matched_window_exact_control,
        name="whole_matched_window_exact_control",
    )
    surrogate_window = _timing_stats(
        whole_matched_window_surrogate_target,
        name="whole_matched_window_surrogate_target",
    )
    surrogate_mean = float(surrogate_window["mean_seconds"])
    speedup = None if surrogate_mean == 0.0 else float(exact_window["mean_seconds"]) / surrogate_mean
    return {
        "schema": "onpolicy_isolated_timings.v1",
        "common_step_schedule_sha256": expected_sha256,
        "exact_forward_only": _timing_stats(exact_forward_only, name="exact_forward_only"),
        "exact_costate_forward_backward": _timing_stats(
            exact_costate_forward_backward,
            name="exact_costate_forward_backward",
        ),
        "anchor_fit": _timing_stats(anchor_fit, name="anchor_fit"),
        "surrogate_inference": _timing_stats(
            surrogate_inference,
            name="surrogate_inference",
        ),
        "renderer_vjp": {
            "exact_control": _timing_stats(
                renderer_vjp_exact_control,
                name="renderer_vjp_exact_control",
            ),
            "surrogate_target": _timing_stats(
                renderer_vjp_surrogate_target,
                name="renderer_vjp_surrogate_target",
            ),
        },
        "whole_matched_window": {
            "exact_control": exact_window,
            "surrogate_target": surrogate_window,
            "matched_speedup_exact_over_surrogate": speedup,
            "matched_speedup_defined": speedup is not None,
            "comparison_basis": (
                "sums of symmetric complete per-step operational timers under one common control schedule"
            ),
        },
        "complete_per_step_timer_sums_used_for_window": True,
        "isolated_component_sums_used_for_window": False,
        "control_law_conflation": False,
        "score_claim": False,
    }
