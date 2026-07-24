# SPDX-License-Identifier: MIT
"""Typed, resume-safe event continuation for the DDM family-(d) fitting engine.

The continuation clock is the accepted receiver state plus causal events.  A
budget is a safety cap, never a stage length or handoff condition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

SCHEMA: Final = "DDMEventContinuationV1"
EVENT_MARK_SCHEMA: Final = "ddm_event_mark.v1"
CHARGE_AUDIT_SCHEMA: Final = "ddm_two_part_charge_audit.v1"
RATE_BREAK_EVEN: Final = 25.0 / 37_545_489.0

VisibilityType = Literal["seg-only", "pose-only(frame_0)", "joint"]
NodeKind = Literal["continuation", "solve_interleave", "terminal_band", "governed_stop"]
NODE_KINDS: Final = {"continuation", "solve_interleave", "terminal_band", "governed_stop"}
EXACT_ACCEPTANCE_METRIC: Final = (
    "100*d_seg_R+sqrt(10*d_pose_YUV6_R)+25*archive_bytes/37545489"
)
PROPOSAL_METRIC: Final = "scorer_recursive_rank4_fisher_corrected_J"


class DDMEventContinuationError(ValueError):
    """Fail-closed malformed schedule, event, proposal, or resume state."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: str, *, field_name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DDMEventContinuationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DDMEventContinuationError(f"{field_name} must be a JSON boolean")
    return value


def _strict_int(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DDMEventContinuationError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite_number(value: Any, *, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DDMEventContinuationError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise DDMEventContinuationError(f"{field_name} is outside its finite domain")
    return result


@dataclass(frozen=True, slots=True)
class BudgetCapsV1:
    """Safety/resource caps; none may cause a semantic segment transition."""

    maximum_receiver_verdicts: int
    maximum_wall_seconds: float
    checkpoint_recovery_loss_verdicts: int
    maximum_counted_bytes: int

    def __post_init__(self) -> None:
        ints = (
            self.maximum_receiver_verdicts,
            self.checkpoint_recovery_loss_verdicts,
            self.maximum_counted_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in ints):
            raise DDMEventContinuationError("budget integer caps must be positive")
        if not math.isfinite(self.maximum_wall_seconds) or self.maximum_wall_seconds <= 0.0:
            raise DDMEventContinuationError("maximum_wall_seconds must be finite and positive")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BudgetCapsV1:
        return cls(
            maximum_receiver_verdicts=_strict_int(
                payload["maximum_receiver_verdicts"],
                field_name="maximum_receiver_verdicts",
                minimum=1,
            ),
            maximum_wall_seconds=_finite_number(
                payload["maximum_wall_seconds"],
                field_name="maximum_wall_seconds",
                minimum=0.0,
            ),
            checkpoint_recovery_loss_verdicts=_strict_int(
                payload["checkpoint_recovery_loss_verdicts"],
                field_name="checkpoint_recovery_loss_verdicts",
                minimum=1,
            ),
            maximum_counted_bytes=_strict_int(
                payload["maximum_counted_bytes"],
                field_name="maximum_counted_bytes",
                minimum=1,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "role": "safety_and_resource_caps_only_never_primary_handoff",
            "maximum_receiver_verdicts": self.maximum_receiver_verdicts,
            "maximum_wall_seconds": self.maximum_wall_seconds,
            "checkpoint_recovery_loss_verdicts": self.checkpoint_recovery_loss_verdicts,
            "maximum_counted_bytes": self.maximum_counted_bytes,
        }


@dataclass(frozen=True, slots=True)
class EventNodeV1:
    node_id: str
    kind: NodeKind
    entry_events: tuple[str, ...]
    exit_events: tuple[str, ...]
    next_by_event: Mapping[str, str]
    active_group_policy: str
    checkpoint_events: tuple[str, ...]
    execution_enabled: bool = True
    blocker: str | None = None

    def __post_init__(self) -> None:
        if not self.node_id or not self.entry_events or not self.exit_events:
            raise DDMEventContinuationError("event node id and entry/exit events must be nonempty")
        if self.kind not in NODE_KINDS:
            raise DDMEventContinuationError(f"{self.node_id} has an unknown node kind")
        if len(set(self.entry_events)) != len(self.entry_events) or len(set(self.exit_events)) != len(
            self.exit_events
        ):
            raise DDMEventContinuationError(f"{self.node_id} has duplicate event identities")
        if set(self.next_by_event) != set(self.exit_events):
            raise DDMEventContinuationError(
                f"{self.node_id} must route every and only declared exit event"
            )
        if any(not event or not target for event, target in self.next_by_event.items()):
            raise DDMEventContinuationError(f"{self.node_id} has an empty event route")
        if not set(self.checkpoint_events) <= set(self.exit_events):
            raise DDMEventContinuationError(
                f"{self.node_id} checkpoints an undeclared exit event"
            )
        if not self.active_group_policy:
            raise DDMEventContinuationError(f"{self.node_id} lacks active-group policy")
        if not self.execution_enabled and not self.blocker:
            raise DDMEventContinuationError(f"{self.node_id} is disabled without a blocker")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EventNodeV1:
        return cls(
            node_id=str(payload["node_id"]),
            kind=str(payload["kind"]),  # type: ignore[arg-type]
            entry_events=tuple(str(value) for value in payload["entry_events"]),
            exit_events=tuple(str(value) for value in payload["exit_events"]),
            next_by_event={str(key): str(value) for key, value in dict(payload["next_by_event"]).items()},
            active_group_policy=str(payload["active_group_policy"]),
            checkpoint_events=tuple(str(value) for value in payload["checkpoint_events"]),
            execution_enabled=_strict_bool(
                payload.get("execution_enabled", True),
                field_name=f"{payload.get('node_id', '<unknown>')}.execution_enabled",
            ),
            blocker=None if payload.get("blocker") is None else str(payload["blocker"]),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "entry_events": list(self.entry_events),
            "exit_events": list(self.exit_events),
            "next_by_event": dict(sorted(self.next_by_event.items())),
            "active_group_policy": self.active_group_policy,
            "checkpoint_events": list(self.checkpoint_events),
            "execution_enabled": self.execution_enabled,
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionObservationV1:
    """Two-axis measured acquisition row. No scalar blend exists."""

    proposal_id: str
    score_before: float
    score_after: float
    counted_bytes_before: int
    counted_bytes_after: int
    measured_work: float
    score_receipt_sha256: str
    description_receipt_sha256: str
    evidence_axis: str

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.evidence_axis:
            raise DDMEventContinuationError("acquisition proposal and evidence axis must be nonempty")
        if not all(math.isfinite(value) and value >= 0.0 for value in (self.score_before, self.score_after)):
            raise DDMEventContinuationError("acquisition scores must be finite and nonnegative")
        if (
            isinstance(self.counted_bytes_before, bool)
            or isinstance(self.counted_bytes_after, bool)
            or not isinstance(self.counted_bytes_before, int)
            or not isinstance(self.counted_bytes_after, int)
            or self.counted_bytes_before < 0
            or self.counted_bytes_after < 0
        ):
            raise DDMEventContinuationError("acquisition counted bytes must be nonnegative integers")
        if not math.isfinite(self.measured_work) or self.measured_work <= 0.0:
            raise DDMEventContinuationError("acquisition measured work must be finite and positive")
        _digest(self.score_receipt_sha256, field_name="score_receipt_sha256")
        _digest(self.description_receipt_sha256, field_name="description_receipt_sha256")

    @property
    def g_s(self) -> float:
        return (self.score_before - self.score_after) / self.measured_work

    @property
    def g_l(self) -> float:
        return (self.counted_bytes_before - self.counted_bytes_after) / self.measured_work

    @property
    def exact_score_admissible(self) -> bool:
        return self.score_after < self.score_before

    def to_payload(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "S_before": self.score_before,
            "S_after": self.score_after,
            "counted_bytes_before": self.counted_bytes_before,
            "counted_bytes_after": self.counted_bytes_after,
            "measured_work": self.measured_work,
            "g_S": self.g_s,
            "g_L": self.g_l,
            "score_receipt_sha256": self.score_receipt_sha256,
            "description_receipt_sha256": self.description_receipt_sha256,
            "evidence_axis": self.evidence_axis,
        }


def pareto_acquisition_order(
    rows: Sequence[AcquisitionObservationV1],
) -> tuple[AcquisitionObservationV1, ...]:
    """Nondominated rows first; stable typed proposal ID is the only tie-break."""

    if len({row.proposal_id for row in rows}) != len(rows):
        raise DDMEventContinuationError("acquisition proposal IDs must be unique")

    def dominated(row: AcquisitionObservationV1) -> bool:
        return any(
            other.proposal_id != row.proposal_id
            and other.g_s >= row.g_s
            and other.g_l >= row.g_l
            and (other.g_s > row.g_s or other.g_l > row.g_l)
            for other in rows
        )

    return tuple(
        sorted(
            rows,
            key=lambda row: (
                dominated(row),
                not row.exact_score_admissible,
                row.proposal_id,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class ChargeSectionV1:
    section_id: str
    selected_by_video: bool
    counted_bytes: int
    parseback_sha256: str
    role: str

    def __post_init__(self) -> None:
        if not self.section_id or not self.role:
            raise DDMEventContinuationError("charge section id and role must be nonempty")
        if (
            isinstance(self.counted_bytes, bool)
            or not isinstance(self.counted_bytes, int)
            or self.counted_bytes < 0
        ):
            raise DDMEventContinuationError("charge section bytes must be nonnegative")
        _digest(self.parseback_sha256, field_name="parseback_sha256")
        if self.selected_by_video and self.counted_bytes == 0:
            raise DDMEventContinuationError(
                f"video-selected section {self.section_id} cannot be declared free"
            )


def audit_two_part_charge(
    sections: Sequence[ChargeSectionV1],
    *,
    archive_bytes: int,
    fixed_interpreter_sha256: str,
) -> dict[str, Any]:
    """OP-GC1-2: all video-selected information is charged and conserved."""

    _digest(fixed_interpreter_sha256, field_name="fixed_interpreter_sha256")
    if isinstance(archive_bytes, bool) or archive_bytes < 0:
        raise DDMEventContinuationError("archive_bytes must be a nonnegative integer")
    if len({section.section_id for section in sections}) != len(sections):
        raise DDMEventContinuationError("charge section IDs must be unique")
    total = sum(section.counted_bytes for section in sections)
    if total != archive_bytes:
        raise DDMEventContinuationError(
            f"charged section sum {total} differs from archive bytes {archive_bytes}"
        )
    return {
        "schema": CHARGE_AUDIT_SCHEMA,
        "fixed_video_independent_interpreter_sha256": fixed_interpreter_sha256,
        "archive_bytes": archive_bytes,
        "charged_section_bytes": total,
        "conserved": True,
        "sections": [
            {
                "section_id": section.section_id,
                "selected_by_video": section.selected_by_video,
                "counted_bytes": section.counted_bytes,
                "parseback_sha256": section.parseback_sha256,
                "role": section.role,
            }
            for section in sections
        ],
    }


@dataclass(frozen=True, slots=True)
class ContinuationStateV1:
    node_id: str
    accepted_state_id: str
    accepted_verdicts: int
    event_sequence: int
    emitted_event_ids: tuple[str, ...] = ()
    pose_gate_history: tuple[float, ...] = ()
    terminal: bool = False

    def __post_init__(self) -> None:
        if not self.node_id or not self.accepted_state_id:
            raise DDMEventContinuationError("continuation state identities must be nonempty")
        if self.accepted_verdicts < 0 or self.event_sequence < 0:
            raise DDMEventContinuationError("continuation counters must be nonnegative")
        if len(set(self.emitted_event_ids)) != len(self.emitted_event_ids):
            raise DDMEventContinuationError("resume state repeats a causal event ID")
        if any(not math.isfinite(value) or value < 0.0 for value in self.pose_gate_history):
            raise DDMEventContinuationError("pose-gate history must be finite and nonnegative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "ddm_event_continuation_state.v1",
            "node_id": self.node_id,
            "accepted_state_id": self.accepted_state_id,
            "accepted_verdicts": self.accepted_verdicts,
            "event_sequence": self.event_sequence,
            "emitted_event_ids": list(self.emitted_event_ids),
            "pose_gate_history": list(self.pose_gate_history),
            "terminal": self.terminal,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContinuationStateV1:
        if payload.get("schema") != "ddm_event_continuation_state.v1":
            raise DDMEventContinuationError("continuation resume-state schema differs")
        return cls(
            node_id=str(payload["node_id"]),
            accepted_state_id=str(payload["accepted_state_id"]),
            accepted_verdicts=_strict_int(
                payload["accepted_verdicts"],
                field_name="accepted_verdicts",
            ),
            event_sequence=_strict_int(
                payload["event_sequence"],
                field_name="event_sequence",
            ),
            emitted_event_ids=tuple(str(value) for value in payload["emitted_event_ids"]),
            pose_gate_history=tuple(float(value) for value in payload.get("pose_gate_history", ())),
            terminal=_strict_bool(payload["terminal"], field_name="terminal"),
        )


@dataclass(frozen=True, slots=True)
class DDMEventContinuationV1:
    graph_id: str
    initial_node_id: str
    nodes: tuple[EventNodeV1, ...]
    budget_caps: BudgetCapsV1
    proposal_metric_selector: str
    exact_acceptance_metric: str
    box_tolerance_policy: Mapping[str, Any]
    visibility_policy: Mapping[str, str]
    terminal_hooks: Mapping[str, Mapping[str, Any]]
    telemetry_fields: tuple[str, ...]
    execution_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.graph_id or not self.proposal_metric_selector or not self.exact_acceptance_metric:
            raise DDMEventContinuationError("event graph identities and metrics must be nonempty")
        if self.proposal_metric_selector != PROPOSAL_METRIC:
            raise DDMEventContinuationError("event graph proposal metric selector differs")
        if self.exact_acceptance_metric != EXACT_ACCEPTANCE_METRIC:
            raise DDMEventContinuationError("event graph exact acceptance metric differs")
        if not isinstance(self.execution_allowed, bool):
            raise DDMEventContinuationError("event graph execution_allowed must be boolean")
        by_id = {node.node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes) or self.initial_node_id not in by_id:
            raise DDMEventContinuationError("event graph node IDs differ or initial node is missing")
        for node in self.nodes:
            unknown = set(node.next_by_event.values()) - set(by_id) - {"STOP"}
            if unknown:
                raise DDMEventContinuationError(
                    f"{node.node_id} routes to unknown nodes {sorted(unknown)}"
                )
        reachable = {self.initial_node_id}
        while True:
            expanded = reachable | {
                target
                for node_id in reachable
                for target in by_id[node_id].next_by_event.values()
                if target != "STOP"
            }
            if expanded == reachable:
                break
            reachable = expanded
        if reachable != set(by_id):
            raise DDMEventContinuationError(
                f"event graph has unreachable nodes {sorted(set(by_id) - reachable)}"
            )
        required_visibility = {
            "frame_0": "pose-only(frame_0)",
            "fine_chroma": "seg-only",
            "shared_visible": "joint",
            "resize_null": "gauge_fixed_out",
            "blind_coordinates": "excluded",
        }
        if dict(self.visibility_policy) != required_visibility:
            raise DDMEventContinuationError("visibility parametrization policy differs")
        required_telemetry = {
            *(f"Q{index}" for index in range(1, 8)),
            "lever_engage",
            "term_inert",
            "liveness",
            "S_before",
            "S_after",
            "counted_bytes_before",
            "counted_bytes_after",
            "measured_work",
            "g_S",
            "g_L",
            "delta_S_per_wall_clock_hour",
            "delta_bytes_per_step",
            "box_milestone_crossed",
            "rollback_reason",
        }
        if not required_telemetry <= set(self.telemetry_fields):
            raise DDMEventContinuationError(
                f"event telemetry misses {sorted(required_telemetry - set(self.telemetry_fields))}"
            )
        required_box_policy = {
            "descent_box_role": "milestone_not_stop",
            "describe_solve_box_role": "tolerance_stop",
            "ms2r_role": "proposal_ordering_prior_until_describe_solve",
            "global_fallback_declared": True,
            "descent_continuation": "exact_receiver_realized_delta_S_lt_zero",
        }
        if any(
            self.box_tolerance_policy.get(key) != value
            for key, value in required_box_policy.items()
        ):
            raise DDMEventContinuationError("box/two-channel continuation policy differs")
        for required_hook in ("fork_head_solve", "head_offset_solver", "ms2_terminal_solve", "mc_finisher"):
            if required_hook not in self.terminal_hooks:
                raise DDMEventContinuationError(f"event graph lacks {required_hook} hook")
        if self.terminal_hooks["mc_finisher"].get("execution_enabled") is not False:
            raise DDMEventContinuationError("MC finisher must remain preregistered and disabled")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DDMEventContinuationV1:
        if payload.get("schema") != SCHEMA:
            raise DDMEventContinuationError(f"event graph schema must be {SCHEMA}")
        forbidden = {
            "stages",
            "stage_count",
            "maximum_steps_per_stage",
            "verdict_interval_steps",
            "fixed_stage_targets",
        }
        present = forbidden & set(payload)
        if present:
            raise DDMEventContinuationError(
                f"event continuation contains forbidden fixed-stage actuators {sorted(present)}"
            )
        return cls(
            graph_id=str(payload["graph_id"]),
            initial_node_id=str(payload["initial_node_id"]),
            nodes=tuple(EventNodeV1.from_payload(row) for row in payload["nodes"]),
            budget_caps=BudgetCapsV1.from_payload(payload["budget_caps"]),
            proposal_metric_selector=str(payload["proposal_metric_selector"]),
            exact_acceptance_metric=str(payload["exact_acceptance_metric"]),
            box_tolerance_policy=dict(payload["box_tolerance_policy"]),
            visibility_policy={str(key): str(value) for key, value in dict(payload["visibility_policy"]).items()},
            terminal_hooks={str(key): dict(value) for key, value in dict(payload["terminal_hooks"]).items()},
            telemetry_fields=tuple(str(value) for value in payload["telemetry_fields"]),
            execution_allowed=_strict_bool(
                payload.get("execution_allowed", False),
                field_name="execution_allowed",
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "graph_id": self.graph_id,
            "primary_clock": "accepted_receiver_states_and_causal_events",
            "initial_node_id": self.initial_node_id,
            "nodes": [node.to_payload() for node in self.nodes],
            "budget_caps": self.budget_caps.to_payload(),
            "proposal_metric_selector": self.proposal_metric_selector,
            "exact_acceptance_metric": self.exact_acceptance_metric,
            "box_tolerance_policy": dict(self.box_tolerance_policy),
            "visibility_policy": dict(self.visibility_policy),
            "terminal_hooks": {
                key: dict(value) for key, value in sorted(self.terminal_hooks.items())
            },
            "telemetry_fields": list(self.telemetry_fields),
            "execution_allowed": self.execution_allowed,
        }

    @property
    def semantic_hash(self) -> str:
        return _sha256(self.to_payload())

    def initial_state(self, *, accepted_state_id: str) -> ContinuationStateV1:
        return ContinuationStateV1(
            node_id=self.initial_node_id,
            accepted_state_id=accepted_state_id,
            accepted_verdicts=0,
            event_sequence=0,
        )

    def advance(
        self,
        state: ContinuationStateV1,
        *,
        event: str,
        accepted_state_id: str,
        telemetry: Mapping[str, Any],
    ) -> tuple[ContinuationStateV1, dict[str, Any]]:
        """Apply one causal event with append-only/deduplicated resume semantics."""

        if state.terminal:
            raise DDMEventContinuationError("terminal continuation cannot advance")
        node = {row.node_id: row for row in self.nodes}.get(state.node_id)
        if node is None:
            raise DDMEventContinuationError("resume state names an unknown event node")
        if event not in node.exit_events:
            raise DDMEventContinuationError(
                f"event {event!r} is not a declared exit from {node.node_id}"
            )
        if not node.execution_enabled:
            raise DDMEventContinuationError(
                f"event node {node.node_id} is disabled: {node.blocker}"
            )
        missing = set(self.telemetry_fields) - set(telemetry)
        if missing:
            raise DDMEventContinuationError(f"event mark misses telemetry {sorted(missing)}")
        score_before = _finite_number(
            telemetry["S_before"],
            field_name="S_before",
            minimum=0.0,
        )
        score_after = _finite_number(
            telemetry["S_after"],
            field_name="S_after",
            minimum=0.0,
        )
        bytes_before = _strict_int(
            telemetry["counted_bytes_before"],
            field_name="counted_bytes_before",
        )
        bytes_after = _strict_int(
            telemetry["counted_bytes_after"],
            field_name="counted_bytes_after",
        )
        measured_work = _finite_number(
            telemetry["measured_work"],
            field_name="measured_work",
            minimum=0.0,
        )
        if measured_work <= 0.0:
            raise DDMEventContinuationError("measured_work must be positive")
        expected_g_s = (score_before - score_after) / measured_work
        expected_g_l = (bytes_before - bytes_after) / measured_work
        g_s = _finite_number(telemetry["g_S"], field_name="g_S")
        g_l = _finite_number(telemetry["g_L"], field_name="g_L")
        if not math.isclose(g_s, expected_g_s, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise DDMEventContinuationError("g_S differs from measured score/work")
        if not math.isclose(g_l, expected_g_l, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise DDMEventContinuationError("g_L differs from measured bytes/work")
        _finite_number(
            telemetry["delta_S_per_wall_clock_hour"],
            field_name="delta_S_per_wall_clock_hour",
        )
        delta_bytes = _finite_number(
            telemetry["delta_bytes_per_step"],
            field_name="delta_bytes_per_step",
        )
        if not math.isclose(
            delta_bytes,
            float(bytes_after - bytes_before),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise DDMEventContinuationError(
                "delta_bytes_per_step differs from measured archive bytes"
            )
        box_crossed = _strict_bool(
            telemetry["box_milestone_crossed"],
            field_name="box_milestone_crossed",
        )
        liveness = telemetry.get("liveness")
        if not isinstance(liveness, Mapping) or not {
            "accepted_batch_fraction",
            "weights_stepped",
            "frozen",
        } <= set(liveness):
            raise DDMEventContinuationError("event mark liveness stamp is incomplete")
        target = node.next_by_event.get(event)
        if target is None:
            raise DDMEventContinuationError(f"event {event!r} lacks a deterministic route")
        q7 = telemetry.get("Q7")
        if not isinstance(q7, Mapping):
            raise DDMEventContinuationError("Q7 telemetry must be a mapping")
        d_seg = _finite_number(q7.get("d_seg"), field_name="Q7.d_seg", minimum=0.0)
        accepted = score_after < score_before
        rollback_reason = telemetry["rollback_reason"]
        if rollback_reason is not None and (
            not isinstance(rollback_reason, str) or not rollback_reason
        ):
            raise DDMEventContinuationError("rollback_reason must be null or nonempty text")
        if event == "exact_joint_delta_nonnegative_rollback" and (
            accepted or rollback_reason is None
        ):
            raise DDMEventContinuationError(
                "nonnegative rollback requires rejected exact score and a reason"
            )
        if event == "box_milestone_crossed" and (not accepted or not box_crossed):
            raise DDMEventContinuationError(
                "box milestone requires an accepted exact score and crossed flag"
            )
        if accepted and rollback_reason is not None:
            raise DDMEventContinuationError("accepted score cannot carry rollback_reason")
        accepted_state_after = accepted_state_id if accepted else state.accepted_state_id
        event_identity = {
            "graph_id": self.graph_id,
            "graph_hash": self.semantic_hash,
            "sequence": state.event_sequence + 1,
            "from_node": node.node_id,
            "event": event,
            "accepted_state_before": state.accepted_state_id,
            "accepted_state_after": accepted_state_after,
        }
        event_id = _sha256(event_identity)
        if event_id in state.emitted_event_ids:
            raise DDMEventContinuationError("causal event ID repeats on resume")
        accepted_verdicts = state.accepted_verdicts + int(accepted)
        if accepted_verdicts > self.budget_caps.maximum_receiver_verdicts:
            raise DDMEventContinuationError("receiver-verdict safety cap exceeded")
        next_state = ContinuationStateV1(
            node_id=node.node_id if target == "STOP" else target,
            accepted_state_id=accepted_state_after,
            accepted_verdicts=accepted_verdicts,
            event_sequence=state.event_sequence + 1,
            emitted_event_ids=(*state.emitted_event_ids, event_id),
            pose_gate_history=(
                *state.pose_gate_history,
                d_seg,
            )[-5:],
            terminal=target == "STOP",
        )
        mark = {
            "schema": EVENT_MARK_SCHEMA,
            "event_id": event_id,
            "identity": event_identity,
            "telemetry": dict(telemetry),
            "accepted": accepted,
            "rollback_reason": telemetry["rollback_reason"],
            "checkpoint_required": (
                event in node.checkpoint_events
                or accepted
                or telemetry["rollback_reason"] is not None
            ),
            "state_after": next_state.to_payload(),
        }
        return next_state, mark


def build_j8e_event_continuation(
    *,
    maximum_receiver_verdicts: int,
    maximum_wall_seconds: float,
    maximum_counted_bytes: int,
    execution_allowed: bool = False,
) -> DDMEventContinuationV1:
    """Build the canonical #688 graph; counts are safety caps, never stages."""

    telemetry_fields = (
        *(f"Q{index}" for index in range(1, 8)),
        "lever_engage",
        "term_inert",
        "liveness",
        "S_before",
        "S_after",
        "counted_bytes_before",
        "counted_bytes_after",
        "measured_work",
        "g_S",
        "g_L",
        "delta_S_per_wall_clock_hour",
        "delta_bytes_per_step",
        "box_milestone_crossed",
        "rollback_reason",
    )
    continuation_events = (
        "box_milestone_crossed",
        "exact_joint_delta_nonnegative_rollback",
        "economic_or_dynamics_stop",
    )
    return DDMEventContinuationV1(
        graph_id="ddm_j8e_688_scorer_recursive_event_continuation",
        initial_node_id="resume_boundary_recondition",
        nodes=(
            EventNodeV1(
                node_id="resume_boundary_recondition",
                kind="continuation",
                entry_events=("ws3_receiver_closed_start_bound",),
                exit_events=(
                    "first_component_safe_n600_residual_admission",
                    "economic_or_dynamics_stop",
                ),
                next_by_event={
                    "first_component_safe_n600_residual_admission": (
                        "costate_ranked_joint_continuation"
                    ),
                    "economic_or_dynamics_stop": "STOP",
                },
                active_group_policy=(
                    "scorer_recursive_costate_pareto_nondominance_stable_id"
                ),
                checkpoint_events=(
                    "first_component_safe_n600_residual_admission",
                    "economic_or_dynamics_stop",
                ),
            ),
            EventNodeV1(
                node_id="costate_ranked_joint_continuation",
                kind="continuation",
                entry_events=("first_component_safe_n600_residual_admission",),
                exit_events=(
                    "exact_n600_d_seg_pose_finish_latch",
                    "ncde_solve_basin",
                    *continuation_events,
                ),
                next_by_event={
                    "exact_n600_d_seg_pose_finish_latch": "pose_protected_finish",
                    "ncde_solve_basin": "terminal_solve",
                    "box_milestone_crossed": "costate_ranked_joint_continuation",
                    "exact_joint_delta_nonnegative_rollback": (
                        "costate_ranked_joint_continuation"
                    ),
                    "economic_or_dynamics_stop": "STOP",
                },
                active_group_policy=(
                    "scorer_recursive_costate_pareto_nondominance_stable_id"
                ),
                checkpoint_events=(
                    "exact_n600_d_seg_pose_finish_latch",
                    "ncde_solve_basin",
                    *continuation_events,
                ),
            ),
            EventNodeV1(
                node_id="pose_protected_finish",
                kind="continuation",
                entry_events=("exact_n600_d_seg_pose_finish_latch",),
                exit_events=("ncde_solve_basin", *continuation_events),
                next_by_event={
                    "ncde_solve_basin": "terminal_solve",
                    "box_milestone_crossed": "pose_protected_finish",
                    "exact_joint_delta_nonnegative_rollback": "pose_protected_finish",
                    "economic_or_dynamics_stop": "STOP",
                },
                active_group_policy=(
                    "scorer_recursive_costate_pareto_with_pose_trust_region"
                ),
                checkpoint_events=("ncde_solve_basin", *continuation_events),
            ),
            EventNodeV1(
                node_id="terminal_solve",
                kind="solve_interleave",
                entry_events=("ncde_solve_basin",),
                exit_events=("exact_solve_accepted", "solve_rollback_governed_stop"),
                next_by_event={
                    "exact_solve_accepted": "pose_protected_finish",
                    "solve_rollback_governed_stop": "STOP",
                },
                active_group_policy="ms2_metric_active_quotient_blocks",
                checkpoint_events=(
                    "exact_solve_accepted",
                    "solve_rollback_governed_stop",
                ),
                execution_enabled=False,
                blocker="DDM_MS2_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE",
            ),
        ),
        budget_caps=BudgetCapsV1(
            maximum_receiver_verdicts=maximum_receiver_verdicts,
            maximum_wall_seconds=maximum_wall_seconds,
            checkpoint_recovery_loss_verdicts=1,
            maximum_counted_bytes=maximum_counted_bytes,
        ),
        proposal_metric_selector="scorer_recursive_rank4_fisher_corrected_J",
        exact_acceptance_metric=(
            "100*d_seg_R+sqrt(10*d_pose_YUV6_R)+25*archive_bytes/37545489"
        ),
        box_tolerance_policy={
            "descent_box_role": "milestone_not_stop",
            "describe_solve_box_role": "tolerance_stop",
            "ms2r_role": "proposal_ordering_prior_until_describe_solve",
            "global_fallback_declared": True,
            "descent_continuation": "exact_receiver_realized_delta_S_lt_zero",
        },
        visibility_policy={
            "frame_0": "pose-only(frame_0)",
            "fine_chroma": "seg-only",
            "shared_visible": "joint",
            "resize_null": "gauge_fixed_out",
            "blind_coordinates": "excluded",
        },
        terminal_hooks={
            "fork_head_solve": {
                "event": "ws3_receiver_closed_start_bound",
                "execution_enabled": False,
                "blocker": "DDM_MATCHED_UPDATE_RMS_RECEIPT_MISSING",
            },
            "head_offset_solver": {
                "event": "ncde_solve_basin",
                "execution_enabled": False,
                "blocker": "DDM_HEAD_OFFSET_RECEIVER_CUSTODY_MISSING",
            },
            "ms2_terminal_solve": {
                "event": "ncde_solve_basin",
                "execution_enabled": False,
                "blocker": "DDM_MS2_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE",
            },
            "mc_finisher": {
                "event": "post_descent",
                "execution_enabled": False,
                "blocker": "PREREGISTRATION_ONLY",
            },
        },
        telemetry_fields=telemetry_fields,
        execution_allowed=execution_allowed,
    )


__all__ = [
    "CHARGE_AUDIT_SCHEMA",
    "EVENT_MARK_SCHEMA",
    "RATE_BREAK_EVEN",
    "SCHEMA",
    "AcquisitionObservationV1",
    "BudgetCapsV1",
    "ChargeSectionV1",
    "ContinuationStateV1",
    "DDMEventContinuationError",
    "DDMEventContinuationV1",
    "EventNodeV1",
    "VisibilityType",
    "audit_two_part_charge",
    "build_j8e_event_continuation",
    "pareto_acquisition_order",
]
