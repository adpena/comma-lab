# SPDX-License-Identifier: MIT
"""Exact finite-state allocation over already-evaluated PVSA archives.

G83 is a decision seam, not a scorer, archive generator, inverse solver, or
candidate claim.  It accepts a finite prefix-closed family of *exact archive
bytes* whose complete 600-sample ``upstream/evaluate.py`` component rows are
bound to one caller-supplied custody contract.  It then:

* validates conditional actuator prerequisites, conflicts, and execution order;
* prices every state from its own archive byte length and exact component row;
* applies the nonlinear contest score without per-component thresholds;
* Pareto-prunes only under same-axis, same-custody monotonic dominance;
* selects the globally cheapest exact state; and
* emits an executable add/remove/rollback route between exact states.

The current G80 wire realizes the zero-actuator semantic base and one G74
actuator.  :func:`exact_state_from_g80_build` and
:func:`exact_state_from_g82_lowering` bind those committed interfaces directly.
Future conditional actuator versions can use :func:`exact_archive_state` only
after their own exact archive bytes and full upstream row exist.  The allocator
never manufactures missing states or extrapolates component deltas.

``research_only=True`` is deliberate: this module invokes neither inflate nor
the evaluator and cannot move the canonical frontier pointer.  It can select
only among rows supplied by a caller after those external operations close.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Literal

from tac import score_geometry
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetSnapshot,
    score_sublevel_against_dynamic_frontier,
    score_transition_against_dynamic_frontier,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.taskspace_g82_tsppv2_pvsa1_lowering_v1 import (
    TSPPV2ToPVSA1LoweringV1,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    CompactActuatorTypeV1,
    CompactPVSAArchiveBuildV1,
)

SCHEMA: Final = "tac.taskspace_g83_pvsa_exact_archive_state_allocator.v1"
EVALUATOR_ENTRYPOINT: Final = "upstream/evaluate.py"
REFERENCE_BYTES: Final = score_geometry.CONTEST_REFERENCE_BYTES
FULL_SAMPLE_COUNT: Final = 600
FULL_OUTPUT_FRAME_COUNT: Final = 1200
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class G83ExactArchiveAllocatorError(ValueError):
    """The exact archive-state or its authority/custody contract is invalid."""


class ExactEvalAxisV1(StrEnum):
    CONTEST_CPU = "contest_cpu"
    CONTEST_CUDA = "contest_cuda"

    @property
    def evidence_grade(self) -> str:
        if self is ExactEvalAxisV1.CONTEST_CPU:
            return "[contest-CPU]"
        return "[contest-CUDA]"


class ArchiveStateOriginV1(StrEnum):
    G80_PVSA = "g80_pvsa"
    G82_LOWERING = "g82_lowering"
    UPSTREAM_EXACT_ARCHIVE = "upstream_exact_archive"


class ArchiveTransitionKindV1(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    ROLLBACK = "rollback"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise G83ExactArchiveAllocatorError(f"{label} must be canonical lowercase SHA-256")
    return value


def _require_id(value: object, *, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise G83ExactArchiveAllocatorError(f"{label} must be a bounded canonical identifier")
    return value


def _require_utc(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise G83ExactArchiveAllocatorError(f"{label} must be a timezone-aware UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise G83ExactArchiveAllocatorError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise G83ExactArchiveAllocatorError(f"{label} must be UTC")
    return value


def _require_finite_nonnegative(value: object, *, label: str, bounded_one: bool = False) -> float:
    if type(value) not in (int, float):
        raise G83ExactArchiveAllocatorError(f"{label} must be a finite numeric value")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or (bounded_one and numeric > 1.0):
        suffix = " in [0,1]" if bounded_one else " >= 0"
        raise G83ExactArchiveAllocatorError(f"{label} must be finite and{suffix}")
    return numeric


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _pvsa1_decoder_transition_id(actuator_type: CompactActuatorTypeV1) -> str:
    return f"pvsa1:{int(actuator_type)}:{actuator_type.name}"


@dataclass(frozen=True, slots=True)
class ExactEvaluationCustodyV1:
    """One immutable upstream-evaluation context shared by every state row."""

    axis: ExactEvalAxisV1
    hardware_substrate: str
    evaluator_source_sha256: str
    upstream_snapshot_sha256: str
    runtime_tree_sha256: str
    target_video_sha256: str
    file_list_sha256: str
    evaluation_context_id: str
    custody_epoch_id: str
    evaluator_entrypoint: Literal["upstream/evaluate.py"] = EVALUATOR_ENTRYPOINT
    sample_count: Literal[600] = FULL_SAMPLE_COUNT
    output_frame_count: Literal[1200] = FULL_OUTPUT_FRAME_COUNT
    reference_bytes: Literal[37_545_489] = REFERENCE_BYTES

    def __post_init__(self) -> None:
        if type(self.axis) is not ExactEvalAxisV1:
            raise G83ExactArchiveAllocatorError("custody axis must be an exact authority enum")
        for name in (
            "evaluator_source_sha256",
            "upstream_snapshot_sha256",
            "runtime_tree_sha256",
            "target_video_sha256",
            "file_list_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"custody.{name}")
        _require_id(self.evaluation_context_id, label="custody.evaluation_context_id")
        _require_id(self.custody_epoch_id, label="custody.custody_epoch_id")
        if type(self.hardware_substrate) is not str or not self.hardware_substrate.strip():
            raise G83ExactArchiveAllocatorError("custody hardware_substrate must be nonempty")
        if (
            self.evaluator_entrypoint != EVALUATOR_ENTRYPOINT
            or self.sample_count != FULL_SAMPLE_COUNT
            or self.output_frame_count != FULL_OUTPUT_FRAME_COUNT
            or self.reference_bytes != REFERENCE_BYTES
        ):
            raise G83ExactArchiveAllocatorError("custody is not the complete canonical n600 evaluator contract")

    @property
    def sha256(self) -> str:
        return _sha256(_canonical_json(asdict(self)))


@dataclass(frozen=True, slots=True)
class ExactUpstreamComponentRowV1:
    """Complete exact component row for one evaluated archive object."""

    archive_sha256: str
    archive_bytes: int
    d_seg: float
    d_pose: float
    final_score: float
    output_video_sha256: str
    evaluation_receipt_sha256: str
    custody_sha256: str
    measured_at_utc: str
    axis: ExactEvalAxisV1
    evidence_grade: str
    evaluator_entrypoint: Literal["upstream/evaluate.py"] = EVALUATOR_ENTRYPOINT
    sample_count: Literal[600] = FULL_SAMPLE_COUNT
    output_frame_count: Literal[1200] = FULL_OUTPUT_FRAME_COUNT
    reference_bytes: Literal[37_545_489] = REFERENCE_BYTES
    inflate_completed: Literal[True] = True
    evaluation_completed: Literal[True] = True
    proxy: Literal[False] = False
    advisory: Literal[False] = False

    def __post_init__(self) -> None:
        _require_sha256(self.archive_sha256, label="row.archive_sha256")
        _require_sha256(self.output_video_sha256, label="row.output_video_sha256")
        _require_sha256(self.evaluation_receipt_sha256, label="row.evaluation_receipt_sha256")
        _require_sha256(self.custody_sha256, label="row.custody_sha256")
        _require_utc(self.measured_at_utc, label="row.measured_at_utc")
        if type(self.archive_bytes) is not int or self.archive_bytes <= 0:
            raise G83ExactArchiveAllocatorError("row.archive_bytes must be an exact positive integer")
        d_seg = _require_finite_nonnegative(self.d_seg, label="row.d_seg", bounded_one=True)
        d_pose = _require_finite_nonnegative(self.d_pose, label="row.d_pose")
        final_score = _require_finite_nonnegative(self.final_score, label="row.final_score")
        if type(self.axis) is not ExactEvalAxisV1 or self.evidence_grade != self.axis.evidence_grade:
            raise G83ExactArchiveAllocatorError("row authority axis/evidence grade is inconsistent")
        if (
            self.evaluator_entrypoint != EVALUATOR_ENTRYPOINT
            or self.sample_count != FULL_SAMPLE_COUNT
            or self.output_frame_count != FULL_OUTPUT_FRAME_COUNT
            or self.reference_bytes != REFERENCE_BYTES
            or self.inflate_completed is not True
            or self.evaluation_completed is not True
            or self.proxy is not False
            or self.advisory is not False
        ):
            raise G83ExactArchiveAllocatorError("row is partial, proxy, advisory, or not upstream n600")
        recomputed = score_geometry.contest_score(
            d_seg,
            d_pose,
            self.archive_bytes,
            reference_bytes=self.reference_bytes,
        )
        if final_score != recomputed:
            raise G83ExactArchiveAllocatorError("row.final_score differs from exact component recomposition")


@dataclass(frozen=True, slots=True)
class ConditionalActuatorV1:
    """Typed state-machine declaration; no score threshold is attached."""

    actuator_id: str
    decoder_transition_id: str
    prerequisites: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.actuator_id, label="actuator_id")
        _require_id(self.decoder_transition_id, label="decoder_transition_id")
        if (
            type(self.prerequisites) is not tuple
            or type(self.conflicts) is not tuple
            or any(type(value) is not str for value in (*self.prerequisites, *self.conflicts))
        ):
            raise G83ExactArchiveAllocatorError("actuator prerequisites/conflicts must be exact tuples")
        for value in (*self.prerequisites, *self.conflicts):
            _require_id(value, label=f"actuator {self.actuator_id} relation")
        if (
            len(set(self.prerequisites)) != len(self.prerequisites)
            or len(set(self.conflicts)) != len(self.conflicts)
            or self.actuator_id in self.prerequisites
            or self.actuator_id in self.conflicts
            or set(self.prerequisites) & set(self.conflicts)
        ):
            raise G83ExactArchiveAllocatorError("actuator dependency/conflict relation is inconsistent")


@dataclass(frozen=True, slots=True)
class ExactArchiveStateV1:
    """One exact selected archive plus its complete exact evaluator row."""

    state_id: str
    selected_actuators: tuple[str, ...]
    selected_decoder_transition_ids: tuple[str, ...]
    archive_payload: bytes = field(repr=False)
    expected_archive_sha256: str
    component_row: ExactUpstreamComponentRowV1
    origin: ArchiveStateOriginV1
    archive_validation_receipt_sha256: str
    wire_contract_id: str

    def __post_init__(self) -> None:
        _require_id(self.state_id, label="state_id")
        _require_sha256(self.expected_archive_sha256, label="state.expected_archive_sha256")
        _require_sha256(
            self.archive_validation_receipt_sha256,
            label="state.archive_validation_receipt_sha256",
        )
        _require_id(self.wire_contract_id, label="state.wire_contract_id")
        if (
            type(self.selected_actuators) is not tuple
            or type(self.selected_decoder_transition_ids) is not tuple
            or any(
                type(value) is not str for value in (*self.selected_actuators, *self.selected_decoder_transition_ids)
            )
        ):
            raise G83ExactArchiveAllocatorError("state actuator and decoder-transition ids must be exact tuples")
        for value in (*self.selected_actuators, *self.selected_decoder_transition_ids):
            _require_id(value, label=f"state {self.state_id} actuator")
        if (
            len(self.selected_actuators) != len(self.selected_decoder_transition_ids)
            or len(set(self.selected_actuators)) != len(self.selected_actuators)
            or len(set(self.selected_decoder_transition_ids)) != len(self.selected_decoder_transition_ids)
        ):
            raise G83ExactArchiveAllocatorError("state actuator/decoder-transition identity is duplicated or unpaired")
        if type(self.archive_payload) is not bytes or not self.archive_payload:
            raise G83ExactArchiveAllocatorError("state archive payload must be nonempty exact bytes")
        if type(self.component_row) is not ExactUpstreamComponentRowV1:
            raise G83ExactArchiveAllocatorError("state lacks an exact upstream component row")
        if type(self.origin) is not ArchiveStateOriginV1:
            raise G83ExactArchiveAllocatorError("state origin must be an exact enum")
        actual_sha256 = _sha256(self.archive_payload)
        if (
            actual_sha256 != self.expected_archive_sha256
            or self.component_row.archive_sha256 != actual_sha256
            or self.component_row.archive_bytes != len(self.archive_payload)
        ):
            raise G83ExactArchiveAllocatorError("state bytes/SHA/component row do not bind the same archive object")

    @property
    def archive_sha256(self) -> str:
        return self.expected_archive_sha256

    @property
    def archive_bytes(self) -> int:
        return len(self.archive_payload)

    @property
    def exact_score(self) -> float:
        return self.component_row.final_score


def exact_archive_state(
    *,
    state_id: str,
    selected_actuators: tuple[str, ...],
    selected_decoder_transition_ids: tuple[str, ...],
    archive_payload: bytes,
    expected_archive_sha256: str,
    component_row: ExactUpstreamComponentRowV1,
    archive_validation_receipt_sha256: str,
    wire_contract_id: str,
) -> ExactArchiveStateV1:
    """Bind a future receiver-closed exact archive after upstream evaluation."""

    return ExactArchiveStateV1(
        state_id=state_id,
        selected_actuators=selected_actuators,
        selected_decoder_transition_ids=selected_decoder_transition_ids,
        archive_payload=archive_payload,
        expected_archive_sha256=expected_archive_sha256,
        component_row=component_row,
        origin=ArchiveStateOriginV1.UPSTREAM_EXACT_ARCHIVE,
        archive_validation_receipt_sha256=archive_validation_receipt_sha256,
        wire_contract_id=wire_contract_id,
    )


def exact_state_from_g80_build(
    *,
    state_id: str,
    selected_actuators: tuple[str, ...],
    build: CompactPVSAArchiveBuildV1,
    component_row: ExactUpstreamComponentRowV1,
    archive_validation_receipt_sha256: str,
) -> ExactArchiveStateV1:
    """Bind the exact G80 selected outer archive, never a synthetic byte sum."""

    if type(build) is not CompactPVSAArchiveBuildV1:
        raise G83ExactArchiveAllocatorError("G80 adapter requires CompactPVSAArchiveBuildV1")
    selected = build.outer_build.selected
    if len(selected_actuators) != len(build.selected.actuators):
        raise G83ExactArchiveAllocatorError("G80 actuator tuple length differs from parsed compact wire")
    return ExactArchiveStateV1(
        state_id=state_id,
        selected_actuators=selected_actuators,
        selected_decoder_transition_ids=tuple(
            _pvsa1_decoder_transition_id(row.actuator_type) for row in build.selected.actuators
        ),
        archive_payload=selected.archive_bytes,
        expected_archive_sha256=selected.archive_sha256,
        component_row=component_row,
        origin=ArchiveStateOriginV1.G80_PVSA,
        archive_validation_receipt_sha256=archive_validation_receipt_sha256,
        wire_contract_id=build.selected.wire_policy_id,
    )


def exact_state_from_g82_lowering(
    *,
    state_id: str,
    selected_actuators: tuple[str, ...],
    lowering: TSPPV2ToPVSA1LoweringV1,
    component_row: ExactUpstreamComponentRowV1,
    use_actuated: bool,
) -> ExactArchiveStateV1:
    """Bind either exact G82 zero-actuator baseline or one-actuator lowering."""

    if type(lowering) is not TSPPV2ToPVSA1LoweringV1 or type(use_actuated) is not bool:
        raise G83ExactArchiveAllocatorError("G82 adapter requires exact lowering and explicit state selector")
    build = lowering.compact_actuated if use_actuated else lowering.semantic_baseline
    expected_count = 1 if use_actuated else 0
    if len(selected_actuators) != expected_count:
        raise G83ExactArchiveAllocatorError("G82 selected_actuators disagrees with selected lowering state")
    selected = build.outer_build.selected
    return ExactArchiveStateV1(
        state_id=state_id,
        selected_actuators=selected_actuators,
        selected_decoder_transition_ids=tuple(
            _pvsa1_decoder_transition_id(row.actuator_type) for row in build.selected.actuators
        ),
        archive_payload=selected.archive_bytes,
        expected_archive_sha256=selected.archive_sha256,
        component_row=component_row,
        origin=ArchiveStateOriginV1.G82_LOWERING,
        archive_validation_receipt_sha256=lowering.receipt.sha256,
        wire_contract_id=build.selected.wire_policy_id,
    )


@dataclass(frozen=True, slots=True)
class ParetoDominanceV1:
    dominated_state_id: str
    dominating_state_id: str
    same_axis_and_custody: Literal[True] = True
    monotone_axes: Literal["d_seg,d_pose,archive_bytes"] = "d_seg,d_pose,archive_bytes"


@dataclass(frozen=True, slots=True)
class ExactArchiveTransitionV1:
    from_state_id: str
    to_state_id: str
    kind: ArchiveTransitionKindV1
    actuator_ids: tuple[str, ...]
    score_transition: score_geometry.ScoreTransitionAudit


@dataclass(frozen=True, slots=True)
class ActuatorDispositionV1:
    actuator_id: str
    globally_selected: bool
    local_add_deltas: tuple[float, ...]
    locally_beneficial_somewhere: bool
    locally_harmful_somewhere: bool
    classification: Literal[
        "locally_beneficial_globally_rejected",
        "locally_harmful_globally_selected",
        "globally_selected_without_local_paradox",
        "globally_rejected_without_local_paradox",
    ]


@dataclass(frozen=True, slots=True)
class ExactArchiveAllocationV1:
    """Deterministic global choice and exact state-machine route."""

    custody_sha256: str
    frontier_pointer_sha256: str
    frontier_target_score: float
    current_state_id: str
    selected_state_id: str
    selected_archive_sha256: str
    selected_archive_bytes: int
    selected_d_seg: float
    selected_d_pose: float
    selected_exact_score: float
    signed_frontier_slack: float
    beats_dynamic_frontier: bool
    pareto_frontier_state_ids: tuple[str, ...]
    dominated: tuple[ParetoDominanceV1, ...]
    route: tuple[ExactArchiveTransitionV1, ...]
    dispositions: tuple[ActuatorDispositionV1, ...]
    state_scores: tuple[tuple[str, float], ...]
    schema: Literal["tac.taskspace_g83_pvsa_exact_archive_state_allocator.v1"] = SCHEMA
    selection_rule: Literal["minimum_exact_nonlinear_score_then_archive_bytes_then_actuator_tuple_then_state_id"] = (
        "minimum_exact_nonlinear_score_then_archive_bytes_then_actuator_tuple_then_state_id"
    )
    component_thresholds_used: Literal[False] = False
    scorer_invoked: Literal[False] = False
    evaluator_invoked: Literal[False] = False
    pointer_moved: Literal[False] = False
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def to_receipt_bytes(self) -> bytes:
        """Serialize custody/decision evidence without embedding archive payloads."""

        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def _validate_actuator_registry(
    actuators: tuple[ConditionalActuatorV1, ...],
) -> tuple[dict[str, ConditionalActuatorV1], dict[str, int]]:
    if type(actuators) is not tuple or any(type(row) is not ConditionalActuatorV1 for row in actuators):
        raise G83ExactArchiveAllocatorError("actuators must be an exact typed tuple")
    ids = tuple(row.actuator_id for row in actuators)
    if len(set(ids)) != len(ids):
        raise G83ExactArchiveAllocatorError("actuator registry contains duplicate ids")
    rows = dict(zip(ids, actuators, strict=True))
    order = {actuator_id: index for index, actuator_id in enumerate(ids)}
    for row in actuators:
        unknown = (set(row.prerequisites) | set(row.conflicts)) - rows.keys()
        if unknown:
            raise G83ExactArchiveAllocatorError(f"actuator {row.actuator_id} references unknown relations")
        if any(order[parent] >= order[row.actuator_id] for parent in row.prerequisites):
            raise G83ExactArchiveAllocatorError(
                f"actuator {row.actuator_id} prerequisite is not earlier in decoder transition order"
            )
        for conflict in row.conflicts:
            if row.actuator_id not in rows[conflict].conflicts:
                raise G83ExactArchiveAllocatorError("actuator conflicts must be declared symmetrically")
    return rows, order


def _validate_state_machine(
    states: tuple[ExactArchiveStateV1, ...],
    *,
    actuator_rows: dict[str, ConditionalActuatorV1],
    actuator_order: dict[str, int],
) -> dict[tuple[str, ...], ExactArchiveStateV1]:
    if not states or any(type(row) is not ExactArchiveStateV1 for row in states):
        raise G83ExactArchiveAllocatorError("states must be a nonempty exact typed tuple")
    if len({row.state_id for row in states}) != len(states):
        raise G83ExactArchiveAllocatorError("state ids are not unique")
    by_selection: dict[tuple[str, ...], ExactArchiveStateV1] = {}
    for state in states:
        selected = state.selected_actuators
        if selected in by_selection:
            raise G83ExactArchiveAllocatorError("multiple archives claim the same selected actuator state")
        if any(actuator_id not in actuator_rows for actuator_id in selected):
            raise G83ExactArchiveAllocatorError(f"state {state.state_id} contains an unknown actuator")
        if tuple(sorted(selected, key=actuator_order.__getitem__)) != selected:
            raise G83ExactArchiveAllocatorError(f"state {state.state_id} violates decoder transition order")
        selected_set = set(selected)
        prefix: set[str] = set()
        for actuator_id, decoder_transition_id in zip(
            selected,
            state.selected_decoder_transition_ids,
            strict=True,
        ):
            row = actuator_rows[actuator_id]
            if decoder_transition_id != row.decoder_transition_id:
                raise G83ExactArchiveAllocatorError(
                    f"state {state.state_id} relabels the exact decoder transition for {actuator_id}"
                )
            if not set(row.prerequisites).issubset(prefix):
                raise G83ExactArchiveAllocatorError(
                    f"state {state.state_id} activates {actuator_id} before its prerequisites"
                )
            if selected_set & set(row.conflicts):
                raise G83ExactArchiveAllocatorError(f"state {state.state_id} contains conflicting actuators")
            prefix.add(actuator_id)
        by_selection[selected] = state
    if () not in by_selection:
        raise G83ExactArchiveAllocatorError("state family lacks the zero-actuator semantic baseline")
    for selection in tuple(by_selection):
        for stop in range(len(selection)):
            if selection[:stop] not in by_selection:
                raise G83ExactArchiveAllocatorError("state family is not prefix-closed for exact rollback")
    return by_selection


def _validate_custody(
    states: tuple[ExactArchiveStateV1, ...],
    *,
    custody: ExactEvaluationCustodyV1,
) -> None:
    if type(custody) is not ExactEvaluationCustodyV1:
        raise G83ExactArchiveAllocatorError("allocator requires caller-supplied exact custody")
    for state in states:
        row = state.component_row
        if row.custody_sha256 != custody.sha256:
            raise G83ExactArchiveAllocatorError(f"state {state.state_id} carries stale or foreign custody")
        if (
            row.axis is not custody.axis
            or row.evidence_grade != custody.axis.evidence_grade
            or row.evaluator_entrypoint != custody.evaluator_entrypoint
            or row.sample_count != custody.sample_count
            or row.output_frame_count != custody.output_frame_count
            or row.reference_bytes != custody.reference_bytes
        ):
            raise G83ExactArchiveAllocatorError(f"state {state.state_id} is cross-axis or incomplete")


def _dominates(left: ExactArchiveStateV1, right: ExactArchiveStateV1) -> bool:
    left_axes = (left.component_row.d_seg, left.component_row.d_pose, left.archive_bytes)
    right_axes = (right.component_row.d_seg, right.component_row.d_pose, right.archive_bytes)
    return all(a <= b for a, b in zip(left_axes, right_axes, strict=True)) and any(
        a < b for a, b in zip(left_axes, right_axes, strict=True)
    )


def _edge_kind(
    source: tuple[str, ...],
    target: tuple[str, ...],
) -> tuple[ArchiveTransitionKindV1, tuple[str, ...]] | None:
    if len(target) == len(source) + 1 and target[:-1] == source:
        return ArchiveTransitionKindV1.ADD, (target[-1],)
    if len(source) == len(target) + 1 and source[:-1] == target:
        return ArchiveTransitionKindV1.REMOVE, (source[-1],)
    if len(source) >= len(target) + 2 and source[: len(target)] == target:
        return ArchiveTransitionKindV1.ROLLBACK, source[len(target) :]
    return None


def _transition(
    source: ExactArchiveStateV1,
    target: ExactArchiveStateV1,
    *,
    kind: ArchiveTransitionKindV1,
    actuator_ids: tuple[str, ...],
    frontier: DynamicFrontierTargetSnapshot,
    now_utc_iso: str | None,
) -> ExactArchiveTransitionV1:
    audit = score_transition_against_dynamic_frontier(
        frontier,
        before_d_seg=source.component_row.d_seg,
        before_d_pose=source.component_row.d_pose,
        before_archive_bytes=source.archive_bytes,
        after_d_seg=target.component_row.d_seg,
        after_d_pose=target.component_row.d_pose,
        after_archive_bytes=target.archive_bytes,
        reference_bytes=REFERENCE_BYTES,
        now_utc_iso=now_utc_iso,
    )
    if audit.before_score != source.exact_score or audit.after_score != target.exact_score:
        raise G83ExactArchiveAllocatorError("transition score differs from exact component-row score")
    return ExactArchiveTransitionV1(
        from_state_id=source.state_id,
        to_state_id=target.state_id,
        kind=kind,
        actuator_ids=actuator_ids,
        score_transition=audit,
    )


def _route(
    current: ExactArchiveStateV1,
    selected: ExactArchiveStateV1,
    *,
    by_selection: dict[tuple[str, ...], ExactArchiveStateV1],
    frontier: DynamicFrontierTargetSnapshot,
    now_utc_iso: str | None,
) -> tuple[ExactArchiveTransitionV1, ...]:
    if current.state_id == selected.state_id:
        return ()
    selections = tuple(sorted(by_selection, key=lambda row: (len(row), row)))
    queue: deque[tuple[str, ...]] = deque((current.selected_actuators,))
    previous: dict[tuple[str, ...], tuple[tuple[str, ...], ArchiveTransitionKindV1, tuple[str, ...]] | None] = {
        current.selected_actuators: None
    }
    while queue:
        source = queue.popleft()
        if source == selected.selected_actuators:
            break
        neighbors: list[tuple[tuple[str, ...], ArchiveTransitionKindV1, tuple[str, ...]]] = []
        for target in selections:
            edge = _edge_kind(source, target)
            if edge is not None:
                neighbors.append((target, edge[0], edge[1]))
        neighbors.sort(key=lambda item: (item[1].value, len(item[0]), item[0]))
        for target, kind, actuator_ids in neighbors:
            if target not in previous:
                previous[target] = (source, kind, actuator_ids)
                queue.append(target)
    target_selection = selected.selected_actuators
    if target_selection not in previous:
        raise G83ExactArchiveAllocatorError("selected state is unreachable through exact add/remove/rollback states")
    reversed_edges: list[tuple[tuple[str, ...], tuple[str, ...], ArchiveTransitionKindV1, tuple[str, ...]]] = []
    cursor = target_selection
    while cursor != current.selected_actuators:
        predecessor = previous[cursor]
        if predecessor is None:  # pragma: no cover - guarded by cursor inequality
            raise G83ExactArchiveAllocatorError("route predecessor chain is corrupt")
        source, kind, actuator_ids = predecessor
        reversed_edges.append((source, cursor, kind, actuator_ids))
        cursor = source
    result: list[ExactArchiveTransitionV1] = []
    for source, target, kind, actuator_ids in reversed(reversed_edges):
        result.append(
            _transition(
                by_selection[source],
                by_selection[target],
                kind=kind,
                actuator_ids=actuator_ids,
                frontier=frontier,
                now_utc_iso=now_utc_iso,
            )
        )
    return tuple(result)


def allocate_exact_archive_state(
    *,
    states: tuple[ExactArchiveStateV1, ...],
    actuators: tuple[ConditionalActuatorV1, ...],
    current_state_id: str,
    custody: ExactEvaluationCustodyV1,
    frontier: DynamicFrontierTargetSnapshot,
    now_utc_iso: str | None = None,
) -> ExactArchiveAllocationV1:
    """Choose the globally minimum exact state under one fresh dynamic pointer."""

    verify_dynamic_frontier_target_snapshot(frontier, now_utc_iso=now_utc_iso)
    actuator_rows, actuator_order = _validate_actuator_registry(actuators)
    by_selection = _validate_state_machine(
        states,
        actuator_rows=actuator_rows,
        actuator_order=actuator_order,
    )
    _validate_custody(states, custody=custody)
    by_id = {state.state_id: state for state in states}
    if current_state_id not in by_id:
        raise G83ExactArchiveAllocatorError("current_state_id does not identify an exact state")

    sublevels: dict[str, score_geometry.ScoreSublevelAudit] = {}
    for state in states:
        sublevel = score_sublevel_against_dynamic_frontier(
            frontier,
            d_seg=state.component_row.d_seg,
            d_pose=state.component_row.d_pose,
            archive_bytes=state.archive_bytes,
            reference_bytes=REFERENCE_BYTES,
            now_utc_iso=now_utc_iso,
        )
        if sublevel.score != state.exact_score:
            raise G83ExactArchiveAllocatorError("dynamic sublevel score differs from exact component row")
        sublevels[state.state_id] = sublevel

    dominated: list[ParetoDominanceV1] = []
    pareto_states: list[ExactArchiveStateV1] = []
    for candidate in states:
        dominators = tuple(
            other for other in states if other.state_id != candidate.state_id and _dominates(other, candidate)
        )
        if dominators:
            winner = min(
                dominators,
                key=lambda row: (row.exact_score, row.archive_bytes, row.selected_actuators, row.state_id),
            )
            dominated.append(
                ParetoDominanceV1(
                    dominated_state_id=candidate.state_id,
                    dominating_state_id=winner.state_id,
                )
            )
        else:
            pareto_states.append(candidate)
    if not pareto_states:  # pragma: no cover - finite strict dominance always has a minimal element
        raise G83ExactArchiveAllocatorError("Pareto frontier unexpectedly empty")
    selected = min(
        pareto_states,
        key=lambda row: (row.exact_score, row.archive_bytes, row.selected_actuators, row.state_id),
    )

    add_deltas: dict[str, list[float]] = {row.actuator_id: [] for row in actuators}
    selections = tuple(by_selection)
    for source_selection in selections:
        for target_selection in selections:
            edge = _edge_kind(source_selection, target_selection)
            if edge is None or edge[0] is not ArchiveTransitionKindV1.ADD:
                continue
            source = by_selection[source_selection]
            target = by_selection[target_selection]
            add_deltas[edge[1][0]].append(target.exact_score - source.exact_score)
    dispositions: list[ActuatorDispositionV1] = []
    selected_ids = set(selected.selected_actuators)
    for row in actuators:
        deltas = tuple(add_deltas[row.actuator_id])
        beneficial = any(delta < 0.0 for delta in deltas)
        harmful = any(delta > 0.0 for delta in deltas)
        globally_selected = row.actuator_id in selected_ids
        if beneficial and not globally_selected:
            classification = "locally_beneficial_globally_rejected"
        elif harmful and globally_selected:
            classification = "locally_harmful_globally_selected"
        elif globally_selected:
            classification = "globally_selected_without_local_paradox"
        else:
            classification = "globally_rejected_without_local_paradox"
        dispositions.append(
            ActuatorDispositionV1(
                actuator_id=row.actuator_id,
                globally_selected=globally_selected,
                local_add_deltas=deltas,
                locally_beneficial_somewhere=beneficial,
                locally_harmful_somewhere=harmful,
                classification=classification,
            )
        )

    selected_sublevel = sublevels[selected.state_id]
    route = _route(
        by_id[current_state_id],
        selected,
        by_selection=by_selection,
        frontier=frontier,
        now_utc_iso=now_utc_iso,
    )
    verify_dynamic_frontier_target_snapshot(frontier, now_utc_iso=now_utc_iso)
    return ExactArchiveAllocationV1(
        custody_sha256=custody.sha256,
        frontier_pointer_sha256=frontier.pointer_sha256,
        frontier_target_score=frontier.target_score,
        current_state_id=current_state_id,
        selected_state_id=selected.state_id,
        selected_archive_sha256=selected.archive_sha256,
        selected_archive_bytes=selected.archive_bytes,
        selected_d_seg=selected.component_row.d_seg,
        selected_d_pose=selected.component_row.d_pose,
        selected_exact_score=selected.exact_score,
        signed_frontier_slack=selected_sublevel.signed_target_slack,
        beats_dynamic_frontier=selected_sublevel.inside_strict_sublevel,
        pareto_frontier_state_ids=tuple(
            row.state_id
            for row in sorted(
                pareto_states,
                key=lambda item: (item.exact_score, item.archive_bytes, item.selected_actuators, item.state_id),
            )
        ),
        dominated=tuple(
            sorted(
                dominated,
                key=lambda item: (item.dominated_state_id, item.dominating_state_id),
            )
        ),
        route=route,
        dispositions=tuple(dispositions),
        state_scores=tuple(sorted((row.state_id, row.exact_score) for row in states)),
    )


__all__ = [
    "SCHEMA",
    "ActuatorDispositionV1",
    "ArchiveStateOriginV1",
    "ArchiveTransitionKindV1",
    "ConditionalActuatorV1",
    "ExactArchiveAllocationV1",
    "ExactArchiveStateV1",
    "ExactArchiveTransitionV1",
    "ExactEvalAxisV1",
    "ExactEvaluationCustodyV1",
    "ExactUpstreamComponentRowV1",
    "G83ExactArchiveAllocatorError",
    "ParetoDominanceV1",
    "allocate_exact_archive_state",
    "exact_archive_state",
    "exact_state_from_g80_build",
    "exact_state_from_g82_lowering",
]
