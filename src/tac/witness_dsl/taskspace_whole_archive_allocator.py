# SPDX-License-Identifier: MIT
"""Whole-object nonlinear marginal admission for counted P/G/A[/T] bundles.

This module is deliberately an allocator, not a scorer or candidate builder.
Every state is rebuilt as one canonical monolithic member, priced under both
canonical outer ZIP encodings, decoded twice through an injected receiver, and
measured through an injected receipt-producing callback.  Admission delegates
the exact finite three-axis transition to the live dynamic-frontier wrapper.

The returned arithmetic is research-only.  It establishes neither contest
score authority nor candidate, evaluation, originality, or promotion status.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

from tac.score_geometry import CONTEST_REFERENCE_BYTES, ScoreTransitionAudit
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetSnapshot,
    score_transition_against_dynamic_frontier,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.taskspace_monolithic_pga_receiver import (
    TaskspaceMonolithicPGARole,
    build_taskspace_monolithic_pga_archive,
    parse_taskspace_monolithic_pga_member,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    DEFAULT_MAX_MEMBER_BYTES,
    OuterArchiveEncoding,
    ParsedTaskspaceOuterArchive,
    TaskspaceOuterArchiveBuild,
    parse_taskspace_outer_archive,
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PROPOSAL_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_RECEIPT_BYTES: Final = 16 * 1024 * 1024


class TaskspaceWholeArchiveAllocatorError(RuntimeError):
    """A proposal, archive, receiver, measurement, or pointer failed custody."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise TaskspaceWholeArchiveAllocatorError(f"{field_name} must be canonical lowercase SHA-256 hex")
    return value


def _require_positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise TaskspaceWholeArchiveAllocatorError(f"{field_name} must be an exact positive integer")
    return value


def _require_nonempty_bytes(value: object, *, field_name: str) -> bytes:
    if type(value) is not bytes or not value:
        raise TaskspaceWholeArchiveAllocatorError(f"{field_name} must be nonempty immutable bytes")
    return value


def _require_stage_id(value: object, *, field_name: str = "stage_id") -> str:
    if type(value) is not str or not value or len(value) > 256 or not value.isascii():
        raise TaskspaceWholeArchiveAllocatorError(f"{field_name} must be bounded nonempty ASCII")
    return value


def _require_finite_distance(value: object, *, field_name: str, bounded_one: bool) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise TaskspaceWholeArchiveAllocatorError(f"{field_name} must be a finite nonnegative float")
    if bounded_one and value > 1.0:
        raise TaskspaceWholeArchiveAllocatorError(f"{field_name} must not exceed one")
    return value


@dataclass(frozen=True, slots=True)
class TaskspaceSectionBundleV1:
    """Exact counted section bytes transformed by allocator proposals."""

    predictor_packet: bytes
    generative_correction_packet: bytes
    coupled_preimage_packet: bytes
    terminal_quotient_packet: bytes | None = None

    def __post_init__(self) -> None:
        _require_nonempty_bytes(self.predictor_packet, field_name="predictor_packet")
        _require_nonempty_bytes(
            self.generative_correction_packet,
            field_name="generative_correction_packet",
        )
        _require_nonempty_bytes(self.coupled_preimage_packet, field_name="coupled_preimage_packet")
        if self.terminal_quotient_packet is not None:
            _require_nonempty_bytes(
                self.terminal_quotient_packet,
                field_name="terminal_quotient_packet",
            )

    @property
    def role_payloads(self) -> tuple[tuple[TaskspaceMonolithicPGARole, bytes], ...]:
        rows: tuple[tuple[TaskspaceMonolithicPGARole, bytes], ...] = (
            (TaskspaceMonolithicPGARole.PREDICTOR, self.predictor_packet),
            (
                TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION,
                self.generative_correction_packet,
            ),
            (TaskspaceMonolithicPGARole.COUPLED_PREIMAGE, self.coupled_preimage_packet),
        )
        if self.terminal_quotient_packet is not None:
            rows += ((TaskspaceMonolithicPGARole.TERMINAL_QUOTIENT, self.terminal_quotient_packet),)
        return rows

    @property
    def raw_section_nbytes(self) -> int:
        return sum(len(payload) for _role, payload in self.role_payloads)

    @property
    def bundle_sha256(self) -> str:
        digest = hashlib.sha256()
        for role, payload in self.role_payloads:
            encoded_role = role.value.encode("ascii")
            digest.update(len(encoded_role).to_bytes(2, "big"))
            digest.update(encoded_role)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


TaskspaceProposalTransform: TypeAlias = Callable[[TaskspaceSectionBundleV1], TaskspaceSectionBundleV1]


@dataclass(frozen=True, slots=True)
class TaskspaceWholeArchiveProposalV1:
    """One ordered atomic transformation of the exact current section bundle."""

    proposal_id: str
    transform: TaskspaceProposalTransform

    def __post_init__(self) -> None:
        if type(self.proposal_id) is not str or _PROPOSAL_ID_RE.fullmatch(self.proposal_id) is None:
            raise TaskspaceWholeArchiveAllocatorError("proposal_id must be 1-128 ASCII alphanumeric/._:- characters")
        if not callable(self.transform):
            raise TaskspaceWholeArchiveAllocatorError("proposal transform must be callable")


@dataclass(frozen=True, slots=True)
class TaskspaceReceiverRequestV1:
    """Exact archive custody passed to the injected semantic receiver."""

    stage_id: str
    encoding: OuterArchiveEncoding
    archive_bytes: bytes
    archive_sha256: str
    archive_nbytes: int
    member_sha256: str
    member_nbytes: int

    def __post_init__(self) -> None:
        _require_stage_id(self.stage_id)
        if type(self.encoding) is not OuterArchiveEncoding:
            raise TaskspaceWholeArchiveAllocatorError("receiver encoding is outside the closed outer-codec set")
        _require_nonempty_bytes(self.archive_bytes, field_name="receiver archive_bytes")
        _require_sha256(self.archive_sha256, field_name="receiver archive_sha256")
        _require_positive_int(self.archive_nbytes, field_name="receiver archive_nbytes")
        _require_sha256(self.member_sha256, field_name="receiver member_sha256")
        _require_positive_int(self.member_nbytes, field_name="receiver member_nbytes")
        if len(self.archive_bytes) != self.archive_nbytes:
            raise TaskspaceWholeArchiveAllocatorError("receiver archive byte count mismatch")
        if _sha256(self.archive_bytes) != self.archive_sha256:
            raise TaskspaceWholeArchiveAllocatorError("receiver archive SHA-256 mismatch")


@dataclass(frozen=True, slots=True)
class TaskspaceReceiverReceiptV1:
    """Receiver-declared decoded-output identity bound to one exact archive."""

    stage_id: str
    encoding: OuterArchiveEncoding
    archive_sha256: str
    archive_nbytes: int
    member_sha256: str
    member_nbytes: int
    decoded_output_sha256: str
    decoded_output_nbytes: int
    receiver_receipt_payload: bytes

    def __post_init__(self) -> None:
        _require_stage_id(self.stage_id)
        if type(self.encoding) is not OuterArchiveEncoding:
            raise TaskspaceWholeArchiveAllocatorError("receiver receipt encoding is invalid")
        _require_sha256(self.archive_sha256, field_name="receipt archive_sha256")
        _require_positive_int(self.archive_nbytes, field_name="receipt archive_nbytes")
        _require_sha256(self.member_sha256, field_name="receipt member_sha256")
        _require_positive_int(self.member_nbytes, field_name="receipt member_nbytes")
        _require_sha256(self.decoded_output_sha256, field_name="decoded_output_sha256")
        _require_positive_int(self.decoded_output_nbytes, field_name="decoded_output_nbytes")
        payload = _require_nonempty_bytes(
            self.receiver_receipt_payload,
            field_name="receiver_receipt_payload",
        )
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise TaskspaceWholeArchiveAllocatorError("receiver receipt payload exceeds the bounded limit")

    @property
    def receiver_receipt_sha256(self) -> str:
        return _sha256(self.receiver_receipt_payload)


TaskspaceReceiverCallback: TypeAlias = Callable[[TaskspaceReceiverRequestV1], TaskspaceReceiverReceiptV1]


@dataclass(frozen=True, slots=True)
class TaskspaceMeasurementRequestV1:
    """Selected exact archive and decoded identity presented for measurement."""

    stage_id: str
    selected_encoding: OuterArchiveEncoding
    archive_bytes: bytes
    archive_sha256: str
    archive_nbytes: int
    member_sha256: str
    member_nbytes: int
    decoded_output_sha256: str
    decoded_output_nbytes: int
    receiver_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_stage_id(self.stage_id)
        if type(self.selected_encoding) is not OuterArchiveEncoding:
            raise TaskspaceWholeArchiveAllocatorError("measurement encoding is invalid")
        _require_nonempty_bytes(self.archive_bytes, field_name="measurement archive_bytes")
        _require_sha256(self.archive_sha256, field_name="measurement archive_sha256")
        _require_positive_int(self.archive_nbytes, field_name="measurement archive_nbytes")
        _require_sha256(self.member_sha256, field_name="measurement member_sha256")
        _require_positive_int(self.member_nbytes, field_name="measurement member_nbytes")
        _require_sha256(self.decoded_output_sha256, field_name="measurement decoded_output_sha256")
        _require_positive_int(self.decoded_output_nbytes, field_name="measurement decoded_output_nbytes")
        _require_sha256(self.receiver_receipt_sha256, field_name="measurement receiver_receipt_sha256")
        if len(self.archive_bytes) != self.archive_nbytes:
            raise TaskspaceWholeArchiveAllocatorError("measurement archive byte count mismatch")
        if _sha256(self.archive_bytes) != self.archive_sha256:
            raise TaskspaceWholeArchiveAllocatorError("measurement archive SHA-256 mismatch")


@dataclass(frozen=True, slots=True)
class TaskspaceRealizedMeasurementReceiptV1:
    """Realized component distances bound to the selected archive/output object."""

    stage_id: str
    archive_sha256: str
    archive_nbytes: int
    member_sha256: str
    member_nbytes: int
    decoded_output_sha256: str
    decoded_output_nbytes: int
    receiver_receipt_sha256: str
    d_seg: float
    d_pose: float
    measurement_receipt_payload: bytes

    def __post_init__(self) -> None:
        _require_stage_id(self.stage_id)
        _require_sha256(self.archive_sha256, field_name="measurement receipt archive_sha256")
        _require_positive_int(self.archive_nbytes, field_name="measurement receipt archive_nbytes")
        _require_sha256(self.member_sha256, field_name="measurement receipt member_sha256")
        _require_positive_int(self.member_nbytes, field_name="measurement receipt member_nbytes")
        _require_sha256(
            self.decoded_output_sha256,
            field_name="measurement receipt decoded_output_sha256",
        )
        _require_positive_int(
            self.decoded_output_nbytes,
            field_name="measurement receipt decoded_output_nbytes",
        )
        _require_sha256(
            self.receiver_receipt_sha256,
            field_name="measurement receipt receiver_receipt_sha256",
        )
        _require_finite_distance(self.d_seg, field_name="d_seg", bounded_one=True)
        _require_finite_distance(self.d_pose, field_name="d_pose", bounded_one=False)
        payload = _require_nonempty_bytes(
            self.measurement_receipt_payload,
            field_name="measurement_receipt_payload",
        )
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise TaskspaceWholeArchiveAllocatorError("measurement receipt payload exceeds the bounded limit")

    @property
    def measurement_receipt_sha256(self) -> str:
        return _sha256(self.measurement_receipt_payload)


TaskspaceMeasurementCallback: TypeAlias = Callable[
    [TaskspaceMeasurementRequestV1],
    TaskspaceRealizedMeasurementReceiptV1,
]


@dataclass(frozen=True, slots=True)
class TaskspaceWholeArchiveStateV1:
    """One fully rebuilt, decoded, selected, and measured whole-object state."""

    stage_id: str
    bundle: TaskspaceSectionBundleV1
    archive_build: TaskspaceOuterArchiveBuild
    stored_receiver_receipt: TaskspaceReceiverReceiptV1
    deflated_receiver_receipt: TaskspaceReceiverReceiptV1
    selected_receiver_receipt: TaskspaceReceiverReceiptV1
    measurement_receipt: TaskspaceRealizedMeasurementReceiptV1

    @property
    def raw_section_nbytes(self) -> int:
        return self.bundle.raw_section_nbytes

    @property
    def member_nbytes(self) -> int:
        return self.archive_build.selected.member_nbytes

    @property
    def selected_archive_nbytes(self) -> int:
        return self.archive_build.selected.archive_nbytes


@dataclass(frozen=True, slots=True)
class TaskspaceSectionDeltaV1:
    """Raw uncompressed section-size movement for one logical role."""

    role: TaskspaceMonolithicPGARole
    before_nbytes: int
    after_nbytes: int
    delta_nbytes: int


@dataclass(frozen=True, slots=True)
class TaskspaceWholeArchiveProposalAuditV1:
    """Exact trial, nonlinear transition, and optional committed-prefix rebuild."""

    proposal_id: str
    proposal_index: int
    accepted: bool
    decision: Literal["accepted_exact_total_delta_negative", "rejected_exact_total_delta_nonnegative"]
    before_stage_id: str
    trial_state: TaskspaceWholeArchiveStateV1
    accepted_prefix_state: TaskspaceWholeArchiveStateV1 | None
    section_deltas: tuple[TaskspaceSectionDeltaV1, ...]
    raw_section_bytes_delta: int
    monolithic_member_bytes_delta: int
    stored_archive_bytes_delta: int
    deflated_archive_bytes_delta: int
    selected_archive_bytes_delta: int
    before_selected_encoding: OuterArchiveEncoding
    trial_selected_encoding: OuterArchiveEncoding
    score_transition: ScoreTransitionAudit


@dataclass(frozen=True, slots=True)
class TaskspaceWholeArchiveAllocatorTruthV1:
    """Non-permissive authority labels for the allocator result."""

    authority: Literal["derived_arithmetic_research_only"] = field(
        default="derived_arithmetic_research_only",
        init=False,
    )
    research_only: bool = field(default=True, init=False)
    candidate_claim: bool = field(default=False, init=False)
    score_claim: bool = field(default=False, init=False)
    evaluation_claim: bool = field(default=False, init=False)
    originality_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    scorer_dispatch_performed_by_allocator: bool = field(default=False, init=False)
    pointer_moved: bool = field(default=False, init=False)
    global_optimality_claim: bool = field(default=False, init=False)
    sign_reversal_safe: bool = field(default=False, init=False)
    caller_order_is_search_prior: bool = field(default=True, init=False)
    endpoint_proposal_generator_only: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class TaskspaceWholeArchiveAllocationV1:
    """Ordered greedy proposal trace under exact current-prefix rebuilding.

    This is deliberately not a global-selection result.  Exact G14 evidence
    contains composition sign reversals, so a proposal rejected on one prefix
    may become beneficial in a different joint context.  Consumers may harvest
    these states as endpoint proposals, but only a complete same-base action
    universe (or admissible branch-and-bound closure) may select globally.
    """

    frontier_snapshot: DynamicFrontierTargetSnapshot
    baseline_state: TaskspaceWholeArchiveStateV1
    final_state: TaskspaceWholeArchiveStateV1
    proposal_audits: tuple[TaskspaceWholeArchiveProposalAuditV1, ...]
    accepted_proposal_ids: tuple[str, ...]
    rejected_proposal_ids: tuple[str, ...]
    truth: TaskspaceWholeArchiveAllocatorTruthV1


def _strict_parse_bundle_archive(
    exact: ParsedTaskspaceOuterArchive,
    bundle: TaskspaceSectionBundleV1,
    *,
    max_member_bytes: int,
) -> ParsedTaskspaceOuterArchive:
    parsed = parse_taskspace_outer_archive(
        exact.archive_bytes,
        expected_encoding=exact.encoding,
        expected_archive_sha256=exact.archive_sha256,
        expected_member_sha256=exact.member_sha256,
        max_member_bytes=max_member_bytes,
    )
    member = parse_taskspace_monolithic_pga_member(
        parsed.member_bytes,
        max_member_bytes=max_member_bytes,
    )
    observed = tuple((section.role, section.payload) for section in member.sections)
    if observed != bundle.role_payloads:
        raise TaskspaceWholeArchiveAllocatorError("strict monolithic parse-back changed P/G/A[/T] section identity")
    return parsed


def _receiver_request(stage_id: str, parsed: ParsedTaskspaceOuterArchive) -> TaskspaceReceiverRequestV1:
    return TaskspaceReceiverRequestV1(
        stage_id=stage_id,
        encoding=parsed.encoding,
        archive_bytes=parsed.archive_bytes,
        archive_sha256=parsed.archive_sha256,
        archive_nbytes=parsed.archive_nbytes,
        member_sha256=parsed.member_sha256,
        member_nbytes=parsed.member_nbytes,
    )


def _bind_receiver_receipt(
    request: TaskspaceReceiverRequestV1,
    receipt: object,
) -> TaskspaceReceiverReceiptV1:
    if type(receipt) is not TaskspaceReceiverReceiptV1:
        raise TaskspaceWholeArchiveAllocatorError("receiver callback must return exact TaskspaceReceiverReceiptV1")
    expected = (
        request.stage_id,
        request.encoding,
        request.archive_sha256,
        request.archive_nbytes,
        request.member_sha256,
        request.member_nbytes,
    )
    observed = (
        receipt.stage_id,
        receipt.encoding,
        receipt.archive_sha256,
        receipt.archive_nbytes,
        receipt.member_sha256,
        receipt.member_nbytes,
    )
    if observed != expected:
        raise TaskspaceWholeArchiveAllocatorError("receiver receipt is not bound to the exact requested archive/member")
    return receipt


def _decode_twice(
    request: TaskspaceReceiverRequestV1,
    *,
    receiver_callback: TaskspaceReceiverCallback,
    frontier_snapshot: DynamicFrontierTargetSnapshot,
    now_utc_iso: str | None,
) -> TaskspaceReceiverReceiptV1:
    verify_dynamic_frontier_target_snapshot(frontier_snapshot, now_utc_iso=now_utc_iso)
    first = _bind_receiver_receipt(request, receiver_callback(request))
    verify_dynamic_frontier_target_snapshot(frontier_snapshot, now_utc_iso=now_utc_iso)
    second = _bind_receiver_receipt(request, receiver_callback(request))
    verify_dynamic_frontier_target_snapshot(frontier_snapshot, now_utc_iso=now_utc_iso)
    if second != first:
        raise TaskspaceWholeArchiveAllocatorError("receiver double-decode was not byte-identical and receipt-identical")
    return first


def _measurement_request(
    stage_id: str,
    selected: ParsedTaskspaceOuterArchive,
    receiver_receipt: TaskspaceReceiverReceiptV1,
) -> TaskspaceMeasurementRequestV1:
    return TaskspaceMeasurementRequestV1(
        stage_id=stage_id,
        selected_encoding=selected.encoding,
        archive_bytes=selected.archive_bytes,
        archive_sha256=selected.archive_sha256,
        archive_nbytes=selected.archive_nbytes,
        member_sha256=selected.member_sha256,
        member_nbytes=selected.member_nbytes,
        decoded_output_sha256=receiver_receipt.decoded_output_sha256,
        decoded_output_nbytes=receiver_receipt.decoded_output_nbytes,
        receiver_receipt_sha256=receiver_receipt.receiver_receipt_sha256,
    )


def _bind_measurement_receipt(
    request: TaskspaceMeasurementRequestV1,
    receipt: object,
) -> TaskspaceRealizedMeasurementReceiptV1:
    if type(receipt) is not TaskspaceRealizedMeasurementReceiptV1:
        raise TaskspaceWholeArchiveAllocatorError(
            "measurement callback must return exact TaskspaceRealizedMeasurementReceiptV1"
        )
    expected = (
        request.stage_id,
        request.archive_sha256,
        request.archive_nbytes,
        request.member_sha256,
        request.member_nbytes,
        request.decoded_output_sha256,
        request.decoded_output_nbytes,
        request.receiver_receipt_sha256,
    )
    observed = (
        receipt.stage_id,
        receipt.archive_sha256,
        receipt.archive_nbytes,
        receipt.member_sha256,
        receipt.member_nbytes,
        receipt.decoded_output_sha256,
        receipt.decoded_output_nbytes,
        receipt.receiver_receipt_sha256,
    )
    if observed != expected:
        raise TaskspaceWholeArchiveAllocatorError(
            "measurement receipt is not bound to the exact selected archive and decoded output"
        )
    return receipt


def _evaluate_state(
    bundle: TaskspaceSectionBundleV1,
    *,
    stage_id: str,
    frontier_snapshot: DynamicFrontierTargetSnapshot,
    receiver_callback: TaskspaceReceiverCallback,
    measurement_callback: TaskspaceMeasurementCallback,
    max_member_bytes: int,
    now_utc_iso: str | None,
) -> TaskspaceWholeArchiveStateV1:
    verify_dynamic_frontier_target_snapshot(frontier_snapshot, now_utc_iso=now_utc_iso)
    build = build_taskspace_monolithic_pga_archive(
        bundle.predictor_packet,
        bundle.generative_correction_packet,
        bundle.coupled_preimage_packet,
        terminal_quotient_packet=bundle.terminal_quotient_packet,
        max_member_bytes=max_member_bytes,
    )
    stored = _strict_parse_bundle_archive(build.stored, bundle, max_member_bytes=max_member_bytes)
    deflated = _strict_parse_bundle_archive(build.deflated, bundle, max_member_bytes=max_member_bytes)
    if stored.member_sha256 != deflated.member_sha256 or stored.member_bytes != deflated.member_bytes:
        raise TaskspaceWholeArchiveAllocatorError("STORE and DEFLATE decode to different monolithic members")

    stored_receipt = _decode_twice(
        _receiver_request(stage_id, stored),
        receiver_callback=receiver_callback,
        frontier_snapshot=frontier_snapshot,
        now_utc_iso=now_utc_iso,
    )
    deflated_receipt = _decode_twice(
        _receiver_request(stage_id, deflated),
        receiver_callback=receiver_callback,
        frontier_snapshot=frontier_snapshot,
        now_utc_iso=now_utc_iso,
    )
    if (
        stored_receipt.decoded_output_sha256,
        stored_receipt.decoded_output_nbytes,
    ) != (
        deflated_receipt.decoded_output_sha256,
        deflated_receipt.decoded_output_nbytes,
    ):
        raise TaskspaceWholeArchiveAllocatorError(
            "STORE and DEFLATE receiver outputs have different decoded identities"
        )

    selected_receipt = stored_receipt if build.selected.encoding is OuterArchiveEncoding.STORED else deflated_receipt
    request = _measurement_request(stage_id, build.selected, selected_receipt)
    verify_dynamic_frontier_target_snapshot(frontier_snapshot, now_utc_iso=now_utc_iso)
    measurement = _bind_measurement_receipt(request, measurement_callback(request))
    verify_dynamic_frontier_target_snapshot(frontier_snapshot, now_utc_iso=now_utc_iso)
    return TaskspaceWholeArchiveStateV1(
        stage_id=stage_id,
        bundle=bundle,
        archive_build=build,
        stored_receiver_receipt=stored_receipt,
        deflated_receiver_receipt=deflated_receipt,
        selected_receiver_receipt=selected_receipt,
        measurement_receipt=measurement,
    )


def _finite_transition(
    before: TaskspaceWholeArchiveStateV1,
    after: TaskspaceWholeArchiveStateV1,
    *,
    frontier_snapshot: DynamicFrontierTargetSnapshot,
    reference_bytes: int,
    now_utc_iso: str | None,
) -> ScoreTransitionAudit:
    transition = score_transition_against_dynamic_frontier(
        frontier_snapshot,
        before_d_seg=before.measurement_receipt.d_seg,
        before_d_pose=before.measurement_receipt.d_pose,
        before_archive_bytes=before.selected_archive_nbytes,
        after_d_seg=after.measurement_receipt.d_seg,
        after_d_pose=after.measurement_receipt.d_pose,
        after_archive_bytes=after.selected_archive_nbytes,
        reference_bytes=reference_bytes,
        now_utc_iso=now_utc_iso,
    )
    finite_fields = (
        transition.before_score,
        transition.after_score,
        transition.seg_term_delta,
        transition.pose_term_delta,
        transition.rate_term_delta,
        transition.exact_score_delta,
        transition.exact_score_improvement,
        transition.after_signed_target_slack,
    )
    if any(not math.isfinite(value) for value in finite_fields):
        raise TaskspaceWholeArchiveAllocatorError("score transition contains a non-finite exact field")
    optional_fields = (
        transition.linearized_score_delta_at_before,
        transition.nonlinear_remainder,
    )
    if any(value is not None and not math.isfinite(value) for value in optional_fields):
        raise TaskspaceWholeArchiveAllocatorError("score transition contains a non-finite diagnostic field")
    if transition.improves_score != (transition.exact_score_delta < 0.0):
        raise TaskspaceWholeArchiveAllocatorError("canonical transition admission flag is inconsistent")
    return transition


def _require_repeat_measurement(
    trial: TaskspaceWholeArchiveStateV1,
    prefix: TaskspaceWholeArchiveStateV1,
) -> None:
    trial_identity = (
        trial.bundle,
        trial.archive_build,
        trial.stored_receiver_receipt.encoding,
        trial.stored_receiver_receipt.archive_sha256,
        trial.stored_receiver_receipt.decoded_output_sha256,
        trial.deflated_receiver_receipt.encoding,
        trial.deflated_receiver_receipt.archive_sha256,
        trial.deflated_receiver_receipt.decoded_output_sha256,
        trial.measurement_receipt.archive_sha256,
        trial.measurement_receipt.member_sha256,
        trial.measurement_receipt.decoded_output_sha256,
        trial.measurement_receipt.d_seg,
        trial.measurement_receipt.d_pose,
    )
    prefix_identity = (
        prefix.bundle,
        prefix.archive_build,
        prefix.stored_receiver_receipt.encoding,
        prefix.stored_receiver_receipt.archive_sha256,
        prefix.stored_receiver_receipt.decoded_output_sha256,
        prefix.deflated_receiver_receipt.encoding,
        prefix.deflated_receiver_receipt.archive_sha256,
        prefix.deflated_receiver_receipt.decoded_output_sha256,
        prefix.measurement_receipt.archive_sha256,
        prefix.measurement_receipt.member_sha256,
        prefix.measurement_receipt.decoded_output_sha256,
        prefix.measurement_receipt.d_seg,
        prefix.measurement_receipt.d_pose,
    )
    if prefix_identity != trial_identity:
        raise TaskspaceWholeArchiveAllocatorError(
            "accepted-prefix rebuild/re-measurement drifted from the admitted trial"
        )


def _section_deltas(
    before: TaskspaceSectionBundleV1,
    after: TaskspaceSectionBundleV1,
) -> tuple[TaskspaceSectionDeltaV1, ...]:
    before_rows = dict(before.role_payloads)
    after_rows = dict(after.role_payloads)
    roles = tuple(dict.fromkeys((*before_rows, *after_rows)))
    return tuple(
        TaskspaceSectionDeltaV1(
            role=role,
            before_nbytes=len(before_rows.get(role, b"")),
            after_nbytes=len(after_rows.get(role, b"")),
            delta_nbytes=len(after_rows.get(role, b"")) - len(before_rows.get(role, b"")),
        )
        for role in roles
    )


def allocate_taskspace_whole_archive(
    baseline_bundle: TaskspaceSectionBundleV1,
    proposals: Sequence[TaskspaceWholeArchiveProposalV1],
    *,
    frontier_snapshot: DynamicFrontierTargetSnapshot,
    receiver_callback: TaskspaceReceiverCallback,
    measurement_callback: TaskspaceMeasurementCallback,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    now_utc_iso: str | None = None,
) -> TaskspaceWholeArchiveAllocationV1:
    """Greedily trace ordered bundle transforms with negative exact total delta.

    A rejected trial never becomes the current prefix.  An accepted trial is
    rebuilt, double-decoded, and remeasured before it becomes the input to the
    next proposal.  The dynamic pointer is reopened around every external
    callback and again inside every canonical score transition.  The result is
    an order-dependent acquisition trace, not an optimum: sign-reversing joint
    bundles must be rebuilt as same-base endpoints and arbitrated by the G33
    complete-universe controller.
    """

    if type(baseline_bundle) is not TaskspaceSectionBundleV1:
        raise TaskspaceWholeArchiveAllocatorError("baseline_bundle has the wrong exact type")
    if type(frontier_snapshot) is not DynamicFrontierTargetSnapshot:
        raise TaskspaceWholeArchiveAllocatorError("frontier_snapshot has the wrong exact type")
    if not callable(receiver_callback) or not callable(measurement_callback):
        raise TaskspaceWholeArchiveAllocatorError("receiver and measurement callbacks must be callable")
    _require_positive_int(reference_bytes, field_name="reference_bytes")
    _require_positive_int(max_member_bytes, field_name="max_member_bytes")
    try:
        ordered = tuple(proposals)
    except TypeError as exc:
        raise TaskspaceWholeArchiveAllocatorError("proposals must be a finite sequence") from exc
    if any(type(proposal) is not TaskspaceWholeArchiveProposalV1 for proposal in ordered):
        raise TaskspaceWholeArchiveAllocatorError("every proposal must have exact proposal type")
    proposal_ids = tuple(proposal.proposal_id for proposal in ordered)
    if len(set(proposal_ids)) != len(proposal_ids):
        raise TaskspaceWholeArchiveAllocatorError("duplicate proposal IDs are forbidden")

    verify_dynamic_frontier_target_snapshot(frontier_snapshot, now_utc_iso=now_utc_iso)
    baseline = _evaluate_state(
        baseline_bundle,
        stage_id="baseline",
        frontier_snapshot=frontier_snapshot,
        receiver_callback=receiver_callback,
        measurement_callback=measurement_callback,
        max_member_bytes=max_member_bytes,
        now_utc_iso=now_utc_iso,
    )
    current = baseline
    audits: list[TaskspaceWholeArchiveProposalAuditV1] = []
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []

    for proposal_index, proposal in enumerate(ordered):
        before = current
        before_bundle_identity = before.bundle.bundle_sha256
        try:
            trial_bundle = proposal.transform(before.bundle)
        except Exception as exc:
            raise TaskspaceWholeArchiveAllocatorError(f"proposal {proposal.proposal_id!r} transform failed") from exc
        if type(trial_bundle) is not TaskspaceSectionBundleV1:
            raise TaskspaceWholeArchiveAllocatorError(
                f"proposal {proposal.proposal_id!r} returned the wrong exact bundle type"
            )
        if before.bundle.bundle_sha256 != before_bundle_identity:
            raise TaskspaceWholeArchiveAllocatorError(
                f"proposal {proposal.proposal_id!r} mutated the current prefix in place"
            )

        trial = _evaluate_state(
            trial_bundle,
            stage_id=f"proposal_trial:{proposal_index}:{proposal.proposal_id}",
            frontier_snapshot=frontier_snapshot,
            receiver_callback=receiver_callback,
            measurement_callback=measurement_callback,
            max_member_bytes=max_member_bytes,
            now_utc_iso=now_utc_iso,
        )
        trial_transition = _finite_transition(
            before,
            trial,
            frontier_snapshot=frontier_snapshot,
            reference_bytes=reference_bytes,
            now_utc_iso=now_utc_iso,
        )
        accepted = trial_transition.exact_score_delta < 0.0
        prefix: TaskspaceWholeArchiveStateV1 | None = None
        transition = trial_transition
        if accepted:
            prefix = _evaluate_state(
                trial_bundle,
                stage_id=f"accepted_prefix:{len(accepted_ids)}:{proposal.proposal_id}",
                frontier_snapshot=frontier_snapshot,
                receiver_callback=receiver_callback,
                measurement_callback=measurement_callback,
                max_member_bytes=max_member_bytes,
                now_utc_iso=now_utc_iso,
            )
            _require_repeat_measurement(trial, prefix)
            transition = _finite_transition(
                before,
                prefix,
                frontier_snapshot=frontier_snapshot,
                reference_bytes=reference_bytes,
                now_utc_iso=now_utc_iso,
            )
            if transition.exact_score_delta >= 0.0:
                raise TaskspaceWholeArchiveAllocatorError(
                    "accepted-prefix rebuild no longer has negative exact total delta"
                )
            current = prefix
            accepted_ids.append(proposal.proposal_id)
            decision: Literal[
                "accepted_exact_total_delta_negative",
                "rejected_exact_total_delta_nonnegative",
            ] = "accepted_exact_total_delta_negative"
        else:
            rejected_ids.append(proposal.proposal_id)
            decision = "rejected_exact_total_delta_nonnegative"

        deltas = _section_deltas(before.bundle, trial.bundle)
        audits.append(
            TaskspaceWholeArchiveProposalAuditV1(
                proposal_id=proposal.proposal_id,
                proposal_index=proposal_index,
                accepted=accepted,
                decision=decision,
                before_stage_id=before.stage_id,
                trial_state=trial,
                accepted_prefix_state=prefix,
                section_deltas=deltas,
                raw_section_bytes_delta=sum(row.delta_nbytes for row in deltas),
                monolithic_member_bytes_delta=trial.member_nbytes - before.member_nbytes,
                stored_archive_bytes_delta=(
                    trial.archive_build.stored.archive_nbytes - before.archive_build.stored.archive_nbytes
                ),
                deflated_archive_bytes_delta=(
                    trial.archive_build.deflated.archive_nbytes - before.archive_build.deflated.archive_nbytes
                ),
                selected_archive_bytes_delta=(trial.selected_archive_nbytes - before.selected_archive_nbytes),
                before_selected_encoding=before.archive_build.selected.encoding,
                trial_selected_encoding=trial.archive_build.selected.encoding,
                score_transition=transition,
            )
        )

    verify_dynamic_frontier_target_snapshot(frontier_snapshot, now_utc_iso=now_utc_iso)
    return TaskspaceWholeArchiveAllocationV1(
        frontier_snapshot=frontier_snapshot,
        baseline_state=baseline,
        final_state=current,
        proposal_audits=tuple(audits),
        accepted_proposal_ids=tuple(accepted_ids),
        rejected_proposal_ids=tuple(rejected_ids),
        truth=TaskspaceWholeArchiveAllocatorTruthV1(),
    )


__all__ = [
    "TaskspaceMeasurementCallback",
    "TaskspaceMeasurementRequestV1",
    "TaskspaceProposalTransform",
    "TaskspaceRealizedMeasurementReceiptV1",
    "TaskspaceReceiverCallback",
    "TaskspaceReceiverReceiptV1",
    "TaskspaceReceiverRequestV1",
    "TaskspaceSectionBundleV1",
    "TaskspaceSectionDeltaV1",
    "TaskspaceWholeArchiveAllocationV1",
    "TaskspaceWholeArchiveAllocatorError",
    "TaskspaceWholeArchiveAllocatorTruthV1",
    "TaskspaceWholeArchiveProposalAuditV1",
    "TaskspaceWholeArchiveProposalV1",
    "TaskspaceWholeArchiveStateV1",
    "allocate_taskspace_whole_archive",
]
