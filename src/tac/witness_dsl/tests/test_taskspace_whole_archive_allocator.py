# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from tac.canonical_frontier_pointer import POINTER_SCHEMA_VERSION
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    load_dynamic_frontier_target,
)
from tac.witness_dsl.taskspace_monolithic_pga_receiver import (
    TaskspaceMonolithicPGARole,
    parse_taskspace_monolithic_pga_member,
)
from tac.witness_dsl.taskspace_outer_archive_codec import parse_taskspace_outer_archive
from tac.witness_dsl.taskspace_whole_archive_allocator import (
    TaskspaceMeasurementRequestV1,
    TaskspaceProposalTransform,
    TaskspaceRealizedMeasurementReceiptV1,
    TaskspaceReceiverReceiptV1,
    TaskspaceReceiverRequestV1,
    TaskspaceSectionBundleV1,
    TaskspaceWholeArchiveAllocatorError,
    TaskspaceWholeArchiveProposalV1,
    allocate_taskspace_whole_archive,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pointer_payload(score: float) -> dict[str, object]:
    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": POINTER_SCHEMA_VERSION,
        "our_local_frontier_contest_cpu": {
            "score": score,
            "axis": "contest_cpu",
            "archive_sha256": "a" * 64,
            "lane_id": "synthetic_allocator_pointer",
            "hardware_substrate": "synthetic_cpu",
            "measured_at_utc": now,
            "evidence_grade": "[synthetic-test-only]",
            "source_path": "synthetic-test-only.json",
            "extra": {},
        },
        "our_local_frontier_contest_cuda": None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": None,
        "upstream_leaderboard_snapshot_at_utc": None,
        "last_refreshed_utc": now,
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "synthetic-test-do-not-run",
        "refresh_provenance": {"synthetic_test": True},
        "effective_frontier": None,
    }


def _write_pointer(repo_root: Path, *, score: float = 0.8) -> Path:
    path = repo_root / ".omx/state/canonical_frontier_pointer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_pointer_payload(score), sort_keys=True), encoding="utf-8")
    return path


def _snapshot(repo_root: Path):
    _write_pointer(repo_root)
    return load_dynamic_frontier_target(repo_root=repo_root)


def _bundle(*, g: bytes = b"G:BASE", terminal: bytes | None = None) -> TaskspaceSectionBundleV1:
    return TaskspaceSectionBundleV1(
        predictor_packet=b"PREDICTOR:" + b"P" * 192,
        generative_correction_packet=g,
        coupled_preimage_packet=b"COUPLED-PREIMAGE:" + b"A" * 96,
        terminal_quotient_packet=terminal,
    )


def _decoded_identity(member_bytes: bytes) -> bytes:
    member = parse_taskspace_monolithic_pga_member(member_bytes)
    output = bytearray()
    for section in member.sections:
        role = section.role.value.encode("ascii")
        output.extend(len(role).to_bytes(2, "big"))
        output.extend(role)
        output.extend(len(section.payload).to_bytes(8, "big"))
        output.extend(section.payload)
    return bytes(output)


class _SyntheticReceiver:
    def __init__(
        self,
        *,
        wrong_archive_binding: bool = False,
        nondeterministic_receipt: bool = False,
        encoding_dependent_output: bool = False,
    ) -> None:
        self.wrong_archive_binding = wrong_archive_binding
        self.nondeterministic_receipt = nondeterministic_receipt
        self.encoding_dependent_output = encoding_dependent_output
        self.calls: list[TaskspaceReceiverRequestV1] = []

    def __call__(self, request: TaskspaceReceiverRequestV1) -> TaskspaceReceiverReceiptV1:
        parsed = parse_taskspace_outer_archive(
            request.archive_bytes,
            expected_encoding=request.encoding,
            expected_archive_sha256=request.archive_sha256,
            expected_member_sha256=request.member_sha256,
        )
        decoded = _decoded_identity(parsed.member_bytes)
        if self.encoding_dependent_output:
            decoded += request.encoding.value.encode("ascii")
        self.calls.append(request)
        receipt_object = {
            "archive_sha256": request.archive_sha256,
            "call": len(self.calls) if self.nondeterministic_receipt else 0,
            "decoded_output_sha256": _sha256(decoded),
            "encoding": request.encoding.value,
            "stage_id": request.stage_id,
        }
        return TaskspaceReceiverReceiptV1(
            stage_id=request.stage_id,
            encoding=request.encoding,
            archive_sha256=("f" * 64 if self.wrong_archive_binding else request.archive_sha256),
            archive_nbytes=request.archive_nbytes,
            member_sha256=request.member_sha256,
            member_nbytes=request.member_nbytes,
            decoded_output_sha256=_sha256(decoded),
            decoded_output_nbytes=len(decoded),
            receiver_receipt_payload=json.dumps(
                receipt_object,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii"),
        )


def _marker_metrics(member_bytes: bytes) -> tuple[float, float]:
    member = parse_taskspace_monolithic_pga_member(member_bytes)
    g = member.section(TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION).payload
    if b"POSE_WIN" in g:
        return 0.0012, 0.005
    if b"SEG_ONLY_TINY" in g:
        return 0.00099, 0.03
    if b"ACCEPT_SECOND" in g:
        return 0.0005, 0.01
    if b"REJECT_FIRST" in g or b"RAW_A" in g or b"RAW_B" in g:
        return 0.002, 0.02
    return 0.001, 0.02


class _SyntheticMeasurement:
    def __init__(
        self,
        *,
        wrong_archive_binding: bool = False,
        nonfinite: bool = False,
        prefix_drift: bool = False,
        mutate_pointer: Path | None = None,
    ) -> None:
        self.wrong_archive_binding = wrong_archive_binding
        self.nonfinite = nonfinite
        self.prefix_drift = prefix_drift
        self.mutate_pointer = mutate_pointer
        self.calls: list[TaskspaceMeasurementRequestV1] = []
        self._mutated = False

    def __call__(
        self,
        request: TaskspaceMeasurementRequestV1,
    ) -> TaskspaceRealizedMeasurementReceiptV1:
        parsed = parse_taskspace_outer_archive(
            request.archive_bytes,
            expected_encoding=request.selected_encoding,
            expected_archive_sha256=request.archive_sha256,
            expected_member_sha256=request.member_sha256,
        )
        d_seg, d_pose = _marker_metrics(parsed.member_bytes)
        if self.prefix_drift and request.stage_id.startswith("accepted_prefix:"):
            d_pose += 0.001
        if self.nonfinite:
            d_pose = math.nan
        self.calls.append(request)
        if self.mutate_pointer is not None and not self._mutated:
            self._mutated = True
            self.mutate_pointer.write_text(
                json.dumps(_pointer_payload(0.79), sort_keys=True),
                encoding="utf-8",
            )
        receipt_payload = json.dumps(
            {
                "archive_sha256": request.archive_sha256,
                "d_pose": d_pose,
                "d_seg": d_seg,
                "decoded_output_sha256": request.decoded_output_sha256,
                "stage_id": request.stage_id,
            },
            allow_nan=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return TaskspaceRealizedMeasurementReceiptV1(
            stage_id=request.stage_id,
            archive_sha256=("e" * 64 if self.wrong_archive_binding else request.archive_sha256),
            archive_nbytes=request.archive_nbytes,
            member_sha256=request.member_sha256,
            member_nbytes=request.member_nbytes,
            decoded_output_sha256=request.decoded_output_sha256,
            decoded_output_nbytes=request.decoded_output_nbytes,
            receiver_receipt_sha256=request.receiver_receipt_sha256,
            d_seg=d_seg,
            d_pose=d_pose,
            measurement_receipt_payload=receipt_payload,
        )


def _run(
    tmp_path: Path,
    proposals: tuple[TaskspaceWholeArchiveProposalV1, ...] = (),
    *,
    receiver: _SyntheticReceiver | None = None,
    measurement: _SyntheticMeasurement | None = None,
):
    return allocate_taskspace_whole_archive(
        _bundle(),
        proposals,
        frontier_snapshot=_snapshot(tmp_path),
        receiver_callback=receiver or _SyntheticReceiver(),
        measurement_callback=measurement or _SyntheticMeasurement(),
    )


def test_raw_section_ranking_reverses_after_monolithic_deflate(tmp_path: Path) -> None:
    incompressible = hashlib.shake_256(b"allocator-raw-A").digest(900)
    proposals = (
        TaskspaceWholeArchiveProposalV1(
            "raw_a",
            lambda bundle: replace(
                bundle,
                generative_correction_packet=bundle.generative_correction_packet + b":RAW_A:" + incompressible,
            ),
        ),
        TaskspaceWholeArchiveProposalV1(
            "raw_b",
            lambda bundle: replace(
                bundle,
                generative_correction_packet=bundle.generative_correction_packet + b":RAW_B:" + b"Z" * 1800,
            ),
        ),
    )

    result = _run(tmp_path, proposals)
    raw_a, raw_b = result.proposal_audits

    assert raw_a.accepted is raw_b.accepted is False
    assert raw_a.raw_section_bytes_delta < raw_b.raw_section_bytes_delta
    assert raw_a.selected_archive_bytes_delta > raw_b.selected_archive_bytes_delta
    assert raw_a.deflated_archive_bytes_delta > raw_b.deflated_archive_bytes_delta
    assert raw_a.monolithic_member_bytes_delta == raw_a.raw_section_bytes_delta
    assert raw_b.monolithic_member_bytes_delta == raw_b.raw_section_bytes_delta


def test_seg_harming_pose_winning_negative_total_delta_is_accepted(tmp_path: Path) -> None:
    proposal = TaskspaceWholeArchiveProposalV1(
        "pose_win",
        lambda bundle: replace(
            bundle,
            generative_correction_packet=bundle.generative_correction_packet + b":POSE_WIN:",
        ),
    )

    result = _run(tmp_path, (proposal,))
    audit = result.proposal_audits[0]

    assert audit.score_transition.seg_term_delta > 0.0
    assert audit.score_transition.pose_term_delta < 0.0
    assert audit.score_transition.exact_score_delta < 0.0
    assert audit.accepted is True
    assert result.accepted_proposal_ids == ("pose_win",)
    assert audit.accepted_prefix_state is result.final_state


def test_stale_snapshot_is_refused_before_archive_callbacks(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    stale = replace(
        snapshot,
        last_refreshed_utc=(datetime.now(UTC) - timedelta(hours=25)).isoformat(),
    )
    receiver = _SyntheticReceiver()
    measurement = _SyntheticMeasurement()

    with pytest.raises(DynamicFrontierTargetError, match="24-hour"):
        allocate_taskspace_whole_archive(
            _bundle(),
            (),
            frontier_snapshot=stale,
            receiver_callback=receiver,
            measurement_callback=measurement,
        )

    assert receiver.calls == []
    assert measurement.calls == []


def test_swapped_pointer_is_refused_before_archive_callbacks(tmp_path: Path) -> None:
    path = _write_pointer(tmp_path, score=0.8)
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)
    path.write_text(json.dumps(_pointer_payload(0.79), sort_keys=True), encoding="utf-8")
    receiver = _SyntheticReceiver()

    with pytest.raises(DynamicFrontierTargetError, match="changed after snapshot"):
        allocate_taskspace_whole_archive(
            _bundle(),
            (),
            frontier_snapshot=snapshot,
            receiver_callback=receiver,
            measurement_callback=_SyntheticMeasurement(),
        )

    assert receiver.calls == []


def test_pointer_swap_during_measurement_is_refused(tmp_path: Path) -> None:
    path = _write_pointer(tmp_path, score=0.8)
    snapshot = load_dynamic_frontier_target(repo_root=tmp_path)

    with pytest.raises(DynamicFrontierTargetError, match="changed after snapshot"):
        allocate_taskspace_whole_archive(
            _bundle(),
            (),
            frontier_snapshot=snapshot,
            receiver_callback=_SyntheticReceiver(),
            measurement_callback=_SyntheticMeasurement(mutate_pointer=path),
        )


def test_measurement_archive_binding_mismatch_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TaskspaceWholeArchiveAllocatorError, match="measurement receipt is not bound"):
        _run(tmp_path, measurement=_SyntheticMeasurement(wrong_archive_binding=True))


def test_receiver_archive_binding_mismatch_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TaskspaceWholeArchiveAllocatorError, match="receiver receipt is not bound"):
        _run(tmp_path, receiver=_SyntheticReceiver(wrong_archive_binding=True))


def test_repeat_run_and_double_decode_are_deterministic(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    first_receiver = _SyntheticReceiver()
    second_receiver = _SyntheticReceiver()
    measurement = _SyntheticMeasurement()
    first = allocate_taskspace_whole_archive(
        _bundle(),
        (),
        frontier_snapshot=snapshot,
        receiver_callback=first_receiver,
        measurement_callback=measurement,
    )
    second = allocate_taskspace_whole_archive(
        _bundle(),
        (),
        frontier_snapshot=snapshot,
        receiver_callback=second_receiver,
        measurement_callback=measurement,
    )

    assert len(first_receiver.calls) == len(second_receiver.calls) == 4
    assert [request.encoding for request in first_receiver.calls] == [
        first.baseline_state.archive_build.stored.encoding,
        first.baseline_state.archive_build.stored.encoding,
        first.baseline_state.archive_build.deflated.encoding,
        first.baseline_state.archive_build.deflated.encoding,
    ]
    assert first == second
    assert first.baseline_state.archive_build.stored.archive_bytes == (
        second.baseline_state.archive_build.stored.archive_bytes
    )
    assert first.baseline_state.archive_build.deflated.archive_bytes == (
        second.baseline_state.archive_build.deflated.archive_bytes
    )


def test_nondeterministic_double_decode_receipt_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TaskspaceWholeArchiveAllocatorError, match="double-decode"):
        _run(tmp_path, receiver=_SyntheticReceiver(nondeterministic_receipt=True))


def test_store_deflate_decoded_identity_mismatch_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TaskspaceWholeArchiveAllocatorError, match="different decoded identities"):
        _run(tmp_path, receiver=_SyntheticReceiver(encoding_dependent_output=True))


def test_rejected_proposal_does_not_mutate_next_prefix(tmp_path: Path) -> None:
    observed_second_inputs: list[bytes] = []

    def second_transform(bundle: TaskspaceSectionBundleV1) -> TaskspaceSectionBundleV1:
        observed_second_inputs.append(bundle.generative_correction_packet)
        return replace(
            bundle,
            generative_correction_packet=bundle.generative_correction_packet + b":ACCEPT_SECOND:",
        )

    proposals = (
        TaskspaceWholeArchiveProposalV1(
            "reject_first",
            lambda bundle: replace(
                bundle,
                generative_correction_packet=bundle.generative_correction_packet + b":REJECT_FIRST:",
            ),
        ),
        TaskspaceWholeArchiveProposalV1("accept_second", second_transform),
    )

    result = _run(tmp_path, proposals)

    assert [audit.accepted for audit in result.proposal_audits] == [False, True]
    assert observed_second_inputs == [b"G:BASE"]
    assert b"REJECT_FIRST" not in result.final_state.bundle.generative_correction_packet
    assert b"ACCEPT_SECOND" in result.final_state.bundle.generative_correction_packet
    assert result.proposal_audits[1].before_stage_id == "baseline"


def test_accepted_prefix_is_rebuilt_and_remeasured(tmp_path: Path) -> None:
    receiver = _SyntheticReceiver()
    measurement = _SyntheticMeasurement()
    proposal = TaskspaceWholeArchiveProposalV1(
        "pose_win",
        lambda bundle: replace(
            bundle,
            generative_correction_packet=bundle.generative_correction_packet + b":POSE_WIN:",
        ),
    )

    result = _run(tmp_path, (proposal,), receiver=receiver, measurement=measurement)

    assert len(receiver.calls) == 12  # baseline, trial, prefix; STORE+DEFLATE; twice each
    assert [request.stage_id for request in measurement.calls] == [
        "baseline",
        "proposal_trial:0:pose_win",
        "accepted_prefix:0:pose_win",
    ]
    assert result.final_state.stage_id == "accepted_prefix:0:pose_win"


def test_accepted_prefix_measurement_drift_is_refused(tmp_path: Path) -> None:
    proposal = TaskspaceWholeArchiveProposalV1(
        "pose_win",
        lambda bundle: replace(
            bundle,
            generative_correction_packet=bundle.generative_correction_packet + b":POSE_WIN:",
        ),
    )

    with pytest.raises(TaskspaceWholeArchiveAllocatorError, match="rebuild/re-measurement drifted"):
        _run(tmp_path, (proposal,), measurement=_SyntheticMeasurement(prefix_drift=True))


def test_axis_local_seg_win_is_rejected_when_exact_total_worsens(tmp_path: Path) -> None:
    proposal = TaskspaceWholeArchiveProposalV1(
        "seg_only_tiny",
        lambda bundle: replace(
            bundle,
            generative_correction_packet=bundle.generative_correction_packet + b":SEG_ONLY_TINY:",
        ),
    )

    result = _run(tmp_path, (proposal,))
    audit = result.proposal_audits[0]

    assert audit.score_transition.seg_term_delta < 0.0
    assert audit.score_transition.pose_term_delta > 0.0
    assert audit.score_transition.exact_score_delta > 0.0
    assert audit.accepted is False


def test_duplicate_proposal_ids_fail_before_callbacks(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    receiver = _SyntheticReceiver()
    duplicate = TaskspaceWholeArchiveProposalV1("same", lambda bundle: bundle)

    with pytest.raises(TaskspaceWholeArchiveAllocatorError, match="duplicate proposal IDs"):
        allocate_taskspace_whole_archive(
            _bundle(),
            (duplicate, duplicate),
            frontier_snapshot=snapshot,
            receiver_callback=receiver,
            measurement_callback=_SyntheticMeasurement(),
        )

    assert receiver.calls == []


def test_nonfinite_measurement_is_refused_at_typed_receipt_boundary(tmp_path: Path) -> None:
    with pytest.raises(TaskspaceWholeArchiveAllocatorError, match="d_pose must be a finite"):
        _run(tmp_path, measurement=_SyntheticMeasurement(nonfinite=True))


def test_terminal_section_delta_and_whole_prices_are_reported(tmp_path: Path) -> None:
    proposal = TaskspaceWholeArchiveProposalV1(
        "add_terminal",
        lambda bundle: replace(bundle, terminal_quotient_packet=b"T" * 73),
    )

    result = _run(tmp_path, (proposal,))
    audit = result.proposal_audits[0]
    terminal = next(row for row in audit.section_deltas if row.role is TaskspaceMonolithicPGARole.TERMINAL_QUOTIENT)

    assert terminal.before_nbytes == 0
    assert terminal.after_nbytes == terminal.delta_nbytes == 73
    assert audit.raw_section_bytes_delta == 73
    assert audit.monolithic_member_bytes_delta > audit.raw_section_bytes_delta
    assert type(audit.stored_archive_bytes_delta) is int
    assert type(audit.deflated_archive_bytes_delta) is int
    assert type(audit.selected_archive_bytes_delta) is int


def test_zero_delta_proposal_is_rejected_and_truth_is_non_authoritative(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        (TaskspaceWholeArchiveProposalV1("identity", lambda bundle: bundle),),
    )
    audit = result.proposal_audits[0]

    assert audit.score_transition.exact_score_delta == 0.0
    assert audit.accepted is False
    assert result.final_state is result.baseline_state
    assert result.truth.authority == "derived_arithmetic_research_only"
    assert result.truth.research_only is True
    assert result.truth.candidate_claim is False
    assert result.truth.score_claim is False
    assert result.truth.evaluation_claim is False
    assert result.truth.originality_claim is False
    assert result.truth.promotion_eligible is False
    assert result.truth.ready_for_exact_eval_dispatch is False
    assert result.truth.scorer_dispatch_performed_by_allocator is False
    assert result.truth.pointer_moved is False
    assert result.truth.global_optimality_claim is False
    assert result.truth.sign_reversal_safe is False
    assert result.truth.caller_order_is_search_prior is True
    assert result.truth.endpoint_proposal_generator_only is True


def test_wrong_transform_return_type_fails_closed(tmp_path: Path) -> None:
    wrong_transform = cast("TaskspaceProposalTransform", lambda _bundle: b"not-a-bundle")
    proposal = TaskspaceWholeArchiveProposalV1("wrong_type", wrong_transform)

    with pytest.raises(TaskspaceWholeArchiveAllocatorError, match="wrong exact bundle type"):
        _run(tmp_path, (proposal,))
