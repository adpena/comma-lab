# SPDX-License-Identifier: MIT
"""Adversarial structural tests for the G83 exact archive-state allocator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tac import score_geometry
from tac.canonical_frontier_pointer import POINTER_SCHEMA_VERSION
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    load_dynamic_frontier_target,
)
from tac.witness_dsl.taskspace_g83_pvsa_exact_archive_state_allocator_v1 import (
    ArchiveStateOriginV1,
    ArchiveTransitionKindV1,
    ConditionalActuatorV1,
    ExactArchiveStateV1,
    ExactEvalAxisV1,
    ExactEvaluationCustodyV1,
    ExactUpstreamComponentRowV1,
    G83ExactArchiveAllocatorError,
    allocate_exact_archive_state,
    exact_archive_state,
    exact_state_from_g80_build,
    exact_state_from_g82_lowering,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    build_compact_pvsa_archive,
)

_ROOT = Path(__file__).resolve().parents[4]
_SEMANTIC_PATH = (
    _ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "fresh_v15_semantic_base_n600_20260726"
    / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
)
_SEMANTIC_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
_NOW = "2026-07-27T00:00:00+00:00"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pointer_payload(*, upstream_score: float = 0.1) -> dict[str, object]:
    now = datetime.now(UTC).isoformat()
    entry = {
        "score": upstream_score,
        "rank": 1,
        "name": "g83-synthetic-structural-fixture",
        "pr_number": 9001,
        "pr_url": "https://invalid.example/g83-fixture",
    }
    return {
        "schema_version": POINTER_SCHEMA_VERSION,
        "our_local_frontier_contest_cpu": None,
        "our_local_frontier_contest_cuda": None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": {
            "best_entry": dict(entry),
            "entries": [dict(entry)],
        },
        "upstream_leaderboard_snapshot_at_utc": now,
        "last_refreshed_utc": now,
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "synthetic-fixture-do-not-run",
        "refresh_provenance": {"fixture": True},
        "effective_frontier": None,
    }


def _frontier(tmp_path: Path, *, score: float = 0.1):
    path = tmp_path / ".omx/state/canonical_frontier_pointer.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_pointer_payload(upstream_score=score)), encoding="utf-8")
    return load_dynamic_frontier_target(repo_root=tmp_path)


def _custody(
    *,
    axis: ExactEvalAxisV1 = ExactEvalAxisV1.CONTEST_CPU,
    epoch: str = "g83-epoch-v1",
) -> ExactEvaluationCustodyV1:
    return ExactEvaluationCustodyV1(
        axis=axis,
        hardware_substrate=f"synthetic-{axis.value}-structural-fixture",
        evaluator_source_sha256="1" * 64,
        upstream_snapshot_sha256="2" * 64,
        runtime_tree_sha256="3" * 64,
        target_video_sha256="4" * 64,
        file_list_sha256="5" * 64,
        evaluation_context_id="g83-exact-upstream-context-v1",
        custody_epoch_id=epoch,
    )


def _row(
    payload: bytes,
    custody: ExactEvaluationCustodyV1,
    *,
    d_seg: float,
    d_pose: float,
    axis: ExactEvalAxisV1 | None = None,
    custody_sha256: str | None = None,
) -> ExactUpstreamComponentRowV1:
    row_axis = custody.axis if axis is None else axis
    return ExactUpstreamComponentRowV1(
        archive_sha256=_sha(payload),
        archive_bytes=len(payload),
        d_seg=d_seg,
        d_pose=d_pose,
        final_score=score_geometry.contest_score(d_seg, d_pose, len(payload)),
        output_video_sha256="6" * 64,
        evaluation_receipt_sha256="7" * 64,
        custody_sha256=custody.sha256 if custody_sha256 is None else custody_sha256,
        measured_at_utc=_NOW,
        axis=row_axis,
        evidence_grade=row_axis.evidence_grade,
    )


def _state(
    state_id: str,
    selected_actuators: tuple[str, ...],
    custody: ExactEvaluationCustodyV1,
    *,
    archive_bytes: int,
    d_seg: float,
    d_pose: float,
    byte_tag: int,
    row_axis: ExactEvalAxisV1 | None = None,
    row_custody_sha256: str | None = None,
) -> ExactArchiveStateV1:
    payload = bytes((byte_tag,)) * archive_bytes
    row = _row(
        payload,
        custody,
        d_seg=d_seg,
        d_pose=d_pose,
        axis=row_axis,
        custody_sha256=row_custody_sha256,
    )
    return exact_archive_state(
        state_id=state_id,
        selected_actuators=selected_actuators,
        selected_decoder_transition_ids=tuple(f"decode-{actuator_id}-v1" for actuator_id in selected_actuators),
        archive_payload=payload,
        expected_archive_sha256=_sha(payload),
        component_row=row,
        archive_validation_receipt_sha256="8" * 64,
        wire_contract_id="g83-fixture-wire-v1",
    )


def _actuators() -> tuple[ConditionalActuatorV1, ...]:
    return (
        ConditionalActuatorV1(actuator_id="A", decoder_transition_id="decode-A-v1"),
        ConditionalActuatorV1(actuator_id="B", decoder_transition_id="decode-B-v1"),
        ConditionalActuatorV1(actuator_id="C", decoder_transition_id="decode-C-v1"),
    )


def _paradox_states(custody: ExactEvaluationCustodyV1) -> tuple[ExactArchiveStateV1, ...]:
    """Finite exact rows whose nonlinear global optimum is A+C.

    B helps when added to the baseline but is globally rejected.  C hurts when
    added to the baseline, yet its composition with A is the global optimum.
    The values are structural fixtures, not empirical score evidence.
    """

    return (
        _state("base", (), custody, archive_bytes=1_000, d_seg=0.0020, d_pose=0.0010, byte_tag=1),
        _state("a", ("A",), custody, archive_bytes=1_050, d_seg=0.0019, d_pose=0.0009, byte_tag=2),
        _state("b", ("B",), custody, archive_bytes=1_100, d_seg=0.0010, d_pose=0.0004, byte_tag=3),
        _state("c", ("C",), custody, archive_bytes=1_010, d_seg=0.0021, d_pose=0.0012, byte_tag=4),
        _state("ab", ("A", "B"), custody, archive_bytes=1_140, d_seg=0.0018, d_pose=0.00085, byte_tag=5),
        _state("ac", ("A", "C"), custody, archive_bytes=1_200, d_seg=0.0006, d_pose=0.0001, byte_tag=6),
        _state(
            "abc",
            ("A", "B", "C"),
            custody,
            archive_bytes=1_300,
            d_seg=0.0015,
            d_pose=0.0008,
            byte_tag=7,
        ),
    )


def test_custody_and_component_rows_are_content_addressed_and_exact() -> None:
    custody = _custody()
    payload = b"exact-archive-fixture"
    row = _row(payload, custody, d_seg=0.001, d_pose=0.0001)

    assert len(custody.sha256) == 64
    assert row.archive_sha256 == _sha(payload)
    assert row.archive_bytes == len(payload)
    assert row.final_score == score_geometry.contest_score(0.001, 0.0001, len(payload))


def test_component_row_refuses_recomposed_score_mismatch_proxy_and_partial_n600() -> None:
    custody = _custody()
    row = _row(b"archive", custody, d_seg=0.001, d_pose=0.0001)

    with pytest.raises(G83ExactArchiveAllocatorError, match="recomposition"):
        replace(row, final_score=row.final_score + 1e-12)
    with pytest.raises(G83ExactArchiveAllocatorError, match=r"partial|proxy"):
        replace(row, proxy=True)  # type: ignore[arg-type]
    with pytest.raises(G83ExactArchiveAllocatorError, match=r"partial|upstream"):
        replace(row, sample_count=599)  # type: ignore[arg-type]


def test_exact_state_refuses_archive_sha_and_component_row_identity_drift() -> None:
    custody = _custody()
    payload = b"archive-one"
    row = _row(payload, custody, d_seg=0.001, d_pose=0.0001)

    with pytest.raises(G83ExactArchiveAllocatorError, match="same archive object"):
        ExactArchiveStateV1(
            state_id="bad-sha",
            selected_actuators=(),
            selected_decoder_transition_ids=(),
            archive_payload=payload,
            expected_archive_sha256="9" * 64,
            component_row=row,
            origin=ArchiveStateOriginV1.UPSTREAM_EXACT_ARCHIVE,
            archive_validation_receipt_sha256="8" * 64,
            wire_contract_id="g83-fixture-wire-v1",
        )
    with pytest.raises(G83ExactArchiveAllocatorError, match="same archive object"):
        exact_archive_state(
            state_id="bad-row",
            selected_actuators=(),
            selected_decoder_transition_ids=(),
            archive_payload=b"archive-two",
            expected_archive_sha256=_sha(b"archive-two"),
            component_row=row,
            archive_validation_receipt_sha256="8" * 64,
            wire_contract_id="g83-fixture-wire-v1",
        )


def test_global_nonlinear_selection_exposes_both_local_global_paradoxes(tmp_path: Path) -> None:
    custody = _custody()
    allocation = allocate_exact_archive_state(
        states=_paradox_states(custody),
        actuators=_actuators(),
        current_state_id="base",
        custody=custody,
        frontier=_frontier(tmp_path),
    )
    dispositions = {row.actuator_id: row for row in allocation.dispositions}

    assert allocation.selected_state_id == "ac"
    assert allocation.selected_exact_score == score_geometry.contest_score(0.0006, 0.0001, 1_200)
    assert allocation.selected_exact_score < allocation.frontier_target_score
    assert allocation.beats_dynamic_frontier is True
    assert dispositions["B"].locally_beneficial_somewhere is True
    assert dispositions["B"].globally_selected is False
    assert dispositions["B"].classification == "locally_beneficial_globally_rejected"
    assert dispositions["C"].locally_harmful_somewhere is True
    assert dispositions["C"].globally_selected is True
    assert dispositions["C"].classification == "locally_harmful_globally_selected"
    assert allocation.component_thresholds_used is False
    assert allocation.scorer_invoked is False
    assert allocation.research_only is True


def test_route_uses_exact_rollback_then_conditional_add(tmp_path: Path) -> None:
    custody = _custody()
    allocation = allocate_exact_archive_state(
        states=_paradox_states(custody),
        actuators=_actuators(),
        current_state_id="abc",
        custody=custody,
        frontier=_frontier(tmp_path),
    )

    assert tuple(row.kind for row in allocation.route) == (
        ArchiveTransitionKindV1.ROLLBACK,
        ArchiveTransitionKindV1.ADD,
    )
    assert allocation.route[0].from_state_id == "abc"
    assert allocation.route[0].to_state_id == "a"
    assert allocation.route[0].actuator_ids == ("B", "C")
    assert allocation.route[1].from_state_id == "a"
    assert allocation.route[1].to_state_id == "ac"
    assert allocation.route[1].actuator_ids == ("C",)
    assert all(
        edge.score_transition.after_score == dict(allocation.state_scores)[edge.to_state_id]
        for edge in allocation.route
    )


def test_route_can_remove_before_recomposing_selected_state(tmp_path: Path) -> None:
    custody = _custody()
    allocation = allocate_exact_archive_state(
        states=_paradox_states(custody),
        actuators=_actuators(),
        current_state_id="b",
        custody=custody,
        frontier=_frontier(tmp_path),
    )

    assert tuple(row.kind for row in allocation.route) == (
        ArchiveTransitionKindV1.REMOVE,
        ArchiveTransitionKindV1.ADD,
        ArchiveTransitionKindV1.ADD,
    )
    assert tuple(row.to_state_id for row in allocation.route) == ("base", "a", "ac")


def test_pareto_pruning_uses_only_monotone_complete_axes(tmp_path: Path) -> None:
    custody = _custody()
    allocation = allocate_exact_archive_state(
        states=_paradox_states(custody),
        actuators=_actuators(),
        current_state_id="base",
        custody=custody,
        frontier=_frontier(tmp_path),
    )
    dominated = {row.dominated_state_id: row for row in allocation.dominated}

    assert dominated["ab"].dominating_state_id == "b"
    assert dominated["abc"].dominating_state_id == "ac"
    assert all(row.monotone_axes == "d_seg,d_pose,archive_bytes" for row in dominated.values())
    assert "ac" in allocation.pareto_frontier_state_ids


def test_missing_baseline_unknown_actuator_and_duplicate_selection_are_refused(tmp_path: Path) -> None:
    custody = _custody()
    base, a, *_ = _paradox_states(custody)
    frontier = _frontier(tmp_path)

    with pytest.raises(G83ExactArchiveAllocatorError, match="zero-actuator"):
        allocate_exact_archive_state(
            states=(a,),
            actuators=_actuators(),
            current_state_id="a",
            custody=custody,
            frontier=frontier,
        )
    unknown = replace(a, state_id="unknown", selected_actuators=("Z",))
    with pytest.raises(G83ExactArchiveAllocatorError, match="unknown actuator"):
        allocate_exact_archive_state(
            states=(base, unknown),
            actuators=_actuators(),
            current_state_id="base",
            custody=custody,
            frontier=frontier,
        )
    duplicate = replace(a, state_id="a-copy")
    with pytest.raises(G83ExactArchiveAllocatorError, match="same selected actuator"):
        allocate_exact_archive_state(
            states=(base, a, duplicate),
            actuators=_actuators(),
            current_state_id="base",
            custody=custody,
            frontier=frontier,
        )


def test_dependency_order_missing_prefix_and_conflicting_subset_are_refused(tmp_path: Path) -> None:
    custody = _custody()
    frontier = _frontier(tmp_path)
    base = _state("base", (), custody, archive_bytes=10, d_seg=0.01, d_pose=0.01, byte_tag=1)
    a = _state("a", ("A",), custody, archive_bytes=11, d_seg=0.009, d_pose=0.009, byte_tag=2)
    b = _state("b", ("B",), custody, archive_bytes=12, d_seg=0.008, d_pose=0.008, byte_tag=3)
    ab = _state("ab", ("A", "B"), custody, archive_bytes=13, d_seg=0.007, d_pose=0.007, byte_tag=4)
    dependent = (
        ConditionalActuatorV1(actuator_id="A", decoder_transition_id="decode-A-v1"),
        ConditionalActuatorV1(
            actuator_id="B",
            decoder_transition_id="decode-B-v1",
            prerequisites=("A",),
        ),
    )

    with pytest.raises(G83ExactArchiveAllocatorError, match="before its prerequisites"):
        allocate_exact_archive_state(
            states=(base, b),
            actuators=dependent,
            current_state_id="base",
            custody=custody,
            frontier=frontier,
        )
    with pytest.raises(G83ExactArchiveAllocatorError, match="prefix-closed"):
        allocate_exact_archive_state(
            states=(base, ab),
            actuators=dependent,
            current_state_id="base",
            custody=custody,
            frontier=frontier,
        )
    conflicts = (
        ConditionalActuatorV1(
            actuator_id="A",
            decoder_transition_id="decode-A-v1",
            conflicts=("B",),
        ),
        ConditionalActuatorV1(
            actuator_id="B",
            decoder_transition_id="decode-B-v1",
            conflicts=("A",),
        ),
    )
    with pytest.raises(G83ExactArchiveAllocatorError, match="conflicting"):
        allocate_exact_archive_state(
            states=(base, a, ab),
            actuators=conflicts,
            current_state_id="base",
            custody=custody,
            frontier=frontier,
        )


def test_exact_decoder_transition_identity_cannot_be_relabelled(tmp_path: Path) -> None:
    custody = _custody()
    base, a, *_ = _paradox_states(custody)
    relabelled = replace(
        a,
        state_id="a-relabelled",
        selected_decoder_transition_ids=("decode-B-v1",),
    )

    with pytest.raises(G83ExactArchiveAllocatorError, match="relabels"):
        allocate_exact_archive_state(
            states=(base, relabelled),
            actuators=_actuators(),
            current_state_id="base",
            custody=custody,
            frontier=_frontier(tmp_path),
        )


def test_asymmetric_conflict_and_late_prerequisite_registry_are_refused(tmp_path: Path) -> None:
    custody = _custody()
    states = (_paradox_states(custody)[0],)
    frontier = _frontier(tmp_path)

    with pytest.raises(G83ExactArchiveAllocatorError, match="symmetrically"):
        allocate_exact_archive_state(
            states=states,
            actuators=(
                ConditionalActuatorV1(
                    actuator_id="A",
                    decoder_transition_id="decode-A-v1",
                    conflicts=("B",),
                ),
                ConditionalActuatorV1(actuator_id="B", decoder_transition_id="decode-B-v1"),
            ),
            current_state_id="base",
            custody=custody,
            frontier=frontier,
        )
    with pytest.raises(G83ExactArchiveAllocatorError, match="not earlier"):
        allocate_exact_archive_state(
            states=states,
            actuators=(
                ConditionalActuatorV1(
                    actuator_id="A",
                    decoder_transition_id="decode-A-v1",
                    prerequisites=("B",),
                ),
                ConditionalActuatorV1(actuator_id="B", decoder_transition_id="decode-B-v1"),
            ),
            current_state_id="base",
            custody=custody,
            frontier=frontier,
        )


def test_foreign_custody_and_cross_axis_rows_fail_closed(tmp_path: Path) -> None:
    custody = _custody()
    frontier = _frontier(tmp_path)
    foreign = _state(
        "foreign",
        (),
        custody,
        archive_bytes=10,
        d_seg=0.001,
        d_pose=0.001,
        byte_tag=1,
        row_custody_sha256="a" * 64,
    )
    cross_axis = _state(
        "cross-axis",
        (),
        custody,
        archive_bytes=10,
        d_seg=0.001,
        d_pose=0.001,
        byte_tag=2,
        row_axis=ExactEvalAxisV1.CONTEST_CUDA,
    )

    with pytest.raises(G83ExactArchiveAllocatorError, match="stale or foreign custody"):
        allocate_exact_archive_state(
            states=(foreign,),
            actuators=(),
            current_state_id="foreign",
            custody=custody,
            frontier=frontier,
        )
    with pytest.raises(G83ExactArchiveAllocatorError, match="cross-axis"):
        allocate_exact_archive_state(
            states=(cross_axis,),
            actuators=(),
            current_state_id="cross-axis",
            custody=custody,
            frontier=frontier,
        )


def test_dynamic_pointer_mutation_during_allocation_is_refused(tmp_path: Path) -> None:
    custody = _custody()
    states = (_paradox_states(custody)[0],)
    frontier = _frontier(tmp_path, score=0.2)
    path = tmp_path / ".omx/state/canonical_frontier_pointer.json"
    path.write_text(json.dumps(_pointer_payload(upstream_score=0.19)), encoding="utf-8")

    with pytest.raises(DynamicFrontierTargetError, match="changed after snapshot"):
        allocate_exact_archive_state(
            states=states,
            actuators=(),
            current_state_id="base",
            custody=custody,
            frontier=frontier,
        )


def test_receipt_is_deterministic_self_hashing_and_omits_archive_payloads(tmp_path: Path) -> None:
    custody = _custody()
    states = _paradox_states(custody)
    frontier = _frontier(tmp_path)
    first = allocate_exact_archive_state(
        states=states,
        actuators=_actuators(),
        current_state_id="base",
        custody=custody,
        frontier=frontier,
    )
    second = allocate_exact_archive_state(
        states=states,
        actuators=_actuators(),
        current_state_id="base",
        custody=custody,
        frontier=frontier,
    )

    assert first.to_receipt_bytes() == second.to_receipt_bytes()
    assert first.receipt_sha256 == _sha(first.to_receipt_bytes())
    assert bytes((6,)) * 1_200 not in first.to_receipt_bytes()


def test_g80_adapter_binds_real_selected_outer_archive_bytes() -> None:
    if not _SEMANTIC_PATH.is_file():
        pytest.skip("retained fresh V15 semantic P is absent")
    semantic = _SEMANTIC_PATH.read_bytes()
    assert _sha(semantic) == _SEMANTIC_SHA256
    build = build_compact_pvsa_archive(
        semantic_p_archive=semantic,
        actuator_payloads=(),
        maximum_semantic_archive_bytes=2 << 20,
        maximum_member_bytes=2 << 20,
        maximum_section_bytes=1 << 20,
    )
    selected = build.outer_build.selected
    custody = _custody()
    row = _row(selected.archive_bytes, custody, d_seg=0.001, d_pose=0.0001)
    state = exact_state_from_g80_build(
        state_id="g80-zero",
        selected_actuators=(),
        build=build,
        component_row=row,
        archive_validation_receipt_sha256="8" * 64,
    )

    assert state.origin is ArchiveStateOriginV1.G80_PVSA
    assert state.archive_payload == selected.archive_bytes
    assert state.archive_bytes == selected.archive_nbytes
    assert state.archive_sha256 == selected.archive_sha256
    assert state.selected_decoder_transition_ids == ()


def test_g80_and_g82_adapters_reject_wrong_interface_types() -> None:
    custody = _custody()
    row = _row(b"archive", custody, d_seg=0.001, d_pose=0.0001)

    with pytest.raises(G83ExactArchiveAllocatorError, match="G80 adapter"):
        exact_state_from_g80_build(  # type: ignore[arg-type]
            state_id="bad-g80",
            selected_actuators=(),
            build=object(),
            component_row=row,
            archive_validation_receipt_sha256="8" * 64,
        )
    with pytest.raises(G83ExactArchiveAllocatorError, match="G82 adapter"):
        exact_state_from_g82_lowering(  # type: ignore[arg-type]
            state_id="bad-g82",
            selected_actuators=(),
            lowering=object(),
            component_row=row,
            use_actuated=False,
        )
