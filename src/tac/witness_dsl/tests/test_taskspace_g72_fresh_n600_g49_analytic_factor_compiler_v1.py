# SPDX-License-Identifier: MIT
"""G72 role/custody/proposal mechanics; no score or candidate evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 as g72
from tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 import (
    EXACT_JOINT_STAGE_ADMISSION_OWED,
    FRESH_BATCH16_MARGIN_CUSTODY_OWED,
    FRESH_V15_BASE_SCORER_CACHE_OWED,
    G49_ROLE_WIRE_OWED,
    POSE_AUTHORITY_OR_FINAL_REPLAY_OWED,
    V15_ROLE_AWARE_DECODER_OWED,
    G72AnalyticFactorCompilerError,
    G72StagePlanV1,
    G72WholeObjectMeasurementV1,
    admit_exact_whole_object_change,
    audit_g72_readiness,
    derive_v9_boundary_shearlet_stage_proposals,
    g72_stage_plan,
    prove_current_g49_role_collision,
    reopen_stage_checkpoint,
    write_stage_checkpoint,
)

_SCORER_FIELD_SHAPE = (120, 384, 512)


_QUARANTINE_V15_PRODUCER_PIN = pytest.mark.xfail(
    strict=True,
    reason=(
        "QUARANTINED 2026-09-03 (ddm_ql1): retired July taskspace lineage, no live consumer. "
        "The sealed V15 compile receipt's producer_custody pins "
        "src/tac/optimization/direct_description_carrier_compose.py at 3e1f69bb/156,551 B; HEAD is "
        "6fef110d/160,470 B. Drift commits: 9934d488b then 36f4b2947 (both 2026-08-20). "
        "This is NOT a hash swap: 36f4b2947 adds key non_empty_member_payload_count to "
        "prove_carrier_archive_fail_closed, which the V15 producer embeds as "
        "receipt.fail_closed_mutation_proof, so a regenerated receipt legitimately differs. "
        "FIRE TRIGGER: a scorer-authorized arm re-runs tools/measure_ddm_v15_scorer_solved_templates.py "
        "(it solves through exact R + SegNet, which ddm_ql1's charter forbade) and either refreshes "
        "producer_custody against a bit-exact compile receipt -- then DELETE this mark -- or records "
        "real output drift. strict=True means this mark FAILS the moment the lineage is repaired. "
        "Owning memos: .omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md and "
        ".omx/research/ddm_cd1_working_tree_debt_landing_20260903.md"
    ),
)


@pytest.fixture(scope="module")
def proposal_stage_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.ones(_SCORER_FIELD_SHAPE, dtype=np.uint8)
    described = np.ones(_SCORER_FIELD_SHAPE, dtype=np.uint8)
    target[0, 10:20, 20:50] = 0
    margins = np.broadcast_to(
        np.asarray(0.05, dtype=np.float32),
        _SCORER_FIELD_SHAPE,
    )
    return target, margins, described


def test_current_g49_role_collision_is_a_behavioral_blocker() -> None:
    proof = prove_current_g49_role_collision()

    assert proof["payloads_byte_identical"] is True
    assert proof["role_preserved_by_current_g49_wire"] is False
    assert proof["road_payload_sha256"] == proof["undrivable_payload_sha256"]
    assert proof["blocker"] == G49_ROLE_WIRE_OWED


def test_real_v9_fisher_shearlet_stage_derivation_preserves_role(
    proposal_stage_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    target, margins, described = proposal_stage_fields
    proposals = derive_v9_boundary_shearlet_stage_proposals(
        stage=g72_stage_plan()[0],
        target_cells=target,
        target_margins=margins,
        described_cells=described,
        minimum_component_sites=8,
        maximum_components_per_pair_role=2,
    )

    assert len(proposals) == 4
    assert {row.atom.role for row in proposals} == {"Road"}
    assert {row.atom.pair_index for row in proposals} == {0}
    assert {row.atom.amplitude_q4 for row in proposals} == {
        -40,
        -80,
        40,
        80,
    }
    assert all(row.fisher_priority > 0.0 for row in proposals)
    assert len({row.fingerprint for row in proposals}) == len(proposals)
    assert proposals == tuple(
        sorted(
            proposals,
            key=lambda row: (-row.fisher_priority, row.candidate_id),
        )
    )


def test_proposal_geometry_uses_exact_component_membership_not_bbox_mismatch() -> None:
    target = np.broadcast_to(
        np.asarray(0, dtype=np.uint8),
        _SCORER_FIELD_SHAPE,
    )
    described = np.zeros(_SCORER_FIELD_SHAPE, dtype=np.uint8)
    margins = np.broadcast_to(
        np.asarray(0.05, dtype=np.float32),
        _SCORER_FIELD_SHAPE,
    )

    # A 40-site ring encloses, but is disconnected from, a 16-site island.
    # The old bbox-based geometry path incorrectly charged the island's sites
    # to the ring as well.  Equal per-site Fisher mass makes the exact 40/16
    # component-site ratio independently visible in proposal priority.
    described[0, 10, 10:21] = 1
    described[0, 20, 10:21] = 1
    described[0, 10:21, 10] = 1
    described[0, 10:21, 20] = 1
    described[0, 12:16, 12:16] = 1

    proposals = derive_v9_boundary_shearlet_stage_proposals(
        stage=g72_stage_plan()[0],
        target_cells=target,
        target_margins=margins,
        described_cells=described,
        minimum_component_sites=1,
        maximum_components_per_pair_role=4096,
    )
    ring = next(row for row in proposals if row.candidate_id == "road_0_0_sh_d0_a1")
    island = next(row for row in proposals if row.candidate_id == "road_0_1_sh_d0_a1")

    assert (ring.atom.center_y, ring.atom.center_x) == (15, 15)
    assert (island.atom.center_y, island.atom.center_x) == (14, 14)
    assert ring.fisher_priority / island.fisher_priority == pytest.approx(40 / 16)


def test_stage_partition_is_exactly_five_by_120_and_refuses_drift() -> None:
    plan = g72_stage_plan()

    assert [(row.pair_start, row.pair_stop_exclusive) for row in plan] == [
        (0, 120),
        (120, 240),
        (240, 360),
        (360, 480),
        (480, 600),
    ]
    with pytest.raises(G72AnalyticFactorCompilerError, match="canonical"):
        G72StagePlanV1(
            stage_index=1,
            pair_start=121,
            pair_stop_exclusive=240,
        )


def test_stage_checkpoint_is_atomic_immutable_and_resume_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal_stage_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    target, margins, described = proposal_stage_fields
    stage = g72_stage_plan()[0]
    proposals = derive_v9_boundary_shearlet_stage_proposals(
        stage=stage,
        target_cells=target,
        target_margins=margins,
        described_cells=described,
        minimum_component_sites=8,
        maximum_components_per_pair_role=2,
    )
    monkeypatch.setattr(
        g72,
        "require_ssd_output_root",
        lambda path: Path(path).resolve(),
    )
    custody = {
        "semantic_archive_sha256": "1" * 64,
        "semantic_compile_receipt_sha256": "2" * 64,
        "g46_target_receipt_sha256": "3" * 64,
        "g51_operand_receipt_sha256": "4" * 64,
    }

    path = write_stage_checkpoint(
        output_root=tmp_path / "g72",
        stage=stage,
        target_cells=target,
        target_margins=margins,
        described_cells=described,
        minimum_component_sites=8,
        maximum_components_per_pair_role=2,
        proposals=proposals,
        previous_checkpoint_sha256=None,
        **custody,
    )
    first = path.read_bytes()
    parsed = json.loads(first)
    reopened = reopen_stage_checkpoint(
        path,
        expected_checkpoint_sha256=parsed["checkpoint_sha256"],
    )
    same = write_stage_checkpoint(
        output_root=tmp_path / "g72",
        stage=stage,
        target_cells=target,
        target_margins=margins,
        described_cells=described,
        minimum_component_sites=8,
        maximum_components_per_pair_role=2,
        proposals=proposals,
        previous_checkpoint_sha256=None,
        **custody,
    )

    assert same == path
    assert path.read_bytes() == first
    assert reopened["pair_range"] == [0, 120]
    assert len(reopened["proposals"]) == 4
    with pytest.raises(G72AnalyticFactorCompilerError, match="resume custody"):
        reopen_stage_checkpoint(
            path,
            expected_checkpoint_sha256="f" * 64,
        )


def test_joint_admission_uses_only_the_complete_score_tradeoff() -> None:
    current = G72WholeObjectMeasurementV1(
        archive_sha256="1" * 64,
        measurement_receipt_sha256="a" * 64,
        evaluator_closure_sha256="e" * 64,
        target_custody_receipt_sha256="f" * 64,
        evidence_axis="[macOS-CPU advisory]",
        archive_bytes=100_000,
        segmentation_error_count=10_000,
        pose_squared_error_sum=0.36,
    )
    proposed = G72WholeObjectMeasurementV1(
        archive_sha256="2" * 64,
        measurement_receipt_sha256="b" * 64,
        evaluator_closure_sha256="e" * 64,
        target_custody_receipt_sha256="f" * 64,
        evidence_axis="[macOS-CPU advisory]",
        archive_bytes=100_100,
        segmentation_error_count=9_000,
        pose_squared_error_sum=0.361,
    )
    accepted = admit_exact_whole_object_change(
        current=current,
        proposed=proposed,
    )
    expensive = G72WholeObjectMeasurementV1(
        archive_sha256="3" * 64,
        measurement_receipt_sha256="c" * 64,
        evaluator_closure_sha256="e" * 64,
        target_custody_receipt_sha256="f" * 64,
        evidence_axis="[macOS-CPU advisory]",
        archive_bytes=200_000,
        segmentation_error_count=10_000,
        pose_squared_error_sum=0.36,
    )
    rejected = admit_exact_whole_object_change(
        current=current,
        proposed=expensive,
    )

    assert proposed.archive_bytes > current.archive_bytes
    assert proposed.d_pose > current.d_pose
    assert accepted["admitted"] is True
    assert float(accepted["joint_score_delta"]) < 0.0
    assert accepted["independent_component_thresholds_used"] is False
    assert rejected["admitted"] is False
    assert float(rejected["joint_score_delta"]) > 0.0


@_QUARANTINE_V15_PRODUCER_PIN
def test_real_current_custody_reopens_and_fails_closed_at_true_missing_seams() -> None:
    root = Path(__file__).resolve().parents[4]
    run = (
        root
        / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
        / "fresh_v15_semantic_base_n600_20260726"
    )
    semantic_receipt = run / "ddm_v15_scorer_solved_templates_n600_receipt.json"
    semantic_archive = run / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
    g46_audit = (
        root
        / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
        / "g46_teacher_batch_geometry_audit_v2_20260726.json"
    )
    g51_aggregate = Path(
        "/Volumes/VertigoDataTier/pact/taskspace_fresh_scorer_planes_n600_20260726/aggregate_receipt.json"
    )
    if not all(
        path.is_file()
        for path in (
            semantic_receipt,
            semantic_archive,
            g46_audit,
            g51_aggregate,
        )
    ):
        pytest.skip("retained G46/G51/V15 production custody is absent")

    receipt = audit_g72_readiness(
        semantic_compile_receipt_path=semantic_receipt,
        semantic_archive_path=semantic_archive,
        semantic_producer_root=root,
        g46_batch_geometry_audit_path=g46_audit,
        g51_operand_aggregate_path=g51_aggregate,
    )

    assert receipt["fresh_semantic_lineage"]["lineage_reopened"] is True
    assert (
        receipt["g46_target_custody"]["target_labels_sha256"]
        == receipt["g51_direct_task_operand_custody"]["target_labels_sha256"]
    )
    assert receipt["g51_direct_task_operand_custody"]["stage_count"] == 5
    assert receipt["g51_direct_task_operand_custody"]["pairs_per_stage"] == 120
    assert receipt["g51_direct_task_operand_custody"]["y0_y1_source_derived_and_recursively_reopened"] is True
    assert receipt["g51_direct_task_operand_custody"]["fresh_pose_target_authority"] is False
    assert receipt["production_compile_ready"] is False
    assert receipt["open_blockers"] == [
        FRESH_BATCH16_MARGIN_CUSTODY_OWED,
        FRESH_V15_BASE_SCORER_CACHE_OWED,
        G49_ROLE_WIRE_OWED,
        V15_ROLE_AWARE_DECODER_OWED,
        POSE_AUTHORITY_OR_FINAL_REPLAY_OWED,
        EXACT_JOINT_STAGE_ADMISSION_OWED,
    ]
