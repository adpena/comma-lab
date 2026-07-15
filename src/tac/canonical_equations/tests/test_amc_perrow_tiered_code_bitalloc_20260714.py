from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_equations.amc_perrow_tiered_code_bitalloc_20260714 import (
    AXIS,
    BASELINE_ARCHIVE_BYTES,
    BASELINE_D_POSE,
    BASELINE_D_SEG,
    BASELINE_SOURCE_ARTIFACT,
    CHECKPOINT_SHA256,
    CUSTODY_SOURCE_ARTIFACT,
    D_SEG_LABEL,
    EQUATION_ID,
    EXACT_CONTEST_CPU_TRANSFER_STATUS,
    FRESH_JOINT_N600_STATUS,
    GT_SHA256,
    OWED_TRAIN_TIME_FLAGS,
    PROMOTION_ELIGIBLE,
    RAW_RESULT_DIRS_STATUS,
    SCORE_CLAIM,
    TIERED_ARCHIVE_BYTES,
    TIERED_CODE_QAT_REACTIVATION_CRITERION,
    TIERED_D_SEG,
    TIERED_SOURCE_ARTIFACT,
    amc_perrow_tiered_code_bitalloc_law,
    build_amc_perrow_tiered_code_bitalloc_v1,
    populate_amc_perrow_tiered_code_bitalloc_v1,
)
from tac.canonical_equations.registry import (
    get_equation_by_id,
    load_registry_events_lenient,
)


def test_exact_pair_local_additive_law() -> None:
    assert amc_perrow_tiered_code_bitalloc_law([1, 2, 0], [10, 20, 10]) == pytest.approx(3 / 40)
    assert amc_perrow_tiered_code_bitalloc_law(
        np.array([1, 2], dtype=np.int64),
        np.array([10, 20], dtype=np.uint64),
    ) == pytest.approx(3 / 30)
    assert amc_perrow_tiered_code_bitalloc_law(
        np.array([1.0, 2.0], dtype=np.float32), [10, 20]
    ) == pytest.approx(3 / 30)


@pytest.mark.parametrize(
    ("mismatches", "pixels", "message"),
    [
        ([], [], "non-empty"),
        ([1], [1, 2], "matching lengths"),
        ([float("nan")], [1], "finite"),
        ([float("inf")], [1], "finite"),
        ([1.5], [2], "integral"),
        ([True], [2], "integral"),
        ([np.bool_(False)], [2], "integral"),
        ([np.float64(1.25)], [2], "integral"),
        ([-1], [2], "non-negative"),
        ([np.int64(-1)], [2], "non-negative"),
        ([1], [0], "greater than zero"),
        ([3], [2], "not exceed"),
    ],
)
def test_law_rejects_invalid_counts(mismatches, pixels, message) -> None:
    with pytest.raises(ValueError, match=message):
        amc_perrow_tiered_code_bitalloc_law(mismatches, pixels)


def test_anchor_labels_numbers_custody_and_advisory_boundary() -> None:
    equation = build_amc_perrow_tiered_code_bitalloc_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 2
    baseline, tiered = equation.empirical_anchors
    assert baseline.inputs["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert baseline.inputs["gt_sha256"] == GT_SHA256
    assert baseline.empirical_output["archive_bytes"] == BASELINE_ARCHIVE_BYTES
    assert baseline.empirical_output["d_seg"] == BASELINE_D_SEG
    assert baseline.empirical_output["d_pose"] == BASELINE_D_POSE
    assert baseline.source_artifact == BASELINE_SOURCE_ARTIFACT
    assert baseline.provenance.source_path == BASELINE_SOURCE_ARTIFACT
    assert tiered.source_artifact == TIERED_SOURCE_ARTIFACT
    assert tiered.provenance.source_path == TIERED_SOURCE_ARTIFACT
    assert tiered.empirical_output["archive_bytes_label"] == "MEASURED"
    assert tiered.empirical_output["d_seg_label"] == D_SEG_LABEL
    assert tiered.empirical_output["archive_bytes"] == TIERED_ARCHIVE_BYTES
    assert tiered.empirical_output["d_seg"] == TIERED_D_SEG
    assert len(tiered.empirical_output["archive_bytes"]) == 6
    assert len(tiered.empirical_output["d_seg"]) == 6
    assert set(tiered.empirical_output["d_pose"].values()) == {"OWED"}
    assert tiered.inputs["baseline_uniform_custody"] == "byte-identical 6/6"
    assert tiered.inputs["baseline_uniform_custody_source_artifact"] == CUSTODY_SOURCE_ARTIFACT
    assert equation.domain_of_validity["axis"] == AXIS
    assert equation.domain_of_validity["score_claim"] is SCORE_CLAIM
    assert equation.domain_of_validity["promotion_eligible"] is PROMOTION_ELIGIBLE
    assert equation.provenance.measurement_axis == AXIS
    assert equation.provenance.source_path == TIERED_SOURCE_ARTIFACT
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False
    for metadata in (baseline.inputs, tiered.inputs, equation.domain_of_validity):
        assert metadata["raw_result_dirs"] == RAW_RESULT_DIRS_STATUS
        assert metadata["fresh_joint_n600_d_pose_d_seg"] == FRESH_JOINT_N600_STATUS
        assert metadata["exact_contest_cpu_transfer"] == EXACT_CONTEST_CPU_TRANSFER_STATUS
        assert metadata["pointer"] == "UNCHANGED"
        assert metadata["score_claim"] is False
        assert metadata["promotion_eligible"] is False
    assert equation.domain_of_validity["TieredCodeQATLever"] == "OWED_NOT_BUILT"
    assert equation.domain_of_validity["TieredCodeQATLever_reactivation_criterion"] == (
        TIERED_CODE_QAT_REACTIVATION_CRITERION
    )
    assert equation.provenance.rejection_reason == TIERED_CODE_QAT_REACTIVATION_CRITERION


def test_population_is_explicit_and_duplicate_safe_in_temporary_registry(tmp_path) -> None:
    registry = tmp_path / "canonical_equations_registry.jsonl"
    lock = tmp_path / "canonical_equations_registry.jsonl.lock"
    module = importlib.import_module(
        "tac.canonical_equations.amc_perrow_tiered_code_bitalloc_20260714"
    )
    assert module.EQUATION_ID == EQUATION_ID
    assert not registry.exists()
    populated = populate_amc_perrow_tiered_code_bitalloc_v1(
        path=registry, lock_path=lock, agent="codex", subagent_id="finding2_test"
    )
    assert get_equation_by_id(EQUATION_ID, path=registry).equation_id == populated.equation_id
    assert len(load_registry_events_lenient(registry)) == 1


def test_live_registry_has_no_duplicate_equation_id() -> None:
    registry = Path(".omx/state/canonical_equations_registry.jsonl")
    matching = [
        event
        for event in load_registry_events_lenient(registry)
        if event.get("equation_id") == EQUATION_ID
    ]
    assert len(matching) <= 1


def test_tool_consumers_exist_and_owed_flags_are_not_dsl_flags() -> None:
    assert Path("tools/apply_amc_saliency_tiered_bitalloc_witness.py").is_file()
    for dotted in (
        "tac.canonical_equations.witness_measured_reverse_waterfill_20260713",
        "tac.frontier_exact_bitalloc",
    ):
        importlib.import_module(dotted)
    from tac.witness_dsl import lever_registry

    for flag in OWED_TRAIN_TIME_FLAGS:
        assert flag not in lever_registry.dsl_referenced_flags()
        assert flag not in lever_registry.dsl_emitted_flags()
        assert all(flag not in flags for flags in lever_registry.lever_factories().values())
