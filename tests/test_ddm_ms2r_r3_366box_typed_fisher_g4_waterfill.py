"""Regression tests for the Task #701 cross-chain admission."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tac.optimization.ddm_ms2r_r3_366box_typed_fisher_g4_waterfill import (
    BLOCKERS,
    RATE_SCORE_PER_BYTE,
    VERDICT,
    DDM366BoxAdmissionError,
    build_artifacts,
    canonical_bytes,
    pose_score_derivative,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / (
    ".omx/research/configs/"
    "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_20260725.json"
)
OUTPUT_ROOT = REPO / (
    ".omx/research/"
    "ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_20260725T162107Z"
)


def _inputs() -> tuple[dict, dict, dict]:
    config = json.loads(CONFIG.read_text())
    values = {}
    custody = {}
    for name, binding in config["inputs"].items():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = REPO / path
        values[name] = json.loads(path.read_text())
        custody[name] = binding
    config_custody = {
        "path": str(CONFIG.relative_to(REPO)),
        "bytes": CONFIG.stat().st_size,
        "sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "schema": config["schema"],
    }
    return values, custody, config_custody


@pytest.fixture(scope="module")
def artifacts():
    values, custody, config_custody = _inputs()
    return build_artifacts(
        values,
        input_custody=custody,
        config_custody=config_custody,
        available_memory_bytes=80 * (1 << 30),
    )


def test_complete_metric_bundle_does_not_imply_executable_homotopy(artifacts) -> None:
    preflight = artifacts.preflight
    assert preflight["verdict"] == VERDICT
    assert preflight["blockers"] == list(BLOCKERS)
    facts = preflight["custody_facts"]
    assert facts["metric_bundle_complete"] is True
    assert facts["metric_bundle_pose_tube_capability_present"] is True
    assert facts["prior_r3_admission_recalled"] is True
    assert facts["prior_r3_finite_dimension_duals"] == 0
    assert facts["solve_local_pose_tube_active"] is False
    assert facts["pc1_tube_claim"] is False
    assert facts["rg3_missing_exact_blocks"] == 25
    assert facts["fully_materialized_occupied_metric_buckets"] == 0
    assert facts["ev2_assigned_pair_cell_bytes"] == 0
    assert facts["rd1_finite_duals"] == 0
    assert preflight["execution"]["homotopy_launched"] is False


def test_priced_table_separates_settled_controls_from_task_rungs(artifacts) -> None:
    table = artifacts.priced_rung_table
    assert table["measured_task_rungs"] == []
    preregistered = table["preregistered_rungs"]
    assert [row["target_error_budget"] for row in preregistered] == [
        17_927,
        17_928,
        32_791,
        77_383,
        121_975,
        136_839,
    ]
    assert all(
        row["execution_status"] == "BLOCKED_PRECONDITION_NOT_RUN"
        and row["candidate"] is None
        and row["epistemic_status"] == "DERIVED_PREREGISTRATION_NOT_MEASURED"
        and row["active_pose_tube_required"] is True
        and row["real_receiver_uint8_parseback_required"] is True
        and row["real_coder_race_required"] is True
        for row in preregistered
    )
    assert len(table["settled_non_rung_controls"]) == 2
    exact, box = table["settled_non_rung_controls"]
    assert exact["error_budget_used"] == 17_927
    assert exact["description_bytes_total"] == 409_526_925
    assert box["error_budget_used"] == 136_839
    assert box["description_bytes_total"] == 291_205_400
    assert box["description_bytes_by_stream"] == {
        "container_receiver_overhead": 1_452,
        "predictor_payload": 291_203_948,
    }
    assert box["receiver_parseback_byte_identity"] is True
    assert exact["eligible_as_task_rung"] is False
    assert box["eligible_as_task_rung"] is False
    assert table["knee"] is None
    assert table["r6_candidate_ready"] is False
    assert table["formulation_falsifier"]["evaluated"] is False
    assert table["rate_score_per_byte_exact"] == RATE_SCORE_PER_BYTE


def test_rd1_backfill_preserves_all_162_null_cells(artifacts) -> None:
    backfill = artifacts.rd1_backfill
    assert backfill["source_cell_count"] == 162
    assert backfill["rung_measured_cell_count"] == 0
    assert backfill["lambda_measured_cell_count"] == 0
    assert backfill["still_null_lambda_cell_count"] == 162
    assert len(backfill["cells"]) == 162
    assert all(
        row["lambda_bytes_per_D_dimension"] is None
        and row["actionable_for_train_decision"] is False
        and row["task_701_homotopy_exchange_rates"] == []
        for row in backfill["cells"]
    )


def test_pose_exchange_rate_is_derived_at_each_control(artifacts) -> None:
    for row in artifacts.priced_rung_table["settled_non_rung_controls"]:
        assert row["dS_dd_pose"] == pytest.approx(
            pose_score_derivative(row["d_pose"]),
            abs=0.0,
        )
    assert pytest.approx(6.658589531221714e-7) == RATE_SCORE_PER_BYTE


def test_drift_to_false_rg3_closure_is_rejected() -> None:
    values, custody, config_custody = _inputs()
    drifted = deepcopy(values)
    drifted["rg3"]["g3_top24_coverage"]["coverage_proven"] = True
    drifted["rg3"]["g3_top24_coverage"]["missing_block_count"] = 0
    with pytest.raises(DDM366BoxAdmissionError, match="RG3 coverage differs"):
        build_artifacts(
            drifted,
            input_custody=custody,
            config_custody=config_custody,
            available_memory_bytes=80 * (1 << 30),
        )


def test_drift_to_fake_active_pose_tube_is_rejected() -> None:
    values, custody, config_custody = _inputs()
    drifted = deepcopy(values)
    drifted["pc1"]["admission"]["tube_claim"] = True
    with pytest.raises(DDM366BoxAdmissionError, match="PC1 tube claim differs"):
        build_artifacts(
            drifted,
            input_custody=custody,
            config_custody=config_custody,
            available_memory_bytes=80 * (1 << 30),
        )


def test_above_threshold_ram_observations_are_resume_deterministic() -> None:
    values, custody, config_custody = _inputs()
    at_threshold = build_artifacts(
        values,
        input_custody=custody,
        config_custody=config_custody,
        available_memory_bytes=20 * (1 << 30),
    )
    above_threshold = build_artifacts(
        values,
        input_custody=custody,
        config_custody=config_custody,
        available_memory_bytes=80 * (1 << 30),
    )
    assert canonical_bytes(at_threshold.preflight) == canonical_bytes(
        above_threshold.preflight
    )
    memory = at_threshold.preflight["memory_preflight"]
    assert memory["passes_threshold"] is True
    assert memory["observed_relation"] == "AT_LEAST_REQUIRED"
    assert memory["exact_available_bytes_persisted"] is False


def test_committed_artifacts_are_canonical_and_cross_bound() -> None:
    receipt = json.loads((OUTPUT_ROOT / "receipt.json").read_text())
    table = json.loads((OUTPUT_ROOT / "priced_rung_table.json").read_text())
    backfill = json.loads((OUTPUT_ROOT / "rd1_162_dual_backfill.json").read_text())
    preflight = json.loads(
        (
            OUTPUT_ROOT
            / "stage_checkpoints/01_cross_chain_preflight.json"
        ).read_text()
    )
    assert canonical_bytes(table) == (OUTPUT_ROOT / "priced_rung_table.json").read_bytes()
    assert canonical_bytes(backfill) == (
        OUTPUT_ROOT / "rd1_162_dual_backfill.json"
    ).read_bytes()
    assert canonical_bytes(preflight) == (
        OUTPUT_ROOT / "stage_checkpoints/01_cross_chain_preflight.json"
    ).read_bytes()
    assert receipt["preflight"]["content_sha256"] == preflight["content_sha256"]
    assert receipt["priced_rung_table"]["content_sha256"] == table["content_sha256"]
    assert receipt["rd1_backfill"]["content_sha256"] == backfill["content_sha256"]
    assert receipt["r6_candidate_ready"] is False
    assert receipt["formulation_falsifier_reached"] is False
