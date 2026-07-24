"""Tests for the fail-closed DDM MS2R R3 typed-waterfill admission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tac.ddm_campaign_evidence_join import canonical_bytes
from tac.optimization.ddm_ms2r_r3_typed_fisher_g4_waterfill import (
    CODER_ROSTER,
    METRIC_ID,
    REPRESENTATION_FAMILIES,
    R3Config,
    run,
)

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / (
    ".omx/research/configs/"
    "ddm_ms2r_r3_typed_fisher_g4_waterfill_20260724.json"
)
OUTPUT_ROOT = REPO / (
    ".omx/research/"
    "ddm_ms2r_r3_typed_fisher_g4_waterfill_20260724T211804Z"
)
OUTPUT = OUTPUT_ROOT / "receipt.json"
R2_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/"
    "ddm_ms2r_tolerance_capped_solve_r2_20260724T181428Z/"
    "stage_checkpoints/04_candidate/archive.zip"
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(OUTPUT.read_text())


def test_live_replay_matches_committed_receipt_when_external_control_exists(
    receipt: dict,
) -> None:
    if not R2_ARCHIVE.is_file():
        pytest.skip("sealed external R2 archive is unavailable")
    assert run(CONFIG, repository_root=REPO) == receipt


def test_actual_admission_preserves_r2_and_refuses_fake_composition(
    receipt: dict,
) -> None:
    assert receipt["verdict"] == (
        "BLOCKED_NO_COMPOSABLE_TYPED_ACTUATOR_STREAM; "
        "R2_CONTROL_REMAINS_CHEAPEST_RECEIVER_CLOSED_BOX_MEMBER"
    )
    box = receipt["box_result"]
    assert box["new_receiver_closed_candidate_emitted"] is False
    assert box["bytes"] == 291_205_400
    assert box["seg_errors"] == box["allowed_errors"] == 136_839
    assert box["beats_r2_bytes"] is False
    assert box["r2_measurement_batch_size"] == 16
    assert box["r3_batch32_candidate_measurement"] == (
        "NOT_RUN_NO_COMPOSABLE_CANDIDATE"
    )
    assert receipt["pointer_delta"] == "NONE"


def test_actual_admission_exposes_exact_missing_foreign_keys(
    receipt: dict,
) -> None:
    summary = receipt["oracle_admission"]["source_summary"]
    assert summary["pf2_bucket_count"] == 1_200
    assert summary["pf2_buckets_without_actuator_foreign_key"] == 1_200
    assert summary["ms4d_rows_with_actuator_secant_not_applicable"] == 1_200
    assert summary["ms4d_direct_blocks_unreachable_by_counted_coordinates"] == 25
    assert summary["ms4d_scorer_batch_size"] == 32
    assert summary["oracle_coverage"]["counts"] == {
        "WRAPPED": 14,
        "TYPED-GAP": 7,
    }


def test_pricing_keeps_accounting_slopes_separate_from_duals(
    receipt: dict,
) -> None:
    pricing = receipt["rd1_pricing"]
    assert pricing["metric_id"] == METRIC_ID
    assert pricing["cell_count"] == 162
    assert pricing["beneficial_accounting_slope_count"] == 29
    assert pricing["finite_dimension_dual_count"] == 0
    assert pricing["actionable_for_train_decision_count"] == 0
    assert len(pricing["endpoint_secants"]) == 3
    assert all(
        row["measured_endpoint_secant_bytes_per_D_improvement"] > 0.0
        for row in pricing["endpoint_secants"]
    )
    assert all(
        row["lambda_bytes_per_D_dimension"] is None
        and row["actionable_for_train_decision"] is False
        for row in pricing["cells"]
    )
    assert sum(
        row["realized_output_min_nonzero_step_uint8"] is not None
        for row in pricing["cells"]
    ) == 38


def test_representation_and_coder_races_are_not_run_on_histograms(
    receipt: dict,
) -> None:
    race = receipt["representation_race"]
    assert race["bucket_count"] == 162
    assert race["family_roster"] == list(REPRESENTATION_FAMILIES)
    assert race["coder_roster"] == list(CODER_ROSTER)
    assert race["assigned_bucket_count"] == 0
    assert race["coder_race_completed_bucket_count"] == 0
    for assignment in race["assignments"]:
        assert assignment["selected_family"] is None
        assert all(
            family["status"] == "NOT_RUN_MISSING_ADMISSIBLE_STREAM"
            for family in assignment["families"]
        )
        assert all(
            coder["bytes"] is None and coder["parseback_exact"] is False
            for coder in assignment["coder_race"]
        )
        assert assignment["fri_event_floor_bits"] is None


def test_headline_is_built_but_blocked_on_the_three_unactivated_components(
    receipt: dict,
) -> None:
    headline = receipt["headline"]
    assert headline["status"] == "HEADLINE_BLOCKED"
    assert headline["headline_eligible"] is False
    assert headline["blockers"] == [
        "TYPED_SUBPROBLEM_ALTERNATION_NOT_ACTIVE",
        "TYPED_BLOCK_ATLAS_NOT_ACTIVE",
        "PER_DIMENSION_EFFECTIVE_QUANTA_NOT_ACTIVE",
    ]
    assert headline["diagnostic_distortions"]["realized_d_seg"] == pytest.approx(
        136_839 / (600 * 512 * 384),
        abs=0.0,
    )


def test_atomic_stage_checkpoints_match_embedded_receipt_state(
    receipt: dict,
) -> None:
    assert OUTPUT.read_bytes() == canonical_bytes(receipt)
    stage_map = {
        "01_oracle_admission.json": "oracle_admission",
        "02_pricing.json": "rd1_pricing",
        "03_representation_race.json": "representation_race",
        "04_compression_telemetry.json": "compression_progress_telemetry",
        "05_headline.json": "headline",
    }
    for filename, field in stage_map.items():
        stage = json.loads(
            (OUTPUT_ROOT / "stage_checkpoints" / filename).read_text()
        )
        assert stage == receipt[field]


def test_config_refuses_euclidean_metric() -> None:
    payload = json.loads(CONFIG.read_text())
    payload["metric_id"] = "euclidean_naive"
    with pytest.raises(ValidationError):
        R3Config.model_validate(payload)
