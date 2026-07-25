from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.run_ddm_rs1_feature_relay_solve import (
    EVIDENCE_AXIS,
    EXPECTED_BINDINGS,
    POINTER,
    RelayAdmissionError,
    RelayAuditConfig,
    build_admission_receipt,
)


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _json(path: Path, payload: object) -> dict[str, object]:
    return _write(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
    )


def _config(tmp_path: Path, *, metric: str = "categorical_margin_fisher_gram") -> Path:
    generic = _write(tmp_path / "generic.txt", b"bound")
    bindings = {name: dict(generic) for name in EXPECTED_BINDINGS}
    bindings["pre_se_locus_memo"] = _write(
        tmp_path / "pre_se.md",
        (
            b"encoder.model.blocks.1.2.se (1,144,96,128) "
            b"encoder.model.blocks.2.2.se (1,288,48,64)"
        ),
    )
    bindings["at1_tracked_receipt"] = _json(
        tmp_path / "at1_tracked.json",
        {"gaze_contraction": {"measured_relay_depths": ["scorer_plane_y", "camera_input_x"]}},
    )
    bindings["at1_gaze_atlas"] = _json(
        tmp_path / "at1_gaze.json",
        {
            "measured_relay_depths": ["scorer_plane_y", "camera_input_x"],
            "unmeasured_internal_layers_claimed": False,
        },
    )
    bindings["sn1_tracked_receipt"] = _json(
        tmp_path / "sn1.json",
        {"schema": "ddm_sn1_segnet_telemetry_asymmetry_receipt.v1"},
    )
    bindings["sn1_telemetry"] = _write(
        tmp_path / "sn1.jsonl",
        b'{"schema":"ddm_sn1_segnet_telemetry.aggregate.v1"}\n',
    )
    bindings["ms4_bundle"] = _json(
        tmp_path / "bundle.json",
        {"schema": "ddm_metric_custody_bundle.v1", "status": "COMPLETE"},
    )
    bindings["ms4_seg_metric"] = _json(
        tmp_path / "seg.json",
        {
            "schema": "ddm_seg_metric_custody.direct_scorer_intrinsic.v2",
            "head_rank": 4,
            "metric_mode": "DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT",
        },
    )
    bindings["j8f_receipt"] = _json(
        tmp_path / "j8f.json",
        {
            "verdict": "READY_TO_FIRE_DDM_EVENT_CONTINUATION",
            "score_claim": False,
            "pointer_moved": False,
            "range_gauge_projected_arm": {"archive": {"parseback_exact": True}},
        },
    )
    config = {
        "schema": "ddm_rs1_feature_relay_solve_config.v1",
        "lane_id": "lane_test",
        "run_id": "run_test",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "pointer": f"{POINTER} UNMOVED",
        "evidence_axis": EVIDENCE_AXIS,
        "main_landing_review_required": True,
        "metric_primary": metric,
        "euclidean_control_only": True,
        "realized_acceptance": (
            "END_ONLY_REALIZED_RECEIVER_PARSEBACK_UINT8_R_FROZEN_SCORERS"
        ),
        "hard_tail_block_count": 24,
        "bounded_n600_block_count": 64,
        "station_bundle": None,
        "source_bindings": bindings,
        "declared_pre_se_receipts": [
            {
                "path": "experiments/results/pre_se_locus_20260713/receipt.json",
                "bytes": 123,
                "sha256": "a" * 64,
            },
            {
                "path": (
                    "experiments/results/pre_se_multi_source_reopen_20260713/"
                    "receipt.json"
                ),
                "bytes": None,
                "sha256": "b" * 64,
            },
        ],
    }
    path = tmp_path / "a" / "b" / "configs" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config))
    return path


def test_current_shape_refuses_missing_internal_station_dynamics(tmp_path: Path) -> None:
    config = RelayAuditConfig.from_path(_config(tmp_path))
    receipt = build_admission_receipt(config)
    assert receipt["verdict"] == "BLOCKED_INTERNAL_STATION_DYNAMICS_NOT_CUSTODIED"
    assert receipt["gates"]["admitted"] is False
    assert receipt["execution"]["g3_top24_run"] == "NOT_RUN_INPUT_ADMISSION_REFUSED"
    assert receipt["execution"]["relay_radius"] is None
    assert receipt["source_interpretation"][
        "at1_internal_station_layers_explicitly_unmeasured"
    ] is True
    assert receipt["source_interpretation"][
        "sn1_telemetry_is_aggregate_without_internal_fisher_or_jacobian"
    ] is True
    assert "BLOCK2_TO_BLOCK3_MEASURED_SEGMENT_JACOBIAN" in receipt["missing_edges"]
    assert receipt["no_family_kill"] is True


def test_config_refuses_euclidean_primary_metric(tmp_path: Path) -> None:
    with pytest.raises(RelayAdmissionError, match="typed/authority contract differs"):
        RelayAuditConfig.from_path(_config(tmp_path, metric="euclidean"))


def test_source_hash_drift_fails_closed(tmp_path: Path) -> None:
    path = _config(tmp_path)
    config = RelayAuditConfig.from_path(path)
    config.source_bindings["sn1_telemetry"].resolve(config.repo_root).write_bytes(b"drift")
    with pytest.raises(RelayAdmissionError, match="identity differs"):
        build_admission_receipt(config)


def test_measured_station_bundle_unlocks_only_predictive_g3_solve(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    config_payload = json.loads(config_path.read_bytes())
    segment_ids = [
        "range_a_input_to_block2_pre_se",
        "block2_pre_se_to_block3_pre_se",
        "block3_pre_se_to_rank4_head",
    ]
    bundle = {
        "schema": "ddm_rs1_measured_station_bundle.v1",
        "measurement_status": "MEASURED",
        "research_only": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "hard_tail_block_count": 24,
        "input_station_id": "range_a_input",
        "station_chain": ["block2_pre_se", "block3_pre_se", "rank4_head"],
        "actuator_dimension": 2,
        "actuator_metric": [[0.1, 0.0], [0.0, 0.2]],
        "actuator_euclidean_control": {
            "metric_kind": "euclidean_l2_control",
            "measurement_status": "MEASURED",
        },
        "stations": [
            {
                "station_id": "block2_pre_se",
                "layer_path": "encoder.model.blocks.1.2.se.forward_pre",
                "target_delta": [1.0, -0.5],
                "metric_gram": [[3.0, 0.0], [0.0, 1.0]],
                "metric_kind": "categorical_margin_fisher_gram",
                "evidence_sha256": "1" * 64,
                "measurement_status": "MEASURED",
                "euclidean_control": {
                    "metric_kind": "euclidean_l2_control",
                    "measurement_status": "MEASURED",
                    "target_delta_l2": 1.118033988749895,
                },
            },
            {
                "station_id": "block3_pre_se",
                "layer_path": "encoder.model.blocks.2.2.se.forward_pre",
                "target_delta": [0.4, 0.8],
                "metric_gram": [[2.0, 0.25], [0.25, 1.0]],
                "metric_kind": "categorical_margin_fisher_gram",
                "evidence_sha256": "2" * 64,
                "measurement_status": "MEASURED",
                "euclidean_control": {
                    "metric_kind": "euclidean_l2_control",
                    "measurement_status": "MEASURED",
                    "target_delta_l2": 0.894427190999916,
                },
            },
            {
                "station_id": "rank4_head",
                "layer_path": "POST_R_PENULTIMATE_HEAD_QUOTIENT",
                "target_delta": [0.3],
                "metric_gram": [[4.0]],
                "metric_kind": "categorical_margin_fisher_gram",
                "evidence_sha256": "3" * 64,
                "measurement_status": "MEASURED",
                "euclidean_control": {
                    "metric_kind": "euclidean_l2_control",
                    "measurement_status": "MEASURED",
                    "target_delta_l2": 0.3,
                },
            },
        ],
        "segments": [
            {
                "segment_id": segment_ids[0],
                "source_id": "range_a_input",
                "target_id": "block2_pre_se",
                "jacobian": [[1.0, 0.0], [0.5, 1.0]],
                "evidence_sha256": "4" * 64,
                "measurement_status": "MEASURED",
            },
            {
                "segment_id": segment_ids[1],
                "source_id": "block2_pre_se",
                "target_id": "block3_pre_se",
                "jacobian": [[1.0, 0.2], [0.0, 0.5]],
                "evidence_sha256": "5" * 64,
                "measurement_status": "MEASURED",
            },
            {
                "segment_id": segment_ids[2],
                "source_id": "block3_pre_se",
                "target_id": "rank4_head",
                "jacobian": [[0.5, -0.25]],
                "evidence_sha256": "6" * 64,
                "measurement_status": "MEASURED",
            },
        ],
        "continuity_secants": [
            {
                "candidate_id": f"g3-{index:02d}",
                "segments": [
                    {
                        "segment_id": segment_ids[0],
                        "source_delta": [0.0, 0.0],
                        "realized_target_delta": [0.0, 0.0],
                        "linearized_target_delta": [0.0, 0.0],
                        "residual_l2": 0.0,
                        "measurement_status": "MEASURED",
                        "evidence_sha256": "7" * 64,
                    },
                    {
                        "segment_id": segment_ids[1],
                        "source_delta": [0.0, 0.0],
                        "realized_target_delta": [0.0, 0.0],
                        "linearized_target_delta": [0.0, 0.0],
                        "residual_l2": 0.0,
                        "measurement_status": "MEASURED",
                        "evidence_sha256": "8" * 64,
                    },
                    {
                        "segment_id": segment_ids[2],
                        "source_delta": [0.0, 0.0],
                        "realized_target_delta": [0.0],
                        "linearized_target_delta": [0.0],
                        "residual_l2": 0.0,
                        "measurement_status": "MEASURED",
                        "evidence_sha256": "9" * 64,
                    },
                ],
                "measurement_status": "MEASURED",
            }
            for index in range(24)
        ],
    }
    config_payload["station_bundle"] = _json(tmp_path / "station_bundle.json", bundle)
    config_path.write_text(json.dumps(config_payload))

    config = RelayAuditConfig.from_path(config_path)
    receipt = build_admission_receipt(config)
    assert receipt["verdict"] == "READY_FOR_G3_TOP24_REALIZED_RADIUS_MEASUREMENT"
    assert receipt["missing_edges"] == []
    assert receipt["gates"]["admitted"] is True
    assert receipt["predicted_solve"]["relay"]["predicted_only"] is True
    assert receipt["predicted_solve"]["direct"]["predicted_only"] is True
    assert receipt["predicted_solve"]["used_for_acceptance"] is False
    assert receipt["station_bundle"]["validated"] is True
    assert receipt["station_bundle"]["candidate_count"] == 24
    assert (
        receipt["execution"]["g3_top24_run"]
        == "PREDICTIVE_SOLVES_COMPLETE_AWAIT_REALIZED_ENDPOINT_LADDER"
    )
    assert (
        receipt["execution"]["bounded_n600_run"]
        == "NOT_RUN_G3_REALIZED_ENDPOINT_LADDER_OWED"
    )
