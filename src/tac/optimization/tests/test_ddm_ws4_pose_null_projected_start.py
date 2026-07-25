# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tac.optimization.ddm_ws4_pose_null_projected_start import (
    CRITICAL_RATIO,
    WSEG_N600,
    WS4PoseNullError,
    build_arbitration_receipt,
    classify_projection_components,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_bytes())


def _inputs() -> dict:
    return {
        "ws1": _read(
            ".omx/research/ddm_ws1_seglex96_filtered_warmstart_20260724T022500Z/"
            "ddm_ws1_seglex96_filtered_warmstart_receipt.json"
        ),
        "ws2": _read(".omx/research/ddm_ws2_warm_start_custody_producer_receipt_20260724.json"),
        "dm2": _read(
            ".omx/research/ddm_dm2_l3_realization_race_25_rows_20260724T133300Z/"
            "ddm_dm2_l3_realization_race_receipt.json"
        ),
        "dm4": _read(
            ".omx/research/ddm_dm4_targeted_realization_cures_20260724T142722Z/"
            "ddm_dm4_targeted_realization_cures_receipt.json"
        ),
        "ws3_arbitration": _read(".omx/research/ddm_ws3_warm_start_slope_arbitration_receipt_20260724.json"),
        "cc3": _read(".omx/research/ddm_cc3_mixed_coder_receiver_receipt_20260725.json"),
        "j9_ticket": _read(".omx/research/configs/ddm_j9_366_geometry_escape_cure_20260725.json"),
    }


def test_sealed_inputs_produce_empty_lawful_projection_set() -> None:
    result = classify_projection_components(**_inputs())
    assert result["projection"]["projected_component_count"] == 0
    assert result["projection"]["dm4_projector_invoked"] is False
    assert result["temporal_suffix"]["pose_coupling"] == "POSE_BENEFICIAL"
    assert result["dm2_dm4_rows"]["wseg_foreign_key_joinable_row_indices"] == []
    assert result["cc3_composition"]["required_pose_projection"] is False


def test_dm4_cross_instance_overlay_fails_closed_if_partial_join_is_invented() -> None:
    inputs = _inputs()
    forged = copy.deepcopy(inputs["dm4"])
    row = next(row for row in forged["rows"] if row["row_index"] == 5)
    row["source_decision_path"] = "invented.json"
    row["source_decision_sha256"] = "0" * 64
    row["wseg_correction_id"] = "invented"
    inputs["dm4"] = forged
    with pytest.raises(WS4PoseNullError, match="acquired W_seg join keys"):
        classify_projection_components(**inputs)


def test_ws3_schema_fail_fast_receipt_preserves_registered_decision() -> None:
    source = _inputs()["ws3_arbitration"]
    wseg = {
        "path": "/ssd/W_seg_perp.zip",
        "sha256": WSEG_N600["archive_sha256"],
        "bytes": WSEG_N600["archive_bytes"],
    }
    wjoint = {
        "path": "/ssd/W_joint_step50_live.zip",
        "sha256": "1" * 64,
        "bytes": 138813,
        "parameter_shadow": "live_resume_state",
    }
    result = build_arbitration_receipt(
        ws3_arbitration=source,
        wseg_perp_custody=wseg,
        wjoint_step50_custody=wjoint,
        inputs={},
    )
    assert result["critical_ratio"] == CRITICAL_RATIO
    assert result["registered_slope_verdict"]["reason"] == "SEG_REGRESSION"
    assert result["selected_warm_start"] == "W_joint_step50_live"
    assert result["execution_allowed"] is False
