# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_measure_ddm_pf2_test",
    ROOT / "tools" / "measure_ddm_pf2_dimension_conditioned_two_type.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return {
        "schema": "DDMPF2DimensionConditionedTwoTypeConfigV1",
        "run_id": "ddm_pf2_test_run",
        "mc1_config_path": ".omx/research/mc1_config.json",
        "mc1_config_sha256": "a" * 64,
        "mc1_receipt_path": ".omx/research/mc1_receipt.json",
        "mc1_receipt_sha256": "b" * 64,
        "pf1_receipt_path": ".omx/research/pf1_receipt.json",
        "pf1_receipt_sha256": "c" * 64,
        "g4_receipt_path": ".omx/research/g4_receipt.json",
        "g4_receipt_sha256": "d" * 64,
        "v12_receipt_path": ".omx/research/v12_receipt.json",
        "v12_receipt_sha256": "e" * 64,
        "xi_fiber_path": "/Volumes/VertigoDataTier/pact/xi.xtdl1",
        "xi_fiber_sha256": "f" * 64,
        "identity_fiber_path": "/Volumes/VertigoDataTier/pact/identity.xtdl1",
        "identity_fiber_sha256": "1" * 64,
        "xi_574_receipt_path": "/Volumes/VertigoDataTier/pact/xi_receipt.json",
        "xi_574_receipt_sha256": "2" * 64,
        "checkpoint_root": "/Volumes/VertigoDataTier/pact/ddm_pf2_test",
    }


def test_config_fails_closed_on_score_claim() -> None:
    payload = _config()
    payload["score_claim"] = True
    with pytest.raises(ValidationError):
        MODULE.PF2Config.model_validate(payload)


def test_config_requires_primary_ssd() -> None:
    payload = _config()
    payload["checkpoint_root"] = "/tmp/ddm_pf2"
    with pytest.raises(ValidationError, match="primary SSD"):
        MODULE.PF2Config.model_validate(payload)


def test_config_pins_n600_threads_and_projection() -> None:
    value = MODULE.PF2Config.model_validate(_config())
    assert value.pair_count == 600
    assert value.scorer_threads == 4
    assert value.pose_projection_alpha == 0.75


def test_rate_row_is_exact_content_control_not_train_route(
    tmp_path: Path,
) -> None:
    events = np.full((4, 3, 5), MODULE.EVENT_SENTINEL, dtype=np.uint8)
    events[0, 1, 2] = 1
    events[3, 2, 4] = 13
    row = MODULE._rate_row(
        bucket_id="road_lane__cell__transient",
        event_codes=events,
        root=tmp_path,
    )
    assert row["identical_content_parseback"] is True
    assert row["metric_status"] == MODULE.IDENTICAL_CONTENT_CODER_CONTROL
    assert row["verdict_eligible"] is True
    assert row["waterfill_eligible"] is False
    assert row["train_decision"] == "HOLD_METRIC_ACTIVE_TOLERANCE_PRICING_OWED"


def test_dr2b_reference_preserves_contest_units_and_refuses_fake_lambda() -> None:
    receipt = {
        "u1_lossy_tolerance_ladder": {
            "e2_semantic_boundary_samples": [
                {
                    "probe_id": "semantic_low_margin",
                    "measurement_status": "MEASURED",
                    "epistemic_status": "MEASURED_WINDOW_PLUS_DERIVED_EXACT_N600_REBASE",
                    "fisher_margin": {
                        "metric": "frozen_head_top1_top2_margin_over_weight_normal",
                        "top1_class": 3,
                        "top2_class": 0,
                        "margin": 1.25,
                        "head_normal_norm": 2.5,
                        "flip_distance": 0.5,
                    },
                    "n600_rebase": {
                        "delta_bytes": -12,
                        "delta_d_seg": -0.001,
                        "delta_d_pose": 0.002,
                        "joint_delta": -0.01,
                    },
                }
            ]
        }
    }
    rows = MODULE._dr2b_metric_reference(receipt)
    assert rows[0]["joint_delta_contest_units"] == -0.01
    assert rows[0]["lambda"] is None
    assert rows[0]["lambda_status"].startswith("OWED_")
