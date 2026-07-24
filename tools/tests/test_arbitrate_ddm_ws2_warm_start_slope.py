# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tools.arbitrate_ddm_ws2_warm_start_slope import (
    EXPECTED_R_STAR,
    _load_receipt,
    _refusal_payload,
    arbitrate_formulation_stop,
)


def test_expected_critical_ratio_is_the_preregistered_value() -> None:
    assert EXPECTED_R_STAR == 4.1215446777965665


def test_full_run_receipt_requires_an_exact_four_step_endpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ddm_joint_descent_full_run_receipt.v1",
                "bounded_verification": True,
                "global_step": 3,
                "baseline_verdict": {},
                "final_stage_verdict": {},
                "score_claim": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DirectDescriptionError, match="global_step=3"):
        _load_receipt(path, "fixture")


def test_refusal_payload_preserves_nonpromotion_scope(tmp_path: Path) -> None:
    producer = tmp_path / "producer.json"
    w_seg = tmp_path / "w_seg.json"
    w_joint = tmp_path / "w_joint.json"
    producer.write_text("{}\n", encoding="utf-8")
    w_seg.write_text("{}\n", encoding="utf-8")

    payload = _refusal_payload(
        reason="W_seg stopped at global_step=1",
        producer_path=producer,
        w_seg_path=w_seg,
        w_joint_path=w_joint,
    )

    assert payload["verdict"] == "REFUSE_INCOMPLETE_FOUR_STEP_WINDOW"
    assert payload["required_global_step"] == 4
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["main_review_required"] is True
    assert payload["inputs"]["W_joint_full_run_receipt"]["available"] is False


def test_ws3_formulation_stop_uses_registered_seg_regression_terminal_law(
    tmp_path: Path,
) -> None:
    producer = tmp_path / "producer.json"
    w_seg = tmp_path / "w_seg.json"
    proposal = tmp_path / "proposal.json"
    w_joint = tmp_path / "w_joint.json"
    producer.write_text(
        json.dumps(
            {
                "schema": "ddm_ws2_warm_start_custody_producer.v1",
                "sealed_batch16_endpoint_comparison": {
                    "W_seg": {
                        "sealed_batch16_d_seg": 0.024124510023328993,
                        "sealed_batch16_d_pose": 146.3649324958955,
                    },
                    "W_joint": {
                        "sealed_batch16_d_seg": 0.07051923116048177,
                        "sealed_batch16_d_pose": 36.6181847780574,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    w_seg.write_text(
        json.dumps(
            {
                "schema": "ddm_joint_descent_full_run_receipt.v1",
                "bounded_verification": True,
                "global_step": 0,
                "campaign_blocker": (
                    "BLOCKED_REALIZED_NO_PURE_PRICED_DESCENT_AFTER_SHRINK_LADDER"
                ),
                "baseline_verdict": {"d_seg": 0.024, "d_pose": 146.0},
                "score_claim": False,
            }
        ),
        encoding="utf-8",
    )
    proposal.write_text(
        json.dumps(
            {
                "schema": "ddm_joint_descent_chunked_stage_verdict.v1",
                "global_step": 1,
                "d_seg": 0.025,
                "d_pose": 145.0,
                "pure_priced_delta": {"accepted": True},
                "warm_start_component_safe_residual_admitted": False,
                "score_claim": False,
            }
        ),
        encoding="utf-8",
    )
    w_joint.write_text(
        json.dumps(
            {
                "schema": "ddm_joint_descent_full_run_receipt.v1",
                "bounded_verification": True,
                "global_step": 4,
                "baseline_verdict": {"d_seg": 0.071, "d_pose": 37.0},
                "final_stage_verdict": {"d_seg": 0.070, "d_pose": 36.0},
                "pose_finish_engage_state": {
                    "exact_verdict_steps": [0, 1, 2, 3, 4]
                },
                "score_claim": False,
            }
        ),
        encoding="utf-8",
    )

    row = arbitrate_formulation_stop(
        producer_path=producer,
        w_seg_path=w_seg,
        w_seg_terminal_proposal_path=proposal,
        w_joint_path=w_joint,
    )

    assert row["selected_warm_start"] == "W_joint"
    assert row["registered_slope_verdict"]["reason"] == "SEG_REGRESSION"
    assert row["critical_ratio"] == EXPECTED_R_STAR
    assert row["main_review_required"] is True
