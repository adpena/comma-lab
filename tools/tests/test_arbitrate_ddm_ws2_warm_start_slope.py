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
