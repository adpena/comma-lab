# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tools.arbitrate_ddm_ws2_warm_start_slope import (
    EXPECTED_R_STAR,
    _load_receipt,
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

    with pytest.raises(DirectDescriptionError, match="custody differs"):
        _load_receipt(path, "fixture")
