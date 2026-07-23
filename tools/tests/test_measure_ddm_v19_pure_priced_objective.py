# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tools.measure_ddm_v19_pure_priced_objective import (
    DDMV19PurePricedObjectiveConfigV1,
    _delta_payload,
)


def _config(**updates: object) -> DDMV19PurePricedObjectiveConfigV1:
    values: dict[str, object] = {
        "run_id": "fixture_ddm_v19",
        "pair_ids": (447, 53, 416, 296, 547, 278, 501, 346),
        "v17_config_path": "v17_config.json",
        "v17_config_sha256": "1" * 64,
        "v17_receipt_path": "v17_receipt.json",
        "v17_receipt_sha256": "2" * 64,
        "v17_problem_path": "v17_problem.json",
        "v17_problem_sha256": "3" * 64,
        "grammar_archive_path": "grammar.zip",
        "grammar_archive_sha256": "4" * 64,
    }
    values.update(updates)
    return DDMV19PurePricedObjectiveConfigV1(**values)


def test_v19_config_seals_continuity_screen_and_false_authority() -> None:
    config = _config()
    assert config.preuint8_scales_q8 == (128, 192, 256)
    assert config.research_only is True
    assert config.execution_allowed is False
    assert config.score_claim is False
    with pytest.raises(ValueError, match="v17 continuity"):
        _config(pair_ids=tuple(reversed(config.pair_ids)))
    with pytest.raises(ValueError, match="preregistration"):
        _config(preuint8_scales_q8=(256,))


def test_delta_payload_has_no_collateral_cap_and_uses_strict_joint_price() -> None:
    before = {"d_seg": "0.025053024292", "d_pose": "162.796878513138", "archive_bytes": 135_328}
    after = {"d_seg": "0.025002797445", "d_pose": "162.797857368493", "archive_bytes": 135_529}
    result = _delta_payload(before, after)
    assert result["accepted"] is True
    assert result["joint_delta"] < 0.0
    assert result["collateral_cap_applied"] is False
    assert set(result) == {
        "seg_term",
        "pose_term",
        "rate_term",
        "joint_delta",
        "accepted",
        "delta_d_seg",
        "delta_d_pose",
        "delta_archive_bytes",
        "acceptance_authority",
        "collateral_cap_applied",
    }

