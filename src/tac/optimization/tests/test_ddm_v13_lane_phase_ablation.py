# SPDX-License-Identifier: MIT
"""Tests for the receiver-closed DDM v13 Lane phase-symbol ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tac.optimization.ddm_v13_lane_phase_ablation import (
    DDMV13LanePhaseAblationConfigV1,
    phase_only_knots,
)
from tac.optimization.direct_description_carrier_compose import LaneDriftKnotV1

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("pair_count", [64, 600])
def test_checked_in_phase_ablation_configs_are_false_authority(pair_count: int) -> None:
    path = REPO_ROOT / f".omx/research/configs/ddm_v13_lane_phase_ablation_n{pair_count}_20260722.json"
    config = DDMV13LanePhaseAblationConfigV1.model_validate_json(path.read_bytes())
    assert config.pair_count == pair_count
    assert config.scorer_batch_size == 16
    assert config.max_measurements_per_invocation == 1
    assert config.research_only is True
    assert config.execution_allowed is False
    assert config.score_claim is False
    assert config.d_seg_claim is False
    assert config.d_pose_claim is False
    assert len(config.typed_config_hash()) == 64


def test_phase_ablation_config_refuses_n600_partial_window() -> None:
    path = REPO_ROOT / ".omx/research/configs/ddm_v13_lane_phase_ablation_n600_20260722.json"
    payload = json.loads(path.read_bytes())
    payload["pair_start"] = 1
    with pytest.raises(ValidationError, match="exact pairs"):
        DDMV13LanePhaseAblationConfigV1.model_validate(payload)


def test_phase_only_knots_remove_geometry_width_and_zero_phase_rows() -> None:
    rows = (
        LaneDriftKnotV1(
            line_index=3,
            pair_index=9,
            center_c0_delta_q24=11,
            center_c3_delta_q8=-7,
            width_delta_q8=5,
            phase_delta_q8=13,
        ),
        LaneDriftKnotV1(line_index=4, pair_index=10, width_delta_q8=8),
    )
    assert phase_only_knots(rows) == (LaneDriftKnotV1(line_index=3, pair_index=9, phase_delta_q8=13),)
