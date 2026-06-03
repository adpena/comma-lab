# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from tac.analysis.nerv_scorer_objective import (
    PEIRCE_P1_CONTEST_SCORER_GEOMETRY,
    SCORER_ONLY_OBJECTIVE_AUTHORITY,
    ContestScorerGeometry,
    ContestScorerGeometryError,
)
from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score

REPO_ROOT = Path(__file__).resolve().parents[3]


def _source(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _top_level_assignments(source: str) -> dict[str, object]:
    assignments: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                assignments[target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return assignments


def _numeric_constants_in_score_assignment(source: str) -> set[float]:
    constants: set[float] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "score" for target in node.targets):
            continue
        for child in ast.walk(node.value):
            if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)):
                constants.add(float(child.value))
    return constants


def test_formula_constants_match_upstream_evaluate_by_source_inspection() -> None:
    geometry = ContestScorerGeometry()
    eval_source = _source("upstream/evaluate.py")
    frame_utils_source = _source("upstream/frame_utils.py")
    modules_source = _source("upstream/modules.py")

    assert {100.0, 10.0, 25.0}.issubset(_numeric_constants_in_score_assignment(eval_source))

    frame_utils_assignments = _top_level_assignments(frame_utils_source)
    assert frame_utils_assignments["seq_len"] == geometry.seq_len
    assert frame_utils_assignments["camera_size"] == geometry.camera_size_wh
    assert frame_utils_assignments["segnet_model_input_size"] == (
        geometry.scorer_input_width,
        geometry.scorer_input_height,
    )
    assert "x = x[:, -1, ...] # Use only last frame" in modules_source
    assert "size=(segnet_model_input_size[1], segnet_model_input_size[0])" in modules_source

    seg_dist = 0.00125
    pose_dist = 0.00034
    archive_bytes = 123_456
    expected = 100.0 * seg_dist + math.sqrt(10.0 * pose_dist) + (25.0 * archive_bytes / ORIGINAL_VIDEO_BYTES)
    assert geometry.formula_value(
        seg_dist=seg_dist,
        pose_dist=pose_dist,
        archive_bytes=archive_bytes,
    ) == pytest.approx(expected)
    assert geometry.formula_value(
        seg_dist=seg_dist,
        pose_dist=pose_dist,
        archive_bytes=archive_bytes,
    ) == pytest.approx(
        contest_formula_score(
            seg_dist=seg_dist,
            pose_dist=pose_dist,
            archive_bytes=archive_bytes,
        )
    )
    assert geometry.rate_score_per_byte == pytest.approx(25.0 / ORIGINAL_VIDEO_BYTES)
    assert geometry.per_byte_marginal == pytest.approx(25.0 / ORIGINAL_VIDEO_BYTES)


def test_scorer_downsample_and_camera_contract() -> None:
    geometry = PEIRCE_P1_CONTEST_SCORER_GEOMETRY

    assert geometry.scorer_interpolate_size_hw == (384, 512)
    assert geometry.scorer_input_size_wh == (512, 384)
    assert geometry.camera_size_wh == (1164, 874)


def test_segnet_pair_frame_mask_is_frame0_zero_frame1_active() -> None:
    geometry = ContestScorerGeometry()

    assert geometry.segnet_pair_frame_mask() == (0.0, 1.0)
    assert geometry.segnet_pair_frame_mask(num_pairs=3) == (
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
    )


def test_posenet_pair_frame_mask_keeps_both_frames_active() -> None:
    geometry = ContestScorerGeometry()

    assert geometry.posenet_pair_frame_mask() == (1.0, 1.0)
    assert geometry.posenet_pair_frame_mask(num_pairs=2) == (1.0, 1.0, 1.0, 1.0)


def test_pose_marginal_numeric_and_fail_closed_at_nonpositive_pose() -> None:
    geometry = ContestScorerGeometry()
    d_pose = 3.4e-5

    assert geometry.pose_marginal_coefficient(d_pose) == pytest.approx(5.0 / math.sqrt(10.0 * d_pose))
    assert geometry.pose_marginal_coefficient(d_pose) == pytest.approx(271.163072273)

    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ContestScorerGeometryError):
            geometry.pose_marginal_coefficient(bad)


def test_geometry_outputs_remain_false_authority() -> None:
    payload = ContestScorerGeometry().as_false_authority_payload()

    assert payload["authority"] == "false_authority_scorer_geometry_contract_no_score_claim"
    for key in (
        "score_claim",
        "score_claim_valid",
        "frontier_score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
    ):
        assert payload[key] is False
        assert SCORER_ONLY_OBJECTIVE_AUTHORITY[key] is False
