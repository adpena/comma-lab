from __future__ import annotations

import json
import math

import numpy as np

from tac.optimization.frame_pair_curriculum import (
    CurriculumConfig,
    FramePairCurriculumError,
    build_frame_pair_curriculum,
    render_markdown,
)


def test_curriculum_maps_non_overlapping_frames_to_scorer_roles() -> None:
    frame_axis = np.array(
        [
            [0.0, 1.0, 0.0],
            [10.0, 1.0, 0.0],
            [0.0, 5.0, 0.0],
            [2.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )

    payload = build_frame_pair_curriculum(
        frame_axis,
        config=CurriculumConfig(top_k_frames=4, top_k_pairs=2, sampling_floor=0.0),
    )

    assert payload["score_claim"] is False
    assert payload["frame_rows"][0]["frame_role"] == "pair_first_pose_only"
    assert payload["frame_rows"][1]["frame_role"] == "pair_last_segnet_and_pose"
    assert payload["pair_rows"][0]["first_frame"] == 0
    assert payload["pair_rows"][0]["last_frame"] == 1
    assert payload["pair_rows"][0]["seg_l1"] == 10.0
    assert payload["pair_rows"][0]["pose_l1"] == 2.0
    assert payload["pair_rows"][1]["pose_l1"] == 10.0
    assert math.isclose(payload["sampling_probability_checks"]["seg_frame_prob_sum"], 1.0)
    assert math.isclose(payload["sampling_probability_checks"]["pose_frame_prob_sum"], 1.0)
    assert math.isclose(payload["sampling_probability_checks"]["pair_pose_prob_sum"], 1.0)
    assert any(
        layer["mask_family"] == "segnet_boundary_or_high_gradient_ring"
        for layer in payload["adjustment_layers"]
    )
    assert any(
        layer["mask_family"] == "full_frame_low_frequency_or_horizon_weighted"
        for layer in payload["adjustment_layers"]
    )


def test_curriculum_carries_response_policy_blocking_sparse_widening() -> None:
    frame_axis = np.ones((2, 3), dtype=np.float64)
    response_plan = {
        "prohibitions": [
            {"rule": "do_not_widen_coordinate_sparse_residual_sidecar", "reason": "over budget"}
        ]
    }

    payload = build_frame_pair_curriculum(frame_axis, response_plan=response_plan)

    assert payload["response_policy"]["charged_sparse_residual_widening_allowed"] is False
    assert (
        payload["adjustment_layers"][0]["byte_contract"]["preferred_first_materialization"]
        == "prefer_byte_neutral_masked_runtime_knobs_before_charged_sparse_pixels"
    )


def test_curriculum_rejects_odd_frame_non_overlapping_input() -> None:
    frame_axis = np.ones((3, 3), dtype=np.float64)

    try:
        build_frame_pair_curriculum(frame_axis)
    except FramePairCurriculumError as exc:
        assert "even frame count" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected odd frame rejection")


def test_curriculum_json_and_markdown_are_safe() -> None:
    payload = build_frame_pair_curriculum(np.ones((2, 3), dtype=np.float64))
    encoded = json.dumps(payload)
    md = render_markdown(payload)

    assert "score_claim" in encoded
    assert "LL Frame/Pair Curriculum" in md
