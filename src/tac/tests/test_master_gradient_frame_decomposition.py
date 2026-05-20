from __future__ import annotations

import json
import math

import numpy as np

from tac.master_gradient_frame_decomposition import (
    FrameDecompositionConfig,
    MasterGradientFrameDecompositionError,
    decompose_per_pair_gradient_to_frames,
    json_ready_decomposition,
    render_markdown,
)
from tac.cathedral_consumers.per_frame_sensitivity_consumer import consume_candidate


def test_non_overlapping_projection_respects_scorer_input_topology() -> None:
    arr = np.zeros((2, 2, 3), dtype=np.float64)
    arr[0, 0, :] = [1.0, 2.0, 3.0]
    arr[1, 1, :] = [4.0, 6.0, 8.0]

    payload = decompose_per_pair_gradient_to_frames(
        arr,
        axis_coefficients=(10.0, 1.0, 0.5),
        config=FrameDecompositionConfig(topology="non_overlapping", top_k_frames=4),
    )

    frame_axis = payload["frame_axis_l1"]
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["n_pairs"] == 2
    assert payload["n_frames"] == 4
    assert payload["scorer_input_topology"]["segnet"] == "last_frame_only"
    assert payload["scorer_input_topology"]["posenet"] == "both_frames"
    np.testing.assert_allclose(frame_axis[0], [0.0, 1.0, 0.75])
    np.testing.assert_allclose(frame_axis[1], [10.0, 1.0, 0.75])
    np.testing.assert_allclose(frame_axis[2], [0.0, 3.0, 2.0])
    np.testing.assert_allclose(frame_axis[3], [40.0, 3.0, 2.0])
    assert payload["conservation_ok"] is True
    assert payload["top_frames"][0]["frame_index"] == 3
    assert payload["top_frames"][0]["rank"] == 1


def test_sliding_projection_is_available_but_marked_exploratory() -> None:
    arr = np.ones((1, 2, 3), dtype=np.float64)

    payload = decompose_per_pair_gradient_to_frames(
        arr,
        axis_coefficients=(1.0, 1.0, 1.0),
        config=FrameDecompositionConfig(topology="sliding", top_k_frames=3),
    )

    assert payload["n_frames"] == 3
    assert "exploratory sliding" in payload["scorer_input_topology"]["upstream_pairing"]
    np.testing.assert_allclose(payload["frame_axis_l1"][0], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(payload["frame_axis_l1"][1], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(payload["frame_axis_l1"][2], [1.0, 0.5, 0.5])
    assert payload["conservation_ok"] is True


def test_projection_rejects_aggregate_tensor_shape() -> None:
    arr = np.zeros((5, 3), dtype=np.float64)

    try:
        decompose_per_pair_gradient_to_frames(arr, axis_coefficients=(1.0, 1.0, 1.0))
    except MasterGradientFrameDecompositionError as exc:
        assert "N_bytes, N_pairs, 3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape rejection")


def test_projection_rejects_negative_or_partial_config() -> None:
    arr = np.zeros((1, 1, 3), dtype=np.float64)

    try:
        decompose_per_pair_gradient_to_frames(arr, axis_coefficients=(1.0, -1.0, 1.0))
    except MasterGradientFrameDecompositionError as exc:
        assert "non-negative" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected coefficient rejection")

    try:
        FrameDecompositionConfig(pose_first_frame_share=math.nan).validate()
    except MasterGradientFrameDecompositionError as exc:
        assert "pose_first_frame_share" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected config rejection")


def test_json_and_markdown_outputs_are_operator_safe() -> None:
    arr = np.ones((1, 1, 3), dtype=np.float64)
    payload = decompose_per_pair_gradient_to_frames(arr, axis_coefficients=(1.0, 1.0, 1.0))

    ready = json_ready_decomposition(payload)
    encoded = json.dumps(ready)
    md = render_markdown(payload)

    assert "frame_axis_l1" not in ready
    assert "score_claim" in encoded
    assert "Master-Gradient Per-Frame Decomposition" in md


def test_cathedral_consumer_surfaces_frame_order_without_score_authority(tmp_path) -> None:
    arr = np.ones((1, 2, 3), dtype=np.float64)
    payload = json_ready_decomposition(
        decompose_per_pair_gradient_to_frames(
            arr,
            axis_coefficients=(1.0, 1.0, 1.0),
            config=FrameDecompositionConfig(top_k_frames=2),
        )
    )
    path = tmp_path / "frame_decomp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    verdict = consume_candidate({"per_frame_decomposition_json": str(path)})

    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    assert verdict["score_claim"] is False
    assert verdict["consumer_signal_kind"] == "per_frame_sensitivity_routing"
    assert verdict["recommended_frame_order"]
