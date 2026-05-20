from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.inflate_postprocess_surface import RawVideoShape
from tac.optimization.scorer_gradient_sparse_residual import (
    ScorerGradientSparseConfig,
    build_plan_from_gradient_selection,
    local_pair_eval_worse_or_null,
    pair_component_delta,
    select_budgeted_gradient_residuals,
    select_gradient_aligned_residuals,
)


def test_select_gradient_aligned_residuals_keeps_descent_pixels_only() -> None:
    shape = RawVideoShape(frames=4, height=1, width=3, channels=3)
    gradient = np.zeros((2, 1, 3, 3), dtype=np.float32)
    residual = np.zeros((2, 1, 3, 3), dtype=np.int16)
    gradient[0, 0, 0] = [2.0, 0.0, 0.0]
    residual[0, 0, 0] = [-1, 0, 0]
    gradient[0, 0, 1] = [2.0, 0.0, 0.0]
    residual[0, 0, 1] = [1, 0, 0]
    gradient[1, 0, 2] = [-4.0, 0.0, 0.0]
    residual[1, 0, 2] = [1, 0, 0]

    selection = select_gradient_aligned_residuals(
        gradient=gradient,
        residual=residual,
        shape=shape,
        frame_indices=[2, 3],
        top_k_pixels=2,
        max_abs_delta=1,
    )

    assert selection.indices.tolist() == [11, 6]
    assert selection.values.tolist() == [[1, 0, 0], [-1, 0, 0]]
    assert selection.candidate_count == 3
    assert selection.rejected_non_descent_count == 1


def test_gradient_selection_builds_charged_sparse_plan() -> None:
    shape = RawVideoShape(frames=2, height=1, width=2, channels=3)
    gradient = np.array([[[[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]], dtype=np.float32)
    residual = np.array([[[[1, 0, 0], [0, -1, 0]]]], dtype=np.int16)
    selection = select_gradient_aligned_residuals(
        gradient=gradient,
        residual=residual,
        shape=shape,
        frame_indices=[1],
        top_k_pixels=2,
        max_abs_delta=1,
    )

    plan = build_plan_from_gradient_selection(
        selection=selection,
        shape=shape,
        config=ScorerGradientSparseConfig(top_k_pixels=2, rate_cap_bytes=2048),
    )

    assert plan.sparse["n_kept"] == 2
    assert plan.packed_bytes > 0
    assert plan.selected_gain_sum > 0.0


def test_budgeted_gradient_residuals_waterfill_by_utility_per_byte() -> None:
    shape = RawVideoShape(frames=2, height=1, width=4, channels=3)
    gradient = np.zeros((1, 1, 4, 3), dtype=np.float32)
    residual = np.zeros((1, 1, 4, 3), dtype=np.int16)
    gradient[0, 0, 0] = [-8.0, 0.0, 0.0]
    residual[0, 0, 0] = [1, 0, 0]
    gradient[0, 0, 1] = [-7.0, 0.0, 0.0]
    residual[0, 0, 1] = [1, 0, 0]
    gradient[0, 0, 2] = [-3.0, 0.0, 0.0]
    residual[0, 0, 2] = [1, 0, 0]
    gradient[0, 0, 3] = [10.0, 0.0, 0.0]
    residual[0, 0, 3] = [1, 0, 0]

    selection = select_budgeted_gradient_residuals(
        gradient=gradient,
        residual=residual,
        shape=shape,
        frame_indices=[1],
        top_k_pixels=3,
        max_abs_delta=1,
        saliency_mask=np.array([[[1.0, 1.0, 0.0, 1.0]]], dtype=np.float32),
        byte_costs=np.array([[[4.0, 1.0, 1.0, 1.0]]], dtype=np.float32),
        budget_limit=4.0,
    )

    assert selection.indices.tolist() == [5]
    assert selection.values.tolist() == [[1, 0, 0]]
    assert selection.candidate_count == 4
    assert selection.rejected_non_descent_count == 1
    assert selection.rejected_by_saliency_count == 1
    assert selection.budget_used == 1.0


def test_gradient_selection_rejects_mismatched_frame_space() -> None:
    shape = RawVideoShape(frames=2, height=2, width=2, channels=3)
    gradient = np.zeros((1, 1, 1, 3), dtype=np.float32)
    residual = np.ones((1, 1, 1, 3), dtype=np.int16)

    with pytest.raises(ValueError, match="frame-space shape"):
        select_gradient_aligned_residuals(
            gradient=gradient,
            residual=residual,
            shape=shape,
            frame_indices=[0],
            top_k_pixels=1,
            max_abs_delta=1,
        )

    with pytest.raises(ValueError, match="frame-space shape"):
        select_budgeted_gradient_residuals(
            gradient=gradient,
            residual=residual,
            shape=shape,
            frame_indices=[0],
            top_k_pixels=1,
            max_abs_delta=1,
        )


def test_gradient_selection_rejects_out_of_range_frame_indices() -> None:
    shape = RawVideoShape(frames=2, height=1, width=1, channels=3)
    gradient = np.zeros((1, 1, 1, 3), dtype=np.float32)
    residual = np.ones((1, 1, 1, 3), dtype=np.int16)

    with pytest.raises(ValueError, match="frame_index out of range"):
        select_gradient_aligned_residuals(
            gradient=gradient,
            residual=residual,
            shape=shape,
            frame_indices=[2],
            top_k_pixels=1,
            max_abs_delta=1,
        )

    with pytest.raises(ValueError, match="frame_index out of range"):
        select_budgeted_gradient_residuals(
            gradient=gradient,
            residual=residual,
            shape=shape,
            frame_indices=[-1],
            top_k_pixels=1,
            max_abs_delta=1,
        )


def test_local_pair_veto_requires_no_improvement_and_some_regression() -> None:
    delta = pair_component_delta(
        {"pose_dist": 0.10, "seg_dist": 0.20},
        {"pose_dist": 0.11, "seg_dist": 0.20},
    )
    assert local_pair_eval_worse_or_null(delta) is True

    mixed = pair_component_delta(
        {"pose_dist": 0.10, "seg_dist": 0.20},
        {"pose_dist": 0.09, "seg_dist": 0.21},
    )
    assert local_pair_eval_worse_or_null(mixed) is False

    null = pair_component_delta(
        {"pose_dist": 0.10, "seg_dist": 0.20},
        {"pose_dist": 0.10, "seg_dist": 0.20},
    )
    assert local_pair_eval_worse_or_null(null) is False
