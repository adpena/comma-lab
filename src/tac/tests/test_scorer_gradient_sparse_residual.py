from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.inflate_postprocess_surface import RawVideoShape
from tac.optimization.scorer_gradient_sparse_residual import (
    ScorerGradientSparseConfig,
    build_plan_from_gradient_selection,
    compute_pair_pose_mse_vjp,
    global_pose_score_costate_scale,
    local_pair_eval_worse_or_null,
    pair_component_delta,
    scale_pair_pose_mse_vjp_for_global_score,
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


def test_global_pose_costate_uses_sqrt_after_population_mean() -> None:
    distortions = np.asarray([1.0, 9.0, 16.0], dtype=np.float64)
    mean = float(distortions.mean())
    scale = global_pose_score_costate_scale(
        global_mean_pose_dist=mean,
        sample_count=len(distortions),
    )

    epsilon = 1.0e-6
    before = np.sqrt(10.0 * mean)
    perturbed = distortions.copy()
    perturbed[1] += epsilon
    finite_difference = (np.sqrt(10.0 * float(perturbed.mean())) - before) / epsilon

    assert finite_difference == pytest.approx(
        scale.pair_mse_vjp_scale,
        rel=1.0e-6,
    )
    # The incorrect local-pair objective would use 5/sqrt(10*d_i) and assign
    # different weights to different pairs.  The evaluator's global costate
    # assigns one shared multiplier to every raw pair-MSE VJP.
    assert scale.pair_mse_vjp_scale != pytest.approx(5.0 / np.sqrt(10.0 * distortions[1]))


def test_global_pose_costate_scales_float_vjp_without_dtype_drift() -> None:
    raw = np.ones((2, 1, 2, 3), dtype=np.float32)
    scaled, receipt = scale_pair_pose_mse_vjp_for_global_score(
        raw,
        global_mean_pose_dist=4.0,
        sample_count=600,
    )

    expected = 5.0 / (600 * np.sqrt(40.0))
    assert scaled.dtype == np.float32
    assert np.all(scaled == np.float32(expected))
    assert receipt.pair_mse_vjp_scale == pytest.approx(expected)


def test_pair_pose_mse_vjp_stays_before_population_sqrt() -> None:
    torch = pytest.importorskip("torch")

    class TinyPoseNet:
        @staticmethod
        def preprocess_input(value: object) -> object:
            return value

        @staticmethod
        def __call__(value: object) -> dict[str, object]:
            tensor = value
            first_six = tensor.mean(dim=(-2, -1)).reshape(tensor.shape[0], 6)
            return {"pose": torch.cat((first_six, first_six), dim=1)}

    baseline = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    target = np.ones((2, 2, 2, 3), dtype=np.uint8)
    gradient, metrics = compute_pair_pose_mse_vjp(
        baseline_pair_hwc=baseline,
        target_pair_hwc=target,
        posenet=TinyPoseNet(),
        device="cpu",
    )

    assert gradient.dtype == np.float32
    assert gradient.shape == baseline.shape
    assert metrics["pair_pose_dist"] == pytest.approx(1.0)
    assert metrics["vjp_objective"] == ("UPSTREAM_PER_SAMPLE_POSE_MSE_BEFORE_GLOBAL_MEAN_AND_SQRT")
    # Six pose coordinates, each averaging four pixels, then MSE over six:
    # d/dpixel mean((mean(pixel)-1)^2) = -2 / (6 * 4).
    assert np.all(gradient == pytest.approx(-2.0 / 24.0))


@pytest.mark.parametrize(
    ("mean_pose_dist", "sample_count"),
    [
        (0.0, 600),
        (-1.0, 600),
        (float("nan"), 600),
        (1.0, 0),
        (1.0, True),
    ],
)
def test_global_pose_costate_refuses_non_population_coordinates(
    mean_pose_dist: float,
    sample_count: int,
) -> None:
    with pytest.raises(ValueError):
        global_pose_score_costate_scale(
            global_mean_pose_dist=mean_pose_dist,
            sample_count=sample_count,
        )
