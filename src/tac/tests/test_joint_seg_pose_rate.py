from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.optimization.joint_seg_pose_rate import (
    JointSolveError,
    MarginBandConfig,
    derive_hyperplane_channel_band,
    derive_margin_rgb_band,
    generated_fill_predictor,
    pose_score_derivative,
    solve_interval_frame,
    solve_measured_waterfill,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tools import measure_joint_seg_pose_rate as measurement_tool


def _operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(camera_h=6, camera_w=8, scorer_h=3, scorer_w=4)


def _bounded_unreachable_numerator_case() -> tuple[
    DisjointResizeOperator, np.ndarray, int, np.ndarray, np.ndarray
]:
    op = DisjointResizeOperator.build(camera_h=3, camera_w=9, scorer_h=1, scorer_w=4)
    source = np.zeros((3, 9, 1), dtype=np.uint8)
    source_numerators, denominator = op.apply_numerators(source)
    predictor = source.copy()
    predictor[1, 0, 0] = 1
    band = np.zeros((1, 4, 1), dtype=np.float64)
    band[0, 0, 0] = 2 / denominator
    return op, source_numerators, denominator, band, predictor


def test_margin_band_is_cached_margin_over_explicit_lipschitz() -> None:
    margin = np.array([[1.0, 2.0]])
    winner = np.array([[0, 1]])
    rival = np.array([[1, 2]])
    pullback = np.array([[[1.0, 2.0, 4.0], [2.0, 1.0, 0.0]]])
    norms = np.array([[2.0, 4.0]])
    got = derive_hyperplane_channel_band(
        margin, winner, rival, pullback, norms,
        MarginBandConfig(scale=1.0, local_lipschitz=1.0, max_rgb_radius=2.0),
    )
    np.testing.assert_allclose(got.feature_flip_distance, [[0.5, 0.5]])
    np.testing.assert_allclose(got.channel_radii[0, 0], [1 / 6, 1 / 12, 1 / 24])
    np.testing.assert_allclose(got.channel_radii[0, 1], [1 / 12, 1 / 6, 2.0])


def test_measured_local_lipschitz_field_uses_stored_q_factorization() -> None:
    margin = np.array([[6.0, 6.0]])
    winner = np.array([[0, 1]])
    rival = np.array([[1, 2]])
    q = np.array([[[1.0, 0.5, 0.0], [1.0, 0.5, 0.25]]])
    norms = np.ones((1, 2))
    local_lipschitz = np.array([[2.0, 4.0]])
    got = derive_hyperplane_channel_band(
        margin,
        winner,
        rival,
        q,
        norms,
        MarginBandConfig(scale=1.0, local_lipschitz=999.0, max_rgb_radius=8.0),
        local_lipschitz_field=local_lipschitz,
    )
    np.testing.assert_allclose(got.channel_radii[0, 0], [1.0, 2.0, 8.0])
    np.testing.assert_allclose(got.channel_radii[0, 1], [0.5, 1.0, 2.0])


def test_measured_local_lipschitz_field_fails_closed_on_bad_geometry() -> None:
    with pytest.raises(JointSolveError, match="Lipschitz field"):
        derive_hyperplane_channel_band(
            np.ones((2, 2)),
            np.zeros((2, 2), dtype=np.int8),
            np.ones((2, 2), dtype=np.int8),
            np.ones((2, 2, 3)),
            np.ones((2, 2)),
            MarginBandConfig(scale=1.0, local_lipschitz=1.0, max_rgb_radius=2.0),
            local_lipschitz_field=np.ones((1, 2)),
        )


def test_positive_isotropic_band_is_forbidden() -> None:
    with pytest.raises(JointSolveError, match="isotropic"):
        derive_margin_rgb_band(np.ones((2, 2)), MarginBandConfig(scale=1.0, local_lipschitz=1.0, max_rgb_radius=2.0))


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_margin_band_fails_closed(bad: float) -> None:
    with pytest.raises(JointSolveError):
        derive_margin_rgb_band(np.ones((2, 2)), MarginBandConfig(scale=bad, local_lipschitz=1.0, max_rgb_radius=1.0))


def test_generated_fill_uses_target_only_and_has_camera_geometry() -> None:
    op = _operator()
    target = np.arange(3 * 4 * 3, dtype=np.float64).reshape(3, 4, 3)
    pred = generated_fill_predictor(op, target)
    assert pred.shape == (6, 8, 3)
    assert pred.dtype == np.uint8
    np.testing.assert_allclose(op.apply(pred), np.rint(target), atol=1e-12)


def test_zero_band_joint_frame_is_exact_and_custodied() -> None:
    op = _operator()
    source = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    target_num, den = op.apply_numerators(source)
    target = target_num.astype(np.float64) / den
    pred = generated_fill_predictor(op, target)
    solved = solve_interval_frame(op, target_num, den, np.zeros((3, 4)), predictor=pred)
    got_num, got_den = op.apply_numerators(solved.frame)
    assert got_den == den
    np.testing.assert_array_equal(got_num, target_num)
    assert solved.telemetry.maximum_projection_error == 0.0
    assert solved.telemetry.exact_blocks == target_num.size
    assert solved.conservative_exact_source_numerator_fallback_map is None


def test_positive_band_moves_only_inside_interval() -> None:
    op = _operator()
    source = np.random.default_rng(7).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    target_num, den = op.apply_numerators(source)
    target = target_num.astype(np.float64) / den
    pred = generated_fill_predictor(op, target)
    band = np.full((3, 4), 4.0)
    solved = solve_interval_frame(op, target_num, den, band, predictor=pred)
    delta = np.abs(solved.chosen_numerators - target_num)
    assert np.all(delta <= solved.band_radius_numerators)
    assert solved.telemetry.maximum_projection_error == 0.0


def test_bounded_unreachable_numerator_default_still_fails_closed() -> None:
    op, source_numerators, denominator, band, predictor = (
        _bounded_unreachable_numerator_case()
    )
    with pytest.raises(JointSolveError, match="INFEASIBLE_EXHAUSTIVE"):
        solve_interval_frame(
            op,
            source_numerators,
            denominator,
            band,
            predictor=predictor,
        )


def test_conservative_exact_source_numerator_fallback_is_opt_in_and_custodied() -> None:
    op, source_numerators, denominator, band, predictor = (
        _bounded_unreachable_numerator_case()
    )
    solved = solve_interval_frame(
        op,
        source_numerators,
        denominator,
        band,
        predictor=predictor,
        conservative_exact_source_numerator_fallback=True,
    )
    np.testing.assert_array_equal(solved.chosen_numerators, source_numerators)
    assert solved.conservative_exact_source_numerator_fallback_map is not None
    assert np.count_nonzero(solved.conservative_exact_source_numerator_fallback_map) == 1
    assert solved.conservative_exact_source_numerator_fallback_map[0, 0, 0] == 1
    assert solved.binding_map[0, 0, 0] == 0
    assert solved.telemetry.target_numerator_l1_shift == 0
    assert solved.telemetry.binding_counts["slack"] == source_numerators.size
    assert "conservative_exact_source_numerator_fallback" not in solved.telemetry.__dict__


def test_conservative_exact_source_numerator_fallback_keeps_budget_fail_closed() -> None:
    op, source_numerators, denominator, band, predictor = (
        _bounded_unreachable_numerator_case()
    )
    with pytest.raises(JointSolveError, match="exhausted node budget"):
        solve_interval_frame(
            op,
            source_numerators,
            denominator,
            band,
            predictor=predictor,
            max_nodes_per_block=1,
            conservative_exact_source_numerator_fallback=True,
        )


def test_conservative_exact_source_numerator_fallback_requires_exact_center() -> None:
    op, source_numerators, denominator, band, predictor = (
        _bounded_unreachable_numerator_case()
    )
    source_numerators[0, 0, 0] = 2
    band.fill(0.0)
    with pytest.raises(JointSolveError, match="INFEASIBLE_EXHAUSTIVE"):
        solve_interval_frame(
            op,
            source_numerators,
            denominator,
            band,
            predictor=predictor,
            conservative_exact_source_numerator_fallback=True,
        )


def test_pose_derivative_and_crossover() -> None:
    assert pose_score_derivative(0.0) == math.inf
    assert pose_score_derivative(2.5e-4) == pytest.approx(100.0)


def test_hard_verdict_uses_last_frame_seg_logits_and_full_pair_pose() -> None:
    class FakeSegNet:
        @staticmethod
        def preprocess_input(pair: torch.Tensor) -> torch.Tensor:
            assert tuple(pair.shape) == (1, 2, 3, 2, 3)
            return pair[:, -1]

        @staticmethod
        def __call__(frames: torch.Tensor) -> torch.Tensor:
            assert tuple(frames.shape) == (1, 3, 2, 3)
            logits = torch.zeros((1, 2, *frames.shape[-2:]), dtype=torch.float32)
            selected_class = int(frames[0, 0, 0, 0].item() > 0.5)
            logits[0, selected_class] = 1.0
            return logits

    class FakePoseNet:
        seen_shape: tuple[int, ...] | None = None

        @staticmethod
        def preprocess_input(pair: torch.Tensor) -> torch.Tensor:
            return pair

        def __call__(self, pair: torch.Tensor) -> dict[str, torch.Tensor]:
            self.seen_shape = tuple(pair.shape)
            pose = torch.zeros((pair.shape[0], 6), dtype=torch.float32)
            pose[:, 0] = pair[:, 0].mean(dim=(1, 2, 3)) + pair[:, 1].mean(
                dim=(1, 2, 3)
            )
            return {"pose": pose}

    posenet = FakePoseNet()
    frame0 = np.zeros((2, 3, 3), dtype=np.uint8)
    frame1 = np.ones((2, 3, 3), dtype=np.uint8)
    gt_f1_labels = np.ones((2, 3), dtype=np.int64)
    target_pose = np.zeros(6, dtype=np.float64)
    target_pose[0] = 1.0

    verdict, winner, rival = measurement_tool._hard_verdict(
        FakeSegNet(),
        posenet,
        torch,
        frame0,
        frame1,
        gt_f1_labels,
        target_pose,
    )

    assert verdict["d_seg"] == 0.0
    assert verdict["d_pose"] == 0.0
    np.testing.assert_array_equal(winner, gt_f1_labels)
    np.testing.assert_array_equal(rival, np.zeros_like(gt_f1_labels))
    assert posenet.seen_shape == (1, 2, 3, 2, 3)


def test_hard_verdict_rejects_flattened_two_frame_seg_logits() -> None:
    class FakeSegNet:
        @staticmethod
        def preprocess_input(pair: torch.Tensor) -> torch.Tensor:
            return pair.reshape(-1, *pair.shape[2:])

        @staticmethod
        def __call__(frames: torch.Tensor) -> torch.Tensor:
            return torch.zeros((2, 2, *frames.shape[-2:]), dtype=torch.float32)

    with pytest.raises(JointSolveError, match=r"batch.*equal.*B=1"):
        measurement_tool._hard_verdict(
            FakeSegNet(),
            object(),
            torch,
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.ones((2, 3, 3), dtype=np.uint8),
            np.ones((2, 3), dtype=np.int64),
            np.zeros(6, dtype=np.float64),
        )


def test_vjp_winner_mismatch_writes_durable_refusal(tmp_path: Path) -> None:
    winner = np.zeros((2, 3), dtype=np.int64)
    custodied_winner = winner.copy()
    custodied_winner[1, 2] = 1
    rival = np.ones((2, 3), dtype=np.int64)
    custodied_rival = rival.copy()
    custodied_rival[0, 1] = 2

    with pytest.raises(JointSolveError, match=r"winner differs.*1 pixels"):
        measurement_tool._gate_vjp_arrangement(
            stage_dir=tmp_path,
            pair_id=2,
            config_sha256="a" * 64,
            source_control={"d_seg": 0.0},
            custodied_winner=custodied_winner,
            custodied_rival=custodied_rival,
            hard_oracle_winner=winner,
            hard_oracle_rival=rival,
        )

    refusal = json.loads((tmp_path / "pair_0002.hard_oracle_refusal.json").read_text())
    assert refusal["custodied_vs_hard_oracle_winner_disagreement_pixels"] == 1
    assert refusal["inference_vs_vjp_rival_disagreement_pixels"] == 1
    assert "proposal-only" in refusal["verdict_scope"]


def test_vjp_rival_only_drift_is_counted_and_admitted(tmp_path: Path) -> None:
    winner = np.zeros((2, 3), dtype=np.int64)
    rival = np.ones((2, 3), dtype=np.int64)
    custodied_rival = rival.copy()
    custodied_rival[1, 2] = 2

    telemetry = measurement_tool._gate_vjp_arrangement(
        stage_dir=tmp_path,
        pair_id=2,
        config_sha256="a" * 64,
        source_control={"d_seg": 0.0},
        custodied_winner=winner.copy(),
        custodied_rival=custodied_rival,
        hard_oracle_winner=winner,
        hard_oracle_rival=rival,
    )

    assert telemetry["custodied_vs_hard_oracle_winner_disagreement_pixels"] == 0
    assert telemetry["inference_vs_vjp_rival_disagreement_pixels"] == 1
    assert "not winner/Seg authority" in telemetry[
        "inference_vs_vjp_rival_disagreement_scope"
    ]
    assert not (tmp_path / "pair_0002.hard_oracle_refusal.json").exists()


def test_repair_pose_proposal_shrinks_linear_budget_but_records_full_limit() -> None:
    pose_j_y = np.zeros((6, 2, 1, 1, 3), dtype=np.float64)
    pose_j_y[:, 0, 0, 0, 0] = 1.0
    frame0_direction = np.array([[[1.0, 0.0, 0.0]]])
    full_tau_pose = 0.25
    shrink = 0.5

    proposal = measurement_tool._repair_pose_linear_proposal(
        pose_j_y,
        np.zeros_like(frame0_direction),
        frame0_direction,
        np.zeros(6, dtype=np.float64),
        full_tau_pose,
        shrink,
    )

    assert proposal["feasible"] is True
    assert proposal["effective_linear_pose_budget"] == pytest.approx(0.125)
    assert proposal["full_hard_oracle_pose_limit"] == full_tau_pose
    assert proposal["planned_predictor_step_pose_mse"] == pytest.approx(0.125)
    assert proposal["selected_step"] == pytest.approx(math.sqrt(0.125))
    assert proposal["selected_step"] < shrink


def test_bindingness_sidecar_persists_full_maps_and_validates_resume(
    tmp_path: Path,
) -> None:
    binding0 = np.array([[[0, 1, 2], [2, 1, 0]]], dtype=np.uint8)
    binding1 = np.array([[[2, 0, 1], [0, 1, 2]]], dtype=np.uint8)
    fallback0 = np.array([[[0, 1, 0], [1, 0, 1]]], dtype=np.uint8)
    arrays, fallback_present = measurement_tool._bindingness_arrays(
        binding0=binding0,
        binding1=binding1,
        positive_seg_radius=np.array([[True, False]]),
        fallback0=fallback0,
        fallback1=None,
    )
    path = tmp_path / "pair_0003.bindingness.npz"
    reference = measurement_tool._publish_bindingness_sidecar(
        path,
        pair_id=3,
        config_sha256="c" * 64,
        arrays=arrays,
        fallback_present=fallback_present,
    )
    metadata = measurement_tool._validate_bindingness_sidecar(
        reference,
        pair_id=3,
        config_sha256="c" * 64,
        expected_arrays=arrays,
    )

    assert metadata["map_semantics"]["frame_binding_maps"] == (
        "0 slack, 1 lower, 2 upper interval binding"
    )
    assert metadata["fallback_map_present_in_solver_result"] == {
        "frame0_exact_source_fallback": True,
        "frame1_exact_source_fallback": False,
    }
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame0_binding_map"], binding0)
        np.testing.assert_array_equal(data["frame1_binding_map"], binding1)
        np.testing.assert_array_equal(
            data["positive_seg_radius_map"],
            np.broadcast_to(np.array([[[True], [False]]]), binding1.shape),
        )
        np.testing.assert_array_equal(
            data["frame0_exact_source_fallback_map"], fallback0.astype(bool)
        )
        assert not np.any(data["frame1_exact_source_fallback_map"])

    assert measurement_tool._publish_bindingness_sidecar(
        path,
        pair_id=3,
        config_sha256="c" * 64,
        arrays=arrays,
        fallback_present=fallback_present,
    ) == reference
    bad_reference = {**reference, "sha256": "0" * 64}
    with pytest.raises(JointSolveError, match="byte custody"):
        measurement_tool._validate_bindingness_sidecar(
            bad_reference,
            pair_id=3,
            config_sha256="c" * 64,
        )


def test_pose_bindingness_classifies_inactive_active_and_zero_tau() -> None:
    inactive = measurement_tool._pose_constraint_activity(
        pose_proposal={
            "feasible": True,
            "selected_step": 1.0,
            "planned_predictor_step_pose_mse": 1e-8,
            "effective_linear_pose_budget": 1e-4,
        },
        hard_d_pose=2e-8,
        tau_pose=1e-4,
        repair_shrink=1.0,
    )
    assert inactive["linear"]["classification"] == "inactive_slack"
    assert inactive["hard"]["classification"] == "inactive_slack"
    assert inactive["hard"]["d_pose_over_tau_pose"] == pytest.approx(2e-4)
    assert inactive["preregistered_hypothesis_at_or_below_crossover"] is True
    assert inactive["preregistered_hypothesis_confirmed_for_pair"] is True

    active = measurement_tool._pose_constraint_activity(
        pose_proposal={
            "feasible": True,
            "selected_step": 0.25,
            "planned_predictor_step_pose_mse": 1e-4,
            "effective_linear_pose_budget": 1e-4,
        },
        hard_d_pose=1e-4,
        tau_pose=1e-4,
        repair_shrink=1.0,
    )
    assert active["linear"]["classification"] == "active"
    assert active["hard"]["classification"] == "active"
    assert active["preregistered_hypothesis_confirmed_for_pair"] is False

    linear_only = measurement_tool._pose_constraint_activity(
        pose_proposal={
            "feasible": True,
            "selected_step": 0.25,
            "planned_predictor_step_pose_mse": 5e-5,
            "effective_linear_pose_budget": 1e-4,
        },
        hard_d_pose=2e-5,
        tau_pose=1e-4,
        repair_shrink=1.0,
    )
    assert linear_only["linear"]["classification"] == "active"
    assert linear_only["hard"]["classification"] == "inactive_slack"
    assert linear_only["preregistered_hypothesis_confirmed_for_pair"] is True
    assert linear_only["linear_proposer_inactive_for_pair"] is False
    assert linear_only["linear_and_hard_inactive_for_pair"] is False

    zero = measurement_tool._pose_constraint_activity(
        pose_proposal={
            "feasible": True,
            "selected_step": 1.0,
            "planned_predictor_step_pose_mse": 0.0,
            "effective_linear_pose_budget": 0.0,
        },
        hard_d_pose=0.0,
        tau_pose=0.0,
        repair_shrink=1.0,
    )
    assert zero["hard"]["classification"] == "active_zero_tau_equality"
    assert zero["hard"]["d_pose_over_tau_pose"] is None
    assert zero["preregistered_hypothesis_confirmed_for_pair"] is False


def test_pose_bindingness_does_not_verdict_outside_preregistered_crossover() -> None:
    result = measurement_tool._pose_constraint_activity(
        pose_proposal={
            "feasible": True,
            "selected_step": 1.0,
            "planned_predictor_step_pose_mse": 1e-8,
            "effective_linear_pose_budget": 1e-3,
        },
        hard_d_pose=1e-8,
        tau_pose=1e-3,
        repair_shrink=1.0,
    )
    assert result["preregistered_hypothesis_at_or_below_crossover"] is False
    assert result["preregistered_hypothesis_confirmed_for_pair"] is None


def test_rung_e_records_joint_chosen_yhat_residual_provenance() -> None:
    chosen0 = np.array([[[1, 2, 3]]], dtype=np.int64)
    chosen1 = np.array([[[5, 7, 9]]], dtype=np.int64)
    predictor0 = np.array([[[1, 1, 3]]], dtype=np.int64)
    predictor1 = np.array([[[4, 7, 8]]], dtype=np.int64)
    point = measurement_tool._rung_e_rate_point(
        chosen0=chosen0,
        chosen1=chosen1,
        predictor0=predictor0,
        predictor1=predictor1,
        rate0={"brotli_q11_bytes": 11, "zstd_19_bytes": 12},
        rate1={"brotli_q11_bytes": 13, "zstd_19_bytes": 14},
    )

    assert point["shared_provenance"]["rung"] == "E"
    assert "chosen_yhat" in point["shared_provenance"]["point"]
    assert point["frames"]["frame0"]["residual_nonzero_count"] == 1
    assert point["frames"]["frame1"]["residual_nonzero_count"] == 2
    assert point["frames"]["frame0"]["measured_brotli_q11_bytes"] == 11


def test_waterfill_refuses_to_force_flat_curves() -> None:
    got = solve_measured_waterfill(
        [{"bytes": 10, "distortion": 0.1}, {"bytes": 20, "distortion": 0.1}],
        [{"bytes": 10, "distortion": 0.01}, {"bytes": 20, "distortion": 0.01}],
    )
    assert got["status"] == "INCONCLUSIVE_FLAT_OR_NOISY"


def test_waterfill_returns_measured_secant_candidate() -> None:
    got = solve_measured_waterfill(
        [{"bytes": 10, "distortion": 0.2}, {"bytes": 20, "distortion": 0.1}],
        [{"bytes": 10, "distortion": 0.001}, {"bytes": 20, "distortion": 0.0005}],
    )
    assert got["status"] == "MEASURED_SECANT_KKT_CANDIDATE"
    assert got["derived_pose_seg_crossover_d_pose"] == 2.5e-4
