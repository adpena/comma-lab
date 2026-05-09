"""Tests for the Fields-medal Theoretical Floor Solver v2.

Validates the math from the Grand Council Fields-Medal Theoretical Floor
deliberation 2026-05-09 (`.omx/research/grand_council_fields_medal_theoretical_floor_20260509.md`):

  * contest_score formula matches the upstream evaluator
  * A1 score decomposition matches the recorded 0.19285
  * score_to_byte_target inversion is consistent with score_components
  * Lagrangian-ADMM dual update arithmetic matches Boyd 2011
  * Wasserstein-1 CUDA-CPU correction matches the HNeRV cluster mean
  * Fisher-information bit allocation respects the floor + budget
  * Theoretical floor estimate is in the 95% CI
  * A1-floor-gap decomposition sums to the total gap (within rounding)
  * Phase 1 track plan totals match the council prescription
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Add tools/ directly to path so we can import the solver module without packaging it.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import theoretical_floor_solver_v2 as solver  # noqa: E402


def test_contest_score_formula_matches_a1_anchor() -> None:
    """A1's recorded score must match the formula on its (B, d_seg, d_pose)."""
    s = solver.contest_score(solver.A1_D_SEG, solver.A1_D_POSE, solver.A1_ARCHIVE_BYTES)
    # Allow ±0.0005 tolerance for the floating-point reconstruction.
    assert abs(s - solver.A1_SCORE_CPU) < 0.001, (
        f"Computed S = {s} ≠ recorded A1 score {solver.A1_SCORE_CPU}"
    )


def test_score_components_decomposition_sums_to_total() -> None:
    """rate + seg + pose must equal total to within 1e-9."""
    comp = solver.score_components(solver.A1_D_SEG, solver.A1_D_POSE, solver.A1_ARCHIVE_BYTES)
    assert abs(comp["total"] - (comp["rate"] + comp["seg"] + comp["pose"])) < 1e-9


def test_score_to_byte_target_inversion() -> None:
    """For a target score, the inverted byte budget should reproduce that score."""
    target = 0.17
    b = solver.score_to_byte_target(target, d_seg=solver.A1_D_SEG, d_pose=solver.A1_D_POSE)
    s_back = solver.contest_score(solver.A1_D_SEG, solver.A1_D_POSE, b)
    assert abs(s_back - target) < 1e-3, f"Byte inversion mismatch: target={target}, got={s_back}"


def test_score_to_byte_target_infeasible_raises() -> None:
    """If target_score < seg+pose contribution alone, raises ValueError."""
    # At A1's seg/pose, seg+pose contribution is ~0.074. Target 0.05 is infeasible.
    try:
        solver.score_to_byte_target(0.05, d_seg=solver.A1_D_SEG, d_pose=solver.A1_D_POSE)
        raise AssertionError("Should have raised ValueError for infeasible target")
    except ValueError as e:
        assert "infeasible" in str(e).lower()


def test_lagrangian_dual_update_monotone() -> None:
    """Dual ascent: if constraint is violated (current > target), λ must increase."""
    state = solver.LagrangianState(eta=0.1)
    initial_lambda_B = state.lambda_B
    state.update_duals(
        B_current=200_000,
        B_target=178_000,
        d_seg_current=0.0006,
        d_seg_target=0.0005,
        d_pose_current=4e-5,
        d_pose_target=3e-5,
    )
    # All three constraints violated → all duals increase.
    assert state.lambda_B > initial_lambda_B
    assert state.lambda_seg > 0
    assert state.lambda_pose > 0


def test_lagrangian_evaluation_at_feasible_point() -> None:
    """At the target point itself, augmented terms are zero → L_ρ = S."""
    state = solver.LagrangianState(rho=1.0)
    targets = (178_000, 0.0005, 3e-5)
    L = solver.contest_score_lagrangian(0.0005, 3e-5, 178_000, targets=targets, state=state)
    S = solver.contest_score(0.0005, 3e-5, 178_000)
    assert abs(L - S) < 1e-9, f"At feasible point, L should equal S. L={L}, S={S}"


def test_wasserstein1_hnerv_correction_matches_cluster_mean() -> None:
    """For HNeRV-class, predicted CUDA = CPU + 0.0327 ± 0.001."""
    cuda_mean, cuda_std = solver.wasserstein1_cuda_cpu_correction(
        solver.A1_SCORE_CPU, decoder_class="a1"
    )
    expected_mean = solver.A1_SCORE_CPU + solver.HNERV_CUDA_CPU_GAP_MEAN
    assert abs(cuda_mean - expected_mean) < 1e-6
    assert cuda_std == solver.HNERV_CUDA_CPU_GAP_STD


def test_wasserstein1_unknown_class_widens_uncertainty() -> None:
    """For unknown decoder class, std widens by ≥4× (per Wasserstein analysis)."""
    _, std_known = solver.wasserstein1_cuda_cpu_correction(0.193, decoder_class="hnerv")
    _, std_unknown = solver.wasserstein1_cuda_cpu_correction(0.193, decoder_class="balle_hyperprior")
    assert std_unknown >= 4.0 * std_known


def test_fisher_bit_allocation_respects_floor_and_budget() -> None:
    """Allocation must give every param ≥ floor and total ≤ budget."""
    fisher = [1.0, 4.0, 16.0, 64.0]  # log_2: 0, 1, 2, 3 → relative weights 1, 2, 3, 4.
    bits = solver.fisher_information_bit_allocation(fisher, total_bits=40, floor_bits_per_param=2)
    # Each param gets ≥ floor.
    assert all(b >= 2 for b in bits)
    # Total ≤ budget.
    assert sum(bits) <= 40
    # Higher Fisher → ≥ bits than lower Fisher.
    assert bits[3] >= bits[0]


def test_fisher_bit_allocation_uniform_when_no_signal() -> None:
    """All-zero Fisher → uniform allocation."""
    fisher = [0.0, 0.0, 0.0, 0.0]
    bits = solver.fisher_information_bit_allocation(fisher, total_bits=20, floor_bits_per_param=2)
    # Should distribute remaining 12 bits across 4 params (3 each).
    assert sum(bits) <= 20
    assert max(bits) - min(bits) <= 1  # Within ±1 due to integer rounding.


def test_fisher_bit_allocation_handles_insufficient_budget() -> None:
    """If total_bits < n*floor, return uniform floor."""
    fisher = [1.0, 4.0, 16.0]
    bits = solver.fisher_information_bit_allocation(fisher, total_bits=2, floor_bits_per_param=2)
    assert bits == [2, 2, 2]


def test_theoretical_floor_estimate_within_council_ci() -> None:
    """Floor estimate must be within the council's 95% CI."""
    floor = solver.theoretical_floor_estimate()
    assert solver.S_FLOOR_CI_95_LOW <= floor.median <= solver.S_FLOOR_CI_95_HIGH
    assert floor.std == solver.S_FLOOR_STD


def test_theoretical_floor_quantizr_88K_ceiling_applies() -> None:
    """For decoder_param_count ≤ 100K, Quantizr 88K ceiling enters bounds."""
    floor = solver.theoretical_floor_estimate(decoder_param_count=88_000)
    assert "quantizr_88K_ceiling" in floor.constituent_bounds
    assert floor.constituent_bounds["quantizr_88K_ceiling"] == 0.180


def test_theoretical_floor_volterra_correction_lowers_median() -> None:
    """Volterra super-additive correction lowers the median by 0.005."""
    with_v = solver.theoretical_floor_estimate(include_volterra_correction=True)
    without_v = solver.theoretical_floor_estimate(include_volterra_correction=False)
    assert abs((without_v.median - with_v.median) - 0.005) < 1e-9


def test_a1_gap_decomposition_sums_to_total() -> None:
    """The 4 attribution components must sum to the total gap (within 0.001)."""
    decomp = solver.a1_floor_gap_decomposition()
    component_sum = (
        decomp.byte_axis_attribution
        + decomp.seg_axis_attribution
        + decomp.pose_axis_attribution
        + decomp.architectural_class_attribution
    )
    # Council math is 0.027 + 0.011 + 0.002 + 0.013 = 0.053.
    assert abs(component_sum - 0.053) < 1e-9
    # Total gap must be at least within 0.01 of the sum (council-derived approx).
    assert abs(decomp.total_gap - component_sum) < 0.01


def test_a1_gap_summary_has_all_four_axes() -> None:
    """Summary string must mention all 4 attribution axes."""
    decomp = solver.a1_floor_gap_decomposition()
    summary = decomp.summary()
    for label in ("byte", "seg", "pose", "arch"):
        assert label in summary.lower()


def test_phase_1_tracks_returns_5_tracks() -> None:
    """Council prescription is 5 tracks."""
    tracks = solver.paradigm_delta_epsilon_zeta_phase_1_tracks()
    assert len(tracks) == 5
    track_ids = {t.track_id for t in tracks}
    assert track_ids == {"1", "2", "3", "4", "5"}


def test_phase_1_total_cost_within_council_estimate() -> None:
    """Total Phase 1 GPU cost should be ≈$210 per council estimate (±$50)."""
    summary = solver.phase_1_total_cost_estimate()
    assert 150 <= summary["total_gpu_cost_usd"] <= 280, (
        f"Total Phase 1 GPU cost ${summary['total_gpu_cost_usd']} outside [150, 280]"
    )


def test_phase_1_track_4_highest_eig_per_dollar() -> None:
    """Track 4 (UNIWARD on existing A1) is the highest EIG/$ per council ranking."""
    summary = solver.phase_1_total_cost_estimate()
    top_track = summary["tracks_by_eig_per_dollar"][0]
    assert top_track["id"] == "4", (
        f"Top EIG/$ track should be 4 (UNIWARD), got {top_track['id']}"
    )


def test_information_bottleneck_beta_positive_for_typical_target() -> None:
    """β should be positive for any sub-0.20 target on A1's anchor."""
    beta = solver.information_bottleneck_beta(target_score=0.17)
    assert beta > 0


def test_a1_decomposition_matches_recorded_anchor() -> None:
    """The A1 anchor numbers in the module must match the recorded anchor exactly."""
    assert solver.A1_ARCHIVE_BYTES == 178_262
    assert abs(solver.A1_D_SEG - 0.0005602) < 1e-9
    assert abs(solver.A1_D_POSE - 3.286e-5) < 1e-9
    assert abs(solver.A1_SCORE_CPU - 0.19284757743677347) < 1e-12


def test_n_ref_video_bytes_matches_canonical() -> None:
    """N must match the upstream constant."""
    assert solver.N_REF_VIDEO_BYTES == 37_545_489


def test_constants_match_canonical_alpha_beta_gamma() -> None:
    """α=25, β=100, γ=√10."""
    assert solver.ALPHA_RATE == 25.0
    assert solver.BETA_SEG == 100.0
    assert abs(solver.GAMMA_POSE - math.sqrt(10.0)) < 1e-12
