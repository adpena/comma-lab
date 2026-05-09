from __future__ import annotations

import json
import math

import pytest

from tools import theoretical_floor_solver_v2 as solver


def test_contest_score_matches_a1_cpu_anchor_with_rounded_components() -> None:
    score = solver.contest_score(
        solver.A1_D_SEG,
        solver.A1_D_POSE,
        solver.A1_ARCHIVE_BYTES,
    )

    assert score == pytest.approx(solver.A1_SCORE_CPU, abs=1e-6)


def test_theoretical_floor_default_matches_council_median() -> None:
    floor = solver.theoretical_floor_estimate()
    volterra_floor = solver.theoretical_floor_estimate(include_volterra_correction=True)

    assert floor.median == pytest.approx(0.140)
    assert volterra_floor.median == pytest.approx(0.135)
    assert floor.ci_95_low <= floor.median <= floor.ci_95_high


def test_score_to_byte_target_inverts_score_formula() -> None:
    target_bytes = solver.score_to_byte_target(0.17)
    score = solver.contest_score(solver.A1_D_SEG, solver.A1_D_POSE, target_bytes)

    assert score == pytest.approx(0.17, abs=solver.ALPHA_RATE / solver.N_REF_VIDEO_BYTES)


def test_fisher_allocation_respects_total_bits_and_floor() -> None:
    bits = solver.fisher_information_bit_allocation([0.0, 1.0, 4.0], total_bits=10)

    assert sum(bits) <= 10
    assert min(bits) >= 1
    assert bits[2] >= bits[1] >= bits[0]


def test_fisher_allocation_rejects_infeasible_floor() -> None:
    with pytest.raises(ValueError, match="smaller than the requested floor"):
        solver.fisher_information_bit_allocation([1.0, 2.0, 3.0], total_bits=2)


def test_wasserstein_cuda_cpu_correction_for_hnerv_cluster() -> None:
    mean, std = solver.wasserstein1_cuda_cpu_correction(solver.A1_SCORE_CPU, decoder_class="a1")

    assert mean == pytest.approx(solver.A1_SCORE_CPU + solver.HNERV_CUDA_CPU_GAP_MEAN)
    assert std == pytest.approx(solver.HNERV_CUDA_CPU_GAP_STD)


def test_lagrangian_dual_update_moves_against_positive_residuals() -> None:
    state = solver.LagrangianState(eta=0.5)
    state.update_duals(
        B_current=120.0,
        B_target=100.0,
        d_seg_current=0.2,
        d_seg_target=0.1,
        d_pose_current=0.03,
        d_pose_target=0.01,
    )

    assert state.lambda_B == pytest.approx(10.0)
    assert state.lambda_seg == pytest.approx(0.05)
    assert state.lambda_pose == pytest.approx(0.01)


def test_phase_1_cost_summary_is_ordered_by_eig_per_dollar() -> None:
    summary = solver.phase_1_total_cost_estimate()
    eigs = [row["eig_per_dollar"] for row in summary["tracks_by_eig_per_dollar"]]

    assert eigs == sorted(eigs, reverse=True)
    assert math.isclose(summary["total_gpu_cost_usd"], 210.0)


def test_cli_help_surface_is_discoverable() -> None:
    help_text = solver.build_arg_parser().format_help()

    assert "--target-score" in help_text
    assert "--decoder-class" in help_text
    assert "--json" in help_text


def test_cli_json_payload_is_planning_only(capsys: pytest.CaptureFixture[str]) -> None:
    rc = solver.main(["--json", "--target-score", "0.17"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "theoretical_floor_solver_v2_cli.v1"
    assert payload["score_claim"] is False
    assert payload["evidence_grade"] == "planning_math_only"
    assert payload["byte_target_at_a1_distortion"] == solver.score_to_byte_target(0.17)
