from __future__ import annotations

import math

import pytest

from tac.canonical_equations.ddm_cn4_arc_consolidation_20260811 import (
    CPU_CUDA_EQUATION_ID,
    MULTISTART_EQUATION_ID,
    REALIZATION_EQUATION_ID,
    build_lc2_device_delta_anchor,
    build_ps135_multistart_equation,
    build_pz4_realization_yield_anchor,
)
from tac.canonical_equations.evaluators import (
    EvaluatorError,
    eval_cpu_cuda_score_gap,
    eval_radius2_multistart_singleton_escape,
    eval_realization_breakeven_bytes,
    populate_lawref_evaluators,
    resolve_equation_value,
)


def _sidecar(tmp_path, name: str = "receipt.json"):
    path = tmp_path / name
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_cpu_cuda_evaluator_uses_registered_cuda_minus_cpu_sign() -> None:
    got = eval_cpu_cuda_score_gap(
        {"score_cpu": 0.20728492781521812, "score_cuda": 0.16959899569230852}
    )
    assert got == pytest.approx(-0.03768593212290960)


def test_cpu_cuda_evaluator_refuses_nonfinite() -> None:
    with pytest.raises(EvaluatorError, match="finite"):
        eval_cpu_cuda_score_gap({"score_cpu": math.inf, "score_cuda": 0.1})


def test_cpu_cuda_evaluator_refuses_negative_scores() -> None:
    with pytest.raises(EvaluatorError, match="non-negative"):
        eval_cpu_cuda_score_gap({"score_cpu": -0.1, "score_cuda": 0.1})


def test_realization_breakeven_evaluator_matches_contest_rate() -> None:
    assert eval_realization_breakeven_bytes({"realized_recovery_s": 0.001}) == pytest.approx(
        0.001 * 37_545_489 / 25
    )


def test_realization_breakeven_evaluator_refuses_negative() -> None:
    with pytest.raises(EvaluatorError, match="non-negative"):
        eval_realization_breakeven_bytes({"realized_recovery_s": -1.0})


def test_multistart_evaluator_reports_escape() -> None:
    got = eval_radius2_multistart_singleton_escape(
        {
            "pair_count": 600,
            "accepted_rows": 597,
            "score_before": 0.2072899013894104,
            "score_after": 0.18474482031130968,
            "d_pose_before": 0.00015904,
            "d_pose_after": 0.000030088120534088604,
        }
    )
    assert got["escaped"] is True
    assert got["accepted_fraction"] == pytest.approx(597 / 600)
    assert got["score_reduction"] == pytest.approx(0.02254508107810072)


def test_multistart_evaluator_no_accept_is_not_escape() -> None:
    got = eval_radius2_multistart_singleton_escape(
        {
            "pair_count": 600,
            "accepted_rows": 0,
            "score_before": 0.18,
            "score_after": 0.18,
            "d_pose_before": 1e-5,
            "d_pose_after": 1e-5,
        }
    )
    assert got["escaped"] is False


@pytest.mark.parametrize("pair_count", [0, -1])
def test_multistart_evaluator_refuses_nonpositive_population(pair_count: int) -> None:
    with pytest.raises(EvaluatorError, match="pair_count"):
        eval_radius2_multistart_singleton_escape(
            {
                "pair_count": pair_count,
                "accepted_rows": 0,
                "score_before": 1.0,
                "score_after": 1.0,
                "d_pose_before": 1.0,
                "d_pose_after": 1.0,
            }
        )


@pytest.mark.parametrize("accepted_rows", [-1, 601])
def test_multistart_evaluator_refuses_invalid_accept_count(accepted_rows: int) -> None:
    with pytest.raises(EvaluatorError, match="accepted_rows"):
        eval_radius2_multistart_singleton_escape(
            {
                "pair_count": 600,
                "accepted_rows": accepted_rows,
                "score_before": 1.0,
                "score_after": 1.0,
                "d_pose_before": 1.0,
                "d_pose_after": 1.0,
            }
        )


def test_device_anchor_carries_identical_archive_and_opposite_sign(tmp_path) -> None:
    anchor = build_lc2_device_delta_anchor(source_receipt=_sidecar(tmp_path))
    assert anchor.inputs["archive_bytes"] == 187_226
    assert anchor.empirical_output["cuda_minus_cpu_score"] < 0
    assert anchor.empirical_output["sign_opposes_pr102_precedent"] is True


def test_device_anchor_residual_measures_old_cluster_miss(tmp_path) -> None:
    anchor = build_lc2_device_delta_anchor(source_receipt=_sidecar(tmp_path))
    assert anchor.residual == pytest.approx(0.0706859321229096)


def test_realization_anchor_carries_exact_yield(tmp_path) -> None:
    anchor = build_pz4_realization_yield_anchor(source_receipt=_sidecar(tmp_path))
    assert anchor.empirical_output["realization_fraction"] == pytest.approx(4089 / 19221)
    assert anchor.empirical_output["decoder_required_bytes_restored"] == 15_132
    assert anchor.empirical_output["score_recovery_status"] == "UNMEASURED"
    assert "realized_recovery_s" not in anchor.inputs


def test_realization_anchor_does_not_claim_receiver_envelope_as_candidate(tmp_path) -> None:
    anchor = build_pz4_realization_yield_anchor(source_receipt=_sidecar(tmp_path))
    assert anchor.inputs["envelope_archive_bytes"] == 168_005
    assert anchor.inputs["realized_archive_bytes"] == 183_137
    assert anchor.provenance.score_claim_valid is False


def test_ps135_equation_has_real_anchor_and_declared_routes(tmp_path) -> None:
    equation = build_ps135_multistart_equation(source_receipt=_sidecar(tmp_path))
    assert equation.equation_id == MULTISTART_EQUATION_ID
    assert len(equation.empirical_anchors) == 1
    assert equation.canonical_producers
    assert equation.canonical_consumers


def test_ps135_equation_keeps_cuda_transfer_open(tmp_path) -> None:
    equation = build_ps135_multistart_equation(source_receipt=_sidecar(tmp_path))
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["cuda_transfer"] == "OPEN_HYPOTHESIS"
    assert equation.domain_of_validity["score_claim"] is False


def test_ps135_equation_records_both_complete_passes(tmp_path) -> None:
    equation = build_ps135_multistart_equation(source_receipt=_sidecar(tmp_path))
    empirical = equation.empirical_anchors[0].empirical_output
    assert empirical["pass1_accepted_rows"] == 597
    assert empirical["pass2_accepted_rows"] == 517
    assert empirical["pass2_archive_bytes"] == 187_221
    assert empirical["pass2_score"] == pytest.approx(0.17952896607020802)


def test_lawref_population_registers_all_three_cn4_evaluators() -> None:
    ids = populate_lawref_evaluators()
    assert CPU_CUDA_EQUATION_ID in ids
    assert REALIZATION_EQUATION_ID in ids
    assert MULTISTART_EQUATION_ID in ids


def test_lawref_resolves_cpu_cuda_delta() -> None:
    populate_lawref_evaluators()
    assert resolve_equation_value(
        CPU_CUDA_EQUATION_ID, {"score_cpu": 0.2, "score_cuda": 0.15}
    ) == pytest.approx(-0.05)
