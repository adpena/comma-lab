# SPDX-License-Identifier: MIT
"""Triality checks for the frozen-replay convex-head contraction law."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.frozen_replay_convex_head_contraction_20260713 import (
    EQUATION_ID,
    build_frozen_replay_convex_head_contraction_v1,
    cached_exact_label_teacher_calls,
    derive_spectral_scale_contraction,
    populate_frozen_replay_convex_head_contraction_v1,
)
from tac.canonical_equations.registry import query_equations
from tac.witness_dsl.frozen_replay_convex_head_policy import FrozenReplayConvexHeadPolicy


def test_spectral_scale_ridge_derives_exact_curvature_and_contraction() -> None:
    result = derive_spectral_scale_contraction(
        data_eigenvalue_min=2.0,
        data_eigenvalue_max=6.0,
    )

    assert result["ridge_lambda"] == pytest.approx(6.0)
    assert result["mu"] == pytest.approx(8.0)
    assert result["smoothness_L"] == pytest.approx(12.0)
    assert result["step_size_eta"] == pytest.approx(0.1)
    assert result["contraction_gamma"] == pytest.approx(0.2)
    assert result["derived_gamma_upper_bound"] == pytest.approx(1.0 / 3.0)


def test_rank_deficient_design_saturates_the_one_third_bound() -> None:
    result = derive_spectral_scale_contraction(
        data_eigenvalue_min=0.0,
        data_eigenvalue_max=4.0,
    )

    assert result["ridge_lambda"] == pytest.approx(4.0)
    assert result["mu"] == pytest.approx(4.0)
    assert result["smoothness_L"] == pytest.approx(8.0)
    assert result["step_size_eta"] == pytest.approx(1.0 / 6.0)
    assert result["contraction_gamma"] == pytest.approx(1.0 / 3.0)
    assert result["contraction_gamma"] <= result["derived_gamma_upper_bound"]


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (-1.0, 4.0, "positive semidefinite"),
        (0.0, 0.0, "positive scale"),
        (5.0, 4.0, "cannot exceed"),
        (float("nan"), 4.0, "finite numbers"),
        (0.0, float("inf"), "finite numbers"),
        (False, 4.0, "finite numbers"),
        (0.0, True, "finite numbers"),
        ("zero", 4.0, "finite numbers"),
    ],
)
def test_spectral_derivation_invalid_domain_fails_closed(
    minimum: object,
    maximum: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_spectral_scale_contraction(
            data_eigenvalue_min=minimum,  # type: ignore[arg-type]
            data_eigenvalue_max=maximum,  # type: ignore[arg-type]
        )


def test_cached_exact_label_teacher_call_law_composes_anchor_and_difference_terms() -> None:
    assert cached_exact_label_teacher_calls(
        fresh_anchor_samples=600,
        paired_difference_samples=0,
        labels_per_difference=2,
    ) == 600
    assert cached_exact_label_teacher_calls(
        fresh_anchor_samples=3,
        paired_difference_samples=4,
        labels_per_difference=2,
    ) == 11


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fresh_anchor_samples", -1),
        ("fresh_anchor_samples", True),
        ("paired_difference_samples", 1.5),
        ("labels_per_difference", False),
    ],
)
def test_teacher_call_law_rejects_invalid_count_custody(field: str, value: object) -> None:
    counts: dict[str, object] = {
        "fresh_anchor_samples": 1,
        "paired_difference_samples": 2,
        "labels_per_difference": 2,
    }
    counts[field] = value
    with pytest.raises(ValueError, match="nonnegative integers"):
        cached_exact_label_teacher_calls(**counts)  # type: ignore[arg-type]


def test_measured_equation_scopes_authority_and_names_default_off_consumers() -> None:
    equation = build_frozen_replay_convex_head_contraction_v1()

    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["verdict"] == "GO"
    assert anchor.empirical_output["heldout_costate_cosine"] == pytest.approx(
        0.0014157933865487525
    )
    assert anchor.empirical_output["teacher_call_amortization_x"] == pytest.approx(12.0)
    assert anchor.empirical_output["fit_prediction_rmse_residual"] <= anchor.empirical_output[
        "fit_prediction_rmse_bound"
    ]
    assert anchor.empirical_output["per_state_gradient_variance"] == pytest.approx(
        7.21498595203425e-14
    )
    assert anchor.empirical_output["max_parameter_ratio_above_scale_floor"] < anchor.empirical_output[
        "executed_contraction_gamma"
    ]
    assert anchor.inputs["n_pairs"] == 600
    assert anchor.inputs["receipt_sha256"] == (
        "067ce197d30fa9e2c7c4bda48ac671af550e0a00f126289ba5b30946d44fc4b1"
    )
    assert equation.predicted_vs_empirical_residual == {
        "ideal_theorem_gamma_upper_bound": 1.0 / 3.0,
        "executed_gamma_rounding_above_ideal_upper_bound": pytest.approx(
            0.3333333461703458 - 1.0 / 3.0
        ),
        "parameter_contraction_bound_violation": 0.0,
        "objective_contraction_bound_violation": 0.0,
        "teacher_amortization_5x_shortfall": 0.0,
        "round1_early_cosine_bar_shortfall": 0.0,
    }
    assert equation.domain_of_validity["scope_level"] == "formulation x instance"
    assert equation.domain_of_validity["research_only"] is True
    assert equation.domain_of_validity["review_status"] == "self-audited-UNREVIEWED_BY_MAIN"
    assert any("on-policy" in exclusion for exclusion in equation.domain_of_validity["excluded"])
    assert any("MPS" in exclusion for exclusion in equation.domain_of_validity["excluded"])
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False
    assert equation.canonical_consumers == (
        "tac.scorer_surrogate.frozen_replay_convex_head",
        "tools.probe_frozen_replay_convex_head",
        "tac.witness_dsl.frozen_replay_convex_head_policy",
    )
    assert equation.canonical_producers == ("tools.probe_frozen_replay_convex_head",)

    policy = FrozenReplayConvexHeadPolicy()
    contract = policy.compile_measurement_contract()
    assert policy.live_training_enabled is False
    assert policy.research_only is True
    assert policy.score_claim is False
    assert policy.promotion_eligible is False
    assert policy.fallback == "full_exact_teacher"
    assert policy.train_state_count == 480
    assert policy.heldout_state_count == 120
    assert policy.fit_epochs == 15
    assert policy.effective_training_state_steps == 7_200
    assert contract["fit_epochs"] == 15
    assert contract["effective_training_state_steps"] == 7_200
    assert contract["live_trainer_argv"] == []
    assert contract["exact_target_costate_tensor_absent"] is True
    assert contract["feature_includes_source_labels_and_margins"] is True
    assert contract["full_batch_deterministic_gradient_descent"] is True
    assert str(contract["constant_provenance"]["fit_epochs"]).startswith("DERIVED")
    assert str(contract["ridge_policy"]).startswith("DERIVED")
    assert str(contract["step_policy"]).startswith("DERIVED")
    cached_calls = cached_exact_label_teacher_calls(
        fresh_anchor_samples=600,
        paired_difference_samples=0,
        labels_per_difference=2,
    )
    assert policy.effective_training_state_steps / cached_calls == pytest.approx(12.0)


def test_population_round_trips_only_through_isolated_locked_registry(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_frozen_replay_convex_head_contraction_v1(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="round2_convex_head_equation",
    )

    rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line]
    loaded = query_equations(path=registry)
    assert registry.exists()
    assert lock.exists()
    assert populated.equation_id == EQUATION_ID
    assert [equation.equation_id for equation in loaded] == [EQUATION_ID]
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_id"] == EQUATION_ID
    assert rows[0]["agent"] == "pytest"
    assert rows[0]["subagent_id"] == "round2_convex_head_equation"
    assert rows[0]["notes"] == "round2-95kill; fixed-replay; cached-exact-labels; research-only"
    assert not list(tmp_path.glob("*.tmp.*"))
