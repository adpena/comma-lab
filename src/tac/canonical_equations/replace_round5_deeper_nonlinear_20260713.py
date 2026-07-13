# SPDX-License-Identifier: MIT
"""Canonical laws for REPLACE round-5 deeper/nonlinear localization."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "replace_round5_deeper_nonlinear_localization_v1"
DAG_FEED = ".omx/research/replace_round5_deeper_nonlinear_DAG_FEED_20260713.md"
AXIS = "[macOS-CPU advisory; NumPy-fp64 convex; CPU-torch exact costates]"
RECEIPT = "experiments/results/replace_round5_deeper_nonlinear_20260713/receipt.json"
RECEIPT_SHA256 = "38033922bd39cb48f72a154ddd622c41b18be0f137ede56fe4c76873e7bfe98f"
MEASUREMENT_UTC = "2026-07-13T21:03:47.961403Z"


def _unit_fraction(value: float, *, name: str, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real fraction")
    result = float(value)
    lower_ok = result > 0.0 if strict else result >= 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        bracket = "(0,1]" if strict else "[0,1]"
        raise ValueError(f"{name} must lie in {bracket}")
    return result


def post_se_sparse_teacher_economics(
    *, feature_cut_fraction: float, selected_area_fraction: float, anchor_calls: int
) -> dict[str, float | int | None]:
    """Compose cut and label FLOPs, retaining acquisition calls as campaign cost.

    The result is conditional: it does not assert an implemented sparse kernel or
    wall-clock speedup, and it does not erase the post-SE global-state dependency.
    """

    cut = _unit_fraction(feature_cut_fraction, name="feature_cut_fraction")
    area = _unit_fraction(selected_area_fraction, name="selected_area_fraction")
    if isinstance(anchor_calls, bool) or not isinstance(anchor_calls, int) or anchor_calls < 0:
        raise ValueError("anchor_calls must be a nonnegative integer")
    coefficient = cut + (1.0 - cut) * area
    if coefficient >= 1.0:
        reduction = 1.0
        break_even: float | None = None
    else:
        reduction = 1.0 / coefficient
        break_even = anchor_calls / (1.0 - coefficient)
    return {
        "anchor_calls_A": anchor_calls,
        "feature_cut_fraction_p": cut,
        "selected_area_fraction_q": area,
        "conditional_c_label": coefficient,
        "conditional_variable_cost_reduction_x": reduction,
        "break_even_future_steps_D": break_even,
    }


def equal_exact_call_branch_design(
    *, horizons: Iterable[int], exact_calls_per_horizon: int
) -> dict[str, object]:
    """Seal an h=0-anchored equal-exact-call branch comparison."""

    values = tuple(horizons)
    if (
        not values
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
        or len(values) != len(set(values))
        or tuple(sorted(values)) != values
        or 0 not in values
    ):
        raise ValueError("horizons must be unique sorted nonnegative integers including h=0")
    if (
        isinstance(exact_calls_per_horizon, bool)
        or not isinstance(exact_calls_per_horizon, int)
        or exact_calls_per_horizon <= 0
    ):
        raise ValueError("exact_calls_per_horizon must be a positive integer")
    return {
        "horizons": values,
        "baseline_horizon": 0,
        "exact_calls_per_horizon": exact_calls_per_horizon,
        "total_exact_call_budget": len(values) * exact_calls_per_horizon,
        "advance_rule": "h>0 must beat h=0 on audited full-facet error",
    }


def query_refuse_audit_budget(
    *, total_cells: int, targeted_fraction: float, random_audit_fraction: float
) -> dict[str, float | int]:
    """Derive deterministic targeted plus positive-propensity audit counts."""

    if isinstance(total_cells, bool) or not isinstance(total_cells, int) or total_cells < 2:
        raise ValueError("total_cells must be an integer >= 2")
    targeted = _unit_fraction(targeted_fraction, name="targeted_fraction", strict=True)
    audit = _unit_fraction(random_audit_fraction, name="random_audit_fraction", strict=True)
    targeted_count = max(1, math.ceil(targeted * total_cells))
    audit_count = max(1, math.ceil(audit * total_cells))
    remaining = total_cells - targeted_count
    if remaining <= 0 or audit_count > remaining:
        raise ValueError("audit sample must fit outside the targeted set")
    queried = targeted_count + audit_count
    return {
        "total_cells": total_cells,
        "targeted_count": targeted_count,
        "random_audit_count": audit_count,
        "queried_count": queried,
        "realized_query_fraction": queried / total_cells,
        "random_audit_positive_propensity": audit_count / remaining,
    }


def build_replace_round5_deeper_nonlinear_v1() -> CanonicalEquation:
    """Build the feature-source-scoped negative and the query-audit anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DAG_FEED,
        reactivation_criteria=(
            "recompute for a dense-label learner, transition-complete FORE/on-policy support "
            "with Z,A,R,Z-prime custody, a different replay distribution or seed, or an "
            "evaluator-equivalent witness successor"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_arm64_cpu_numpy_fp64_and_cpu_torch_teacher",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="replace_round5_deeper_nonlinear_v9_n600_seed455_20260713",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "n_states": 600,
            "train_states": 480,
            "heldout_states": 120,
            "checkpoint_epochs": [150, 251, 275],
            "selected_prefix_cells_per_state": 2311,
            "prefix_cells_per_state": 49152,
            "realized_area_fraction": 0.047017415364583336,
            "deep_feature_count": 116,
            "ordered_class_pair_count": 20,
            "nonlinear_seeds": [455, 456, 457],
            "receipt_sha256": RECEIPT_SHA256,
        },
        predicted_output={
            "heldout_retained_mass_fraction_minimum": 0.47,
            "nonlinear_seed_population_std_maximum": 0.03,
            "teacher_started_call_budget_maximum": 600,
            "disagreement_high_low_error_ratio_minimum": 1.25,
        },
        empirical_output={
            "verdict": "KILL-CHEAP-LOCALIZATION-FAMILY-BY-FEATURE-SOURCE",
            "verdict_scope": "FAMILY x FEATURE-SOURCE x FIXED REPLAY",
            "convex_deeper_retained_mass_fraction": 0.13046753525944724,
            "nonlinear_ensemble_retained_mass_fraction": 0.29462633883840517,
            "nonlinear_seed_retained_mass": [
                0.2773190155117359,
                0.27591107023490496,
                0.2796787058394356,
            ],
            "nonlinear_seed_population_std": 0.0015544032936474037,
            "oracle_retained_mass_fraction": 0.5278150212253758,
            "ensemble_ece": 0.18620396272803974,
            "block2_cut_fraction": 0.04214211147013728,
            "block3_cut_fraction": 0.07129461126470672,
            "post_se_tileability": "not-independently-tileable-after-first-se",
            "campaign_honest_teacher_starts": 600,
            "teacher_retries": 0,
            "conditional_c_label": 0.11495993827820083,
            "conditional_variable_cost_reduction_x": 8.698682471279858,
            "pay_only_on_support_admitted": False,
            "query_disagreement_high_low_error_ratio": 189.8129248528991,
            "query_disagreement_error_spearman": 0.8656102517385542,
            "query_random_audit_positive_propensity": 0.01042704249231747,
            "query_research_gate_pass": True,
            "branch_horizon_status": "blocked-not-identified",
            "FORE_weights_applied": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.1753736611615948,
        source_artifact=RECEIPT,
        measurement_method=(
            "preregistered fixed n600 replay; exact CPU SegNet input-costate support labels; "
            "twenty 116-column post-SE class-pair RankRLS Moore-Penrose heads followed by "
            "three deterministic pair-gated MLP seeds with train-only-dev early stopping; "
            "untouched heldout top-2311 selection; 4-percent disagreement query plus "
            "1-percent positive-propensity randomized audit"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Post-SE support localization, conditional cost, and query audit",
        one_line_summary=(
            "Post-SE cut cost composes as p+(1-p)q, but the registered deeper convex and "
            "small nonlinear heads miss the 47-percent support-retention gate."
        ),
        latex_form=(
            r"\rho_k=\frac{\sum_{i\in\operatorname{TopK}(s,k)}\|\lambda_i\|_2^2}"
            r"{\sum_i\|\lambda_i\|_2^2},\qquad "
            r"C_{teacher}=A+[p_{cut}+(1-p_{cut})q]D"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.replace_round5_deeper_nonlinear_20260713:"
            "post_se_sparse_teacher_economics"
        ),
        domain_of_validity={
            "scope_level": "family x feature-source x fixed replay",
            "included": [
                "fixed V9 n600 seed455 replay with 480 train and 120 heldout states",
                "block2/block3 post-SE feature source with full-frame SE-state custody",
                "twenty ordered-class-pair convex RankRLS Moore-Penrose heads",
                "three-seed width-32 pair-gated MLP ensemble with train-only-dev early stop",
                "matched top-2311 exact input-costate support target",
            ],
            "excluded": [
                "dense-label feature learners or larger nonlinear/attention families",
                "transition-complete FORE or on-policy query policies",
                "other replay distributions or seeds",
                "implemented sparse exact-teacher kernels or realized wall-clock speedups",
                "evaluator score, archive bytes, contest-CPU, CUDA, MPS, or promotion authority",
            ],
            "research_only": True,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
            "authority": AXIS,
        },
        units_in={
            "costate_mass": "squared SegNet input-gradient units",
            "feature_selected_fractions": "dimensionless",
            "anchor_calls": "campaign-honest exact-teacher starts",
        },
        units_out={
            "retained_mass_fraction": "dimensionless",
            "conditional_teacher_coefficient": "dense-teacher variable-cost fraction",
            "query_audit_propensity": "dimensionless probability",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "nonlinear_retained_mass_shortfall": 0.1753736611615948,
            "convex_deeper_retained_mass_shortfall": 0.33953246474055276,
            "oracle_headroom_over_gate": 0.057815021225375795,
            "nonlinear_seed_std_headroom": 0.028445596706352594,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.scorer_surrogate.replace_round5_deeper_nonlinear",
            "tools.probe_replace_round5_deeper_nonlinear",
            "tac.witness_dsl.replace_round5_deeper_nonlinear_policy",
        ),
        canonical_producers=("tools.probe_replace_round5_deeper_nonlinear",),
        provenance=provenance,
    )


def populate_replace_round5_deeper_nonlinear_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append through the locked helper; main review owns shared registration."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_replace_round5_deeper_nonlinear_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="replace-round5; feature-source-family-kill; query-audit; research-only",
    )
    return equation


__all__ = [
    "AXIS",
    "DAG_FEED",
    "EQUATION_ID",
    "RECEIPT",
    "RECEIPT_SHA256",
    "build_replace_round5_deeper_nonlinear_v1",
    "equal_exact_call_branch_design",
    "populate_replace_round5_deeper_nonlinear_v1",
    "post_se_sparse_teacher_economics",
    "query_refuse_audit_budget",
]
