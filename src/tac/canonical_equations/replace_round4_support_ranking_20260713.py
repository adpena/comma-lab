# SPDX-License-Identifier: MIT
"""Canonical support-retention and conditional teacher-cost laws for REPLACE round 4."""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "replace_round4_support_ranking_v1"
MEASUREMENT_UTC = "2026-07-13T19:28:32.454641Z"
DAG_FEED = ".omx/research/replace_round4_support_ranking_DAG_FEED_20260713.md"
AXIS = "[macOS-CPU advisory; NumPy-fp64 convex fit; frozen CPU SegNet exact costates]"
RECEIPT = "experiments/results/replace_round4_support_ranking_20260713/receipt.json"
RECEIPT_SHA256 = "6ccbf0e10691dc39c94b77aaefdfe7d9ac3a38b32962bfa5eefcb1107f627222"


def support_retention_law(
    *,
    retained_l2_square_mass: float,
    total_l2_square_mass: float,
    selected_cells: int,
    total_cells: int,
) -> dict[str, float | int]:
    """Derive matched-area retention, uplift, and masked-exact cosine."""

    masses = (retained_l2_square_mass, total_l2_square_mass)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in masses
    ):
        raise ValueError("support masses must be finite real numbers")
    if not 0.0 <= retained_l2_square_mass <= total_l2_square_mass or total_l2_square_mass <= 0:
        raise ValueError("support masses require 0 <= retained <= positive total")
    cells = (selected_cells, total_cells)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in cells):
        raise ValueError("support cell counts must be integers")
    if not 0 < selected_cells <= total_cells:
        raise ValueError("support cell counts require 0 < selected <= total")
    area = selected_cells / total_cells
    retained = retained_l2_square_mass / total_l2_square_mass
    return {
        "retained_l2_square_mass": retained_l2_square_mass,
        "total_l2_square_mass": total_l2_square_mass,
        "selected_cells": selected_cells,
        "total_cells": total_cells,
        "area_fraction": area,
        "retained_mass_fraction": retained,
        "uplift_over_uniform_area": retained / area,
        "conditional_masked_exact_costate_cosine": math.sqrt(retained),
    }


def conditional_sparse_teacher_economics(
    *, prefix_fraction: float, selected_area_fraction: float
) -> dict[str, float]:
    """Compose prefix and selected-label fractions without claiming sparse wall time."""

    values = (prefix_fraction, selected_area_fraction)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("cost fractions must be finite and in [0,1]")
    coefficient = prefix_fraction + (1.0 - prefix_fraction) * selected_area_fraction
    return {
        "prefix_fraction": float(prefix_fraction),
        "selected_area_fraction": float(selected_area_fraction),
        "conditional_composed_label_coefficient": coefficient,
        "conditional_variable_cost_reduction_x": 1.0 / coefficient,
    }


def build_replace_round4_support_ranking_v1() -> CanonicalEquation:
    """Build the closed laws and the exact-optimum family-scoped negative anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DAG_FEED,
        reactivation_criteria=(
            "recompute for a deeper prefix, nonlinear or dense-label ranker, transition-complete "
            "FORE/on-policy support, different replay distribution, or different seed"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_arm64_cpu_numpy_fp64_and_cpu_torch_teacher",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="replace_round4_support_ranking_v9_n600_seed455_20260713",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "n_states": 600,
            "train_states": 480,
            "heldout_states": 120,
            "checkpoint_epochs": [150, 251, 275],
            "selected_prefix_cells_per_state": 2311,
            "prefix_cells_per_state": 49152,
            "realized_area_fraction": 0.047017415364583336,
            "global_feature_count": 84,
            "block_feature_count": 44,
            "ordered_class_pair_count": 20,
            "seed": 455,
            "receipt_sha256": RECEIPT_SHA256,
        },
        predicted_output={
            "heldout_retained_mass_fraction_minimum": 0.47,
            "heldout_ece_refusal_maximum": 0.05,
            "oracle_retained_mass_fraction": 0.5278150212253758,
        },
        empirical_output={
            "verdict": "NO_GO_SHALLOW_CHEAP_FEATURE_CONVEX_LOCALIZATION",
            "family_verdict": (
                "FAMILY_LEVEL_NEGATIVE_SIGNAL__SHALLOW_PRESE_CHEAP_FEATURE_"
                "CONVEX_LOCALIZERS_ONLY"
            ),
            "winning_rung": "pairwise-rank-pair-block-44",
            "winning_retained_mass_fraction": 0.20172451295048283,
            "winning_uplift_over_uniform_area": 4.290421142597201,
            "winning_conditional_masked_exact_cosine": 0.44913752120089323,
            "winning_heldout_ece": 0.003073753177168275,
            "weighted_global_retained_mass_fraction": 0.19865776607447305,
            "weighted_pair_block_retained_mass_fraction": 0.19771315378268864,
            "oracle_retained_mass_fraction": 0.5278150212253758,
            "prefix_fraction": 0.005714118050141177,
            "conditional_composed_label_coefficient": 0.05246287035291876,
            "conditional_variable_cost_reduction_x": 19.061099655298698,
            "conditional_wall_clock_claim": False,
            "exact_teacher_unique_states": 600,
            "teacher_retries": 0,
            "FORE_weights_applied": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.26827548704951715,
        source_artifact=RECEIPT,
        measurement_method=(
            "preregistered fixed n600 replay; exact CPU SegNet input-costate support labels; "
            "one 84-column global weighted-top-k head, twenty 44-column weighted-top-k block "
            "heads, and twenty implicit all-pairs RankRLS block heads; symmetric-eigh "
            "rank-truncated Moore-Penrose optima; train-only 16-bin isotonic calibration; "
            "deterministic top-2311 heldout selection"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Matched-area support retention and conditional sparse-teacher cost",
        one_line_summary=(
            "Top-k support retention is a mass ratio; sparse-teacher economics compose as "
            "p+(1-p)q, while the registered shallow convex rankers miss the 47-percent gate."
        ),
        latex_form=(
            r"S_k(s)=\operatorname{TopK}(s,k),\quad "
            r"\rho_k=\frac{\sum_{i\in S_k(s)}\|\lambda_i\|_2^2}"
            r"{\sum_i\|\lambda_i\|_2^2},\quad "
            r"\cos(\lambda,M_{S_k}\lambda)=\sqrt{\rho_k};\qquad "
            r"C_{teacher}=A+[p+(1-p)q]D"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.replace_round4_support_ranking_20260713:"
            "support_retention_law"
        ),
        domain_of_validity={
            "scope_level": "family x fixed replay",
            "included": [
                "first local pre-SE 32-channel prefix plus source margin and class-pair chart",
                "global or ordered-class-pair convex weighted-top-k squared heads",
                "ordered-class-pair convex implicit all-positive-negative RankRLS heads",
                "rank threshold eps times feature width times maximum eigenvalue",
                "fixed V9 n600 seed455 replay and local macOS CPU exact-costate evidence",
            ],
            "excluded": [
                "deeper scorer features with global squeeze-excite state",
                "nonlinear, dense-label, or trainable feature learners",
                "transition-complete FORE or on-policy query policies",
                "sparse exact-teacher kernels and realized wall-clock speedups",
                "evaluator score, archive bytes, contest-CPU, CUDA, MPS, or promotion authority",
            ],
            "research_only": True,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
            "authority": AXIS,
        },
        units_in={
            "costate_mass": "squared SegNet input-gradient units",
            "selected_total_cells": "prefix lattice cells",
            "prefix_selected_area_fraction": "dimensionless",
        },
        units_out={
            "retained_mass_fraction": "dimensionless",
            "conditional_masked_costate_cosine": "dimensionless",
            "conditional_teacher_coefficient": "dense-teacher variable-cost fraction",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "winning_retained_mass_shortfall": 0.26827548704951715,
            "oracle_headroom_over_gate": 0.057815021225375795,
            "oracle_headroom_over_winner": 0.32609050827489297,
            "winner_ece_headroom_below_refusal": 0.04692624682283173,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.scorer_surrogate.replace_round4_support_ranking",
            "tools.probe_replace_round4_support_ranking",
            "tac.witness_dsl.replace_round4_support_ranking_policy",
        ),
        canonical_producers=("tools.probe_replace_round4_support_ranking",),
        provenance=provenance,
    )


def populate_replace_round4_support_ranking_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append through the locked registry helper; main review owns shared registration."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_replace_round4_support_ranking_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="replace-round4; family-scoped-no-go; support-ranking; research-only",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_replace_round4_support_ranking_v1",
    "conditional_sparse_teacher_economics",
    "populate_replace_round4_support_ranking_v1",
    "support_retention_law",
]
