# SPDX-License-Identifier: MIT
"""Canonical law separating estimator variance from fixed-call exact progress.

The n600-composed, six-bit terminal-polish A/B measured lower exhaustive UGC gradient variance than
plain DisARM, but UGC lost to the existing (1+1)-ES control at the same 64 exact-objective calls.
The law prevents a variance-only result from being promoted as a score-progress result.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ugc_terminal_polish_variance_cost_progress_separation_v1"
_UTC = "2026-07-12T17:35:35Z"
_MEMO = ".omx/research/ugc_terminal_polish_ab_20260712.md"
_RECEIPT = "experiments/results/ugc_terminal_polish_ab_20260712/measurement_receipt.json"
_AXIS = "[macOS-CPU advisory . frozen CPU-torch exact cells . NON-PROMOTABLE]"

UGC_EXACT_TRACE_VARIANCE = 1.2364343091476812e-12
DISARM_EXACT_TRACE_VARIANCE = 1.8123283088760382e-12
UGC_TO_DISARM_VARIANCE_RATIO = UGC_EXACT_TRACE_VARIANCE / DISARM_EXACT_TRACE_VARIANCE
UGC_IMPROVEMENT_PER_EVAL = 1.1756251378815252e-7
ES_IMPROVEMENT_PER_EVAL = 1.285825766295795e-7
UGC_TO_ES_PROGRESS_RATIO = UGC_IMPROVEMENT_PER_EVAL / ES_IMPROVEMENT_PER_EVAL


def build_ugc_terminal_polish_variance_cost_progress_separation_v1() -> CanonicalEquation:
    """Build the measured, non-promotable variance/cost/progress separation law."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "re-measure on a new archive/support/probability geometry or a materially larger exact-call "
            "budget; do not generalize this instance/formulation-scoped loss to the UGC family"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="apple_m5_max_cpu",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ugc_terminal_polish_n600_composed_k6_budget64_20260712",
        measurement_utc=_UTC,
        inputs={
            "archive_sha256": "9c2afa96abdd6fa401bbdfa7a29a7f26ef67c70540656b6fd9ffd87d0bb91d6c",
            "n_pairs_authority": 600,
            "active_pairs": [144, 147, 150, 153, 156, 159],
            "probabilities": [1 / 24, 1 / 24, 1 / 24, 0.5, 0.5, 0.5],
            "ugc_tau": 1 / 12,
            "exact_function_eval_budget_per_arm": 64,
            "seed": 396400,
        },
        predicted_output={
            "hypothesis": "UGC's boundary variance reduction improves fixed-call exact polish progress",
        },
        empirical_output={
            "ugc_exact_trace_variance": UGC_EXACT_TRACE_VARIANCE,
            "disarm_exact_trace_variance": DISARM_EXACT_TRACE_VARIANCE,
            "ugc_to_disarm_variance_ratio": UGC_TO_DISARM_VARIANCE_RATIO,
            "ugc_improvement_per_eval": UGC_IMPROVEMENT_PER_EVAL,
            "one_plus_one_es_improvement_per_eval": ES_IMPROVEMENT_PER_EVAL,
            "ugc_to_es_progress_ratio": UGC_TO_ES_PROGRESS_RATIO,
            "ugc_delta_s": -7.5240008824417615e-6,
            "one_plus_one_es_delta_s": -8.229284904293088e-6,
            "verdict": "UGC_LOSES_INSTANCE_FORMULATION_SCOPED",
        },
        residual=abs(1.0 - UGC_TO_ES_PROGRESS_RATIO),
        source_artifact=_RECEIPT,
        measurement_method=(
            "same-budget exact black-box A/B; n600 base-cell composition with fresh frozen CPU-torch "
            "verification of every final canonical-16 chunk; exhaustive 64-state estimator moments"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="UGC variance reduction does not imply fixed-call terminal-polish progress",
        one_line_summary=(
            "UGC/DisARM exhaustive variance ratio is 0.682235, but UGC/(1+1)-ES exact progress-per-"
            "call ratio is 0.914296 at K=6 and B=64 because estimator call cost reduces proposals."
        ),
        latex_form=(
            r"r_V=\operatorname{tr}\Sigma_{\rm UGC}/\operatorname{tr}\Sigma_{\rm DisARM}"
            r"=0.682235,\qquad r_P=(-\Delta S_{\rm UGC}/B)/(-\Delta S_{\rm ES}/B)=0.914296;"
            r"\quad r_V<1\not\Rightarrow r_P>1"
        ),
        python_callable_module_path="tac.through_r.mc_finisher:measure_estimator_variance",
        domain_of_validity={
            "scope_level": "instance/formulation",
            "archive_sha256": "9c2afa96abdd6fa401bbdfa7a29a7f26ef67c70540656b6fd9ffd87d0bb91d6c",
            "objective": "n600-composed exact S with six direction-pinned pair-local mask bits",
            "probability_geometry": "three p=1/24 boundary coordinates plus three p=1/2 interior",
            "budget": "64 exact function evaluations per variance arm and search arm",
            "authority": _AXIS,
            "exclusions": [
                "not contest-CPU/CUDA score evidence",
                "not a UGC-family death verdict",
                "not transferable to new supports, archives, probability geometries, or budgets",
            ],
        },
        units_in={
            "function_eval_budget": "exact frozen-scorer calls",
            "gradient": "dS/d Bernoulli logit",
        },
        units_out={
            "trace_variance": "S^2 per logit^2",
            "improvement_per_eval": "negative Delta S per exact function evaluation",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "ugc_progress_deficit_fraction_vs_es": abs(1.0 - UGC_TO_ES_PROGRESS_RATIO),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.through_r.mc_finisher.DirectionPinnedMaskFinisher",
            "tools.ugc_terminal_polish_ab",
        ),
        canonical_producers=(
            "tac.through_r.mc_finisher.exact_bernoulli_estimator_moments",
            "tac.through_r.mc_finisher.measure_estimator_variance",
        ),
        provenance=provenance,
    )


def populate_ugc_terminal_polish_variance_cost_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append a latest-row-wins copy through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ugc_terminal_polish_variance_cost_progress_separation_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "ugc_terminal_polish_ab_20260712; UGC exact variance win but same-budget exact-progress "
            "loss; instance/formulation scoped; DSL N/A"
        ),
    )
    return equation


__all__ = [
    "DISARM_EXACT_TRACE_VARIANCE",
    "EQUATION_ID",
    "ES_IMPROVEMENT_PER_EVAL",
    "UGC_EXACT_TRACE_VARIANCE",
    "UGC_IMPROVEMENT_PER_EVAL",
    "UGC_TO_DISARM_VARIANCE_RATIO",
    "UGC_TO_ES_PROGRESS_RATIO",
    "build_ugc_terminal_polish_variance_cost_progress_separation_v1",
    "populate_ugc_terminal_polish_variance_cost_equation",
]
