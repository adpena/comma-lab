# SPDX-License-Identifier: MIT
"""Canonical fixed-k SFESS cached-replay ranking law.

On the sealed six-bit objective, every non-degenerate SFESS arm reached its
within-cardinality minimum at 64 counted cache lookups.  The best such arm,
``k=5``, still remained above exact enumeration and the registered `(1+1)-ES`
control.  The ``k=6`` equality is a one-state structural control and therefore
contains no score-function-estimator evidence.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "sfess_fixed_k_cached_replay_ranking_v1"
MEASUREMENT_UTC = "2026-07-12T21:45:20Z"
RECEIPT = (
    "experiments/results/sfess_cached_replay_ugc64_20260712T214520Z/"
    "measurement_receipt.json"
)
AXIS = "[macOS-CPU advisory . frozen CPU-torch exact cells . NON-PROMOTABLE]"
OBJECTIVE_TABLE_SHA256 = "249c19af0b8c117412de491e944bcacb6194c870c9d9ec57d5c93b5e55f1a979"
RECEIPT_SHA256 = "aa296c61fde712f9a2207ff5ecf9298c2506c92e3a48af8ac2af3d9bc83e6c9e"
BEST_NONDEGENERATE_K = 5
BEST_SFESS_S = 0.19080429731336374
EXACT_ENUMERATION_S = 0.19080359202934188
SFESS_EXACT_GAP_S = BEST_SFESS_S - EXACT_ENUMERATION_S
COMPARISON_NOISE_FLOOR_S = 1.0e-12


def build_sfess_fixed_k_cached_replay_ranking_v1() -> CanonicalEquation:
    """Build the measured instance/formulation-scoped SFESS ranking law."""

    # FROM-LITERATURE at the derivation point: Klas Wijk, Ricardo Vinuesa,
    # Hossein Azizpour (2024), "Revisiting Score Function Estimators for
    # k-Subset Sampling", arXiv:2407.16058.  It supplies the conditional-
    # Bernoulli score and M-sample leave-one-out control variate.
    # FROM-LITERATURE: Manuel Fernandez, Stuart Williams (2010), "Closed-Form
    # Expression for the Poisson-Binomial Probability Density Function",
    # DOI:10.1109/TAES.2010.5461658.  It supplies the DFT normalizer.
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=RECEIPT,
        reactivation_criteria=(
            "register a non-enumerable support and re-run the same strict-gated returned-state "
            "comparison at a pre-registered matched query budget; do not generalize this six-bit "
            "instance NO-GO to the SFESS family"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_m5_max_cpu",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="sfess_cached_ugc64_k_ladder_budget64_20260712",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "objective_table_sha256": OBJECTIVE_TABLE_SHA256,
            "objective_states": 64,
            "n_bits": 6,
            "k_values": [1, 2, 3, 4, 5],
            "degenerate_controls": [0, 6],
            "samples_per_gradient": 5,
            "conditional_probabilities": "constant p_i=k/6",
            "function_eval_budget_per_arm": 64,
            "seed": 396_400,
            "retention_rule": "strict_gated_returned_state",
            "comparison_noise_floor_s": COMPARISON_NOISE_FLOOR_S,
        },
        predicted_output={
            "hypothesis": "fixed-cardinality SFESS changes the same-budget baseline ranking"
        },
        empirical_output={
            "best_nondegenerate_k": BEST_NONDEGENERATE_K,
            "best_sfess_s": BEST_SFESS_S,
            "exact_enumeration_s": EXACT_ENUMERATION_S,
            "one_plus_one_es_s": EXACT_ENUMERATION_S,
            "sfess_minus_exact_gap_s": SFESS_EXACT_GAP_S,
            "all_nondegenerate_arms_reached_within_k_minimum": True,
            "k6_is_one_state_control_not_estimator_evidence": True,
            "same_budget_ranking_changed": False,
            "verdict": "NO-GO",
            "scorer_calls": 0,
        },
        residual=SFESS_EXACT_GAP_S,
        source_artifact=RECEIPT,
        measurement_method=(
            "clean-room DFT conditional-Bernoulli SFESS replay over one SHA-pinned measured "
            "64-state table; five estimator samples plus one separately counted strict exact "
            "swap gate with the registered 1e-12 comparison floor; 64 counted cache lookups per "
            "arm; no scorer or repacker calls"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=COMPARISON_NOISE_FLOOR_S,
        noise_floor_provenance=(
            "inherited 1e-12 exact-composition verification tolerance from the sealed UGC receipt; "
            "single-seed across-seed variance UNKNOWN"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Fixed-k SFESS does not change the cached six-bit same-budget ranking",
        one_line_summary=(
            "At B=64, k=5 SFESS returned S=0.19080429731336374, which is "
            "7.052840218513268e-7 above exact enumeration and (1+1)-ES; k=6 is degenerate."
        ),
        latex_form=(
            r"S^*_{\mathrm{SFESS},1:5}=S^*_{k=5}=0.19080429731336374,\quad "
            r"S^*_{\mathrm{enum}}=S^*_{(1+1)\mathrm{ES}}=0.19080359202934188,\quad "
            r"\Delta=7.052840218513268\times10^{-7}>\epsilon_S=10^{-12}"
        ),
        python_callable_module_path=(
            "tac.sfess_cached_replay:sfess_leave_one_out_gradient"
        ),
        domain_of_validity={
            "scope_level": "instance/formulation",
            "objective_table_sha256": OBJECTIVE_TABLE_SHA256,
            "objective": "six direction-pinned pair-local terminal edit bits",
            "budget": "64 counted cached objective lookups per arm",
            "authority": AXIS,
            "review_status_at_measurement": "recovery-written-UNREVIEWED",
            "exclusions": [
                "not contest-CPU/CUDA score evidence",
                "not a live scorer-costate provider",
                "not a SFESS-family death verdict",
                "not transferable to non-enumerable supports, new archives, or new budgets",
                "k=6 one-state support is not estimator evidence",
            ],
        },
        units_in={
            "objective": "contest S values inherited from the measured cache",
            "function_eval_budget": "counted cached objective lookups",
            "gradient": "dS/d conditional-Bernoulli logit",
        },
        units_out={
            "returned_best": "contest S on the cached local-advisory objective",
            "ranking_gap": "contest S",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "best_sfess_minus_exact_enumeration_s": SFESS_EXACT_GAP_S,
            "gap_over_noise_floor": SFESS_EXACT_GAP_S / COMPARISON_NOISE_FLOOR_S,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_sfess_cached_replay",
            "tac.sfess_cached_replay.SFESSFixedKSearch",
        ),
        canonical_producers=(
            "tac.sfess_cached_replay.sfess_leave_one_out_gradient",
            "tools.probe_sfess_cached_replay",
        ),
        provenance=provenance,
    )


def populate_sfess_fixed_k_cached_replay_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append a latest-row-wins copy through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_sfess_fixed_k_cached_replay_ranking_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "sfess_cached_replay_ugc64_20260712; exact enumeration remains lower; "
            "instance/formulation-scoped NO-GO; terminal-objective DSL only"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "BEST_NONDEGENERATE_K",
    "BEST_SFESS_S",
    "COMPARISON_NOISE_FLOOR_S",
    "EQUATION_ID",
    "EXACT_ENUMERATION_S",
    "MEASUREMENT_UTC",
    "OBJECTIVE_TABLE_SHA256",
    "RECEIPT",
    "RECEIPT_SHA256",
    "SFESS_EXACT_GAP_S",
    "build_sfess_fixed_k_cached_replay_ranking_v1",
    "populate_sfess_fixed_k_cached_replay_equation",
]
