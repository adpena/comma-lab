#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register canonical equation `multiop_composition_pareto_optimal_savings_v1`.

FORMALIZATION_PENDING per Catalog #344. Predicted-only; macOS-CPU advisory
NON-PROMOTABLE per Catalog #127/#192/#323. Carries one [predicted] anchor from
the closed-form multi-op composition prediction sweep
(`tools/canvas_multiop_composition_closed_form_prediction_sweep.py`).

The equation formalizes the PARADOX-CLOSER Half 2 closed-form prediction:

    ΔS_multiop_optimal(ops, frontier) =
        ΔS_best_single_op
        - max(0, <grad_axes, dykstra_allocation>) / 37_545_489

where:
  - ΔS_best_single_op = best single-op predicted score reduction from the 12
    canvas operators
  - grad_axes = per-substrate-axis (seg, pose, rate) gradient L2 norms derived
    from the canvas archive-aggregate cells
  - dykstra_allocation = optimal byte-budget allocation from the META-LIFT-2
    Pareto polytope Dykstra alternating-projections solver (rate ∩ SegNet ∩
    PoseNet ∩ Cauchy-Schwarz polytope intersection)

DOMAIN: the multi-op extra synergy term is BOUNDED by the Dykstra-feasible
polytope intersection; at archive-aggregate canvas resolution (single
pair_idx=0/frame_idx=0 coordinate) the operators cannot compose and the extra
term collapses to ~0 → multi-op ≈ single-op (operating-point saturation).
"""

from __future__ import annotations

from tac.canonical_equations import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
    query_equations,
    register_canonical_equation,
)
from tac.provenance import build_provenance_for_research_sidecar

EQUATION_ID = "multiop_composition_pareto_optimal_savings_v1"
SWEEP_MEMO = (
    ".omx/research/canvas_multiop_composition_closed_form_prediction_sweep_landed_20260527.md"
)


def build_equation() -> CanonicalEquation:
    # Closed-form predicted anchor from the sweep: at fec6 6bae0201 archive-
    # aggregate canvas resolution, 0 operators compose → predicted multi-op
    # extra delta ~ -2.5e-19 → multi-op ≈ single-op (operating-point saturation
    # at the available ledger resolution; single-op locally optimal).
    predicted_extra = -2.549318241666795e-19
    # The "empirical" output here is the closed-form Dykstra-solver-computed
    # value (NOT a paid run); residual is 0 because the prediction IS the
    # closed-form computation (this is a FORMALIZATION_PENDING predicted anchor).
    empirical_extra = -2.549318241666795e-19

    anchor = EmpiricalAnchor(
        anchor_id="multiop_composition_closed_form_fec6_aggregate_predicted_20260527",
        measurement_utc="2026-05-27T13:14:00Z",
        inputs={
            "frontier_archive_sha256": (
                "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
            ),
            "frontier_which": "fec6_fallback",
            "canvas_cell_count": 3,
            "canvas_resolution": "archive_aggregate_pair0_frame0",
            "n_productive_operators": 0,
            "dykstra_feasible": True,
            "dykstra_feasibility_residual": 3.3928667952091836e-12,
            "axis": "[predicted]",
        },
        predicted_output={"multiop_extra_delta_s": predicted_extra},
        empirical_output={"closed_form_dykstra_extra_delta_s": empirical_extra},
        residual=abs(predicted_extra - empirical_extra),
        source_artifact=SWEEP_MEMO,
        measurement_method=(
            "canvas_multiop_composition_closed_form_prediction_sweep_dykstra_polytope"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=SWEEP_MEMO,
            reactivation_criteria=(
                "per_pair_fp64_decomposition_ledger_lands_then_re_run_sweep_with_"
                "multi_pair_canvas_then_3plus_paired_cpu_cuda_anchors"
            ),
            measurement_axis="[predicted]",
            hardware_substrate="macos_arm64",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Multi-op composition Pareto-optimal savings",
        one_line_summary=(
            "Optimal multi-op composition delta = best single-op delta minus the "
            "Dykstra-feasible polytope synergy term; synergy collapses to 0 at "
            "archive-aggregate canvas resolution."
        ),
        latex_form=(
            r"\Delta S_{\mathrm{multiop}}^{*} = \Delta S_{\mathrm{single}}^{*} "
            r"- \frac{\max(0, \langle g_{\mathrm{axes}}, x_{\mathrm{Dykstra}} "
            r"\rangle)}{37{,}545{,}489}"
        ),
        python_callable_module_path=(
            "tools.canvas_multiop_composition_closed_form_prediction_sweep:main"
        ),
        domain_of_validity={
            "candidate_family": ["pair_frame_5d_canvas_12_operator_composition"],
            "operation": [
                "full_drop",
                "repair",
                "masked",
                "feathered",
                "replace_one",
                "replace_many",
                "merge_pair",
                "reorder_pair",
                "drop_frame",
                "synthesize_frame",
                "motion_conditional",
                "temporal_coherence",
            ],
            "axes": ["[predicted]", "[contest-CPU]", "[contest-CUDA T4]"],
            "canvas_resolution_required_for_synergy": "multi_pair_or_multi_frame",
            "identity_required": [
                "frontier_archive_sha256",
                "canvas_cell_count",
                "dykstra_feasible",
            ],
            "formalization_status": "FORMALIZATION_PENDING",
            "promotable": False,
            "score_claim": False,
        },
        units_in={
            "best_single_op_delta_s": "float_score_units",
            "per_axis_gradient_l2_norms": "float_per_axis_gradient_magnitude",
            "dykstra_allocation": "float_byte_budget_units",
            "archive_byte_delta": "float_archive_bytes",
        },
        units_out={
            "multiop_optimal_delta_s": "float_score_units_negative_is_better",
            "multiop_extra_delta_s": "float_score_units_synergy_beyond_single_op",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.measurement_method: anchor.residual},
        last_calibration_utc="2026-05-27T13:14:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.cathedral_autopilot_autonomous_loop",
            "tac.pareto_polytope_unified_solver.solver",
            "tac.optimization.decoder_q_pairset_acquisition",
        ),
        canonical_producers=(
            "tools.canvas_multiop_composition_closed_form_prediction_sweep",
            "tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=SWEEP_MEMO,
            reactivation_criteria=(
                "per_pair_fp64_decomposition_ledger_lands_OR_3plus_paired_cpu_cuda_anchors"
            ),
            measurement_axis="[predicted]",
            hardware_substrate="macos_arm64",
        ),
    )


def main() -> int:
    existing = {e.equation_id for e in query_equations()}
    if EQUATION_ID in existing:
        print(f"equation {EQUATION_ID} already registered; skipping (append-only).")
        return 0
    eq = build_equation()
    registered = register_canonical_equation(
        eq,
        agent="claude",
        subagent_id="canvas_multiop_RESUME1",
        notes=(
            "FORMALIZATION_PENDING per Catalog #344; PARADOX-CLOSER Half 2 "
            "closed-form multi-op composition prediction; predicted-only "
            "macOS-CPU advisory NON-PROMOTABLE per Catalog #127/#192/#323"
        ),
    )
    print(f"registered canonical equation: {registered.equation_id}")
    print(f"  anchors: {len(registered.empirical_anchors)}")
    print(f"  residual: {registered.predicted_vs_empirical_residual}")
    print(f"  consumers: {len(registered.canonical_consumers)}")
    print(f"  producers: {len(registered.canonical_producers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
