# SPDX-License-Identifier: MIT
"""Canonical equation: step-native activation L-infinity-at-edge optimality (DAG FEED-07b lever #2,
capstone lever #5 — the in-code UNSWEPT step-native chart; #310 sweep is the owed anchor).

For a PIECEWISE-CONSTANT argmax target (the SegNet partition), a step-native / partition-indicator
chart is the topology-matched representation: it places the full approximation error INSIDE the
codim-1 flip band (L-infinity error AT the edge, O(1) params per edge, no Gibbs ringing), whereas a
smooth sine/relu chart pays a spatially-extended Gibbs/blur penalty for every edge it sharpens:

    err_stepnative(x) ~ 0 for dist(x, boundary) > w_flip;   err_smooth(x) ~ Gibbs ripple, O(1/N) decay

In the LIVE levelset trainer the step-native route is the hosc BETA-ANNEAL (FEED-fb): hosc is
``tanh(beta * sin(omega * u))`` and ``beta -> inf`` IS the step limit, so ``--hosc-beta-end >
--hosc-beta`` step-sharpens the activation as the SDF partition forms. CAVEAT (MEASURED, DAG FEED
2026-06-25a + FEED-ly): FIXED high beta (beta=4 constant from scratch) DIVERGES (tanh saturation ->
vanishing grad); the step limit must be APPROACHED by anneal, never started at. A true
``--activation step_basis`` choice does NOT exist in this trainer's argparse (choices are
wire/hosc/relu) — the discrete step_basis chart + FINER++ bias-init are BUILD-NEEDED (#310 sisters).

Anchors (honest tiers): the deep-math L-infinity-at-edge optimality is
INFERRED_FROM_DOMAIN_LITERATURE (piecewise-constant approximation theory; deep-math ch.1-6
"Amortizing the Argmax": MCF/spectral-bias erases sub-band structure a step chart keeps); the
capstone's best NON-step measured d_seg 0.004445 (~4.4x above the ~0.001 need, ep100; ep450 cap arm
0.002447 ~2.8x) bounds what the smooth chart achieved WITHOUT the lever — the step-native delta
itself is ASSUMED_AWAITING_VERIFICATION until the #310 sweep lands.

DSL leg: ``tac.witness_dsl.curriculum_dsl.StepNativeActivation`` (the beta-anneal over the real
``--hosc-beta`` / ``--hosc-beta-end`` / ``--hosc-beta-anneal`` flags).

means != ends: advisory anchors, NON-PROMOTABLE; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "step_native_activation_edge_optimality_v1"

_UTC = "2026-07-07T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- MEASURED constants (the non-step chart's best, bounding the unswept lever) -------------------
CAPSTONE_BEST_NONSTEP_DSEG_EP100 = 0.004445   # all-class-dir+cap @ep100 (~4.4x above ~0.001 need)
CAPSTONE_BEST_NONSTEP_DSEG_EP450 = 0.002447   # same arm @ep400/450 (~2.8x above the need)
DSEG_NEED_APPROX = 0.001                      # the approximate witness d_seg need (capstone framing)


def hosc_step_limit_beta_ratio(beta_start: float, beta_end: float) -> float:
    """The step-sharpening ratio beta_end/beta_start of the hosc anneal (the trainer's step-native
    route: tanh(beta*sin) -> step as beta -> inf). Must be > 1 for a genuine step-anneal; == 1 is
    the bit-identical no-anneal path. Raises on non-positive betas (fail-closed)."""
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError(f"hosc betas must be > 0, got start={beta_start!r} end={beta_end!r}")
    return float(beta_end) / float(beta_start)


def build_step_native_activation_edge_optimality_v1() -> CanonicalEquation:
    """Build the step-native L-infinity-at-edge optimality equation (owed to the #310 sweep)."""

    anchor_nonstep_bound = EmpiricalAnchor(
        anchor_id="capstone_nonstep_best_dseg_bound_20260625",
        measurement_utc="2026-06-25T00:00:00Z",
        inputs={
            "vehicle": "capstone all-class-dir(+cap) coord-INR (witness-build a922483dfc636ccc3 + "
                       "ep450 daemons, DAG FEEDs 2026-06-25t/u)",
            "authority": "frozen CPU-torch SegNet argmax (MLX gradient-only, NEVER MPS)",
            "activation": "non-step (relu/smooth chart; step/gauss UNSWEPT at measurement time)",
        },
        predicted_output={"deep_math": "smooth charts pay Gibbs/blur per edge; gap to need remains"},
        empirical_output={
            "best_dseg_ep100": CAPSTONE_BEST_NONSTEP_DSEG_EP100,
            "best_dseg_ep450": CAPSTONE_BEST_NONSTEP_DSEG_EP450,
            "need": DSEG_NEED_APPROX,
            "verdict": ("non-step chart converged ~2.8-4.4x ABOVE the ~0.001 need and mildly "
                        "overfit past ep400 => further gain needs a NEW lever (the step-native "
                        "chart is the named unswept #5 lever), not more epochs"),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="capstone lever sweep + ep450 daemons on frozen CPU-torch SegNet argmax",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="supersede with the #310 step-native sweep rows when they land",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_step_delta_owed = EmpiricalAnchor(
        anchor_id="step_native_delta_owed_310_sweep_20260707",
        measurement_utc=_UTC,
        inputs={
            "literature": ("piecewise-constant approximation theory (L-infinity-at-edge, O(1) "
                           "params/edge, no Gibbs) + deep-math 'Amortizing the Argmax' ch.1-6"),
            "trainer_route": "--hosc-beta -> --hosc-beta-end anneal (tanh(beta*sin) -> step limit)",
            "caveat": ("MEASURED: FIXED beta=4 from scratch DIVERGES (tanh saturation); anneal "
                       "the step limit, never start at it (DAG FEED 2026-06-25a + FEED-ly)"),
        },
        predicted_output={
            "claim": "step-native chart concentrates error in the flip band => d_seg drop at ~0 bytes",
        },
        empirical_output={
            "status": "OWED — the #310 step-native sweep (beta-anneal arm A/B) is the owed anchor",
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="derivation only (no step-native run yet; #310 sweep owed)",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="step_native_activation_edge_optimality.step_delta",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Step-native activation L-infinity-at-edge optimality: for piecewise-constant argmax "
              "targets the step chart is topology-matched (error inside the flip band, no Gibbs); "
              "trainer route = hosc beta-anneal (beta->inf is the step limit; fixed-high-beta diverges)"),
        one_line_summary=(
            "Step-native charts put all error at the edge (no Gibbs) for the piecewise-constant "
            "argmax; non-step best 0.004445 bounds the unswept lever; delta OWED to #310."
        ),
        latex_form=(
            r"\sigma_\beta(u)=\tanh(\beta\sin(\omega u)),\ \ \lim_{\beta\to\infty}\sigma_\beta="
            r"\mathrm{step};\quad \|e_{\mathrm{step}}\|_{L^\infty(\Omega\setminus B_{w})}\approx 0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.step_native_activation_edge_optimality_20260707:"
            "hosc_step_limit_beta_ratio"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": ("tac.witness_dsl.curriculum_dsl.StepNativeActivation "
                      "(--hosc-beta/--hosc-beta-end/--hosc-beta-anneal)"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("approach the step limit by ANNEAL only (fixed high beta diverges, measured); "
                     "a discrete --activation step_basis choice + FINER++ bias-init are BUILD-NEEDED "
                     "(#310) — this trainer's argparse offers wire/hosc/relu only"),
        },
        units_in={"beta_start": "dimensionless", "beta_end": "dimensionless"},
        units_out={"ratio": "dimensionless_gt_1_for_anneal"},
        empirical_anchors=(anchor_nonstep_bound, anchor_step_delta_owed),
        predicted_vs_empirical_residual={
            # the non-step chart's measured gap-to-need (ep450 arm): 0.002447/0.001 - 1 = 1.447 —
            # the headroom the step-native lever is predicted to attack; #310 sweep re-anchors this.
            "nonstep_gap_to_need_ep450": abs(
                CAPSTONE_BEST_NONSTEP_DSEG_EP450 / DSEG_NEED_APPROX - 1.0),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.curriculum_dsl",),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="step_native_activation_edge_optimality.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_step_native_activation_edge_optimality_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the step-native edge-optimality law into the canonical
    registry (mirrors ``populate_focal_gradient_concentration_equation``; latest-row-wins query
    semantics). Equations leg of DAG FEED-07b lever #2; DSL leg = ``StepNativeActivation``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_step_native_activation_edge_optimality_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="step_native_activation_edge_optimality_20260707 (equations leg of FEED-07b lever #2; "
              "DSL leg = StepNativeActivation; #310 sweep = owed anchor)",
    )
    return eq


__all__ = [
    "CAPSTONE_BEST_NONSTEP_DSEG_EP100",
    "CAPSTONE_BEST_NONSTEP_DSEG_EP450",
    "DSEG_NEED_APPROX",
    "EQUATION_ID",
    "build_step_native_activation_edge_optimality_v1",
    "hosc_step_limit_beta_ratio",
    "populate_step_native_activation_edge_optimality_equation",
]
