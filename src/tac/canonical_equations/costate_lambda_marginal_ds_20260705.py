# SPDX-License-Identifier: MIT
"""Canonical equation: the COSTATE λ = marginal-ΔS shadow-price field (task #303 Phase A).

The Hamiltonian meta-layer's missing fourth object (memory
``project_meta_layer_above_triality_hamiltonian_control_costate_20260703``): the triality
{DAG=state x(t) · DSL=control u(t) · equations=law S} is completed by the costate

    λ_i(t) = ∂S/∂x_i        (the sensitivity of the goal-value to state)

against the contest score law S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37_545_489:

    λ_seg   = 100                       (exact, constant)
    λ_pose  = 5 / sqrt(10·d_pose)       (exact, state-dependent — the operating-point
                                         crossover in CLAUDE.md "SegNet vs PoseNet")
    λ_bytes = 25 / 37_545_489           (exact, constant; ≈ 6.659e-7 S/byte)

and the measured control-side chain along a trajectory (per stage, LOCAL window because
λ(t) is TIME-VARYING — the #205 tau-onset transient decays ~10x to steady creep):

    dS/dep = λ_seg·(dd_seg/dep) + λ_pose·(dd_pose/dep) + λ_bytes·(dbytes/dep)
             ± sqrt(Σ (λ_i·se_i)²)      (OLS slope stderr, independence approx)

MEASURED empirical anchor: the #303 shadow-controller BACKTEST on the completed #205
run log (n600 advisory verdicts, [macOS-CPU advisory] NON-PROMOTABLE). Rediscovered
without being told: the tau-onset spike is WATCH-not-act at ep325 (transient), SUSTAINED
erosion flags at ep350 (3rd tau verdict), rollback costate +0.166..0.187 S recoverable.
Honest residuals recorded: the ep350 25-ep horizon projection landed IN band but 7x high
centrally (transient decay); the ep450 windowed projection landed BELOW band (deceleration
+ pose offset) — λ(t) drift is real, linear extrapolation overpredicts under decay.

means != ends: advisory shadow rows, never a score; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "costate_lambda_marginal_ds_v1"
_UTC = "2026-07-05T07:30:00Z"
_ADVISORY = "macOS-CPU advisory NON-PROMOTABLE"
_PREDICTED = "predicted"
_BACKTEST_MEMO = ".omx/research/costate_controller_design_20260705.md"
_RUN_205 = "experiments/results/levelset_n600_witness_20260703T120444Z/run.log"

# score-law constants (mirrors tac.witness_control.costate_estimator)
_SEG_W = 100.0
_RATE = 25.0 / 37_545_489.0


def costate_vector(d_pose: float) -> tuple[float, float, float]:
    """(λ_seg, λ_pose, λ_bytes) at the measured operating point d_pose > 0."""
    if d_pose <= 0.0:
        raise ValueError("λ_pose = 5/sqrt(10·d_pose) diverges at d_pose <= 0 "
                         "(UNIDENTIFIABLE operating point)")
    return (_SEG_W, 5.0 / math.sqrt(10.0 * d_pose), _RATE)


def chained_ds_depoch(dseg_slope: float, dpose_slope: float, dbytes_slope: float,
                      d_pose: float) -> float:
    """dS/dep = λ·dx/dep at the operating point (the control-side chain rule)."""
    lam_seg, lam_pose, lam_bytes = costate_vector(d_pose)
    return lam_seg * dseg_slope + lam_pose * dpose_slope + lam_bytes * dbytes_slope


def build_costate_lambda_marginal_ds_v1() -> CanonicalEquation:
    """Build the costate definition equation with the #205 backtest anchor."""
    anchor = EmpiricalAnchor(
        anchor_id="costate_shadow_backtest_205_20260705",
        measurement_utc=_UTC,
        inputs={
            "run_log": _RUN_205,
            "verdicts": "n600 advisory verdict rows ep0..525 (CE ep25-275, tau ep300-525)",
            "tool": "tools/costate_shadow_report.py --as-of-epoch {300,325,350,400,450,525}",
        },
        predicted_output={
            "ep325": "WATCH (transition transient, do not act)",
            "ep350": "DIVERGING_ERASING -> ROLLBACK (ΔS -0.1655) + STOP (ΔS -0.0825 "
                     "band [-0.1552,-0.0097] over 25ep)",
            "ep450_windowed_25ep": "+0.0060 band [+0.0018,+0.0103] continued-creep cost",
        },
        empirical_output={
            "ep325_realized": "correct: the ep300->325 rise was onset-transient-shaped; "
                              "erosion only confirmable at the 3rd tau verdict",
            "ep350_horizon_realized": "+0.0119 implied-S over ep350->375 — INSIDE the "
                                      "predicted band (central 7x high: transient decay)",
            "ep450_horizon_realized": "+0.0004 implied-S over ep450->475 — BELOW the band "
                                      "(creep deceleration + pose still descending)",
            "rollback_recoverable": "+0.166..+0.187 S (advisory implied), best=ep300",
            "l7_muon_costates": "UNIDENTIFIABLE (stages never ran in any available log) — "
                                "returned honestly, not guessed",
        },
        residual=0.0706,  # |predicted 0.0825 − realized 0.0119| ep350 horizon (in-band but wide)
        source_artifact=_BACKTEST_MEMO,
        measurement_method="shadow-controller backtest: as-of truncated replay over the real log",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BACKTEST_MEMO,
            reactivation_criteria=("re-anchor when a run reaches l7/Muon stages (their "
                                   "costates are the named identifiability gaps) or when a "
                                   "second-order λ-decay model lands (the ep450 below-band "
                                   "miss is its trigger)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Costate λ = ∂S/∂x: (100, 5/sqrt(10·d_pose), 25/37545489) + chained "
              "dS/dep = λ·dx/dep ± sqrt(Σ(λ_i·se_i)²) — the triality's fourth object"),
        one_line_summary=(
            "The marginal-ΔS shadow price: exact score-law partials + measured windowed "
            "trajectory chain; BACKTESTED on #205 (WATCH@325, EROSION@350, rollback +0.17 S)."
        ),
        latex_form=(
            r"\lambda=\Big(100,\ \frac{5}{\sqrt{10\,d_{pose}}},\ \frac{25}{37545489}\Big),\quad"
            r"\frac{dS}{dep}=\lambda\cdot\frac{dx}{dep}\ \pm\ "
            r"\sqrt{\textstyle\sum_i(\lambda_i\,se_i)^2}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.costate_lambda_marginal_ds_20260705:chained_ds_depoch"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "state": "aggregate (d_seg, d_pose, blob_bytes) verdict rows; per-class λ is a "
                     "NAMED GAP until #253/#255 attribution feeds per-class rows",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": ("λ(t) is time-varying: use the LOCAL window (5 same-stage verdicts); "
                     "linear horizon projection overpredicts under transient decay "
                     "(ep450 below-band miss recorded)"),
        },
        units_in={"dseg_slope": "d_seg per epoch", "dpose_slope": "d_pose per epoch",
                  "dbytes_slope": "bytes per epoch", "d_pose": "dimensionless"},
        units_out={"dS_depoch": "S per epoch (advisory implied-S units)"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "ep350_horizon_25ep_in_band_central_gap": 0.0706,
            "ep450_horizon_25ep_below_band_gap": 0.0056,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_control.shadow_controller",),
        canonical_producers=(
            "tools/costate_shadow_report.py",
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="costate_lambda_marginal_ds.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_costate_lambda_marginal_ds_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the costate λ = marginal-ΔS equation (#303 Phase A)
    into the canonical registry (mirrors ``populate_adaptive_eps_cfl_edge_tracking_equation``;
    latest-row-wins query semantics). This is the EQUATIONS leg of the Hamiltonian meta-layer
    triality (#247): the costate λ COMPLETES {DAG=state · DSL=control · equations=law} with the
    fourth object λ_i = ∂S/∂x_i (memory
    ``project_meta_layer_above_triality_hamiltonian_control_costate_20260703``). Was ORPHANED
    (built + tested but never imported into ``__init__`` / wired via a populate_*)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_costate_lambda_marginal_ds_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="costate_lambda_marginal_ds_20260705_deorphan_20260706 (equations leg of the "
              "Hamiltonian meta-layer triality #247/#303)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_costate_lambda_marginal_ds_v1",
    "chained_ds_depoch",
    "costate_vector",
    "populate_costate_lambda_marginal_ds_equation",
]
