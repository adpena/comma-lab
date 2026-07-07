# SPDX-License-Identifier: MIT
"""Canonical equations: the §15 $0 solver pack — (A) junction Young-angle sigma_ij fit +
(B) weak-KAM power-law tail exit rule. MEASURED 2026-07-07 (viscosity hunt §7 + §4 bridges).

(A) ``junction_young_angle_sigma_fit_v1`` (spec'd in viscosity_theory_alignment_hunt_20260707.md
§9.2): the frozen SegNet's triple-junction angles define, via Young's law
(sigma_jk/sin th_i = sigma_ik/sin th_j = sigma_ij/sin th_k), a pairwise surface-tension matrix
sigma_ij; the all-ones default (today's scalar ``--length-weight``) imposes Herring 120-120-120
and fights the target at junctions. MEASURED (tools/fit_junction_sigma_youngs_law.py, cached
n600 GT argmax, [macOS-CPU advisory]): 3256 clean triple junctions; mean |angle - 120| = 36.2
deg; 19.3% of junctions have an arc >= 180 deg (positive-tension equilibrium impossible there —
a Herring violation by itself); the dominant BULK triple Road-Undrivable-Movable (n=1815) is
near-Herring ([120.0, 123.7, 116.3] deg — all-ones approximately CORRECT for bulk), while every
Lane-involving triple is far from it (Lane 28.4 deg at R-L-U; MyCar 167.8 deg at R-L-MyCar).
Fitted sigma (geometric-mean-1 gauge; bootstrap 95% CIs): **sigma[Road-Lane] = 0.377
[0.317, 0.441]** — the uniform length weight over-penalizes lane boundary length ~2.7x relative
to the scorer's own junction geometry (a named mechanism feeding lane erasure); sigma[Road-MyCar]
= 1.78, sigma[Lane-MyCar] = 1.76 (hood boundary stiff); 4 of 7 fitted pairs exclude 1.0.

(B) ``weak_kam_powerlaw_tail_exit_v1`` (hunt §4, "fit owed" — now run): late descent on the
binding class is power-law O(1/t)-class, so exponential plateau detectors fire EARLY. MEASURED
(tools/fit_powerlaw_plateau_detector.py): long900 (19 pts, ep1-900) prefers the power law by
dAIC +45.4 (alpha = 0.51, ci95 [0.053, 0.554]); the exponential fit's asymptote 0.002464 was
BROKEN by the measured trajectory (0.002017@ep900, 22% below the exp floor) — the quantified
"meat left on the bone" failure of exponential-window detectors. Early stage windows are
exponential-preferred (mod32cap CE dAIC -20.4, tau dAIC -46.9) — consistent with §4's
off-obstruction-set exponential regime. Exit rule: extrapolated remaining meat
d(t_now) - d(t_now + horizon) under the AIC-preferred model, never a window slope
(``tac.witness_control.powerlaw_exit:powerlaw_meat_exit``). Pre-registered alpha_lane <
alpha_road: UNRESOLVED APPARATUS GAP (no per-class d_seg trajectory is logged by any run;
duty-to-measure recorded).

means != ends: advisory geometry/shape calibrations, NOT scores; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

SIGMA_EQUATION_ID = "junction_young_angle_sigma_fit_v1"
POWERLAW_EQUATION_ID = "weak_kam_powerlaw_tail_exit_v1"

_UTC = "2026-07-07T18:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"
_HUNT_MEMO = ".omx/research/viscosity_theory_alignment_hunt_20260707.md"
_PACK_MEMO = ".omx/research/solver_pack_junction_sigma_powerlaw_20260707.md"
_SIGMA_JSON = "experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json"
_POWERLAW_JSON = "experiments/results/solver_pack_20260707/powerlaw_detector/powerlaw_fits.json"

# --- (A) MEASURED sigma_ij (geometric-mean-1 gauge; canonical class order Road0/Lane1/
#     Undrivable2/Movable3/MyCar4; NaN = pair unobserved at min 30 junctions/triple) -------------
SIGMA_ROAD_LANE = 0.377          # ci95 [0.317, 0.441] — EXCLUDES all-ones (the headline)
SIGMA_ROAD_UNDRIVABLE = 1.085    # ci95 [0.994, 1.206] — consistent with all-ones
SIGMA_ROAD_MOVABLE = 1.006       # ci95 [0.918, 1.125] — consistent with all-ones
SIGMA_ROAD_MYCAR = 1.779         # ci95 [1.459, 2.150] — EXCLUDES all-ones
SIGMA_LANE_UNDRIVABLE = 0.738    # ci95 [0.568, 0.922] — EXCLUDES all-ones
SIGMA_LANE_MYCAR = 1.764         # ci95 [1.434, 2.119] — EXCLUDES all-ones
SIGMA_UNDRIVABLE_MOVABLE = 1.048  # ci95 [0.955, 1.185] — consistent with all-ones
N_CLEAN_JUNCTIONS = 3256
MEAN_ABS_DEV_FROM_120_DEG = 36.2
FRAC_JUNCTIONS_ARC_GE_180 = 0.193

_NAN = float("nan")
FITTED_SIGMA_MATRIX = (
    # Road      Lane      Undriv    Movable   MyCar
    (0.0, SIGMA_ROAD_LANE, SIGMA_ROAD_UNDRIVABLE, SIGMA_ROAD_MOVABLE, SIGMA_ROAD_MYCAR),
    (SIGMA_ROAD_LANE, 0.0, SIGMA_LANE_UNDRIVABLE, _NAN, SIGMA_LANE_MYCAR),
    (SIGMA_ROAD_UNDRIVABLE, SIGMA_LANE_UNDRIVABLE, 0.0, SIGMA_UNDRIVABLE_MOVABLE, _NAN),
    (SIGMA_ROAD_MOVABLE, _NAN, SIGMA_UNDRIVABLE_MOVABLE, 0.0, _NAN),
    (SIGMA_ROAD_MYCAR, SIGMA_LANE_MYCAR, _NAN, _NAN, 0.0),
)

# --- (B) MEASURED tail-shape fits -----------------------------------------------------------------
LONG900_ALPHA = 0.510            # power-law exponent, ep1-900 window (19 points)
LONG900_ALPHA_CI95 = (0.053, 0.554)
LONG900_DAIC_EXP_MINUS_POW = 45.37   # positive => power law preferred
LONG900_EXP_ASYMPTOTE_BROKEN_BY = 4.47e-4  # exp floor 0.002464 vs measured 0.002017@ep900
MOD32CAP_CE_DAIC_EXP_MINUS_POW = -20.35    # early stage: exponential preferred
MOD32CAP_TAU_DAIC_EXP_MINUS_POW = -46.91   # tau stage: exponential preferred


def fitted_sigma_matrix() -> tuple[tuple[float, ...], ...]:
    """The measured 5x5 pairwise surface-tension matrix (geometric-mean-1 gauge; NaN =
    unobserved pair). Canonical class order Road0/Lane1/Undrivable2/Movable3/MyCar4."""
    return FITTED_SIGMA_MATRIX


def build_junction_young_angle_sigma_fit_v1() -> CanonicalEquation:
    """Build the junction sigma_ij equation with the MEASURED n600 Young-angle anchor
    (anchor (i) of the hunt §9.2 owed list; the sigma-weighted-length A/B remains OWED)."""
    anchor = EmpiricalAnchor(
        anchor_id="junction_young_sigma_n600_gt_argmax_20260707",
        measurement_utc=_UTC,
        inputs={
            "gt_cache": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz (lstars, ALL 600 pairs)",
            "detector": "2x2 plaquette, exactly-3-distinct-classes; clean = 3 circular label "
                        "transitions on a radius-4 circle (120 samples); subpixel refine SKIPPED "
                        "(stated; +-0.5px corner noise averaged + bootstrapped)",
            "fit": "Young's law sine ratios -> weighted log-LS across class-triples; gauge "
                   "geometric-mean(sigma)=1 (all-ones = the null); bootstrap 200, seed 0",
        },
        predicted_output={
            "herring_null": "all angles 120 deg; all sigma_ij = 1 (today's scalar length weight)",
        },
        empirical_output={
            "n_clean_junctions": N_CLEAN_JUNCTIONS,
            "mean_abs_dev_from_120_deg": MEAN_ABS_DEV_FROM_120_DEG,
            "frac_junctions_arc_ge_180": FRAC_JUNCTIONS_ARC_GE_180,
            "sigma_road_lane": SIGMA_ROAD_LANE,
            "sigma_road_lane_ci95": [0.317, 0.441],
            "sigma_road_mycar": SIGMA_ROAD_MYCAR,
            "sigma_lane_mycar": SIGMA_LANE_MYCAR,
            "sigma_lane_undrivable": SIGMA_LANE_UNDRIVABLE,
            "pairs_excluding_all_ones": "4 of 7 fitted",
            "bulk_triple_near_herring": "Road-Undrivable-Movable n=1815: [120.0, 123.7, 116.3] deg",
            "verdict": ("all-ones is right for BULK junctions and measurably wrong at every "
                        "Lane-involving junction; uniform length weight over-penalizes lane "
                        "boundary length ~2.7x (sigma_RL=0.377)"),
            "consistency_caveat": ("3 usable triples -> the log-LS system is exactly solvable "
                                   "(residuals 0 by construction); cross-triple consistency has "
                                   "no leverage yet"),
        },
        residual=0.0,
        source_artifact=_SIGMA_JSON,
        measurement_method=("tools/fit_junction_sigma_youngs_law.py (vectorized numpy; 1.3 s; "
                            "peak RSS 2.8 GiB; deterministic seed 0)"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SIGMA_JSON,
            reactivation_criteria=("OWED anchor (ii) per hunt §9.2: the A/B — sigma_ij-weighted "
                                   "length term vs uniform, n600 through R, junction-local d_seg "
                                   "attribution; requires the --length-sigma-matrix "
                                   "TrainerSupportGap to close (DSL Regularizer factory first)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=SIGMA_EQUATION_ID,
        name=("Junction Young-angle sigma_ij fit: the frozen scorer's triple-junction angles "
              "invert to a pairwise tension matrix; sigma[Road-Lane]=0.377 excludes all-ones"),
        one_line_summary=(
            "Scorer junctions are near-Herring for bulk classes, far from it wherever Lane is "
            "involved; the all-ones length weight over-penalizes lane boundaries ~2.7x."
        ),
        latex_form=(
            r"\frac{\sigma_{jk}}{\sin\theta_i}=\frac{\sigma_{ik}}{\sin\theta_j}"
            r"=\frac{\sigma_{ij}}{\sin\theta_k},\qquad \hat\sigma_{RL}=0.377\;[0.317,0.441]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.junction_young_sigma_and_powerlaw_exit_20260707"
            ":fitted_sigma_matrix"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": ("LANDED 2026-07-07: --length-sigma-matrix (trainer flag, default all-ones = "
                      "byte-identical code path; faithful per-interface gather sigma[top1,top2] on "
                      "the decision-margin length term) held by the DSL LengthSigma Lever factory "
                      "(tac.witness_dsl.curriculum_dsl; fitted-20260707 preset = this fit); "
                      "resolver tac.boundary_math.length_sigma; A/B still the OWED anchor"),
            "measurement_axis": ["macOS-CPU advisory"],
            "note": ("sigma fitted on the FROZEN scorer's GT partition (target geometry), gauge "
                     "geometric-mean 1; 3 pairs unobserved (Lane-Movable, Movable-MyCar, "
                     "Undrivable-MyCar: <30 junctions); treatment arm for councils, NOT the "
                     "clean baseline (hunt §7 verdict)"),
        },
        units_in={"theta_i": "degrees_interior_angle"},
        units_out={"sigma_ij": "relative_surface_tension_geomean_1"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            # Herring-null deviation, normalized: measured mean |dev|/120 vs predicted 0
            "herring_null_mean_abs_angle_dev_frac": MEAN_ABS_DEV_FROM_120_DEG / 120.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            # (refined 2026-07-07, sigma lever landed) the consumption path is CLOSED:
            "tac.witness_dsl.curriculum_dsl:LengthSigma",
            "tac.boundary_math.length_sigma",
            "experiments/train_levelset_witness_realized_through_R_mlx.py:--length-sigma-matrix",
        ),
        canonical_producers=("tools/fit_junction_sigma_youngs_law.py",),
        provenance=build_provenance_for_predicted(
            model_id="junction_young_angle_sigma_fit.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def build_weak_kam_powerlaw_tail_exit_v1() -> CanonicalEquation:
    """Build the weak-KAM power-law tail exit equation with the MEASURED long900 + mod32cap
    retro-fit anchor (the hunt §4 'fit owed' — now run)."""
    anchor = EmpiricalAnchor(
        anchor_id="powerlaw_tail_retrofit_long900_mod32cap_20260707",
        measurement_utc=_UTC,
        inputs={
            "long900": "witcap_kd_c1_long900 durable log (19 eval points, ep1-900, total d_seg)",
            "mod32cap": ("live run verdict telemetry (READ-ONLY), stage windows CE ep1-299 (11 "
                         "pts) / tau 300-725 (18 pts) / Muon 726+ (4 pts, low-confidence)"),
            "models": "d(t)=a+b*t^-alpha vs d(t)=a+b*exp(-t/tau_e); profile LS + golden refine; "
                      "AIC (n ln(RSS/n)+2k); bootstrap 200 seed 0; a clamped >= 0",
        },
        predicted_output={
            "hunt_s4_prediction": ("late/binding-class descent power-law (exp detectors fire "
                                   "early); early stages exponential"),
            "preregistered": "alpha_lane < alpha_road (needs per-class trajectories)",
        },
        empirical_output={
            "long900_preferred": "power_law (dAIC exp-pow = +45.37)",
            "long900_alpha": LONG900_ALPHA,
            "long900_alpha_ci95": list(LONG900_ALPHA_CI95),
            "long900_exp_asymptote_broken_by": LONG900_EXP_ASYMPTOTE_BROKEN_BY,
            "long900_exp_asymptote_vs_measured": "exp floor 0.002464 vs measured 0.002017@ep900",
            "mod32cap_ce_dAIC": MOD32CAP_CE_DAIC_EXP_MINUS_POW,
            "mod32cap_tau_dAIC": MOD32CAP_TAU_DAIC_EXP_MINUS_POW,
            "alpha_lane_lt_alpha_road": ("UNRESOLVED_APPARATUS_GAP — no run logs per-class d_seg "
                                         "trajectories (verdict rows are total-only; annulus "
                                         "sidecar has 2 per-class snapshots)"),
            "verdict": ("longest window power-law-preferred + exp floor measurably broken => "
                        "exponential plateau detectors DO leave meat on the bone; early stage "
                        "windows exponential-preferred, matching the theory's regime split"),
        },
        residual=0.0,
        source_artifact=_POWERLAW_JSON,
        measurement_method=("tools/fit_powerlaw_plateau_detector.py (deterministic profile-LS; "
                            "seeded bootstrap; $0 CPU retro-fit on logged trajectories)"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_POWERLAW_JSON,
            reactivation_criteria=("per-class d_seg verdict telemetry (duty-to-measure; "
                                   "score-neutral read-only => defaults ON per the default-off-"
                                   "is-orphaned-signal rule) -> run the pre-registered "
                                   "alpha_lane < alpha_road check; recalibrate on any new "
                                   "long-budget trajectory"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=POWERLAW_EQUATION_ID,
        name=("Weak-KAM power-law tail exit: late witness descent is a+b*t^-alpha (long900 "
              "alpha=0.51, dAIC +45.4 over exponential); exit on extrapolated remaining meat"),
        one_line_summary=(
            "long900 is power-law (exp floor broken by 22%): exponential plateau detectors fire "
            "early; exit on extrapolated remaining meat under a+b*t^-alpha, not window slope."
        ),
        latex_form=(
            r"d_{seg}(t)=a+b\,t^{-\alpha};\quad"
            r"\mathrm{meat}(t_{now},T)=d(t_{now})-d(T);\quad"
            r"\hat\alpha_{long900}=0.51,\ \Delta\mathrm{AIC}=+45.4"
        ),
        python_callable_module_path=(
            "tac.witness_control.powerlaw_exit:powerlaw_meat_exit"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "witcap_kd_c1_coord_inr"],
            "lever": ("exit/plateau detection in the costate shadow controller + a future "
                      "ExitEvent(criterion='powerlaw_meat') gap-kind DSL criterion"),
            "measurement_axis": ["macOS-CPU advisory"],
            "note": ("alpha depends on the window origin (t0 recorded per fit); Muon window n=4 "
                     "is low-confidence; per-class alpha (the binding-class claim) is the OWED "
                     "half — total-d_seg fits only until per-class telemetry lands"),
        },
        units_in={"t": "epochs_since_window_start", "horizon": "epochs", "meat_floor": "d_seg"},
        units_out={"remaining_meat": "d_seg", "alpha": "dimensionless_exponent"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            # the theory predicts power-law preference on the longest window: dAIC sign check
            "long900_powerlaw_preferred": 0.0 if LONG900_DAIC_EXP_MINUS_POW > 0 else 1.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_control.producer_bridge",
                             "tac.witness_control.powerlaw_exit"),
        canonical_producers=("tools/fit_powerlaw_plateau_detector.py",),
        provenance=build_provenance_for_predicted(
            model_id="weak_kam_powerlaw_tail_exit.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def populate_solver_pack_equations(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> list[CanonicalEquation]:
    """Idempotent APPEND-ONLY registration of both solver-pack equations into the canonical
    registry (mirrors ``populate_boundary_distance_weight_calibration_equation``; latest-row-wins
    query semantics). The EQUATIONS leg of the §15 solver pack (memo
    ``solver_pack_junction_sigma_powerlaw_20260707.md``)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eqs = [build_junction_young_angle_sigma_fit_v1(),
           build_weak_kam_powerlaw_tail_exit_v1()]
    for eq in eqs:
        register_canonical_equation(
            eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
            notes="solver_pack_junction_sigma_powerlaw_20260707 (equations leg; hunt §7/§4 "
                  "bridges measured)",
        )
    return eqs


__all__ = [
    "FITTED_SIGMA_MATRIX",
    "FRAC_JUNCTIONS_ARC_GE_180",
    "LONG900_ALPHA",
    "LONG900_ALPHA_CI95",
    "LONG900_DAIC_EXP_MINUS_POW",
    "MEAN_ABS_DEV_FROM_120_DEG",
    "N_CLEAN_JUNCTIONS",
    "POWERLAW_EQUATION_ID",
    "SIGMA_EQUATION_ID",
    "SIGMA_ROAD_LANE",
    "build_junction_young_angle_sigma_fit_v1",
    "build_weak_kam_powerlaw_tail_exit_v1",
    "fitted_sigma_matrix",
    "populate_solver_pack_equations",
]
