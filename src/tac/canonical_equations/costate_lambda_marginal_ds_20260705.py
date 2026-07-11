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
    anchor_waterlines = EmpiricalAnchor(
        # (b) SCORE-ECONOMICS: cell<->byte exchange + distance-to-target waterlines — advisory-fold
        # 2026-07-10 (ADVISORY_evaluator_video_geometry / ADVISORY_vehicle_line_synthesis). This
        # equation already owns λ_bytes = 25/37_545_489 and λ_pose = 5/sqrt(10·d_pose); the seg-CELL
        # price 8.4771050347e-7 (=100/117_964_800) is held by resize_exploit_flip_fix_frontier_v1
        # law-2. This anchor pins the two DERIVED exchange rates + the single-axis waterlines from the
        # local pointer 0.19110 — the costate integrated to the sub-0.19 / sub-0.15 targets.
        anchor_id="score_economics_cell_byte_waterlines_20260710",
        measurement_utc="2026-07-10T00:00:00Z",
        inputs={
            "score_law": "S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37_545_489",
            "n600_seg_surface": "600·384·512 = 117,964,800 scored cells",
            "rate_denom": "37,545,489 bytes (single-video videos/0.mkv; evaluate.py:55-69)",
            "pointer": "0.19109982419209975 [contest-CPU] (canonical_frontier_pointer)",
            "advisory": ".omx/research/ADVISORY_evaluator_video_geometry_20260710.md + "
                        "ADVISORY_vehicle_line_synthesis_20260710.md",
            "custody_sha256": "evaluate.py 7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd4"
                              "0baa4b (frozen checkout git 991b317c41fe3aac657e0f0cb88fd831b2e4185a)",
        },
        predicted_output={
            "question": "exact single-axis exchange rates + distance-to-sub-0.19 / sub-0.15 waterlines",
        },
        empirical_output={
            "seg_cell_price_S": 8.4771050347e-7,       # 100/117_964_800 (held by resize_exploit law-2)
            "archive_byte_price_S": 6.6585895312e-7,   # 25/37_545_489 (= λ_bytes above)
            "cell_buys_bytes": 1.273108,               # 1 corrected Seg cell ≡ 1.273108 bytes (no Pose)
            "byte_needs_cells": 0.785479,              # 1 added byte must save > 0.785479 cells (no Pose)
            "waterline_sub019_bytes": 1651.74,         # from pointer 0.19110, one-axis-held-fixed
            "waterline_sub019_cells": 1297.41,
            "waterline_sub015_bytes": 61724.52,
            "waterline_sub015_cells": 48483.33,
            "pose_caveat": (
                "NO fixed pose-per-byte: the sqrt(10·d_pose) marginal 5/sqrt(10·d_pose) is operating-"
                "point dependent; use the exact sqrt DIFFERENCE for finite moves, never the tangent. "
                "Advisory arithmetic corrections: sqrt(10·0.018)=0.424264 (NOT ~0.02); 0.022/0.00161="
                "13.6646 and the 0.022 row is n24, not an n600 matched ratio."
            ),
            "verdict_scope": (
                "DERIVED (exact score-law arithmetic). The waterlines are one-axis-held-fixed "
                "EQUIVALENCES, NOT predictions — recompute Pose + component interactions at the "
                "realized byte-closed candidate; means != ends, pointer 0.19110 UNMOVED."
            ),
        },
        residual=0.0,
        source_artifact=".omx/research/ADVISORY_evaluator_video_geometry_20260710.md",
        measurement_method="exact score-law arithmetic on the frozen evaluator constants + local pointer",
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/ADVISORY_evaluator_video_geometry_20260710.md",
            reactivation_criteria=(
                "recompute the waterlines when the canonical pointer moves; recompute the Pose "
                "contribution at the realized candidate's measured d_pose (the sqrt term is not linear)"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    anchor_organ_seal = EmpiricalAnchor(
        # (c) AMORTIZABILITY of this costate λ by a LEARNED predictor — the #426 costate-organ
        # adversarial-review SEAL (2026-07-11). This equation gives λ EXACTLY (partials above);
        # #426 amortizes it (predict λ from state without recomputing). This anchor pins HOW WELL
        # it amortizes + WHICH predictor is faithful, so the look-ahead-bias correction is never
        # re-derived. Fresh-Fable reviewer OVERTURNED the author's n=5 headline.
        anchor_id="costate_organ_426_seal_walkforward_20260711",
        measurement_utc="2026-07-11T12:00:00Z",
        inputs={
            "organ": "tac.witness_control.{lambda_net,costate_panel,control_alphabet,"
                     "prototype_router} + tac.witness_dsl.costate_agent_dsl",
            "trajectory": "#205 (levelset_v752_baseline_20260710T185913Z) advisory verdicts, "
                          "grew 5->8 same-stage intervals during review (free out-of-sample)",
            "tool": "tools/lambda_net_backtest.py --archs (chunked)",
            "gate": "tri-gate = LOO ∧ WALK-FORWARD (deployment-faithful) ∧ binding-floor 0.8",
        },
        predicted_output={
            "memo_era_claim_n5": "ridge-SOLVE fMAE 0.003379 vs persistence 0.005632 = 4.2x/"
                                 "per-class — the ORIGINAL (Einstein) headline",
        },
        empirical_output={
            "look_ahead_artifact": "the n=5 LOO 4.2x was a LOOK-AHEAD artifact; under "
                                   "walk-forward ridge LOSES to persistence (0.003535 vs 0.002896)",
            "winner_family": "prototype/Bregman/BSF: E_prototype WF 0.002513, E_bregman 0.002501, "
                             "F_bsf 0.002513 (tri-gate PASS, ~14% better than persistence at n=8) "
                             "— the Rudin/Nielsen/Goodfire arms are the MEASURED winners",
            "interpretability_tax_gone": "the +8% interpretable-head cost is ELIMINATED — the "
                                         "interpretable prototype head is ALSO the best predictor",
            "analytic_partials_exact": "λ_seg partials exact-linear (lstsq residual 5.1e-7; fitted "
                                       "class weights recover canonical GT areas) — DERIVED",
            "plateau_regime_loss": "amortized λ LOSES to persistence in PLATEAU regimes (0.0007 vs "
                                   "0.0017-0.0030); the meta-λ self-monitor flags it live (trust 0.72)",
            "self_activation": "rehabilitated-NOT-decisive (flips best at n=8 but 5/8 folds, p≈0.36)",
            "shipped_mode": "SINGLE_BEST=prototype (parsimony + Rudin); re-arbitrated per accrual",
            "verdict_scope": "INSTANCE — n=1 trajectory / 1 vehicle / 8 intervals, fold variance ~30x; "
                             "GEPA/Bregman adoption margins ~0.5% WITHIN NOISE = provisional-until-"
                             "accrual (the triality-native compounding ledger is the cure). MEANS "
                             "(the organ never ships); pointer 0.19108282 UNMOVED.",
        },
        residual=0.000639,  # ridge WF deficit vs persistence (0.003535−0.002896) = the look-ahead-bias magnitude
        source_artifact=".omx/research/costate_organ_capabilities_limits_envelope_20260711.md",
        measurement_method="walk-forward tri-gate backtest over the #205 trajectory (n=8 same-stage "
                           "intervals); LOO look-ahead bias measured + extincted",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/costate_organ_capabilities_limits_envelope_20260711.md",
            reactivation_criteria=("re-arbitrate the winning architecture as the trajectory ledger "
                                   "accrues (>=3 trajectory records; the VAPO crossover is DETECTED "
                                   "not scheduled); the ~0.5% Bregman/GEPA margins resolve only "
                                   "out-of-noise with more out-of-sample intervals"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_scorer_arms_430 = EmpiricalAnchor(
        # (d) SCORER-MODEL ARMS + #430 coherent schedule (P0 build 2026-07-11). The λ
        # law gains trajectory-INDEPENDENT scorer-side instruments (the n=1-fragility
        # cure attempts) + the first MEASURED schedule-level consumption of λ.
        anchor_id="scorer_model_arms_430_backtest_20260711",
        measurement_utc="2026-07-11T13:50:00Z",
        inputs={
            "arms": "tac.witness_control.scorer_model_arms (H/I/J/K/L/M) + "
                    "schedule_backtest (#430)",
            "trajectory": "#205 levelset_v752_baseline (10 verdicts / 9 intervals at "
                          "measurement — grew during the build, free out-of-sample)",
            "caches": "gt_n96.npz margins/lstars + 192 REAL comma10k masks "
                      "(experiments/results/comma10k_regime_prior) + σ_cc′ #382",
            "tool": "tools/schedule_bundle_backtest.py (durable JSON artifact)",
        },
        predicted_output={
            "survey_order": "#428 survey: metric relaxation (#1) before learned "
                            "surrogate; scorer priors should cure the n=1 fragility",
            "430": "state-gated coordinated bundles beat independently-clocked floors "
                   "(2607.08716 selective-intervention, MEASURED-elsewhere +8.3pp)",
        },
        empirical_output={
            "smoothed_argmax_exact": "per-class ∂(smoothed d_seg)/∂δ_c through the real "
                                     "frozen SegNet's cached margins: finite-diff rel "
                                     "gap 5.6e-7 (ZERO model error; ε=τ median margin)",
            "ball_agreement": "margin-surrogate flip set vs advection-ball label-change "
                              "set: IoU 0.732, prec/rec 0.844 — faithful (2306.04431)",
            "comma10k_prior": "rarity reweight Lane 3.51x / Movable 1.22x from SegNet's "
                              "real training distribution — converges with the measured "
                              "flip crux WITHOUT any trajectory or scorer forward",
            "forecast_gate_honest_negative": "ALL scorer-prior formulations are forecast-"
                                             "NEUTRAL-or-worse: φ-rescale INERT (≡ ridge, "
                                             "the solve refits it away); shrink-to-prior "
                                             "WORSE at early folds (κ-scale variance at "
                                             "n=2). verdict_scope: formulation ×2. The "
                                             "arms' measured wins: binding AUROC 1.0 + "
                                             "the audited sensors + per-class λ for #430",
            "430_replay": "organ's state-gated cascade beats the hand-scheduled #205 "
                          "curriculum on all 3 replay models: ∫d_seg·dep 6.283 vs 8.649 "
                          "(−27%) on the walk-forward-winning prototype model; self-"
                          "replay MAE 0.0054 / final gap 1.6e-4; selective ≈ always-on "
                          "in-model (Δ0.4%: transient-only prefix, a gate always active "
                          "— the gate question is NOT in-model-resolvable; live A/B owed)",
            "plateau_onset": "at 9 intervals the WHOLE model family loses walk-forward "
                             "to persistence (E_bregman 0.002839 vs 0.002792) — the "
                             "envelope's plateau-regime prediction confirmed out-of-"
                             "sample; meta-λ prefer_persistence is the correct posture",
            "verdict_scope": "INSTANCE (1 trajectory) for the replay; the smoothed-"
                             "argmax exactness + comma10k shares are trajectory-free. "
                             "MEANS; pointer 0.19108282 UNMOVED.",
        },
        residual=0.000047,  # E_bregman WF deficit vs persistence at 9 intervals (plateau onset)
        source_artifact=".omx/research/scorer_model_arms_430_schedule_20260711.md",
        measurement_method="tournament backtest (LOO∧WF∧binding) + early-fold walk-forward "
                           "+ ball-agreement audit + 4-policy model-based schedule replay",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/scorer_model_arms_430_schedule_20260711.md",
            reactivation_criteria=("re-arbitrate per organ-ledger accrual; the learned "
                                   "surrogate (survey #2) unblocks only by beating the "
                                   "zero-model-error smoothed-argmax baseline; the #430 "
                                   "LIVE A/B is the ticket's standing gates_owed"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
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
        empirical_anchors=(anchor, anchor_waterlines, anchor_organ_seal,
                           anchor_scorer_arms_430),
        predicted_vs_empirical_residual={
            "ep350_horizon_25ep_in_band_central_gap": 0.0706,
            "ep450_horizon_25ep_below_band_gap": 0.0056,
            "score_economics_waterlines_derived": 0.0,
            "organ_426_amortization_ridge_walkforward_deficit": 0.000639,
            "scorer_arms_430_plateau_onset_wf_deficit": 0.000047,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.shadow_controller",
            "tools/costate_digest.py",  # surfaces the score-economics waterlines at SessionStart
        ),
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
