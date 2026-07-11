# SPDX-License-Identifier: MIT
"""Canonical equations: the witness-NATIVE curriculum laws derived by the 2026-07-05 T3
curriculum-derivation symposium (task #302;
``.omx/research/council_grand_symposium_curriculum_derivation_20260705.md``).

Four laws, each converting a PR95/wall-clock-inherited schedule element into a derived form:

1. ``curriculum_handoff_critical_nucleus_v1`` — the CE->tau boundary is a per-class READINESS
   TRIGGER, not an epoch:  fire <=> [forall scored class c: nucleus(c)] AND [CE plateau at
   |rel slope| <= 1e-4 over a 25-ep window, min-stage >= 250].  Nucleus = the measured
   resolution-portable survival knee pi_1 = w/sigma ~ 5 (Tao pi-ledger) realized as
   part_frac > 0 AND within-flip below threshold. Anchors: the C1 plateau-eps miscalibration
   (eps 1e-3 fires ep151 MID-DESCENT on #205's own CE trace = 15% CE-floor loss) + the #205
   tau-onset erosion (d_seg 0.004752@300 -> 0.006568@400 with a sub-nucleus lane).

2. ``ema_window_pi_group_v1`` — the EMA decay is a WINDOW, N_ema = 1/(1-rho) steps; the
   dimensionless group pi_ema = N_ema / N_stage is what transfers across vehicles, NOT rho.
   rho=0.997 (Quantizr-inherited) = 333 steps = 4.4 ep at 75 steps/ep -> averages only ~1.6%
   of a 274-ep finisher (too SHORT for Polyak averaging) and lags 78x in early fast-descent
   (too LONG for tracking). Fix is BUILT: ``--ema-decay-finisher`` (SWA-style; A/B owed).

3. ``muon_switch_conditioning_criterion_v1`` — engage Muon when the partition is FORMED
   (no class below nucleus — Muon measurably cannot nucleate a zero-mass class) AND the
   tau-stage verdict has plateaued (remaining error conditioning-limited, kappa~19 basin);
   with warm-start momentum + LR anneal (GAP1+2). PR95's "stage 8" placement is the right
   ORDER for the wrong REASON (their clock).

4. ``rewarmup_beta2_memory_window_v1`` — a stage-transition LR rewarmup window must cover
   AdamW's second-moment re-accumulation memory: rewarmup_steps >= 1/(1-beta2).  Run-2's
   20 ep x 75 steps/ep = 1500 >= 1000 satisfies it; the earlier 8-ep value (600) did NOT.
   Shape (cosine) + floor (0.1) remain underived — honestly open.

means != ends: advisory/derivation laws, NOT scores; pointer 0.19110 UNMOVED.
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

_UTC = "2026-07-05T06:30:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"
_MEMO = ".omx/research/council_grand_symposium_curriculum_derivation_20260705.md"
_MODULE = "tac.canonical_equations.curriculum_derivation_laws_20260705"

# --- derived constants (each traceable to a measured anchor; see the memo audit table) ------------
HANDOFF_PLATEAU_REL_EPS = 1e-4      # C1 recalibration (1e-3 fires ep151 mid-descent on #205 CE)
HANDOFF_PLATEAU_WINDOW_EP = 25      # C1 recalibration window
HANDOFF_MIN_STAGE_EP = 250          # C1 recalibration min-stage floor
NUCLEUS_PI1_KNEE = 5.0              # pi_1 = w/sigma survival knee (measured sigma0.8/94% vs 1.5/49%)
EMA_STEPS_PER_EPOCH_RUN2 = 75       # n600 / accum 8 (verified per-step ema.update, trainer:4759)
EMA_LAG_FACTOR_EARLY_MEASURED = 78.0  # cert C3: measured early-stage shadow lag
MUON_DSEG_GAIN_VS_ADAMW = -0.32     # fork A/B 20260622: -32% d_seg, gap widening
MUON_COLD_SWITCH_TRANSIENT = 0.08   # cert B5: +8% d_seg at the cold ep726 switch, ~75 ep recovery

# --- Law 5 (phase-tail hand-off, 2026-07-10 flicker reframe) -------------------------------------
LABEL_FLOOR_DSEG = 0.005318         # GT stride-2 spike rate = temporal-majority persistence floor
LABEL_FLOOR_BAND = (0.00496, 0.00700)  # converged-residual band (witness popout 0.00496 -> +margin)
PHASE_NEED_BAND = (0.00077, 0.00118)   # sub-0.15 d_seg need (openpilot lane-band) = 4.5-7x below floor
PHASE_EXISTENCE_DSEG = 0.00086     # real-frame content through R (FEED-ma) — below the floor
BLINK_BACK_FRACTION = 0.418        # single-frame revert share => the PREDICTABLE-phase gradient share
PHASE_WEIGHT_FRACTION_OF_SUBPIX = 0.4  # w_p = 0.4*w_subpix (derived from the blink-back fraction)


def handoff_ready(rel_slope: float, epochs_in_stage: int,
                  per_class_nucleus_ok: list[bool] | tuple[bool, ...],
                  *, eps: float = HANDOFF_PLATEAU_REL_EPS,
                  min_stage_epochs: int = HANDOFF_MIN_STAGE_EP) -> bool:
    """Law 1 callable: CE->tau fires iff plateaued AND every scored class is above nucleus."""
    plateaued = abs(float(rel_slope)) <= float(eps)
    seasoned = int(epochs_in_stage) >= int(min_stage_epochs)
    nucleus = all(bool(b) for b in per_class_nucleus_ok)
    return bool(plateaued and seasoned and nucleus)


def ema_window_steps(rho: float) -> float:
    """Law 2 callable: the EMA's effective averaging window N_ema = 1/(1-rho) steps."""
    if not 0.0 < float(rho) < 1.0:
        raise ValueError(f"rho={rho} must be in (0, 1)")
    return 1.0 / (1.0 - float(rho))


def pi_ema(rho: float, stage_steps: int) -> float:
    """Law 2 callable: the dimensionless group pi_ema = N_ema / N_stage (what should transfer)."""
    if int(stage_steps) <= 0:
        raise ValueError(f"stage_steps={stage_steps} must be > 0")
    return ema_window_steps(rho) / float(stage_steps)


def muon_switch_ready(tau_stage_plateaued: bool,
                      per_class_nucleus_ok: list[bool] | tuple[bool, ...]) -> bool:
    """Law 3 callable: Muon engages iff the partition is formed AND tau has plateaued."""
    return bool(tau_stage_plateaued) and all(bool(b) for b in per_class_nucleus_ok)


def min_rewarmup_epochs(beta2: float, steps_per_epoch: int) -> int:
    """Law 4 callable: rewarmup window (epochs) >= AdamW second-moment memory 1/(1-beta2) steps."""
    if not 0.0 < float(beta2) < 1.0:
        raise ValueError(f"beta2={beta2} must be in (0, 1)")
    if int(steps_per_epoch) <= 0:
        raise ValueError(f"steps_per_epoch={steps_per_epoch} must be > 0")
    return math.ceil((1.0 / (1.0 - float(beta2))) / float(steps_per_epoch))


def phase_tail_ready(rel_slope: float, d_seg_latest: float, stage_is_label_smooth: bool,
                     *, floor_band: tuple[float, float] = LABEL_FLOOR_BAND,
                     flat_eps: float = HANDOFF_PLATEAU_REL_EPS) -> bool:
    """Law 5 callable: the label->phase hand-off fires iff the LABEL-SMOOTH flow has settled
    at the temporal-majority persistence floor (analogue of Law 1's CE->tau nucleus trigger,
    on the TEMPORAL/finest-persistence axis). Fire <=> label-smooth stage AND d_seg within the
    converged-residual band AND flat (|rel slope| <= plateau eps). Below the floor is the
    APPEARANCE-PHASE regime (T1 + #360), the ONLY descent to the sub-0.15 need band."""
    lo, hi = float(floor_band[0]), float(floor_band[1])
    at_floor = lo <= float(d_seg_latest) <= hi
    flat = abs(float(rel_slope)) <= float(flat_eps)
    return bool(stage_is_label_smooth) and at_floor and flat


def _sidecar_prov(reactivation: str):
    return build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=reactivation,
        measurement_axis=_ADVISORY,
        hardware_substrate="apple_m5_max_cpu_mlx",
    )


def _pred_prov(model_id: str):
    return build_provenance_for_predicted(
        model_id=model_id,
        inputs_sha256="0" * 64,
        measurement_axis=_PREDICTED,
        hardware_substrate="apple_m5_max_cpu_mlx",
    )


def build_curriculum_handoff_critical_nucleus_v1() -> CanonicalEquation:
    """Law 1: the CE->tau boundary is a per-class readiness trigger, not an epoch."""
    anchor = EmpiricalAnchor(
        anchor_id="handoff_c1_miscalibration_plus_205_tau_creep_20260705",
        measurement_utc=_UTC,
        inputs={
            "ce_trace": "#205 run.log CE-phase rel slopes (25-ep averaged), mined 2026-07-04",
            "erosion_trace": "#205 run.log tau-phase verdicts (no-seed arm)",
            "survival_knee": "MCF-proxy smoothing survival, lane w=6px (facet-4 memo)",
        },
        predicted_output={
            "naive_default": "plateau eps 1e-3 == 'stage done' (the event-trigger default)",
        },
        empirical_output={
            "eps_1e3_fires": "ep151 (rel slope -8.22e-4) while d_seg 0.005473@150 -> 0.004752@300 "
                             "still descending = 15% CE-floor loss if fired",
            "eps_1e4_separates": "rel slope -9.38e-5 only at ep250->275 (true knee)",
            "sub_nucleus_erosion": "d_seg 0.004752@300 -> 0.006568@400 (lane part_frac 0: MCF "
                                   "erodes what CE never formed; Muon cannot nucleate)",
            "knee": "sigma0.8 -> 94% lane survival vs sigma1.5 -> 49% => pi_1 = w/sigma ~ 5",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="$0 log mining of #205 run.log + facet-4 survival probe (no new runs)",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=_sidecar_prov(
            "re-derive the plateau constants if the vehicle's CE timescale shifts (they are "
            "trajectory-calibrated); the nucleus guard itself is a BUILD (run-3 spec item 1b)"),
    )
    # (#302 build, LANDED 2026-07-05) the per-class nucleus guard + readiness telemetry that
    # IMPLEMENT the law's forall-class clause. Additive anchor: the law was DERIVED (anchor above);
    # this records that its measured trigger is now BUILT (default-OFF, byte-identical), so the
    # NEXT run passively yields the per-class validation data the law needs to be confirmed.
    build_anchor = EmpiricalAnchor(
        anchor_id="handoff_nucleus_guard_build_landed_20260705",
        measurement_utc="2026-07-05T20:30:00Z",
        inputs={
            "trainer": "experiments/train_levelset_witness_realized_through_R_mlx.py "
                       "(_evt_nucleus_counts/_evt_nucleus_stats/_evt_nucleus_satisfied/"
                       "_evt_readiness_row/_evt_reanchor_epoch + _verdict_dseg_dpose_nucleus_chunked)",
            "flags": "--curriculum-nucleus-guard / --curriculum-nucleus-within-flip / "
                     "--handoff-readiness-telemetry / --curriculum-reanchor-levers; "
                     "--curriculum-plateau-rel-eps default recalibrated 1e-3->1e-4",
            "gauge": "tac.witness_dsl.gauge.CurriculumGauge.HANDOFF_NUCLEUS",
        },
        predicted_output={
            "forall_class_clause": "MEASURED at verdict cadence: fire CE->tau only when every scored "
                                   "class is BORN (part_frac>0) AND FORMED (within-flip<=thresh)",
        },
        empirical_output={
            "byte_identity_off": "all flags default-OFF => nucleus_ready True, verdict path unchanged "
                                 "(_verdict_dseg_dpose_chunked), no readiness row => the #205 path is "
                                 "byte-identical (existing event/closed-loop tests 39/39 green)",
            "tests": "21 pure-function tests (guard/trigger/reanchor/counts/readiness) green",
            "cap_never_hangs": "the hardcoded cap still fires unconditionally even if nucleus never "
                               "satisfies (verified)",
            "readiness_telemetry": "handoff_readiness row emitted per verdict (observability-first: "
                                   "the next run yields per-class validation data even trigger-OFF)",
        },
        residual=0.0,
        source_artifact=".omx/research/event_handoff_perclass_lambda_20260705.md",
        measurement_method="build landing (default-OFF byte-identity + 21 unit tests); no new "
                           "training run — the per-class nucleus DATA is a run-3 A/B (net-S gated)",
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        provenance=_sidecar_prov(
            "confirm the within-flip threshold empirically from the NEXT run's handoff_readiness "
            "telemetry (the theoretical knee pi_1~5 is the operational proxy target)"),
    )
    return CanonicalEquation(
        equation_id="curriculum_handoff_critical_nucleus_v1",
        name=("CE->tau handoff = readiness trigger: fire <=> plateau(|rel slope|<=1e-4, 25-ep "
              "window, min-stage 250) AND forall class: nucleus (pi_1=w/sigma>~5; part_frac>0)"),
        one_line_summary=(
            "CE->tau is a per-class readiness TRIGGER, not an epoch: MCF erodes sub-nucleus "
            "classes (measured #205 creep) so CE must form every class + plateau first."
        ),
        latex_form=(
            r"\mathrm{fire} \iff |\dot L_{CE}/L_{CE}| \le 10^{-4} \wedge t \ge 250 \wedge "
            r"\forall c:\ \pi_1^{(c)} = w_c/\sigma \gtrsim 5"
        ),
        python_callable_module_path=f"{_MODULE}:handoff_ready",
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": "--curriculum-event-triggered (recalibrated) + per-class nucleus guard (BUILD)",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "plateau constants are trajectory-calibrated on #205's CE; re-mine on a "
                    "materially different vehicle; the l7 converge-fire guard (review C2) is a "
                    "PRECONDITION for enabling event mode at all",
        },
        units_in={"rel_slope": "per_epoch_fraction", "epochs_in_stage": "epochs",
                  "per_class_nucleus_ok": "bool_per_scored_class"},
        units_out={"fire": "bool"},
        empirical_anchors=(anchor, build_anchor),
        predicted_vs_empirical_residual={
            # naive eps 1e-3 vs measured knee: fires 124+ epochs early on #205's trace
            "naive_eps_early_fire_epochs": 124.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=_pred_prov("curriculum_handoff_critical_nucleus.v1"),
    )


def build_ema_window_pi_group_v1() -> CanonicalEquation:
    """Law 2: EMA decay is a window; pi_ema = N_ema/N_stage is the transferable group."""
    anchor = EmpiricalAnchor(
        anchor_id="ema_per_step_cadence_and_78x_lag_20260705",
        measurement_utc=_UTC,
        inputs={
            "cadence": "ema.update(model) per optimizer step, VERIFIED trainer:4759 "
                       "(75 steps/ep at n600/accum-8)",
            "house_value": "rho=0.997 (Quantizr non-negotiable; PR95 ran 29,650 ep)",
        },
        predicted_output={
            "window_steps_at_0997": 333.3,
            "window_epochs_at_0997_run2": 333.3 / EMA_STEPS_PER_EPOCH_RUN2,
            "finisher_coverage_run2": "333 / (274 ep x 75) ~ 1.6% of the Muon stage",
        },
        empirical_output={
            "early_lag": "up to 78x EMA-shadow lag in early fast-descent (cert C3, the "
                         "EMA-shadow-lag artifact) — the tracking half of the pi violation",
            "fix_built": "--ema-decay-finisher EXISTS (theta* TIER-2 MUST-3, default None = "
                         "bit-identical); finisher A/B value 0.9995 derived from window ~ "
                         "0.1-0.3 x finisher steps; UNFIRED",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="source inspection (cadence) + cert C3 measured lag; window arithmetic",
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        provenance=_sidecar_prov(
            "byte-closed A/B of --ema-decay-finisher 0.9995 vs None on a run-3-class arm; the "
            "0.997 house default is UNCHANGED until that A/B lands"),
    )
    return CanonicalEquation(
        equation_id="ema_window_pi_group_v1",
        name=("EMA decay as a window: N_ema = 1/(1-rho) steps; the transferable invariant is "
              "pi_ema = N_ema/N_stage (0.997 = 333 steps = 1.6% of run-2's finisher: too short "
              "to Polyak-average, 78x lag when tracking early)"),
        one_line_summary=(
            "rho does not transfer; pi_ema = window/stage does. Early wants tracking, finisher "
            "wants Polyak averaging (~0.1-0.3 x stage) — built as --ema-decay-finisher."
        ),
        latex_form=(
            r"N_{ema} = \frac{1}{1-\rho},\quad \pi_{ema} = \frac{N_{ema}}{N_{stage}};\quad "
            r"\rho_{fin}: \pi_{ema} \in [0.1, 0.3]"
        ),
        python_callable_module_path=f"{_MODULE}:pi_ema",
        domain_of_validity={
            "vehicle": ["any per-step-EMA trainer; constants cited for the levelset witness"],
            "lever": "--ema-decay-finisher (BUILT, default None)",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "the deployed-checkpoint authority (EMA shadow, house 0.997) is UNCHANGED "
                    "until a byte-closed A/B; this law licenses the A/B, not a default flip",
        },
        units_in={"rho": "dimensionless_decay", "stage_steps": "optimizer_steps"},
        units_out={"pi_ema": "dimensionless_fraction"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"finisher_pi_ema_at_0997": 333.3 / (274.0 * 75.0)},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=_pred_prov("ema_window_pi_group.v1"),
    )


def build_muon_switch_conditioning_criterion_v1() -> CanonicalEquation:
    """Law 3: Muon engages on formation + plateau, not on PR95's clock."""
    anchor = EmpiricalAnchor(
        anchor_id="muon_fork_gain_cold_transient_and_nonucleation_20260705",
        measurement_utc=_UTC,
        inputs={
            "fork_ab": "muon_vs_adamw_from_stage4_convergence_arm_20260622 (identical fork)",
            "cold_switch": "cert B5: FIRE-arm ep725 0.004316 -> ep750 0.004674 (+8%), ~75 ep "
                           "recovery; #205 spike +0.000357",
            "no_nucleation": "#205: Muon@726 sharpened but never nucleated the zero-mass lane "
                             "(facet-4 sec 2.1)",
            "placement_provenance": "cert A7 cites 'PR95 stage-8 placement' verbatim (inherited)",
        },
        predicted_output={
            "criterion": "engage <=> tau-stage verdict plateaued AND no scored class below "
                         "nucleus; kappa~19 boundary basin is Muon's payoff regime",
        },
        empirical_output={
            "dseg_gain_vs_adamw": MUON_DSEG_GAIN_VS_ADAMW,
            "cold_switch_transient": MUON_COLD_SWITCH_TRANSIENT,
            "lr_ratio_at_engage": "muon-lr 2e-3 = 7.8x the frozen base-LR at ep726 (cert A8)",
            "gaps_fixed": "--muon-warm-start-momentum + --muon-lr-final-frac 0.1 (deep-dive "
                          "GAP1+2; ACTIVE in run-2)",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="$0 mining of the fork A/B + cert B5/A7/A8 + facet-4 nucleation trace",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=_sidecar_prov(
            "trigger-form engage (Muon on tau-plateau + nucleus-complete, fixed epoch as CAP) is "
            "a BUILD (Muon is verified NOT event-fireable today); until built, the fixed epoch "
            "stays with warm-start + LR-anneal active"),
    )
    return CanonicalEquation(
        equation_id="muon_switch_conditioning_criterion_v1",
        name=("Muon engage criterion: switch when the partition is FORMED (no class below "
              "nucleus — Muon cannot nucleate) AND tau-stage verdict plateaued (error is "
              "conditioning-limited, kappa~19); warm-start momentum + anneal LR to 0.1"),
        one_line_summary=(
            "PR95 stage-8 placement = right ORDER, wrong CLOCK; derived engage is a trigger: "
            "partition formed + tau plateau, never before nucleation."
        ),
        latex_form=(
            r"t_{Muon} = \inf\{t: \mathrm{plateau}_\tau(t) \wedge \forall c\ \pi_1^{(c)} "
            r"\gtrsim 5\};\quad \eta_{Muon}(t) \downarrow 0.1\,\eta_0,\ v_0 \leftarrow m_{AdamW}"
        ),
        python_callable_module_path=f"{_MODULE}:muon_switch_ready",
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": "--muon-start-epoch (today a fixed CAP) + --muon-warm-start-momentum + "
                     "--muon-lr-final-frac",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "the -32% gain is measured; its ATTRIBUTION to natural-gradient-ness is "
                    "conjectural (the #284 FALSE-FRIEND catch) — the criterion rests on the "
                    "measured no-nucleation + plateau facts, not the attribution",
        },
        units_in={"tau_stage_plateaued": "bool", "per_class_nucleus_ok": "bool_per_scored_class"},
        units_out={"engage": "bool"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"cold_switch_transient_frac": MUON_COLD_SWITCH_TRANSIENT},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=_pred_prov("muon_switch_conditioning_criterion.v1"),
    )


def build_rewarmup_beta2_memory_window_v1() -> CanonicalEquation:
    """Law 4: rewarmup window must cover AdamW's second-moment memory."""
    anchor = EmpiricalAnchor(
        anchor_id="rewarmup_window_vs_beta2_memory_20260705",
        measurement_utc=_UTC,
        inputs={
            "incident": "margin-engage spike-skip (stale regime -> gnorm explosion -> "
                        "all-batches-skipped stall) + FEED-ft#3 stale-moment tau-jump root cause",
            "run2": "20 ep x 75 steps/ep = 1500 steps; beta2=0.999 => memory 1000 steps",
            "cert_prior": "8 ep = 600 steps < 1000 (the earlier cert value UNDER the bound)",
        },
        predicted_output={"bound": "rewarmup_steps >= 1/(1-beta2)"},
        empirical_output={
            "run2_satisfies": True,
            "cert8_satisfies": False,
            "open": "cosine shape + floor 0.1 underived; critical-slowing exponent "
                    "(dwell ~ 1/gap^2) never matched — the profile is a bound-satisfying guess",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="arithmetic against verified config values + the measured incidents",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=_sidecar_prov(
            "an A/B (8 vs 20 ep at fixed beta2, or beta2 sweep at fixed window) would convert "
            "INFERRED to VERIFIED; PROVISIONAL until then — run-3 keeps 20/0.1/cosine"),
    )
    return CanonicalEquation(
        equation_id="rewarmup_beta2_memory_window_v1",
        name=("Stage-transition rewarmup window >= AdamW second-moment memory: "
              "rewarmup_steps >= 1/(1-beta2) (run-2: 1500 >= 1000 OK; prior 8-ep: 600 FAILED)"),
        one_line_summary=(
            "With reset-moments ON, v re-accumulates over 1/(1-beta2) steps; the LR rewarmup "
            "must span that window or full-LR steps divide by unconverged moments."
        ),
        latex_form=r"T_{rw} \cdot S_{ep} \ge \frac{1}{1-\beta_2}",
        python_callable_module_path=f"{_MODULE}:min_rewarmup_epochs",
        domain_of_validity={
            "vehicle": ["any AdamW trainer with --stage-transition-reset-moments"],
            "lever": "--stage-transition-rewarmup-epochs",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "PROVISIONAL-PENDING-VERIFICATION (Catalog #363 Round-3): the bound is "
                    "literature-derived and consistent with the measured incidents, but no "
                    "isolated A/B has confirmed it is the binding constant",
        },
        units_in={"beta2": "dimensionless_decay", "steps_per_epoch": "optimizer_steps_per_epoch"},
        units_out={"min_rewarmup_epochs": "epochs"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"run2_margin_steps": 1500.0 - 1000.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=_pred_prov("rewarmup_beta2_memory_window.v1"),
    )


def build_label_floor_to_phase_tail_handoff_v1() -> CanonicalEquation:
    """Law 5: the label-smooth flow has a FINEST-PERSISTENCE floor (the temporal-majority
    oracle), and reaching it is the DERIVED transition into the appearance-phase tail — the
    curriculum is a co-equal dimension of the one level-set flow, not a bolt-on stage."""
    anchor = EmpiricalAnchor(
        anchor_id="label_floor_equals_gt_spike_rate_phase_tail_20260710",
        measurement_utc="2026-07-10T00:00:00Z",
        inputs={
            "gt_spike_rate": "n600 cached lstars stride-2 single-frame spike rate (flicker memo Part D)",
            "witness_residual": "#205 popout floor d_seg 0.00496-0.0052 (L67/FEED-03u)",
            "existence_proof": "real-frame content through R d_seg 0.00086 (FEED-ma SIGNAL A)",
            "blink_back": "n600 single-frame revert fraction (flicker memo §2.2)",
        },
        predicted_output={
            "naive_default": "a flat d_seg at ~0.0053 == 'converged, advance/early-stop' "
                             "(the scalar plateau green)",
        },
        empirical_output={
            "floor_is_spike_rate": f"witness converged residual ~0.005 == GT spike rate "
                                   f"{LABEL_FLOOR_DSEG} (temporal-majority oracle) — a Morse-Smale "
                                   f"persistence threshold at one resolution/temporal cell, NOT a "
                                   f"converged minimum",
            "need_below_floor": f"sub-0.15 need band {PHASE_NEED_BAND} is 4.5-7x BELOW the floor; "
                                f"no smooth-label witness reaches it (eq independent_flicker_"
                                f"jitter_dseg_floor_smooth_optimal_v1)",
            "pierceable_by_phase": f"appearance-phase channel reaches d_seg {PHASE_EXISTENCE_DSEG} "
                                   f"< floor (FEED-ma) — the phase is deterministic, per-pair-"
                                   f"storable, low-dim (advected by ξ)",
            "weight_derivation": f"predictable-phase gradient share = blink-back fraction "
                                 f"{BLINK_BACK_FRACTION} => w_p = {PHASE_WEIGHT_FRACTION_OF_SUBPIX}"
                                 f"*w_subpix (NOT tuned)",
            "engage_at_l7": "the forces engage at the l7/sharpening boundary = the flow's FINEST "
                            "curvelet/persistence scale, where the sub-pixel phase lives (the "
                            "energy's scale structure places the phase there, not a hand-set epoch)",
        },
        residual=0.0,
        source_artifact=".omx/research/flicker_transform_geometry_term_design_20260710.md",
        measurement_method="$0 n600 cached-argmax analysis + exact R ops (no scorer forward; "
                           "flicker deep-dive 2026-07-10)",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=_sidecar_prov(
            "the floor value + need band are n600-measured; the phase-tail's net-S is a run-time "
            "A/B (T1 + #360 build owed) — re-mine the floor if the vehicle/scorer changes"),
    )
    return CanonicalEquation(
        equation_id="label_floor_to_phase_tail_handoff_v1",
        name=("label->phase hand-off: fire <=> label-smooth stage AND d_seg in "
              "[0.00496,0.00700] around the GT spike rate 0.005318 AND flat; below the floor "
              "is the appearance-phase regime (T1 + #360), the only sub-0.15 descent"),
        one_line_summary=(
            "the label-smooth flow's finest-persistence floor = GT spike rate 0.0053; reaching it "
            "is the DERIVED switch to the appearance-phase tail, not a converged stop."
        ),
        latex_form=(
            r"\mathrm{fire} \iff \mathrm{label\text{-}smooth} \wedge d_{seg}\in[0.00496,0.00700]"
            r" \wedge |\dot d_{seg}/d_{seg}|\le 10^{-4};\quad d_{seg}^{floor}=0.005318"
            r"=\mathrm{spike\ rate};\ w_p=0.4\,w_{subpix}"
        ),
        python_callable_module_path=f"{_MODULE}:phase_tail_ready",
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": "phase_advection_consistency (T1) + p0_forces (#360) + --seg-spike-downweight "
                     "(T2/#274); the SENSE detector = tac.witness_control.label_floor_detector",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "the floor is n600-measured on THIS video/scorer (FAMILY-level for smooth-label "
                    "witnesses); the phase-tail forces are BUILD-owed (T1 + #360); w_p derivation "
                    "assumes #360's per-term gnorm telemetry for the 0.4*subpix target",
        },
        units_in={"rel_slope": "per_epoch_fraction", "d_seg_latest": "argmax_disagreement_rate",
                  "stage_is_label_smooth": "bool"},
        units_out={"fire": "bool"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            # naive plateau early-stop vs the 4.5-7x sub-floor descent it abandons
            "sub_floor_descent_ratio_abandoned_by_early_stop": 5.7,
        },
        last_calibration_utc="2026-07-10T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.label_floor_detector",
            "tac.witness_control.shadow_controller",
            "tac.witness_dsl.schedule_readback",
        ),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=_pred_prov("label_floor_to_phase_tail_handoff.v1"),
    )


ALL_CURRICULUM_DERIVATION_BUILDERS = (
    build_curriculum_handoff_critical_nucleus_v1,
    build_ema_window_pi_group_v1,
    build_muon_switch_conditioning_criterion_v1,
    build_rewarmup_beta2_memory_window_v1,
    build_label_floor_to_phase_tail_handoff_v1,
)


def populate_curriculum_derivation_laws_equations(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> tuple[CanonicalEquation, ...]:
    """Idempotent APPEND-ONLY registration of the FOUR witness-native curriculum-derivation laws
    (#302) into the canonical registry (mirrors ``populate_adaptive_eps_cfl_edge_tracking_equation``;
    latest-row-wins query semantics). These are the EQUATIONS leg of the schedule/curriculum DSL
    triality (#334): ``curriculum_handoff_critical_nucleus_v1`` ↔ ``Curriculum(handoff="event")``,
    ``rewarmup_beta2_memory_window_v1`` ↔ ``Transition`` (reheat), ``ema_window_pi_group_v1`` ↔ the
    EMA decay, ``muon_switch_conditioning_criterion_v1`` ↔ the Muon stage. Was ORPHANED (built +
    tested but never registered / imported into ``__init__``)."""
    from tac.canonical_equations.registry import register_canonical_equation

    out: list[CanonicalEquation] = []
    for builder in ALL_CURRICULUM_DERIVATION_BUILDERS:
        eq = builder()
        register_canonical_equation(
            eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
            notes="curriculum_derivation_laws_302_20260705 (equations leg of schedule/curriculum DSL #334)",
        )
        out.append(eq)
    return tuple(out)


__all__ = [
    "ALL_CURRICULUM_DERIVATION_BUILDERS",
    "EMA_LAG_FACTOR_EARLY_MEASURED",
    "EMA_STEPS_PER_EPOCH_RUN2",
    "HANDOFF_MIN_STAGE_EP",
    "HANDOFF_PLATEAU_REL_EPS",
    "HANDOFF_PLATEAU_WINDOW_EP",
    "MUON_COLD_SWITCH_TRANSIENT",
    "MUON_DSEG_GAIN_VS_ADAMW",
    "NUCLEUS_PI1_KNEE",
    "LABEL_FLOOR_DSEG",
    "PHASE_NEED_BAND",
    "PHASE_WEIGHT_FRACTION_OF_SUBPIX",
    "build_curriculum_handoff_critical_nucleus_v1",
    "build_ema_window_pi_group_v1",
    "build_label_floor_to_phase_tail_handoff_v1",
    "build_muon_switch_conditioning_criterion_v1",
    "build_rewarmup_beta2_memory_window_v1",
    "ema_window_steps",
    "handoff_ready",
    "min_rewarmup_epochs",
    "muon_switch_ready",
    "phase_tail_ready",
    "pi_ema",
    "populate_curriculum_derivation_laws_equations",
]
