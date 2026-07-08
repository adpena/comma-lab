# SPDX-License-Identifier: MIT
"""Canonical equations: T5 CRUCIBLE measured laws (2026-07-07) — the EQUATIONS leg of the
crucible's requirement-G triality integration ("all must be integrated into triality
especially DSL and tasks or it's ephemeral", operator 2026-07-07).

Eight laws distilled from the six seat positions in ``.omx/research/t5_crucible/``
(S2 schedule/curriculum · S3 costate/spectrum · S4 rate · S5 lever-ledger · S6
pose+byte-close), each anchored to the REAL artifact it was measured on. Every row is
`[macOS-CPU advisory]` / `[macOS-MLX research-signal]` NON-PROMOTABLE — NONE of these are
contest-authority rows; pointer contest-CPU **0.19110 UNMOVED** (everything here is MEANS).

The measured substrate is the council-designed CLEAN BASELINE run **mod32cap**
(``experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z``, best d_seg
0.0033662 @ ep650 EMA) — a deliberate T3 clean config (eik-0, no seeding/band/prior/birth),
NOT a missing-research gap.

Registration tool: ``tools/register_t5_crucible_equations_20260707.py`` (idempotent,
append-only 'registered' events + 3 'anchor_appended' updates to existing rows).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    ASSUMED_AWAITING_VERIFICATION,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

_UTC = "2026-07-07T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_MLX_SIGNAL = "[macOS-MLX research-signal]"
_HW = "m5_max_cpu"
_RATE_DENOM = 37_545_489

_S2 = ".omx/research/t5_crucible/position_S2_schedule_curriculum_20260707.md"
_S3 = ".omx/research/t5_crucible/position_S3_costate_20260707.md"
_S4 = ".omx/research/t5_crucible/position_S4_rate_20260707.md"
_S5 = ".omx/research/t5_crucible/position_S5_lever_ledger_20260707.md"
_S6 = ".omx/research/t5_crucible/position_S6_pose_byteclose_20260707.md"
_MOD32CAP_RESULT = (
    "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_train_result.json"
)
_LANE_SHARE_PROBE = (
    "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/lane_share_probe_ep225_n600.json"
)
_SPECTRUM_JSON = "experiments/results/t5_s3_hvp_lanczos_20260707/spectrum_ep650_K8_s0.json"
_S6_WEIGHTSONLY = "reports/t5_s6_byteclose_mod32cap_ep650_weightsonly_20260707.json"
_S6_BAND = "reports/t5_s6_byteclose_mod32cap_ep650_band_20260707.json"
_S6_POSESN = "reports/t5_s6_byteclose_mod32cap_ep650_posesn_20260707.json"

# ---- measured constants (each labeled at its anchor) --------------------------------------
MOD32CAP_BEST_DSEG = 0.0033662          # ep650 EMA-best, n600 verdict [advisory]
ANNEAL_FREEZE_EPOCH = 726               # Muon fire = anneal freeze (mod32cap)
ANNEAL_DENOM_EPOCH = 1000               # --anneal-epochs default = run length
HOSC_BETA_TRUNCATED = 3.177             # of derived endpoint 4.00
SOFTMAX_TAU_TRUNCATED = 0.2157          # of configured endpoint 0.05
MUON_COLD_QUENCH_FRAC = 0.275           # 0.0034139@725 -> 0.0043514@750
TAU_STAGE_ASYMPTOTE = 0.003377          # exponential fit (dAIC -47), tau_e=79ep
MUON_STAGE_ASYMPTOTE = 0.003236         # exponential fit, tau_e=305ep (extrapolated)
MUON_RECOVERY_TAU_E_EP = 305
MUON_REMAINING_BUDGET_EP = 274
BASE_BROTLI_BYTES = 61_838              # weights stream int8+brotli-11
BASE_H0_BYTES = 61_303                  # order-0 entropy floor of the same stream
CODE_BROTLI_BYTES = 20_355              # code stream int8+brotli-11
CODE_H0_BYTES = 31_799
CODE_TEMPORAL_DELTA_BYTES = 33_411      # PR95-L25 temporal delta on our code: +64% (NEGATIVE)
MOD32CAP_ARCHIVE_BYTES = 83_406         # S4 probe, zip DEFLATED
MOD32CAP_ARCHIVE_BYTES_BYTECLOSE = 83_430  # S6 real byte-close tool, bit-exact PASS
MOD48_CAPACITY_TOLL_BYTES = 20_200      # INFERRED (measured shapes + brotli ratios)
BIG3_FLIP_SHARE = 0.361                 # Road 20.3 + Undriv 12.7 + MyCar 3.1 (%)
ISLAND_FLIP_SHARE = 0.639               # Movable 44.8% + Lane 19.1% of d_seg (un-born)
T3_DSEG_NEED = 0.00092
POSE_T1_DPOSE_MAX = 1.51e-4             # at d_seg 0.00092 / rate 0.0602
POSE_T3_DPOSE_MAX = 3.2e-5
R1_STORE_NOTHING_DPOSE_FLOOR = 0.0011   # measured plateau (ep1074/1093, #245)
POSE_BLIND_DPOSE = 125.833              # mod32cap costate_shadow.jsonl
RITZ_EP650_K8 = (-369.7, -94.1, -18.5, 14.6, 61.5, 93.9, 139.3)
RITZ_GRAD_NORM = 0.787

ANNEAL_TRUNCATION_EQUATION_ID = "anneal_truncation_fixed_clock_defect_v1"
FINISHER_BUDGET_EQUATION_ID = "finisher_transient_budget_and_meat_exhaustion_v1"
ENTROPY_FLOOR_EQUATION_ID = "weights_at_order0_entropy_floor_v1"
ARCHIVE_RATE_EQUATION_ID = "mod32cap_measured_archive_rate_and_capacity_toll_v1"
POSE_WALL_EQUATION_ID = "pose_second_wall_t1_feasibility_bound_v1"
ISLANDS_FLOOR_EQUATION_ID = "islands_necessity_floor_big3_only_v1"
LEDGER_TRUTH_EQUATION_ID = "activation_ledger_not_run_truth_v1"
SPECTRUM_EQUATION_ID = "gn_hessian_spectrum_indefinite_at_ema_best_v1"


# ---- callables ----------------------------------------------------------------------------
def effective_anneal_value(start: float, end: float, t_freeze: int, t_denom: int) -> float:
    """The silently-truncated effective anneal endpoint when a consumer freezes the anneal
    at ``t_freeze`` while the denominator is bound to the fixed clock ``t_denom`` (linear
    form): value = start + (end-start)·min(t_freeze/t_denom, 1)."""
    frac = min(max(float(t_freeze) / float(t_denom), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * frac


def finisher_budget_feasible(tau_e_epochs: float, remaining_budget_epochs: float) -> bool:
    """Finisher transient×budget law: a finisher whose recovery time constant exceeds its
    remaining epoch budget cannot pay back its own switch transient (mod32cap receipt:
    tau_e=305 > 274 -> ended +11% above the ep650 entry best)."""
    return float(tau_e_epochs) < float(remaining_budget_epochs)


def coder_slack_fraction(coded_bytes: float, order0_floor_bytes: float) -> float:
    """coded/H0 ratio: ~1.0 (mod32cap base weights: 1.009) means ZERO coder slack — only
    fewer symbols (bit-depth) or lower symbol entropy (in-training shaping) move the rate;
    <1.0 (code stream: 0.64) means context/structure remains that a coder already exploits."""
    return float(coded_bytes) / float(order0_floor_bytes)


def predict_rate_term(archive_bytes: float) -> float:
    """Exact contest rate term for a byte-closed archive of ``archive_bytes``."""
    return 25.0 * float(archive_bytes) / _RATE_DENOM


def big3_only_dseg_floor(dseg_best: float = MOD32CAP_BEST_DSEG,
                         island_flip_share: float = ISLAND_FLIP_SHARE) -> float:
    """The d_seg floor if ALL big-3 boundary jitter were removed but no island (lane/movable)
    class is ever born: floor = island_share × d_seg (mod32cap: 0.639×0.0033662 ≈ 0.00215
    > the 0.00092 T_3 need ⇒ island birth is NECESSARY for T_3 — arithmetic, not preference)."""
    return float(island_flip_share) * float(dseg_best)


def t1_feasible_dpose_max(dseg: float = 0.00092, rate: float = 0.0602,
                          pointer: float = 0.19110) -> float:
    """The largest d_pose compatible with S < pointer at the given d_seg/rate working point:
    d_pose_max = ((pointer − 100·d_seg − rate)²)/10 (from S = 100·d_seg + √(10·d_pose) + rate).
    Negative headroom returns 0.0 (infeasible at any pose)."""
    headroom = float(pointer) - 100.0 * float(dseg) - float(rate)
    if headroom <= 0.0:
        return 0.0
    return headroom * headroom / 10.0


def raw_flag_runs_visible_to_ledger() -> bool:
    """Apparatus law: the activation ledger records ONLY --dsl-lever-path launches; every
    historical raw-flag launch.sh run is invisible to it. launch.sh + run.log are the run
    ground truth; the ledger is NOT (until the S5-R1 argv→lever ingest lands)."""
    return False


def indefiniteness_ratio() -> float:
    """|λ₋|/λ_max at the mod32cap ep650 EMA-best (K=8 Ritz, PROVISIONAL): 369.7/139.3 ≈ 2.65."""
    return abs(RITZ_EP650_K8[0]) / RITZ_EP650_K8[-1]


# ---- builders ------------------------------------------------------------------------------
def build_anneal_truncation_fixed_clock_defect_v1() -> CanonicalEquation:
    """S2 M-S2-2: anneal denominators bound to a fixed clock + an early consumer freeze
    silently truncate the effective anneal endpoints — a consumer-precondition violation class."""
    anchor_truncation = EmpiricalAnchor(
        anchor_id="mod32cap_anneal_truncated_beta_tau_at_muon_freeze_20260707",
        measurement_utc=_UTC,
        inputs={"run": "mod32cap clean baseline", "freeze_epoch": ANNEAL_FREEZE_EPOCH,
                "anneal_denominator_epoch": ANNEAL_DENOM_EPOCH,
                "derived_beta_end": 4.00, "configured_tau_end": 0.05},
        predicted_output={"hosc_beta_at_freeze": effective_anneal_value(1.0, 4.0, 726, 1000)},
        empirical_output={
            "hosc_beta_ep726_and_ep1000": HOSC_BETA_TRUNCATED,
            "softmax_temp_ep726_and_ep1000": SOFTMAX_TAU_TRUNCATED,
            "note": "loss_terms rows at ep726 AND ep1000 read identically -> the Muon freeze "
                    "truncated BOTH anneals at 72.6% of their paths; the control's 0.0033662 "
                    "best sat on an INCOMPLETE anneal (beta 3.177/4.00, tau 0.216/0.05)",
        },
        residual=0.0,
        source_artifact=_MOD32CAP_RESULT,
        measurement_method="run_log_loss_terms_rows_ep726_ep1000",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S2,
            reactivation_criteria="re-measure on any run where the finisher fires at an event "
                                  "epoch != --anneal-epochs; fix = --anneal-epochs bound to the "
                                  "finisher CAP or anneal-complete as fire precondition",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )
    anchor_mechanism = EmpiricalAnchor(
        anchor_id="anneal_freeze_mechanism_source_verified_20260707",
        measurement_utc=_UTC,
        inputs={"surface": "experiments/train_levelset_witness_realized_through_R_mlx.py:2318-2400",
                "functions": ["_hosc_beta_for_epoch", "_softmax_temp_for_epoch"]},
        predicted_output={"mechanism": "freeze at muon-start; denominator = --anneal-epochs"},
        empirical_output={"mechanism_confirmed": True},
        residual=0.0,
        source_artifact=_S2,
        measurement_method="source_inspection",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S2,
            reactivation_criteria="re-verify if the trainer's anneal extraction changes",
            measurement_axis="[predicted]",
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=ANNEAL_TRUNCATION_EQUATION_ID,
        name="Anneal-truncation fixed-clock defect (consumer-precondition violation class)",
        one_line_summary=(
            "fixed-clock anneal denominators + an early consumer freeze silently truncate "
            "endpoints (mod32cap: beta 3.177/4.00, tau 0.216/0.05 at the ep726 Muon freeze); "
            "fix: anneal-complete precondition"
        ),
        latex_form=(
            r"v_{\mathrm{eff}} = v_0 + (v_1{-}v_0)\,\tfrac{t_{\mathrm{freeze}}}{t_{\mathrm{denom}}}"
            r"\ \ne\ v_1\ \text{when}\ t_{\mathrm{freeze}} < t_{\mathrm{denom}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.t5_crucible_measured_laws_20260707:effective_anneal_value"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-MLX research-signal"],
            "note": "generalizes to ANY (anneal, consumer) pair where the consumer fires before "
                    "the anneal clock completes; the P2 draft's completion-guarantee contract "
                    "(denominators bound to events, never a fixed clock) is the class fix",
        },
        units_in={"start": "anneal_units", "end": "anneal_units", "t_freeze": "epochs",
                  "t_denom": "epochs"},
        units_out={"effective_endpoint": "anneal_units"},
        empirical_anchors=(anchor_truncation, anchor_mechanism),
        predicted_vs_empirical_residual={"run_log_loss_terms_rows_ep726_ep1000": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",
            "tools/launch_witness_run.py",
        ),
        canonical_producers=(
            ".omx/research/t5_crucible/position_S2_schedule_curriculum_20260707.md",
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="anneal_truncation_fixed_clock_defect.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )


def build_finisher_transient_budget_and_meat_exhaustion_v1() -> CanonicalEquation:
    """S2 M-S2-1/4/5: cold-quench + meat-exhaustion + transient×budget finisher law."""
    anchor_quench = EmpiricalAnchor(
        anchor_id="mod32cap_cold_muon_quench_275pct_never_rebeat_ep650_20260707",
        measurement_utc=_UTC,
        inputs={"run": "mod32cap", "muon_fire_epoch": 726, "warm_start_momentum": False,
                "muon_lr_final_frac": 1.0},
        predicted_output={"transition": "cold switch quench expected (built #269 lever OFF)"},
        empirical_output={
            "dseg_ep725": 0.0034139, "dseg_ep750": 0.0043514,
            "quench_frac": MUON_COLD_QUENCH_FRAC,
            "dseg_ep1000": 0.0037373,
            "note": "+27.5% transition quench; recovery incomplete — ep1000 = +11% ABOVE the "
                    "ep650 best 0.0033662; the finisher NEVER re-beat its entry point",
        },
        residual=0.0,
        source_artifact=_MOD32CAP_RESULT,
        measurement_method="n600_verdict_history_41_rows",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S2,
            reactivation_criteria="RECESS-4: resume ep650 BEST -> Muon with --muon-warm-start-momentum "
                                  "+ --muon-lr-final-frac 0.1 (the #270 restart pattern)",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )
    anchor_meat = EmpiricalAnchor(
        anchor_id="mod32cap_tau_stage_meat_exhaustion_76_125ep_past_20260707",
        measurement_utc=_UTC,
        inputs={"stage": "tau (ep300-725)", "fit": "powerlaw_meat_exit / fit_tail_models",
                "preferred_model": "exponential", "delta_aic": -47},
        predicted_output={"asymptote": TAU_STAGE_ASYMPTOTE},
        empirical_output={
            "asymptote": TAU_STAGE_ASYMPTOTE, "tau_e_epochs": 79,
            "remaining_meat_at_plus300ep": 5.5e-6,
            "note": "tau stage exhausted by ~ep600-650; 76-125 epochs ran past saturation before "
                    "the fixed Muon boundary — ~35% of the run's wall-clock was past-exhaustion "
                    "or regression",
        },
        residual=0.0,
        source_artifact=_MOD32CAP_RESULT,
        measurement_method="tac.witness_control.powerlaw_exit deterministic fit",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S2,
            reactivation_criteria="wire ExitEvent(powerlaw_meat) into the clean early-stop arming "
                                  "path (S2 BUILD rank 5); per-class meat split = S2 RECESS-3",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )
    anchor_budget = EmpiricalAnchor(
        anchor_id="mod32cap_finisher_recovery_tau_e_305_exceeds_budget_274_20260707",
        measurement_utc=_UTC,
        inputs={"stage": "Muon finisher (ep750-1000)", "fit": "exponential (11 points)"},
        predicted_output={"failure_mode": "TRANSIENT x BUDGET, not paradigm"},
        empirical_output={
            "muon_asymptote_extrapolated": MUON_STAGE_ASYMPTOTE,
            "recovery_tau_e_epochs": MUON_RECOVERY_TAU_E_EP,
            "remaining_budget_epochs": MUON_REMAINING_BUDGET_EP,
            "note": "tau_e=305 > 274 budget -> the step never settles; the extrapolated asymptote "
                    "0.003236 < the tau asymptote 0.003377 means the kappa-buster premise is NOT "
                    "falsified — the SCHEDULE around it was the failure (asymptote comparison is "
                    "an 11-point extrapolation, ADVISORY, routed to S2 RECESS-4/RECESS-1)",
        },
        residual=0.0,
        source_artifact=_MOD32CAP_RESULT,
        measurement_method="exponential_tail_fit_11_points_extrapolated",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S2,
            reactivation_criteria="S2 RECESS-4 finisher-schedule A/B in anger + RECESS-1 "
                                  "HVP-Lanczos kappa trace verify/kill the premise",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=FINISHER_BUDGET_EQUATION_ID,
        name="Finisher transient×budget law + cold-quench + meat-exhaustion (mod32cap decomposition)",
        one_line_summary=(
            "finisher viable only if recovery tau_e < budget AND warm anneal-complete entry; "
            "mod32cap: +27.5% cold quench, 76-125 ep past meat, tau_e 305 > 274 -> ended +11% "
            "above entry best (budget, not paradigm)"
        ),
        latex_form=(
            r"\text{viable} \iff \tau_e < T_{\mathrm{budget}}\ \wedge\ \text{warm}\ \wedge\ "
            r"\text{anneal-complete};\ \ \Delta d_{seg}^{\mathrm{cold}} = +27.5\%"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.t5_crucible_measured_laws_20260707:finisher_budget_feasible"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "stage": "AdamW->Muon finisher on the mod32cap clean baseline",
            "measurement_axis": ["macOS-MLX research-signal"],
            "note": "event exits with epoch CAPS + anneal-complete precondition + regression guard "
                    "are the derived fixes (S2 BUILD ranks 1/2/4/5); exits alone save ~35% "
                    "wall-clock at zero score cost on this trajectory",
        },
        units_in={"tau_e_epochs": "epochs", "remaining_budget_epochs": "epochs"},
        units_out={"viable": "bool"},
        empirical_anchors=(anchor_quench, anchor_meat, anchor_budget),
        predicted_vs_empirical_residual={"n600_verdict_history_41_rows": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",
            "tools/costate_digest.py",
            "src/tac/witness_control/powerlaw_exit.py",
        ),
        canonical_producers=(
            ".omx/research/t5_crucible/position_S2_schedule_curriculum_20260707.md",
            "src/tac/witness_control/powerlaw_exit.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="finisher_transient_budget_and_meat_exhaustion.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )


def build_weights_at_order0_entropy_floor_v1() -> CanonicalEquation:
    """S4 P1: the base weights stream sits AT its order-0 entropy floor (zero coder slack)."""
    anchor_floor = EmpiricalAnchor(
        anchor_id="mod32cap_base_weights_brotli_at_order0_floor_20260707",
        measurement_utc=_UTC,
        inputs={"checkpoint": "mod32cap ep650 EMA-BEST (111,103 counted params)",
                "grammar": "int8 symmetric + brotli-11 (shipped)"},
        predicted_output={"coder_slack": "none expected if brotli exhausts order-0 structure"},
        empirical_output={
            "base_brotli_bytes": BASE_BROTLI_BYTES, "base_h0_bytes": BASE_H0_BYTES,
            "base_ratio": 1.009, "lzma_9e_bytes": 62_488,
            "code_brotli_bytes": CODE_BROTLI_BYTES, "code_h0_bytes": CODE_H0_BYTES,
            "code_ratio": 0.64,
            "note": "base at 100.9% of H0 -> ZERO coder slack (only bit-depth/waterfill or "
                    "in-training entropy shaping move base rate; coder migration is dead); the "
                    "code stream is 36% BELOW H0 (context structure remains, already exploited)",
        },
        residual=0.0,
        source_artifact=_S4,
        measurement_method="inline_probe_real_byte_counts_on_ema_best_npz",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S4,
            reactivation_criteria="re-measure after WeightEntropyPenaltyMLX A/B (S4 R2) or any "
                                  "bit-depth change (Arm-E #336 waterfill, S4 R1)",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    anchor_l25_negative = EmpiricalAnchor(
        anchor_id="pr95_l25_temporal_delta_on_witness_code_measured_negative_20260707",
        measurement_utc=_UTC,
        inputs={"transform": "PR95-L25 temporal-delta on the (1200,32) code table",
                "baseline_bytes": CODE_BROTLI_BYTES},
        predicted_output={"pr95_l25_expectation": "smaller (smooth temporal trajectory prior)"},
        empirical_output={
            "temporal_delta_bytes": CODE_TEMPORAL_DELTA_BYTES,
            "delta_pct": "+64%",
            "note": "MEASURED NEGATIVE — cargo-cult DROP with receipt: the witness code table is "
                    "not a smooth temporal trajectory at int8 granularity",
        },
        residual=0.0,
        source_artifact=_S4,
        measurement_method="inline_probe_real_byte_counts_on_ema_best_npz",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S4,
            reactivation_criteria="none foreseen (recorded so nobody re-opens L25 on this payload)",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    anchor_grammar_v2 = EmpiricalAnchor(
        anchor_id="grammar_rev2_free_rate_wins_minus_2p8kb_20260707",
        measurement_utc=_UTC,
        inputs={"changes": ["column-major code stream pre-brotli",
                            "brotli the manifest then STORED 1-char zip member"]},
        predicted_output={"combined_bytes": "~81,620 -> rate 0.05434"},
        empirical_output={
            "code_col_major_bytes": 19_022, "code_col_major_delta": -1_333,
            "brotli_manifest_stored_bytes": 82_970, "brotli_manifest_stored_delta": -436,
            "combined_estimate_bytes": 81_620,
            "note": "-2.8 KB ≈ -0.0019 rate for free (byte-close tool change only); nuance: with "
                    "a plain-text manifest DEFLATED beats STORED by 816 B — brotli the manifest "
                    "FIRST, then the #79 STORED/1-char floor applies",
        },
        residual=0.0,
        source_artifact=_S4,
        measurement_method="inline_probe_real_byte_counts_on_ema_best_npz",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S4,
            reactivation_criteria="S4 R4: fold into levelset_byte_close_and_eval.py with "
                                  "bit-identical dequantized params + full parity gate",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=ENTROPY_FLOOR_EQUATION_ID,
        name="Base weights at the order-0 entropy floor (zero coder slack); code stream 36% below",
        one_line_summary=(
            "base weights brotli = 100.9% of order-0 H0 -> zero coder slack (only bit-depth or "
            "entropy shaping move base rate); code 36% below H0; L25 temporal-delta NEGATIVE "
            "(+64%); grammar rev2 -2.8 KB free"
        ),
        latex_form=(
            r"\frac{|\mathrm{brotli}(W_{int8})|}{H_0(W_{int8})} = 1.009 \Rightarrow "
            r"\text{coder slack} \approx 0;\ \frac{|\mathrm{brotli}(C)|}{H_0(C)} = 0.64"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.t5_crucible_measured_laws_20260707:coder_slack_fraction"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "grammar": "int8 symmetric + brotli-11 LVLS1 blob",
            "measurement_axis": ["macOS-CPU advisory"],
            "byte_closed": True,
        },
        units_in={"coded_bytes": "bytes", "order0_floor_bytes": "bytes"},
        units_out={"slack_ratio": "dimensionless"},
        empirical_anchors=(anchor_floor, anchor_l25_negative, anchor_grammar_v2),
        predicted_vs_empirical_residual={"inline_probe_real_byte_counts_on_ema_best_npz": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/levelset_byte_close_and_eval.py",
            "tac.witness_dsl.gauge",
            "tools/apply_sensitivity_bitalloc_witness.py",
        ),
        canonical_producers=(
            ".omx/research/t5_crucible/position_S4_rate_20260707.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="weights_at_order0_entropy_floor.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )


def build_mod32cap_measured_archive_rate_and_capacity_toll_v1() -> CanonicalEquation:
    """S4+S6: the FIRST measured archive/rate rows for the clean baseline + the mod48 toll."""
    anchor_s4 = EmpiricalAnchor(
        anchor_id="mod32cap_ep650_archive_83406B_rate_005553_20260707",
        measurement_utc=_UTC,
        inputs={"checkpoint": "mod32cap ep650 EMA-BEST", "grammar": "LVLS1 int8+brotli, zip DEFLATED",
                "counted_streams": {"base": 61_838, "code": 20_355, "manifest_zip": 1_213}},
        predicted_output={"rate_term": predict_rate_term(MOD32CAP_ARCHIVE_BYTES)},
        empirical_output={
            "archive_zip_bytes": MOD32CAP_ARCHIVE_BYTES,
            "rate_term": 0.05553,
            "note": "S4 inline probe mirroring quantize_levelset_blob exactly; the DAG's earlier "
                    "0.05499/82.6 KB was a different checkpoint state — the ep650-BEST row is the "
                    "clean-baseline authority going forward",
        },
        residual=0.0,
        source_artifact=_S4,
        measurement_method="inline_probe_real_byte_counts_on_ema_best_npz",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S4,
            reactivation_criteria="re-measure on any new best checkpoint or grammar change",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    anchor_s6_crosscheck = EmpiricalAnchor(
        anchor_id="mod32cap_ep650_byteclose_83430B_bitexact_crosscheck_20260707",
        measurement_utc=_UTC,
        inputs={"tool": "tools/levelset_byte_close_and_eval.py (real byte-close, weights-only)",
                "params": 111_095, "so_freq_along": 8},
        predicted_output={"cross_check": "within container noise of S4's 83,406"},
        empirical_output={
            "archive_zip_bytes": MOD32CAP_ARCHIVE_BYTES_BYTECLOSE,
            "rate_term": 0.05555,
            "bit_exact_2pair_gate": "PASS",
            "delta_vs_s4_independent_accounting_bytes": 24,
            "note": "independent cross-check within 24 B; bit-exact decode gate PASS on all 3 "
                    "compositions run today (weightsonly/band/posesn)",
        },
        residual=0.0,
        source_artifact=_S6_WEIGHTSONLY,
        measurement_method="real_byte_close_tool_bit_exact_gate",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S6_WEIGHTSONLY,
            reactivation_criteria="S6 R-1: n600 realized-parity row on the clean baseline (G5); "
                                  "byte-closes of mod32cap MUST pass --so-freq-along 8 until the "
                                  "G1 cfg-consume fix lands",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    anchor_capacity_toll = EmpiricalAnchor(
        anchor_id="mod32_to_mod48_capacity_toll_20p2kb_inferred_20260707",
        measurement_utc=_UTC,
        inputs={"scaling_tensors": ["code (1200 x mod)", "film.weight (768 x mod)"],
                "brotli_ratios_held": {"film": 0.825, "code": 0.530}},
        predicted_output={"toll_bytes": MOD48_CAPACITY_TOLL_BYTES, "toll_rate": 0.0135},
        empirical_output={
            "status": "INFERRED from measured mod32 shapes + per-stream brotli ratios",
            "pays_iff": "delta_d_seg < -1.35e-4",
            "note": "capacity is SECONDARY and now COSTED; S4 R5 pre-registers the projection — "
                    "if the realized mod48 blob differs by >10% the ratio assumption is falsified",
        },
        residual=0.0,
        source_artifact=_S4,
        measurement_method="inferred_from_measured_shapes_and_ratios",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S4,
            reactivation_criteria="S4 R5: the mod48 arm's own realized blob measures the truth",
            measurement_axis="[predicted]",
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=ARCHIVE_RATE_EQUATION_ID,
        name="mod32cap clean-baseline measured archive rate (83.4 KB / 0.0555) + mod48 capacity toll",
        one_line_summary=(
            "clean-baseline measured archive: 83,406 B -> rate 0.05553, cross-checked 83,430 B "
            "bit-exact byte-close (within 24 B); mod32->48 toll +20.2 KB = +0.0135 rate "
            "(INFERRED; pays iff dd_seg < -1.35e-4)"
        ),
        latex_form=(
            r"\mathrm{rate} = \tfrac{25\,B}{37{,}545{,}489};\ B_{\mathrm{mod32cap}} = 83{,}406\;"
            r"\text{(probe)} \approx 83{,}430\;\text{(byte-close)};\ \Delta B_{32\to48} \approx +20.2\,\mathrm{KB}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.t5_crucible_measured_laws_20260707:predict_rate_term"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "checkpoint": "mod32cap ep650 EMA-BEST",
            "measurement_axis": ["macOS-CPU advisory"],
            "byte_closed": True,
            "note": "archive bytes are host-independent (zip stat); score semantics advisory; "
                    "budget context: mod32+band+xi = 110-117 KB EXCEEDS the ~105 KB sub-0.15 "
                    "headroom -> the compress-half is NOT optional if band+pose+capacity compose",
        },
        units_in={"archive_bytes": "bytes"},
        units_out={"rate_term": "score_units_contest_rate_term"},
        empirical_anchors=(anchor_s4, anchor_s6_crosscheck, anchor_capacity_toll),
        predicted_vs_empirical_residual={"real_byte_close_tool_bit_exact_gate": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/costate_digest.py",
            "tac.witness_dsl.gauge",
        ),
        canonical_producers=(
            ".omx/research/t5_crucible/position_S4_rate_20260707.md",
            ".omx/research/t5_crucible/position_S6_pose_byteclose_20260707.md",
            "tools/levelset_byte_close_and_eval.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="mod32cap_measured_archive_rate_and_capacity_toll.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )


def build_pose_second_wall_t1_feasibility_bound_v1() -> CanonicalEquation:
    """S6 §5B: pose is the SECOND WALL — the T_1-feasible d_pose bound and the measured misses."""
    anchor_bound = EmpiricalAnchor(
        anchor_id="pose_t1_feasible_dpose_bound_1p51em4_derived_20260707",
        measurement_utc=_UTC,
        inputs={"score_law": "S = 100*d_seg + sqrt(10*d_pose) + rate (upstream/evaluate.py)",
                "working_point": {"d_seg": 0.00092, "rate": 0.0602}},
        predicted_output={"t1_dpose_max": POSE_T1_DPOSE_MAX, "t3_dpose_max": POSE_T3_DPOSE_MAX},
        empirical_output={
            "t1_dpose_max": POSE_T1_DPOSE_MAX,
            "t3_dpose_max": POSE_T3_DPOSE_MAX,
            "note": "exact arithmetic from the score law + TODAY's measured rate rows; even a "
                    "perfect d_seg run cannot cross 0.19110 unless the pose mechanism beats R1's "
                    "measured floor by ~7x",
        },
        residual=0.0,
        source_artifact=_S6,
        measurement_method="exact_arithmetic_from_score_law_and_measured_rates",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S6,
            reactivation_criteria="re-derive at any new measured (d_seg, rate) working point",
            measurement_axis="[predicted]",
            hardware_substrate=_HW,
        ),
    )
    anchor_r1_floor = EmpiricalAnchor(
        anchor_id="r1_store_nothing_dpose_floor_0p0011_misses_7x_20260707",
        measurement_utc=_UTC,
        inputs={"run": "R1 (#245) store-nothing pose trained through-R, w_pose=0 for the null",
                "epochs": "ep1074/1093 plateau"},
        predicted_output={"pose_term_at_floor": 0.105},
        empirical_output={
            "dpose_floor": R1_STORE_NOTHING_DPOSE_FLOOR,
            "pose_term": 0.105,
            "miss_factor_vs_t1_bound": 7,
            "dseg_held": 0.0046,
            "note": "R1 trained with w_pose=0 — 'the null was never used for pose'; the unfired L3 "
                    "mechanism (w_pose>0 + FiLM-on-xi + null-texture) is the designed lever; if it "
                    "floors above 1.5e-4 the symposium L1 Jacobian fallback activates",
        },
        residual=0.0,
        source_artifact=".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md",
        measurement_method="dag_feed_rows_7496_7527_r1_run",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S6,
            reactivation_criteria="S6 M5/R-5: the bounded pose-ON n600 smoke (operator GO); kill "
                                  "pose-ON-as-designed if converged d_pose > 1.5e-4",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    anchor_pose_blind = EmpiricalAnchor(
        anchor_id="mod32cap_pose_blind_term_35p5_unsubmittable_20260707",
        measurement_utc=_UTC,
        inputs={"run": "mod32cap (w_pose=0 by design)", "source": "costate_shadow.jsonl"},
        predicted_output={"pose_term": 35.5},
        empirical_output={
            "d_pose": POSE_BLIND_DPOSE,
            "pose_term": 35.5,
            "note": "sqrt(10*125.833) ≈ 35.5 — a pose-blind row is UNSUBMITTABLE; a stored sidecar "
                    "does NOT fix it (the scorer runs PoseNet on the FRAMES, source-verified); pose "
                    "must be TRAINED-IN",
        },
        residual=0.0,
        source_artifact=_S6,
        measurement_method="costate_shadow_jsonl_operating_point",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S6,
            reactivation_criteria="pose-ON pointer-run design (S6 P1 two-track) supersedes",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=POSE_WALL_EQUATION_ID,
        name="Pose second-wall T_1 feasibility bound (d_pose <= 1.51e-4 at the measured working point)",
        one_line_summary=(
            "T_1-feasible d_pose <= 1.51e-4 (T_3 <= 3.2e-5) at d_seg 0.00092/rate 0.0602; R1 "
            "floor 0.0011 -> term 0.105 = 7x miss; pose-blind term ~35.5 unsubmittable -> pose "
            "is the SECOND wall"
        ),
        latex_form=(
            r"d_{pose}^{\max} = \tfrac{(S_{\mathrm{ptr}} - 100\,d_{seg} - \mathrm{rate})^2}{10}"
            r" = 1.51\times10^{-4}\ \text{at}\ (0.00092,\ 0.0602)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.t5_crucible_measured_laws_20260707:t1_feasible_dpose_max"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": "pose OPEN on the witness (memory L68/L69); the L3 mechanism's achievable "
                    "floor is UNMEASURED (S6 M5 measures it); the 3.4e-5 ancestor number is "
                    "BORROWED and never cited as solved",
        },
        units_in={"dseg": "segnet_argmax_dseg", "rate": "score_units_contest_rate_term",
                  "pointer": "contest_score"},
        units_out={"dpose_max": "posenet_mse_dpose"},
        empirical_anchors=(anchor_bound, anchor_r1_floor, anchor_pose_blind),
        predicted_vs_empirical_residual={"dag_feed_rows_7496_7527_r1_run": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/costate_digest.py",
            "tac.witness_dsl.curriculum_dsl",
        ),
        canonical_producers=(
            ".omx/research/t5_crucible/position_S6_pose_byteclose_20260707.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="pose_second_wall_t1_feasibility_bound.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )


def build_islands_necessity_floor_big3_only_v1() -> CanonicalEquation:
    """S5 headline theorem: island birth is NECESSARY for T_3 (arithmetic, not preference)."""
    anchor_probe = EmpiricalAnchor(
        anchor_id="mod32cap_island_flip_share_639pct_big3_floor_0p00215_20260707",
        measurement_utc=_UTC,
        inputs={"probe": "lane_share_probe_ep225_n600 (witness-alone, mod32cap run dir)",
                "d_seg_best": MOD32CAP_BEST_DSEG},
        predicted_output={"big3_only_floor": big3_only_dseg_floor()},
        empirical_output={
            "island_share": {"movable": 0.448, "lane": 0.191},
            "big3_jitter_share": {"road": 0.203, "undrivable": 0.127, "mycar": 0.031},
            "big3_only_floor": 0.00215,
            "t3_need": T3_DSEG_NEED,
            "within_class_unborn": {"lane": 0.839, "movable": 0.931},
            "note": "perfect removal of ALL big-3 jitter with zero island birth leaves d_seg "
                    "~0.639 x 0.0033662 ~ 0.00215 > the 0.00092 T_3 need => island birth "
                    "NECESSARY for T_3; caveat welded: shares are witness-alone ep225 upper "
                    "bounds, but the within-class un-born fractions independently transfer "
                    "(live part_frac lane=movable=0 all run)",
        },
        residual=0.0,
        source_artifact=_LANE_SHARE_PROBE,
        measurement_method="n600_witness_alone_flip_share_probe_ep225",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_LANE_SHARE_PROBE,
            reactivation_criteria="re-measure shares on any island-arm checkpoint (the un-born "
                                  "fractions are the transferable quantity)",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=ISLANDS_FLOOR_EQUATION_ID,
        name="Islands-necessity floor: big-3-only d_seg floor 0.00215 > 0.00092 T_3 need",
        one_line_summary=(
            "un-born islands (movable 44.8% + lane 19.1%) carry 63.9% of d_seg -> big-3-only "
            "floors at ~0.00215 > 0.00092 T_3 need: island birth NECESSARY for T_3 "
            "(arithmetic, not preference)"
        ),
        latex_form=(
            r"d_{seg}^{\mathrm{floor,big3}} = s_{\mathrm{isl}}\cdot d_{seg}^{\mathrm{best}} "
            r"= 0.639 \times 0.0033662 \approx 0.00215 > 0.00092 = d_{seg}^{T_3}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.t5_crucible_measured_laws_20260707:big3_only_dseg_floor"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "checkpoint": "mod32cap (no-birth clean baseline BY DESIGN)",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "composes with #333 (~97% of d_seg mass in the ~4.7%-area annulus): "
                    "recovery-per-byte is maximized by levers acting ON the annulus (birth + "
                    "boundary position), not bulk capacity",
        },
        units_in={"dseg_best": "segnet_argmax_dseg", "island_flip_share": "fraction"},
        units_out={"dseg_floor": "segnet_argmax_dseg"},
        empirical_anchors=(anchor_probe,),
        predicted_vs_empirical_residual={"n600_witness_alone_flip_share_probe_ep225": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",
            "tools/costate_digest.py",
        ),
        canonical_producers=(
            ".omx/research/t5_crucible/position_S5_lever_ledger_20260707.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="islands_necessity_floor_big3_only.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )


def build_activation_ledger_not_run_truth_v1() -> CanonicalEquation:
    """S5 apparatus law: the activation ledger is NOT run truth; launch.sh is."""
    anchor_audit = EmpiricalAnchor(
        anchor_id="ledger_7_rows_vs_launch_sh_10_raw_fired_20260707",
        measurement_utc=_UTC,
        inputs={"ledger": ".omx/state/lever_activation_ledger.jsonl (7 rows)",
                "ground_truth": "experiments/results/levelset_n600_witness_*/launch.sh + run.log"},
        predicted_output={"ledger_reflects_runs": False},
        empirical_output={
            "raw_flag_fired_but_ledger_blind": [
                "Muon", "PoseDecouple", "AnalyticLaneRenderBand (ep300+ in #205, UNATTRIBUTED)",
                "AmplifyIsland", "PersistenceTopology", "SeedIslandBirth components",
                "EikonalViscosity", "StiefelW", "BoundaryDistance", "CacheGtSkeleton",
            ],
            "note": "~10 of the '36 never-fired' DID raw-flag fire per launch.sh; the ledger "
                    "records ONLY --dsl-lever-path launches (launch_witness_run.py:1075-1082); "
                    "'fired in a poisoned (spike-guard-frozen) run' earns a fired row, NEVER a "
                    "measured verdict; fix R1 = argv->lever reverse-map + engagement-predicate "
                    "backfill",
        },
        residual=0.0,
        source_artifact=_S5,
        measurement_method="launch_sh_run_log_audit_vs_ledger",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S5,
            reactivation_criteria="S5 R1: launch-time argv->lever ENGAGED ingest wired so the "
                                  "ledger finally reads real runs (crucible requirement G.5)",
            measurement_axis="[predicted]",
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=LEDGER_TRUTH_EQUATION_ID,
        name="Activation ledger != run truth (launch.sh is ground truth) — apparatus law",
        one_line_summary=(
            "activation ledger records only --dsl-lever launches; raw-flag launch.sh runs are "
            "invisible (~10 of '36 never-fired' raw-fired) -> run-config claims cite launch.sh, "
            "never the ledger"
        ),
        latex_form=r"\mathrm{ledger} \subsetneq \mathrm{runs};\ \mathrm{truth} = \mathrm{launch.sh}",
        python_callable_module_path=(
            "tac.canonical_equations.t5_crucible_measured_laws_20260707:raw_flag_runs_visible_to_ledger"
        ),
        domain_of_validity={
            "apparatus": ["lever_activation_ledger", "launch_witness_run"],
            "measurement_axis": ["predicted"],
            "note": "sister of the default-off-is-orphaned-signal non-negotiable: 'held but "
                    "never fired' AND 'fired but never recorded' are both orphaned signal",
        },
        units_in={},
        units_out={"visible": "bool"},
        empirical_anchors=(anchor_audit,),
        predicted_vs_empirical_residual={"launch_sh_run_log_audit_vs_ledger": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/launch_witness_run.py",
            "tools/costate_digest.py",
        ),
        canonical_producers=(
            ".omx/research/t5_crucible/position_S5_lever_ledger_20260707.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="activation_ledger_not_run_truth.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate=_HW,
        ),
    )


def build_gn_hessian_spectrum_indefinite_at_ema_best_v1() -> CanonicalEquation:
    """S3 first measurement (PROVISIONAL, K=8): the ep650 EMA-best is strongly INDEFINITE."""
    anchor_spectrum = EmpiricalAnchor(
        anchor_id="mod32cap_ep650_ritz_k8_indefinite_2p65x_20260707",
        measurement_utc=_UTC,
        inputs={"checkpoint": "mod32cap ep650 EMA-BEST", "pair_subset_K": 8, "seed": 0,
                "lanczos_iterations": 7,
                "anneal_state": {"hosc_beta": HOSC_BETA_TRUNCATED,
                                 "softmax_temp": SOFTMAX_TAU_TRUNCATED,
                                 "note": "truncated-anneal state (honors the S2 M2 defect)"}},
        predicted_output={"in_basin": "quadratic basin expected if 2nd-order exhausted"},
        empirical_output={
            "ritz_values": list(RITZ_EP650_K8),
            "lambda_neg_over_lambda_max": 2.65,
            "grad_norm": RITZ_GRAD_NORM,
            "reading": "STRONGLY INDEFINITE — best point NOT 2nd-order exhausted; TerminalSolve "
                       "in-basin precondition NOT met; the cold Muon fire was curvature-blind",
            "review_status": "fresh-eyes-UNREVIEWED (written by the seat itself)",
        },
        residual=0.0,
        source_artifact=_SPECTRUM_JSON,
        measurement_method="hvp_lanczos_k8_pair_subset_7_iters",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SPECTRUM_JSON,
            reactivation_criteria="full-P RECESS-R1 (K=128): kill band |lambda_-|/lambda_max < 0.1 "
                                  "-> capacity/basis wall (strengthens Arm A either way); also "
                                  "subject to the +4.3% self-orient checkpoint-reconstruction gap",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )
    anchor_transfer = EmpiricalAnchor(
        anchor_id="k8_to_full_p_spectrum_transfer_preregistered_20260707",
        measurement_utc=_UTC,
        inputs={"known_gap": "#341 subset-solve gap: K=8 SOLVE overfits +5.1% net (estimation "
                             "has a different failure surface, but transfer is load-bearing)"},
        predicted_output={"indefiniteness_persists_at_full_P": True},
        empirical_output={
            "status": "PRE-REGISTERED, unmeasured",
            "note": "K=8 -> full-P transfer is the load-bearing assumption; the full-P recess "
                    "either confirms or kills; additional caveat: the ep650 checkpoint "
                    "reconstruction carries a +4.3% self-orient gap (state not persisted)",
        },
        residual=0.0,
        source_artifact=_S3,
        measurement_method="preregistered_recess",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S3,
            reactivation_criteria="run the pre-registered full-P (K=128) Lanczos recess "
                                  "(experiments/t5_s3_hvp_lanczos_probe.py; ~3-4 h, <8 GiB)",
            measurement_axis="[predicted]",
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=SPECTRUM_EQUATION_ID,
        name="GN/Hessian spectrum STRONGLY INDEFINITE at the ep650 EMA-best [PROVISIONAL, K=8]",
        one_line_summary=(
            "PROVISIONAL (K=8): ep650 EMA-best Ritz [-369.7..+139.3], |lambda_-|=2.65x "
            "lambda_max, grad 0.787 -> NOT 2nd-order exhausted; TerminalSolve in-basin NOT met; "
            "cold Muon curvature-blind"
        ),
        latex_form=(
            r"\lambda(H_{ep650})\big|_{K=8} \in [-369.7,\ +139.3];\ "
            r"\tfrac{|\lambda_-|}{\lambda_{\max}} = 2.65;\ \|\nabla\| = 0.787"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.t5_crucible_measured_laws_20260707:indefiniteness_ratio"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "checkpoint": "mod32cap ep650 EMA-BEST (truncated-anneal state; ep1000 is a "
                          "cold-quench artifact, NOT a spectrum target)",
            "measurement_axis": ["macOS-CPU advisory"],
            "status": "PROVISIONAL — K=8 pair subset; full-P (K=128) recess pre-registered; "
                      "+4.3% self-orient checkpoint-reconstruction caveat; fresh-eyes-unreviewed",
        },
        units_in={},
        units_out={"indefiniteness_ratio": "dimensionless"},
        empirical_anchors=(anchor_spectrum, anchor_transfer),
        predicted_vs_empirical_residual={"hvp_lanczos_k8_pair_subset_7_iters": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/costate_digest.py",
            "tac.witness_dsl.curriculum_dsl",
        ),
        canonical_producers=(
            "experiments/t5_s3_hvp_lanczos_probe.py",
            ".omx/research/t5_crucible/position_S3_costate_20260707.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="gn_hessian_spectrum_indefinite_at_ema_best.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )


# ---- anchor updates for EXISTING equations (no duplication) --------------------------------
def build_muon_cold_quench_anchor_for_existing_row() -> EmpiricalAnchor:
    """New measured anchor for ``muon_finisher_schedule_warmstart_and_lr_anneal_v1``: the
    mod32cap clean-baseline cold-Muon fire is a SECOND, independent measurement of the
    cold-start-quench half of that law (the warm-start net verdict remains AWAITING #270)."""
    return EmpiricalAnchor(
        anchor_id="mod32cap_cold_muon_fire_ep726_quench_275pct_20260707",
        measurement_utc=_UTC,
        inputs={"run": "mod32cap clean baseline", "muon_fire_epoch": 726,
                "warm_start_momentum": False, "muon_lr_final_frac": 1.0},
        predicted_output={"cold_start_spike": "expected (the equation's own law)"},
        empirical_output={
            "quench_frac": MUON_COLD_QUENCH_FRAC,
            "dseg_ep725": 0.0034139, "dseg_ep750": 0.0043514, "dseg_ep1000": 0.0037373,
            "note": "T5-crucible S2 M-S2-1: +27.5% cold quench; the finisher never re-beat the "
                    "ep650 best 0.0033662 (ended +11% above) — second measured receipt for the "
                    "cold-start half; warm-start/anneal net d_seg remains #270-gated AWAITING",
        },
        residual=0.0,
        source_artifact=_MOD32CAP_RESULT,
        measurement_method="n600_verdict_history_41_rows",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S2,
            reactivation_criteria="the #270 warm-start+anneal arm measures the fix half",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )


def build_lane_band_byteclose_anchor_for_existing_row() -> EmpiricalAnchor:
    """New measured anchor for ``lane_band_camera_frame_rd_rate_v1``: the band's counted cost
    through the REAL byte-close tool (bit-exact gate PASS), correcting 'near-zero-byte' framing."""
    return EmpiricalAnchor(
        anchor_id="lane_band_lbnd2_real_byteclose_41562B_marginal_002767_20260707",
        measurement_utc=_UTC,
        inputs={"tool": "tools/levelset_byte_close_and_eval.py --lane-render-band (u_mask on)",
                "checkpoint": "mod32cap ep650 EMA-BEST", "codec": "LBND2 RD"},
        predicted_output={"marginal_rate": 0.02767},
        empirical_output={
            "archive_with_band_bytes": 125_267,
            "band_marginal_bytes": 41_562,
            "band_marginal_rate": 0.02767,
            "bit_exact_2pair_gate": "PASS",
            "note": "T5-crucible S6: the band is NOT 'near-zero byte' — 41.5 KB counted through "
                    "the real byte-close (LBND4 drops it to +0.02057 once the decode is inlined, "
                    "gap G2); its d_seg payment must clear this bar (S6 M2)",
        },
        residual=0.0,
        source_artifact=_S6_BAND,
        measurement_method="real_byte_close_tool_bit_exact_gate",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S6_BAND,
            reactivation_criteria="S6 M2: band ROI numerator (net delta-S through parity); G2 "
                                  "LBND4 decode inline re-prices to +0.02057",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )


def build_pose_carrier_byteclose_anchor_for_existing_row() -> EmpiricalAnchor:
    """New measured anchor for ``store_nothing_pose_carrier_rate_collapse_vs_dpose_v1``: the
    carrier's counted cost through the REAL byte-close tool with derive-H live (H_bytes=0)."""
    return EmpiricalAnchor(
        anchor_id="pose_store_nothing_real_byteclose_6929B_marginal_000464_20260707",
        measurement_utc=_UTC,
        inputs={"tool": "tools/levelset_byte_close_and_eval.py --pose-carrier "
                        "--pose-carrier-mode store_nothing",
                "checkpoint": "mod32cap ep650 EMA-BEST",
                "xi_coder": "delta_ar", "q_levels": 4096},
        predicted_output={"marginal_rate": 0.00464},
        empirical_output={
            "archive_with_pose_bytes": 90_393,
            "pose_marginal_bytes": 6_929,
            "pose_marginal_rate": 0.00464,
            "coded_xi_bytes": 6_367,
            "H_bytes": 0,
            "bit_exact_2pair_gate": "PASS",
            "note": "T5-crucible S6: derive-H LIVE at decode (rule-118 free exp_se3 + plane "
                    "homography); FINDING-1's 52,135 B -> 6,929 B (7.5x); q_levels unswept "
                    "(S6 M4 sweeps 1024/256 against the 1.51e-4 d_pose budget)",
        },
        residual=0.0,
        source_artifact=_S6_POSESN,
        measurement_method="real_byte_close_tool_bit_exact_gate",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_S6_POSESN,
            reactivation_criteria="S6 M3/M4: realized-d_pose-through-real-decode row + q_levels "
                                  "sweep on the store-nothing-trained ckpt",
            measurement_axis=_ADVISORY,
            hardware_substrate=_HW,
        ),
    )


ALL_BUILDERS = (
    build_anneal_truncation_fixed_clock_defect_v1,
    build_finisher_transient_budget_and_meat_exhaustion_v1,
    build_weights_at_order0_entropy_floor_v1,
    build_mod32cap_measured_archive_rate_and_capacity_toll_v1,
    build_pose_second_wall_t1_feasibility_bound_v1,
    build_islands_necessity_floor_big3_only_v1,
    build_activation_ledger_not_run_truth_v1,
    build_gn_hessian_spectrum_indefinite_at_ema_best_v1,
)

ANCHOR_UPDATES = (
    ("muon_finisher_schedule_warmstart_and_lr_anneal_v1",
     build_muon_cold_quench_anchor_for_existing_row),
    ("lane_band_camera_frame_rd_rate_v1",
     build_lane_band_byteclose_anchor_for_existing_row),
    ("store_nothing_pose_carrier_rate_collapse_vs_dpose_v1",
     build_pose_carrier_byteclose_anchor_for_existing_row),
)

__all__ = [
    "ALL_BUILDERS",
    "ANCHOR_UPDATES",
    "ANNEAL_TRUNCATION_EQUATION_ID",
    "ARCHIVE_RATE_EQUATION_ID",
    "ENTROPY_FLOOR_EQUATION_ID",
    "FINISHER_BUDGET_EQUATION_ID",
    "ISLANDS_FLOOR_EQUATION_ID",
    "LEDGER_TRUTH_EQUATION_ID",
    "POSE_WALL_EQUATION_ID",
    "SPECTRUM_EQUATION_ID",
    "big3_only_dseg_floor",
    "build_activation_ledger_not_run_truth_v1",
    "build_anneal_truncation_fixed_clock_defect_v1",
    "build_finisher_transient_budget_and_meat_exhaustion_v1",
    "build_gn_hessian_spectrum_indefinite_at_ema_best_v1",
    "build_islands_necessity_floor_big3_only_v1",
    "build_lane_band_byteclose_anchor_for_existing_row",
    "build_mod32cap_measured_archive_rate_and_capacity_toll_v1",
    "build_muon_cold_quench_anchor_for_existing_row",
    "build_pose_carrier_byteclose_anchor_for_existing_row",
    "build_pose_second_wall_t1_feasibility_bound_v1",
    "build_weights_at_order0_entropy_floor_v1",
    "coder_slack_fraction",
    "effective_anneal_value",
    "finisher_budget_feasible",
    "indefiniteness_ratio",
    "predict_rate_term",
    "raw_flag_runs_visible_to_ledger",
    "t1_feasible_dpose_max",
]
