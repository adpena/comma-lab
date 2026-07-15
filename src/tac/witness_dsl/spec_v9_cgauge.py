# SPDX-License-Identifier: MIT
"""SPEC_v9_cgauge — the V9·CGauge launch config as a typed, validated DSL program.

The OWED artifact from ``vehicle_v9_cgauge_naming_20260711.md`` ("SPEC_v9_cgauge +
the DSL designation"), authored by the 2026-07-11 Einstein/Fable master-action pass
(memo ``.omx/research/cgauge_master_action_and_parametrization_20260711.md``).

**V9·CGauge IS** (60-second read): the v7.5.2 single covariant trunk (texture trunk
DROPPED on both axes) + the sealed launch-1 lever set, with EVERY constant placed on
the value-provenance ladder against the CGauge master action
(``cgauge_master_action_v1``) and the #223 parametrization laws
(``cgauge_whitney_moddim_v1`` / ``cgauge_nyquist_bank_frequency_v1`` /
``cgauge_curvelet_parabolic_bank_v1`` / ``cgauge_beta2_window_v1``), plus the ONE
theory-forced addition: the T1 appearance-phase advection term armed for the terminal
band — because the flicker-floor theorem (``gt_scoredframe_spike_rate_equals_witness_
flicker_floor_v1``) PROVES no label-smooth witness can descend below 0.005318, and the
sub-0.15 need (0.00077–0.00118) is 4.5–7x below it. Pose rides the dedicated dxi
channel (banked R1 shape); mod-dim stays at the SAFE measured anchor 32 with the
DERIVED target 19 gated on the #299 Arm-A / terminal-band eff_rank read (live harvest:
eff_rank 16.4 and rising at ep125).

CONTAINMENT: this module BUILDS + VALIDATES + COMPILES the program ($0, pure); it never
launches. LAUNCH = operator-GO. Pointer 0.19108282 UNMOVED — a config is MEANS.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# The V9 delta over crucible_v752(self_orient=False) — every entry justified in
# V9_CGAUGE_PROVENANCE below (flags verified against the live trainer parser).
# ---------------------------------------------------------------------------
_V9_CGAUGE_DELTA: dict[str, Any] = {
    # T1 appearance-phase advection (theory-FORCED by the flicker-floor law; #424 built,
    # default-OFF; SEAL + n600 A/B owed BEFORE launch — encoded in the purpose string).
    # weight 0.4 = Law-5 PHASE_WEIGHT_FRACTION_OF_SUBPIX (derived from the measured
    # blink-back fraction 0.418); start-epoch = the terminal band anchor (the same
    # ep726 placement provenance as muon/pose-finish — the trunk must be formed first).
    "--seg-phase-advect-weight": 0.4,
    "--seg-phase-advect-start-epoch": 726,
    # Active structured-core prefit and generated/table pose-carrier companion
    # values. These must travel through the typed DSL with the active switches;
    # trainer-side argparse defaults are not a scientific configuration owner.
    "--structured-init-include-lane": True,
    "--structured-init-thresh": 0.5,
    "--structured-init-steps": 600,
    "--structured-init-lr": 5e-3,
    "--structured-init-subsample": 8192,
    "--structured-init-sdf-clip": 20.0,
    "--pose-carrier-residual-scale": 1.0,
    "--pose-carrier-s-t": 0.044,
    "--pose-carrier-s-r": 0.0,
    "--pose-carrier-pitch": 0.0,
    # State-gate sensor companions. The MLX trainer parser historically supplied
    # these defaults, but V9 is a typed program: the compiled argv must carry the
    # actual sensor geometry/cadence so every backend consumes one authority.
    "--annulus-telemetry": True,
    "--annulus-band": 2.0,
    "--annulus-bottom-k": 0.05,
    "--annulus-plateau-rel-eps": 1e-4,
    "--annulus-plateau-dwell-windows": 4,
    "--annulus-plateau-min-epochs": 150,
    "--curriculum-nucleus-within-flip": 0.5,
    "--curriculum-nucleus-min-part-frac": 0.0,
    "--jacobian-basin-telemetry": True,
    "--jacobian-basin-t0": True,
    "--jacobian-basin-k-pairs": 32,
    "--jacobian-basin-every": 4,
    "--jacobian-basin-stratify-t": True,
    "--jacobian-basin-sigma-floor": 1e-4,
    "--jacobian-basin-f-basin": 1.0,
    "--jacobian-basin-quorum-q": 0.8,
    # Basis-family custody (2026-07-15 PR95/Fourier/RGB cargo-cult sweep, #508):
    # the coordinate front-end is the #1 measured lever family and must NOT ride
    # the trainer argparse default. legacy_fourier_ab_control is the DELIBERATE
    # A/B-control state per the no-Fourier-basis doctrine (genuine curvelet/
    # shearlet stay opt-in behind the c1 optimal-form receipt slot until the owed
    # n600 through-R no-regression A/B); --self-orient False makes the owed-16
    # refuted directional front-end drop explicit (--no-self-orient) instead of
    # default-inherited.
    "--basis": "legacy_fourier_ab_control",
    "--self-orient": False,
    # Explicit inactive/derived companions are still configuration: omission
    # would silently re-delegate scientific ownership to an argparse default.
    "--seed-anneal-epochs": 0,
    "--seed-anneal-shape": "linear",
    "--seed-blend": 1.0,
    "--seed-lr": 0.02,
    "--containment-mode": "shield",
    "--containment-damp": 0.1,
    "--muon-adamw-lr": 1e-4,
    "--tail-live-mq": False,
}

_V9_CGAUGE_REMOVE: tuple[str, ...] = ()

# ---------------------------------------------------------------------------
# Value-provenance manifest (rung x form x derivation edge) for every constant the
# V9 identity is ABOUT. Rungs: derived_live > derived_at_config > measured_anchor >
# hardcoded_waiver (tac.witness_dsl.lawref ladder). Forms per the master action's
# value-form vocabulary (tac.canonical_equations.cgauge_master_action_20260711).
# Constants NOT listed here inherit the sealed crucible_v752 provenance unchanged
# (SPEC_v75 / SYNTHESIS_v3 — already ladder-placed at sealing).
# ---------------------------------------------------------------------------
V9_CGAUGE_PROVENANCE: dict[str, dict[str, str]] = {
    "--annulus-telemetry": {
        "value": "True", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "required measured source for the wired annulus_plateau event; explicit rather than inherited from the MLX parser.",
    },
    "--annulus-band": {
        "value": "2", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "#333 GT-margin annulus used by the realized through-R boundary sensor and temporal screw.",
    },
    "--annulus-bottom-k": {
        "value": "0.05", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "bottom-five-percent GT-margin companion definition retained for sensor telemetry parity.",
    },
    "--annulus-plateau-rel-eps": {
        "value": "0.0001", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "event_wirings ANNULUS_PLATEAU_REL_EPS; calibrated sister of the curriculum plateau band.",
    },
    "--annulus-plateau-dwell-windows": {
        "value": "4", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "event_wirings ANNULUS_PLATEAU_DWELL_WINDOWS; four verdict points must hold the plateau.",
    },
    "--annulus-plateau-min-epochs": {
        "value": "150", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "event_wirings ANNULUS_PLATEAU_MIN_EPOCHS dwell anchor for a formed boundary.",
    },
    "--curriculum-nucleus-within-flip": {
        "value": "0.5", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "critical-nucleus formed-partition threshold consumed by lane_nucleus and birth handoff sensors.",
    },
    "--curriculum-nucleus-min-part-frac": {
        "value": "0", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "strict positive-mass critical-nucleus boundary; zero is the comparison floor, not a guessed class area.",
    },
    "--jacobian-basin-telemetry": {
        "value": "True", "rung": "derived_at_config", "form": "SCALAR",
        "law": "pose_jacobian_basin_conditioning_v1",
        "note": "required real sigma_min source for pose-finish conditioning; explicit in the typed program.",
    },
    "--jacobian-basin-t0": {
        "value": "True", "rung": "derived_at_config", "form": "SCALAR",
        "law": "pose_jacobian_basin_conditioning_v1",
        "note": "near-free render-gradient companion observer remains enabled beside the authoritative T1 probe.",
    },
    "--jacobian-basin-k-pairs": {
        "value": "32", "rung": "derived_at_config", "form": "SCALAR",
        "law": "pose_jacobian_basin_conditioning_v1",
        "note": "canonical deterministic motion-stratified T1 subsample size.",
    },
    "--jacobian-basin-every": {
        "value": "4", "rung": "derived_at_config", "form": "SCALAR",
        "law": "pose_jacobian_basin_conditioning_v1",
        "note": "canonical cadence of one T1 conditioning probe per four sensor verdicts.",
    },
    "--jacobian-basin-stratify-t": {
        "value": "True", "rung": "derived_at_config", "form": "SCALAR",
        "law": "pose_jacobian_basin_conditioning_v1",
        "note": "motion stratification retains the low-motion conditioning tail in the real 6x6 probe.",
    },
    "--jacobian-basin-sigma-floor": {
        "value": "0.0001", "rung": "measured_anchor", "form": "SCALAR",
        "law": "pose_jacobian_basin_conditioning_v1",
        "note": "canonical basin quorum floor for per-pair sigma_min.",
    },
    "--jacobian-basin-f-basin": {
        "value": "1", "rung": "derived_at_config", "form": "SCALAR",
        "law": "pose_jacobian_basin_conditioning_v1",
        "note": "terminal-policy conditioning fraction; the rolling sigma plateau remains the firing authority.",
    },
    "--jacobian-basin-quorum-q": {
        "value": "0.8", "rung": "measured_anchor", "form": "SCALAR",
        "law": "pose_jacobian_basin_conditioning_v1",
        "note": "canonical fraction of sampled pairs required above the conditioning floor.",
    },
    "--seed-anneal-epochs": {
        "value": "0", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "V9 keeps the seed compose weight constant while witness-alone formation and birth-completion provide the absorption handoff.",
    },
    "--seed-anneal-shape": {
        "value": "linear", "rung": "derived_at_config", "form": "LINEAR",
        "law": "cgauge_master_action_v1",
        "note": "explicit dormant shape companion; it becomes active only if seed-anneal-epochs is amended above zero.",
    },
    "--seed-blend": {
        "value": "1", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "full GT-island appearance initialization for the training-only protected residual seed.",
    },
    "--seed-lr": {
        "value": "0.02", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "dedicated protected-seed AdamW learning-rate anchor; separate from witness optimizer geometry.",
    },
    "--containment-mode": {
        "value": "shield", "rung": "derived_at_config", "form": "POLYTOPE_KKT",
        "law": "cgauge_master_action_v1",
        "note": "canonical protected-gradient projection that prevents the bulk gradient from washing out the island seed.",
    },
    "--containment-damp": {
        "value": "0.1", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "explicit dormant damp-mode companion; shield mode is active in V9.",
    },
    "--muon-adamw-lr": {
        "value": "0.0001", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "derived fallback-group rate 0.1 times the typed base lr 0.001 during the Muon stage.",
    },
    "--tail-live-mq": {
        "value": "False", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "explicitly selects the sealed tau-halving fallback; compile emits the existing parser's negative BooleanOptionalAction form.",
    },
    "--structured-init-steps": {
        "value": "600", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "short structured-core prefit anchor from the wired MLX authority; "
                "now emitted with the active switch so Torch cannot inherit a parser default.",
    },
    "--structured-init-lr": {
        "value": "0.005", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "wired structured-core AdamW prefit anchor; scientific ownership is the V9 DSL.",
    },
    "--structured-init-subsample": {
        "value": "8192", "rung": "hardcoded_waiver", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "bounded-memory prefit sample count carried from the wired authority; no score claim.",
    },
    "--structured-init-sdf-clip": {
        "value": "20", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "structured signed-distance target clip used by the validated prefit path.",
    },
    "--structured-init-thresh": {
        "value": "0.5", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "majority threshold defining the shared static-core partition.",
    },
    "--structured-init-include-lane": {
        "value": "True", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "explicit structured-core role selection; emitted to prevent parser-default drift.",
    },
    "--pose-carrier-residual-scale": {
        "value": "1", "rung": "derived_at_config", "form": "SCALAR",
        "law": "store_nothing_pose_carrier_rate_collapse_vs_dpose_v1",
        "note": "identity scaling for the table dxi residual; the trainable coordinate is unwarped.",
    },
    "--pose-carrier-s-t": {
        "value": "0.044", "rung": "measured_anchor", "form": "SCALAR",
        "law": "store_nothing_pose_carrier_rate_collapse_vs_dpose_v1",
        "note": "n600 first-24 self-fit selected 0.044 (stepping_instability_diagnostic_"
                "20260705); matches the canonical generated/store-nothing xi source in "
                "store_nothing_pose_carrier_rate_dpose_20260702.py. The broader 0.16 "
                "full-ground pre-residual optimum is a different authority surface.",
    },
    "--pose-carrier-s-r": {
        "value": "0", "rung": "measured_anchor", "form": "SCALAR",
        "law": "store_nothing_pose_carrier_rate_collapse_vs_dpose_v1",
        "note": "generated/store-nothing carrier rotation calibration anchor.",
    },
    "--pose-carrier-pitch": {
        "value": "0", "rung": "measured_anchor", "form": "SCALAR",
        "law": "store_nothing_pose_carrier_rate_collapse_vs_dpose_v1",
        "note": "EON generated-carrier pitch anchor.",
    },
    "--mod-dim": {
        "value": "32", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_whitney_moddim_v1",
        "note": "DERIVED target = 19 (Whitney 2*8+1 + 2 gauge margin); 32 is the SAFE "
                "sealed anchor. GATED-ON-HARVEST: adopt 19 iff the #299 Arm-A / "
                "terminal-band eff_rank read shows d_seg-neutrality (live ep125: "
                "eff_rank 16.4 rising; ~10/32 dims ablation-neutral-or-harmful).",
    },
    "--hidden-dim": {
        "value": "96", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_whitney_moddim_v1",
        "note": "sized to rank-8 + gauge margin per the naming-memo spec; NO texture-"
                "trunk hidden budget on either axis (capacity wasted where d_cov~0).",
    },
    "--max-bank-freq": {
        "value": "64", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_nyquist_bank_frequency_v1",
        "note": "inside the through-R usable band (ceiling 128, near-all-pass <=64); "
                "along-tangent allocation sqrt(64)=8 is parabolic-C2-optimal "
                "(cgauge_curvelet_parabolic_bank_v1); the dash comb is a C2 violation "
                "cured by the class-targeted comb term, not a bank re-scale.",
    },
    "--lane-band-dash-comb": {
        "value": "True", "rung": "derived_at_config", "form": "POLYNOMIAL",
        "law": "cgauge_curvelet_parabolic_bank_v1",
        "note": "the DEDICATED C2-violation term the parabolic law forces (openpilot "
                "lane polynomial + comb phase = a polynomial-form value).",
    },
    "--basis": {
        "value": "legacy_fourier_ab_control", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "EXPLICIT basis-family custody (#508 cargo-cult sweep 2026-07-15): the "
                "legacy directional-Fourier bank stays the deliberate A/B CONTROL per the "
                "no-Fourier-basis doctrine — a TRACKED, REASONED state, never a silent "
                "argparse default and never a silent curvelet flip; windowed_curvelet/"
                "compact_shearlet fold only via the c1 optimal-form receipt slot after "
                "the owed n600 through-R no-regression A/B.",
    },
    "--self-orient": {
        "value": "False", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_master_action_v1",
        "note": "the owed-16 A/B refuted the self-orient directional front-end on this "
                "trunk (the v752(self_orient=False) identity); emitted explicitly as "
                "--no-self-orient so the drop is typed custody, not an argparse-default "
                "inheritance. NB the -48% label was NEVER a curvelet frame (NO-FAKE "
                "audit 2026-07-14): it is a self-oriented directional-FOURIER feature.",
    },
    "--adam-beta2": {
        "value": "0.999", "rung": "measured_anchor", "form": "SCALAR",
        "law": "cgauge_beta2_window_v1",
        "note": "inside the DERIVED admissible window [0.9867, 0.9996] at S=75, "
                "T_c=100ep; the T0 candidate 0.9999999 is derived-REJECTED; the point "
                "value inside the window stays the #222 A/B's to move.",
    },
    "--seg-phase-advect-weight": {
        "value": "0.4", "rung": "derived_at_config", "form": "SCALAR",
        "law": "gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1",
        "note": "theory-FORCED term (no smooth witness pierces the 0.005318 floor); "
                "weight = Law-5 PHASE_WEIGHT_FRACTION_OF_SUBPIX = 0.4 (from the "
                "measured blink-back fraction 0.418). SEAL + n600 A/B owed pre-launch.",
    },
    "--seg-phase-advect-start-epoch": {
        "value": "726", "rung": "measured_anchor", "form": "SCALAR",
        "law": "dseg_covariant_gauge_decomposition_v1",
        "note": "terminal-band placement (needs a formed trunk; same anchor as muon/"
                "pose-finish). DERIVED-LIVE form owed: the label_floor detector event "
                "(engage when the run reaches the floor) — BUILD-OWED as a trainer "
                "event hook; until then the epoch anchor is the static approximation.",
    },
    "--eikonal-weight": {
        "value": "0.01", "rung": "measured_anchor", "form": "SCALAR",
        "law": "adaptive_eps_cfl_edge_tracking_v1",
        "note": "the L13 stable value; the CFL-edge adaptive CURE is FALSIFIED_MECHANISM "
                "at n600 (FEED-06g) so the weight stays a measured anchor, NOT a DE.",
    },
    "--lr": {
        "value": "1e-3 cosine->1e-4", "rung": "measured_anchor", "form": "ODE",
        "law": "cgauge_master_action_v1",
        "note": "under the SATURATED_ALWAYS_CLIPS regime (live harvest 3f) lr x clip is "
                "the SOLE magnitude control — an ODE-form value (cosine flow); loss "
                "weights are direction-only controls.",
    },
    "--tail-tau-halving": {
        "value": "0.5", "rung": "derived_at_config", "form": "SELF_DERIVING",
        "law": "tau_eps_hbar_one_dequantization_two_scales_v1",
        "note": "the tau tail is costate-controlled (tail-stop-marginal-s = a marginal-"
                "DS readout = the canonical SELF_DERIVING value).",
    },
    "--pose-carrier": {
        "value": "True (dxi 6+k)", "rung": "derived_at_config", "form": "POLYTOPE_KKT",
        "law": "posenet_luma_chroma_sensitivity_asymmetry_v1",
        "note": "pose's +2.1-2.4 partition-invisible DOF route through the dedicated "
                "dxi/steering channel (the banked R1 archive shape); the terminal "
                "pose-finish is an output-space SOLVE surface (S1/S2), not a trunk knob.",
    },
}


def derive_v9_cgauge_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = "experiments/results/__v9_cgauge__",
):
    """Build the V9·CGauge :class:`TypedWitnessConfig` — v752(self_orient=False) + the
    theory-forced V9 delta. Fail-CLOSED: refuses a config the DSL cannot validate.

    $0 / pure: never launches (LAUNCH = operator-GO, CONTAINMENT).
    """
    from tac.witness_autoconfig import derive_crucible_v752_config

    v752 = derive_crucible_v752_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir,
        self_orient=False,
    )
    merged = {**v752.base, **_V9_CGAUGE_DELTA}
    for rm in _V9_CGAUGE_REMOVE:
        merged.pop(rm, None)
    purpose = (
        "V9 CGauge (Covariant-Gauge witness; vehicle_v9_cgauge_naming_20260711): the "
        "v7.5.2 single covariant trunk (texture trunk DROPPED both axes) + sealed "
        "launch-1 levers, every constant ladder-placed against cgauge_master_action_v1 "
        "+ the #223 sizing laws, + the theory-forced T1 appearance-phase term armed for "
        "the terminal band (flicker-floor law: no smooth witness pierces 0.005318; the "
        "need is 4.5-7x below). GATES BEFORE LAUNCH (operator-GO): T1 SEAL + n600 A/B "
        "owed (L86); mod-dim 32->19 gated on #299 Arm-A/terminal eff_rank; lane-repair "
        "reversal falsification pre-registered at annulus_plateau engage (harvest 3b). "
        "MEANS until a byte-closed n600 exact row < 0.19108282."
    )
    typed = v752.model_copy(update={
        "name": "v9_cgauge",
        "base": merged,
        "purpose": purpose,
    })
    viol = typed.validate_program()
    if viol:
        raise ValueError(
            f"v9_cgauge DSL-authored-config gate: {len(viol)} WitnessProgram.validate "
            f"violation(s): {viol[:4]}")
    return typed


def compile_v9_cgauge_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = "experiments/results/__v9_cgauge__",
) -> tuple:
    """Compile V9·CGauge: (typed_config, argv_tuple). Pure / $0 — no dispatch."""
    typed = derive_v9_cgauge_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir)
    argv = typed.to_program().compile_trainer_argv()
    return typed, argv


# ===========================================================================
# Task #432 — the COHERENT STATE-GATED-SCHEDULE ARM (V9·CGauge as design spec
# and optimal config), materialized as a LAUNCH-READY typed DSL program.
#
# #430 (scorer_model_arms_430_schedule_20260711.md) measured that the organ's
# state-gated cascade (island-birth → boundary-form → τ-sharpen⊕repair → finish)
# beats the hand-scheduled #205 shares on all three replay models, and its
# OperatorGoTicket named "the witness-DSL compile of the bundle" as gates_owed.
# THIS is that compile: the cascade realized in the REAL trainer flag space
# (never-invent-flags) on the V9·CGauge base, with every un-realizable bundle
# element explicitly dispositioned (organ-advisory or BUILD-OWED) — never
# silently dropped.
#
# CONTAINMENT: builds + validates + compiles only. LAUNCH = operator-GO.
# ===========================================================================

# The #432 delta over the v9_cgauge base (each row justified in
# V9_CGAUGE_432_PROVENANCE; flags verified against the live trainer parser).
_V9_CGAUGE_432_DELTA: dict[str, Any] = {
    # V9·CGauge parametrization (operator-directed intended change, task #432 +
    # vehicle_v9_cgauge_naming_20260711 §parametrization): mod-dim 32 -> 19 =
    # Whitney 2*8+1 = 17 + 2 gauge margin (cgauge_whitney_moddim_v1 evaluator -> 19).
    "--mod-dim": 19,
}

V9_CGAUGE_432_PROVENANCE: dict[str, dict[str, str]] = {
    "--mod-dim": {
        "value": "19", "rung": "derived_at_config", "form": "SCALAR",
        "law": "cgauge_whitney_moddim_v1",
        "note": "task #432 operator-directed adoption of the DERIVED Whitney value "
                "(2*8+1 = 17 + 2 gauge margin = 19; evaluator-executable). Supports: "
                "live #205 harvest eff_rank 16.4 (rising) < 19 with ~10/32 dims "
                "ablation-neutral-or-harmful (32 demonstrably over-parametrized); "
                "UU-2: mod-17-19 valid for SDF-like charts, which THIS trunk is; "
                "rate: latent table shrinks ~40% (D18 ~7KB class). RISK bounded by "
                "the pre-registered #299 Arm-A rule: if the arm shows "
                "d_seg-residual > +2% vs the mod-32 control, revert to 32 (the #205 "
                "banked run IS the mod-32 control). NOTE this arm therefore doubles "
                "as #299 Arm-A on the SPEC_v9 base, exactly as the master-action §6 "
                "redesign requires (arms must run WITH the phase machinery active).",
    },
    "--seg-phase-advect-weight": {
        "value": "0.4 (composed as the PhaseAdvectionConsistency LEVER)",
        "rung": "derived_at_config", "form": "PDE_FIELD",
        "law": "gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1",
        "note": "the V9 theory-forced T1 term, composed at the LAUNCH wrapper as a "
                "DSL Lever (triality: a lever is not built until it is a Lever "
                "factory AND composed) so it rides the expected-active-levers "
                "manifest — built-but-not-composed extinction (advisory P0-1).",
    },
    "schedule": {
        "value": "state-gated cascade (see V9_CGAUGE_432_CASCADE_REALIZATION)",
        "rung": "derived_at_config", "form": "SELF_DERIVING",
        "law": "costate_lambda_marginal_ds_v1",
        "note": "#430 coherent schedule: gates fire on wired trainer sensors "
                "(lane_nucleus / annulus_plateau / powerlaw_meat / sigma_min_plateau "
                "/ tau-event / birth-completion) with epoch values demoted to "
                "fail-safe BACKSTOP caps — the trainer-native realization of the "
                "measured-winning selective-intervention shape (2607.08716 + the "
                "#430 backtest, −27.4% ∫d_seg·dep on the WF-winning replay model).",
    },
}

# The #430 CASCADE_STAGES → trainer-flag realization map (the witness-DSL compile
# of the organ's bundle — the gates_owed item from the #430 OperatorGoTicket).
# Stage names MUST mirror tac.witness_control.schedule_backtest.CASCADE_STAGES
# (cross-checked by test). Each bundle lever is either REALIZED (the trainer flag
# set that fires it on a state sensor), ALWAYS-ON (no gating needed), or honestly
# dispositioned ORGAN-ADVISORY / BUILD-OWED (no trainer flag exists;
# never-invent-flags forbids a stub).
V9_CGAUGE_432_CASCADE_REALIZATION: dict[str, dict] = {
    "island_birth": {
        "gate_realization": "active from ep0 (Movable unborn at init by construction); "
                            "stage EXIT is state-gated: --birth-completion-event "
                            "(tau_persist 0.8, area band 0.25, ramp-down to 0.2)",
        "bundle": {
            "island_amplify": "REALIZED: --seed-islands + --amplify-* + LADDER homotopy "
                              "(--ladder-* per-class birth/anneal; λ-gates 0.0 = "
                              "UNGATED — a nonzero per-class λ floor has NO measured "
                              "derivation yet; setting one would be a guess)",
            "area_constraint": "REALIZED: --area-constraint-birth (classes 1,3)",
            "persistence": "REALIZED: --persistence-* (warmup 275) + cldice",
        },
    },
    "boundary_form": {
        "gate_realization": "unify-τ continuous flow (--seg-form-unify-tau) + "
                            "--tau-advance-mode event (τ advances on the relaxation "
                            "sensor, min-dwell floor 250 = waived backstop)",
        "bundle": {
            "seg": "ALWAYS-ON (w_seg 100, the score law)",
            "eikonal": "ALWAYS-ON small+flat (0.01; CFL-adaptive cure "
                       "FALSIFIED_MECHANISM at n600 — FEED-06g)",
        },
    },
    "sharpen_repair": {
        "gate_realization": "wired state sensors: lane_nucleus (lane-band), "
                            "annulus_plateau (chroma boundary + temporal screw); "
                            "epoch values are BACKSTOP caps only",
        "bundle": {
            "chroma_boundary": "REALIZED: --seg-chroma-boundary-start-event annulus_plateau",
            "lane_edge": "REALIZED: --lane-band-start-event lane_nucleus (+ dash-comb)",
            "thin_lane": "REALIZED via LADDER lane homotopy + persistence recall "
                         "(inverse_thickness amplify-persist)",
            "subpix": "REALIZED: #360 temporal-screw (--seg-temporal-screw-start-event "
                      "annulus_plateau) + T1 phase-advection (ep726 static approx; "
                      "event form = N7 BUILD-OWED: label_floor detector trainer hook)",
            "margin_saliency": "ORGAN-ADVISORY: msal_uni measured INERT as texture "
                               "proxy (L76); exact through-R S_R reachability is the "
                               "registered ladder rung (never-fired, duty-queued) — "
                               "NOT composed here",
        },
    },
    "finish": {
        "gate_realization": "muon on powerlaw_meat (backstop 726) + pose-finish on "
                            "sigma_min_plateau (#383 gate) + tail τ-halving with "
                            "tail-stop-marginal-s (a costate readout, SELF_DERIVING)",
        "bundle": {
            "weight_entropy": "ALWAYS-ON (λ=15, measured BINDING ~15% direction "
                              "share, 91→83KB): NO trainer start-event flag exists "
                              "for plateau-gating it; gating late would also change "
                              "the measured rate behavior — dispositioned ALWAYS-ON "
                              "with the plateau-gate variant ORGAN-ADVISORY",
        },
    },
    "_organ_advisory": {
        "note": "the cascade's per-class-λ BUDGET SHIFTS (the #430 _shift_toward "
                "mechanism, ≤10% share mass onto the gated bundle's best-λ lever) "
                "are per-verdict loss-weight mutations — FORBIDDEN in-run (loss "
                "weights at stage boundaries only; live-config mutation = "
                "operator-GO). They remain the ORGAN'S advisory loop: the "
                "score-neutral shadow observer auto-starts with every governed "
                "launch; its recommendations are OperatorGoTickets.",
    },
}

# Expected-active-lever manifest for the #432 launch config (advisory P0-1 gate):
# the v752 launch set + the T1 phase lever. Amend ONLY via reviewed amendment.
V9_CGAUGE_432_EXPECTED_LEVERS: tuple[str, ...] = (
    "seg_form_unify_tau", "tail_k_warm_restart", "n323_ladder_island_homotopy",
    "R7_polyak_finisher", "v75_area_constraint_birth", "v75_birth_completion_event",
    "n287_dash_comb", "temporal_screw_consistency",
    "pose_finish_conditioning_gate",
    "phase_advection_consistency",
)


def derive_v9_cgauge_432_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = "experiments/results/__v9_cgauge_432__",
):
    """Build the task-#432 coherent-arm :class:`TypedWitnessConfig` — the v9_cgauge
    base + the #432 delta (mod-dim 19). Fail-CLOSED on DSL validation.

    $0 / pure: never launches (LAUNCH = operator-GO, CONTAINMENT).
    """
    typed = derive_v9_cgauge_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir)
    merged = {**typed.base, **_V9_CGAUGE_432_DELTA}
    purpose = (
        "task #432 V9 CGauge COHERENT STATE-GATED-SCHEDULE ARM (the #430 bundle's "
        "witness-DSL compile; vehicle_v9_cgauge_naming_20260711 as design spec): the "
        "v9_cgauge single covariant trunk + mod-dim 19 (Whitney 17 + 2 gauge margin, "
        "cgauge_whitney_moddim_v1; operator-directed intended change — this arm "
        "doubles as #299 Arm-A on the SPEC_v9 base, phase machinery active per the "
        "master-action §6 redesign) + T1 phase-advection composed as a LEVER (0.4 @ "
        "ep726 static approx; label_floor event = N7 BUILD-OWED) + the state-gated "
        "cascade realized on wired trainer sensors (lane_nucleus / annulus_plateau / "
        "powerlaw_meat / sigma_min_plateau / tau-event / birth-completion; epoch "
        "values = fail-safe backstops). Per-class-lambda budget shifts remain "
        "ORGAN-ADVISORY (no in-run config mutation; operator-GO). FRESH start "
        "(mod-19 cannot warm-start mod-32 checkpoints); resumable-from-disk with "
        "per-stage checkpoints (--ckpt-every 25 --stage-checkpoints). CONTROL = the "
        "#205 banked mod-32 baseline (ep0-225). GATES BEFORE LAUNCH (operator-GO): "
        "T1 SEAL + n600 A/B owed (L86); the lane-reversal falsification stands "
        "pre-registered at annulus_plateau engage. MEANS until a byte-closed n600 "
        "exact row < 0.19108282."
    )
    typed = typed.model_copy(update={
        "name": "v9_cgauge_432",
        "base": merged,
        "purpose": purpose,
    })
    viol = typed.validate_program()
    if viol:
        raise ValueError(
            f"v9_cgauge_432 DSL-authored-config gate: {len(viol)} WitnessProgram.validate "
            f"violation(s): {viol[:4]}")
    return typed


def compile_v9_cgauge_432_launch_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = "experiments/results/__v9_cgauge_432__",
):
    """The launcher-facing #432 cfg — the ONE object satisfying the duck-typed cfg
    protocol ``tools/launch_witness_run.py`` consumes (mirrors
    ``compile_crucible_v752_launch_config``): composes the #383 pose conditioning
    gate + the T1 :func:`~tac.witness_dsl.curriculum_dsl.PhaseAdvectionConsistency`
    lever + the amber stability values (the #205 launch's SPEC §1.1 set, HELD so the
    arm's only deltas are the intended ones), enforces the expected-active-lever
    manifest fail-closed, and reuses crucible_v7's constants/governance manifests
    (v9_cgauge_432 changes only base flags + levers + purpose — never LawRef
    constants nor schedule-WHEN governance tokens).

    CONTAINMENT: pure / $0 — returns the config; it never launches.
    means != ends: only a byte-closed n600 exact row < 0.19108282 moves the pointer.
    """
    from tac.witness_autoconfig import (
        CrucibleV7LaunchConfig,
        _crucible_v7_argv_pairs,
        compile_crucible_v7_config,
    )
    from tac.witness_dsl.curriculum_dsl import (
        PhaseAdvectionConsistency,
        PoseFinishConditioningGate,
    )
    from tac.witness_dsl.typed_config import TypedLever, build_launch_manifest
    from tac.witness_stability import AMBER as _AMBER

    typed = derive_v9_cgauge_432_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir)
    _gate = PoseFinishConditioningGate()
    # T1 as a composed LEVER (weight/start match the base flags the v9 delta already
    # emits — the lever is the triality-visible owner; overrides win on compile and
    # are value-identical, so composition cannot drift the argv).
    _t1 = PhaseAdvectionConsistency(weight=0.4, start_epoch=726)
    _updates: dict = {
        "levers": (*tuple(typed.levers), TypedLever(name=_gate.name, overrides=dict(_gate.overrides), epochs_delta=_gate.epochs_delta, notes=_gate.notes), TypedLever(name=_t1.name, overrides=dict(_t1.overrides), epochs_delta=_t1.epochs_delta, notes=_t1.notes)),
        # amber HELD from the #205 launch (SPEC §1.1 explicit values — the arm must
        # not silently differ from the control on stability):
        "base": {**typed.base,
                 "--grad-clip": float(_AMBER.grad_clip),
                 "--pose-grad-coeff-max": float(_AMBER.pose_grad_coeff_max),
                 "--grad-normalize": "per-param"},
    }
    typed = typed.model_copy(update=_updates)
    viol = typed.validate_program()
    if viol:
        raise ValueError(
            f"v9_cgauge_432 launch-config gate: lever composition produced "
            f"{len(viol)} WitnessProgram.validate violation(s): {viol[:4]}")
    got = tuple(lv.name for lv in typed.levers)
    if sorted(got) != sorted(V9_CGAUGE_432_EXPECTED_LEVERS):
        raise ValueError(
            "v9_cgauge_432 expected-active-lever gate: composed lever set "
            f"{sorted(got)} != expected {sorted(V9_CGAUGE_432_EXPECTED_LEVERS)} "
            "(built-but-not-composed / silently-dropped lever — advisory P0-1). "
            "Update V9_CGAUGE_432_EXPECTED_LEVERS ONLY via a reviewed amendment.")
    argv = typed.to_program().compile_trainer_argv()
    emitted_names = sorted({f for f, _ in _crucible_v7_argv_pairs(argv)})
    dsl_manifest = build_launch_manifest(
        program_name="v9_cgauge_432", emitted_flag_names=emitted_names,
        typed_config_hash=typed.typed_config_hash(), typed_validated=True)
    dsl_manifest["expected_active_levers"] = list(V9_CGAUGE_432_EXPECTED_LEVERS)
    dsl_manifest["expected_stability"] = {
        "grad_clip": float(typed.base["--grad-clip"]),
        "pose_grad_coeff_max": float(typed.base["--pose-grad-coeff-max"]),
        "grad_normalize": str(typed.base["--grad-normalize"]),
        "per_group_grad_clip": True,
        "composed_as": "explicit-values (HELD from the #205 amber launch; no preset)",
    }
    v7_compiled = compile_crucible_v7_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir)
    constants = dict(v7_compiled.constants_manifest)
    # T1 engage epoch = DERIVED-AT-CONFIG (schedule-provenance gate class DERIVED — the
    # honest form: the trainer wires T1 as a pure epoch gate, so CAP-with-sensor would
    # LAUNDER a sensor that does not govern it; the label_floor event trainer hook is
    # N7 BUILD-OWED). Derivation: the flicker-floor law FORCES the phase term as a
    # terminal-band (floor->phase-tail, Law-5) treatment on a FORMED trunk; the
    # terminal band's registered anchor is the muon cap (726 = nu-law settle + floor
    # derivation, the same co-anchor pose-finish uses) => T1 start := muon cap.
    _muon_cap = int(typed.base.get("--muon-start-epoch", 726))
    _pa_start = int(typed.base.get("--seg-phase-advect-start-epoch", 0))
    if _pa_start != _muon_cap:
        raise ValueError(
            f"v9_cgauge_432 T1-placement gate: emitted --seg-phase-advect-start-epoch "
            f"{_pa_start} != the muon terminal-band co-anchor {_muon_cap} — the DERIVED "
            "constants-manifest entry would lie about the emitted value. Re-derive both "
            "from the same anchor (never patch one side).")
    constants["seg_phase_advect_start_epoch"] = {
        "value": _muon_cap,
        "equation_id": "gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1",
        "ladder_class": "derived_at_config",
        "fallback_used": False,
        "inputs": {
            "terminal_band_anchor_muon_cap": _muon_cap,
            "placement_law": "Law-5 floor->phase-tail (flicker design memo §curriculum)",
        },
        "note": (
            "T1 phase-advection engage epoch := the terminal-band anchor (muon cap; the "
            "phase term needs a formed trunk and treats the post-floor tail). STATIC "
            "APPROXIMATION of the label_floor detector event (N7 BUILD-OWED: "
            "--seg-phase-advect-start-event trainer hook); until built, the value is "
            "compiled from the flicker-floor law's placement + the muon-cap co-anchor, "
            "never hand-typed independently."),
    }
    return CrucibleV7LaunchConfig(
        typed=typed,
        constants_manifest=constants,
        dsl_program_manifest=dict(dsl_manifest),
        schedule_governance=dict(v7_compiled.schedule_governance),
    )


# ===========================================================================
# 2026-07-13 — TRULY-OPTIMAL EVENT-NATIVE V9 + MATCHED MOD19/MOD32 FAMILY A/B
# ===========================================================================

V9_CGAUGE_IDEAL_EXPECTED_ADDITIONS: tuple[str, ...] = (
    # The v7.5.2 taper flags historically lived anonymously in ``base``.  V9
    # isolation requires a real Lever owner + LawRefs so TAPER-off can remove
    # the whole scientific declaration instead of emitting a forbidden zero.
    "dseg_aware_taper",
    "unified_tau_eikonal_hold",
    "n292_closed_loop_eikonal_control",
    "R7_beta2_window_rewarmup",
    "FEED_08a_length_sigma",
    "tie_locus_displacement",
    "margin_band_satisficing",
    # C1 control leg: the S_R source selector is meaningful only when the
    # margin-saliency loss itself is active.  Keep this in every ideal arm so
    # the named S_R treatment differs from its control by exactly one Boolean.
    "margin_saliency",
)

V9_CGAUGE_IDEAL_SR_EXPECTED_ADDITION = "lever4_margin_saliency_reachability"
V9_CGAUGE_IDEAL_SR_EQUATION_ID = "margin_saliency_reachability_replaces_texture_proxy_v1"

V9_CGAUGE_IDEAL_EXCLUSIONS: dict[str, str] = {
    # margin_band_satisficing REMOVED from exclusions 2026-07-13: sibling provenance fix landed
    # (a79f5d68cd; m_safe DERIVED via LawRef + fail-closed) → now composed ON in core + both A/B arms.
    "step_native_endpoint": "ISOLATE: endpoint/activation basin; never stack into the matched core A/B",
    "fresh_finer_start": "ISOLATE: fresh-start representation basin; incompatible with this trunk control",
    "ot_head_offset": "EXCLUDE: matched measurement was +5.1% worse; formulation remains isolated",
    "eikonal_viscosity": "ISOLATE: can reproduce thin-lane smoothing pressure; test only after proactive hold",
    "horizon_weighted_margin": "ISOLATE: favorable geometry but its treatment weight lacks V9 custody",
    "aa_coverage": "ISOLATE: 4x supersample compute treatment; not part of the family-dimension test",
    "hardness_oversample": "ISOLATE: requires equal-step accounting before score attribution",
    "code_spectral_entropy": "EXCLUDE: assumed weight and no matched V9 score/rate receipt",
    "dense_phase_carrier": "ISOLATE at byte-close: n600 rate/score A/B owed; not a trainer-core flag",
}


def _typed_ideal_lever(lever):
    """Lossless curriculum-DSL Lever -> typed-config adapter."""
    from tac.witness_dsl.typed_config import TypedLever

    # Canonical symmetric codec (tac.witness_dsl.lawref) — the same declarations are
    # decoded back to LawRef objects by TypedLever.to_dsl() on the deserialized path,
    # so the dsl_compile_hash self-recompile sees identical LawRef custody.
    from tac.witness_dsl.lawref import lawref_to_declaration

    declarations = {
        flag: lawref_to_declaration(ref) for flag, ref in lever.lawrefs.items()
    }

    return TypedLever(
        name=lever.name,
        overrides=dict(lever.overrides),
        epochs_delta=int(lever.epochs_delta),
        notes=str(lever.notes),
        lawrefs=dict(lever.lawrefs),
        lawref_declarations=declarations,
        constant_manifest=dict(lever.constant_manifest),
        runtime_receipt_schemas=dict(lever.runtime_receipt_schemas),
    )


def _derive_manifest_from_emitted_argv(constants: dict, argv: tuple[str, ...]) -> dict:
    """Make emitted argv the single scalar-value owner, then fail closed on drift.

    The inherited V7 manifest recorded ``hosc_beta_end=10`` while the program emitted
    ``--hosc-beta-end 3.177``.  For every manifest key with an identically named emitted
    flag, this helper derives the manifest value from the compiled DSL argv and immediately
    re-checks identity.  This is deliberately applied only to the new ideal configs so the
    review-gated historical compiler remains untouched.
    """
    from tac.witness_autoconfig import _crucible_v7_argv_pairs

    emitted = dict(_crucible_v7_argv_pairs(argv))
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in constants.items()}
    for key, rec in out.items():
        if not isinstance(rec, dict) or "value" not in rec:
            continue
        flag = "--" + str(key).replace("_", "-")
        if flag not in emitted:
            continue
        raw = emitted[flag]
        old = rec["value"]
        if raw is None:
            value = True
        elif isinstance(old, bool):
            value = str(raw).lower() not in ("0", "false", "no")
        elif isinstance(old, int) and not isinstance(old, bool):
            value = int(raw)
        elif isinstance(old, float):
            value = float(raw)
        else:
            value = str(raw)
        rec["value"] = value
        rec["single_value_owner"] = "compiled_dsl_argv"
        rec["inherited_manifest_value_replaced"] = old
        check = rec["value"]
        if check != value:
            raise ValueError(
                f"manifest parity REFUSE: {key} records {check!r}, emitted {flag}={value!r}")
    return out


def _merge_lever_constant_manifests(constants: dict, levers: tuple) -> dict:
    """Merge eager LawRef rows into launcher constants under canonical keys."""
    out = {key: (dict(value) if isinstance(value, dict) else value)
           for key, value in constants.items()}
    for lever in levers:
        for flag, record in lever.constant_manifest.items():
            key = flag.removeprefix("--").replace("-", "_")
            # A treatment replaces an inherited scalar only through this
            # explicit owning Lever record (STEP's beta_end=8 is the case at
            # hand); bare argv never reaches this merge surface.
            out[key] = dict(record)
            out[key]["single_value_owner"] = f"dsl_lever:{lever.name}"
            out[key]["emitted_flag"] = flag
    return out


def compile_v9_cgauge_ideal_launch_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = "experiments/results/v9_cgauge_truly_optimal_core_20260713",
    mod_dim: int = 19,
    program_name: str = "v9_cgauge_truly_optimal_core",
    with_reachability: bool = False,
):
    """Compile the held event-native V9 core or either matched family A/B arm.

    Pure construction only: this function validates and compiles a WitnessProgram; it
    cannot launch.  ``mod_dim`` is restricted to the pre-registered 19-vs-32 family test.
    The two family A/B programs are byte-identical in scientific argv except
    ``--mod-dim`` (their ``--out-dir`` custody paths necessarily differ).
    ``with_reachability`` selects the C1 treatment over the matched weighted-
    margin-saliency control; its only scientific argv delta is the DSL-owned
    ``--margin-saliency-reachability`` Boolean.
    """
    if int(mod_dim) not in (19, 32):
        raise ValueError(f"ideal V9 family A/B admits mod_dim 19 or 32 only, got {mod_dim!r}")

    from tac.witness_autoconfig import CrucibleV7LaunchConfig, _crucible_v7_argv_pairs
    from tac.witness_dsl.curriculum_dsl import (
        Beta2WindowRewarmup,
        ClosedLoopEikonalControl,
        DsegAwareTaper,
        LengthSigma,
        Lever,
        MarginBandSatisficing,
        MarginSaliency,
        MarginSaliencyReachability,
        TieLocusDisplacement,
    )
    from tac.witness_dsl.typed_config import build_launch_manifest

    wrapped = compile_v9_cgauge_432_launch_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir)
    typed = wrapped.typed

    taper = DsegAwareTaper(
        strength=1.0, scale=0.0, floor=0.05, scientific_declaration=True)
    taper_flags = frozenset(taper.overrides)
    missing_taper_flags = sorted(taper_flags - set(typed.base))
    if missing_taper_flags:
        raise ValueError(
            f"{program_name} taper ownership REFUSE: inherited base lacks {missing_taper_flags}")
    if any(typed.base[flag] != value for flag, value in taper.overrides.items()):
        raise ValueError(
            f"{program_name} taper ownership REFUSE: Lever values do not match inherited base")

    # The new trainer actuator owns λ_k = λ_0 + (λ_N-λ_0)k/N on the persisted event
    # τ rung and primes λ BEFORE assigning the lower τ.  Values are the settled V9
    # retention transition (0.01 -> 0.05), not the older F0 constant-hold rescue.
    eik_hold = Lever(
        "unified_tau_eikonal_hold",
        overrides={
            "--seg-form-unify-tau": True,
            "--tau-advance-mode": "event",
            "--eikonal-weight": 0.01,
            "--eikonal-weight-end": 0.05,
        },
        notes=("eikonal_retention_couples_to_tau_rung_v1: lambda_k=0.01+(0.05-0.01)k/N; "
               "persisted rung is the single actuator; trainer primes retention before lower tau"),
    )
    additions = (
        _typed_ideal_lever(taper),
        _typed_ideal_lever(eik_hold),
        _typed_ideal_lever(ClosedLoopEikonalControl(
            eikonal_bump=0.05, eikonal_max=0.20, max_bumps=2,
            stop_after_windows=3, min_sustained_windows=3)),
        _typed_ideal_lever(Beta2WindowRewarmup(beta2=0.999, steps_per_epoch=75)),
        _typed_ideal_lever(LengthSigma("fitted-20260707")),
        _typed_ideal_lever(TieLocusDisplacement(
            weight=0.3, start_epoch=0, v_band=1.0,
            edge_weight_source="pa_flipmass",
            edge_weight_path="reports/pa_edge_weights.json", ref_domain="seg384")),
        # MarginBandSatisficing: sibling provenance fix LANDED (a79f5d68cd) — m_safe now DERIVED via
        # LawRef (headroom*delta_R=0.03918), fail-closed invariant. Operator 2026-07-13: "belongs in the
        # ideal v9 cgauge". ON in core + BOTH A/B arms (identical → mod-dim FAMILY A/B stays unconfounded).
        _typed_ideal_lever(MarginBandSatisficing(weight=0.2)),
        # C1 matched control.  Reachability is a source multiplier inside this
        # loss and is inert when margin_saliency_weight==0; compose the existing
        # margin-saliency Lever on BOTH arms.  No event-native margin-saliency
        # engage sensor exists, so use the
        # always-on value 0 rather than laundering the factory's historical
        # naked ep900 default through a launchable config.
        _typed_ideal_lever(MarginSaliency(start_epoch=0, window=0)),
    )
    if with_reachability:
        additions += (_typed_ideal_lever(MarginSaliencyReachability(window=0)),)
    purpose = (
        "HELD operator-GO V9 CGauge event-native family A/B: form nuclei; persisted tau-rung "
        "retention prime; one-rung sharpen; actuated lane/chroma/screw plus tie-locus/anisotropic-"
        "length repair; beta2-window re-treatment; Muon only on its stable-trunk gate; terminal "
        "appearance-phase handoff. Exact n600 per-class through-R and complete stage checkpoints; "
        "MEANS only until receiver-closed exact evaluation. Fire after the 95%-kill P0."
    )
    typed = typed.model_copy(update={
        "name": str(program_name),
        "purpose": purpose,
        "base": {
            **{flag: value for flag, value in typed.base.items() if flag not in taper_flags},
            "--mod-dim": int(mod_dim),
            # The trainer refuses S_R with a larger pair micro-batch because
            # provider weights are pair-local.  Pin 1 in the shared control,
            # not only in the treatment, so C1 stays clean.
            "--micro-batch-pairs": 1,
            "--verdict-pairs": 0,
        },
        "levers": tuple(typed.levers) + additions,
    })
    violations = typed.validate_program()
    if violations:
        raise ValueError(
            f"{program_name} DSL gate: {len(violations)} WitnessProgram.validate violation(s): "
            f"{violations[:4]}")

    got = tuple(lv.name for lv in typed.levers)
    expected = tuple(V9_CGAUGE_432_EXPECTED_LEVERS) + V9_CGAUGE_IDEAL_EXPECTED_ADDITIONS
    if with_reachability:
        expected += (V9_CGAUGE_IDEAL_SR_EXPECTED_ADDITION,)
    if sorted(got) != sorted(expected):
        raise ValueError(
            f"{program_name} expected-lever REFUSE: {sorted(got)} != {sorted(expected)}")
    # MarginBandSatisficing is now REQUIRED-present (sibling provenance fix landed a79f5d68cd; m_safe
    # DERIVED + fail-closed). The expected-lever check above enforces its presence in all three programs.

    argv = tuple(typed.to_program().compile_trainer_argv())
    emitted_pairs = dict(_crucible_v7_argv_pairs(argv))
    required = {
        "--mod-dim": str(int(mod_dim)),
        "--eikonal-weight": "0.01",
        "--eikonal-weight-end": "0.05",
        "--tau-advance-mode": "event",
        "--length-sigma-matrix": "fitted-20260707",
        "--seg-subpix-edge-weight-source": "pa_flipmass",
        "--stage-transition-rewarmup-epochs": "14",
        "--micro-batch-pairs": "1",
        "--margin-saliency-weight": "1.0",
        "--margin-saliency-start-epoch": "0",
        "--margin-saliency-tau": "0.5",
        "--margin-saliency-target": "0.5",
        "--verdict-pairs": "0",
    }
    mismatches = {
        flag: (emitted_pairs.get(flag), want)
        for flag, want in required.items() if emitted_pairs.get(flag) != want
    }
    if mismatches:
        raise ValueError(f"{program_name} actuation/custody REFUSE: {mismatches}")
    for flag in ("--stage-checkpoints", "--closed-loop-control"):
        if flag not in emitted_pairs:
            raise ValueError(f"{program_name} required Boolean actuator missing: {flag}")
    reachability_emitted = "--margin-saliency-reachability" in emitted_pairs
    if reachability_emitted is not bool(with_reachability):
        raise ValueError(
            f"{program_name} S_R selector REFUSE: emitted={reachability_emitted} "
            f"but with_reachability={bool(with_reachability)}")

    manifest = build_launch_manifest(
        program_name=str(program_name),
        emitted_flag_names=sorted(emitted_pairs),
        typed_config_hash=typed.typed_config_hash(),
        typed_validated=True,
    )
    manifest.update({
        "expected_active_levers": list(expected),
        "excluded_levers": dict(V9_CGAUGE_IDEAL_EXCLUSIONS),
        "ab_decision_rule": {
            "metric": "n600 per-class through-R residual d_seg",
            "formula": "(d_seg_mod19 - d_seg_mod32) / d_seg_mod32",
            "threshold": 0.02,
            "action": "revert_to_mod32_if_residual_gt_threshold",
            "threshold_provenance": "DERIVED pre-registered cgauge_whitney_moddim_v1 family gate",
        },
        "held": True,
        "operator_go_required": True,
        "fire_after": "95%-kill P0 to avoid GPU-timing contention",
        "sr_ab_contract": {
            "control_config": "v9_cgauge_ideal_mod19",
            "treatment_config": "v9_cgauge_ideal_mod19_sR",
            "treatment": bool(with_reachability),
            "sole_scientific_argv_delta": "--margin-saliency-reachability",
            "micro_batch_pairs": 1,
            "equation_id": V9_CGAUGE_IDEAL_SR_EQUATION_ID,
            "status": "PREPARED_NOT_FIRED",
        },
    })
    constants = _derive_manifest_from_emitted_argv(wrapped.constants_manifest, argv)
    constants = _merge_lever_constant_manifests(constants, tuple(typed.to_program().levers))
    constants["eikonal_retention_tau_rung"] = {
        "value": {"base": 0.01, "end": 0.05},
        "equation_id": "eikonal_retention_couples_to_tau_rung_v1",
        "ladder_class": "derived_at_config",
        "fallback_used": False,
        "inputs": {"actuator": "persisted TauAdvanceController.rung", "fraction": "k/N"},
        "note": "retention is primed before each lower-tau model assignment",
    }
    constants["margin_saliency_reachability"] = {
        "value": bool(with_reachability),
        "equation_id": V9_CGAUGE_IDEAL_SR_EQUATION_ID,
        "ladder_class": "derived_at_config",
        "fallback_used": False,
        "inputs": {
            "control": "exp(-margin/tau)",
            "treatment_multiplier": "normalized cached through-R S_R",
            "micro_batch_pairs": 1,
        },
        "note": "C1 treatment selector; PREPARED_NOT_FIRED and no score authority",
    }
    if float(constants["hosc_beta_end"]["value"]) != float(emitted_pairs["--hosc-beta-end"]):
        raise ValueError("hosc_beta_end constants manifest does not match compiled DSL argv")
    return CrucibleV7LaunchConfig(
        typed=typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance=dict(wrapped.schedule_governance),
    )


def compile_v9_cgauge_ideal_mod19_launch_config(**kwargs):
    kwargs.setdefault("mod_dim", 19)
    kwargs.setdefault("program_name", "v9_cgauge_ideal_mod19")
    kwargs.setdefault("out_dir", "experiments/results/v9_cgauge_ideal_mod19_20260713")
    return compile_v9_cgauge_ideal_launch_config(**kwargs)


def compile_v9_cgauge_ideal_mod32_launch_config(**kwargs):
    kwargs.setdefault("mod_dim", 32)
    kwargs.setdefault("program_name", "v9_cgauge_ideal_mod32")
    kwargs.setdefault("out_dir", "experiments/results/v9_cgauge_ideal_mod32_20260713")
    return compile_v9_cgauge_ideal_launch_config(**kwargs)


def compile_v9_cgauge_ideal_mod19_sR_launch_config(**kwargs):
    """Compile the C1 #268 treatment over the matched mod19 saliency control."""
    if kwargs.get("with_reachability") is False:
        raise ValueError("v9_cgauge_ideal_mod19_sR cannot disable its S_R treatment")
    kwargs.setdefault("mod_dim", 19)
    kwargs.setdefault("program_name", "v9_cgauge_ideal_mod19_sR")
    kwargs.setdefault("out_dir", "experiments/results/v9_cgauge_ideal_mod19_sR_20260715")
    kwargs["with_reachability"] = True
    return compile_v9_cgauge_ideal_launch_config(**kwargs)


V9_CGAUGE_ISO_CONFIG_IDS: tuple[str, ...] = (
    "v9_cgauge_432_taper_off",
    "v9_cgauge_432_horizon_iso",
    "v9_cgauge_432_step_iso",
)

_TAPER_FLAGS = frozenset({
    "--dseg-aware-taper",
    "--dseg-aware-taper-strength",
    "--dseg-aware-taper-scale",
    "--dseg-aware-taper-floor",
})


def _assert_iso_lever_custody(lever, *, trainer_path=None, consumer_locations=None) -> None:
    """Refuse an ISO Lever without parser, consumer, LawRef, or receipt custody.

    ``consumer_locations`` is injectable for adversarial tests; production derives
    it from the real trainer AST.  This is the requested missing-consumer refusal,
    not a prose declaration.
    """
    from pathlib import Path

    from tac.v9_provenance_gates import _trainer_consumers
    from tac.witness_dsl.curriculum_dsl import TRAINER_REL, build_real_trainer_parser

    root = Path(__file__).resolve().parents[3]
    trainer = Path(trainer_path or root / TRAINER_REL)
    parser = build_real_trainer_parser(trainer)
    actions = {opt: action for action in parser._actions for opt in action.option_strings}
    consumers = (dict(consumer_locations) if consumer_locations is not None
                 else dict(_trainer_consumers(trainer, root)))
    violations: list[str] = []
    for flag in lever.overrides:
        action = actions.get(flag)
        dest = getattr(action, "dest", None)
        if action is None:
            violations.append(f"{flag}: absent from the real trainer parser")
        elif not consumers.get(dest or ""):
            violations.append(f"{flag}: no executable trainer consumer for args.{dest}")
        if not lever.runtime_receipt_schemas.get(flag):
            violations.append(f"{flag}: missing runtime receipt schema")
    for flag, ref in lever.lawrefs.items():
        rec = lever.constant_manifest.get(flag)
        if rec is None or rec.get("equation_id") != ref.equation_id:
            violations.append(f"{flag}: LawRef/compiler-record mismatch")
    if violations:
        raise ValueError(
            f"{lever.name} ISO provenance/consumer REFUSE: " + "; ".join(violations))


def _compile_v9_cgauge_iso_launch_config(
    variant: str,
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str,
):
    """Compile one top-three matched ISO arm over the ideal mod19 control."""
    from tac.witness_autoconfig import CrucibleV7LaunchConfig, _crucible_v7_argv_pairs
    from tac.witness_dsl.curriculum_dsl import HorizonWeightedMargin, StepNativeActivation

    if variant not in {"taper_off", "horizon_iso", "step_iso"}:
        raise ValueError(f"unknown V9 ISO variant {variant!r}")
    config_id = f"v9_cgauge_432_{variant}"
    wrapped = compile_v9_cgauge_ideal_launch_config(
        gt_cache_path,
        num_pairs=num_pairs,
        epochs=epochs,
        out_dir=out_dir,
        mod_dim=19,
        program_name="v9_cgauge_ideal_mod19",
    )
    typed = wrapped.typed
    control_pairs = dict(_crucible_v7_argv_pairs(tuple(typed.to_program().compile_trainer_argv())))
    levers = list(typed.levers)
    delta_lever = None
    duty = {"taper_off": 78.9, "horizon_iso": 47.3, "step_iso": 34.2}[variant]

    if variant == "taper_off":
        matches = [lever for lever in levers if lever.name == "dseg_aware_taper"]
        if len(matches) != 1:
            raise ValueError(
                f"{config_id} REFUSE: expected exactly one base DsegAwareTaper Lever, got {len(matches)}")
        _assert_iso_lever_custody(matches[0])
        delta_lever = matches[0]
        levers = [lever for lever in levers if lever.name != "dseg_aware_taper"]
    elif variant == "horizon_iso":
        stage_boundary = int(typed.base.get("--muon-start-epoch", 726))
        raw = HorizonWeightedMargin(
            weight=0.15,
            target=0.5,
            margin_lo=0.3,
            margin_hi=0.5,
            row_lo=96,
            row_hi=288,
            start_epoch=stage_boundary,
            window=0,
            stage_share_derived_live=True,
            scientific_declaration=True,
        )
        if len(raw.lawrefs) != 7:
            raise ValueError(
                f"{config_id} REFUSE: HorizonWeightedMargin needs seven scientific LawRefs, "
                f"got {len(raw.lawrefs)}")
        _assert_iso_lever_custody(raw)
        delta_lever = _typed_ideal_lever(raw)
        levers.append(delta_lever)
    else:
        raw = StepNativeActivation(
            beta_start=1.0,
            beta_end=8.0,
            anneal="linear",
            window=0,
            basis="annealed_hosc",
            omega=1.0,
            finer_bias_init=False,
            scientific_declaration=True,
        )
        _assert_iso_lever_custody(raw)
        beta_rec = raw.constant_manifest.get("--hosc-beta-end")
        if beta_rec is None or float(beta_rec["value"]) != 8.0:
            raise ValueError(
                f"{config_id} REFUSE: beta_end=8 requires its own LawRef/compiler record")
        delta_lever = _typed_ideal_lever(raw)
        levers.append(delta_lever)

    purpose = (
        f"PREPARED_NOT_FIRED V9 CGauge {variant} matched one-Lever isolation arm; "
        f"duty-to-measure={duty:.1f}% of remaining descent. Same seed/order/steps as "
        "v9_cgauge_ideal_mod19. Operator-GO + governed lane claim still required; "
        "means only, pointer 0.19108/0.18804 UNMOVED."
    )
    typed = typed.model_copy(update={
        "name": config_id,
        "purpose": purpose,
        "out_dir": out_dir,
        "levers": tuple(levers),
    })
    violations = typed.validate_program()
    if violations:
        raise ValueError(
            f"{config_id} DSL gate: {len(violations)} WitnessProgram.validate violation(s): "
            f"{violations[:4]}")

    argv = tuple(typed.to_program().compile_trainer_argv())
    treatment_pairs = dict(_crucible_v7_argv_pairs(argv))
    absent = "<ABSENT>"
    diff = {
        flag: (
            control_pairs.get(flag, absent),
            treatment_pairs.get(flag, absent),
        )
        for flag in sorted(set(control_pairs) | set(treatment_pairs))
        if flag != "--out-dir" and (
            flag not in control_pairs
            or flag not in treatment_pairs
            or control_pairs[flag] != treatment_pairs[flag]
        )
    }
    if variant == "taper_off":
        expected_diff = {flag: (control_pairs[flag], absent) for flag in sorted(_TAPER_FLAGS)}
    elif variant == "horizon_iso":
        expected_diff = {
            flag: (absent, treatment_pairs[flag]) for flag in sorted(delta_lever.overrides)
        }
    else:
        expected_diff = {"--hosc-beta-end": (control_pairs["--hosc-beta-end"], "8.0")}
    if diff != expected_diff:
        raise ValueError(
            f"{config_id} one-Lever-delta REFUSE: argv diff {diff} != expected {expected_diff}")

    rebound = wrapped._rebind_typed(typed)
    manifest = dict(rebound.dsl_program_manifest)
    exclusions = dict(manifest.get("excluded_levers", {}))
    if variant == "horizon_iso":
        exclusions.pop("horizon_weighted_margin", None)
    elif variant == "step_iso":
        exclusions.pop("step_native_endpoint", None)
    manifest.update({
        "expected_active_levers": [lever.name for lever in typed.levers],
        "excluded_levers": exclusions,
        "iso_contract": {
            "config_id": config_id,
            "control_config": "v9_cgauge_ideal_mod19",
            "duty_to_measure_percent": duty,
            "argv_diff": {flag: list(values) for flag, values in diff.items()},
            "one_lever_delta": True,
            "status": "PREPARED_NOT_FIRED_OPERATOR_GO_REQUIRED",
        },
    })
    constants = dict(wrapped.constants_manifest)
    if variant == "taper_off":
        for flag in _TAPER_FLAGS:
            constants.pop(flag.removeprefix("--").replace("-", "_"), None)
    constants = _derive_manifest_from_emitted_argv(constants, argv)
    constants = _merge_lever_constant_manifests(constants, tuple(typed.to_program().levers))
    return CrucibleV7LaunchConfig(
        typed=typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance=dict(wrapped.schedule_governance),
    )


def compile_v9_cgauge_432_taper_off_launch_config(**kwargs):
    kwargs.setdefault("out_dir", "experiments/results/v9_cgauge_432_taper_off_20260715")
    return _compile_v9_cgauge_iso_launch_config("taper_off", **kwargs)


def compile_v9_cgauge_432_horizon_iso_launch_config(**kwargs):
    kwargs.setdefault("out_dir", "experiments/results/v9_cgauge_432_horizon_iso_20260715")
    return _compile_v9_cgauge_iso_launch_config("horizon_iso", **kwargs)


def compile_v9_cgauge_432_step_iso_launch_config(**kwargs):
    kwargs.setdefault("out_dir", "experiments/results/v9_cgauge_432_step_iso_20260715")
    return _compile_v9_cgauge_iso_launch_config("step_iso", **kwargs)


__all__ = [
    "V9_CGAUGE_432_CASCADE_REALIZATION",
    "V9_CGAUGE_432_EXPECTED_LEVERS",
    "V9_CGAUGE_432_PROVENANCE",
    "V9_CGAUGE_IDEAL_EXCLUSIONS",
    "V9_CGAUGE_IDEAL_EXPECTED_ADDITIONS",
    "V9_CGAUGE_IDEAL_SR_EQUATION_ID",
    "V9_CGAUGE_IDEAL_SR_EXPECTED_ADDITION",
    "V9_CGAUGE_PROVENANCE",
    "V9_CGAUGE_ISO_CONFIG_IDS",
    "compile_v9_cgauge_432_horizon_iso_launch_config",
    "compile_v9_cgauge_432_step_iso_launch_config",
    "compile_v9_cgauge_432_taper_off_launch_config",
    "compile_v9_cgauge_432_launch_config",
    "compile_v9_cgauge_config",
    "compile_v9_cgauge_ideal_launch_config",
    "compile_v9_cgauge_ideal_mod19_launch_config",
    "compile_v9_cgauge_ideal_mod19_sR_launch_config",
    "compile_v9_cgauge_ideal_mod32_launch_config",
    "derive_v9_cgauge_432_config",
    "derive_v9_cgauge_config",
]
