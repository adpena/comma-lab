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
        "levers": tuple(typed.levers) + (
            TypedLever(name=_gate.name, overrides=dict(_gate.overrides),
                       epochs_delta=_gate.epochs_delta, notes=_gate.notes),
            TypedLever(name=_t1.name, overrides=dict(_t1.overrides),
                       epochs_delta=_t1.epochs_delta, notes=_t1.notes),
        ),
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


__all__ = [
    "V9_CGAUGE_432_CASCADE_REALIZATION",
    "V9_CGAUGE_432_EXPECTED_LEVERS",
    "V9_CGAUGE_432_PROVENANCE",
    "V9_CGAUGE_PROVENANCE",
    "compile_v9_cgauge_432_launch_config",
    "compile_v9_cgauge_config",
    "derive_v9_cgauge_432_config",
    "derive_v9_cgauge_config",
]
