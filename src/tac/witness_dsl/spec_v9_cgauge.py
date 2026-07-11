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


__all__ = [
    "V9_CGAUGE_PROVENANCE",
    "compile_v9_cgauge_config",
    "derive_v9_cgauge_config",
]
