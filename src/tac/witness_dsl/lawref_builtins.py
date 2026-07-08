# SPDX-License-Identifier: MIT
"""Built-in LawRefs — the first 3 constants compiled from canonical equations.

Task #351 stage 4: prove the LawRef mechanism end-to-end against REAL T5-crucible
artifacts (no autoconfig touched). Each LawRef here resolves (via
``tac.witness_dsl.lawref.resolve``) to a KNOWN sealed value — bit-matching the
sealed number IS the validation (see ``src/tac/tests/test_lawref_*``):

  * ``S_STAR_FORFEIT_MATCHED_EXIT``  s* = ν·forfeit → 6.897090…e-6  (rounds to the
    sealed 6.8971e-6). ν from the wave-A trace-probe artifact (P-CT1), forfeit the
    P-CT3 recovery constant. ladder = derived_at_config.
  * ``TAU_STAR_MASLOV_Q90``  τ* = m_q(q90)/ln5 → 0.4619441215759677  (bit-identical
    to the artifact's stored ``tau_star.q90``). ladder = measured_anchor.
  * ``R_STAR_CRITICAL_NUCLEUS``  r* = 0.95·σ_eff → 1.425  (B18 island-release radius;
    σ_eff = 1.5 px dilation-knee). ladder = derived_at_config.

These are the migration SEED: crucible_v6's 5 constants (τ_end, ν laws, β_hosc,
adaptive-ε clamps, s_fit) fold into this module as LawRefs in the #351 follow-up
(see the landing memo's migration plan). Kept as module-level constants because
construction is pure (no filesystem I/O until ``resolve``).

Provenance anchors: ``.omx/research/t5_crucible/ORCHESTRATION_LEDGER.md`` (req T +
the P-CT1/P-CT3/τ-confirm folds) + the artifacts under
``.omx/research/t5_crucible/artifacts/``.
"""
from __future__ import annotations

from tac.witness_dsl.lawref import (
    LADDER_DERIVED_AT_CONFIG,
    LADDER_MEASURED_ANCHOR,
    InputRef,
    LawRef,
)

_ARTIFACTS = ".omx/research/t5_crucible/artifacts"
_TRACE_PROBES = f"{_ARTIFACTS}/trace_probes_levelset_n600_witness_mod32cap_20260706T115554Z.json"
_TAU_CONFIRM_END = f"{_ARTIFACTS}/tau_mq_confirm_end_20260708.json"
_TAU_KNEE_PTAU2 = f"{_ARTIFACTS}/tau_knee_ptau2_20260708.json"
_NU_TAU_SOFTPLUS_KEY = "probes/ct1/result/stages/tau_softplus/nu_per_ep"

# s* = ν·forfeit — the forfeit-matched TAU->FIN exit trigger (P-CT3).
# ν is CONFIG-CONDITIONAL (mod32cap / tau_softplus stage — the P-CT1 lesson).
S_STAR_FORFEIT_MATCHED_EXIT = LawRef(
    equation_id="forfeit_matched_exit_v1",
    inputs={
        "nu": InputRef.anchor(
            _TRACE_PROBES,
            "probes/ct1/result/stages/tau_softplus/nu_per_ep",
            "P-CT1 wave-A exponential-meat ν-refit, tau_softplus stage (0.012653/ep)",
            config_tags={"schedule": "mod32cap", "stage": "tau_softplus"},
        ),
        "forfeit": InputRef.literal(
            0.0005450778537325913,
            "P-CT3 forfeit_S_vs_stage_best at reference fire ep625 "
            "(+5.450779e-4 S recovery vs one-cadence-late fire)",
        ),
    },
    ladder_class=LADDER_DERIVED_AT_CONFIG,
)

# τ* = m_q/ln5 — Maslov interface half-width; q90 convention (END_ep1000 leg).
TAU_STAR_MASLOV_Q90 = LawRef(
    equation_id="tau_star_maslov_quantile_v1",
    inputs={
        "m_q": InputRef.anchor(
            _TAU_CONFIRM_END,
            "results/0/table/m_q/q90",
            "τ-confirm END_ep1000 q90 GT-margin (16-pair advisory; 0.7434703826904296)",
            config_tags={"leg": "END_ep1000", "convention": "q90"},
        ),
    },
    ladder_class=LADDER_MEASURED_ANCHOR,
)

# r* = 0.95·σ_eff — B18 island-release radius (critical nucleus).
R_STAR_CRITICAL_NUCLEUS = LawRef(
    equation_id="critical_nucleus_release_v1",
    inputs={
        "coeff": InputRef.literal(
            0.95, "B18 release coefficient 0.674·√2 = 0.9532 ≈ 0.95 (operator law)"
        ),
        "sigma_eff": InputRef.literal(
            1.5,
            "0630 island dilation-knee probe σ_eff (px); knee r*≈1.43 straddles native dash",
            config_tags={"sigma_probe": "island_dilation_knee_0630"},
        ),
    },
    ladder_class=LADDER_DERIVED_AT_CONFIG,
)

# ══════════════════════════════════════════════════════════════════════════
# crucible_v6 constant migration (#351 follow-up). VALUE-IDENTITY IS THE LAW:
# each LawRef resolves BIT-IDENTICALLY to the value ``derive_crucible_v6_config``
# currently hardcodes in ``_CRUCIBLE_V6_DELTAS`` (bit-match test per constant).
#
# Two tiers:
#   CONSUMED  — the variant EMITS these; ``derive_crucible_v6_config`` resolves
#               them via the resolver and the emitted launch.sh stays byte-
#               identical (the resolved value == the sealed literal).
#   LIBRARY   — the migration-plan laws NOT emitted by this variant (ν-family
#               window laws, the P-CON absolute persistence bar, the adaptive-ε
#               saturation alarm). Registered as canonical constant-compiler
#               entries (the req-T single-source-of-truth) + bit-match tested,
#               but NEVER wired into the argv (wiring a non-emitted flag would
#               break value-identity — that is a bug, not a migration).
# ══════════════════════════════════════════════════════════════════════════

# ── CONSUMED: τ_end 0.31 — the P-TAU2 knee probe's chosen launch τ. ──────────
TAU_END_KNEE_LAUNCH = LawRef(
    equation_id="tau_end_knee_launch_v1",
    inputs={
        "launch_tau": InputRef.anchor(
            _TAU_KNEE_PTAU2,
            "launch_tau",
            "P-TAU2 knee probe launch_tau 0.31 (within knee band [0.19072,0.54294]; "
            "≈ mod32cap ep650-best τ=0.3098). v6 §1.4a τ_end re-anchor.",
            config_tags={"schedule": "mod32cap"},
        ),
    },
    ladder_class=LADDER_MEASURED_ANCHOR,
    fallback=0.31,
    fallback_waiver_reason=(
        "sealed-v6.4 crucible τ_end literal 0.31; P-TAU2 knee artifact unavailable at config time"
    ),
)

# ── CONSUMED: β_end 10.0 — hosc-β anneal endpoint PIN (linear-replica law). ──
HOSC_BETA_FIREBAND_PIN = LawRef(
    equation_id="hosc_beta_fireband_pin_v1",
    inputs={
        "beta_end": InputRef.literal(
            10.0,
            "v6.3 MAJOR-2(i) β-pin 10.0: β is LINEAR over the SHARED --anneal-epochs 3000 den, so "
            "endpoint 10.0 reproduces the mod32cap control's β(ep) slope on [1,726] to ≤0.1% (the ν/"
            "ep650-best/fire-band anchors were measured at the control's JOINT β state). RE-DERIVE on "
            "any change to muon-start-epoch (726 freeze) / --anneal-epochs / --hosc-beta-anneal.",
        ),
    },
    ladder_class=LADDER_DERIVED_AT_CONFIG,
)

# ── CONSUMED: LR-anneal den 1000 — the control vehicle's own --epochs den. ───
LR_CONTROL_DENOMINATOR = LawRef(
    equation_id="lr_control_denominator_v1",
    inputs={
        "control_den": InputRef.literal(
            1000,
            "v6.4 MAJOR-2(ii) LR-pin: the mod32cap CONTROL ran its LR cosine at --epochs=1000, so "
            "--lr-anneal-epochs 1000 reproduces the control LR(ep) on [1,726] BIT-IDENTICALLY (unlike "
            "β the LR curvature is not endpoint-rephasable => own denominator). INT (str '1000').",
        ),
    },
    ladder_class=LADDER_DERIVED_AT_CONFIG,
)

# ── CONSUMED: LR-hold-frac 1.0 — NO hold (control never held LR pre-freeze). ─
LR_HOLD_FRAC_NO_HOLD = LawRef(
    equation_id="lr_hold_frac_no_hold_v1",
    inputs={
        "hold_frac": InputRef.literal(
            1.0,
            "v6.4 LR-pin: --lr-hold-frac 1.0 = NO hold. The control's Muon freeze (726) < the LR den "
            "(1000), so it never held LR pre-freeze => hold-frac 1.0 is the bit-identical-cosine choice.",
        ),
    },
    ladder_class=LADDER_DERIVED_AT_CONFIG,
)

# ── LIBRARY: ν-family window laws (P-CT1 exponential-meat; tau_softplus). ────
# settle = 3/ν  (bit-reproduces the artifact's stored settle_3_over_nu_ep = 237.09).
SETTLE_WINDOW_TAU_SOFTPLUS = LawRef(
    equation_id="settle_window_v1",
    inputs={
        "coeff": InputRef.literal(3.0, "settle = 3 e-folds (3·τ_e) of the exponential meat"),
        "nu": InputRef.anchor(
            _TRACE_PROBES,
            _NU_TAU_SOFTPLUS_KEY,
            "P-CT1 tau_softplus ν per-ep (0.012653/ep)",
            config_tags={"schedule": "mod32cap", "stage": "tau_softplus"},
        ),
    },
    ladder_class=LADDER_MEASURED_ANCHOR,
)

# tail-cycle floor = 3/ν + 150  (bit-reproduces the stored tail_cycle_floor_ep = 387.09).
TAIL_CYCLE_FLOOR_TAU_SOFTPLUS = LawRef(
    equation_id="tail_cycle_floor_v1",
    inputs={
        "coeff": InputRef.literal(3.0, "settle = 3 e-folds"),
        "nu": InputRef.anchor(
            _TRACE_PROBES,
            _NU_TAU_SOFTPLUS_KEY,
            "P-CT1 tau_softplus ν per-ep (0.012653/ep)",
            config_tags={"schedule": "mod32cap", "stage": "tau_softplus"},
        ),
        "tail_extra": InputRef.literal(150.0, "one dwell floor (150 ep) added past settle"),
    },
    ladder_class=LADDER_MEASURED_ANCHOR,
)

# ── LIBRARY: P-CON absolute persistence bar (B17′ fitted; τ-independent). ────
CONLEY_ABSOLUTE_BAR_TAU = LawRef(
    equation_id="conley_absolute_bar_v1",
    inputs={
        "s_fit_logit": InputRef.literal(
            1.7504924172,
            "P-CON B17′ fitted ABSOLUTE persistence bar @ Tau-stage maps (s=21.75, P(s|cert)=0.9574); "
            "near τ-INDEPENDENT while τ varied 4.3× (draft §3.4/§SC row). Draft-cited fit.",
            config_tags={"stage": "tau_softplus"},
        ),
    },
    ladder_class=LADDER_MEASURED_ANCHOR,
)

CONLEY_ABSOLUTE_BAR_MUON = LawRef(
    equation_id="conley_absolute_bar_v1",
    inputs={
        "s_fit_logit": InputRef.literal(
            1.3017706202,
            "P-CON B17′ fitted ABSOLUTE persistence bar @ MuonBest (s=3.75, P(s|cert)=0.9506); "
            "near τ-INDEPENDENT while τ varied 4.3× (draft §3.4/§SC row). Draft-cited fit.",
            config_tags={"stage": "muon_fin"},
        ),
    },
    ladder_class=LADDER_MEASURED_ANCHOR,
)

# ── LIBRARY: adaptive-ε saturation alarm clamp (v6 §row-1; KEPT at τ_end 0.31). ─
ADAPTIVE_EPS_SATURATION_ALARM = LawRef(
    equation_id="adaptive_eps_saturation_alarm_v1",
    inputs={
        "alarm_threshold": InputRef.literal(
            0.7,
            "adaptive-ε saturation ALARM: ε_raw > 0.7 sustained (v6 §row-1; the |c_a(τ)| growth that "
            "drove it RELAXES at τ_end 0.31 but the alarm is KEPT). Control-loop constant.",
        ),
    },
    ladder_class=LADDER_DERIVED_AT_CONFIG,
)


# The crucible_v6 CONSUMED constants, keyed by their ``_CRUCIBLE_V6_DELTAS`` key
# so ``derive_crucible_v6_config`` can resolve them and merge over the sealed
# deltas (resolved value == the literal => launch.sh byte-identical).
CRUCIBLE_V6_CONSUMED_LAWREFS: dict[str, LawRef] = {
    "softmax_temp_end": TAU_END_KNEE_LAUNCH,
    "hosc_beta_end": HOSC_BETA_FIREBAND_PIN,
    "lr_anneal_epochs": LR_CONTROL_DENOMINATOR,
    "lr_hold_frac": LR_HOLD_FRAC_NO_HOLD,
}

# The target config tags the CONSUMED resolution runs under (the mod32cap-lineage
# crucible_v6 vehicle). Matches the τ_end anchor's config_tags so the P-CT1
# conditionality guard PASSES for this vehicle (a mod48 run would fail closed).
CRUCIBLE_V6_CONSUMED_TARGET_TAGS: dict[str, str] = {"schedule": "mod32cap"}


# Canonical name -> LawRef, for iteration / audit.
BUILTIN_LAWREFS: dict[str, LawRef] = {
    "s_star_forfeit_matched_exit": S_STAR_FORFEIT_MATCHED_EXIT,
    "tau_star_maslov_q90": TAU_STAR_MASLOV_Q90,
    "r_star_critical_nucleus": R_STAR_CRITICAL_NUCLEUS,
    # crucible_v6 migration (#351 follow-up):
    "tau_end_knee_launch": TAU_END_KNEE_LAUNCH,
    "hosc_beta_fireband_pin": HOSC_BETA_FIREBAND_PIN,
    "lr_control_denominator": LR_CONTROL_DENOMINATOR,
    "lr_hold_frac_no_hold": LR_HOLD_FRAC_NO_HOLD,
    "settle_window_tau_softplus": SETTLE_WINDOW_TAU_SOFTPLUS,
    "tail_cycle_floor_tau_softplus": TAIL_CYCLE_FLOOR_TAU_SOFTPLUS,
    "conley_absolute_bar_tau": CONLEY_ABSOLUTE_BAR_TAU,
    "conley_absolute_bar_muon": CONLEY_ABSOLUTE_BAR_MUON,
    "adaptive_eps_saturation_alarm": ADAPTIVE_EPS_SATURATION_ALARM,
}

__all__ = [
    "ADAPTIVE_EPS_SATURATION_ALARM",
    "BUILTIN_LAWREFS",
    "CONLEY_ABSOLUTE_BAR_MUON",
    "CONLEY_ABSOLUTE_BAR_TAU",
    "CRUCIBLE_V6_CONSUMED_LAWREFS",
    "CRUCIBLE_V6_CONSUMED_TARGET_TAGS",
    "HOSC_BETA_FIREBAND_PIN",
    "LR_CONTROL_DENOMINATOR",
    "LR_HOLD_FRAC_NO_HOLD",
    "R_STAR_CRITICAL_NUCLEUS",
    "S_STAR_FORFEIT_MATCHED_EXIT",
    "SETTLE_WINDOW_TAU_SOFTPLUS",
    "TAIL_CYCLE_FLOOR_TAU_SOFTPLUS",
    "TAU_END_KNEE_LAUNCH",
    "TAU_STAR_MASLOV_Q90",
]
