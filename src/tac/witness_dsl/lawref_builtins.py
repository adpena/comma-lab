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

# Canonical name -> LawRef, for iteration / audit.
BUILTIN_LAWREFS: dict[str, LawRef] = {
    "s_star_forfeit_matched_exit": S_STAR_FORFEIT_MATCHED_EXIT,
    "tau_star_maslov_q90": TAU_STAR_MASLOV_Q90,
    "r_star_critical_nucleus": R_STAR_CRITICAL_NUCLEUS,
}

__all__ = [
    "BUILTIN_LAWREFS",
    "R_STAR_CRITICAL_NUCLEUS",
    "S_STAR_FORFEIT_MATCHED_EXIT",
    "TAU_STAR_MASLOV_Q90",
]
