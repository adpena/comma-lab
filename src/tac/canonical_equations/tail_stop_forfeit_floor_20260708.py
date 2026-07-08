# SPDX-License-Identifier: MIT
"""Canonical equation: the TAIL PowerPlay attribution-floor law s* = ν·forfeit
(2026-07-08) — the EQUATIONS leg of SEAL-v7-r1 MAJOR-1's reconcile.

SEAL v7 ROUND-1 LENS-2 (`.omx/research/t5_crucible/seal_v7_r1_deepmath_20260708.md`)
MAJOR-1: the TAIL PowerPlay stop HARDCODES ``stop_marginal_s = 1e-4 S/ep``, **14.5×
COARSER** than the campaign's own DERIVED marginal-S floor
``s* = ν·forfeit = 0.012653 · 5.450779e-4 = 6.897e-6 S/ep`` (DRAFT_OPTIMAL_STACK_v6
§2.2f/g), so the constant TRUNCATES the exact power-law tail the module exists to
mine. A LawRef IS derivable — but the ``s*=ν·forfeit`` law was **not in the
canonical registry** (seal grep empty), so nothing bound the tail to it. This module
registers it (the same ``forfeit_matched_exit_v1`` id the LawRef evaluator already
uses in ``tac.canonical_equations.evaluators``) so the DERIVED floor is a real
compile-time constant, not a bare literal.

DERIVATION CHAIN (verified from the ν-law anchors this session):
* ``ν(tau_softplus) = 0.012653`` /ep — the BINDING per-stage exponential-meat fit
  (DRAFT v6 §2.2g, fold-2 P-CT1; the registered single-ν = 0.026210 was NOT
  reproducible and struck). This is the ν the DRAFT's §2.2f/g s* row USES.
* ``forfeit = 5.450779e-4`` S — the P-CT3 restore-EMA-best forfeit (recovery vs the
  shipped ep625 fire; DRAFT v6 §2.2f, VERIFIED full-precision from the trace).
* ``s* = ν·forfeit = 6.8968706687e-6`` S/ep (rounds to 6.897e-6; the DRAFT's
  artifact-full-precision print is 6.8971e-6, a 0.003% display delta with no
  consumer at that resolution).

HONEST NOTE on the seal's ``s*_tail = ν(muon_fin)·forfeit_tail`` reformulation: the
seal offered that FORM (the TAIL runs one stage LATER, so ν(muon_fin)=0.003289 is the
regime-correct rate) as a MORE-conservative/finer floor. But ``0.003289 · 5.450779e-4
= 1.79e-6`` ≠ 6.897e-6, and no ``forfeit_tail`` has been measured — so the ν(muon_fin)
form is a run-2 duty-to-measure, NOT the derived value. This registration binds the
DERIVED magnitude the campaign actually computed: ν(tau_softplus)·forfeit. The
``forfeit_matched_exit_v1`` FORM (s* = ν·forfeit) covers both once forfeit_tail lands.

All rows ``[macOS-MLX research-signal]`` / ``[macOS-CPU advisory]`` NON-PROMOTABLE;
pointer contest-CPU **0.19110 UNMOVED** — everything here is MEANS.

Registration tool: ``tools/register_tail_stop_forfeit_floor_20260708.py`` (idempotent).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

_UTC = "2026-07-08T00:00:00Z"
_MLX_SIGNAL = "[macOS-MLX research-signal]"
_HW = "m5_max_cpu"

FORFEIT_MATCHED_EXIT_EQUATION_ID = "forfeit_matched_exit_v1"

# ---- measured constants (each labeled at its anchor) --------------------------------------
NU_TAU_SOFTPLUS = 0.012653     # ν(tau_softplus) binding per-stage fit (DRAFT v6 §2.2g, P-CT1)
FORFEIT_S = 5.450779e-4        # P-CT3 restore-EMA-best forfeit S (DRAFT v6 §2.2f)
NU_MUON_FIN = 0.003289         # ν(muon_fin) — the seal's run-2 reformulation rate (NOT used here)
S_STAR_DERIVED = NU_TAU_SOFTPLUS * FORFEIT_S  # = 6.8968706687e-6 S/ep (rounds to 6.897e-6)
HARDCODED_COARSE_FLOOR = 1e-4  # the struck HARDCODED value (14.5× coarser; seal MAJOR-1)

_DRAFT_V6 = ".omx/research/t5_crucible/DRAFT_OPTIMAL_STACK_v6_20260708.md"
_SEAL_V7 = ".omx/research/t5_crucible/seal_v7_r1_deepmath_20260708.md"


# ---- callable (the natural-signature math helper the equation points at) -------------------
def forfeit_matched_exit_s_star(nu: float, forfeit: float) -> float:
    """s* = ν·forfeit — the PowerPlay marginal-S attribution floor (S/ep).

    ``nu``: the stage exponential-meat decay rate (S/ep; P-CT1 fit).
    ``forfeit``: the S recovery forfeited by a one-cadence-late fire (P-CT3).
    Mirrors ``tac.canonical_equations.evaluators.eval_forfeit_matched_exit_s_star``
    (the LawRef eval adapter) — SAME arithmetic, natural signature."""
    return float(nu) * float(forfeit)


# ---- builder -------------------------------------------------------------------------------
def build_forfeit_matched_exit_v1() -> CanonicalEquation:
    """s* = ν·forfeit: the forfeit-matched TAU→FIN exit trigger AND the TAIL PowerPlay
    attribution-floor (the SAME law; SEAL-v7-r1 MAJOR-1 reconcile). Binds the TAIL stop to
    the DERIVED 6.897e-6 floor instead of the struck HARDCODED 1e-4."""
    anchor_derivation = EmpiricalAnchor(
        anchor_id="tail_stop_forfeit_floor_s_star_derived_20260708",
        measurement_utc=_UTC,
        inputs={
            "nu_tau_softplus_per_ep": NU_TAU_SOFTPLUS,
            "forfeit_S": FORFEIT_S,
            "nu_source": "DRAFT v6 §2.2g fold-2 P-CT1 binding per-stage fit",
            "forfeit_source": "DRAFT v6 §2.2f P-CT3 restore-EMA-best (VERIFIED full-precision)",
        },
        predicted_output={"s_star_S_per_ep": forfeit_matched_exit_s_star(NU_TAU_SOFTPLUS, FORFEIT_S)},
        empirical_output={
            "s_star_S_per_ep_draft_artifact_precision": 6.8971e-6,
            "s_star_S_per_ep_4dp_nu_recompute": 6.8969e-6,
            "hardcoded_coarse_floor_struck": HARDCODED_COARSE_FLOOR,
            "coarsening_ratio": HARDCODED_COARSE_FLOOR / S_STAR_DERIVED,
            "note": "the DERIVED s* = ν(tau)·forfeit = 6.897e-6 S/ep is 14.5× finer than the "
                    "struck 1e-4 hardcode; the seal's s*_tail = ν(muon_fin)·forfeit_tail "
                    "reformulation (ν=0.003289) is a run-2 duty-to-measure (no forfeit_tail yet)",
        },
        residual=0.0,
        source_artifact=_DRAFT_V6,
        measurement_method="draft_v6_s2_2fg_arithmetic_reexecuted",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SEAL_V7,
            reactivation_criteria="re-fit ν per-stage on any run whose schedule/mod differs from "
                                  "mod32cap (P-CT1 config-conditionality); promote to the "
                                  "s*_tail = ν(muon_fin)·forfeit_tail form once run-1 measures "
                                  "the post-Muon forfeit_tail",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )
    anchor_evaluator = EmpiricalAnchor(
        anchor_id="forfeit_matched_exit_evaluator_source_verified_20260708",
        measurement_utc=_UTC,
        inputs={"evaluator": "tac.canonical_equations.evaluators:eval_forfeit_matched_exit_s_star",
                "lawref_consumer": "tac.witness_control.tail_cycles:stop_marginal_s_lawref"},
        predicted_output={"form": "s* = nu * forfeit"},
        empirical_output={"evaluator_matches_callable": True},
        residual=0.0,
        source_artifact="src/tac/canonical_equations/evaluators.py",
        measurement_method="source_inspection",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SEAL_V7,
            reactivation_criteria="re-verify if the forfeit_matched_exit_v1 evaluator arity changes",
            measurement_axis="[predicted]",
            hardware_substrate=_HW,
        ),
    )
    return CanonicalEquation(
        equation_id=FORFEIT_MATCHED_EXIT_EQUATION_ID,
        name="Forfeit-matched exit / TAIL PowerPlay attribution floor (s* = ν·forfeit)",
        one_line_summary=(
            "s* = ν·forfeit = 0.012653·5.450779e-4 ≈ 6.897e-6 S/ep — the DERIVED PowerPlay "
            "attribution floor (14.5× finer than the struck 1e-4 hardcode); SEAL-v7-r1 MAJOR-1"
        ),
        latex_form=r"s^{*} = \nu \cdot \mathrm{forfeit}",
        python_callable_module_path=(
            "tac.canonical_equations.tail_stop_forfeit_floor_20260708:forfeit_matched_exit_s_star"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-MLX research-signal"],
            "config_conditional": "ν is the mod32cap/this-schedule per-stage fit (P-CT1); re-fit on "
                                  "config change. The FORM s*=ν·forfeit is regime-general (the TAIL "
                                  "stage promotes to ν(muon_fin)·forfeit_tail once forfeit_tail lands)",
        },
        units_in={"nu": "S_per_epoch", "forfeit": "S"},
        units_out={"s_star": "S_per_epoch"},
        empirical_anchors=(anchor_derivation, anchor_evaluator),
        predicted_vs_empirical_residual={"draft_v6_s2_2fg_arithmetic": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.tail_cycles",
            "tac.canonical_equations.evaluators",
        ),
        canonical_producers=(
            _DRAFT_V6,
            _SEAL_V7,
        ),
        provenance=build_provenance_for_predicted(
            model_id="forfeit_matched_exit.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate=_HW,
        ),
    )


ALL_BUILDERS = (build_forfeit_matched_exit_v1,)

__all__ = [
    "ALL_BUILDERS",
    "FORFEIT_MATCHED_EXIT_EQUATION_ID",
    "FORFEIT_S",
    "HARDCODED_COARSE_FLOOR",
    "NU_MUON_FIN",
    "NU_TAU_SOFTPLUS",
    "S_STAR_DERIVED",
    "build_forfeit_matched_exit_v1",
    "forfeit_matched_exit_s_star",
]
