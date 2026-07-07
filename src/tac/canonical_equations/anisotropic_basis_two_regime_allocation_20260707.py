# SPDX-License-Identifier: MIT
"""Canonical equation: anisotropic-basis TWO-REGIME along-tangent frequency allocation
(DAG FEED-07a, 2026-07-07 operator reframe: basis-match + rule-118 = the next-run PRIMARY arms).

The directional (anisotropic/curvelet) Fourier basis carries a high frequency ACROSS each
boundary (``freq_across``, the sharp normal direction) and a lower frequency ALONG the boundary
tangent (``freq_along``). The optimal ``freq_along`` is NOT one number — it depends on WHICH
witness carries the lane class (the one boundary that is along-edge TEXTURE, not a cartoon edge):

  regime = "lane_offloaded"  (lane -> the FREE analytic render band, rule-118: the openpilot
      polynomial band is a deterministic generator compiled into inflate.py at ~0 counted bytes,
      MEASURED lane d_seg 0.00087):  the boundaries the witness still carries are C^2-cartoon
      edges, so Candes-Donoho parabolic scaling (width ~ length^2 for curved smooth singularities)
      applies:

          freq_along = max(4, round(sqrt(freq_across)))          (across=32 -> along ~ 6)

  regime = "lane_carried"  (the witness itself must represent the lane dash comb): the dashes are
      a ~25 cyc/unit on/off modulation ALONG the tangent — along-edge texture that VIOLATES the
      cartoon model. The 4-lens root-cause pass MEASURED the basis 3.2x short of the dash
      frequency at freq_along=8 (25/8 ~ 3.2; R is ALL-PASS to 2px so R is NOT the wall):

          freq_along = min(freq_across, round(8 * 3.2)) = 26     (across=32 -> along = 26)

Anchored by (honest tiers): the MEASURED -48% all-class directional-basis d_seg drop
(VERIFIED_VIA_EMPIRICAL_ANCHOR, DAG FEED 2026-06-25t) + the MEASURED 3.2x along-tangent deficit
(VERIFIED_VIA_EMPIRICAL_ANCHOR, 4-lens FEED 2026-07-03) + the regime-1 parabolic optimum itself,
which is INFERRED_FROM_DOMAIN_LITERATURE (Candes-Donoho 2004 curvelet parabolic scaling) and
ASSUMED_AWAITING_VERIFICATION at the specific along=sqrt(across) point — the next-run A/B is the
owed anchor.

DSL leg: ``tac.witness_dsl.curriculum_dsl.DirectionalBasisRebalance`` emits this law over the
real trainer flags (``--freq-across`` / ``--freq-along`` / ``--n-dir-freqs`` / ``--self-orient``).

means != ends: this is a train-time prior derivation (advisory anchors, NON-PROMOTABLE);
pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "anisotropic_basis_two_regime_allocation_v1"

_UTC = "2026-07-07T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- MEASURED constants (the anchors' load-bearing numbers) ---------------------------------------
ALL_CLASS_DIRECTIONAL_DSEG_DROP = -0.48      # all-class directional basis vs isotropic (lane-only -8%)
LANE_ONLY_DIRECTIONAL_DSEG_DROP = -0.08
DIRECTIONAL_BASELINE_DSEG_N600 = 0.008257    # lever-B baseline the -48% was measured against
DASH_COMB_FREQ_CYC_PER_UNIT = 25.0           # measured lane-dash on/off modulation along the tangent
ALONG_TANGENT_DEFICIT_FACTOR = 3.2           # 25 / 8 at the live freq_along=8 (4-lens root cause)
LANE_CARRIED_REFERENCE_ALONG = 8.0           # the freq_along the deficit was measured at


def freq_along_for_regime(freq_across: int | float, regime: str) -> int:
    """The two-regime along-tangent allocation law (the DSL lever's derivation callable).

    ``lane_offloaded``: parabolic (Candes-Donoho) scaling for the remaining cartoon edges —
    ``max(4, round(sqrt(freq_across)))``. ``lane_carried``: cover the measured 3.2x dash-comb
    deficit — ``min(freq_across, round(8 * 3.2))``. Raises ``ValueError`` on an unknown regime
    (fail-closed; never a silent default)."""
    fa = float(freq_across)
    if fa <= 0:
        raise ValueError(f"freq_across must be > 0, got {freq_across!r}")
    if regime == "lane_offloaded":
        return max(4, round(math.sqrt(fa)))
    if regime == "lane_carried":
        return min(int(round(fa)), round(LANE_CARRIED_REFERENCE_ALONG * ALONG_TANGENT_DEFICIT_FACTOR))
    raise ValueError(f"unknown regime {regime!r} (expected 'lane_offloaded' | 'lane_carried')")


def build_anisotropic_basis_two_regime_allocation_v1() -> CanonicalEquation:
    """Build the two-regime anisotropic-basis allocation equation with its honest-tier anchors."""

    anchor_directional = EmpiricalAnchor(
        anchor_id="all_class_directional_basis_minus48pct_20260625t",
        measurement_utc="2026-06-25T00:00:00Z",
        inputs={
            "vehicle": "ImprovedSegGenerator MLX coord-INR (witness-build subagent a922483dfc636ccc3)",
            "authority": "frozen CPU-torch SegNet argmax (MLX gradient-only, NEVER MPS)",
            "baseline_dseg_n600": DIRECTIONAL_BASELINE_DSEG_N600,
        },
        predicted_output={"claim": "basis-match is PRIOR to capacity; directional basis moves d_seg"},
        empirical_output={
            "all_class_directional_dseg_drop": ALL_CLASS_DIRECTIONAL_DSEG_DROP,
            "lane_only_directional_dseg_drop": LANE_ONLY_DIRECTIONAL_DSEG_DROP,
            "verdict": ("all-class directional/anisotropic Fourier basis = THE decisive ~0-byte "
                        "d_seg lever (-48%); lane-only orientation only -8%; capacity alone "
                        "nothing/HURTS until the basis matches the boundary geometry"),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="lever sweep on frozen CPU-torch SegNet argmax (DAG FEED 2026-06-25t)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria=("re-measure the directional-basis drop on the levelset witness at "
                                   "optimal form (ancestor numbers are lessons, not transfers)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_deficit = EmpiricalAnchor(
        anchor_id="along_tangent_dash_comb_deficit_3p2x_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={
            "run": "#205 ep200 4-lens deep-math pass (DAG FEED 2026-07-03; memo "
                   "lane_dash_residual_root_is_along_tangent_freq_deficit_R_allpass_20260703)",
            "live_freq_along": LANE_CARRIED_REFERENCE_ALONG,
        },
        predicted_output={"hypothesis": "the d_seg residual is R-washout of the dash comb"},
        empirical_output={
            "dash_comb_freq_cyc_per_unit": DASH_COMB_FREQ_CYC_PER_UNIT,
            "deficit_factor": ALONG_TANGENT_DEFICIT_FACTOR,
            "verdict": ("OVERTURNS the R-washout prior: R is ALL-PASS to 2px (0.997 at the 10px "
                        "dash scale). The basis is 3.2x short of the ~25 cyc/unit dash frequency "
                        "ALONG the tangent — an along-edge-texture blindness by construction; "
                        "the lever is along-tangent frequency, not beating R"),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="4-lens residual decomposition (spectral lens 2) on #205 ep200",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria=("recompute the dash-comb spectrum if the lane render geometry "
                                   "changes (new clip / new homography per tac.clip_profile)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_parabolic = EmpiricalAnchor(
        anchor_id="regime1_parabolic_along_eq_sqrt_across_owed_ab_20260707",
        measurement_utc=_UTC,
        inputs={
            "literature": "Candes-Donoho 2004 (curvelets: parabolic scaling is the optimal sparse "
                          "basis for C^2-cartoon curved singularities)",
            "regime": "lane_offloaded (lane -> rule-118 analytic band, MEASURED lane d_seg 0.00087)",
        },
        predicted_output={
            "freq_along_at_across_32": freq_along_for_regime(32, "lane_offloaded"),
            "law": "freq_along = max(4, round(sqrt(freq_across)))",
        },
        empirical_output={
            "status": ("OWED — the specific along=sqrt(across) optimum is NOT yet A/B-measured on "
                       "the witness; the next-run FEED-07a arm-(A) A/B is the owed anchor"),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="derivation from Candes-Donoho parabolic scaling (no run yet)",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="anisotropic_basis_two_regime_allocation.regime1_parabolic",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Anisotropic-basis two-regime along-tangent allocation: lane_offloaded -> "
              "along=max(4,round(sqrt(across))) (Candes-Donoho parabolic); lane_carried -> "
              "along=min(across,round(8*3.2))=26 (measured dash-comb deficit)"),
        one_line_summary=(
            "freq_along = sqrt(freq_across) when the lane rides the free rule-118 band; 26 when "
            "the witness carries the dash comb (measured 3.2x deficit; -48% directional anchor)."
        ),
        latex_form=(
            r"f_{\parallel}=\begin{cases}\max(4,\lfloor\sqrt{f_{\perp}}\rceil)&\text{lane offloaded "
            r"(parabolic)}\\\min(f_{\perp},\lfloor 8\cdot 3.2\rceil)=26&\text{lane carried (dash comb)}"
            r"\end{cases}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.anisotropic_basis_two_regime_allocation_20260707:"
            "freq_along_for_regime"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "lever": ("tac.witness_dsl.curriculum_dsl.DirectionalBasisRebalance "
                      "(--freq-across/--freq-along/--n-dir-freqs/--self-orient)"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("regime-1 (parabolic) holds ONLY when the analytic lane band actually carries "
                     "the lane class (compose AnalyticLaneRenderBand); with the lane carried by the "
                     "witness, parabolic under-allocates by ~3.2x (the measured deficit)"),
        },
        units_in={"freq_across": "cycles_per_unit", "regime": "enum"},
        units_out={"freq_along": "cycles_per_unit_int"},
        empirical_anchors=(anchor_directional, anchor_deficit, anchor_parabolic),
        predicted_vs_empirical_residual={
            # lane_carried regime: law-derived along (26) vs the measured dash-comb need (25 cyc/unit
            # normalized by the reference along=8 -> 25/8*8 = 25): |26 - 25| = 1 cyc/unit.
            "lane_carried_along_vs_dash_comb": abs(
                float(freq_along_for_regime(32, "lane_carried")) - DASH_COMB_FREQ_CYC_PER_UNIT),
            # lane_offloaded regime: OWED to the next-run A/B (no measured optimum yet) -> 0 residual
            # recorded with the ASSUMED_AWAITING_VERIFICATION anchor carrying the honest status.
            "lane_offloaded_along_owed_ab": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.curriculum_dsl",),
        canonical_producers=(
            "experiments/train_witness_realized_through_R_mlx.py",
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="anisotropic_basis_two_regime_allocation.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_anisotropic_basis_two_regime_allocation_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the two-regime anisotropic-basis allocation law into
    the canonical registry (mirrors ``populate_focal_gradient_concentration_equation``; latest-row-wins
    query semantics). This is the EQUATIONS leg of DAG FEED-07a (basis-match + rule-118 = the next-run
    PRIMARY arms); the DSL leg is ``curriculum_dsl.DirectionalBasisRebalance``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_anisotropic_basis_two_regime_allocation_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="anisotropic_basis_two_regime_allocation_20260707 (equations leg of DAG FEED-07a; "
              "DSL leg = DirectionalBasisRebalance)",
    )
    return eq


__all__ = [
    "ALL_CLASS_DIRECTIONAL_DSEG_DROP",
    "ALONG_TANGENT_DEFICIT_FACTOR",
    "DASH_COMB_FREQ_CYC_PER_UNIT",
    "DIRECTIONAL_BASELINE_DSEG_N600",
    "EQUATION_ID",
    "LANE_CARRIED_REFERENCE_ALONG",
    "LANE_ONLY_DIRECTIONAL_DSEG_DROP",
    "build_anisotropic_basis_two_regime_allocation_v1",
    "freq_along_for_regime",
    "populate_anisotropic_basis_two_regime_allocation_equation",
]
