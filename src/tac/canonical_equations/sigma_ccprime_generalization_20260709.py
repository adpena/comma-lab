# SPDX-License-Identifier: MIT
"""sigma_cc' GENERALIZATION law + the fragility CROSS-DERIVATION finding (task #382,
FEED-sigma-ccprime, 2026-07-09).

Two DERIVED claims, both about the per-class-pair surface tension sigma_cc' on the Chan-Vese
length term of the level-set witness (built 2026-07-07 as ``--length-sigma-matrix`` /
``tac.boundary_math.length_sigma`` / DSL ``LengthSigma``):

(A) GENERALIZATION-NOT-ADDITION (DERIVED, code-anchored — the crucible S1-vs-S2 answer).
    The multiphase-Modica-Mortola length integrand is
        L_length(sigma) = mean_{m=0}( sigma[top1,top2] * delta_eps(m) * |grad m| ).
    sigma is a per-interface MULTIPLIER on the SAME gradient channel as the incumbent scalar
    length term; sigma == 1 (uniform / 'all-ones') recovers the incumbent EXACTLY. In code the
    'all-ones' spec resolves to ``None`` and takes the pre-existing unweighted branch, so the loss
    is BYTE-IDENTICAL (not merely multiply-by-1.0). Therefore sigma_cc' GENERALIZES the existing
    length term (it does NOT ADD a new competing gradient term) — it composes without a new
    loss-share confound. ANCHOR: the existing regression test
    ``test_all_ones_matrix_bitwise_identical_to_default`` in test_length_sigma_lever.py.
    Status: VERIFIED_VIA_SOURCE_INSPECTION (byte-identity is a code-path property, tested).

(B) CROSS-DERIVATION DISAGREEMENT (MEASURED, FORMALIZATION_PENDING).
    Two INDEPENDENT laws derive sigma_cc' from the same n600 GT junction artifact:
      - Young's-angle (Herring force-balance): sigma[Road-Lane] = 0.377  (angle far from 120 deg)
      - fragility (sliver-drop fraction):     sigma[Road-Lane] = 1.029  (abundance dilutes drops)
    They DISAGREE on the flip-dominant Road-Lane pair. The angle-based Herring derivation is the
    PRINCIPLED surface-tension readout of the frozen scorer; the fragility/abundance statistic is
    a proxy. Finding: the choice of sigma derivation law is LOAD-BEARING (not a free lunch); the
    incumbent 'fitted-20260707' (angle) preset is the default treatment, and 'fragility-20260709'
    is a SECOND A/B arm (let math arbitrate), NOT an asserted improvement. Notably, fragility USES
    the 19.3% arc>=180 slivers the angle-fit DISCARDS. Status: ASSUMED_AWAITING_VERIFICATION (the
    n600 through-R A/B of either arm is the OWED anchor; both are advisory geometry, NOT scores).

means != ends: advisory geometry calibration, NOT a score; pointer contest-CPU 0.19110 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

GENERALIZATION_EQUATION_ID = "sigma_ccprime_length_generalization_v1"
CROSS_DERIVATION_EQUATION_ID = "sigma_ccprime_fragility_cross_derivation_v1"

_UTC = "2026-07-09T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_SIGMA_JSON = "experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json"
_BUILD_MEMO = ".omx/research/sigma_ccprime_build_20260709.md"

# MEASURED cross-derivation anchors (deterministic reductions of _SIGMA_JSON; geomean-1 gauge).
SIGMA_ROAD_LANE_YOUNGS = 0.377            # Herring angle-inversion (junction_young_angle_sigma_fit_v1)
SIGMA_ROAD_LANE_FRAGILITY = 1.028689074683137  # exp(-f) drop-fraction, geomean-1 (this build)
SIGMA_LANE_UNDRIVABLE_FRAGILITY = 0.7101432121853161  # thin-sliver interface, lowered by both laws


def build_sigma_ccprime_length_generalization_v1() -> CanonicalEquation:
    """(A) The GENERALIZATION law: sigma_cc' is a per-interface multiplier that recovers the
    incumbent length term at sigma==1 (byte-identical code path) — it generalizes, not adds."""
    anchor = EmpiricalAnchor(
        anchor_id="sigma_ccprime_byte_identity_all_ones_20260709",
        measurement_utc=_UTC,
        inputs={"loss": "L_length(sigma)=mean(sigma[top1,top2]*delta_eps(m)*|grad m|) on {m=0}",
                "control": "spec 'all-ones' -> resolver returns None -> pre-existing unweighted branch"},
        predicted_output={"sigma_eq_1_recovers_incumbent": "byte-identical (code path, not *1.0)"},
        empirical_output={
            "classification": "GENERALIZATION (same gradient channel), NOT ADDITION",
            "anchor_test": "test_length_sigma_lever.py::test_all_ones_matrix_bitwise_identical_to_default",
            "consequence": "composes with no new loss-share confound (S1/S2 crucible answer)"},
        residual=0.0,
        source_artifact="experiments/train_levelset_witness_realized_through_R_mlx.py:_eikonal_length_mlx",
        measurement_method="source inspection + byte-identity regression test (existing)",
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BUILD_MEMO,
            reactivation_criteria="none — a code-path property (byte-identity), stable under refactor guard",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=GENERALIZATION_EQUATION_ID,
        name=("sigma_cc' length generalization: per-interface tension multiplier recovers the "
              "incumbent scalar length term at sigma==1 (byte-identical) — generalizes, not adds"),
        one_line_summary=("sigma_cc' is a multiplier on the existing length gradient channel; "
                          "uniform sigma == the incumbent, so it composes without a new loss term."),
        latex_form=(r"L_{\mathrm{len}}(\sigma)=\!\!\int_{\{m=0\}}\!\!\sigma_{c c'}\,"
                    r"\delta_\epsilon(m)\,|\nabla m|\,;\quad \sigma\equiv 1\Rightarrow L_{\mathrm{len}}^{\mathrm{incumbent}}"),
        python_callable_module_path=(
            "tac.boundary_math.length_sigma:resolve_length_sigma_matrix"),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": ("--length-sigma-matrix (default all-ones = byte-identical); DSL LengthSigma; "
                      "GENERALIZATION classification is the crucible S1-vs-S2 answer"),
            "measurement_axis": ["macOS-CPU advisory"]},
        units_in={"sigma_cc_prime": "relative_surface_tension_geomean_1"},
        units_out={"length_loss": "dimensionless_regularizer"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"byte_identity_at_sigma_1": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.curriculum_dsl:LengthSigma",
                             "tac.boundary_math.length_sigma"),
        canonical_producers=("tac.boundary_math.length_sigma:derive_fragility_sigma_from_junction_fit",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BUILD_MEMO, reactivation_criteria="none (code-path property)",
            measurement_axis=_ADVISORY, hardware_substrate="apple_m5_max_cpu"),
    )


def build_sigma_ccprime_fragility_cross_derivation_v1() -> CanonicalEquation:
    """(B) The CROSS-DERIVATION finding: Young's-angle and fragility disagree on Road-Lane
    (0.377 vs 1.029) -> the angle-based Herring derivation is load-bearing. FORMALIZATION_PENDING."""
    anchor = EmpiricalAnchor(
        anchor_id="sigma_ccprime_youngs_vs_fragility_20260709",
        measurement_utc=_UTC,
        inputs={"artifact": _SIGMA_JSON,
                "youngs_law": "sin-ratio inversion of triple-junction angles (drops arc>=180)",
                "fragility_law": "sigma=exp(-k*f), f=dropped/(clean+dropped), geomean-1, k=1"},
        predicted_output={"pre_registered": "if the two laws agree, sigma choice is a free lunch"},
        empirical_output={
            "sigma_road_lane_youngs": SIGMA_ROAD_LANE_YOUNGS,
            "sigma_road_lane_fragility": SIGMA_ROAD_LANE_FRAGILITY,
            "disagreement": "Young's LOWERS Road-Lane (0.377), fragility does NOT (1.029, abundance dilutes)",
            "agreement": "both LOWER the thin Lane-Undrivable sliver (Young's 0.738, fragility 0.710)",
            "verdict": ("sigma derivation law is LOAD-BEARING; angle-based Herring is the principled "
                        "readout, fragility is a proxy that reuses the arc>=180 slivers Young's drops; "
                        "fitted-20260707 is default treatment, fragility-20260709 is a 2nd A/B arm"),
        },
        residual=0.0,
        source_artifact=_SIGMA_JSON,
        measurement_method="tac.boundary_math.length_sigma:derive_fragility_sigma_from_junction_fit (deterministic)",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BUILD_MEMO,
            reactivation_criteria=("OWED: the n600 through-R A/B of fitted-20260707 vs "
                                   "fragility-20260709 vs all-ones, junction-local d_seg attribution"),
            measurement_axis=_ADVISORY, hardware_substrate="apple_m5_max_cpu"),
    )
    return CanonicalEquation(
        equation_id=CROSS_DERIVATION_EQUATION_ID,
        name=("sigma_cc' cross-derivation: Young's-angle vs fragility disagree on Road-Lane "
              "(0.377 vs 1.029) -> the angle-based Herring derivation is load-bearing"),
        one_line_summary=("Two independent sigma derivations disagree on the flip-dominant pair; "
                          "the principled angle law is the default, fragility is a 2nd A/B arm."),
        latex_form=(r"\sigma^{\mathrm{frag}}_{c c'}=\exp(-k\,f_{c c'}),\;"
                    r"f_{c c'}=\tfrac{\#\{arc\ge180\}}{\#\{all\}}\;\neq\;\sigma^{\mathrm{Young}}_{c c'}"),
        python_callable_module_path=(
            "tac.boundary_math.length_sigma:derive_fragility_sigma_from_junction_fit"),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": ("--length-sigma-matrix fragility-20260709 (2nd A/B arm; DSL LengthSigma); "
                      "FORMALIZATION_PENDING until its n600 A/B lands"),
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "advisory geometry; FORMALIZATION_PENDING; NOT a score"},
        units_in={"f_cc_prime": "sliver_drop_fraction", "k": "nats_per_unit_fraction"},
        units_out={"sigma_cc_prime": "relative_surface_tension_geomean_1"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "road_lane_cross_derivation_disagreement_abs":
                abs(SIGMA_ROAD_LANE_YOUNGS - SIGMA_ROAD_LANE_FRAGILITY)},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.curriculum_dsl:LengthSigma",),
        canonical_producers=("tac.boundary_math.length_sigma:derive_fragility_sigma_from_junction_fit",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_BUILD_MEMO,
            reactivation_criteria="the sigma-arm A/B (fitted vs fragility vs all-ones)",
            measurement_axis=_ADVISORY, hardware_substrate="apple_m5_max_cpu"),
    )


def populate_sigma_ccprime_equations(*, path=None, lock_path=None, agent=None, subagent_id=None):
    """Explicit registration (matches the solver-pack pattern; NOT an import side-effect). The
    generalization equation is VERIFIED; the cross-derivation equation is FORMALIZATION_PENDING."""
    from tac.canonical_equations.registry import register_canonical_equation

    eqs = [build_sigma_ccprime_length_generalization_v1(),
           build_sigma_ccprime_fragility_cross_derivation_v1()]
    for eq in eqs:
        register_canonical_equation(
            eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
            notes="sigma_ccprime_build_20260709 (task #382 FEED-sigma-ccprime; equations leg)")
    return eqs


__all__ = [
    "CROSS_DERIVATION_EQUATION_ID",
    "GENERALIZATION_EQUATION_ID",
    "SIGMA_LANE_UNDRIVABLE_FRAGILITY",
    "SIGMA_ROAD_LANE_FRAGILITY",
    "SIGMA_ROAD_LANE_YOUNGS",
    "build_sigma_ccprime_fragility_cross_derivation_v1",
    "build_sigma_ccprime_length_generalization_v1",
    "populate_sigma_ccprime_equations",
]
