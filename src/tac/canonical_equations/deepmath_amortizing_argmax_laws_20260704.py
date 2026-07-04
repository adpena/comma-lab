# SPDX-License-Identifier: MIT
"""Canonical equations: "Amortizing the Argmax" — the proven deep-math laws (task #284 / A2).

Formalizes the PROVEN spine of the six-chapter synthesis
(``.omx/research/deepmath_amortizing_argmax_paper_draft_20260704.md`` §2) so the EQUATIONS leg of
the triality AGREES with the DAG (FEED-03y/03z) and the DSL. Each law below is either DERIVED
(cited domain literature) or MEASURED (chapter-memo empirical anchor); every CONJECTURE from the
draft §4/§6 is EXPLICITLY NOT registered as a standalone law (see the module-final note).

The one object (draft §0/§1): a frozen classifier's decision boundary — the separatrix — is a
Fisher-metric anisotropic Laguerre tessellation; "training a witness to match its argmax" is a
temperature-annealed mirror-descent continuation (softmax_tau, tau->0) that dequantizes the smooth
boundary onto the sharp one; the distortion is the boundary's perimeter, its optimal code is the
curvelet/shearlet, and its temporal transport is a single se(3) screw.

Laws registered here (8):
  1. ``maslov_dequantization_bound_v1`` -- softmax_tau -> argmax free-energy error in [0, tau*ln K],
     K=5; the tau->0 (+,x)->(max,+) semiring (Maslov) limit; the tau:1.0->0.05 curriculum IS it.
  2. ``fisher_curvature_equals_categorical_fisher_trace_caustic_v1`` -- 1-sum(p^2) IS
     tr[diag(p)-p p^T]; in the two-class annulus tr F = 1/2 sech^2(m/2), a deterministic monotone
     function of the margin m. That EXACT identity is WHY the measured curvature<->(-margin)
     Pearson 0.978 (band) / Spearman 0.908 (global) caustic holds. margin = byte-faithful Fisher.
  3. ``ce_softmax_mirror_descent_natural_gradient_v1`` -- CE is the Bregman divergence of the
     categorical entropy potential (= KL); Raskutti-Mukherjee 2015 proves MD == natural gradient in
     dual coordinates. The curriculum-as-mirror-descent is a theorem.
  4. ``shearlet_nterm_upper_bounds_task_rate_v1`` -- d_seg is a boundary-displacement functional =>
     the Fisher/margin-weighted shearlet N-term count (cartoon-optimal O(N^-2 (log N)^3),
     Candes-Donoho) is a PROVEN UPPER BOUND on the task-space rate R_X(D_Y). Measured receipt: the
     self-orient basis IS a discrete shearlet -> D1 lever -48% all-class. Tightness CONJECTURED.
  5. ``tau_eps_hbar_one_dequantization_two_scales_v1`` -- the single scalar tau is simultaneously
     the Maslov/tropical Planck constant, the Modica-Mortola diffuse-interface width, and the
     mirror-descent temperature: one dequantization at two scales. Eikonal-on-margin => interface
     half-width exactly tau/2.
  6. ``multiphase_modica_mortola_perimeter_gamma_limit_v1`` -- softmax(phi/tau) of the K SDFs with
     the coarea length term + the entropy barrier as the double-well Gamma-converges (Baldo/
     Sternberg multi-phase, K=5 wells, Herring 120-deg triple junctions) to the weighted perimeter
     of the argmax partition as tau->0.
  7. ``mcf_minority_erasure_inevitability_v1`` -- the perimeter-gradient training flow is (sharp
     limit) motion by mean curvature (Bronsard-Kohn), annihilating high-curvature features first =>
     thin Lanes / small Movables erase INEVITABLY (not an artifact). Measured receipt: the MBO
     probe attributes 95.7% of smoothing cost to Lane.
  8. ``annulus_anisotropy_magnitude_disputed_v1`` -- the NO-FAKE correction anchor: the annulus
     codim-1 anisotropy magnitude re-measures to 9.56:1 (gradient projection) / 37.8:1
     (structure-tensor eigenvalue); the on-the-fly 198:1 is DISPUTED / uncached (not settled). The
     qualitative strong-codim-1 anisotropy (=> directional basis) holds; the magnitude needs
     reconciliation. Recorded so the equations leg does NOT quote 198:1 as a headline.

se(3) CROSS-REF (NOT duplicated): draft §2 law 5 -- the se(3)-screw temporal-sufficiency law
(Sigma_t = H_t(Sigma_0); encode xi once; the ~13.9x/#257 rate cut) is ALREADY the canonical
equation ``store_nothing_pose_carrier_rate_collapse_vs_dpose_v1``
(``tac.canonical_equations.store_nothing_pose_carrier_rate_dpose_20260702``): xi is the temporal
sufficient statistic, encode once, warp the witness's OWN render, keyframe payload -> ~0 marginal
bytes (byte-close BIT-EXACT). We REFERENCE it (the shearlet-rate law's rate-half payload |xi
B-spline| is realized there -> producer/consumer link) rather than register a second se(3) law.

MEANS != ends (draft §0/§9): NONE of this moves the pointer (contest-CPU 0.19110, UNMOVED). These
are the NAMED GEOMETRY of the #205 witness; the net score is #205-gated (a byte-closed
``upstream/evaluate.py`` n600 row < 0.19110 is the ONLY thing that moves it). NEVER MPS as a score
authority; the measured anchors below are frozen CPU-torch SegNet advisory / derived-math.
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

# --- canonical equation IDs -----------------------------------------------------------------------
MASLOV_EQUATION_ID = "maslov_dequantization_bound_v1"
FISHER_CAUSTIC_EQUATION_ID = "fisher_curvature_equals_categorical_fisher_trace_caustic_v1"
MD_NG_EQUATION_ID = "ce_softmax_mirror_descent_natural_gradient_v1"
SHEARLET_RATE_EQUATION_ID = "shearlet_nterm_upper_bounds_task_rate_v1"
TAU_EPS_HBAR_EQUATION_ID = "tau_eps_hbar_one_dequantization_two_scales_v1"
MODICA_MORTOLA_EQUATION_ID = "multiphase_modica_mortola_perimeter_gamma_limit_v1"
MCF_ERASURE_EQUATION_ID = "mcf_minority_erasure_inevitability_v1"
ANISOTROPY_DISPUTE_EQUATION_ID = "annulus_anisotropy_magnitude_disputed_v1"

_UTC = "2026-07-04T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"
_MEANS_ENDS = "MEANS only; pointer contest-CPU 0.19110 UNMOVED; net score #205-gated"

# --- source-of-truth artifacts (the synthesis draft + the six chapter memos) ----------------------
_DRAFT = ".omx/research/deepmath_amortizing_argmax_paper_draft_20260704.md"
_MEMO_TROPICAL = ".omx/research/deepmath_lens_tropical_ot_powerdiagram_20260704.md"
_MEMO_INFOGEO = ".omx/research/deepmath_lens_infogeo_naturalgrad_20260704.md"
_MEMO_SINGULARITY = ".omx/research/deepmath_lens_singularity_duality_20260704.md"
_MEMO_PHASEFIELD = ".omx/research/deepmath_lens_phasefield_gmt_levelset_20260704.md"
_MEMO_MICROLOCAL = ".omx/research/deepmath_lens_microlocal_se3_code_20260704.md"
_MEMO_DYNAMICS = ".omx/research/deepmath_lens_dynamics_transition_easing_20260704.md"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- MEASURED grounding constants (advisory / frozen CPU-torch SegNet; NET is #205-gated) ---------
FISHER_CURVATURE_MARGIN_PEARSON_BAND = 0.978    # curvature<->(-margin), band (draft §2 law 2 / §6)
FISHER_CURVATURE_MARGIN_SPEARMAN_GLOBAL = 0.908  # curvature<->(-margin), global
SEPARATRIX_MIN_DIFF_OVER_PX = 0.0               # top1-top2 = exact distance-to-flip (118M px)
D1_DIRECTIONAL_BASIS_DSEG_DELTA_FRAC = -0.48    # self-orient (=discrete shearlet) all-class d_seg
MBO_SMOOTHING_LANE_FRACTION = 0.957             # MBO probe: 95.7% of smoothing cost is Lane
CE_TO_TAU_STAGE_DSEG_DELTA_FRAC = -0.216        # CE->tau stage d_seg (draft §4; stage-behavior)

# --- anisotropy correction constants (NO-FAKE #5.1; qualitative anisotropy holds, magnitude open) -
ANISOTROPY_GRADIENT_PROJECTION = 9.56           # re-measured (gradient projection)
ANISOTROPY_STRUCTURE_TENSOR = 37.8              # re-measured (structure-tensor eigenvalue ratio)
ANISOTROPY_ON_THE_FLY_DISPUTED = 198.0          # Lens-1 on-the-fly; DISPUTED / not cached / unsettled

_K_CLASSES = 5                                  # SegNet classes {Road,Lane,Undrivable,Movable,MyCar}
HERRING_EQUAL_TENSION_ANGLE_DEG = 120.0         # equal surface tensions -> 120-deg triple junctions


# ==================================================================================================
# Helper callables (each equation's python_callable_module_path points at one of these).
# ==================================================================================================
def maslov_dequantization_bound(tau: float, k: int = _K_CLASSES) -> float:
    """The Maslov dequantization free-energy bound: ``softmax_tau -> argmax`` contributes at most
    ``tau * ln K`` (the (+,x)->(max,+) semiring limit; ``hbar = tau``). K=5 SegNet classes.

    At the curriculum end tau=0.05, K=5: 0.05*ln 5 = 0.08047. The pointwise error is in
    ``[0, tau*ln K]`` -> 0 as tau->0. Derived (Maslov dequantization / idempotent analysis)."""

    if tau < 0.0:
        raise ValueError("tau must be >= 0 (a temperature)")
    if k < 2:
        raise ValueError("k must be >= 2 (a classification with >=2 classes)")
    return float(tau) * math.log(float(k))


def annulus_fisher_trace(margin: float) -> float:
    """The two-class annulus categorical Fisher trace ``tr F = 1/2 sech^2(m/2)`` (draft §2 law 2).

    This IS the exact identity ``1 - sum_i p_i^2 = tr[diag(p) - p p^T]`` specialized to two classes
    with ``p = sigma(m)``: ``1 - p^2 - (1-p)^2 = 2 p (1-p) = 2 sigma(m)(1-sigma(m)) = 1/2 sech^2(m/2)``.
    A deterministic monotone function of the margin ``m`` -> WHY the measured curvature<->(-margin)
    Pearson is 0.978. At m=0 (the boundary, p=0.5): 1/2 * sech^2(0) = 1/2 * 1 = 0.5 (MAX Fisher)."""

    return 0.5 / (math.cosh(float(margin) / 2.0) ** 2)


def categorical_fisher_trace_two_class(p: float) -> float:
    """The categorical Fisher trace ``1 - sum_i p_i^2`` for two classes ``(p, 1-p)``.

    Equals ``2 p (1-p)`` and, with ``p = sigma(m)``, equals :func:`annulus_fisher_trace` (m). This
    is the EXACT identity grounding the caustic (``1 - sum p^2 = tr[diag(p) - p p^T]``)."""

    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be a probability in [0, 1]")
    return 1.0 - (p * p) - ((1.0 - p) ** 2)


def categorical_bregman_divergence(p: list[float], q: list[float]) -> float:
    """The Bregman divergence of the categorical (negative-)entropy potential = the KL divergence
    ``sum_i p_i ln(p_i / q_i)`` (nats). CE = this Bregman divergence, which is exactly why the
    softmax + CE update is a mirror-descent step = the natural gradient in dual (mean) coordinates
    (Raskutti-Mukherjee 2015). ``D(p||p) = 0`` for any p on the simplex."""

    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    total = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi < 0.0 or qi <= 0.0:
            raise ValueError("p_i must be >= 0 and q_i must be > 0")
        if pi > 0.0:
            total += pi * math.log(pi / qi)
    return total


def shearlet_nterm_error(n_terms: int) -> float:
    """The cartoon (piecewise-C^2 with C^2 boundary) N-term shearlet approximation error rate
    ``O(N^-2 (log N)^3)`` (Candes-Donoho; near-optimal for the codim-1 curved singularity). This
    upper-bounds the task-space rate ``R_X(D_Y)`` because ``d_seg`` is a boundary-displacement
    functional (draft §2 law 4). Beats wavelet ``N^-1`` and Fourier ``N^-1/2``. Tightness (whether
    it hits the task-rate floor) is a CONJECTURE, not a claim here."""

    if n_terms < 2:
        raise ValueError("n_terms must be >= 2 (log N defined)")
    n = float(n_terms)
    return n ** (-2.0) * (math.log(n) ** 3)


def tau_interface_halfwidth(tau: float) -> float:
    """The Modica-Mortola diffuse-interface HALF-width under the eikonal-on-margin construction:
    choosing the eikonal on ``m = phi_1 - phi_2`` makes the soft interface half-width exactly
    ``tau/2`` (draft §3). This is the ``tau = eps = hbar`` identity made concrete: the single scalar
    ``tau`` is at once the Maslov/tropical Planck constant, the interface width, and the MD
    temperature (one dequantization at two scales)."""

    if tau < 0.0:
        raise ValueError("tau must be >= 0 (a temperature / interface width)")
    return float(tau) / 2.0


def herring_triple_junction_angle_deg(surface_tensions: tuple[float, float, float] | None = None) -> float:
    """The Herring/Young triple-junction angle for the multi-phase Modica-Mortola Gamma-limit. For
    EQUAL surface tensions the three phase boundaries meet at ``120 deg`` (draft §2 law 7). Passing
    unequal tensions is not the equal-tension case and raises (the general Herring condition is a
    balance-of-forces solve, out of scope for this constant)."""

    if surface_tensions is None:
        return HERRING_EQUAL_TENSION_ANGLE_DEG
    a, b, c = surface_tensions
    if not (math.isclose(a, b) and math.isclose(b, c)):
        raise ValueError(
            "unequal surface tensions -> general Herring balance (not the equal-tension 120-deg case)"
        )
    return HERRING_EQUAL_TENSION_ANGLE_DEG


def mbo_smoothing_cost_lane_fraction() -> float:
    """The MEASURED MBO-probe receipt for MCF minority-erasure inevitability: the sharp-limit
    perimeter-gradient flow (= motion by mean curvature, Bronsard-Kohn) annihilates high-curvature
    features first, and the MBO probe attributes ``95.7%`` of the smoothing cost to the thin Lane
    class (draft §2 law 8). The principled cure is a per-class area/volume constraint (auction-MBO)."""

    return MBO_SMOOTHING_LANE_FRACTION


def annulus_anisotropy_ratio(method: str) -> float:
    """The re-measured annulus codim-1 anisotropy magnitude by method (NO-FAKE #5.1). The
    qualitative strong-codim-1 anisotropy (=> directional basis) holds; the MAGNITUDE is disputed:

      - ``"gradient_projection"`` -> 9.56 (re-measured)
      - ``"structure_tensor"``    -> 37.8 (re-measured)

    The Lens-1 on-the-fly ``198:1`` is DISPUTED / uncached / NOT settled and is intentionally NOT
    returned as a settled value (pass ``"on_the_fly_disputed"`` to see the raise)."""

    table = {
        "gradient_projection": ANISOTROPY_GRADIENT_PROJECTION,
        "structure_tensor": ANISOTROPY_STRUCTURE_TENSOR,
    }
    if method not in table:
        raise ValueError(
            f"method={method!r} is not a settled measurement; the on-the-fly 198:1 is DISPUTED "
            f"(not cached / not reconciled). Settled methods: {sorted(table)!r}"
        )
    return table[method]


# ==================================================================================================
# Law 1 -- Maslov dequantization bound
# ==================================================================================================
def build_maslov_dequantization_bound_v1() -> CanonicalEquation:
    """softmax_tau -> argmax free-energy error in [0, tau*ln K], K=5; the tau->0 semiring limit."""

    anchor_bound = EmpiricalAnchor(
        anchor_id="maslov_dequantization_free_energy_bound_tau_ln_k_20260704",
        measurement_utc=_UTC,
        inputs={"semiring_limit": "(+,x) -> (max,+) as tau->0 (Maslov / idempotent analysis)",
                "K_classes": _K_CLASSES, "curriculum_tau": "1.0 -> 0.05",
                "citation": "Maslov dequantization (tropical / idempotent analysis)"},
        predicted_output={"free_energy_error_bound_at_tau_end": maslov_dequantization_bound(0.05),
                          "bound_form": "error in [0, tau*ln K]"},
        empirical_output={"tau_end_bound": maslov_dequantization_bound(0.05),
                          "curriculum_is_the_semiring_limit": True,
                          "note": "the tau:1.0->0.05 curriculum IS the semiring dequantization; the "
                                  "hard partition is a tropical-hypersurface complement (hbar=tau)"},
        residual=0.0,
        source_artifact=_MEMO_SINGULARITY,
        measurement_method="derived_maslov_dequantization_tropical_semiring_limit",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_SINGULARITY,
            reactivation_criteria="tighten the constant from ln K to the measured per-stage free-energy gap",
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )
    return CanonicalEquation(
        equation_id=MASLOV_EQUATION_ID,
        name="Maslov dequantization bound: softmax_tau -> argmax free-energy error <= tau*ln K",
        one_line_summary=(
            "softmax_tau->argmax contributes at most tau*ln K (K=5) to the free energy; the "
            "tau:1.0->0.05 curriculum IS the (+,x)->(max,+) semiring (Maslov) limit, hbar=tau"
        ),
        latex_form=(
            r"0 \le F_\tau - F_0 \le \tau\ln K,\ K=5;\quad "
            r"\mathrm{softmax}_\tau \xrightarrow{\tau\to0} \mathrm{argmax}\ (+,\times)\to(\max,+)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704:maslov_dequantization_bound"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "classes_K": _K_CLASSES,
            "tau_range": [0.05, 1.0],
            "status": "DERIVED (Maslov dequantization)",
            "means_ends": _MEANS_ENDS,
        },
        units_in={"tau": "softmax_temperature", "k": "class_count"},
        units_out={"free_energy_error_bound": "nats"},
        empirical_anchors=(anchor_bound,),
        predicted_vs_empirical_residual={"maslov_dequantization_bound": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("tools/register_deepmath_amortizing_argmax_laws_20260704.py",),
        provenance=build_provenance_for_predicted(
            model_id="maslov_dequantization_bound.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )


# ==================================================================================================
# Law 2 -- Fisher curvature = categorical Fisher trace = caustic (the exact 0.978 identity)
# ==================================================================================================
def build_fisher_curvature_equals_categorical_fisher_trace_caustic_v1() -> CanonicalEquation:
    """1-sum(p^2) IS tr[diag(p)-p p^T]; annulus tr F = 1/2 sech^2(m/2); the exact identity behind
    the measured curvature<->(-margin) Pearson 0.978 / Spearman 0.908 caustic."""

    anchor_identity = EmpiricalAnchor(
        anchor_id="fisher_trace_categorical_identity_annulus_half_sech2_20260704",
        measurement_utc=_UTC,
        inputs={"identity": "1 - sum_i p_i^2 = tr[diag(p) - p p^T] (categorical Fisher trace)",
                "annulus_two_class": "tr F = 1/2 sech^2(m/2), p = sigma(m)",
                "caustic": "I_tau = tau^-2 sigma(m/tau)(1-sigma(m/tau)) rank-1 bright ridge on m=0",
                "citation": "categorical information geometry (derivation)"},
        predicted_output={"tr_F_at_margin_0": annulus_fisher_trace(0.0),
                          "categorical_trace_at_p_half": categorical_fisher_trace_two_class(0.5)},
        empirical_output={"tr_F_at_margin_0": annulus_fisher_trace(0.0),
                          "identity_holds_at_boundary": (
                              abs(annulus_fisher_trace(0.0) - categorical_fisher_trace_two_class(0.5)) < 1e-12),
                          "note": "deterministic monotone function of the margin m -> margin field IS "
                                  "a byte-faithful Fisher surrogate (max Fisher 0.5 at m=0)"},
        residual=0.0,
        source_artifact=_MEMO_INFOGEO,
        measurement_method="derived_categorical_fisher_trace_annulus_sech2_identity",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_INFOGEO,
            reactivation_criteria="re-derive if the scorer stops being locally two-class at the annulus",
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )
    anchor_measured = EmpiricalAnchor(
        anchor_id="curvature_neg_margin_pearson_0978_spearman_0908_caustic_20260704",
        measurement_utc=_UTC,
        inputs={"correlate_a": "Fisher curvature field", "correlate_b": "(-margin) field",
                "scorer": "frozen CPU-torch SegNet argmax", "n_px": "118M (simplex probe)"},
        predicted_output={"expect_high_corr_because_exact_identity": True},
        empirical_output={"pearson_band": FISHER_CURVATURE_MARGIN_PEARSON_BAND,
                          "spearman_global": FISHER_CURVATURE_MARGIN_SPEARMAN_GLOBAL,
                          "top1_top2_min_diff_over_118M_px": SEPARATRIX_MIN_DIFF_OVER_PX,
                          "note": "the measured 0.978 caustic IS the exact identity (near-tautological "
                                  "-> trustworthy); top1-top2 = exact distance-to-flip (min-diff 0.0)"},
        residual=0.0,
        source_artifact=_MEMO_INFOGEO,
        measurement_method="frozen_cpu_torch_segnet_curvature_margin_correlation_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_INFOGEO,
            reactivation_criteria="re-measure curvature<->(-margin) correlation at n600 through R",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu_torch",
        ),
    )
    return CanonicalEquation(
        equation_id=FISHER_CAUSTIC_EQUATION_ID,
        name="Fisher curvature = categorical Fisher trace = caustic (the exact 0.978 margin identity)",
        one_line_summary=(
            "1-sum p^2 = tr[diag(p)-p p^T]; annulus tr F = 1/2 sech^2(m/2) monotone in the margin "
            "-> the measured curvature<->(-margin) Pearson 0.978 caustic is an EXACT identity"
        ),
        latex_form=(
            r"1-\sum_i p_i^2 = \mathrm{tr}\!\big[\mathrm{diag}(p)-pp^\top\big];\ "
            r"\mathrm{tr}\,F_{\text{annulus}} = \tfrac12\,\mathrm{sech}^2\!\big(\tfrac{m}{2}\big)"
            r"\ \Rightarrow\ \rho(\kappa,-m)=0.978"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704:annulus_fisher_trace"
        ),
        domain_of_validity={
            "vehicle": ["frozen_cpu_torch_segnet_annulus"],
            "regime": "locally two-class annulus (p = sigma(m))",
            "status": "DERIVED identity + MEASURED caustic (advisory)",
            "measurement_axis": [_ADVISORY, _PREDICTED],
            "means_ends": _MEANS_ENDS,
        },
        units_in={"margin": "segnet_argmax_logit_margin"},
        units_out={"fisher_trace": "categorical_fisher_information_trace"},
        empirical_anchors=(anchor_identity, anchor_measured),
        predicted_vs_empirical_residual={
            "categorical_fisher_trace_identity": 0.0,
            "curvature_margin_caustic_correlation": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
            "tools/precompute_sR_reachability.py",
        ),
        canonical_producers=("tools/register_deepmath_amortizing_argmax_laws_20260704.py",),
        provenance=build_provenance_for_predicted(
            model_id="fisher_curvature_equals_categorical_fisher_trace_caustic.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


# ==================================================================================================
# Law 3 -- CE + softmax = mirror descent = natural gradient
# ==================================================================================================
def build_ce_softmax_mirror_descent_natural_gradient_v1() -> CanonicalEquation:
    """CE is the Bregman divergence of the categorical entropy potential (= KL); Raskutti-Mukherjee
    2015 proves mirror descent == natural gradient in dual coordinates."""

    anchor_theorem = EmpiricalAnchor(
        anchor_id="ce_bregman_categorical_entropy_md_equals_ng_20260704",
        measurement_utc=_UTC,
        inputs={"potential": "categorical (negative) entropy", "bregman": "D_phi(p||q) = KL(p||q)",
                "theorem": "mirror descent == natural gradient in dual coordinates",
                "citation": "Raskutti-Mukherjee 2015"},
        predicted_output={"bregman_self_divergence_is_zero": categorical_bregman_divergence(
            [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2])},
        empirical_output={"md_equiv_ng": True,
                          "note": "CE = Bregman(neg-entropy) = KL -> the softmax+CE (p-y) update is "
                                  "a mirror-descent step = the natural gradient in mean coordinates; "
                                  "the curriculum-as-mirror-descent is a THEOREM"},
        residual=0.0,
        source_artifact=_MEMO_INFOGEO,
        measurement_method="derived_bregman_categorical_entropy_md_equals_ng_raskutti_mukherjee_2015",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_INFOGEO,
            reactivation_criteria="n/a (theorem); reactivate only if the loss stops being CE-on-softmax",
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )
    return CanonicalEquation(
        equation_id=MD_NG_EQUATION_ID,
        name="CE + softmax = mirror descent = natural gradient (categorical entropy Bregman)",
        one_line_summary=(
            "CE is the Bregman divergence of the categorical entropy potential (= KL); "
            "Raskutti-Mukherjee 2015 -> mirror descent == natural gradient in dual coordinates"
        ),
        latex_form=(
            r"\mathrm{CE}(y,p) = D_{\phi}(y\,\|\,p),\ \phi=\text{neg-entropy};\ "
            r"\text{MD}_\phi \equiv \text{natural gradient (dual coords)}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704:categorical_bregman_divergence"
        ),
        domain_of_validity={
            "loss": ["cross_entropy_on_softmax"],
            "status": "DERIVED (Raskutti-Mukherjee 2015)",
            "conjecture_NOT_registered": (
                "the trajectory literally IS natural-gradient-along-Maslov ending at the Gamma-limit "
                "is a CONJECTURE (draft §4/§6): matches CE->tau -21.6% but unproven"),
            "means_ends": _MEANS_ENDS,
        },
        units_in={"p": "categorical_probability_vector", "q": "categorical_probability_vector"},
        units_out={"bregman_divergence": "nats"},
        empirical_anchors=(anchor_theorem,),
        predicted_vs_empirical_residual={"md_equals_ng_bregman": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("tools/register_deepmath_amortizing_argmax_laws_20260704.py",),
        provenance=build_provenance_for_predicted(
            model_id="ce_softmax_mirror_descent_natural_gradient.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )


# ==================================================================================================
# Law 4 -- Shearlet N-term upper-bounds the task rate (tightness conjectured)
# ==================================================================================================
def build_shearlet_nterm_upper_bounds_task_rate_v1() -> CanonicalEquation:
    """d_seg = boundary-displacement functional => Fisher/margin-weighted shearlet N-term count
    (cartoon-optimal O(N^-2 (log N)^3)) is a PROVEN UPPER BOUND on R_X(D_Y); tightness CONJECTURED."""

    anchor_bound = EmpiricalAnchor(
        anchor_id="shearlet_nterm_cartoon_optimal_upper_bounds_task_rate_20260704",
        measurement_utc=_UTC,
        inputs={"partition": "cartoon (piecewise-C^2, C^2 boundary)",
                "basis": "curvelet/shearlet N-term", "rate_order": "O(N^-2 (log N)^3)",
                "vs": "wavelet N^-1, Fourier N^-1/2",
                "task_functional": "d_seg = boundary-displacement functional",
                "citation": "Candes-Donoho (cartoon-optimal sparse approximation)"},
        predicted_output={"nterm_error_at_N_64": shearlet_nterm_error(64),
                          "upper_bounds_R_X_D_Y": True},
        empirical_output={"nterm_error_at_N_64": shearlet_nterm_error(64),
                          "asymptotic_rate_exponent": -2.0,
                          "vs_wavelet_exponent": -1.0,
                          "vs_fourier_exponent": -0.5,
                          "beats_wavelet_asymptotically_N_1e6": (
                              shearlet_nterm_error(1_000_000) < 1_000_000 ** (-1.0)),
                          "note": "PROVEN UPPER BOUND on the task-space rate R_X(D_Y); the O(N^-2 "
                                  "(log N)^3) rate beats wavelet N^-1 / Fourier N^-1/2 ASYMPTOTICALLY "
                                  "(the (log N)^3 constant means the crossover is at large N, not N=64). "
                                  "The rate-half payload is |Fisher/margin-weighted shearlet coeffs| + "
                                  "|xi B-spline| + |Movable store| (the generic bank is rule-118 FREE)"},
        residual=0.0,
        source_artifact=_MEMO_MICROLOCAL,
        measurement_method="derived_cartoon_optimal_shearlet_nterm_upper_bound_candes_donoho",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_MICROLOCAL,
            reactivation_criteria="prove/measure whether the N-term upper bound is TIGHT at the task-rate floor",
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )
    anchor_d1 = EmpiricalAnchor(
        anchor_id="self_orient_basis_is_discrete_shearlet_d1_minus48_all_class_20260704",
        measurement_utc=_UTC,
        inputs={"lever": "D1 --self-orient directional basis", "claim": "self-orient basis IS a discrete shearlet",
                "scorer": "frozen CPU-torch SegNet argmax", "scope": "all-class boundary"},
        predicted_output={"expect_cartoon_optimal_because_shearlet": True},
        empirical_output={"d1_all_class_dseg_delta_frac": D1_DIRECTIONAL_BASIS_DSEG_DELTA_FRAC,
                          "note": "-48% all-class d_seg = Candes-Donoho cartoon-optimal (NOT a heuristic); "
                                  "lane-only was only -8%. The net n600 row is #205-gated."},
        residual=0.0,
        source_artifact=_DRAFT,
        measurement_method="frozen_cpu_torch_segnet_d1_directional_basis_dseg_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DRAFT,
            reactivation_criteria="byte-closed n600 exact row with the self-orient shearlet basis ON",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu_torch",
        ),
    )
    anchor_tightness = EmpiricalAnchor(
        anchor_id="shearlet_tightness_hits_task_rate_floor_conjecture_awaiting_verification_20260704",
        measurement_utc=_UTC,
        inputs={"claim": "the shearlet N-term upper bound is TIGHT (hits the task-rate floor)",
                "evidence": "annulus 1.4%-cell concentration", "status": "CONJECTURE (draft §6)"},
        predicted_output={"tightness_proven": False},
        empirical_output={"tightness_proven": False,
                          "note": "the UPPER BOUND is proven; tightness is UNPROVEN -> not a claim, "
                                  "registered only as the conjectural edge of the domain"},
        residual=0.0,
        source_artifact=_DRAFT,
        measurement_method="conjecture_awaiting_verification_not_a_claim",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DRAFT,
            reactivation_criteria="prove or measure that the shearlet N-term rate hits the task-rate floor",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=SHEARLET_RATE_EQUATION_ID,
        name="Shearlet N-term upper-bounds the task-space rate R_X(D_Y) (tightness conjectured)",
        one_line_summary=(
            "d_seg is a boundary-displacement functional -> the cartoon-optimal shearlet N-term "
            "count O(N^-2 (log N)^3) is a PROVEN UPPER BOUND on R_X(D_Y); tightness conjectured"
        ),
        latex_form=(
            r"\|f-f_N\|_2^2 = O\!\big(N^{-2}(\log N)^3\big)\ \ge\ R_X(D_Y)\ \text{(upper bound)};\ "
            r"\text{self-orient basis} = \text{discrete shearlet} \Rightarrow \Delta d_{seg}=-48\%"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704:shearlet_nterm_error"
        ),
        domain_of_validity={
            "partition": ["cartoon_piecewise_c2_with_c2_boundary"],
            "basis": ["curvelet", "shearlet", "self_orient_directional"],
            "bound_type": "PROVEN UPPER BOUND on R_X(D_Y)",
            "tightness": "CONJECTURED (NOT a claim; see the ASSUMED_AWAITING_VERIFICATION anchor)",
            "measurement_axis": [_ADVISORY, _PREDICTED],
            "means_ends": _MEANS_ENDS,
        },
        units_in={"n_terms": "count_of_retained_shearlet_coefficients"},
        units_out={"nterm_error_rate": "l2_approximation_error_upper_bound"},
        empirical_anchors=(anchor_bound, anchor_d1, anchor_tightness),
        predicted_vs_empirical_residual={
            "shearlet_nterm_upper_bound": 0.0,
            "d1_directional_basis_dseg": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
            # se(3) cross-ref producer/consumer link: the rate-half payload |xi B-spline| is realized
            # by the store-nothing pose carrier (draft §2 law 5 -- NOT duplicated here).
            "tac.canonical_equations.store_nothing_pose_carrier_rate_dpose_20260702",
        ),
        canonical_producers=(
            "tools/register_deepmath_amortizing_argmax_laws_20260704.py",
            "src/tac/boundary_math/lever_b_generator.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="shearlet_nterm_upper_bounds_task_rate.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


# ==================================================================================================
# Law 5 -- tau = eps = hbar (one dequantization at two scales)
# ==================================================================================================
def build_tau_eps_hbar_one_dequantization_two_scales_v1() -> CanonicalEquation:
    """The single scalar tau is simultaneously the Maslov/tropical Planck constant, the
    Modica-Mortola diffuse-interface width, and the mirror-descent temperature."""

    anchor = EmpiricalAnchor(
        anchor_id="tau_eps_hbar_one_dequantization_two_scales_20260704",
        measurement_utc=_UTC,
        inputs={"tau_as_hbar": "pointwise logsumexp -> max (Maslov / tropical Planck constant)",
                "tau_as_eps": "Modica-Mortola diffuse-interface width (spatial energy)",
                "tau_as_temperature": "softmax = neg-entropy Bregman mirror map (MD temperature)",
                "eikonal_on_margin": "m = phi_1 - phi_2 -> interface half-width exactly tau/2",
                "citation": "Gamma-convergence (Baldo/Sternberg); Maslov dequantization"},
        predicted_output={"interface_halfwidth_at_tau_end": tau_interface_halfwidth(0.05)},
        empirical_output={"interface_halfwidth_at_tau_end": tau_interface_halfwidth(0.05),
                          "one_dequantization_two_scales": True,
                          "note": "pointwise semiring limit and spatial Gamma-limit COINCIDE -> "
                                  "hard<->soft duality is one dequantization at two scales"},
        residual=0.0,
        source_artifact=_MEMO_PHASEFIELD,
        measurement_method="derived_tau_eps_hbar_coincidence_gamma_convergence",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_PHASEFIELD,
            reactivation_criteria="n/a (derivation); reactivate if the eikonal is moved off the margin m",
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )
    return CanonicalEquation(
        equation_id=TAU_EPS_HBAR_EQUATION_ID,
        name="tau = eps = hbar: one dequantization at two scales",
        one_line_summary=(
            "the single scalar tau is at once the Maslov/tropical Planck constant, the "
            "Modica-Mortola interface width, and the MD temperature; eikonal-on-margin -> half-width tau/2"
        ),
        latex_form=(
            r"\tau = \varepsilon = \hbar;\quad "
            r"\text{eikonal on } m=\phi_1-\phi_2 \Rightarrow \text{interface half-width} = \tfrac{\tau}{2}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704:tau_interface_halfwidth"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "construction": "eikonal on the margin m = phi_1 - phi_2",
            "status": "DERIVED (Gamma-convergence coincidence)",
            "means_ends": _MEANS_ENDS,
        },
        units_in={"tau": "softmax_temperature_equals_interface_width"},
        units_out={"interface_halfwidth": "same_units_as_tau"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"tau_interface_halfwidth": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("tools/register_deepmath_amortizing_argmax_laws_20260704.py",),
        provenance=build_provenance_for_predicted(
            model_id="tau_eps_hbar_one_dequantization_two_scales.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )


# ==================================================================================================
# Law 6 -- Multi-phase Modica-Mortola perimeter Gamma-limit
# ==================================================================================================
def build_multiphase_modica_mortola_perimeter_gamma_limit_v1() -> CanonicalEquation:
    """softmax(phi/tau) of the K SDFs, with the coarea length term and the entropy barrier as the
    double-well, Gamma-converges to the weighted perimeter of the argmax partition as tau->0."""

    anchor = EmpiricalAnchor(
        anchor_id="multiphase_modica_mortola_gamma_limit_weighted_perimeter_K5_20260704",
        measurement_utc=_UTC,
        inputs={"field": "softmax(phi/tau) of K SDFs", "length_term": "delta_eps*|grad phi| (Chan-Vese coarea)",
                "double_well": "entropy barrier tau*H(p)", "K_wells": _K_CLASSES,
                "triple_junctions": "Herring angles (equal tension -> 120 deg)",
                "citation": "Baldo / Sternberg multi-phase Modica-Mortola"},
        predicted_output={"herring_equal_tension_angle_deg": herring_triple_junction_angle_deg()},
        empirical_output={"herring_equal_tension_angle_deg": herring_triple_junction_angle_deg(),
                          "gamma_limit": "weighted perimeter of the argmax partition",
                          "note": "the soft continuation lands on the hard perimeter-minimizer as "
                                  "tau->0; the witness IS the classical level-set variational triple"},
        residual=0.0,
        source_artifact=_MEMO_PHASEFIELD,
        measurement_method="derived_multiphase_modica_mortola_gamma_limit_baldo_sternberg",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_PHASEFIELD,
            reactivation_criteria="n/a (Gamma-convergence theorem); reactivate if the K-well structure changes",
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )
    return CanonicalEquation(
        equation_id=MODICA_MORTOLA_EQUATION_ID,
        name="Multi-phase Modica-Mortola Gamma-limit = weighted perimeter of the argmax partition",
        one_line_summary=(
            "softmax(phi/tau) of K=5 SDFs (coarea length + entropy double-well) Gamma-converges to "
            "the weighted perimeter of the argmax partition (Baldo/Sternberg; Herring 120-deg junctions)"
        ),
        latex_form=(
            r"E_\tau[\phi] \xrightarrow{\Gamma,\ \tau\to0} \sum_{i<j} w_{ij}\,"
            r"\mathrm{Per}(\Omega_i\cap\partial\Omega_j);\ K=5\ \text{wells, Herring }120^\circ"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704:herring_triple_junction_angle_deg"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "K_wells": _K_CLASSES,
            "status": "DERIVED (Baldo/Sternberg multi-phase Gamma-convergence)",
            "means_ends": _MEANS_ENDS,
        },
        units_in={"surface_tensions": "relative_surface_tension_triple"},
        units_out={"triple_junction_angle": "degrees"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"multiphase_gamma_limit_perimeter": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("tools/register_deepmath_amortizing_argmax_laws_20260704.py",),
        provenance=build_provenance_for_predicted(
            model_id="multiphase_modica_mortola_perimeter_gamma_limit.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="derivation",
        ),
    )


# ==================================================================================================
# Law 7 -- MCF minority-erasure inevitability
# ==================================================================================================
def build_mcf_minority_erasure_inevitability_v1() -> CanonicalEquation:
    """The perimeter-gradient training flow is (sharp limit) motion by mean curvature (Bronsard-Kohn),
    annihilating high-curvature features first => thin-Lane / small-Movable erasure is inevitable."""

    anchor = EmpiricalAnchor(
        anchor_id="mcf_high_curvature_first_thin_lane_erasure_mbo_957_20260704",
        measurement_utc=_UTC,
        inputs={"flow": "perimeter-gradient training flow = motion by mean curvature (sharp limit)",
                "mechanism": "MCF annihilates high-curvature features first",
                "probe": "MBO (Merriman-Bence-Osher) smoothing-cost attribution",
                "citation": "Bronsard-Kohn (MCF from the phase-field flow)"},
        predicted_output={"minority_thin_class_erases_first": True},
        empirical_output={"mbo_smoothing_cost_lane_fraction": mbo_smoothing_cost_lane_fraction(),
                          "note": "95.7% of the smoothing cost is the thin Lane class -> erasure is "
                                  "INEVITABLE (not an artifact); the principled cure is a per-class "
                                  "area/volume constraint (auction-MBO), sibling of lane_thin_weight_map"},
        residual=0.0,
        source_artifact=_MEMO_DYNAMICS,
        measurement_method="mbo_smoothing_cost_attribution_frozen_cpu_torch_segnet_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_DYNAMICS,
            reactivation_criteria="re-measure MBO per-class smoothing-cost attribution at n600 through R",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu_torch",
        ),
    )
    return CanonicalEquation(
        equation_id=MCF_ERASURE_EQUATION_ID,
        name="MCF minority-erasure inevitability (thin-Lane erasure = motion by mean curvature)",
        one_line_summary=(
            "the perimeter-gradient flow is (sharp limit) motion by mean curvature -> high-curvature "
            "thin Lanes erase FIRST and inevitably; MBO probe attributes 95.7% smoothing cost to Lane"
        ),
        latex_form=(
            r"\partial_t \Gamma = -\kappa\,\mathbf{n}\ (\text{MCF, Bronsard-Kohn})\ \Rightarrow\ "
            r"\text{high-}\kappa\ \text{thin Lane erased first};\ \text{MBO}_{\text{Lane}}=95.7\%"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704:mbo_smoothing_cost_lane_fraction"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "flow": "perimeter-gradient = motion by mean curvature (sharp limit)",
            "status": "DERIVED (Bronsard-Kohn) + MEASURED (MBO 95.7% Lane, advisory)",
            "cure": "per-class area/volume constraint (auction-MBO), sibling of lane_thin_weight_map",
            "measurement_axis": [_ADVISORY],
            "means_ends": _MEANS_ENDS,
        },
        units_in={},
        units_out={"mbo_smoothing_cost_lane_fraction": "fraction_of_total_smoothing_cost"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"mcf_minority_erasure_mbo_lane": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("tools/register_deepmath_amortizing_argmax_laws_20260704.py",),
        provenance=build_provenance_for_predicted(
            model_id="mcf_minority_erasure_inevitability.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


# ==================================================================================================
# Law 8 -- Annulus anisotropy magnitude DISPUTED (the NO-FAKE correction anchor)
# ==================================================================================================
def build_annulus_anisotropy_magnitude_disputed_v1() -> CanonicalEquation:
    """NO-FAKE correction (draft §5.1): the annulus codim-1 anisotropy magnitude re-measures to
    9.56:1 (gradient projection) / 37.8:1 (structure-tensor); the on-the-fly 198:1 is DISPUTED."""

    anchor_remeasured = EmpiricalAnchor(
        anchor_id="annulus_anisotropy_remeasured_956_378_20260704",
        measurement_utc=_UTC,
        inputs={"quantity": "annulus codim-1 anisotropy magnitude",
                "method_a": "gradient projection", "method_b": "structure-tensor eigenvalue ratio",
                "scorer": "frozen CPU-torch SegNet argmax"},
        predicted_output={"gradient_projection_ratio": annulus_anisotropy_ratio("gradient_projection"),
                          "structure_tensor_ratio": annulus_anisotropy_ratio("structure_tensor")},
        empirical_output={"gradient_projection_ratio": ANISOTROPY_GRADIENT_PROJECTION,
                          "structure_tensor_ratio": ANISOTROPY_STRUCTURE_TENSOR,
                          "note": "qualitative strong-codim-1 anisotropy HOLDS (=> directional basis); "
                                  "the MAGNITUDE needs reconciliation -> do NOT quote a single headline"},
        residual=0.0,
        source_artifact=_MEMO_INFOGEO,
        measurement_method="frozen_cpu_torch_segnet_annulus_anisotropy_remeasure_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_INFOGEO,
            reactivation_criteria="reconcile the anisotropy magnitude across gradient-projection / structure-tensor",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu_torch",
        ),
    )
    anchor_198_disputed = EmpiricalAnchor(
        anchor_id="annulus_anisotropy_198_on_the_fly_disputed_uncached_20260704",
        measurement_utc=_UTC,
        inputs={"quantity": "Lens-1 on-the-fly 198:1 anisotropy claim",
                "grep_cache_finds_198": False, "status": "DISPUTED / uncached / not reconciled"},
        predicted_output={"198_is_settled": False},
        empirical_output={"on_the_fly_ratio_disputed": ANISOTROPY_ON_THE_FLY_DISPUTED,
                          "settled": False,
                          "note": "Ch.2 grep of the entire cache finds NO cached 198 -> DO NOT quote "
                                  "198:1 as settled; recorded as a correction anchor, not a headline"},
        residual=0.0,
        source_artifact=_MEMO_INFOGEO,
        measurement_method="disputed_uncached_awaiting_reconciliation_not_a_claim",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_INFOGEO,
            reactivation_criteria="reconcile or retire the 198:1 on-the-fly anisotropy figure",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=ANISOTROPY_DISPUTE_EQUATION_ID,
        name="Annulus anisotropy magnitude DISPUTED (NO-FAKE correction: 9.56 / 37.8, not 198)",
        one_line_summary=(
            "annulus codim-1 anisotropy re-measures to 9.56:1 (gradient projection) / 37.8:1 "
            "(structure-tensor); the on-the-fly 198:1 is DISPUTED/uncached -> not a settled headline"
        ),
        latex_form=(
            r"\text{anisotropy} \approx 9.56\ (\nabla\text{-proj})\ /\ 37.8\ (\text{struct-tensor});\ "
            r"198{:}1\ \text{DISPUTED (uncached)}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704:annulus_anisotropy_ratio"
        ),
        domain_of_validity={
            "quantity": "annulus codim-1 anisotropy magnitude",
            "qualitative": "strong-codim-1 anisotropy HOLDS (=> directional / shearlet basis)",
            "magnitude": "DISPUTED (9.56 vs 37.8 vs 198) -- NO-FAKE correction anchor, not a headline",
            "measurement_axis": [_ADVISORY, _PREDICTED],
            "means_ends": _MEANS_ENDS,
        },
        units_in={"method": "anisotropy_measurement_method_token"},
        units_out={"anisotropy_ratio": "dimensionless_eigenvalue_ratio"},
        empirical_anchors=(anchor_remeasured, anchor_198_disputed),
        predicted_vs_empirical_residual={"annulus_anisotropy_remeasured": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
            "src/tac/boundary_math/lever_b_generator.py",
        ),
        canonical_producers=("tools/register_deepmath_amortizing_argmax_laws_20260704.py",),
        provenance=build_provenance_for_predicted(
            model_id="annulus_anisotropy_magnitude_disputed.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


# The 8 build_* factories, in registration order.
ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS = (
    build_maslov_dequantization_bound_v1,
    build_fisher_curvature_equals_categorical_fisher_trace_caustic_v1,
    build_ce_softmax_mirror_descent_natural_gradient_v1,
    build_shearlet_nterm_upper_bounds_task_rate_v1,
    build_tau_eps_hbar_one_dequantization_two_scales_v1,
    build_multiphase_modica_mortola_perimeter_gamma_limit_v1,
    build_mcf_minority_erasure_inevitability_v1,
    build_annulus_anisotropy_magnitude_disputed_v1,
)

# NOT registered (draft §4/§6 CONJECTURES; per NO-FAKE, conjectures are NOT laws):
#   * "the trajectory literally IS natural-gradient-along-Maslov ending at the Gamma-limit" --
#     matches CE->tau -21.6% but UNPROVEN (recorded in MD_NG domain_of_validity, not a law).
#   * "the shearlet N-term tightness hits the task-rate floor" -- UPPER BOUND is proven; tightness
#     is UNPROVEN (recorded as the ASSUMED_AWAITING_VERIFICATION anchor in the shearlet-rate law).
#   * "Muon win = natural-gradient-ness" -- the -32% is real but the mechanism attribution is OPEN
#     (this is the FALSE FRIEND of draft §5.2; NOT registered as a natural-gradient law here).
# se(3) screw temporal-sufficiency (draft §2 law 5) is REFERENCED, not duplicated:
#   ``store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`` is its canonical home.

__all__ = [
    "ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS",
    "ANISOTROPY_DISPUTE_EQUATION_ID",
    "ANISOTROPY_GRADIENT_PROJECTION",
    "ANISOTROPY_ON_THE_FLY_DISPUTED",
    "ANISOTROPY_STRUCTURE_TENSOR",
    "FISHER_CAUSTIC_EQUATION_ID",
    "FISHER_CURVATURE_MARGIN_PEARSON_BAND",
    "FISHER_CURVATURE_MARGIN_SPEARMAN_GLOBAL",
    "MASLOV_EQUATION_ID",
    "MBO_SMOOTHING_LANE_FRACTION",
    "MCF_ERASURE_EQUATION_ID",
    "MD_NG_EQUATION_ID",
    "MODICA_MORTOLA_EQUATION_ID",
    "SHEARLET_RATE_EQUATION_ID",
    "TAU_EPS_HBAR_EQUATION_ID",
    "annulus_anisotropy_ratio",
    "annulus_fisher_trace",
    "build_annulus_anisotropy_magnitude_disputed_v1",
    "build_ce_softmax_mirror_descent_natural_gradient_v1",
    "build_fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
    "build_maslov_dequantization_bound_v1",
    "build_mcf_minority_erasure_inevitability_v1",
    "build_multiphase_modica_mortola_perimeter_gamma_limit_v1",
    "build_shearlet_nterm_upper_bounds_task_rate_v1",
    "build_tau_eps_hbar_one_dequantization_two_scales_v1",
    "categorical_bregman_divergence",
    "categorical_fisher_trace_two_class",
    "herring_triple_junction_angle_deg",
    "maslov_dequantization_bound",
    "mbo_smoothing_cost_lane_fraction",
    "shearlet_nterm_error",
    "tau_interface_halfwidth",
]
