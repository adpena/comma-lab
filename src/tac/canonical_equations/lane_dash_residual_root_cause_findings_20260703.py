# SPDX-License-Identifier: MIT
"""Canonical equations: the lane-dash d_seg residual ROOT-CAUSE findings (triality-sync).

The 4-lens deep-math pass on #205 ep200 (2026-07-03, task #273) + the two closed refinement
probes CONVERGE on ONE residual (the LANE DASHES) seen five ways, and produce FIVE measured laws.
This module formalizes them so the equations leg AGREES with DAG FEED-03t and the DSL gauge charts
(AlongTangentFrequencyGauge / VectorFieldMarginSaliencyGauge / ChromaBoundaryGauge /
FlickerTreatmentGauge / TripleJunctionMarginGauge):

  1. ``contest_R_operator_mtf_allpass_to_2px_v1`` (memory
     [[lane-dash-residual-root-is-along-tangent-freq-deficit-R-allpass]]) -- the contest R operator
     (bicubic up 384->874 -> uint8 -> bilinear down 512x384) is ALL-PASS to ~2px: MTF amplitude 1.00
     to 16 cyc/unit, 0.997 at the 10px dash scale (24 cyc/unit), 0.955 at stem-Nyquist (64), 0.00
     only at 2px (128). R adds ~0 d_seg (noR 0.00424 -> R 0.00423). OVERTURNS the R-washout prior:
     R is NOT the wall; AA-SDF / matched-filter-to-beat-R measured non-binding at ep200.

  2. ``anisotropic_basis_along_tangent_frequency_deficit_v1`` (Lens 2, decision-changing) -- the
     witness's self-orient/curvelet basis is SHARP ACROSS edges (freq_across -> 32,64 = Nyquist) but
     SMOOTH ALONG them (freq_along <= 8 cyc/unit); the lane dashes are a ~25 cyc/unit on/off
     modulation ALONG the tangent (10px period @scorer-512) -> a 3.2x frequency DEFICIT -> the basis
     CANNOT represent the dashes -> they erase finest-first (error ~ 1/persistence). The fix is
     along-tangent FREQUENCY (--n-dir-freqs 2->4 / --bank-n-scales 4->5), NOT a curvelet<->step swap.

  3. ``separatrix_asymmetry_t_subpixel_boundary_localizer_v1`` (probe a8afad40, GREEN) -- the
     cross-boundary margin ratio ``t = M_p/(M_p+M_q)`` is a FREE sub-pixel boundary localizer +
     flip-direction: self-consistent +0.560 disjoint-pixel (all) / +0.448 (lane) / +0.848 (hood),
     along-boundary autocorr +0.696; t ~ Uniform(0,1) => informative. Upgrades margin-saliency #141
     from a SCALAR magnitude to a VECTOR (magnitude + boundary-normal + sub-pixel skew).

  4. ``independent_flicker_jitter_dseg_floor_smooth_optimal_v1`` (probe a949ff63) -- for a GT
     minority-flicker probability q<0.5, an INDEPENDENT witness jitter rate r gives
     ``d_seg = q(1-r) + r(1-q)``, monotone increasing in r (slope 1-2q>0) => minimized at r=0
     (SMOOTH is provably optimal). Replicating flicker to lower d_seg therefore REQUIRES temporal
     CORRELATION (predictability); an independent replicate only raises d_seg. Splits the ~43% spike
     floor into PREDICTABLE (ego-xi / sub-pixel / quant-phase -> REPLICATE+reward) vs IRREDUCIBLE
     (sensor-noise -> DOWN-WEIGHT, smooth-is-optimal).

  5. ``scalar_top1_top2_margin_is_exact_distance_to_flip_v1`` (probe a4c66f2f, CLOSED WEAK -- the
     clean negative that SHARPENS the crux) -- MEASURED at ALL 118M pixels (n600 exact CPU-torch
     SegNet argmax, bit-exact to the cached margins): ``gap13 = top1-top3 >= gap12 = top1-top2``
     everywhere (min(gap13-gap12) = 0.0). => the SCALAR margin top1-top2 IS the exact logit-space
     distance-to-flip; only the 2nd-place class can flip the argmax FIRST; the 3rd+ logit can NEVER
     surface a flip-ONSET fragility the scalar margin missed. Corollary: triple junctions are a
     flip-STRUCTURE DOF, never a flip-ONSET one (0.027% of pixels, ~1-2% of flip mass, dominated by
     Road|Undrivable|Movable car-corners = NOT the lane tail). The lane residual is a codim-1
     Road|Lane FACET (41.4% of all facets) the scalar margin already describes EXACTLY -> aim
     capacity at the FACET (along-tangent-freq + vector-t), not the junction.

means != ends: findings 1-3 are MEASURED (advisory n6 / n600-diagnostic); finding 4 is an analytic
identity (the empirical PREDICTABLE-vs-IRREDUCIBLE split is probe-pending); finding 5 is a MEASURED
n600 law. NONE moves the pointer (0.19110 UNMOVED) -- only a byte-closed exact row does. The per-lever
net d_seg (the along-tangent-freq / vector-t levers) is a #205-class A/B arm (operator GO).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

R_MTF_EQUATION_ID = "contest_r_operator_mtf_allpass_to_2px_v1"
FREQ_DEFICIT_EQUATION_ID = "anisotropic_basis_along_tangent_frequency_deficit_v1"
ASYMMETRY_T_EQUATION_ID = "separatrix_asymmetry_t_subpixel_boundary_localizer_v1"
JITTER_FLOOR_EQUATION_ID = "independent_flicker_jitter_dseg_floor_smooth_optimal_v1"
SCALAR_MARGIN_EQUATION_ID = "scalar_top1_top2_margin_is_exact_distance_to_flip_v1"

_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"

# The durable DAG record (FEED-03t) + the source memory anchor.
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_MEM = "lane_dash_residual_root_is_along_tangent_freq_deficit_R_allpass_20260703.md"

# --- MEASURED grounding constants (advisory; the per-lever net d_seg is #205-gated) ----------------
# R MTF amplitude at the anchor spatial frequencies (cyc/unit through the contest R operator).
R_MTF_ANCHOR_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, 1.00),    # DC
    (16.0, 1.00),   # witness render is band-limited: 98.6% power <= 16 cyc/unit
    (24.0, 0.997),  # the 10px dash scale @scorer-512 -> R barely touches it
    (64.0, 0.955),  # stride-2 stem Nyquist
    (128.0, 0.00),  # 2px -> the only place R fully washes out (5x finer than the dashes)
)
R_NO_R_DSEG = 0.00424   # d_seg WITHOUT the R roundtrip (witness render already band-limited)
R_WITH_R_DSEG = 0.00423  # d_seg WITH the R roundtrip -> R adds ~0 d_seg
# Lens-2 along-tangent frequency deficit.
BASIS_FREQ_ALONG_MAX = 8.0   # the anisotropic basis' along-tangent bandwidth (cyc/unit)
DASH_FREQ_ALONG = 25.0       # the lane-dash on/off modulation along the tangent (~10px period @512)
FREQ_DEFICIT_RATIO = DASH_FREQ_ALONG / BASIS_FREQ_ALONG_MAX  # ~3.125 (~3.2x)
# Asymmetry-t self-consistency (disjoint-pixel), measured n6 advisory.
ASYM_T_SELFCONSISTENCY_ALL = 0.560
ASYM_T_SELFCONSISTENCY_LANE = 0.448
ASYM_T_SELFCONSISTENCY_HOOD = 0.848
ASYM_T_ALONG_BOUNDARY_AUTOCORR = 0.696
# Chroma boundary DOF (a3e9f0bd GREEN, n96 advisory, 100% L*-match to the frozen SegNet).
CHROMA_REMOVAL_LANE_TO_ROAD_FLIP = 0.0754      # constant-luma flips 7.54% of Lane -> Road
CHROMA_REMOVAL_MOVABLE_TO_UNDRIV_FLIP = 0.0438  # constant-luma flips 4.38% of Movable -> Undrivable
CHROMA_FLIPS_IN_MARGIN_LT1_ANNULUS = 0.934      # 93.4% of chroma-flips are in the margin<1 annulus
CHROMA_FLIPS_IN_MARGIN_LT025 = 0.337            # 33.7% at margin<0.25 (the fragile band)
CONSTANT_LUMA_DESAT_ANNULUS_FLIP = 0.031        # desat-only flips 3.1% -> chroma effect is luma-indep
MARGIN_GRAD_ENERGY_LUMA_FRACTION = 0.788        # 78.8% of margin-gradient energy is luma
MARGIN_GRAD_ENERGY_CHROMA_FRACTION = 0.212      # 21.2% is chroma (the independent DOF)
# Scalar-margin exactness (a4c66f2f, n600 exact): gap13 >= gap12 EVERYWHERE.
GAP13_MINUS_GAP12_MIN = 0.0            # measured min over ALL 118M pixels
TRIPLE_JUNCTION_PIXEL_FRACTION = 0.00027   # 0.027% of pixels are near-triple-junctions
TRIPLE_JUNCTION_CAR_CORNER_FRACTION = 0.539  # 53.9% of the tiny triple set = Road|Undriv|Movable
LANE_FACET_FRACTION_OF_FACETS = 0.414        # Road|Lane = 41.4% of all codim-1 facets


def r_mtf_amplitude_at(cyc_per_unit: float) -> float:
    """Piecewise-linear interpolation of the MEASURED contest-R MTF amplitude at a spatial frequency
    ``cyc_per_unit`` (through bicubic-up -> uint8 -> bilinear-down). Anchor points: R_MTF_ANCHOR_POINTS.
    R is ALL-PASS (amplitude ~1) to ~24 cyc/unit (the 10px dash scale) and only nulls at 2px (128)."""
    f = float(cyc_per_unit)
    pts = R_MTF_ANCHOR_POINTS
    if f <= pts[0][0]:
        return pts[0][1]
    if f >= pts[-1][0]:
        return pts[-1][1]
    for (f0, a0), (f1, a1) in zip(pts, pts[1:]):
        if f0 <= f <= f1:
            if f1 == f0:
                return a0
            return a0 + (a1 - a0) * (f - f0) / (f1 - f0)
    return pts[-1][1]


def along_tangent_frequency_deficit_ratio() -> float:
    """The MEASURED along-tangent frequency deficit ``dash_freq / basis_freq_along`` (~3.2x): the lane
    dashes modulate at ~25 cyc/unit ALONG the tangent, the anisotropic basis carries only <=8."""
    return FREQ_DEFICIT_RATIO


def boundary_asymmetry_t(m_p: float, m_q: float) -> float:
    """The cross-boundary margin-ratio sub-pixel localizer ``t = M_p / (M_p + M_q)`` (probe a8afad40).

    ``m_p`` / ``m_q`` are the (non-negative) SegNet argmax margins on the two sides of a codim-1
    boundary. t in (0,1) encodes the sub-pixel boundary POSITION (t=0.5 -> boundary bisects the pair)
    AND, by which side is smaller, the flip DIRECTION. Returns 0.5 for a degenerate zero-margin pair."""
    denom = float(m_p) + float(m_q)
    if denom <= 0.0:
        return 0.5
    return float(m_p) / denom


def independent_jitter_dseg(q: float, r: float) -> float:
    """The independent-jitter d_seg floor ``d_seg = q(1-r) + r(1-q)`` (analytic identity).

    ``q`` = the GT minority-flicker probability (a pixel's class flips in GT with prob q), ``r`` = the
    witness's INDEPENDENT jitter rate at that pixel. For q<0.5 the slope d(d_seg)/dr = 1-2q > 0, so
    d_seg is minimized at r=0 (SMOOTH is provably optimal) -- an independent replicate can only RAISE
    d_seg. Lowering d_seg by replicating flicker therefore REQUIRES temporal correlation (r must track
    the GT flip, not fire independently)."""
    q = float(q)
    r = float(r)
    return q * (1.0 - r) + r * (1.0 - q)


def chroma_margin_gradient_energy_fraction() -> float:
    """The MEASURED chroma share of the SegNet margin-gradient energy (0.212; probe a3e9f0bd) =>
    chroma is an INDEPENDENT boundary d_seg DOF (78.8% luma / 21.2% chroma), not a luma proxy."""
    return MARGIN_GRAD_ENERGY_CHROMA_FRACTION


def gap13_minus_gap12_min() -> float:
    """The MEASURED minimum of ``(top1-top3) - (top1-top2)`` over ALL 118M pixels (n600 exact
    CPU-torch SegNet, a4c66f2f) = 0.0. i.e. gap13 >= gap12 EVERYWHERE => the scalar top1-top2 margin
    IS the exact logit-space distance-to-flip (only the 2nd class can flip the argmax first)."""
    return GAP13_MINUS_GAP12_MIN


def build_contest_R_operator_mtf_allpass_to_2px_v1() -> CanonicalEquation:
    """Build the contest-R MTF all-pass-to-2px canonical equation (FEED-03t; overturns the R-wall)."""

    anchor_mtf = EmpiricalAnchor(
        anchor_id="contest_R_mtf_grating_sweep_allpass_to_2px_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"operator": "bicubic_up_384_to_874 -> uint8 -> bilinear_down_512x384",
                "method": "sinusoidal grating amplitude sweep through the REAL R operator",
                "n_pairs": "n6 advisory"},
        predicted_output={"r_adds_zero_dseg": True},
        empirical_output={
            "mtf_at_0_to_16_cyc": 1.00,
            "mtf_at_24_cyc_dash_scale": 0.997,
            "mtf_at_64_cyc_stem_nyquist": 0.955,
            "mtf_at_128_cyc_2px": 0.00,
            "noR_dseg": R_NO_R_DSEG,
            "withR_dseg": R_WITH_R_DSEG,
            "render_power_below_16_cyc": 0.986,
            "note": "R is all-pass to ~2px (5x finer than the 10px dashes); AA-SDF / matched-filter "
                    "to 'beat R' MEASURED non-binding at ep200; Gibbs ABSENT (hosc working)",
        },
        residual=0.0,
        source_artifact=_MEM,
        measurement_method="grating_amplitude_sweep_through_real_R_n6_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEM,
            reactivation_criteria="re-measure R MTF at n600 + across the seg-frame resolution ladder",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=R_MTF_EQUATION_ID,
        name="Contest R operator is MTF all-pass to ~2px -> R adds ~0 d_seg",
        one_line_summary=(
            "R MTF amplitude 1.00 to 16 cyc/unit, 0.997 at the 10px dash scale (24), 0.00 only at 2px "
            "(128) -> R adds ~0 d_seg (noR 0.00424 -> R 0.00423); the R-washout prior is OVERTURNED"
        ),
        latex_form=(
            r"\mathrm{MTF}_R(f)\approx 1\ (f\le 24\,\mathrm{cyc/unit}),\ \to 0\ \text{only at }f=128\ "
            r"(2\,\mathrm{px})\ \Rightarrow\ \Delta d_{seg}^{R}\approx 0\ (0.00424\to 0.00423)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703:r_mtf_amplitude_at"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "operator": "contest R (bicubic-up -> uint8 -> bilinear-down)",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "measured at ep200 on a band-limited witness render; the sub-2px R boundary "
                    "effect is real but tiny (0.65% within the margin band)",
        },
        units_in={"cyc_per_unit": "spatial_frequency_cycles_per_render_unit"},
        units_out={"mtf_amplitude": "dimensionless_0_1"},
        empirical_anchors=(anchor_mtf,),
        predicted_vs_empirical_residual={"r_mtf_allpass_to_2px": 0.0},
        last_calibration_utc="2026-07-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("experiments/train_levelset_witness_realized_through_R_mlx.py",),
        provenance=build_provenance_for_predicted(
            model_id="contest_R_operator_mtf_allpass_to_2px.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


def build_anisotropic_basis_along_tangent_frequency_deficit_v1() -> CanonicalEquation:
    """Build the along-tangent frequency-deficit canonical equation (FEED-03t Lens 2; the ROOT lever)."""

    anchor_deficit = EmpiricalAnchor(
        anchor_id="anisotropic_basis_along_tangent_freq_deficit_3p2x_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"basis": "self-orient / anisotropic curvelet (freq_across -> 32,64 Nyquist)",
                "dash_scale": "~10px period @scorer-512 = ~25 cyc/unit along the tangent",
                "n205_config": "--n-dir-freqs 2 (the deficit); default 6, #205 runs 2"},
        predicted_output={"basis_blind_to_dash_along_tangent": True},
        empirical_output={
            "basis_freq_along_max": BASIS_FREQ_ALONG_MAX,
            "dash_freq_along": DASH_FREQ_ALONG,
            "deficit_ratio": along_tangent_frequency_deficit_ratio(),
            "lane_flip_fraction": 0.352,
            "lane_over_representation": "55x (0.59% area)",
            "flips_in_margin_lt1_annulus": 0.93,
            "median_boundary_margin": 0.22,
            "erasure_law": "error ~ 1/persistence (1-6px dashes erased 97%, 100+px runs 25%)",
            "note": "dashed edge needs along-tangent bandwidth (wave-atom) the anisotropic curvelet "
                    "lacks by construction; the fix is FREQUENCY, not a curvelet<->step swap",
        },
        residual=0.0,
        source_artifact=_MEM,
        measurement_method="curvelet_lens_flip_attribution_plus_basis_freq_analysis_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEM,
            reactivation_criteria="re-measure the flip-vs-along-tangent-freq attribution at n600",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_lever = EmpiricalAnchor(
        anchor_id="along_tangent_freq_lever_net_dseg_205_gated_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"lever": "--n-dir-freqs 2->4 (and/or --bank-n-scales 4->5)",
                "cost": "~0 archive bytes (deterministic basis); R-safe (<= Nyquist)",
                "stage": "next-run / improved-Muon-restart (NOT #205 mid-run)"},
        predicted_output={"attacks_dash_residual_at_representational_root": True},
        empirical_output={"net_n600_dseg": "205-class A/B arm (operator GO); the #1 ranked ep300+ "
                          "lever; expect it to lower the lane-dash residual at its root"},
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="205_class_byte_closed_ab_arm",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="#205-class arm: --n-dir-freqs 4 vs 2 byte-closed n600 d_seg",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=FREQ_DEFICIT_EQUATION_ID,
        name="Anisotropic basis along-tangent frequency deficit -> lane-dash erasure",
        one_line_summary=(
            "basis freq_along <= 8 cyc/unit vs dash ~25 -> 3.2x deficit -> the basis cannot represent "
            "the dashes -> they erase finest-first (error ~ 1/persistence); the fix is along-tangent freq"
        ),
        latex_form=(
            r"\frac{f_{\text{dash}}^{\parallel}}{f_{\text{basis}}^{\parallel}}=\frac{25}{8}\approx 3.2"
            r"\ \Rightarrow\ \text{dash erasure} \propto 1/\text{persistence}\quad"
            r"(f^{\perp}\to 64\ \text{sharp},\ f^{\parallel}\le 8\ \text{blind})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703:"
            "along_tangent_frequency_deficit_ratio"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "basis": "anisotropic self-orient / curvelet directional Fourier features",
            "residual": "class-1 lane dashes (the finest-scale erasure tail)",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": "the deficit is measured; the net d_seg of raising the frequency is #205-gated",
        },
        units_in={},
        units_out={"deficit_ratio": "dimensionless_frequency_ratio"},
        empirical_anchors=(anchor_deficit, anchor_lever),
        predicted_vs_empirical_residual={"along_tangent_freq_deficit_3p2x": 0.0},
        last_calibration_utc="2026-07-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("experiments/train_levelset_witness_realized_through_R_mlx.py",),
        provenance=build_provenance_for_predicted(
            model_id="anisotropic_basis_along_tangent_frequency_deficit.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


def build_separatrix_asymmetry_t_subpixel_boundary_localizer_v1() -> CanonicalEquation:
    """Build the cross-boundary asymmetry-t sub-pixel localizer canonical equation (FEED-03t; probe
    a8afad40, GREEN -- margin-saliency scalar->vector)."""

    anchor_t = EmpiricalAnchor(
        anchor_id="separatrix_asymmetry_t_selfconsistent_subpixel_localizer_green_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"statistic": "t = M_p/(M_p+M_q) cross-boundary argmax-margin ratio",
                "probe": "a8afad40 (vector-field / separatrix-asymmetry)", "n_pairs": "n6 advisory"},
        predicted_output={"t_is_a_free_subpixel_boundary_localizer": True},
        empirical_output={
            "selfconsistency_disjoint_all": ASYM_T_SELFCONSISTENCY_ALL,
            "selfconsistency_disjoint_lane": ASYM_T_SELFCONSISTENCY_LANE,
            "selfconsistency_disjoint_hood": ASYM_T_SELFCONSISTENCY_HOOD,
            "along_boundary_autocorr_horizontal": ASYM_T_ALONG_BOUNDARY_AUTOCORR,
            "t_distribution": "~Uniform(0,1) => informative",
            "note": "unifies sub-pixel POSITION + flip-DIRECTION; upgrades margin-saliency #141 "
                    "scalar -> vector (magnitude + boundary-normal + skew); QUALIFIED: weakest on "
                    "thin lanes, effect concentrated in the 1-2px flip band",
        },
        residual=0.0,
        source_artifact=_MEM,
        measurement_method="cross_boundary_margin_ratio_selfconsistency_n6_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEM,
            reactivation_criteria="re-measure t self-consistency + the vector-lever net d_seg at n600",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_lever = EmpiricalAnchor(
        anchor_id="asymmetry_t_vector_saliency_lever_net_dseg_205_gated_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"lever": "margin-saliency SCALAR magnitude -> VECTOR (t sub-pixel skew)",
                "status": "DESIGN-STAGE (the training lever is unbuilt; the t diagnostic is measured)"},
        predicted_output={"vector_t_saliency_sharpens_lane_facet": True},
        empirical_output={"net_n600_dseg": "205-class A/B arm (operator GO); composes with the "
                          "along-tangent-freq lever on the SAME lane facet"},
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="205_class_byte_closed_ab_arm",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="build the vector-t saliency lever + #205-class byte-closed A/B",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=ASYMMETRY_T_EQUATION_ID,
        name="Cross-boundary asymmetry t = M_p/(M_p+M_q): free sub-pixel boundary localizer",
        one_line_summary=(
            "t = M_p/(M_p+M_q) is a self-consistent (+0.560 disjoint) sub-pixel boundary position + "
            "flip-direction localizer -> upgrades margin-saliency #141 from scalar to vector"
        ),
        latex_form=(
            r"t=\frac{M_p}{M_p+M_q}\in(0,1)\ :\ \text{sub-pixel boundary position + flip side}"
            r"\quad(\text{self-consistency}\ +0.560\ \text{disjoint},\ t\sim U(0,1))"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703:boundary_asymmetry_t"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "field": "SegNet argmax margin on the codim-1 separatrix",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": "the t self-consistency is measured (n6 advisory); the vector-lever net d_seg is "
                    "#205-gated; weakest on thin lanes",
        },
        units_in={"m_p": "segnet_argmax_margin_side_p", "m_q": "segnet_argmax_margin_side_q"},
        units_out={"t": "dimensionless_subpixel_boundary_position_0_1"},
        empirical_anchors=(anchor_t, anchor_lever),
        predicted_vs_empirical_residual={"asymmetry_t_selfconsistency_positive": 0.0},
        last_calibration_utc="2026-07-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("experiments/train_levelset_witness_realized_through_R_mlx.py",),
        provenance=build_provenance_for_predicted(
            model_id="separatrix_asymmetry_t_subpixel_boundary_localizer.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


def build_independent_flicker_jitter_dseg_floor_smooth_optimal_v1() -> CanonicalEquation:
    """Build the independent-jitter d_seg floor canonical equation (FEED-03t; probe a949ff63 -- the
    replicate-vs-downweight flicker decomposition)."""

    anchor_identity = EmpiricalAnchor(
        anchor_id="independent_jitter_dseg_floor_smooth_optimal_analytic_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"identity": "d_seg = q(1-r) + r(1-q) (argmax-disagreement rate, independent Bernoulli)",
                "q": "GT minority-flicker probability (<0.5)", "r": "witness independent jitter rate"},
        predicted_output={"argmin_r_dseg": 0.0, "slope_d_dseg_dr": "1-2q > 0 for q<0.5"},
        empirical_output={
            "dseg_at_r0": "q (smooth = the minority baseline)",
            "dseg_at_r_eq_q": "2q(1-q) >= q (independent replicate RAISES d_seg)",
            "note": "SMOOTH (r=0) is provably optimal for an INDEPENDENT jitter; lowering d_seg by "
                    "replicating flicker REQUIRES temporal correlation (predictability)",
        },
        residual=0.0,
        source_artifact=_MEM,
        measurement_method="closed_form_argmax_disagreement_rate_identity",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEM,
            reactivation_criteria="none (analytic identity); verify the numeric floor on a flicker slice",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_split = EmpiricalAnchor(
        anchor_id="flicker_predictable_vs_irreducible_split_probe_pending_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"decomposition": "~43% spike floor -> PREDICTABLE (ego-xi/sub-pixel/quant-phase) vs "
                "IRREDUCIBLE (sensor-noise)", "probe": "a949ff63 (flicker decomposition, running)"},
        predicted_output={"predictable_replicate_irreducible_downweight": True},
        empirical_output={"split_fraction": "probe-pending; PREDICTABLE -> REPLICATE+reward+preserve "
                          "d_seg lever (correlated), IRREDUCIBLE -> DOWN-WEIGHT (smooth-is-optimal)"},
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="flicker_autocorrelation_decomposition_probe",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="a949ff63 measures the predictable/irreducible split fraction at n600",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=JITTER_FLOOR_EQUATION_ID,
        name="Independent-jitter d_seg floor: smooth (r=0) provably optimal for q<0.5",
        one_line_summary=(
            "d_seg = q(1-r)+r(1-q) is monotone increasing in r for q<0.5 -> smooth (r=0) is optimal; "
            "replicating flicker to lower d_seg REQUIRES temporal correlation (predictability)"
        ),
        latex_form=(
            r"d_{seg}(r)=q(1-r)+r(1-q),\ \frac{\partial d_{seg}}{\partial r}=1-2q>0\ (q<\tfrac12)"
            r"\ \Rightarrow\ \arg\min_r=0\ (\text{smooth optimal})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703:independent_jitter_dseg"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "regime": "per-pixel minority-flicker q<0.5, independent (uncorrelated) witness jitter",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": "the floor is an analytic identity; the predictable/irreducible split is probe-pending",
        },
        units_in={"q": "gt_minority_flicker_probability", "r": "witness_independent_jitter_rate"},
        units_out={"d_seg": "argmax_disagreement_rate"},
        empirical_anchors=(anchor_identity, anchor_split),
        predicted_vs_empirical_residual={"smooth_optimal_for_independent_jitter": 0.0},
        last_calibration_utc="2026-07-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("experiments/train_levelset_witness_realized_through_R_mlx.py",),
        provenance=build_provenance_for_predicted(
            model_id="independent_flicker_jitter_dseg_floor_smooth_optimal.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


def build_chroma_decides_lane_and_movable_at_annulus_v1() -> CanonicalEquation:
    """Build the chroma-boundary-DOF canonical equation (FEED-03t; probe a3e9f0bd GREEN -- chroma is
    an independent d_seg boundary actuator concentrated on the lane crux + Movable over-dilation)."""

    anchor_chroma = EmpiricalAnchor(
        anchor_id="chroma_removal_flips_lane_and_movable_at_annulus_green_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"ablation": "constant-luma (remove chroma) through the frozen contest SegNet argmax",
                "probe": "a3e9f0bd", "n_pairs": "n96 advisory", "fidelity": "100% L*-match to SegNet"},
        predicted_output={"chroma_is_independent_dseg_boundary_dof": True},
        empirical_output={
            "lane_to_road_flip_fraction": CHROMA_REMOVAL_LANE_TO_ROAD_FLIP,
            "movable_to_undrivable_flip_fraction": CHROMA_REMOVAL_MOVABLE_TO_UNDRIV_FLIP,
            "chroma_flips_in_margin_lt1_annulus": CHROMA_FLIPS_IN_MARGIN_LT1_ANNULUS,
            "chroma_flips_in_margin_lt025": CHROMA_FLIPS_IN_MARGIN_LT025,
            "constant_luma_desat_annulus_flip": CONSTANT_LUMA_DESAT_ANNULUS_FLIP,
            "margin_grad_energy_luma_fraction": MARGIN_GRAD_ENERGY_LUMA_FRACTION,
            "margin_grad_energy_chroma_fraction": MARGIN_GRAD_ENERGY_CHROMA_FRACTION,
            "note": "chroma is an INDEPENDENT d_seg boundary DOF (luma-independent: desat-only flips "
                    "3.1%), concentrated on the LANE crux + the Movable over-dilation, ORTHOGONAL to "
                    "the geometry levers (along-tangent-freq / lane-render-band / sub-pixel-t) -> a "
                    "stacking APPEARANCE attack on the SAME lane facet",
        },
        residual=0.0,
        source_artifact=_MEM,
        measurement_method="constant_luma_ablation_through_frozen_segnet_n96_advisory",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEM,
            reactivation_criteria="re-measure the chroma ablation flip fractions at n600 through R",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_lever = EmpiricalAnchor(
        anchor_id="route_chroma_into_annulus_lever_net_dseg_205_gated_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"lever": "route chroma CAPACITY into the boundary annulus (chroma already ON in #205)",
                "pose_synergy": "pose on the stored sidecar -> the seg-frame chroma is FREED for d_seg "
                "(seg-perp-pose; orphan #227)", "operator": "'Chroma too' 2026-06-25"},
        predicted_output={"chroma_stacks_orthogonally_on_the_lane_facet": True},
        empirical_output={"net_n600_dseg": "205-class A/B arm (operator GO); GREEN candidate; stacks "
                          "with the geometry levers on the same lane facet (geometry + appearance)"},
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="205_class_byte_closed_ab_arm",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="#205-class arm: chroma-into-annulus routing byte-closed n600 d_seg",
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id="chroma_decides_lane_and_movable_at_annulus_v1",
        name="Chroma is an independent d_seg boundary DOF (decides Lane + Movable at the annulus)",
        one_line_summary=(
            "removing chroma flips 7.54% Lane->Road + 4.38% Movable->Undriv, 93.4% in the margin<1 "
            "annulus -> chroma is an independent d_seg DOF (21.2% of margin-gradient energy)"
        ),
        latex_form=(
            r"\Pr[\text{flip}\mid \text{no chroma}]:\ \text{Lane}\to\text{Road}\ 7.54\%,\ "
            r"\text{Movable}\to\text{Undriv}\ 4.38\%;\ E^{\nabla m}_{\text{chroma}}=21.2\%"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703:"
            "chroma_margin_gradient_energy_fraction"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "actuator": "RGB chroma channels (SegNet reads RGB; --chroma default ON in #205)",
            "target": "the Lane crux + the Movable over-dilation, in the margin<1 annulus",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": "the chroma DOF is measured GREEN (n96 advisory); the routing-into-annulus net "
                    "d_seg is #205-gated; orthogonal to the geometry levers (a stacking appearance attack)",
        },
        units_in={},
        units_out={"chroma_margin_gradient_energy_fraction": "dimensionless_0_1"},
        empirical_anchors=(anchor_chroma, anchor_lever),
        predicted_vs_empirical_residual={"chroma_independent_dseg_dof": 0.0},
        last_calibration_utc="2026-07-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("experiments/train_levelset_witness_realized_through_R_mlx.py",),
        provenance=build_provenance_for_predicted(
            model_id="chroma_decides_lane_and_movable_at_annulus.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


def build_scalar_top1_top2_margin_is_exact_distance_to_flip_v1() -> CanonicalEquation:
    """Build the scalar-margin-is-exact-distance-to-flip canonical equation (FEED-03t; probe a4c66f2f
    CLOSED WEAK -- the simplex/triple-junction clean negative that SHARPENS the crux)."""

    anchor_gap = EmpiricalAnchor(
        anchor_id="gap13_ge_gap12_at_all_118M_pixels_n600_exact_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={"claim": "gap13 = top1-top3 >= gap12 = top1-top2 at every pixel",
                "scorer": "frozen contest CPU-torch SegNet argmax, bit-exact to the cached margins",
                "scale": "n600 EXACT (all 600 pairs, 118M pixels)", "probe": "a4c66f2f"},
        predicted_output={"gap13_minus_gap12_min": gap13_minus_gap12_min()},
        empirical_output={
            "gap13_minus_gap12_min": GAP13_MINUS_GAP12_MIN,
            "scalar_margin_is_exact_distance_to_flip": True,
            "triple_junction_pixel_fraction": TRIPLE_JUNCTION_PIXEL_FRACTION,
            "triple_junction_flip_mass": "~1-2%",
            "triple_junction_car_corner_fraction": TRIPLE_JUNCTION_CAR_CORNER_FRACTION,
            "lane_facet_fraction_of_facets": LANE_FACET_FRACTION_OF_FACETS,
            "note": "only the 2nd-place class can flip the argmax FIRST -> the 3rd+ logit can NEVER "
                    "surface a flip-ONSET fragility the scalar margin missed; triple junctions are a "
                    "flip-STRUCTURE DOF (car-corners), NOT the lane tail; the lane residual is a "
                    "codim-1 Road|Lane FACET the scalar margin already describes EXACTLY",
        },
        residual=0.0,
        source_artifact=_MEM,
        measurement_method="n600_exact_cpu_torch_segnet_gap13_vs_gap12_over_all_pixels",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEM,
            reactivation_criteria="none (n600 exact law); re-verify if the SegNet checkpoint changes",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=SCALAR_MARGIN_EQUATION_ID,
        name="Scalar top1-top2 margin IS the exact distance-to-flip (simplex adds no flip-onset DOF)",
        one_line_summary=(
            "gap13>=gap12 at all 118M pixels (min 0.0) => scalar top1-top2 margin IS the exact "
            "distance-to-flip; triple junctions = flip-STRUCTURE (car-corners), not the lane tail"
        ),
        latex_form=(
            r"\min_{\text{pixels}}\big[(z_{(1)}-z_{(3)})-(z_{(1)}-z_{(2)})\big]=0"
            r"\ \Rightarrow\ \text{margin}=z_{(1)}-z_{(2)}\ \text{is exact distance-to-flip}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703:gap13_minus_gap12_min"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "frozen_contest_segnet"],
            "scope": "flip-ONSET (which class flips first); NOT flip-STRUCTURE (triple-junction geometry)",
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "n600 EXACT diagnostic on the frozen scorer (bit-exact to cached margins); a "
                    "clean NEGATIVE for the multi-class-simplex lever -> banks TripleJunctionMarginGauge",
        },
        units_in={},
        units_out={"gap13_minus_gap12_min": "segnet_argmax_logit_units"},
        empirical_anchors=(anchor_gap,),
        predicted_vs_empirical_residual={"gap13_ge_gap12_everywhere": 0.0},
        last_calibration_utc="2026-07-03T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=("experiments/train_levelset_witness_realized_through_R_mlx.py",),
        provenance=build_provenance_for_predicted(
            model_id="scalar_top1_top2_margin_is_exact_distance_to_flip.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "ASYMMETRY_T_EQUATION_ID",
    "ASYM_T_SELFCONSISTENCY_ALL",
    "BASIS_FREQ_ALONG_MAX",
    "DASH_FREQ_ALONG",
    "FREQ_DEFICIT_EQUATION_ID",
    "FREQ_DEFICIT_RATIO",
    "GAP13_MINUS_GAP12_MIN",
    "JITTER_FLOOR_EQUATION_ID",
    "R_MTF_ANCHOR_POINTS",
    "R_MTF_EQUATION_ID",
    "SCALAR_MARGIN_EQUATION_ID",
    "along_tangent_frequency_deficit_ratio",
    "boundary_asymmetry_t",
    "build_anisotropic_basis_along_tangent_frequency_deficit_v1",
    "build_chroma_decides_lane_and_movable_at_annulus_v1",
    "build_contest_R_operator_mtf_allpass_to_2px_v1",
    "build_independent_flicker_jitter_dseg_floor_smooth_optimal_v1",
    "build_scalar_top1_top2_margin_is_exact_distance_to_flip_v1",
    "build_separatrix_asymmetry_t_subpixel_boundary_localizer_v1",
    "chroma_margin_gradient_energy_fraction",
    "gap13_minus_gap12_min",
    "independent_jitter_dseg",
    "r_mtf_amplitude_at",
]
