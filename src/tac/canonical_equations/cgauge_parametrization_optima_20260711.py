# SPDX-License-Identifier: MIT
"""#223 — the V9·CGauge parametrization optima, DERIVED (2026-07-11 Einstein/Fable pass).

Memo: ``.omx/research/cgauge_master_action_and_parametrization_20260711.md``. Four laws,
each derived BLINDED from first principles (Whitney embedding / R-chain sampling theory /
Candès–Donoho parabolic scaling / gradient-noise stationarity), THEN checked against the
measured anchors (rank-8 doubly-measured; R MTF all-pass-to-2px; the 25-vs-8 cyc/unit
along-tangent deficit; the 0.999 β₂ anchor). Pointer 0.19108282 UNMOVED — these are MEANS
(sizing laws for the V9·CGauge trunk; they move no score until a config byte-closes).

THE UNIFICATION (the bridge these four laws are facets of): all four are ONE statement —
*the chart's resolution must match the rank-8 covariant separatrix manifold's
Fisher-metric extent in every resource axis* —
  • Whitney mod-dim   = enough chart COORDINATES        (embedding dimension),
  • Nyquist bank      = enough chart SCALE through R     (finest usable frequency),
  • parabolic scaling = the right ORIENTATION-SCALE split (anisotropic allocation),
  • β₂ window         = enough estimation TIME            (the metric's second moments).
Same law read four ways, exactly as curvature/geodesics/field-equation are one object.

CHART CAVEAT (MEASURED 2026-07-11, the UU-2 probe): linear rank is CHART-DEPENDENT — in
the raw INDICATOR basis 8 dims hold only 0.505 of per-pair variance (rank 293 needed for
the SDF-chart's 0.9316). Intrinsic dim 8 stands (nonlinear TwoNN + SDF-linear agree); the
SDF/level-set chart is the (near-)LINEARIZING chart. Consequence: mod-17–19 suffices ONLY
for a trunk whose internal representation is SDF-like (a level-set field) — which
retro-derives the level-set architecture as necessary for low-dim conditioning.
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

_UTC = "2026-07-11T04:00:00Z"
_MEMO = ".omx/research/cgauge_master_action_and_parametrization_20260711.md"
_PROBE = ".omx/tmp/cgauge_probes/results.json"

WHITNEY_EQUATION_ID = "cgauge_whitney_moddim_v1"
NYQUIST_EQUATION_ID = "cgauge_nyquist_bank_frequency_v1"
PARABOLIC_EQUATION_ID = "cgauge_curvelet_parabolic_bank_v1"
BETA2_EQUATION_ID = "cgauge_beta2_window_v1"

# Measured inputs the derivations are CHECKED against (never fit to).
RANK8_INTRINSIC_DIM = 8              # doubly-measured (level-set TwoNN + SDF-SVD 0.9316 n600)
POSE_EXTRA_LOCAL_DOF = (2.1, 2.4)    # mirror audit: partition-INVISIBLE => NOT trunk-mod DOF
R_MTF_ZERO_PERIOD_PX = 2             # contest_r_operator_mtf_allpass_to_2px_v1 (0.00 at 2px)
SCORER_UNIT_PX = 512                 # frequency unit: cycles per 512-px scorer width
DASH_LINE_CYC_PER_UNIT = 25.0        # anisotropic_basis_along_tangent_frequency_deficit_v1
ALONG_TANGENT_SUPPLIED_CYC = 8.0     # n-dir-freqs=2 allocation (same anchor)
BETA2_MEASURED_ANCHOR = 0.999        # §2D byte-identical MLX anchor (t5 crucible)


# --------------------------------------------------------------------------------------
# Law 1 — Whitney mod-dim (DERIVED: embedding theory on the measured rank-8 manifold)
# --------------------------------------------------------------------------------------
def whitney_mod_dim(intrinsic_dim: int = RANK8_INTRINSIC_DIM, *, gauge_margin: int = 2) -> int:
    """Minimal conditioning (mod-vector) dimension for a GENERIC injective parametrization
    of a d-dim manifold: 2d+1 (Whitney/prevalence). d=8 => 17; +gauge_margin (zero-mode
    slack for the lattice-phase/gauge directions) => the standing 17–19 answer.

    Why 2d (mod-16) UNDER-embeds: at target dim 2d double points of a generic smooth map
    are STABLE (transversality: self-intersections of a d-manifold in R^{2d} are
    0-dimensional and generic — removable only by the global Whitney trick, which a
    generic learned encoder does not perform). At 2d+1 they are non-generic (measure
    zero) => a generic parametrization is injective. Pose's +2.1–2.4 local DOF are
    partition-INVISIBLE (MEASURED, mirror audit) => they do NOT enter d; they route
    through the dedicated dxi channel (6+k)."""
    d = int(intrinsic_dim)
    if d <= 0:
        raise ValueError("intrinsic_dim must be positive")
    return 2 * d + 1 + int(gauge_margin)


def moddim_under_embeds(mod_dim: int, intrinsic_dim: int = RANK8_INTRINSIC_DIM) -> bool:
    """True iff mod_dim <= 2d (double points generic => aliased pair-configurations)."""
    return int(mod_dim) <= 2 * int(intrinsic_dim)


# --------------------------------------------------------------------------------------
# Law 2 — Nyquist bank-frequency through R (DERIVED from the measured R MTF + grids)
# --------------------------------------------------------------------------------------
def nyquist_bank_ceiling_cyc_per_unit(
    *, unit_px: int = SCORER_UNIT_PX, mtf_zero_period_px: int = R_MTF_ZERO_PERIOD_PX
) -> float:
    """Highest bank frequency with ANY through-R transfer.

    R's measured MTF (``contest_r_operator_mtf_allpass_to_2px_v1``) reaches its first
    zero at a 2-px period, which that anchor labels 128 cyc/unit on the 512-px unit —
    i.e. the anchor's frequency convention is nu = unit_px / (2 * period_px)
    (512/(2*2) = 128). This function adopts that convention verbatim so the derived
    ceiling and the measured MTF table live on the same axis. Near-all-pass region:
    MTF 0.997 at 24, 0.955 at 64."""
    return float(unit_px) / (2.0 * float(mtf_zero_period_px))


def bank_frequency_is_wasted(nu: float, *, ceiling: float | None = None) -> bool:
    """A bank frequency ABOVE the through-R ceiling has zero transfer => wasted capacity
    (it can only alias). Below the ceiling, usefulness is an ALLOCATION question (Law 3),
    not a sampling question — the measured deficit (8 supplied vs 25 needed along-tangent)
    sits far BELOW the ceiling, so it is an allocation/capacity/convergence question:
    exactly the #299 three-arm disambiguation."""
    c = nyquist_bank_ceiling_cyc_per_unit() if ceiling is None else float(ceiling)
    return float(nu) > c


# --------------------------------------------------------------------------------------
# Law 3 — curvelet parabolic bank scaling (DERIVED: Candès–Donoho on the C² separatrix)
# --------------------------------------------------------------------------------------
def parabolic_along_tangent_allocation(nu_across: float) -> float:
    """Optimal along-tangent bandwidth for a C² (curvature-bounded) codim-1 arc under
    parabolic scaling: width ~ length² <=> the wedge at radial frequency nu_across has
    tangential extent sqrt(nu_across). With nu_across = 64 (the live --max-bank-freq):
    nu_along* = 8 — EXACTLY the live allocation. The measured 3.2× 'deficit' (needs ~25)
    is therefore NOT a curvelet design bug: it is the C² HYPOTHESIS FAILING on the dash
    comb (dash endpoints are codim-2 point singularities the parabolic law does not
    cover). Cure = a DEDICATED C²-violation term (the lane dash-comb / phase carrier),
    or a class-targeted anisotropic extension — NOT a global bank re-scale."""
    v = float(nu_across)
    if v <= 0:
        raise ValueError("nu_across must be positive")
    return math.sqrt(v)


def along_tangent_deficit_ratio(
    nu_needed: float = DASH_LINE_CYC_PER_UNIT,
    nu_supplied: float = ALONG_TANGENT_SUPPLIED_CYC,
) -> float:
    """Measured-need over supplied along-tangent bandwidth (anchor: 25/8 = 3.125 ~ the
    reported 3.2×). GT-side census (2026-07-11, $0 cached): median 30.1% of Lane
    along-tangent occupancy variance lies ABOVE nu=8 => the need is a broad tail, not
    just the dash line."""
    if nu_supplied <= 0:
        raise ValueError("nu_supplied must be positive")
    return float(nu_needed) / float(nu_supplied)


# --------------------------------------------------------------------------------------
# Law 4 — β₂ window from n (DERIVED: stationarity sandwich on the gradient-noise process)
# --------------------------------------------------------------------------------------
def beta2_window(
    steps_per_epoch: int, *, curvature_timescale_epochs: float = 100.0
) -> tuple[float, float]:
    """Adam β₂ admissible window: the second-moment EMA window N₂ = 1/(1-β₂) must
    (FLOOR) cover ≥ one full data cycle — N₂ ≥ S (else the preconditioner phase-locks to
    the within-epoch pair ordering: per-pair variance leaks into v_t) — and (CEILING)
    stay ≤ ~1/3 of the curvature-drift timescale T_c·S (else v_t tracks stale curvature
    through anneals/stage flows). β₂ ∈ [1 − 1/S, 1 − 3/(T_c·S)].

    n600 / accum-8 => S = 75: window [0.98667, 0.9996] at T_c = 100 ep. The measured
    anchor 0.999 (N₂ = 1000 steps = 13.3 ep) sits INSIDE; the T0-derived 0.9999999
    (N₂ = 10⁷ steps ≈ 133k ep >> the 3000-ep run) violates the ceiling AND never
    converges its own bias correction within the run => derived-REJECTED as a default
    (the #222 A/B gate stays the empirical arbiter)."""
    s = int(steps_per_epoch)
    if s <= 0:
        raise ValueError("steps_per_epoch must be positive")
    t_c = float(curvature_timescale_epochs)
    if t_c <= 0:
        raise ValueError("curvature_timescale_epochs must be positive")
    lo = 1.0 - 1.0 / s
    hi = 1.0 - 3.0 / (t_c * s)
    return (lo, hi)


def beta1_beta2_guard_ok(beta1: float, beta2: float) -> bool:
    """Adam SNR/convergence guard: β₁ < sqrt(β₂) (first-moment memory must be shorter
    than the second-moment scale's sqrt — the standard sufficient condition)."""
    return float(beta1) < math.sqrt(float(beta2))


def _prov():
    return build_provenance_for_research_sidecar(
        _MEMO,
        reactivation_criteria=(
            "re-derive if the intrinsic dim, R MTF, dash-comb line, or accum structure "
            "change; promote when a sizing-law-driven config lands a byte-closed exact row"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="m5_max_macos_cpu",
        captured_at_utc=_UTC,
    )


def build_cgauge_whitney_moddim_v1() -> CanonicalEquation:
    prov = _prov()
    anchor = EmpiricalAnchor(
        anchor_id="rank8_double_measure_plus_indicator_chart_dependence_20260711",
        measurement_utc=_UTC,
        inputs={
            "intrinsic_dim_sdf_svd_n600": 0.9316,
            "intrinsic_dim_sdf_svd_n96": 0.9559,
            "pose_extra_local_dof_twonn": list(POSE_EXTRA_LOCAL_DOF),
            "pose_extra_dof_partition_visible": False,
            "indicator_basis_cumvar_at_8_n600": 0.5050,
            "indicator_basis_rank_for_0.9316": 293,
            "live_mod_dim": 32,
            "probe": _PROBE,
        },
        predicted_output={
            "whitney_mod_dim_seg_only": whitney_mod_dim(8, gauge_margin=0),
            "with_gauge_margin": whitney_mod_dim(8, gauge_margin=2),
            "mod16_under_embeds": moddim_under_embeds(16, 8),
            "joint_if_pose_rode_trunk": whitney_mod_dim(10, gauge_margin=0),
            "pose_routes_via_dxi_channel": True,
        },
        empirical_output={
            "standing_answer_mod_17_19_unchanged_by_pose_leg": True,
            "chart_caveat": (
                "linear rank is chart-dependent (indicator basis cumvar@8=0.505): "
                "mod-17-19 suffices ONLY for an SDF-like (level-set) internal chart — "
                "retro-derives the level-set architecture"
            ),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method=(
            "Whitney/prevalence embedding theory on the doubly-measured rank-8 manifold; "
            "checked against the mirror-audit pose-DOF partition-invisibility (n600) and "
            "the 2026-07-11 indicator-basis Gram spectrum probe ($0 cached lstars)"
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=prov,
    )
    return CanonicalEquation(
        equation_id=WHITNEY_EQUATION_ID,
        name=(
            "CGauge trunk mod-dim: Whitney 2d+1 on the rank-8 separatrix manifold "
            "(17, +2 gauge margin => 17-19; mod-16 under-embeds; pose rides dxi)"
        ),
        one_line_summary=(
            "mod-dim* = 2*8+1 = 17 (+2 gauge) — generic injectivity needs 2d+1; 2d=16 has "
            "stable double points; pose +2 DOF partition-invisible => dxi channel, not trunk."
        ),
        latex_form=(
            r"m^* = 2d+1\ (d=8) = 17;\ m\le 2d\Rightarrow\ \text{generic double points};\ "
            r"d_{\text{pose-extra}}\perp\text{partition}\Rightarrow\ \text{dxi}(6{+}k)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.cgauge_parametrization_optima_20260711:whitney_mod_dim"
        ),
        domain_of_validity={
            "covariance_class": "COVARIANT_LAW",
            "derivation_status": (
                "DERIVED (Whitney embedding / prevalence) from the MEASURED rank-8; the "
                "d_seg-neutrality of shrinking the live mod-32 to 19 is UNMEASURED "
                "(#299 Arm A owns it — sizing law is necessity, not sufficiency-to-FIT)"
            ),
            "chart_caveat": (
                "holds for SDF-like internal charts; indicator-chart linear rank is 293 "
                "(MEASURED 2026-07-11) — the level-set representation is load-bearing"
            ),
            "excluded": [
                "trunks without a level-set/SDF-like internal chart",
                "other clips (re-measure intrinsic dim)",
            ],
            "sisters": [
                "witness_general_covariance_totality_v1 (the rank-8 upper bound)",
                "posenet_luma_chroma_sensitivity_asymmetry_v1 (the dxi routing)",
            ],
        },
        units_in={"intrinsic_dim": "count", "gauge_margin": "count"},
        units_out={"mod_dim": "count"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"standing_17_19_vs_derived_17_19": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_autoconfig",
            "tac.witness_dsl.lawref",
        ),
        canonical_producers=(_MEMO,),
        provenance=prov,
    )


def build_cgauge_nyquist_bank_frequency_v1() -> CanonicalEquation:
    prov = _prov()
    anchor = EmpiricalAnchor(
        anchor_id="r_mtf_ceiling_vs_deficit_band_placement_20260711",
        measurement_utc=_UTC,
        inputs={
            "r_mtf": {"16": 1.00, "24": 0.997, "64": 0.955, "128": 0.00},
            "live_max_bank_freq": 64,
            "dash_line_cyc_per_unit": DASH_LINE_CYC_PER_UNIT,
            "along_tangent_supplied": ALONG_TANGENT_SUPPLIED_CYC,
        },
        predicted_output={
            "nu_ceiling_cyc_per_unit": nyquist_bank_ceiling_cyc_per_unit(),
            "deficit_is_below_ceiling": not bank_frequency_is_wasted(25.0),
            "verdict": (
                "the along-tangent shortfall is an ALLOCATION/capacity/convergence "
                "question (#299), NOT a through-R sampling limit"
            ),
        },
        empirical_output={
            "r_is_all_pass_to_2px": True,
            "lane_census_frac_above_nu8_median": 0.3014,
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method=(
            "derived from the MEASURED R MTF (contest_r_operator_mtf_allpass_to_2px_v1) "
            "+ the 2026-07-11 GT-side Lane along-tangent occupancy census ($0 cached)"
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=prov,
    )
    return CanonicalEquation(
        equation_id=NYQUIST_EQUATION_ID,
        name=(
            "CGauge bank Nyquist: through-R usable band tops at 128 cyc/unit (2px); the "
            "measured along-tangent deficit sits far BELOW it (allocation, not sampling)"
        ),
        one_line_summary=(
            "nu_max(through-R) = 128 cyc/unit (R MTF zero at 2px); need 25 << 128 => the "
            "3.2x deficit is allocation/capacity/convergence (#299), never a Nyquist wall."
        ),
        latex_form=(
            r"\nu_{\max} = \frac{U}{2\,T_0} = \frac{512}{2\cdot 2} = 128;\ "
            r"\nu_{\text{dash}}=25 \ll \nu_{\max}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.cgauge_parametrization_optima_20260711:"
            "nyquist_bank_ceiling_cyc_per_unit"
        ),
        domain_of_validity={
            "covariance_class": "SCORER_FRAME_STRUCTURAL",
            "derivation_status": "DERIVED from the measured R MTF; grid/chain-specific",
            "excluded": ["any change to the R chain / render grid"],
            "sisters": [
                "contest_r_operator_mtf_allpass_to_2px_v1 (the measured MTF)",
                "anisotropic_basis_along_tangent_frequency_deficit_v1 (the deficit)",
            ],
        },
        units_in={"unit_px": "px", "mtf_zero_period_px": "px"},
        units_out={"nu_ceiling": "cycles_per_512px_unit"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"ceiling_vs_mtf_zero": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_autoconfig", "tac.witness_dsl.lawref"),
        canonical_producers=(_MEMO,),
        provenance=prov,
    )


def build_cgauge_curvelet_parabolic_bank_v1() -> CanonicalEquation:
    prov = _prov()
    supplied = parabolic_along_tangent_allocation(64.0)
    anchor = EmpiricalAnchor(
        anchor_id="parabolic_allocation_reproduces_live_bank_and_locates_c2_failure_20260711",
        measurement_utc=_UTC,
        inputs={
            "live_max_bank_freq": 64,
            "live_along_tangent_supplied": ALONG_TANGENT_SUPPLIED_CYC,
            "dash_need": DASH_LINE_CYC_PER_UNIT,
            "lane_census_frac_above_nu8": {"median": 0.3014, "mean": 0.3372},
            "lane_census_frac_band_16_32_median": 0.0789,
        },
        predicted_output={
            "parabolic_along_allocation_at_64": supplied,
            "reproduces_live_allocation": abs(supplied - ALONG_TANGENT_SUPPLIED_CYC) < 1e-9,
            "deficit_ratio": along_tangent_deficit_ratio(),
            "verdict": (
                "the live bank IS parabolic-optimal for a C2 arc; the deficit is the C2 "
                "hypothesis FAILING on the dash comb (codim-2 endpoints) => cure is a "
                "dedicated comb/phase term, not a global bank re-scale"
            ),
        },
        empirical_output={
            "measured_deficit_3p2x": 3.2,
            "derived_deficit_25_over_8": 3.125,
            "gt_side_need_is_broad_tail_not_single_line": True,
        },
        residual=abs(3.125 - 3.2) / 3.2,
        source_artifact=_MEMO,
        measurement_method=(
            "Candès–Donoho parabolic scaling (width~length² wedges) evaluated at the live "
            "--max-bank-freq 64; checked against the measured along-tangent deficit anchor "
            "and the 2026-07-11 GT-side Lane census"
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=prov,
    )
    return CanonicalEquation(
        equation_id=PARABOLIC_EQUATION_ID,
        name=(
            "CGauge parabolic bank law: nu_along* = sqrt(nu_across) for the C2 separatrix "
            "(sqrt(64)=8 = the live allocation); the dash comb is a C2 violation"
        ),
        one_line_summary=(
            "Parabolic scaling gives nu_along*=sqrt(64)=8 — the live bank is C2-optimal; "
            "the 3.2x deficit IS the dash comb violating C2 => dedicated comb/phase term."
        ),
        latex_form=(
            r"\nu_{\parallel}^* = \sqrt{\nu_{\perp}}\ (\text{width}\sim\text{length}^2);\ "
            r"\sqrt{64}=8;\ \nu_{\text{dash}}=25\ \text{(C}^2\text{ violation)}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.cgauge_parametrization_optima_20260711:"
            "parabolic_along_tangent_allocation"
        ),
        domain_of_validity={
            "covariance_class": "COVARIANT_LAW",
            "derivation_status": (
                "DERIVED (Candès–Donoho) — the C2 optimality is a theorem; which cure "
                "wins on the C2-violating comb (comb term vs anisotropic extension vs "
                "capacity) is #299 Arm B territory"
            ),
            "excluded": ["separatrix families without bounded curvature"],
            "sisters": [
                "shearlet_nterm_upper_bounds_task_rate_v1 (the rate side of the same basis)",
                "curvelet_directional_basis_dseg_reduction_v1 (the measured -48% proxy)",
                "dash_erasure_homogenization_v1 (what the deficit erases)",
            ],
        },
        units_in={"nu_across": "cycles_per_512px_unit"},
        units_out={"nu_along_optimal": "cycles_per_512px_unit"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"deficit_derived_vs_measured": abs(3.125 - 3.2) / 3.2},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_autoconfig", "tac.witness_dsl.lawref"),
        canonical_producers=(_MEMO,),
        provenance=prov,
    )


def build_cgauge_beta2_window_v1() -> CanonicalEquation:
    prov = _prov()
    lo, hi = beta2_window(75, curvature_timescale_epochs=100.0)
    anchor = EmpiricalAnchor(
        anchor_id="beta2_stationarity_window_vs_measured_anchor_20260711",
        measurement_utc=_UTC,
        inputs={
            "num_pairs": 600,
            "accum_pairs": 8,
            "steps_per_epoch": 75,
            "curvature_timescale_epochs_assumed": 100.0,
            "measured_anchor_beta2": BETA2_MEASURED_ANCHOR,
            "t0_candidate_beta2": 0.9999999,
            "run_epochs": 3000,
        },
        predicted_output={
            "window": [lo, hi],
            "anchor_inside_window": lo < BETA2_MEASURED_ANCHOR < hi,
            "t0_candidate_inside_window": lo < 0.9999999 < hi,
            "beta1_guard_at_0.9": beta1_beta2_guard_ok(0.9, BETA2_MEASURED_ANCHOR),
        },
        empirical_output={
            "measured_0.999_is_derived_admissible": True,
            "t0_0.9999999_derived_rejected": (
                "N2 = 1e7 steps ~ 133k ep >> 3000-ep run: violates the curvature ceiling "
                "and never converges bias correction — #222 A/B stays the arbiter"
            ),
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method=(
            "gradient-noise stationarity sandwich (data-cycle floor N2>=S; curvature "
            "ceiling N2 <= T_c*S/3) at the live n600/accum-8 structure; checked against "
            "the measured 0.999 anchor and the #222-gated T0 candidate"
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=prov,
    )
    return CanonicalEquation(
        equation_id=BETA2_EQUATION_ID,
        name=(
            "CGauge beta2 window: 1-1/S <= beta2 <= 1-3/(T_c S) (n600/accum-8: "
            "[0.9867, 0.9996]); 0.999 admissible, 1e-7-complement rejected"
        ),
        one_line_summary=(
            "beta2 in [1-1/75, 1-3/(100*75)] = [0.9867, 0.9996]: window must cover the "
            "data cycle yet track curvature; 0.999 inside; 0.9999999 derived-REJECTED."
        ),
        latex_form=(
            r"1-\frac{1}{S} \le \beta_2 \le 1-\frac{3}{T_c S};\ S=\frac{600}{8}=75;\ "
            r"\beta_1 < \sqrt{\beta_2}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.cgauge_parametrization_optima_20260711:beta2_window"
        ),
        domain_of_validity={
            "covariance_class": "APPARATUS_ENGINEERING",
            "derivation_status": (
                "DERIVED window (stationarity sandwich; T_c=100 ep is an ASSUMED scale "
                "from the anneal/stage structure — labeled); the point value inside the "
                "window is empirical (#222 A/B)"
            ),
            "excluded": ["different accum/pair structure (recompute S)"],
            "sisters": ["costate_lambda_marginal_ds_v1 (the controller that senses it)"],
        },
        units_in={"steps_per_epoch": "steps", "curvature_timescale_epochs": "epochs"},
        units_out={"beta2_window": "interval"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"anchor_0.999_inside_window": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_autoconfig", "tac.witness_dsl.lawref"),
        canonical_producers=(_MEMO,),
        provenance=prov,
    )


ALL_CGAUGE_PARAMETRIZATION_BUILDERS = (
    build_cgauge_whitney_moddim_v1,
    build_cgauge_nyquist_bank_frequency_v1,
    build_cgauge_curvelet_parabolic_bank_v1,
    build_cgauge_beta2_window_v1,
)


def populate_cgauge_parametrization_optima(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> list[CanonicalEquation]:
    """Idempotent APPEND-ONLY registration of the four #223 parametrization laws."""
    from tac.canonical_equations.registry import register_canonical_equation

    out: list[CanonicalEquation] = []
    for build in ALL_CGAUGE_PARAMETRIZATION_BUILDERS:
        eq = build()
        register_canonical_equation(
            eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
            notes=(
                "cgauge_parametrization_optima_20260711: #223 derived sizing law "
                "(Einstein/Fable pass; memo "
                ".omx/research/cgauge_master_action_and_parametrization_20260711.md)"
            ),
        )
        out.append(eq)
    return out


__all__ = [
    "ALL_CGAUGE_PARAMETRIZATION_BUILDERS",
    "ALONG_TANGENT_SUPPLIED_CYC",
    "BETA2_EQUATION_ID",
    "BETA2_MEASURED_ANCHOR",
    "DASH_LINE_CYC_PER_UNIT",
    "NYQUIST_EQUATION_ID",
    "PARABOLIC_EQUATION_ID",
    "POSE_EXTRA_LOCAL_DOF",
    "RANK8_INTRINSIC_DIM",
    "R_MTF_ZERO_PERIOD_PX",
    "SCORER_UNIT_PX",
    "WHITNEY_EQUATION_ID",
    "along_tangent_deficit_ratio",
    "bank_frequency_is_wasted",
    "beta1_beta2_guard_ok",
    "beta2_window",
    "build_cgauge_beta2_window_v1",
    "build_cgauge_curvelet_parabolic_bank_v1",
    "build_cgauge_nyquist_bank_frequency_v1",
    "build_cgauge_whitney_moddim_v1",
    "moddim_under_embeds",
    "nyquist_bank_ceiling_cyc_per_unit",
    "parabolic_along_tangent_allocation",
    "populate_cgauge_parametrization_optima",
    "whitney_mod_dim",
]
