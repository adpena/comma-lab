# SPDX-License-Identifier: MIT
"""Canonical equations: level-set WITNESS measured findings (2026-07-01 drift-fix).

TRIALITY drift-fix (operator directive 2026-07-01 "fix drift sources"): the DAG
leg (FEED-ly / FEED-dj / FEED 2026-06-24e / sig-proc memo) carried MEASURED
findings that were NOT yet formalized in the ``tac.canonical_equations`` leg —
a finding in one triality leg missing from the others IS drift. This module
registers the eight measured-but-unanchored rows named in the drift-fix task so
DAG ↔ DSL ↔ equations stay consistent.

Every equation here carries:
  * its DAG FEED cite (in the docstring + one_line_summary),
  * ≥1 EmpiricalAnchor with the MEASURED value + an honest caveat,
  * canonical_producers + canonical_consumers (no orphan equations),
  * an explicit ``domain_of_validity`` recording the axis + the honest limits.

⛔ NO-FAKE + ancestor-rule discipline (supreme): every number below is copied
verbatim from a MEASURED n600/n96 realized-through-R artifact, tagged
``[macOS-CPU advisory]`` / NON-PROMOTABLE. NONE is an exact-eval score — the
pointer is UNMOVED at 0.19110 and moves only through a composed θ* byte-closed
``#202`` exact row (CPU/CUDA, MPS never). Where a finding is a NEGATIVE result
(R near-all-pass) it is registered PRECISELY so it is not re-derived; where a
finding is circular-GT / self-orient-UNVERIFIED (curvelet) the caveat is stated
in ``domain_of_validity`` so no consumer treats it as an n600-realized number.

LAUNCH-CONFIG DECISIONS FLAGGED FOR THE WIRE-IN (#224): the l7-defect
(``l7_linf_sharpening_defect_in_smoothing_flow_v1``) and the
hosc-saturation-vs-step-basis (``step_basis_stability_vs_hosc_saturation_v1``)
equations RECORD the measured behavior but the ACTIVATION / CURRICULUM launch
choice is resolved by the sibling activation-research agent + the wire-in — this
module does NOT change ``witness_autoconfig.proven_base``.

Cross-references:
  * DAG ``.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md``
    FEED-ly (line ~7140) · FEED-dj (line ~1838) · FEED 2026-06-24e (line ~326)
    · argmax/power-diagram (line ~3193) · R-all-pass memo
    ``.omx/research/signal_processing_filter_levers_derived_20260701T014119Z.md``.
  * CLAUDE.md "Canonical equations + models registry — non-negotiable".
  * CLAUDE.md "The Triality — DAG ↔ DSL ↔ equations".
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

_UTC = "2026-07-01T00:00:00Z"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_ADVISORY = "[macOS-CPU advisory]"
_M5 = "m5_max_cpu"


def _sidecar(path: str, reactivation: str) -> object:
    return build_provenance_for_research_sidecar(
        sidecar_path=path,
        reactivation_criteria=reactivation,
        measurement_axis=_ADVISORY,
        hardware_substrate=_M5,
    )


def _predicted(model_id: str) -> object:
    return build_provenance_for_predicted(
        model_id=model_id,
        inputs_sha256="0" * 64,
        measurement_axis="[predicted]",
        hardware_substrate=_M5,
    )


# ───────────────────────── 1. oracle-R d_seg floor by render grid (FEED-ly) ──


_ORACLE_R_FLOOR = {"384": 0.00091, "192": 0.00247, "camera": 0.0}


def predict_oracle_r_dseg_floor(render_grid_px: int | str) -> float:
    """Ideal-frame → R → frozen SegNet argmax d_seg FLOOR at a render grid.

    ``render_grid_px`` ∈ {384, 192, "camera"}. The floor is the d_seg an IDEAL
    (oracle) frame incurs purely from the R observation-map + SegNet stride-2
    stem at that render resolution — the representation-achievable lower bound,
    NOT a byte-closed score. Finer render grid ⇒ lower floor (0.00091@384).
    """
    key = str(render_grid_px)
    if key not in _ORACLE_R_FLOOR:
        raise ValueError(
            f"render_grid_px={render_grid_px!r} must be one of {sorted(_ORACLE_R_FLOOR)}"
        )
    return _ORACLE_R_FLOOR[key]


def build_oracle_r_dseg_floor_by_render_grid_v1() -> CanonicalEquation:
    report = "reports/levelset_gate_discriminators_n600 (FEED-ly, #210)"
    anchors = tuple(
        EmpiricalAnchor(
            anchor_id=f"oracle_r_dseg_floor_grid{grid}_n600_20260701",
            measurement_utc=_UTC,
            inputs={"render_grid_px": grid, "pipeline": "ideal->R->frozen_cpu_torch_segnet_argmax",
                    "n_pairs": 600},
            predicted_output={"dseg_floor": _ORACLE_R_FLOOR[grid]},
            empirical_output={"dseg_floor": _ORACLE_R_FLOOR[grid]},
            residual=0.0,
            source_artifact=f"{_DAG} FEED-ly + tools/levelset_gate_discriminators_n600.py",
            measurement_method="n600_through_R_frozen_cpu_torch_segnet_argmax",
            provenance=_sidecar(report, "re-measure floor at the chosen render grid before an optimal-form run"),
        )
        for grid in ("384", "192")
    )
    return CanonicalEquation(
        equation_id="oracle_r_dseg_floor_by_render_grid_v1",
        name="Oracle-R d_seg floor by render grid (representation-achievable lower bound)",
        one_line_summary=(
            "FEED-ly: ideal->R->SegNet d_seg floor 0.00091@384 / 0.00247@192 / ~0@camera; "
            "witness ~0.003 above => TRAINING headroom (NOT a score)."
        ),
        latex_form=r"d_{seg}^{floor}(g) = \mathrm{argmax\text{-}disagree}\big(\mathrm{SegNet}(R_g(x^\star)),\, L^\star\big),\ g\in\{384,192,\mathrm{cam}\}",
        python_callable_module_path=(
            "tac.canonical_equations.witness_measured_findings_20260701:predict_oracle_r_dseg_floor"
        ),
        domain_of_validity={
            "measurement_axis": [_ADVISORY],
            "n_pairs": 600,
            "pipeline": "ideal_frame->R->frozen_cpu_torch_segnet_argmax",
            "caveat": "representation-achievable floor from an IDEAL input; NOT a byte-closed exact-eval score; pointer UNMOVED 0.19110",
        },
        units_in={"render_grid_px": "pixels_or_camera_token"},
        units_out={"dseg_floor": "fraction_of_pixels_argmax_disagree"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"n600_through_R_frozen_cpu_torch_segnet_argmax": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",
            "experiments.train_levelset_witness_realized_through_R_mlx",
        ),
        canonical_producers=("tools/levelset_gate_discriminators_n600.py",),
        provenance=_predicted("oracle_r_dseg_floor_by_render_grid.v1"),
    )


# ───────────────────── 2. AA / supersample lane-recall lift (FEED-ly) ────────


_AA_LANE_RECALL = {"no_aa": 0.56, "aa_supersample": 0.94}


def predict_aa_lane_recall(supersample_aa: bool) -> float:
    """Lane-class-1 recall through R with vs without supersample/AA render.

    0.56 (no AA) → 0.94 (AA/supersample) = +0.38 at ~0-rate. The lift proves
    the finest-scale lane ERASURE is DOMINANTLY aliasing / observation-map, not
    a physics floor — attackable by the render representation (≈0 extra bytes).
    """
    return _AA_LANE_RECALL["aa_supersample" if supersample_aa else "no_aa"]


def build_aa_supersample_lane_recall_lift_v1() -> CanonicalEquation:
    report = "reports/levelset_gate_discriminators_n600 (FEED-ly, #210)"
    anchors = (
        EmpiricalAnchor(
            anchor_id="aa_supersample_lane_recall_lift_n600_20260701",
            measurement_utc=_UTC,
            inputs={"no_aa_recall": 0.56, "aa_recall": 0.94, "n_pairs": 600,
                    "rate_delta": "~0 (render lever)"},
            predicted_output={"lane_recall_no_aa": 0.56, "lane_recall_aa": 0.94, "lift": 0.38},
            empirical_output={"lane_recall_no_aa": 0.56, "lane_recall_aa": 0.94, "lift": 0.38},
            residual=0.0,
            source_artifact=f"{_DAG} FEED-ly + tools/levelset_gate_discriminators_n600.py",
            measurement_method="n600_lane_class1_recall_through_R_frozen_cpu_torch_segnet_argmax",
            provenance=_sidecar(report, "re-measure lane recall at the chosen supersample factor in the optimal-form run"),
        ),
    )
    return CanonicalEquation(
        equation_id="aa_supersample_lane_recall_lift_v1",
        name="AA/supersample render lifts lane recall (finest-scale erasure is aliasing, not physics)",
        one_line_summary=(
            "FEED-ly: supersample/AA render lifts lane-class-1 recall 0.56->0.94 (+0.38 @~0-rate) "
            "=> finest-scale erasure is DOMINANTLY aliasing/observation-map."
        ),
        latex_form=r"\mathrm{recall}_{lane}(\mathrm{AA}) - \mathrm{recall}_{lane}(\neg\mathrm{AA}) = 0.94 - 0.56 = +0.38",
        python_callable_module_path=(
            "tac.canonical_equations.witness_measured_findings_20260701:predict_aa_lane_recall"
        ),
        domain_of_validity={
            "measurement_axis": [_ADVISORY],
            "n_pairs": 600,
            "metric": "lane_class1_recall_through_R",
            "caveat": "~0-rate render lever; recall proxy for finest-scale erasure; NOT a byte-closed d_seg score",
        },
        units_in={"supersample_aa": "bool"},
        units_out={"lane_recall": "fraction_of_gt_class1_cells_recovered"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"n600_lane_class1_recall_through_R_frozen_cpu_torch_segnet_argmax": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",
            "experiments.train_levelset_witness_realized_through_R_mlx",
        ),
        canonical_producers=("tools/levelset_gate_discriminators_n600.py",),
        provenance=_predicted("aa_supersample_lane_recall_lift.v1"),
    )


# ─────────────── 3. analytic-lane band d_seg recon floor + decomp (FEED-dj) ──


_ANALYTIC_LANE = {
    "band_target": 0.00087,        # ≈ the sub-0.15 lane need (FEED-dj hit this)
    "shape_fn": 0.00046,           # shape false-NEGATIVE (band captures shape < target)
    "dashgap_fp_hardmask": 0.00396,  # dash-gap false-POSITIVE, hard-mask recon (90% of recon d_seg)
    "dashgap_fp_sdf": 0.000193,    # SAME FP in the SDF/argmax framing (~20x collapse, structural containment)
}


def predict_analytic_lane_band_dseg(dash_gated: bool, sdf_framing: bool = True) -> float:
    """Analytic openpilot poly-band lane d_seg (FEED-dj, dseg-index L11 / #144/#145).

    The continuous band captures lane SHAPE better than needed (FN 0.00046) but
    paints lane into the dash GAPS (FP). Ungated hard-mask recon is FP-dominated
    (0.00396 = 90%); the SDF/argmax framing collapses the FP ~20x to 0.000193
    (deep road SDF dominates locally-supported lane SDF in gaps = STRUCTURAL
    containment); dash-gating drives toward the 0.00046 shape floor.
    """
    if dash_gated:
        return _ANALYTIC_LANE["shape_fn"]
    return _ANALYTIC_LANE["dashgap_fp_sdf"] if sdf_framing else _ANALYTIC_LANE["dashgap_fp_hardmask"]


def build_analytic_lane_band_dseg_recon_floor_v1() -> CanonicalEquation:
    report = "reports/levelset_analytic_lane_band_dseg_n600_20260701.json (FEED-dj, #144/#145)"
    anchor_target = EmpiricalAnchor(
        anchor_id="analytic_lane_band_target_dseg_20260701",
        measurement_utc=_UTC,
        inputs={"lane_model": "openpilot_deg3_ground_centerline+width+dash", "condition": "band_recon_target"},
        predicted_output={"band_dseg": _ANALYTIC_LANE["band_target"]},
        empirical_output={"band_dseg": _ANALYTIC_LANE["band_target"], "note": "≈ sub-0.15 lane need"},
        residual=0.0,
        source_artifact=f"{_DAG} FEED-dj + {report}",
        measurement_method="analytic_lane_band_recon_dseg_through_R_frozen_cpu_torch_segnet_argmax",
        provenance=_sidecar(report, "re-measure the fitted band d_seg after per-frame param-optimize through R"),
    )
    anchor_decomp = EmpiricalAnchor(
        anchor_id="analytic_lane_band_dseg_decomposition_20260701",
        measurement_utc=_UTC,
        inputs={"shape_fn": _ANALYTIC_LANE["shape_fn"],
                "dashgap_fp_hardmask": _ANALYTIC_LANE["dashgap_fp_hardmask"],
                "dashgap_fp_sdf": _ANALYTIC_LANE["dashgap_fp_sdf"]},
        predicted_output={"dash_gate_is_critical": True, "sdf_fp_collapse_x": 20.5},
        empirical_output={"shape_fn": _ANALYTIC_LANE["shape_fn"],
                          "dashgap_fp_hardmask": _ANALYTIC_LANE["dashgap_fp_hardmask"],
                          "dashgap_fp_sdf": _ANALYTIC_LANE["dashgap_fp_sdf"],
                          "note": "dash-gap FP dominates hard-mask recon (90%); SDF framing collapses it ~20x structurally"},
        residual=0.0,
        source_artifact=f"{_DAG} FEED-dj/FEED-di + {report}",
        measurement_method="analytic_lane_band_recon_dseg_through_R_frozen_cpu_torch_segnet_argmax",
        provenance=_sidecar(report, "re-measure the dash-gate FP after render-time gating"),
    )
    return CanonicalEquation(
        equation_id="analytic_lane_band_dseg_recon_floor_v1",
        name="Analytic openpilot lane-band d_seg recon floor + dash-gap FP decomposition",
        one_line_summary=(
            "FEED-dj: analytic poly-band lane d_seg 0.00087 (~lane need); FN shape 0.00046, "
            "dash-gap FP 0.00396 hardmask -> 0.000193 SDF => DASH-GATE is THE critical piece."
        ),
        latex_form=r"d_{seg}^{band} = \mathrm{FN}_{shape} + \mathrm{FP}_{dashgap};\ \mathrm{FN}=4.6e{-4},\ \mathrm{FP}_{hard}=3.96e{-3}\xrightarrow{SDF}1.93e{-4}",
        python_callable_module_path=(
            "tac.canonical_equations.witness_measured_findings_20260701:predict_analytic_lane_band_dseg"
        ),
        domain_of_validity={
            "measurement_axis": [_ADVISORY],
            "lane_chart": "openpilot_deg3_ground_frame_centerline_width_dash",
            "distinct_from": ["analytic_lane_render_band_fp_reduction_v1 (FEED-dv uncertainty-gate compose)",
                              "dm2_lane_ipm_polynomial_geodesic_v1"],
            "caveat": "band recon floor + FP decomposition, fit to GT class-1; NOT a byte-closed score; dash-gate is the binding lever",
        },
        units_in={"dash_gated": "bool", "sdf_framing": "bool"},
        units_out={"band_dseg": "fraction_of_pixels_argmax_disagree"},
        empirical_anchors=(anchor_target, anchor_decomp),
        predicted_vs_empirical_residual={"analytic_lane_band_recon_dseg_through_R_frozen_cpu_torch_segnet_argmax": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.boundary_math.lane_sdf_component",
            "tac.boundary_math.analytic_lane_render_band",
            "tac.witness_dsl.curriculum_dsl",
        ),
        canonical_producers=("tools/levelset_analytic_lane_band_dseg_n600.py",),
        provenance=_predicted("analytic_lane_band_dseg_recon_floor.v1"),
    )


# ─────────────── 4. curvelet all-class directional basis d_seg (⚠ circular-GT) ─


def predict_curvelet_dseg_reduction() -> float:
    """All-class directional (curvelet/anisotropic) Fourier basis d_seg reduction.

    MEASURED −48% vs the isotropic basis (basis-match is PRIOR to capacity).
    ⚠ CIRCULAR-GT synthetic + realized self-orient UNVERIFIED — see
    ``domain_of_validity``: this is NOT an n600 realized-through-R self-orienting
    number and MUST NOT be consumed as one.
    """
    return -0.48


def build_curvelet_directional_basis_dseg_reduction_v1() -> CanonicalEquation:
    report = f"{_DAG} FEED-ly / curvelet basis probe"
    anchor = EmpiricalAnchor(
        anchor_id="curvelet_directional_basis_dseg_minus48_circular_gt_20260701",
        measurement_utc=_UTC,
        inputs={"basis": "all_class_directional_curvelet_anisotropic", "baseline": "isotropic_fourier",
                "gt": "circular_synthetic", "n": 96},
        predicted_output={"dseg_reduction_fraction": -0.48},
        empirical_output={"dseg_reduction_fraction": -0.48,
                          "caveat": "CIRCULAR-GT synthetic; realized self-orient UNVERIFIED"},
        residual=0.0,
        source_artifact=report,
        measurement_method="n96_circular_gt_curvelet_basis_dseg_UNVERIFIED_self_orient",
        provenance=_sidecar(report, "re-measure -48% on n600 real GT with REALIZED self-orient (not circular synthetic)"),
    )
    return CanonicalEquation(
        equation_id="curvelet_directional_basis_dseg_reduction_v1",
        name="All-class directional (curvelet) basis d_seg reduction (circular-GT, self-orient UNVERIFIED)",
        one_line_summary=(
            "Curvelet all-class directional basis: -48% d_seg vs isotropic (basis-match PRIOR to capacity); "
            "⚠ n96 circular-GT, realized self-orient UNVERIFIED."
        ),
        latex_form=r"\Delta d_{seg}^{curvelet} / d_{seg}^{iso} = -0.48\ \ (\text{n96 circular-GT; self-orient UNVERIFIED})",
        python_callable_module_path=(
            "tac.canonical_equations.witness_measured_findings_20260701:predict_curvelet_dseg_reduction"
        ),
        domain_of_validity={
            "measurement_axis": [_ADVISORY],
            "n": 96,
            "gt": "circular_synthetic_NOT_real_n600",
            "self_orient": "UNVERIFIED",
            "caveat": "NOT an n600 realized-through-R self-orienting number; -48% is on synthetic circular GT with an oracle orientation; do NOT consume as a realized lever until re-measured (#185 ladder A/B)",
        },
        units_in={},
        units_out={"dseg_reduction_fraction": "signed_fraction_of_baseline_dseg"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"n96_circular_gt_curvelet_basis_dseg_UNVERIFIED_self_orient": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.curriculum_dsl",),
        canonical_producers=("tools/levelset_gate_discriminators_n600.py",),
        provenance=_predicted("curvelet_directional_basis_dseg_reduction.v1"),
    )


# ──────────── 5. step_basis stability vs hosc saturation (FEED 2026-06-24e) ──


def classify_activation_trainability(activation: str, hosc_beta: float | None = None) -> str:
    """Trainability verdict for step-native activations (FEED 2026-06-24e).

    Both ``hosc`` and ``step_basis`` are step-native (square-wave partition
    shape, NO Gibbs overshoot), so both fit the argmax target better than
    sinusoidal SIREN/FINER. BUT ``hosc`` SATURATES gradients at large β
    (trainability risk; finite hosc_beta knob + anneal), while ``step_basis``
    (Σ aₖ·tanh(gₖ(x−cₖ))) is trainability-stable. Theory d_seg-fit rank:
    hosc > step_basis > fkan > FINER≈SIREN. UNMEASURED at n600 — a LAUNCH-CONFIG
    decision flagged for the wire-in (#224).
    """
    a = activation.lower()
    if a == "hosc":
        if hosc_beta is not None and hosc_beta >= 8.0:
            return "step_native_gradient_saturation_risk_at_large_beta"
        return "step_native_best_dseg_fit_but_beta_saturation_risk"
    if a == "step_basis":
        return "step_native_trainability_stable"
    return "sinusoidal_gibbs_overshoot_shallow_flips"


def build_step_basis_stability_vs_hosc_saturation_v1() -> CanonicalEquation:
    report = f"{_DAG} FEED 2026-06-24e (activation research) + hosc_beta anneal unit test (DAG line ~3078)"
    anchor = EmpiricalAnchor(
        anchor_id="step_basis_vs_hosc_saturation_theory_ranked_20260701",
        measurement_utc=_UTC,
        inputs={"candidates": ["hosc", "step_basis", "fkan", "FINER", "SIREN"],
                "hosc_beta_anneal": "unit-tested LINEAR/COSINE 4->8 monotone (DAG line ~3078)"},
        predicted_output={"dseg_fit_rank": ["hosc", "step_basis", "fkan", "FINER", "SIREN"],
                          "hosc_risk": "gradient saturation at large beta",
                          "step_basis": "trainability stable"},
        empirical_output={"hosc_beta_anneal_moves": True,
                          "note": "12-lens THEORY-ranked + hosc_beta anneal unit-tested; n100/n600 d_seg screen UNMEASURED"},
        residual=0.0,
        source_artifact=report,
        measurement_method="activation_screen_theory_ranked_plus_hosc_beta_anneal_unit_test",
        provenance=_sidecar(report, "run the n600 activation screen (hosc/step_basis/finer_gauss/fkan) to MEASURE the d_seg rank"),
    )
    return CanonicalEquation(
        equation_id="step_basis_stability_vs_hosc_saturation_v1",
        name="step_basis trainability-stable vs hosc gradient-saturation (step-native activation axis)",
        one_line_summary=(
            "FEED 2026-06-24e: hosc & step_basis both step-native (no Gibbs); hosc SATURATES grads at large beta, "
            "step_basis is stable; d_seg-fit rank hosc>step_basis (n600 UNMEASURED). LAUNCH-CONFIG #224."
        ),
        latex_form=r"\text{step\_basis}(x)=\sum_k a_k\tanh(g_k(x-c_k));\ \text{hosc: } |\nabla|\to 0\ \text{as}\ \beta\uparrow",
        python_callable_module_path=(
            "tac.canonical_equations.witness_measured_findings_20260701:classify_activation_trainability"
        ),
        domain_of_validity={
            "measurement_axis": [_ADVISORY],
            "status": "THEORY-ranked (12-lens) + hosc_beta anneal unit-tested; n600 d_seg screen UNMEASURED",
            "launch_config_decision": "activation choice resolved by sibling activation-research agent + wire-in (#224); this equation RECORDS behavior, does NOT set witness_autoconfig.proven_base",
        },
        units_in={"activation": "activation_name_token", "hosc_beta": "dimensionless_or_none"},
        units_out={"trainability_verdict": "categorical_token"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"activation_screen_theory_ranked_plus_hosc_beta_anneal_unit_test": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
            "tac.witness_autoconfig",
        ),
        canonical_producers=("experiments.train_levelset_witness_realized_through_R_mlx",),
        provenance=_predicted("step_basis_stability_vs_hosc_saturation.v1"),
    )


# ────────── 6. l7 L∞-sharpening DEFECT in a smoothing flow + per-stage curve ─


_PER_STAGE_DSEG = {"CE": 0.00576, "tau": 0.00417, "l7": 0.00405}


def predict_per_stage_dseg(stage: str) -> float:
    """Witness per-stage n600 d_seg (CE->tau->l7 curriculum descent).

    CE 0.00576 -> tau_softplus 0.00417 -> l7 0.00405. The l7 stage moves d_seg
    only -0.00012 = the MEASURED d_seg-DECOUPLING: L∞ (l7) SHARPENING inside a
    SMOOTHING (curvature) flow is a DEFECT — it barely lowers d_seg (per FEED-ly
    the unified-flow correction: l7 sharpens the WRONG functional).
    """
    s = stage.lower()
    if s not in _PER_STAGE_DSEG:
        raise ValueError(f"stage={stage!r} must be one of {sorted(_PER_STAGE_DSEG)} (CE/tau/l7)")
    return _PER_STAGE_DSEG[s]


def build_l7_linf_sharpening_defect_in_smoothing_flow_v1() -> CanonicalEquation:
    report = "reports/levelset_calib_* n600 per-stage monitor (FEED-ly; 96-monitor +0.1-1.8% vs full n600)"
    anchors = tuple(
        EmpiricalAnchor(
            anchor_id=f"per_stage_dseg_{stage}_n600_20260701",
            measurement_utc=_UTC,
            inputs={"stage": stage, "n_pairs": 600, "monitor": "96-frame (+0.1-1.8% vs full n600)"},
            predicted_output={"dseg": _PER_STAGE_DSEG[stage]},
            empirical_output={"dseg": _PER_STAGE_DSEG[stage]},
            residual=0.0,
            source_artifact=f"{_DAG} FEED-ly + {report}",
            measurement_method="n600_per_stage_dseg_through_R_frozen_cpu_torch_segnet_argmax",
            provenance=_sidecar(report, "re-measure per-stage d_seg on the optimal-form curriculum (with/without l7)"),
        )
        for stage in ("CE", "tau", "l7")
    )
    l7_delta = _PER_STAGE_DSEG["l7"] - _PER_STAGE_DSEG["tau"]  # -0.00012 (the decoupling)
    return CanonicalEquation(
        equation_id="l7_linf_sharpening_defect_in_smoothing_flow_v1",
        name="l7 (L-inf sharpening) is a DEFECT in the smoothing flow (measured d_seg-decoupling)",
        one_line_summary=(
            "FEED-ly: per-stage n600 d_seg CE 0.00576->tau 0.00417->l7 0.00405; l7 moves d_seg only "
            "-0.00012 = L-inf sharpening decouples from d_seg in a smoothing flow. LAUNCH-CONFIG #224."
        ),
        latex_form=r"d_{seg}: 0.00576\xrightarrow{CE}\;0.00417\xrightarrow{\tau}\;0.00405\xrightarrow{l7};\ \Delta_{l7}=-1.2e{-4}\ (\text{decoupled})",
        python_callable_module_path=(
            "tac.canonical_equations.witness_measured_findings_20260701:predict_per_stage_dseg"
        ),
        domain_of_validity={
            "measurement_axis": [_ADVISORY],
            "n_pairs": 600,
            "monitor_note": "96-frame monitor tracks full n600 to +0.1-1.8%",
            "l7_verdict": "L-inf sharpening in a curvature-smoothing flow is a DEFECT (decoupled from d_seg)",
            "launch_config_decision": "whether to KEEP the l7 stage is resolved by the wire-in (#224); this equation RECORDS the measured decoupling",
        },
        units_in={"stage": "curriculum_stage_token_CE_tau_l7"},
        units_out={"dseg": "fraction_of_pixels_argmax_disagree"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"n600_per_stage_dseg_through_R_frozen_cpu_torch_segnet_argmax": abs(l7_delta) * 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",
            "tac.witness_autoconfig",
        ),
        canonical_producers=("experiments.train_levelset_witness_realized_through_R_mlx",),
        provenance=_predicted("l7_linf_sharpening_defect_in_smoothing_flow.v1"),
    )


# ───────────── 7. R transfer function near ALL-PASS (NEGATIVE result) ────────


_R_MTF = {"dc": 1.0, "nyquist_2p1px": 0.842, "wiener_ceiling_db": 1.25}


def predict_r_mtf_at_nyquist() -> float:
    """|H_R| at render-Nyquist (2.1px) — MEASURED 0.842 (R is near ALL-PASS).

    R is NOT a low-pass: |H_R| = 1.0 at DC → 0.842 at Nyquist; 1px dash keeps
    ~91% contrast; uint8 quant negligible. Wiener-inverse ceiling +1.25 dB
    (×1.15) ⇒ R-deconvolution / pre-emphasis-OF-R are DEAD — DO NOT build them.
    The observed low-pass is the GENERATOR (INR NTK spectral bias) + the
    DETECTOR (SegNet stride-2 stem), NOT R.
    """
    return _R_MTF["nyquist_2p1px"]


def build_r_transfer_function_near_all_pass_negative_v1() -> CanonicalEquation:
    memo = ".omx/research/signal_processing_filter_levers_derived_20260701T014119Z.md (commit f206231a4)"
    anchor = EmpiricalAnchor(
        anchor_id="r_mtf_near_all_pass_negative_20260701",
        measurement_utc=_UTC,
        inputs={"H_R_dc": 1.0, "H_R_nyquist_2p1px": 0.842, "wiener_ceiling_db": 1.25,
                "uint8_quant": "negligible"},
        predicted_output={"r_is_low_pass": False, "deconvolution_worth_building": False},
        empirical_output={"H_R_nyquist": 0.842, "wiener_ceiling_db": 1.25,
                          "verdict": "R near ALL-PASS => R-deconvolution/pre-emphasis-OF-R DEAD; low-pass is INR-NTK + SegNet stride-2 stem"},
        residual=0.0,
        source_artifact=memo,
        measurement_method="measured_R_MTF_probe_scratchpad_measure_R_mtf",
        provenance=_sidecar(memo, "N/A — NEGATIVE result; reopen only if the R operator definition changes"),
    )
    return CanonicalEquation(
        equation_id="r_transfer_function_near_all_pass_negative_v1",
        name="R transfer function is near ALL-PASS (NEGATIVE: R-deconvolution is DEAD)",
        one_line_summary=(
            "MEASURED: |H_R|=1.0->0.842 @Nyquist 2.1px, Wiener ceiling +1.25dB => R near ALL-PASS; "
            "R-deconvolution/pre-emphasis-OF-R DEAD; low-pass is INR-NTK + SegNet stride-2 stem."
        ),
        latex_form=r"|H_R(f)|: 1.0\!\to\!0.842\ (@\,f_{Nyq}=2.1\mathrm{px}),\ \text{Wiener}^{-1}\ \le +1.25\,\mathrm{dB}\Rightarrow \text{deconv worthless}",
        python_callable_module_path=(
            "tac.canonical_equations.witness_measured_findings_20260701:predict_r_mtf_at_nyquist"
        ),
        domain_of_validity={
            "measurement_axis": [_ADVISORY],
            "result_type": "NEGATIVE (registered so R-deconvolution is not re-derived)",
            "caveat": "the low-pass is the GENERATOR (INR-NTK) + DETECTOR (SegNet stride-2 stem), NOT R; attack those, not R",
        },
        units_in={},
        units_out={"H_R_at_nyquist": "dimensionless_magnitude_0_to_1"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"measured_R_MTF_probe_scratchpad_measure_R_mtf": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.differentiable_eval_roundtrip",
            "tac.witness_dsl.curriculum_dsl",
        ),
        canonical_producers=(".omx/research/signal_processing_filter_levers_derived_20260701T014119Z.md",),
        provenance=_predicted("r_transfer_function_near_all_pass_negative.v1"),
    )


# ──────────── 8. argmax-of-SDF IS an additively-weighted power diagram ────────


def classify_partition_chart() -> dict:
    """The K=5 argmax-of-SDF partition IS an additively-weighted power diagram.

    argmax over K=5 φ_k = an additively-weighted power diagram (Laguerre /
    Voronoi-with-weights) in R^K; its 1-skeleton = the Morse-Smale separatrix
    graph (FEED-fh: separatrix predicted at AUC 0.9987). The ONE un-covered
    stratum is the codim-2/3 Movable junction (a single smooth φ cannot carve
    multiple disconnected medial-axis car-blobs) = the representational hole.
    """
    return {
        "chart": "additively_weighted_power_diagram_in_R_K",
        "one_skeleton": "morse_smale_separatrix_graph",
        "separatrix_auc": 0.9987,
        "uncovered_stratum": "codim_2_3_movable_junction",
    }


def build_argmax_of_sdf_is_additively_weighted_power_diagram_v1() -> CanonicalEquation:
    report = f"{_DAG} (argmax/power-diagram line ~3193; FEED-fh separatrix AUC 0.9987)"
    anchor = EmpiricalAnchor(
        anchor_id="argmax_sdf_power_diagram_separatrix_auc_20260701",
        measurement_utc=_UTC,
        inputs={"K": 5, "chart": "additively_weighted_power_diagram", "one_skeleton": "morse_smale_separatrix"},
        predicted_output={"separatrix_is_power_diagram_1skeleton": True},
        empirical_output={"separatrix_auc": 0.9987,
                          "uncovered_stratum": "codim_2_3_movable_junction"},
        residual=abs(1.0 - 0.9987),
        source_artifact=report,
        measurement_method="feed_fh_separatrix_prediction_auc",
        provenance=_sidecar(report, "re-measure separatrix AUC if the K-class SDF chart changes"),
    )
    return CanonicalEquation(
        equation_id="argmax_of_sdf_is_additively_weighted_power_diagram_v1",
        name="argmax-of-SDF IS an additively-weighted power diagram (Morse-Smale separatrix chart)",
        one_line_summary=(
            "IDENTITY: argmax of K=5 phi_k = additively-weighted power diagram in R^K; 1-skeleton = "
            "Morse-Smale separatrix (FEED-fh AUC 0.9987); codim-2/3 Movable junction = un-covered stratum."
        ),
        latex_form=r"\arg\max_k \phi_k(x)\ \equiv\ \text{power diagram};\ \partial = \text{Morse-Smale separatrix}\ (\mathrm{AUC}=0.9987)",
        python_callable_module_path=(
            "tac.canonical_equations.witness_measured_findings_20260701:classify_partition_chart"
        ),
        domain_of_validity={
            "measurement_axis": [_ADVISORY],
            "result_type": "representational identity + measured separatrix AUC",
            "uncovered_stratum": "codim-2/3 Movable junction (a single smooth phi cannot carve multi-component Movable)",
        },
        units_in={},
        units_out={"chart_facts": "structured_dict"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"feed_fh_separatrix_prediction_auc": abs(1.0 - 0.9987)},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.boundary_math.lane_sdf_component",
            "tac.witness_dsl.curriculum_dsl",
        ),
        canonical_producers=("tools/levelset_gate_discriminators_n600.py",),
        provenance=_predicted("argmax_of_sdf_is_additively_weighted_power_diagram.v1"),
    )


# ─────────────────────────── aggregator ─────────────────────────────────────


def build_all_witness_measured_findings_20260701() -> tuple[CanonicalEquation, ...]:
    """Build all 8 level-set witness measured-finding equations (2026-07-01 drift-fix)."""
    return (
        build_oracle_r_dseg_floor_by_render_grid_v1(),
        build_aa_supersample_lane_recall_lift_v1(),
        build_analytic_lane_band_dseg_recon_floor_v1(),
        build_curvelet_directional_basis_dseg_reduction_v1(),
        build_step_basis_stability_vs_hosc_saturation_v1(),
        build_l7_linf_sharpening_defect_in_smoothing_flow_v1(),
        build_r_transfer_function_near_all_pass_negative_v1(),
        build_argmax_of_sdf_is_additively_weighted_power_diagram_v1(),
    )


__all__ = [
    "build_all_witness_measured_findings_20260701",
    "build_oracle_r_dseg_floor_by_render_grid_v1",
    "build_aa_supersample_lane_recall_lift_v1",
    "build_analytic_lane_band_dseg_recon_floor_v1",
    "build_curvelet_directional_basis_dseg_reduction_v1",
    "build_step_basis_stability_vs_hosc_saturation_v1",
    "build_l7_linf_sharpening_defect_in_smoothing_flow_v1",
    "build_r_transfer_function_near_all_pass_negative_v1",
    "build_argmax_of_sdf_is_additively_weighted_power_diagram_v1",
    "predict_oracle_r_dseg_floor",
    "predict_aa_lane_recall",
    "predict_analytic_lane_band_dseg",
    "predict_curvelet_dseg_reduction",
    "classify_activation_trainability",
    "predict_per_stage_dseg",
    "predict_r_mtf_at_nyquist",
    "classify_partition_chart",
]
