# SPDX-License-Identifier: MIT
"""Textured power diagram — the corrected scored sufficient statistic (Fable synthesis,
FEED-fable-synthesis, 2026-07-10).

Three laws derived by the synthesis pass over five convergent measured findings
(`.omx/research/fable_synthesis_texture_partition_20260710.md`):

(1) TEXTURED SUFFICIENT STATISTIC (MEASURED composite). The scored object is
    W = (G, xi, T): power-diagram generators G of the frame_1 argmax partition + the ego-screw
    xi(t) + per-class stationary SegNet-legible texture measures T = {t_c}. The partition-only
    formulation (any palette) has a MEASURED realized floor d_seg 0.0416 (zero-R-mixing,
    best palette = per-pair scene mean; all-palette scope, n600); textured realization
    (trained witness) measures 0.0048 (8.7x below); the GT canary is 1.6e-7. Corrected
    indirect-RD floor gains R(T) in [0.003, 0.013] S (per-class statistics, amortized once
    per video; decode-side synthesis is rule-118 free code) and a texture-legibility
    distortion term 100*d_seg*(T) whose MEASURED bracket [1.6e-7, 0.0048] is now the
    floor-dominating uncertainty of the campaign.

(2) SCORER OBLIGATION MATRIX (DERIVED from code + MEASURED Jacobian; REFINED by UNIT C
    `frame0_chromahf_dofs_20260710.md`, landed mid-synthesis). The scorer factorizes
    (frames x frequency-bands) into a 2x2 obligation matrix: frame_0 owes only pose
    (x[:,-1] frame select; n600 MEASURED d_seg 8.5e-9 with random-noise f0); frame_1 chroma-HF
    AT THE yuv6/384 PLANE owes only seg (2x2 box-average null MEASURED op-level 3.4e-6; SegNet
    reads raw RGB at full 384x512; ideal-lever seg authority 2.73e-3 at zero pose cost) —
    a NAIVE camera-res chroma dither does NOT access it (50% luma leak through the 2.28x no-AA
    bilinear downsample; must band-design / pre-image through the exact D kernel); frame_1 luma
    is the only doubly-priced block (11.1x per-plane pose sensitivity, MEASURED); frame_0
    chroma-HF is EXACTLY UNSCORED (dead subspace). W's components have canonical cheap
    embeddings: xi + luma-only pose render -> frame_0 (-67% bytes at sqrt10-pose 0.180 UPPER
    bound, MEASURED); T -> frame_1 384-band-designed chroma first (the chroma-alone BASIN
    legibility CONJECTURE is pre-registered, NOT asserted here). Composition of the two levers
    is ORTHOGONAL by construction (disjoint scorer terms, MEASURED via the two exact-null legs).

(3) FLIP MARGIN STEP LAW (DERIVED; accuracy OWED). For a residual flip at seg pixel p with
    margin deficit mu(p) = z_wrong - z_gt > 0, the optimal through-R camera step is
    d* = D^T grad_x m_p |_footprint (margin-saliency gradient pulled back through the exact
    bilinear-down adjoint, uint8-dead-zone gated), alpha* = mu / ||P_R grad_x m_p||, with a
    secant/Newton 1-D correction and a per-cluster QP for footprint overlap. The naive s0=1
    form MEASURED prediction-vs-realized 0.594 (the missing inner Jacobian is the gap);
    the corrected law's lift is the pre-registered owed row (>=0.85 target).

means != ends: advisory geometry/characterisation, NOT a score; pointer contest-CPU 0.19110
UNMOVED. Conjectures (chroma-alone legibility; PS-statistics sufficiency) are NOT registered.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

TEXTURED_STATISTIC_EQUATION_ID = "textured_power_diagram_sufficient_statistic_v1"
OBLIGATION_MATRIX_EQUATION_ID = "scorer_obligation_matrix_factorization_v1"
FLIP_STEP_LAW_EQUATION_ID = "flip_margin_step_law_v1"

_UTC = "2026-07-10T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_SYNTH_MEMO = ".omx/research/fable_synthesis_texture_partition_20260710.md"

# MEASURED anchors (all pre-existing artifacts; no new compute in this synthesis).
DSEG_FLAT_PAINT_FLOOR_ALL_PALETTE = 0.041622  # palette probe arm 4b, zero-R-mixing, n600
DSEG_TEXTURED_WITNESS_REALIZED = 0.0048       # trained self-orient witness, realized through R
DSEG_GT_CANARY = 1.6e-7                       # U1: real GT frame1 through R (n64)
LUMA_CHROMA_POSE_SENSITIVITY_PER_PLANE = 11.1 # measured per-channel PoseNet Jacobian ratio
FLIP_VERIFY_NAIVE_S0 = 0.594                  # #391 targeted verify under the s0=1 step model
RATE_T_BAND_S = (0.003, 0.013)                # 5 classes x ~1-4KB quantized PS statistics


def build_textured_power_diagram_sufficient_statistic_v1() -> CanonicalEquation:
    """(1) The scored sufficient statistic is (G, xi, T); partition-without-texture floors at
    0.0416 (all-palette, MEASURED); textured realization measures 0.0048; the corrected floor's
    dominant uncertainty is the texture-legibility gap d_seg*(T) in [1.6e-7, 0.0048]."""
    anchor = EmpiricalAnchor(
        anchor_id="textured_statistic_palette_floor_vs_witness_20260710",
        measurement_utc=_UTC,
        inputs={
            "palette_floor": ".omx/research/palette_artifact_probe_20260710.md (arm 4b, n600)",
            "witness_realized": "MEMORY L68/L17 trained self-orient witness through-R",
            "gt_canary": "palette probe U1 (n64, d_seg ~ 1.6e-7)",
            "rate_T_parametrization": "Portilla-Simoncelli class statistics, 5 x ~1-4KB quantized",
        },
        predicted_output={
            "if_partition_sufficient": "any palette realization -> d_seg ~ boundary noise (REFUTED)"},
        empirical_output={
            "flat_paint_floor_all_palette": DSEG_FLAT_PAINT_FLOOR_ALL_PALETTE,
            "textured_witness_realized": DSEG_TEXTURED_WITNESS_REALIZED,
            "gt_canary": DSEG_GT_CANARY,
            "const_colour_tiles": "Undrivable 195/216, Road 1/216, Lane 0/216 (context-domination)",
            "corrected_floor": ("S_floor(W) = R(G) + R(xi) + R(T in [0.003,0.013]) "
                                "+ 100*d_seg*(T in [1.6e-7, 0.0048]) + sqrt(10*d_pose*)"),
            "consequence": ("the campaign's floor-dominating unknown is the texture-legibility "
                            "gap d_seg*(T), a single measurable scalar (rows E-1/E-2)"),
        },
        residual=0.0,
        source_artifact=".omx/research/palette_artifact_probe_verdict_20260710.json",
        measurement_method=("synthesis over pre-existing MEASURED rows (palette probe n600 + "
                            "witness realized + GT canary); no new compute"),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SYNTH_MEMO,
            reactivation_criteria=("rows E-1 (chroma-HF legibility) + E-2 (PS-metamer texture arm) "
                                   "pin d_seg*(T) tighter than the 3.5-order bracket"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=TEXTURED_STATISTIC_EQUATION_ID,
        name=("textured power diagram: the scored sufficient statistic is (G, xi, T); "
              "partition-only realization floors at d_seg 0.0416 (all-palette), textured at 0.0048"),
        one_line_summary=("The partition alone is NOT the sufficient statistic: per-class "
                          "SegNet-legible texture T is load-bearing; its legibility gap "
                          "d_seg*(T) in [1.6e-7, 0.0048] dominates the corrected floor."),
        latex_form=(r"W=(G,\xi,T);\;S_{\mathrm{floor}}(W)=R(G)+R(\xi)+R(T)"
                    r"+100\,d^*_{\mathrm{seg}}(T)+\sqrt{10\,d^*_{\mathrm{pose}}},\;"
                    r"d^*_{\mathrm{seg}}(T)\in[1.6\mathrm{e}{-7},\,4.8\mathrm{e}{-3}]"),
        python_callable_module_path="tac.through_r.palette_realization:realize_partition_through_r",
        domain_of_validity={
            "vehicle": ["task_space_witness", "v8_perclass_carriers"],
            "scope": ("frame_1 frozen CPU-torch SegNet argmax through the pinned R; "
                      "T parametrization (PS statistics) is a candidate, sufficiency CONJECTURED "
                      "(pre-registered row E-2, NOT asserted)"),
            "measurement_axis": ["macOS-CPU advisory"]},
        units_in={"d_seg": "argmax_mismatch_rate", "rate_T": "S_points"},
        units_out={"S_floor": "contest_score_points"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "textured_vs_flat_floor_ratio": DSEG_FLAT_PAINT_FLOOR_ALL_PALETTE
            / DSEG_TEXTURED_WITNESS_REALIZED},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(".omx/research/SPEC_v8.1_20260709.md (carrier design)",
                             "tac.through_r.palette_realization"),
        canonical_producers=("tac.through_r.palette_realization",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SYNTH_MEMO,
            reactivation_criteria="rows E-1/E-2 (the two pre-registered texture falsifiers)",
            measurement_axis=_ADVISORY, hardware_substrate="apple_m5_max_cpu"),
    )


def build_scorer_obligation_matrix_factorization_v1() -> CanonicalEquation:
    """(2) The frames x bands obligation matrix: f0 = pose-only, f1-chromaHF = seg-only,
    f1-luma = doubly priced, f0-chromaHF = exactly unscored (dead subspace)."""
    anchor = EmpiricalAnchor(
        anchor_id="scorer_obligation_matrix_20260710",
        measurement_utc=_UTC,
        inputs={
            "frame_select": "upstream/modules.py:108 x[:,-1] (SegNet sees frame_1 only)",
            "chroma_subsample": "upstream/frame_utils.py:65-72 (2x2 box-average U/V)",
            "jacobian": "per-channel PoseNet Jacobian, gt real pairs (alldim re-read)",
            "frame0_seg_free": "cascade_c_prime 87 perturbation modes seg_delta=0.0",
        },
        predicted_output={"factorization": "2x2 obligation matrix (frames x frequency-bands)"},
        empirical_output={
            "f0_all": ("pose-only (seg obligation identically zero; n600 MEASURED d_seg 8.5e-9 "
                       "with random-noise f0, UNIT C); luma-only f0 = efficient point "
                       "(-67% bytes, sqrt10-pose 0.180 UPPER bound)"),
            "f1_chroma_hf": ("seg-only AT THE yuv6/384 PLANE (box-average null MEASURED 3.4e-6; "
                             "ideal-lever seg authority 2.73e-3 at zero pose cost); naive "
                             "camera-res chroma dither leaks 50% to luma through the 2.28x "
                             "no-AA downsample -> must band-design / pre-image through D"),
            "f1_luma": f"doubly priced ({LUMA_CHROMA_POSE_SENSITIVITY_PER_PLANE}x per-plane pose "
                       "sensitivity vs chroma, MEASURED)",
            "f0_chroma_hf": "EXACTLY UNSCORED (dead subspace: pose-null AND seg-free)",
            "composition": "ORTHOGONAL by construction (disjoint scorer terms; UNIT C P12)",
            "placement": ("xi + luma-only pose render -> f0; T -> f1 384-band-designed chroma "
                          "first (chroma-alone BASIN legibility is a pre-registered CONJECTURE, "
                          "row E-1)"),
        },
        residual=0.0,
        source_artifact=".omx/research/upstream_scorer_alldim_reread_20260710.md",
        measurement_method="line-by-line source inspection + measured per-channel Jacobian",
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SYNTH_MEMO,
            reactivation_criteria=("row E-1 decides the chroma-first placement recommendation "
                                   "(the matrix itself is code-exact and stable)"),
            measurement_axis=_ADVISORY, hardware_substrate="apple_m5_max_cpu"),
    )
    return CanonicalEquation(
        equation_id=OBLIGATION_MATRIX_EQUATION_ID,
        name=("scorer obligation matrix: (frames x bands) factorization — f0 pose-only, "
              "f1-chromaHF seg-only, f1-luma doubly priced, f0-chromaHF dead"),
        one_line_summary=("W's components have canonical cheap embeddings in the scorer's "
                          "channel geometry; only frame_1 luma is jointly constrained."),
        latex_form=(r"\mathrm{Obl}=\begin{pmatrix}(\mathrm{pose},0)&(0,0)\\"
                    r"(\mathrm{pose}\,{\times}11.1,\mathrm{seg})&(0,\mathrm{seg})\end{pmatrix}"
                    r"\;\text{rows}=f_0,f_1;\ \text{cols}=\mathrm{luma},\mathrm{chroma\text{-}HF}"),
        python_callable_module_path=(
            "tac.canonical_equations.posenet_luma_chroma_asymmetry_20260710"),
        domain_of_validity={
            "vehicle": ["task_space_witness", "v8_perclass_carriers"],
            "scope": ("exact for the pinned upstream scorer (modules.py/frame_utils.py verified "
                      "line-by-line); composes the registered luma/chroma asymmetry + frame-select "
                      "laws into ONE factorization"),
            "measurement_axis": ["macOS-CPU advisory"]},
        units_in={"channel_block": "frames_x_frequency_bands"},
        units_out={"obligation": "scored_distortion_terms_owed"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"luma_chroma_per_plane_ratio":
                                         LUMA_CHROMA_POSE_SENSITIVITY_PER_PLANE},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(".omx/research/DUAL_CHAIN_BRIEF_385_20260710.md (build-wave routing)",
                             ".omx/research/SPEC_v8.1_20260709.md"),
        canonical_producers=("tac.canonical_equations.posenet_luma_chroma_asymmetry_20260710",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SYNTH_MEMO,
            reactivation_criteria="row E-1 (chroma-HF legibility probe)",
            measurement_axis=_ADVISORY, hardware_substrate="apple_m5_max_cpu"),
    )


def build_flip_margin_step_law_v1() -> CanonicalEquation:
    """(3) The corrected flip step law: d* = D^T grad m |_footprint, alpha* = mu/||P_R grad m||,
    + secant + cluster QP. The naive s0=1 form measured verify 0.594; lift OWED (row E-3)."""
    anchor = EmpiricalAnchor(
        anchor_id="flip_step_law_naive_residual_20260710",
        measurement_utc=_UTC,
        inputs={
            "exact_outer": "flip_inverse.py composite kernel bit-exact (5.7e-14) + adjoint verified",
            "naive_step": "s0=1 (identity inner Jacobian), 16 LSB toward GT at adjoint pixels",
            "verify_set": "top-512 cheapest flips, 346 pairs, real frozen CPU-torch SegNet",
        },
        predicted_output={"corrected_law": ("d*=D^T grad_x m|_fp (margin-saliency pullback), "
                                            "alpha*=mu/||P_R grad m||, secant + cluster QP")},
        empirical_output={
            "naive_verify": FLIP_VERIFY_NAIVE_S0,
            "collateral": 82,
            "gap_attribution": ("missing inner Jacobian: stem gain != 1, ker(D)+deadzone loss, "
                                "activation curvature (DERIVED; sign+magnitude consistent)"),
            "owed": "row E-3: true-gradient + secant + QP, target verify >= 0.85, collateral <= 10",
        },
        residual=1.0 - FLIP_VERIFY_NAIVE_S0,
        source_artifact="experiments/results/resize_exploit_flip_solver_20260710T024926Z/verdict.json",
        measurement_method="targeted perturbation + re-measure through real SegNet (#391)",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SYNTH_MEMO,
            reactivation_criteria=("OWED row E-3: the corrected-law verify run; <=0.594 refutes "
                                   "the Jacobian gap attribution at this formulation"),
            measurement_axis=_ADVISORY, hardware_substrate="apple_m5_max_cpu"),
    )
    return CanonicalEquation(
        equation_id=FLIP_STEP_LAW_EQUATION_ID,
        name=("flip margin step law: optimal through-R flip fix = margin-saliency gradient "
              "pulled back through the exact resize adjoint, Newton-corrected, cluster-QP'd"),
        one_line_summary=("The 0.594 prediction-vs-realized gap is the missing inner Jacobian; "
                          "the corrected first-order + secant + QP law is derived, lift owed."),
        latex_form=(r"d^*(p)=D^{\top}\nabla_x m_p|_{\mathrm{fp}},\;"
                    r"\alpha^*(p)=\frac{\mu(p)}{\|P_R\nabla_x m_p\|}(1+O(\kappa\mu)),\;"
                    r"\min\|\delta x\|^2\ \mathrm{s.t.}\ J\delta x\ge\mu+\epsilon"),
        python_callable_module_path="tac.through_r.flip_inverse:build_targeted_perturbation",
        domain_of_validity={
            "vehicle": ["through_r_flip_corrector"],
            "scope": ("annulus/boundary flips (98.98% of residual); interior-texture flips are "
                      "T's domain (the #149 wall), NOT this law's; FORMALIZATION of the lift "
                      "PENDING row E-3"),
            "measurement_axis": ["macOS-CPU advisory"]},
        units_in={"mu": "logit_margin_deficit", "delta_x": "camera_uint8_lsb"},
        units_out={"delta_m": "logit_margin"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"naive_s0_verify_gap": 1.0 - FLIP_VERIFY_NAIVE_S0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.through_r.flip_inverse",),
        canonical_producers=("tac.through_r.flip_inverse:verify_targeted_fix",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SYNTH_MEMO,
            reactivation_criteria="row E-3 (corrected-law verify)",
            measurement_axis=_ADVISORY, hardware_substrate="apple_m5_max_cpu"),
    )


def populate_textured_power_diagram_equations(
        *, path=None, lock_path=None, agent=None, subagent_id=None):
    """Explicit registration (solver-pack pattern; NOT an import side-effect)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eqs = [
        build_textured_power_diagram_sufficient_statistic_v1(),
        build_scorer_obligation_matrix_factorization_v1(),
        build_flip_margin_step_law_v1(),
    ]
    for eq in eqs:
        register_canonical_equation(
            eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
            notes="fable_synthesis_texture_partition_20260710 (FEED-fable-synthesis; equations leg)")
    return eqs


__all__ = [
    "DSEG_FLAT_PAINT_FLOOR_ALL_PALETTE",
    "DSEG_GT_CANARY",
    "DSEG_TEXTURED_WITNESS_REALIZED",
    "FLIP_STEP_LAW_EQUATION_ID",
    "FLIP_VERIFY_NAIVE_S0",
    "LUMA_CHROMA_POSE_SENSITIVITY_PER_PLANE",
    "OBLIGATION_MATRIX_EQUATION_ID",
    "RATE_T_BAND_S",
    "TEXTURED_STATISTIC_EQUATION_ID",
    "build_flip_margin_step_law_v1",
    "build_scorer_obligation_matrix_factorization_v1",
    "build_textured_power_diagram_sufficient_statistic_v1",
    "populate_textured_power_diagram_equations",
]
