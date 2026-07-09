# SPDX-License-Identifier: MIT
"""Canonical equation: #169 HORIZON-WEIGHTED MARGIN (0-byte shared-structure d_seg lever).

The frontier residual d_seg flips split by the GT top-2 SegNet argmax MARGIN (``|m| = top1 − top2``
logit gap): the ``<0.05``-margin flips are IRREDUCIBLE frozen-SegNet label-noise (~193× concentrated —
chasing them is FITTING NOISE), while the flips at GT margin ∈ ``[0.3, 0.5]`` are the ONLY ones both
REDUCIBLE and STABLY-DECIDED (oracle ceiling ΔS≈0.024 at margin≥0.3 / 0.012 at margin≥0.5; derivation
``.omx/research/dseg_reducibility_gt_margin_verdict_20260623.md``). 97.8% of the frontier d_seg lives in
the horizon band (SEG rows ~96-288). So the lever is a ONE-SIDED SATISFICING HINGE on the SHARED realized
through-R witness GT-class margin ``m_wit`` (``= gt_class_logit − top_competitor_logit``; the ``_signed``
field #141 — NO 2nd SegNet forward, 0 archive bytes), STRATIFIED to a θ-INDEPENDENT mask:

    mask(x) = 1[row(x) ∈ [row_lo, row_hi)] · 1[m_gt(x) ∈ [lo, hi)]        (horizon rows AND reducible band)
    L_hz    = w_h · ( Σ_x relu(m_target − m_wit(x)) · mask(x) ) / ( Σ_x mask(x) + 1e-6 )

It pushes ONLY the reducible confident-GT band toward the ``m_target`` ceiling and EXCLUDES the ``<lo``
label-noise BY CONSTRUCTION. Zero gradient where ``m_wit ≥ m_target`` (satisficing — do not over-push
into the noise regime).

WHY it is byte-free AND rule-118 FREE. The term adds ZERO trainable params (a TRAIN-TIME loss reweighting,
ships nothing) → the archive is unchanged. The stratified mask is a deterministic function of the FIXED GT
margin field + the row band (θ-independent, precomputed ONCE per pair, numpy — a geometry PRIOR, not
learned/video-derived weights).

TWO PROVABLE DEGENERACIES (VERIFIED_VIA_SOURCE_INSPECTION — the byte-identity contract). ``w_h == 0`` ⇒
the trainer NEVER constructs the branch (default-OFF path) ⇒ the loss graph is byte-identical; the numpy
twin returns EXACTLY ``0.0`` there. An EMPTY mask (no pixel in the band) ⇒ ``Σ mask == 0`` ⇒ the ``+1e-6``
guard makes ``L_hz == 0`` (never a /0). ``m_wit ≥ m_target`` on the band ⇒ ``relu`` is 0 ⇒ ``L_hz == 0``
(satisficing).

est ΔS = MEASURED oracle CEILING 0.012–0.024 (ASSUMED_AWAITING_VERIFICATION as an ACHIEVED move — the
ceiling is the reducible-band UPPER bound; the OWED exit criterion is a CONVERGED n600 byte-close A/B that
the surviving flips shift to HIGHER GT margin). means != ends: this is a train-time prior (advisory,
NON-PROMOTABLE); pointer 0.19110 UNMOVED.

DSL leg: ``tac.witness_dsl.curriculum_dsl.HorizonWeightedMargin`` (--seg-horizon-margin-weight/-target/-lo/
-hi + --seg-horizon-row-lo/-hi + --seg-horizon-margin-start-epoch). Mechanism / reference twin:
``tac.boundary_math.horizon_weighted_margin``.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "horizon_weighted_margin_hinge_v1"

_UTC = "2026-07-09T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- the #169 oracle-ceiling relative-significance anchor numbers (load-bearing, quoted verbatim) -----
ORACLE_CEILING_DELTA_S_MARGIN_GE_030 = 0.024  # oracle ceiling at GT margin >= 0.30 (the [lo,hi] lo edge)
ORACLE_CEILING_DELTA_S_MARGIN_GE_050 = 0.012  # oracle ceiling at GT margin >= 0.50 (the [lo,hi] hi edge)
LEDGER_EST_DELTA_S = 0.018                     # midpoint est_delta_s (relative_significance seed row)
FRONTIER_DSEG_HORIZON_BAND_FRAC = 0.978       # 97.8% of frontier d_seg lives in SEG rows ~96-288


def build_horizon_weighted_margin_v1() -> CanonicalEquation:
    """Build the #169 horizon-weighted margin law with its honest-tier anchors.

    Two anchors: (1) the byte-identity / empty-band / satisficing DEGENERACIES,
    VERIFIED_VIA_SOURCE_INSPECTION (provable in code + pinned by the boundary_math tests); (2) the d_seg
    EFFECT, ASSUMED_AWAITING_VERIFICATION — the 0.012–0.024 is a MEASURED oracle CEILING on the reducible
    band, but the ACHIEVED move is owed a CONVERGED n600 byte-close A/B (exit criterion below)."""

    anchor_degeneracy = EmpiricalAnchor(
        anchor_id="horizon_weighted_margin_off_degeneracy_byte_identical_20260709",
        measurement_utc=_UTC,
        inputs={
            "mechanism": "tac.boundary_math.horizon_weighted_margin.horizon_weighted_margin_loss",
            "tests": "src/tac/boundary_math/tests/test_horizon_weighted_margin.py",
            "trainer_twin": ("experiments/train_levelset_witness_realized_through_R_mlx.py "
                             "(L4953-4962 loss block; L5615-5628 mask precompute)"),
        },
        predicted_output={
            "claim": ("weight==0 ⇒ term==0.0 exactly (trainer skips the branch — byte-identical); empty "
                      "mask ⇒ term==0.0 (the +1e-6 guard, no /0); m_wit>=m_target ⇒ term==0.0 (satisficing)"),
        },
        empirical_output={
            "weight_zero_is_zero": "loss==0.0 exactly (test_weight_zero_is_byte_identical_zero)",
            "empty_band_is_zero": "term==0.0 on an all-out-of-band margin (test_empty_mask_is_zero_no_divzero)",
            "satisficing_is_zero": "term==0.0 when m_wit>=m_target on the band (test_satisficing_zero_above_target)",
            "band_half_open": ("margin==lo IN, margin==hi OUT (test_band_is_half_open_lo_in_hi_out)"),
            "only_horizon_band_upweighted": ("only rows∈[rlo,rhi) AND margin∈[lo,hi) contribute "
                                             "(test_only_reducible_horizon_band_contributes)"),
            "verdict": ("the default-OFF trainer path is byte-identical by construction (the loss block is "
                        "gated on hz_w>0, default 0.0, provider None); the numpy twin's weight==0 / "
                        "empty-mask / satisficing returns are the runtime guarantee, not a promise"),
        },
        residual=0.0,
        source_artifact="src/tac/boundary_math/horizon_weighted_margin.py",
        measurement_method="source inspection + NO-FAKE unit tests (numpy $0) vs the trainer twin",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="src/tac/boundary_math/horizon_weighted_margin.py",
            reactivation_criteria=("re-verify the degeneracies if the hinge/mask formula changes "
                                   "(weight==0, empty-mask, and m_wit>=m_target must stay exactly 0)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_effect_owed = EmpiricalAnchor(
        anchor_id="horizon_weighted_margin_effect_owed_converged_ab_169_20260709",
        measurement_utc=_UTC,
        inputs={
            "ledger_row": (".omx/state/lever_relative_significance.jsonl "
                           "(lever horizon_weighted_margin_169)"),
            "source_anchor": ("#169 / .omx/research/dseg_reducibility_gt_margin_verdict_20260623.md / "
                              "relative_significance_reaudit_20260708"),
            "reducible_band": "GT top-2 margin ∈ [0.3, 0.5]; horizon SEG rows ~96-288",
        },
        predicted_output={
            "hypothesis": ("pushing the witness margin up on the reducible confident-GT band lowers d_seg "
                           "up to the oracle ceiling, WITHOUT chasing the <lo irreducible label-noise"),
        },
        empirical_output={
            "oracle_ceiling_delta_s_margin_ge_030": ORACLE_CEILING_DELTA_S_MARGIN_GE_030,
            "oracle_ceiling_delta_s_margin_ge_050": ORACLE_CEILING_DELTA_S_MARGIN_GE_050,
            "est_delta_s_midpoint": LEDGER_EST_DELTA_S,
            "frontier_dseg_horizon_band_frac": FRONTIER_DSEG_HORIZON_BAND_FRAC,
            "verdict_scope": "instance-owed-converged-ab",
            "status": ("OWED — 0.012–0.024 is a MEASURED oracle CEILING (frozen-SegNet, real GT, "
                       "through-R) on the reducible band, NOT an achieved witness move. The exit "
                       "criterion is a CONVERGED n600 byte-close A/B (re-run "
                       "tools/measure_dseg_reducibility_gt_margin.py --n-pairs 600 on ON vs OFF ckpts; "
                       "require the surviving flips shift to HIGHER GT margin, else terminal-finding). "
                       "This landing COMPLETES the fireable-lever triality (default-OFF, byte-identical); "
                       "it makes NO score claim. means != ends: pointer 0.19110 UNMOVED."),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="oracle-ceiling measurement (#169) + relative-significance re-audit; no witness A/B yet",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="horizon_weighted_margin.effect_owed_converged_ab",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("#169 horizon-weighted margin: one-sided satisficing hinge w_h·mean_{mask} relu(m_target − "
              "m_wit) on the SHARED realized through-R witness GT-class margin, STRATIFIED to (horizon "
              "rows) AND (GT margin ∈ [lo,hi)); 0-byte shared-structure d_seg lever"),
        one_line_summary=(
            "Satisficing relu(target−m_wit) hinge on the reducible horizon band (GT margin∈[0.3,0.5), "
            "rows~96-288); 0 bytes; OFF/empty/satisficed⇒0; ΔS ceiling 0.012–0.024 owed n600 A/B (#169)."
        ),
        latex_form=(
            r"L_{hz}=w_h\,\frac{\sum_x \mathrm{relu}(m^{*}-m_{wit}(x))\,\mathbb{1}[r(x)\in[r_0,r_1)]\,"
            r"\mathbb{1}[m_{gt}(x)\in[\ell,h)]}{\sum_x \mathbb{1}[\cdot]+\varepsilon}"
        ),
        python_callable_module_path=(
            "tac.boundary_math.horizon_weighted_margin:horizon_weighted_margin_loss"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "lever": ("tac.witness_dsl.curriculum_dsl.HorizonWeightedMargin "
                      "(--seg-horizon-margin-weight/-target/-lo/-hi, --seg-horizon-row-lo/-hi, "
                      "--seg-horizon-margin-start-epoch)"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("0-byte train-time loss reweighting on the reducible GT-margin band; the d_seg "
                     "SIGN/MAGNITUDE is ASSUMED_AWAITING_VERIFICATION until a CONVERGED n600 byte-close "
                     "A/B measures it (the 0.012–0.024 is an oracle CEILING, not an achieved move)"),
        },
        units_in={"m_wit": "segnet_gt_class_logit_margin", "gt_margin": "segnet_top1_top2_logit_gap",
                  "m_target": "segnet_logit_margin", "weight": "loss_weight"},
        units_out={"L_hz": "seg_loss_contribution"},
        empirical_anchors=(anchor_degeneracy, anchor_effect_owed),
        predicted_vs_empirical_residual={
            # the DEGENERACIES are exact (residual 0, source-verified); the d_seg EFFECT is OWED to the
            # converged n600 A/B (no witness measurement yet) -> 0 recorded with the ASSUMED anchor's status.
            "off_degeneracy_byte_identity": 0.0,
            "dseg_effect_owed_converged_ab": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",       # DSL leg: HorizonWeightedMargin factory
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # the wire-in consumer
        ),
        canonical_producers=(
            "tac.boundary_math.horizon_weighted_margin",
        ),
        provenance=build_provenance_for_predicted(
            model_id="horizon_weighted_margin.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_horizon_weighted_margin_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the #169 horizon-weighted margin law (mirrors
    ``populate_dseg_aware_fourier_taper_equation``; latest-row-wins). This is the EQUATIONS leg of the
    #169 lever build; the DSL leg is ``curriculum_dsl.HorizonWeightedMargin``, the mechanism / reference
    twin is ``tac.boundary_math.horizon_weighted_margin``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_horizon_weighted_margin_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="horizon_weighted_margin_20260709 (equations leg of the #169 lever build; DSL leg = "
              "HorizonWeightedMargin; oracle-ceiling est ΔS owed a converged n600 A/B)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "FRONTIER_DSEG_HORIZON_BAND_FRAC",
    "LEDGER_EST_DELTA_S",
    "ORACLE_CEILING_DELTA_S_MARGIN_GE_030",
    "ORACLE_CEILING_DELTA_S_MARGIN_GE_050",
    "build_horizon_weighted_margin_v1",
    "populate_horizon_weighted_margin_equation",
]
