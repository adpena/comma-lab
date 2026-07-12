# SPDX-License-Identifier: MIT
"""Canonical equation: LEVER-4c ANNULUS-DIRECTED CHROMA-BOUNDARY MATCH (0-byte shared-structure d_seg lever).

SegNet reads RGB ⇒ its per-pixel argmax depends on CHROMA (``rgb − BT.601-luma``), not just luma. The
chroma-DOF probe a3e9f0bd (MEASURED n96, 100% L*-match to the frozen SegNet; #276 chroma-DOF; eq
``chroma_decides_lane_and_movable_at_annulus_v1``) found removing chroma (constant-luma) FLIPS ``7.54%`` of
Lane→Road + ``4.38%`` of Movable→Undrivable, with ``93.4%`` of chroma-flips in the ``margin < 1`` fragile
ANNULUS (→ ``33.7%`` at margin<0.25); the flips are LUMA-INDEPENDENT (desat-only still flips ``3.1%``) and
the SegNet margin-gradient energy is ``78.8%`` luma / ``21.2%`` chroma. So chroma is a PROVEN INDEPENDENT
d_seg BOUNDARY SHARPENER, ORTHOGONAL to the geometry levers. The witness UNDER-exploits it (its rendered
chroma converges to a near per-class CONSTANT palette; nothing supervises per-pixel chroma).

So LEVER-4c is an additive chroma-MATCH loss on the SHARED realized-through-R render ``f1`` (the SAME render
the SegNet forward / ``_signed`` come from — NO 2nd render, NO 2nd SegNet forward), over the θ-INDEPENDENT
fragile annulus:

    chroma(x) = rgb(x) − BT.601_luma(x)·[1,1,1]            (LUMA-INVARIANT ⇒ ORTHOGONAL to every luma lever)
    ann(x)    = 1[ m_gt(x) < band ]                         (fragile-annulus mask; band=1 ⇒ 93.4% of flips)
    L_c       = w · ( Σ_x ‖chroma(f1)(x) − chroma(GT)(x)‖² · ann(x) ) / ( Σ_x ann(x) + 1e-6 )

It pulls the per-pixel RGB head (``self.out = Linear(hidden, 3)``, which HAS per-pixel chroma capacity — the
constant palette is a convergence HABIT, not a structural ceiling) to paint the boundary chroma the constant
palette can't. Because chroma is LUMA-INVARIANT (subtracting BT.601 luma leaves ``rgb + c·[1,1,1]``
unchanged) the term is a per-pixel chroma MATCH at the boundary, NOT a full-RGB reconstruction.

WHY it is byte-free AND rule-118 FREE. The term adds ZERO trainable params (a TRAIN-TIME loss reweighting,
ships nothing) → the archive is unchanged. The GT chroma target + annulus mask are a deterministic function
of the FIXED GT frame + the FIXED GT margin field (θ-independent, precomputed ONCE per pair, numpy — a
geometry PRIOR, not learned/video-derived weights).

THREE PROVABLE DEGENERACIES (VERIFIED_VIA_SOURCE_INSPECTION — the byte-identity contract). ``w == 0`` ⇒ the
trainer NEVER constructs the branch (default-OFF path, ``chroma_bnd_w > 0.0`` gate) AND the providers stay
``None`` ⇒ the loss graph is byte-identical; the numpy twin returns EXACTLY ``0.0`` there. An EMPTY annulus
(no pixel with ``m_gt < band``) ⇒ ``Σ ann == 0`` ⇒ the ``+1e-6`` guard makes ``L_c == 0`` (never a /0). A
PERFECT chroma match on the annulus ⇒ the squared error is 0 ⇒ ``L_c == 0``.

est ΔS = UNMEASURED (ASSUMED_AWAITING_VERIFICATION). The 7.54% / 4.38% / 93.4% are a MEASURED chroma-REMOVAL
ABLATION (the DOF EXISTENCE proof), NOT the ADD-BACK score ΔS of THIS chroma-MATCH loss term through the
witness. The OWED exit criterion is a CONVERGED n600 byte-close A/B (ON vs OFF ckpts; the surviving annulus
flips must shift toward the GT chroma, else terminal-finding). means != ends: this is a train-time prior
(advisory, NON-PROMOTABLE); the canonical ``reports/latest.md`` pointer is unchanged by this law.

DSL leg: ``tac.witness_dsl.curriculum_dsl.SegChromaBoundary`` (--seg-chroma-boundary-weight/-margin-band/
-start-epoch). Mechanism / reference twin: ``tac.boundary_math.chroma_boundary_match``. Source DOF equation:
``chroma_decides_lane_and_movable_at_annulus_v1`` (tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703).
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

EQUATION_ID = "chroma_boundary_annulus_match_hinge_v1"

_UTC = "2026-07-09T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- the chroma-DOF probe a3e9f0bd anchor numbers (load-bearing, quoted verbatim; the SOURCE ablation) ----
CHROMA_REMOVAL_LANE_TO_ROAD_FLIP = 0.0754       # constant-luma flips 7.54% of Lane -> Road
CHROMA_REMOVAL_MOVABLE_TO_UNDRIV_FLIP = 0.0438  # constant-luma flips 4.38% of Movable -> Undrivable
CHROMA_FLIPS_IN_MARGIN_LT1_ANNULUS = 0.934      # 93.4% of chroma-flips are in the margin<1 annulus
CHROMA_FLIPS_IN_MARGIN_LT025 = 0.337            # 33.7% at margin<0.25 (the fragile band)
CHROMA_DESAT_ONLY_ANNULUS_FLIP = 0.031          # constant-luma DESAT still flips 3.1% (luma-independent)
MARGIN_GRAD_ENERGY_CHROMA_FRACTION = 0.212      # 21.2% of SegNet margin-gradient energy is chroma (the DOF)


def build_chroma_boundary_annulus_match_hinge_v1() -> CanonicalEquation:
    """Build the LEVER-4c chroma-boundary-match law with its honest-tier anchors.

    Two anchors: (1) the byte-identity / empty-annulus / perfect-match DEGENERACIES,
    VERIFIED_VIA_SOURCE_INSPECTION (provable in code + pinned by the boundary_math tests); (2) the d_seg
    EFFECT, ASSUMED_AWAITING_VERIFICATION — the 7.54%/4.38%/93.4% are a MEASURED chroma-REMOVAL ABLATION (the
    DOF EXISTENCE proof), but the ADD-BACK score ΔS of the chroma-MATCH term through the witness is UNMEASURED
    (exit criterion below)."""

    anchor_degeneracy = EmpiricalAnchor(
        anchor_id="chroma_boundary_match_off_degeneracy_byte_identical_20260709",
        measurement_utc=_UTC,
        inputs={
            "mechanism": "tac.boundary_math.chroma_boundary_match.chroma_boundary_loss",
            "tests": "src/tac/boundary_math/tests/test_chroma_boundary_match.py",
            "trainer_twin": ("experiments/train_levelset_witness_realized_through_R_mlx.py "
                             "(L4933-4944 loss block; L5731-5752 GT-chroma/annulus provider precompute)"),
        },
        predicted_output={
            "claim": ("weight==0 ⇒ term==0.0 exactly (trainer skips the branch AND providers None — "
                      "byte-identical); empty annulus ⇒ term==0.0 (the +1e-6 guard, no /0); perfect chroma "
                      "match on the annulus ⇒ term==0.0"),
        },
        empirical_output={
            "weight_zero_is_zero": "loss==0.0 exactly (test_weight_zero_is_byte_identical_zero)",
            "empty_annulus_is_zero": "term==0.0 on all-out-of-band margins (test_empty_annulus_is_zero_no_divzero)",
            "perfect_match_is_zero": "term==0.0 when witness chroma == GT chroma (test_perfect_match_is_zero)",
            "luma_invariant": ("chroma(rgb + c·[1,1,1]) == chroma(rgb) (test_chroma_is_luma_invariant) ⇒ "
                               "ORTHOGONAL to every luma lever"),
            "only_annulus_upweighted": ("only pixels with GT margin < band contribute "
                                        "(test_only_fragile_annulus_contributes)"),
            "verdict": ("the default-OFF trainer path is byte-identical by construction (the loss block is "
                        "gated on chroma_bnd_w>0, default 0.0, providers None); the numpy twin's weight==0 / "
                        "empty-annulus / perfect-match returns are the runtime guarantee, not a promise"),
        },
        residual=0.0,
        source_artifact="src/tac/boundary_math/chroma_boundary_match.py",
        measurement_method="source inspection + NO-FAKE unit tests (numpy $0) vs the trainer twin",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="src/tac/boundary_math/chroma_boundary_match.py",
            reactivation_criteria=("re-verify the degeneracies if the chroma/annulus/term formula changes "
                                   "(weight==0, empty-annulus, and perfect-match must stay exactly 0; chroma "
                                   "must stay luma-invariant)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_effect_owed = EmpiricalAnchor(
        anchor_id="chroma_boundary_match_effect_owed_converged_ab_276_20260709",
        measurement_utc=_UTC,
        inputs={
            "source_dof_equation": "chroma_decides_lane_and_movable_at_annulus_v1",
            "source_probe": "a3e9f0bd (n96, 100% L*-match to frozen SegNet) / #276 chroma-DOF / DAG FEED",
            "reducible_locus": "fragile annulus GT top-2 margin < 1 (93.4% of chroma-flips)",
        },
        predicted_output={
            "hypothesis": ("supervising the witness's per-pixel rendered chroma toward the GT chroma at the "
                           "fragile annulus recovers the annulus d_seg flips chroma decides (Lane/Road, "
                           "Movable/Undriv), independent of the luma levers"),
        },
        empirical_output={
            "chroma_removal_lane_to_road_flip": CHROMA_REMOVAL_LANE_TO_ROAD_FLIP,
            "chroma_removal_movable_to_undriv_flip": CHROMA_REMOVAL_MOVABLE_TO_UNDRIV_FLIP,
            "chroma_flips_in_margin_lt1_annulus": CHROMA_FLIPS_IN_MARGIN_LT1_ANNULUS,
            "chroma_flips_in_margin_lt025": CHROMA_FLIPS_IN_MARGIN_LT025,
            "chroma_desat_only_annulus_flip": CHROMA_DESAT_ONLY_ANNULUS_FLIP,
            "margin_grad_energy_chroma_fraction": MARGIN_GRAD_ENERGY_CHROMA_FRACTION,
            "verdict_scope": "instance-owed-converged-ab",
            "status": ("OWED — the 7.54%/4.38%/93.4% are a MEASURED chroma-REMOVAL ABLATION (the DOF "
                       "EXISTENCE proof, eq chroma_decides_lane_and_movable_at_annulus_v1), NOT the ADD-BACK "
                       "score ΔS of THIS chroma-MATCH loss term through the witness. That ΔS is UNMEASURED. "
                       "The exit criterion is a CONVERGED n600 byte-close A/B (ON vs OFF ckpts; the surviving "
                       "annulus flips must shift toward the GT chroma, else terminal-finding). This landing "
                       "COMPLETES the fireable-lever triality (default-OFF, byte-identical); it makes NO "
                       "score claim. means != ends: canonical reports/latest.md pointer UNMOVED."),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="chroma-removal ablation (probe a3e9f0bd) — the DOF existence; no add-back witness A/B yet",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="chroma_boundary_match.effect_owed_converged_ab",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("LEVER-4c annulus-directed chroma-boundary match: additive w·mean_{ann} ‖chroma(f1) − "
              "chroma(GT)‖² on the SHARED realized-through-R render over the fragile annulus 1[GT margin < "
              "band]; chroma := rgb − BT.601-luma (LUMA-INVARIANT); 0-byte shared-structure d_seg lever"),
        one_line_summary=(
            "Luma-invariant chroma-match hinge on the fragile annulus (GT margin<1, 93.4% of chroma-flips); "
            "0 bytes; OFF/empty/perfect-match⇒0; DOF-measured removal 7.54%/4.38%, add-back ΔS owed n600 A/B."
        ),
        latex_form=(
            r"L_{c}=w\,\frac{\sum_x \lVert \mathrm{chroma}(f_1)(x)-\mathrm{chroma}(\mathrm{GT})(x)\rVert^2\,"
            r"\mathbb{1}[m_{gt}(x)<\text{band}]}{\sum_x \mathbb{1}[m_{gt}(x)<\text{band}]+\varepsilon},\quad "
            r"\mathrm{chroma}(rgb)=rgb-\mathrm{luma}_{601}(rgb)\cdot[1,1,1]"
        ),
        python_callable_module_path=(
            "tac.boundary_math.chroma_boundary_match:chroma_boundary_loss"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "lever": ("tac.witness_dsl.curriculum_dsl.SegChromaBoundary "
                      "(--seg-chroma-boundary-weight/-margin-band/-start-epoch)"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("0-byte train-time chroma-match loss on the fragile annulus; the d_seg SIGN/MAGNITUDE "
                     "of the ADD-BACK is ASSUMED_AWAITING_VERIFICATION until a CONVERGED n600 byte-close A/B "
                     "measures it (the 7.54%/4.38%/93.4% are a REMOVAL ABLATION, not an achieved move). "
                     "The loss is now routed through the micro-batch twin with per-pair annulus normalization; "
                     "training drift is waived, but score authority remains owed."),
        },
        units_in={"f1_rgb": "witness_rendered_rgb_through_R", "gt_rgb": "gt_frame_rgb_at_segnet_res",
                  "gt_margin": "segnet_top1_top2_logit_gap", "weight": "loss_weight", "band": "segnet_logit_margin"},
        units_out={"L_c": "seg_loss_contribution"},
        empirical_anchors=(anchor_degeneracy, anchor_effect_owed),
        predicted_vs_empirical_residual={
            # the DEGENERACIES are exact (residual 0, source-verified); the d_seg add-back EFFECT is OWED to
            # the converged n600 A/B (no witness measurement yet) -> 0 recorded with the ASSUMED anchor status.
            "off_degeneracy_byte_identity": 0.0,
            "dseg_effect_owed_converged_ab": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",       # DSL leg: SegChromaBoundary factory
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # the wire-in consumer
        ),
        canonical_producers=(
            "tac.boundary_math.chroma_boundary_match",
        ),
        provenance=build_provenance_for_predicted(
            model_id="chroma_boundary_annulus_match_hinge.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_chroma_boundary_annulus_match_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the LEVER-4c chroma-boundary-match law (mirrors
    ``populate_horizon_weighted_margin_equation``; latest-row-wins). This is the EQUATIONS leg of the
    LEVER-4c build; the DSL leg is ``curriculum_dsl.SegChromaBoundary``, the mechanism / reference twin is
    ``tac.boundary_math.chroma_boundary_match``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_chroma_boundary_annulus_match_hinge_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="chroma_boundary_match_20260709 (equations leg of the LEVER-4c build; DSL leg = "
              "SegChromaBoundary; DOF-measured removal ablation, add-back ΔS owed a converged n600 A/B)",
    )
    return eq


__all__ = [
    "CHROMA_DESAT_ONLY_ANNULUS_FLIP",
    "CHROMA_FLIPS_IN_MARGIN_LT025",
    "CHROMA_FLIPS_IN_MARGIN_LT1_ANNULUS",
    "CHROMA_REMOVAL_LANE_TO_ROAD_FLIP",
    "CHROMA_REMOVAL_MOVABLE_TO_UNDRIV_FLIP",
    "EQUATION_ID",
    "MARGIN_GRAD_ENERGY_CHROMA_FRACTION",
    "build_chroma_boundary_annulus_match_hinge_v1",
    "populate_chroma_boundary_annulus_match_equation",
]
