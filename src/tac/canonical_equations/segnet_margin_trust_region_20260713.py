# SPDX-License-Identifier: MIT
"""SegNet first-block margin trust-region law and measured local disposition."""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

SEGNET_MARGIN_TRUST_REGION_EQUATION_ID = "segnet_margin_trust_region_v1"
_SPEC = ".omx/research/frozen_segnet_cheap_validation_spec_20260713.md"
_MEMO = ".omx/research/frozen_segnet_cheap_validation_20260713.md"
_RECEIPT = "experiments/results/segnet_validation_certificate_20260713T015633Z/receipt.json"
_UTC = "2026-07-13T01:56:33Z"


def build_segnet_margin_trust_region_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria="supply a real suffix upper-bound artifact or measure a different pre-registered holdout formulation",
        measurement_axis="[macOS-CPU advisory; research_only]",
        hardware_substrate="apple_macos_cpu_numpy_torch",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="segnet_margin_trust_region_pair0_three_regimes_20260713",
        measurement_utc=_UTC,
        inputs={
            "pair": 0,
            "saved_regimes": ["early", "boundary", "late"],
            "holdout_candidates": 58,
            "seed": 20260712,
        },
        predicted_output={
            "falsifier": "NO-GO if a proxy accept worsens held CE/d_seg/d_pose or component economics fail 1.3x"
        },
        empirical_output={
            "proxy_accepts": 3,
            "proxy_rejects": 55,
            "dseg_unsafe_accepts": 0,
            "joint_unsafe_accepts": 2,
            "derived_speedup_k2": 0.9869128501255486,
            "derived_speedup_k4": 0.9818936607306264,
            "sequence_integrated_speedup": "UNKNOWN",
            "verdict": "NO_GO",
            "verdict_scope": "pair0; sealed early/boundary/late; blocks[0]; registered ladder; macOS-CPU advisory; this formulation only",
            "review_status": "fresh-eyes-reviewed(1)-CLEAN",
        },
        residual=1.0,
        source_artifact=_RECEIPT,
        measurement_method="disjoint empirical feature-ball holdout with identical-frame exact SegNet/Pose controls and derived component economics",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance="within-run exact functional comparison floor is zero; timing and across-seed floors remain UNKNOWN",
    )
    return CanonicalEquation(
        equation_id=SEGNET_MARGIN_TRUST_REGION_EQUATION_ID,
        name="Frozen-SegNet first-block margin trust region",
        one_line_summary="A true suffix pairwise-logit upper bound turns anchor-correct margins into a sufficient no-worsening feature ball.",
        latex_form=(r"m_p=z_{p,a_p}(h_0)-\max_{c\ne a_p}z_{p,c}(h_0),\quad "
                    r"r_h=\min_{p:\,a_p=y_p}m_p/L_p,\quad \|h-h_0\|_\infty<r_h"),
        python_callable_module_path=("tac.boundary_math.segnet_validation_certificate:derive_feature_trust_region"),
        domain_of_validity={
            "research_only": True,
            "review_status": "fresh-eyes-reviewed(1)-CLEAN",
            "derivation": "triangle inequality on every protected pixel and competing pairwise-logit difference",
            "included": ("fixed frozen SegNet and target labels", "strict feature-ball inequality", "actual suffix pairwise-logit upper bounds"),
            "excluded": ("local Jacobian presented as a neighborhood Lipschitz upper bound", "first-block Jacobian presented as a downstream suffix bound", "empirical proxy described as certified, proof, or provably safe"),
            "empirical_proxy": "Lhat is calibrated on a pre-registered subset and evaluated on disjoint exact-SegNet holdouts; PROXY_ACCEPT is advisory only",
            "cheap_path": "exact conv_stem -> bn1 -> blocks[0] prefix followed by an O(feature-array) trust-region check",
            "prefix_canary": "prefix-only feature must be bitwise equal and hash-equal to the blocks[0] feature captured during a full frozen-SegNet forward",
            "empirical_admission": "an accepted exact worsening in CE, d_seg, or d_pose is an unsafe accept and fails empirical admission",
            "throughput_authority": "inherited YOPO component rows plus current component timings are DERIVED economics only; GO requires a sequence-integrated measured whole-step >=1.3x",
            "fallback_cost": "proxy rejection selects DSL full_teacher_and_refresh, so component economics use rejection_rate times the custody-bound inherited exact-teacher forward/backward time; exact SegNet/Pose forwards are safety evidence only",
            "missing_bound": "without an actual suffix upper bound, no rigorous positive radius is claimed",
            "citation": "YOPO reuse inspiration: Zhang, Zhang, Lu, Zhu, Dong (2019), arXiv:1905.00877; the trust-region derivation imports no external theorem",
            "authority": "DERIVED law plus measured local disposition; score_claim=false; promotion_eligible=false; fresh-eyes-reviewed(1)-CLEAN",
        },
        units_in={"m_p": "pairwise_logit_units", "L_p": "pairwise_logit_units_per_feature_linf", "h": "first_block_feature_units"},
        units_out={"r_h": "first_block_feature_linf"},
        empirical_anchors=(anchor,), predicted_vs_empirical_residual={"joint_unsafe_accepts": 2.0},
        last_calibration_utc="2026-07-13T00:00:00Z", next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=("tac.witness_dsl.segnet_validation_certificate_policy", "tools.probe_segnet_validation_certificate"),
        canonical_producers=("tac.boundary_math.segnet_validation_certificate",),
        provenance=provenance,
    )


def populate_segnet_margin_trust_region_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_segnet_margin_trust_region_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="task454; research_only; fresh-eyes-reviewed(1)-CLEAN; pointer unmoved",
    )
    return equation


__all__ = [
    "SEGNET_MARGIN_TRUST_REGION_EQUATION_ID",
    "build_segnet_margin_trust_region_v1",
    "populate_segnet_margin_trust_region_v1",
]
