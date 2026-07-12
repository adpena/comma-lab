# SPDX-License-Identifier: MIT
"""Canonical law for exact component-safe coefficient-prefix selection.

The law is an offline receiver/rate oracle.  It does not authorize a trainer
flag or a score claim: every candidate plane is rendered through the canonical
receiver and R, both frozen-scorer components are compared independently, and
ZIP bytes are counted on the exact reconstructed archive.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "jrd_exact_coefficient_prefix_selection_v1"

_UTC = "2026-07-12T22:39:09Z"
_MEMO = ".omx/research/codex_findings_jrd_coeff_prefix_20260712T224139Z_codex.md"
_RECEIPT = (
    "experiments/results/jrd_coeff_prefix_probe_20260712T221747Z/"
    "measurement_receipt.json"
)


def build_jrd_exact_coefficient_prefix_selection_v1() -> CanonicalEquation:
    """Build the exact-Pareto prefix-selection law and its scoped fixture anchor."""

    anchor = EmpiricalAnchor(
        anchor_id="jrd_prefix_v752_pair0_exact_r_20260712",
        measurement_utc=_UTC,
        inputs={
            "fixture": (
                "single frozen v7.5.2 checkpoint staged for V9 apply-pass; pair 0; "
                "macOS CPU advisory"
            ),
            "enumeration": "18 sections x 2 families x 8 nonzero int8 prefix planes",
            "families": ["uniform", "laplace_dead_zone"],
            "component_tolerances": {"d_seg": 0.0, "d_pose": 0.0},
            "receiver": "canonical NumPy-fp32 receiver plus exact R",
            "source_artifact": _RECEIPT,
        },
        predicted_output={
            "falsifier": (
                "NO-GO if no exact combined replay both shrinks the ZIP and remains "
                "componentwise non-worse than the sealed baseline"
            )
        },
        empirical_output={
            "response_rows": 288,
            "sealed_sections": 18,
            "sealed_coefficients": 71_223,
            "baseline_archive_bytes": 83_905,
            "selected_archive_bytes": 81_154,
            "archive_bytes_saved": 2_751,
            "raw_precision_bits_removed": 40_416,
            "baseline_d_seg": 0.023157755533854168,
            "selected_d_seg": 0.0218505859375,
            "delta_d_seg": -0.0013071695963541678,
            "baseline_d_pose": 116.59830629690003,
            "selected_d_pose": 92.42743674059255,
            "delta_d_pose": -24.17086955630748,
            "accepted_combined_steps": 5,
            "rejected_combined_steps": 2,
            "fixture_verdict": "GO",
            "task_verdict": "NEEDS-MORE",
            "verdict_scope": (
                "INSTANCE: v7.5.2 pair-0 fixture only; transfer to V9/v8, other pairs, "
                "early/boundary/late saved regimes, contest CPU, and contest CUDA is UNKNOWN"
            ),
        },
        residual=0.0,
        source_artifact=_RECEIPT,
        measurement_method=(
            "complete deterministic prefix enumeration plus exact combined receiver/R/scorer "
            "replay and exact ZIP byte accounting"
        ),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_RECEIPT,
            reactivation_criteria=(
                "rerun on one sealed non-live typed V9/v8 payload, then require early, "
                "boundary, and late saved-regime replay before any family verdict"
            ),
            measurement_axis="[macOS-CPU advisory] NON-PROMOTABLE",
            hardware_substrate="macos_arm64_cpu",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Exact component-safe last-plane coefficient-prefix selection",
        one_line_summary=(
            "Admit a coefficient prefix only when exact receiver/R replay is Seg- and "
            "Pose-nonworse and exact ZIP bytes strictly decrease."
        ),
        latex_form=(
            r"A_{s,f,k}=\mathbf 1[d_{seg}(s,f,k)\le d^0_{seg}+\epsilon_{seg}]"
            r"\mathbf 1[d_{pose}(s,f,k)\le d^0_{pose}+\epsilon_{pose}],\quad "
            r"k^*_{s,f}=\max\{k:A_{s,f,k}=1\};\quad "
            r"\operatorname{accept}(q)=A(q)\wedge B(q)<B_{current}"
        ),
        python_callable_module_path=(
            "tac.packet_compiler.jrd_coefficient_prefix:select_best_byte_safe"
        ),
        domain_of_validity={
            "included": (
                "content-addressed sealed coefficient payload",
                "complete nested uniform and analytic Laplace dead-zone prefix enumeration",
                "exact receiver, R, frozen SegNet, frozen PoseNet, and ZIP byte replay",
            ),
            "excluded": (
                "proxy losses, logits, argmax-only agreement, or weighted-score compensation",
                "path-name inference used as V9/v8 payload custody",
                "score or promotion claims without upstream/evaluate.py on exact archive bytes",
            ),
            "control_law": (
                "event-conditioned tested predicate: componentwise non-worse at the repeat-"
                "measured noise floor and strict exact-ZIP shrink; enumerate every plane"
            ),
            "named_recess_measurement": "controls/baseline_repeat.json",
            "noise_floor": {"d_seg": 0.0, "d_pose": 0.0},
            "measurement_axis": "macOS-CPU advisory",
            "promotion_eligible": False,
            "score_claim": False,
            "review_status": (
                "recovery-written-UNREVIEWED at construction; post-write review provenance, "
                "when present, is recorded in experiments/results/"
                "jrd_coeff_prefix_probe_20260712T221747Z/adversarial_review_receipt.json"
            ),
            "dsl_leg": (
                "N/A-with-reason: offline receiver/byte-allocator oracle over a frozen "
                "payload; no trainer, curriculum, launch, or actuator configuration changes"
            ),
        },
        units_in={
            "coefficient_prefix_plane": "removed_low_int8_bits",
            "d_seg": "frozen_SegNet_exact_R_distortion",
            "d_pose": "frozen_PoseNet_exact_R_distortion",
            "archive_bytes": "exact_ZIP_bytes",
        },
        units_out={
            "last_safe_plane": "removed_low_int8_bits",
            "archive_bytes_saved": "exact_ZIP_bytes",
            "admitted": "boolean",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"fixture_gate_rederivation": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_jrd_coefficient_prefix",
            "allocator_planning_input.json",
            _MEMO,
        ),
        canonical_producers=(
            "tac.packet_compiler.jrd_coefficient_prefix",
            "tools.probe_jrd_coefficient_prefix",
            _RECEIPT,
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "a sealed non-live typed V9/v8 coefficient payload plus content-addressed "
                "early, boundary, and late saved regimes becomes available"
            ),
            measurement_axis="[macOS-CPU advisory] NON-PROMOTABLE",
            hardware_substrate="macos_arm64_cpu",
        ),
    )


def populate_jrd_exact_coefficient_prefix_selection_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append the equation through the canonical registry writer."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_jrd_exact_coefficient_prefix_selection_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "JRD pair-0 fixture GO: 2751 exact ZIP bytes removed with zero-tolerance "
            "component guard; V9/v8 task NEEDS-MORE; DSL N/A-with-reason"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_jrd_exact_coefficient_prefix_selection_v1",
    "populate_jrd_exact_coefficient_prefix_selection_v1",
]
