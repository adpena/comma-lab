# SPDX-License-Identifier: MIT
"""Empirical-anchor builders for Task #494 authority-ladder receipts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    EmpiricalAnchor,
)
from tac.provenance.contract import (
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
)

EXACT_REDUCTION_EQUATION_ID = "exact_commutative_reduction_reorder_invariance_v1"
ARGMAX_CERTIFICATE_EQUATION_ID = "interval_argmax_enclosure_certificate_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measurement_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _provenance(path: Path, *, repo: Path, mlx: bool) -> Provenance:
    return Provenance(
        artifact_kind=ProvenanceKind.ADVISORY_NON_PROMOTABLE,
        source_path=str(path.resolve().relative_to(repo.resolve())),
        source_sha256=_sha256(path),
        measurement_axis=(
            "[macOS-MLX research-signal]" if mlx else "[macOS-CPU advisory]"
        ),
        hardware_substrate="macos_arm64_mlx" if mlx else "macos_arm64",
        evidence_grade=(
            ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL
            if mlx
            else ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY
        ),
        promotion_eligible=False,
        score_claim_valid=False,
        captured_at_utc=_measurement_utc(path),
        canonical_helper_invocation=(
            "tac.canonical_equations.throughput_authority_anchors_20260714."
            "build_receipt_anchor"
        ),
        rejection_reason=(
            "research-only throughput MEANS; terminal exact archive evaluation remains contest CPU/CUDA"
        ),
    )


def _anchor_id(kind: str, path: Path) -> str:
    return f"task494_{kind}_{_sha256(path)[:16]}"


def _require_qdq(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = payload.get("contract", {})
    scale_mode = contract.get("activation_scale_mode") or (
        "fixed_calibration"
        if payload.get("schema") == "fixedpoint_scorer_forward_n600.v2"
        else None
    )
    expected_schema = {
        "fixed_calibration": "fixedpoint_scorer_forward_n600.v2",
        "dynamic_exact_absmax": "dynamic_fixedpoint_scorer_forward_n600.v1",
    }.get(scale_mode)
    if payload.get("schema") != expected_schema:
        raise ValueError("QDQ receipt schema/activation-scale-mode mismatch")
    summary = payload.get("summary", {})
    custody = summary.get("cache_custody", {})
    if not (
        summary.get("status") == "MEASURED"
        and summary.get("full_real_n600") is True
        and custody.get("status") == "MEASURED"
        and int(custody.get("pairs", -1)) == 600
        and int(custody.get("unique_pair_indices", -1)) == 600
        and custody.get("observed_pair_indices_sha256")
        == custody.get("expected_pair_indices_sha256")
        and payload.get("contract", {}).get("native_integer_speed_claim") is False
    ):
        raise ValueError("QDQ receipt lacks exact 0..599 custody or feasibility-only labeling")
    return summary


def build_qdq_anchor(path: Path, payload: Mapping[str, Any], *, repo: Path) -> EmpiricalAnchor:
    summary = _require_qdq(payload)
    arm_rows: dict[str, Any] = {}
    residual_candidates: list[float] = []
    for name, row in summary.get("arms", {}).items():
        if row.get("status") != "MEASURED" or name == "fp32_control":
            continue
        full = row.get("segnet", {}).get("full", {})
        flip_fraction = float(full.get("aggregate_flip_fraction", 1.0))
        residual_candidates.append(flip_fraction)
        arm_rows[str(name)] = {
            "aggregate_flip_fraction": flip_fraction,
            "worst_pair_flip_fraction": full.get("worst_pair_flip_fraction"),
            "uncertified_pixels": full.get("uncertified_pixels"),
            "argmax_corpus_sha256": full.get("argmax_corpus_sha256"),
            "argmax_exact_admitted": row.get("argmax_exact_admitted"),
        }
    return EmpiricalAnchor(
        anchor_id=_anchor_id("fixedpoint_qdq_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": "frozen_segnet_real_n600_fixedpoint_argmax_certificate",
            "pair_indices": "exact 0..599",
            "calibration_split": payload.get("contract", {}).get("calibration_split"),
            "heldout_split": payload.get("contract", {}).get("heldout_split"),
            "accumulation": payload.get("contract", {}).get("accumulation"),
            "activation_scale_mode": (
                payload.get("contract", {}).get("activation_scale_mode")
                or "fixed_calibration"
            ),
            "native_integer_speed_claim": False,
        },
        predicted_output={
            "load_bearing_target": "zero argmax flips and zero uncertified pixels",
            "certificate_rule": "winner_lower_bound > every_competitor_upper_bound",
        },
        empirical_output={
            "minimum_argmax_exact_arm": summary.get("minimum_argmax_exact_arm"),
            "minimum_training_tolerance_arm": summary.get("minimum_training_tolerance_arm"),
            "rung2_verdict": summary.get("rung2_verdict"),
            "arms": arm_rows,
            "verdict_scope": summary.get("verdict_scope"),
        },
        residual=min(residual_candidates, default=1.0),
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method=(
            "one-thread CPU-Torch fp32 control versus symmetric WnAn QDQ/fp32 "
            "accumulation with "
            f"{payload.get('contract', {}).get('activation_scale_mode') or 'fixed_calibration'} "
            "activation scaling over exact real pair indices 0..599"
        ),
        provenance=_provenance(path, repo=repo, mlx=False),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_full_r_anchor(path: Path, payload: Mapping[str, Any], *, repo: Path) -> EmpiricalAnchor:
    if payload.get("schema") != "pythagorean_exact_arithmetic_full_r_n600.v2":
        raise ValueError("full-R receipt schema mismatch")
    summary = payload.get("summary", {})
    authority = summary.get("authority", {})
    if not (
        summary.get("complete") is True
        and authority.get("coverage_exact") is True
        and int(authority.get("frames", -1)) == 1200
    ):
        raise ValueError("full-R receipt lacks exact 0..599 x {f0,f1} custody")
    integer = summary.get("fixed_q15_int32_atomic", {})
    empirical_holds = bool(
        integer.get("cross_process_identical")
        and integer.get("exact_numpy_int_corpus_parity")
        and authority.get("within_derived_bound")
    )
    return EmpiricalAnchor(
        anchor_id=_anchor_id("full_r_integer_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": "full_render_R_adjoint_four_axes_real_n600",
            "pair_indices": "exact 0..599",
            "frames": 1200,
            "q_weight_bits": payload.get("contract", {}).get("q_weight_bits"),
            "state_bits_by_boundary": payload.get("contract", {}).get(
                "state_bits_by_boundary"
            ),
        },
        predicted_output={
            "integer_cross_process_identical": True,
            "exact_numpy_integer_corpus_parity": True,
            "within_derived_error_bound": True,
        },
        empirical_output={
            "integer": integer,
            "float": summary.get("float_atomic"),
            "authority": authority,
            "overall_verdict": summary.get("overall_verdict"),
            "verdict_scope": summary.get("verdict_scope"),
        },
        residual=0.0 if empirical_holds else 1.0,
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method="N-process Metal full-R VJP corpus digest plus NumPy fp32/int32 authority",
        provenance=_provenance(path, repo=repo, mlx=True),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_metal_segnet_anchor(
    path: Path, payload: Mapping[str, Any], *, repo: Path
) -> EmpiricalAnchor:
    if payload.get("schema") != "metal_fixedpoint_segnet_n600.v1":
        raise ValueError("Metal SegNet receipt schema mismatch")
    summary = payload.get("summary", {})
    if summary.get("complete") is not True:
        raise ValueError("Metal SegNet receipt is incomplete")
    fidelity = summary.get("fidelity") or {}
    return EmpiricalAnchor(
        anchor_id=_anchor_id("metal_fixedpoint_segnet_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": "custom_metal_fixedpoint_segnet_real_n600",
            "bits": payload.get("contract", {}).get("bits"),
            "precision_assignment": payload.get("contract", {}).get(
                "precision_assignment"
            ),
            "activation_scale_mode": payload.get("contract", {}).get(
                "activation_scale_mode"
            ),
            "n_processes": payload.get("contract", {}).get("n_processes"),
            "pair_indices": "exact 0..599",
        },
        predicted_output={
            "argmax_exact": True,
            "strict_interval_certified": True,
            "cross_process_argmax_identical": True,
            "positive_speed": True,
        },
        empirical_output=summary,
        residual=float(fidelity.get("aggregate_flip_fraction", 1.0)),
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method=(
            "custom direct-int64 Metal frozen-SegNet versus one-thread CPU-Torch fp32 control"
        ),
        provenance=_provenance(path, repo=repo, mlx=True),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_exact_int64_segnet_anchor(
    path: Path, payload: Mapping[str, Any], *, repo: Path
) -> EmpiricalAnchor:
    if payload.get("schema") != "exact_int64_fixedpoint_scorer_n600.v1":
        raise ValueError("exact-int64 SegNet receipt schema mismatch")
    summary = payload.get("summary", {})
    custody = summary.get("cache_custody", {})
    manifest = payload.get("model_manifest", {})
    if not (
        summary.get("status") == "MEASURED"
        and summary.get("full_real_n600") is True
        and custody.get("status") == "MEASURED"
        and int(custody.get("pairs", -1)) == 600
        and int(custody.get("unique_pair_indices", -1)) == 600
        and custody.get("observed_pair_indices_sha256")
        == custody.get("expected_pair_indices_sha256")
        and int(manifest.get("converted_conv2d_count", -1)) == 125
        and manifest.get("accumulation") == "exact_signed_int64"
        and payload.get("contract", {}).get("native_integer_speed_claim") is True
    ):
        raise ValueError("exact-int64 SegNet receipt lacks exact 0..599 integer custody")
    full = summary.get("candidate", {}).get("full", {})
    return EmpiricalAnchor(
        anchor_id=_anchor_id("exact_int64_fixedpoint_segnet_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": "exact_int64_fixedpoint_segnet_real_n600",
            "bits": summary.get("bits"),
            "activation_scale_mode": payload.get("contract", {}).get(
                "activation_scale_mode"
            ),
            "pair_indices": "exact 0..599",
            "converted_conv2d_count": manifest.get("converted_conv2d_count"),
            "accumulation": manifest.get("accumulation"),
            "finalization": manifest.get("finalization"),
        },
        predicted_output={
            "argmax_exact": True,
            "within_static_int64_bound": True,
            "strict_interval_certified_fraction": "reported separately",
        },
        empirical_output={
            "argmax_exact_admitted": summary.get("argmax_exact_admitted"),
            "training_tolerance_admitted": summary.get(
                "training_tolerance_admitted"
            ),
            "candidate_full": full,
            "timing": summary.get("timing"),
            "rung2_integer_verdict": summary.get("rung2_integer_verdict"),
            "verdict_scope": summary.get("verdict_scope"),
        },
        residual=float(full.get("aggregate_flip_fraction", 1.0)),
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method=(
            "exact signed-int64 CPU Conv2d twin with signed W26A26 codes and one fp32 "
            "finalization versus one-thread CPU-Torch fp32 over exact real pairs 0..599"
        ),
        provenance=_provenance(path, repo=repo, mlx=False),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_mixed_int64_segnet_anchor(
    path: Path, payload: Mapping[str, Any], *, repo: Path
) -> EmpiricalAnchor:
    if payload.get("schema") != "mixed_int64_fixedpoint_scorer_n600.v1":
        raise ValueError("mixed exact-int64 SegNet receipt schema mismatch")
    summary = payload.get("summary", {})
    custody = summary.get("cache_custody", {})
    manifest = payload.get("model_manifest", {})
    if not (
        summary.get("status") == "MEASURED"
        and summary.get("full_real_n600") is True
        and custody.get("status") == "MEASURED"
        and int(custody.get("pairs", -1)) == 600
        and int(custody.get("unique_pair_indices", -1)) == 600
        and custody.get("observed_pair_indices_sha256")
        == custody.get("expected_pair_indices_sha256")
        and int(manifest.get("minimum_bits", -1)) == 26
        and int(manifest.get("maximum_bits", -1)) == 30
        and int(manifest.get("converted_conv2d_count", -1)) == 125
        and manifest.get("accumulation") == "exact_signed_int64"
        and manifest.get("assignment_rule")
        == "largest_geometry_safe_bits_with_signed_int64_static_bound"
        and payload.get("contract", {}).get("native_integer_speed_claim") is True
    ):
        raise ValueError("mixed exact-int64 SegNet receipt lacks exact 0..599 integer custody")
    full = summary.get("candidate", {}).get("full", {})
    return EmpiricalAnchor(
        anchor_id=_anchor_id("mixed_int64_fixedpoint_segnet_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": "mixed_exact_int64_fixedpoint_segnet_real_n600",
            "minimum_bits": manifest.get("minimum_bits"),
            "maximum_bits": manifest.get("maximum_bits"),
            "assignment_rule": manifest.get("assignment_rule"),
            "precision_histogram": manifest.get("precision_histogram"),
            "pair_indices": "exact 0..599",
            "converted_conv2d_count": manifest.get("converted_conv2d_count"),
            "accumulation": manifest.get("accumulation"),
            "finalization": manifest.get("finalization"),
        },
        predicted_output={
            "argmax_exact": True,
            "every_layer_within_static_int64_bound": True,
            "precision_assignment_is_label_free": True,
        },
        empirical_output={
            "argmax_exact_admitted": summary.get("argmax_exact_admitted"),
            "training_tolerance_admitted": summary.get(
                "training_tolerance_admitted"
            ),
            "candidate_full": full,
            "timing": summary.get("timing"),
            "rung2_mixed_integer_verdict": summary.get(
                "rung2_mixed_integer_verdict"
            ),
            "verdict_scope": summary.get("verdict_scope"),
        },
        residual=float(full.get("aggregate_flip_fraction", 1.0)),
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method=(
            "geometry-only maximum signed W26..W30 precision per Conv2d under a static "
            "int64 bound, exact signed-int64 CPU MAC, and one fp32 finalization versus "
            "one-thread CPU-Torch fp32 over exact real pairs 0..599"
        ),
        provenance=_provenance(path, repo=repo, mlx=False),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_weight_l1_int64_segnet_anchor(
    path: Path, payload: Mapping[str, Any], *, repo: Path
) -> EmpiricalAnchor:
    if payload.get("schema") != "weight_l1_int64_fixedpoint_scorer_n600.v1":
        raise ValueError("weight-L1 exact-int64 SegNet receipt schema mismatch")
    summary = payload.get("summary", {})
    custody = summary.get("cache_custody", {})
    manifest = payload.get("model_manifest", {})
    if not (
        summary.get("status") == "MEASURED"
        and summary.get("full_real_n600") is True
        and custody.get("status") == "MEASURED"
        and int(custody.get("pairs", -1)) == 600
        and int(custody.get("unique_pair_indices", -1)) == 600
        and custody.get("observed_pair_indices_sha256")
        == custody.get("expected_pair_indices_sha256")
        and int(manifest.get("minimum_bits", -1)) == 26
        and int(manifest.get("maximum_bits", -1)) == 31
        and int(manifest.get("converted_conv2d_count", -1)) == 125
        and manifest.get("accumulation") == "exact_signed_int64"
        and manifest.get("assignment_rule")
        == "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
        and manifest.get("bound_kind")
        == "activation_qmax_times_max_output_quantized_weight_l1"
        and manifest.get("label_or_frame_dependent") is False
        and payload.get("contract", {}).get("native_integer_speed_claim") is True
    ):
        raise ValueError("weight-L1 exact-int64 receipt lacks exact 0..599 integer custody")
    full = summary.get("candidate", {}).get("full", {})
    return EmpiricalAnchor(
        anchor_id=_anchor_id("weight_l1_int64_fixedpoint_segnet_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": "weight_l1_exact_int64_fixedpoint_segnet_real_n600",
            "minimum_bits": manifest.get("minimum_bits"),
            "maximum_bits": manifest.get("maximum_bits"),
            "assignment_rule": manifest.get("assignment_rule"),
            "bound_kind": manifest.get("bound_kind"),
            "precision_histogram": manifest.get("precision_histogram"),
            "label_or_frame_dependent": False,
            "pair_indices": "exact 0..599",
            "converted_conv2d_count": manifest.get("converted_conv2d_count"),
            "accumulation": manifest.get("accumulation"),
            "finalization": manifest.get("finalization"),
        },
        predicted_output={
            "argmax_exact": True,
            "every_layer_within_static_int64_bound": True,
            "precision_assignment_is_label_and_frame_free": True,
        },
        empirical_output={
            "argmax_exact_admitted": summary.get("argmax_exact_admitted"),
            "training_tolerance_admitted": summary.get(
                "training_tolerance_admitted"
            ),
            "candidate_full": full,
            "timing": summary.get("timing"),
            "rung2_weight_l1_integer_verdict": summary.get(
                "rung2_weight_l1_integer_verdict"
            ),
            "verdict_scope": summary.get("verdict_scope"),
        },
        residual=float(full.get("aggregate_flip_fraction", 1.0)),
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method=(
            "per-layer maximum W26..W31 under activation_qmax times exact frozen "
            "quantized-weight L1 int64 bounds, exact CPU MAC, and one fp32 finalization "
            "versus one-thread CPU-Torch fp32 over exact real pairs 0..599"
        ),
        provenance=_provenance(path, repo=repo, mlx=False),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_weight_l1_tie_snap_segnet_anchor(
    path: Path, payload: Mapping[str, Any], *, repo: Path
) -> EmpiricalAnchor:
    if payload.get("schema") != "weight_l1_tie_snap_scorer_n600.v1":
        raise ValueError("weight-L1 tie-snap SegNet receipt schema mismatch")
    summary = payload.get("summary", {})
    custody = summary.get("cache_custody", {})
    manifest = payload.get("model_manifest", {})
    contract = payload.get("contract", {})
    if not (
        summary.get("status") == "MEASURED"
        and summary.get("full_real_n600") is True
        and custody.get("status") == "MEASURED"
        and int(custody.get("pairs", -1)) == 600
        and int(custody.get("unique_pair_indices", -1)) == 600
        and custody.get("observed_pair_indices_sha256")
        == custody.get("expected_pair_indices_sha256")
        and int(manifest.get("minimum_bits", -1)) == 26
        and int(manifest.get("maximum_bits", -1)) == 31
        and int(manifest.get("converted_conv2d_count", -1)) == 125
        and manifest.get("accumulation") == "exact_signed_int64"
        and manifest.get("assignment_rule")
        == "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
        and manifest.get("bound_kind")
        == "activation_qmax_times_max_output_quantized_weight_l1"
        and manifest.get("label_or_frame_dependent") is False
        and contract.get("decision_rule")
        == "lowest class index within epsilon of candidate maximum"
        and contract.get("epsilon_selection")
        == "minimum calibration-exact epsilon; no heldout reselection"
        and contract.get("runtime_label_or_frame_dependent") is False
    ):
        raise ValueError("weight-L1 tie-snap receipt lacks exact 0..599 custody")
    selected = summary.get("minimum_calibration_exact_arm")
    selected_rows = (
        summary.get("arms", {}).get(selected, {}) if isinstance(selected, str) else {}
    )
    full = selected_rows.get("full", {})
    residual = float(full.get("aggregate_flip_fraction", 1.0))
    return EmpiricalAnchor(
        anchor_id=_anchor_id("weight_l1_tie_snap_segnet_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": "weight_l1_exact_int64_tie_snap_segnet_real_n600",
            "minimum_bits": manifest.get("minimum_bits"),
            "maximum_bits": manifest.get("maximum_bits"),
            "precision_histogram": manifest.get("precision_histogram"),
            "bound_kind": manifest.get("bound_kind"),
            "epsilon_ladder": contract.get("epsilon_ladder"),
            "epsilon_selection": contract.get("epsilon_selection"),
            "decision_rule": contract.get("decision_rule"),
            "calibration_split": contract.get("calibration_split"),
            "heldout_start": contract.get("heldout_start"),
            "pair_indices": "exact 0..599",
        },
        predicted_output={
            "calibration_exact": True,
            "heldout_exact_without_reselection": True,
            "full_argmax_exact": True,
            "runtime_decision_is_label_and_frame_free": True,
        },
        empirical_output={
            "argmax_exact_admitted": summary.get("argmax_exact_admitted"),
            "selected_arm": selected,
            "selected_epsilon": summary.get("minimum_calibration_exact_epsilon"),
            "selected_calibration": selected_rows.get("calibration"),
            "selected_heldout": selected_rows.get("heldout"),
            "selected_full": full,
            "rung2_tie_snap_verdict": summary.get("rung2_tie_snap_verdict"),
            "verdict_scope": summary.get("verdict_scope"),
        },
        residual=residual,
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method=(
            "preregistered dyadic lowest-class epsilon tie-snap ladder over weight-L1-safe "
            "W27..W31 exact-int64 logits; minimum calibration-exact epsilon selected on "
            "pairs 0..119 and validated without reselection on pairs 120..599"
        ),
        provenance=_provenance(path, repo=repo, mlx=False),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_weight_l1_class_pair_tie_snap_segnet_anchor(
    path: Path, payload: Mapping[str, Any], *, repo: Path
) -> EmpiricalAnchor:
    if payload.get("schema") != "weight_l1_class_pair_tie_snap_scorer_n600.v1":
        raise ValueError("weight-L1 class-pair tie-snap receipt schema mismatch")
    summary = payload.get("summary", {})
    custody = summary.get("cache_custody", {})
    manifest = payload.get("model_manifest", {})
    contract = payload.get("contract", {})
    if not (
        summary.get("status") == "MEASURED"
        and summary.get("full_real_n600") is True
        and custody.get("status") == "MEASURED"
        and int(custody.get("pairs", -1)) == 600
        and int(custody.get("unique_pair_indices", -1)) == 600
        and custody.get("observed_pair_indices_sha256")
        == custody.get("expected_pair_indices_sha256")
        and int(manifest.get("minimum_bits", -1)) == 26
        and int(manifest.get("maximum_bits", -1)) == 31
        and int(manifest.get("converted_conv2d_count", -1)) == 125
        and manifest.get("accumulation") == "exact_signed_int64"
        and manifest.get("assignment_rule")
        == "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
        and manifest.get("bound_kind")
        == "activation_qmax_times_max_output_quantized_weight_l1"
        and manifest.get("label_or_frame_dependent") is False
        and contract.get("design_split") == [0, 264]
        and contract.get("second_validation_split") == [264, 600]
        and contract.get("candidate_winner_class") == 4
        and contract.get("candidate_runner_class") == 0
        and contract.get("replacement_class") == 0
        and float(contract.get("epsilon", -1.0)) == float(2.0**-19)
        and contract.get("rule_frozen_before_second_validation_access") is True
        and contract.get("second_validation_reselection") is False
        and contract.get("runtime_label_or_frame_dependent") is False
    ):
        raise ValueError("class-pair tie-snap receipt lacks honest exact 0..599 custody")
    decisions = summary.get("class_pair_tie_snap", {})
    full = decisions.get("full", {})
    return EmpiricalAnchor(
        anchor_id=_anchor_id("weight_l1_class_pair_tie_snap_segnet_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": (
                "weight_l1_exact_int64_class_pair_tie_snap_segnet_real_n600"
            ),
            "minimum_bits": manifest.get("minimum_bits"),
            "maximum_bits": manifest.get("maximum_bits"),
            "precision_histogram": manifest.get("precision_histogram"),
            "bound_kind": manifest.get("bound_kind"),
            "design_split": contract.get("design_split"),
            "second_validation_split": contract.get("second_validation_split"),
            "epsilon": contract.get("epsilon"),
            "ordered_candidate_top2": [
                contract.get("candidate_winner_class"),
                contract.get("candidate_runner_class"),
            ],
            "replacement_class": contract.get("replacement_class"),
            "pair_indices": "exact 0..599",
        },
        predicted_output={
            "design_exact": True,
            "second_validation_exact_without_reselection": True,
            "full_argmax_exact": True,
            "runtime_decision_is_label_and_frame_free": True,
        },
        empirical_output={
            "argmax_exact_admitted": summary.get("argmax_exact_admitted"),
            "design_exact": summary.get("design_exact"),
            "second_validation_exact": summary.get("second_validation_exact"),
            "design": decisions.get("design"),
            "second_validation": decisions.get("second_validation"),
            "full": full,
            "rung2_class_pair_tie_snap_verdict": summary.get(
                "rung2_class_pair_tie_snap_verdict"
            ),
            "verdict_scope": summary.get("verdict_scope"),
        },
        residual=float(full.get("aggregate_flip_fraction", 1.0)),
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method=(
            "frozen ordered candidate-top2 (4,0), gap <=2^-19 decision head over "
            "weight-L1-safe W27..W31 exact-int64 logits; designed on pairs 0..263 "
            "and validated without reselection on previously untouched pairs 264..599"
        ),
        provenance=_provenance(path, repo=repo, mlx=False),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def build_integer_r_backend_anchor(
    path: Path, payload: Mapping[str, Any], *, repo: Path
) -> EmpiricalAnchor:
    if payload.get("schema") != "integer_r_adjoint_backend_benchmark.v1":
        raise ValueError("integer R backend receipt schema mismatch")
    if payload.get("status") != "MEASURED" or not payload.get("coverage", {}).get(
        "full_real_n600"
    ):
        raise ValueError("integer R backend receipt lacks full real-n600 measurement")
    admitted = payload.get("admission", {}).get("admitted_for_training") is True
    return EmpiricalAnchor(
        anchor_id=_anchor_id("integer_r_backend_n600", path),
        measurement_utc=_measurement_utc(path),
        inputs={
            "in_domain_context": "integer_order_independent_render_R_adjoint_backend",
            "pair_indices": "exact 0..599",
            "frames": payload.get("coverage", {}).get("frames"),
        },
        predicted_output={
            "within_bound": True,
            "repeat_bit_identical": True,
            "positive_speed": True,
        },
        empirical_output={
            "parity": payload.get("parity"),
            "timing": payload.get("timing"),
            "admission": payload.get("admission"),
        },
        residual=0.0 if admitted else 1.0,
        source_artifact=str(path.resolve().relative_to(repo.resolve())),
        measurement_method="matched custom-Metal integer versus fixed-order float R-adjoint benchmark",
        provenance=_provenance(path, repo=repo, mlx=True),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


__all__ = [
    "ARGMAX_CERTIFICATE_EQUATION_ID",
    "EXACT_REDUCTION_EQUATION_ID",
    "build_exact_int64_segnet_anchor",
    "build_full_r_anchor",
    "build_integer_r_backend_anchor",
    "build_metal_segnet_anchor",
    "build_mixed_int64_segnet_anchor",
    "build_qdq_anchor",
    "build_weight_l1_class_pair_tie_snap_segnet_anchor",
    "build_weight_l1_int64_segnet_anchor",
    "build_weight_l1_tie_snap_segnet_anchor",
]
