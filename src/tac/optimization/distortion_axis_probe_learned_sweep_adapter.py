# SPDX-License-Identifier: MIT
"""Adapt distortion-axis probe verdicts into learned-sweep candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.optimization.mlx_dynamic_learned_sweep import (
    FALSE_AUTHORITY,
    QUALITY_EVIDENCE_SCHEMA,
)
from tac.optimization.normalized_objective import RATE_SCORE_PER_BYTE
from tac.optimization.proxy_candidate_contract import (
    require_no_truthy_authority_fields,
)
from tac.repo_io import write_json_artifact

SCHEMA = "distortion_axis_probe_learned_sweep_candidates.v1"
ROW_SCHEMA = "distortion_axis_probe_learned_sweep_candidate.v1"
SUPPRESSED_ROW_SCHEMA = "distortion_axis_probe_suppressed_candidate.v1"
REFUSAL_SCHEMA = "distortion_axis_probe_learned_sweep_refusal.v1"
TOOL = "tac.optimization.distortion_axis_probe_learned_sweep_adapter"
CLI_TOOL = "tools/adapt_distortion_axis_probes_to_learned_sweep.py"
DEFAULT_VARIANCE_FLOOR = 2.5e-10
DEFAULT_PREDICTED_DELTA_BAND = (-0.025, -0.010)
SEGMENT_THRESHOLD = 0.5
MOTION_PEARSON_THRESHOLD = 0.6
MOTION_AMPLIFICATION_THRESHOLD = 1.5
MACOS_CPU_ADVISORY_AXIS = "[macOS-CPU advisory]"


class DistortionAxisProbeLearnedSweepAdapterError(ValueError):
    """Raised when distortion probe evidence is unsafe for learned-sweep intake."""


def dumps_json(payload: Mapping[str, Any]) -> str:
    """Return deterministic JSON text for adapter artifacts."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
) -> None:
    """Write deterministic JSON through the guarded artifact writer."""

    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_sha256,
    )


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object with adapter-specific validation errors."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DistortionAxisProbeLearnedSweepAdapterError(
            f"{path}: expected JSON object"
        )
    return payload


def file_sha256(path: Path) -> str:
    """Return SHA-256 for a source artifact consumed by this adapter."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_artifact_metadata(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Build deterministic source metadata for verdict manifests."""

    return {
        f"verdict_{index:03d}": {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for index, path in enumerate(paths)
    }


def build_refusal_payload(
    *,
    error: str,
    source_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a durable fail-closed refusal artifact for operator audit."""

    return {
        "schema": REFUSAL_SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "status": "refused",
        "error": str(error),
        "source_artifacts": dict(source_artifacts or {}),
        "allowed_use": "fail_closed_distortion_probe_adapter_refusal",
    }


def build_distortion_axis_probe_learned_sweep_candidates(
    verdicts: Sequence[Mapping[str, Any]],
    *,
    incumbent_score: float,
    top_k: int | None = None,
    source_artifacts: Mapping[str, Any] | None = None,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
) -> dict[str, Any]:
    """Adapt local distortion probe verdicts into planning-only sweep candidates."""

    incumbent = _finite_float(incumbent_score, label="incumbent_score")
    if incumbent <= 0.0:
        raise DistortionAxisProbeLearnedSweepAdapterError(
            "incumbent_score must be positive"
        )
    floor = _finite_float(variance_floor, label="variance_floor")
    if floor < 0.0:
        raise DistortionAxisProbeLearnedSweepAdapterError(
            "variance_floor must be non-negative"
        )
    if top_k is not None and int(top_k) <= 0:
        raise DistortionAxisProbeLearnedSweepAdapterError(
            "top_k must be positive when provided"
        )
    if not verdicts:
        raise DistortionAxisProbeLearnedSweepAdapterError(
            "at least one distortion probe verdict is required"
        )

    candidates: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    advisory_only: list[dict[str, Any]] = []
    for index, verdict in enumerate(verdicts):
        if not isinstance(verdict, Mapping):
            raise DistortionAxisProbeLearnedSweepAdapterError(
                f"verdict {index} must be an object"
            )
        _require_no_truthy_authority(verdict, label=f"verdict {index}")
        if _is_positive_uniward_threshold_break(verdict):
            candidates.append(
                _candidate_from_uniward_threshold_break(
                    verdict,
                    index=index,
                    incumbent_score=incumbent,
                    variance_floor=floor,
                )
            )
        elif _is_motion_neutral(verdict):
            suppressed.append(_suppressed_motion_candidate(verdict, index=index))
        else:
            advisory_only.append(_advisory_verdict(verdict, index=index))

    candidates.sort(
        key=lambda row: (
            float(row["predicted_score_mean"]),
            -float(row["master_gradient_priority"]),
            str(row["candidate_id"]),
        )
    )
    if top_k is not None:
        candidates = candidates[: int(top_k)]
    if not candidates:
        raise DistortionAxisProbeLearnedSweepAdapterError(
            "no positive threshold-breaking distortion candidates found"
        )

    best = candidates[0]
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "quality_evidence_schema": QUALITY_EVIDENCE_SCHEMA,
        "allowed_use": (
            "learned_sweep_candidate_intake_from_distortion_axis_probe_verdicts"
        ),
        "evidence_grade": "macOS-CPU-advisory",
        "evidence_tag": MACOS_CPU_ADVISORY_AXIS,
        "source_artifacts": dict(source_artifacts or {}),
        "execution_bridge_status": (
            "planning_payload_only_selection_adapter_required_before_local_actuation"
        ),
        "summary": {
            "verdict_count": len(verdicts),
            "adapted_candidate_count": len(candidates),
            "suppressed_candidate_count": len(suppressed),
            "advisory_only_count": len(advisory_only),
            "top_k": top_k,
            "incumbent_score": incumbent,
            "variance_floor": floor,
            "best_predicted_score_mean": best["predicted_score_mean"],
            "best_predicted_delta_vs_incumbent": (
                best["predicted_score_mean"] - incumbent
            ),
            "best_non_authoritative_repair_budget_score": best[
                "non_authoritative_normalized_full_video_gain_sum"
            ],
            "best_non_authoritative_repair_budget_bytes_equivalent": best[
                "component_axis_context"
            ]["non_authoritative_rate_budget_bytes_equivalent"],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "candidates": candidates,
        "suppressed_candidates": suppressed,
        "advisory_only_verdicts": advisory_only,
    }


def _candidate_from_uniward_threshold_break(
    verdict: Mapping[str, Any],
    *,
    index: int,
    incumbent_score: float,
    variance_floor: float,
) -> dict[str, Any]:
    actual = _actual_signature(verdict)
    segment_min = _finite_float(
        actual.get(
            "min_segment_textured_avg_weight_combined",
            actual.get("min_segment_textured_avg_weight"),
        ),
        label="min_segment_textured_avg_weight",
    )
    segment_spread = _optional_float(
        actual.get(
            "spread_segment_textured_avg_weight_combined",
            actual.get("spread_segment_textured_avg_weight"),
        )
    )
    valid_segment_count = _optional_int(actual.get("valid_segment_count"))
    pair_indices = _pair_indices(actual)
    delta_band = _predicted_delta_band(verdict)
    conservative_delta = max(delta_band)
    aggressive_delta = min(delta_band)
    conservative_gain = abs(conservative_delta)
    aggressive_gain = abs(aggressive_delta)
    predicted_score = incumbent_score + conservative_delta
    sigma = max((aggressive_gain - conservative_gain) / 2.0, 0.0)
    variance = max(variance_floor, sigma * sigma)
    budget_bytes = conservative_gain / RATE_SCORE_PER_BYTE
    normalized_gain = conservative_gain
    window_gain = normalized_gain * float(CONTEST_EXACT_SAMPLE_COUNT)
    candidate_id = "distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1"
    quality_evidence = {
        "schema": QUALITY_EVIDENCE_SCHEMA,
        **FALSE_AUTHORITY,
        "source_schema": SCHEMA,
        "source_probe_id": str(verdict.get("probe_id") or ""),
        "adapter_tool": CLI_TOOL,
        "calibrated_score_mean": predicted_score,
        "calibrated_gain_mean": normalized_gain,
        "calibrated_delta_mean": conservative_delta,
        "calibration_scope": (
            "macOS_CPU_advisory_distortion_threshold_planning_only_not_auth_score"
        ),
        "predicted_delta_band": list(delta_band),
        "gate_statuses": {
            "calibration": "strict_pass",
            "parity": "strict_pass",
            "production_contract": "strict_pass",
            "effective_spend_triage": "strict_pass",
        },
        "gate_basis": {
            "calibration": "probe_verdict_breaks_predeclared_segment_threshold",
            "parity": "source_verdict_loaded_with_no_truthy_authority_fields",
            "production_contract": (
                "candidate_payload_has_false_authority_and_explicit_context"
            ),
            "effective_spend_triage": (
                "non_authoritative_rate_budget_equivalent_is_positive"
            ),
        },
    }
    return {
        "schema": ROW_SCHEMA,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "candidate_id": candidate_id,
        "source_probe_index": index,
        "source_probe_id": str(verdict.get("probe_id") or ""),
        "family": "distortion_axis_segnet_uniward",
        "rank_score_field": "distortion_axis_predicted_score_mean",
        "predicted_score_mean": predicted_score,
        "predicted_score_variance": variance,
        "predicted_score_source": (
            "distortion_axis_probe9_threshold_break_conservative_predicted_delta"
        ),
        "predicted_score_scope": (
            "macOS_CPU_advisory_full_video_planning_projection_not_auth_score"
        ),
        "predicted_delta_vs_incumbent_score": conservative_delta,
        "predicted_delta_band_vs_incumbent_score": list(delta_band),
        "payload_bytes": 0,
        "selected_pair_indices": pair_indices,
        "selected_pair_count": max(1, len(pair_indices)),
        "pair_indices": pair_indices,
        "non_authoritative_mlx_gain_sum": normalized_gain,
        "non_authoritative_normalized_full_video_gain_sum": normalized_gain,
        "non_authoritative_mlx_window_gain_sum": window_gain,
        "full_video_denominator": CONTEST_EXACT_SAMPLE_COUNT,
        "orthogonality_score": 0.75,
        "master_gradient_priority": _master_gradient_priority(segment_min),
        "component_axis_context": {
            "schema": "distortion_axis_probe_component_axis_context.v1",
            **FALSE_AUTHORITY,
            "primary_axis": "segnet",
            "secondary_axis": "posenet_repair_budget_unmeasured",
            "rate_axis": "non_authoritative_free_budget_equivalent",
            "source_evidence_grade": "macOS-CPU-advisory",
            "non_authoritative_repair_budget_score": normalized_gain,
            "non_authoritative_rate_budget_bytes_equivalent": budget_bytes,
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "posenet_repair_requires_exact_component_probe": True,
        },
        "segnet_context": {
            "schema": "distortion_axis_probe_segnet_context.v1",
            **FALSE_AUTHORITY,
            "segment_min_textured_avg_weight": segment_min,
            "segment_spread_textured_avg_weight": segment_spread,
            "segment_threshold": SEGMENT_THRESHOLD,
            "threshold_broken": True,
            "valid_segment_count": valid_segment_count,
            "wavelet_name": str(actual.get("wavelet_name") or ""),
            "wavelet_levels": _optional_int(actual.get("wavelet_levels")),
            "per_class_segment_count": actual.get("per_class_segment_count"),
            "lowest_segments": _lowest_segments(actual),
        },
        "posenet_context": {
            "schema": "distortion_axis_probe_posenet_context.v1",
            **FALSE_AUTHORITY,
            "status": "repair_budget_candidate_not_measured",
            "required_followup": (
                "paired SegNet/PoseNet component probe before spending saved rate budget"
            ),
        },
        "frame_pair_context": {
            "schema": "distortion_axis_probe_frame_pair_context.v1",
            **FALSE_AUTHORITY,
            "pair_count": _optional_int(actual.get("pair_count")),
            "n_frames_decoded": _optional_int(actual.get("n_frames_decoded")),
            "pair_indices": pair_indices,
            "granularity": ["region", "boundary", "frame", "pair", "full_video"],
        },
        "waterbucket_context": {
            "schema": "distortion_axis_probe_waterbucket_context.v1",
            **FALSE_AUTHORITY,
            "normalized_gain": normalized_gain,
            "projected_delta": conservative_delta,
            "break_even_added_bytes": budget_bytes,
            "added_archive_bytes": 0,
            "byte_budget_margin": budget_bytes,
            "allocation_hint": (
                "treat score-equivalent bytes as repair budget for SegNet/PoseNet"
            ),
        },
        "rate_distortion_lattice_context": {
            "schema": "distortion_axis_probe_rate_distortion_lattice_context.v1",
            **FALSE_AUTHORITY,
            "covered_levels": [
                "bit",
                "byte",
                "pixel",
                "region",
                "boundary",
                "frame",
                "pair",
                "batch",
                "full_video",
            ],
            "bit_byte_signal": "rate_budget_equivalent_only_no_payload_change",
            "pixel_region_boundary_signal": (
                "per_instance_multi_scale_textured_segment_threshold_break"
            ),
            "frame_pair_signal": "probe_source_pairs_and_segments",
            "batch_full_video_signal": "conservative_predicted_full_video_gain",
        },
        "canonical_equation_provenance": {
            "schema": "distortion_axis_probe_canonical_equation_provenance.v1",
            **FALSE_AUTHORITY,
            "canonical_equation_candidate": str(
                verdict.get("sister_canonical_equation_candidate_for_RATIFY_N")
                or verdict.get("canonical_equation_reference")
                or "uniward_per_instance_multi_scale_wavelet_combined_v1"
            ),
            "catalog_status": "FORMALIZATION_PENDING",
        },
        "consumer_payload": {
            "schema": "distortion_axis_probe_local_sweep_spec.v1",
            **FALSE_AUTHORITY,
            "local_actuator_status": "selection_adapter_required",
            "loss_family": "uniward_per_instance_multi_scale_wavelet_segnet",
            "receiver_runtime_status": "training_loss_surface_not_inflate_runtime",
        },
        "quality_evidence": quality_evidence,
        "dispatch_blockers": [
            "distortion_probe_is_macos_cpu_advisory_not_auth_score",
            "byte_closed_archive_required_before_promotion",
            "exact_auth_eval_required_before_score_claim",
            "local_selection_adapter_required_before_run_mlx_dynamic_learned_sweep_local",
        ],
    }


def _suppressed_motion_candidate(
    verdict: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    actual = _actual_signature(verdict)
    return {
        "schema": SUPPRESSED_ROW_SCHEMA,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "source_probe_index": index,
        "source_probe_id": str(verdict.get("probe_id") or ""),
        "candidate_id": "distortion_axis:hinton_kl_motion_aware_temporal_context_w6_v1",
        "family": "distortion_axis_motion_weighted_hinton_kl",
        "suppression_reason": "NEGATIVE_MOTION_NEUTRAL",
        "motion_weighted_uniform_ratio": _optional_float(
            actual.get("motion_amplification_ratio")
        ),
        "motion_kl_pearson": _optional_float(actual.get("motion_kl_pearson")),
        "preserved_signal": (
            "Probe 7 W=6 temporal Hinton KL remains eligible; only the "
            "motion-weighted enhancement is suppressed."
        ),
        "dispatch_blockers": [
            "motion_weighted_uniform_ratio_failed_threshold",
            "motion_kl_pearson_failed_threshold",
            "do_not_spend_budget_on_motion_weighting_without_new_probe",
        ],
    }


def _advisory_verdict(
    verdict: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    return {
        "schema": "distortion_axis_probe_advisory_verdict.v1",
        **FALSE_AUTHORITY,
        "source_probe_index": index,
        "source_probe_id": str(verdict.get("probe_id") or ""),
        "verdict": str(verdict.get("verdict") or ""),
        "reason": "verdict_preserved_as_advisory_signal_not_learned_sweep_candidate",
    }


def _is_positive_uniward_threshold_break(verdict: Mapping[str, Any]) -> bool:
    actual = _actual_signature(verdict)
    verdict_text = str(verdict.get("verdict") or "")
    probe_id = str(verdict.get("probe_id") or "").lower()
    segment_min = _optional_float(
        actual.get(
            "min_segment_textured_avg_weight_combined",
            actual.get("min_segment_textured_avg_weight"),
        )
    )
    return (
        verdict_text.startswith("POSITIVE_SIGNAL")
        and "uniward" in probe_id
        and (
            actual.get("any_segment_below_threshold") is True
            or (segment_min is not None and segment_min < SEGMENT_THRESHOLD)
        )
    )


def _is_motion_neutral(verdict: Mapping[str, Any]) -> bool:
    actual = _actual_signature(verdict)
    verdict_text = str(verdict.get("verdict") or "")
    ratio = _optional_float(actual.get("motion_amplification_ratio"))
    pearson = _optional_float(actual.get("motion_kl_pearson"))
    return verdict_text == "NEGATIVE_MOTION_NEUTRAL" or (
        ratio is not None
        and ratio < MOTION_AMPLIFICATION_THRESHOLD
        and pearson is not None
        and pearson < MOTION_PEARSON_THRESHOLD
    )


def _actual_signature(verdict: Mapping[str, Any]) -> Mapping[str, Any]:
    actual = verdict.get("actual_signature")
    if not isinstance(actual, Mapping):
        raise DistortionAxisProbeLearnedSweepAdapterError(
            f"{verdict.get('probe_id') or 'verdict'} actual_signature must be an object"
        )
    return actual


def _predicted_delta_band(verdict: Mapping[str, Any]) -> tuple[float, float]:
    explicit = verdict.get("predicted_delta_band")
    if isinstance(explicit, Sequence) and not isinstance(explicit, str | bytes):
        values = [
            _finite_float(item, label="predicted_delta_band")
            for item in explicit[:2]
        ]
        if len(values) == 2 and all(value < 0.0 for value in values):
            return (min(values), max(values))
    text = " ".join(
        str(verdict.get(key) or "")
        for key in (
            "recommendation",
            "next_action_on_POSITIVE",
            "next_action_on_POSITIVE_SIGNAL",
            "next_action_on_POSITIVE_BREAKS_THRESHOLD",
        )
    )
    matches = re.findall(r"(-0\.\d+)\s+to\s+(-0\.\d+)", text)
    if matches:
        values = [float(matches[-1][0]), float(matches[-1][1])]
        return (min(values), max(values))
    return DEFAULT_PREDICTED_DELTA_BAND


def _pair_indices(actual: Mapping[str, Any]) -> list[int]:
    indices: set[int] = set()
    metrics = actual.get("per_segment_metrics")
    if isinstance(metrics, Sequence) and not isinstance(metrics, str | bytes):
        for row in metrics:
            if isinstance(row, Mapping):
                parsed = _optional_int(row.get("pair_index"))
                if parsed is not None:
                    indices.add(parsed)
    pair_count = _optional_int(actual.get("pair_count"))
    if not indices and pair_count is not None and pair_count > 0:
        indices.update(range(pair_count))
    return sorted(indices)


def _lowest_segments(actual: Mapping[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    metrics = actual.get("per_segment_metrics")
    if not isinstance(metrics, Sequence) or isinstance(metrics, str | bytes):
        return []
    rows: list[dict[str, Any]] = []
    for row in metrics:
        if not isinstance(row, Mapping):
            continue
        weight = _optional_float(
            row.get(
                "instance_textured_avg_weight_combined",
                row.get("segment_textured_avg_weight"),
            )
        )
        if weight is None:
            continue
        rows.append(
            {
                "pair_index": _optional_int(row.get("pair_index")),
                "class_index": _optional_int(row.get("class_index")),
                "instance_id": _optional_int(row.get("instance_id")),
                "instance_pixel_count": _optional_int(row.get("instance_pixel_count")),
                "instance_textured_count": _optional_int(
                    row.get("instance_textured_count")
                ),
                "instance_textured_avg_weight": weight,
            }
        )
    rows.sort(key=lambda item: float(item["instance_textured_avg_weight"]))
    return rows[:limit]


def _master_gradient_priority(segment_min: float) -> float:
    # Below-threshold segments are the useful inverse-steg signal. Clamp so one
    # unusually tiny local probe cannot dominate all later planner geometry.
    return max(0.0, min(1.0, (SEGMENT_THRESHOLD - segment_min) / SEGMENT_THRESHOLD))


def _require_no_truthy_authority(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise DistortionAxisProbeLearnedSweepAdapterError(str(exc)) from exc


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise DistortionAxisProbeLearnedSweepAdapterError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DistortionAxisProbeLearnedSweepAdapterError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise DistortionAxisProbeLearnedSweepAdapterError(f"{label} must be finite")
    return result


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


__all__ = [
    "CLI_TOOL",
    "DEFAULT_PREDICTED_DELTA_BAND",
    "DEFAULT_VARIANCE_FLOOR",
    "REFUSAL_SCHEMA",
    "ROW_SCHEMA",
    "SCHEMA",
    "SUPPRESSED_ROW_SCHEMA",
    "TOOL",
    "DistortionAxisProbeLearnedSweepAdapterError",
    "build_distortion_axis_probe_learned_sweep_candidates",
    "build_refusal_payload",
    "dumps_json",
    "file_sha256",
    "load_json_object",
    "source_artifact_metadata",
    "write_json",
]
