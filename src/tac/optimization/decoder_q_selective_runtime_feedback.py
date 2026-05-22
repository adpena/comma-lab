# SPDX-License-Identifier: MIT
"""Feedback accounting for DQS1 selective decoder-q runtime probes."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "decoder_q_selective_runtime_feedback.v1"
SIGN_SCHEMA = "decoder_q_surface_sign_calibration_labels.v1"
TOOL = "tac.optimization.decoder_q_selective_runtime_feedback"
PACKET_SCHEMA = "decoder_q_selective_runtime_packet_plan.v1"
LOCAL_ADVISORY_SCORE_AXES = frozenset({"cpu_advisory"})
LOCAL_ADVISORY_EVIDENCE_GRADES = frozenset({"macOS-CPU advisory"})

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class DecoderQSelectiveRuntimeFeedbackError(ValueError):
    """Raised when selective runtime feedback would overstate authority."""


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DecoderQSelectiveRuntimeFeedbackError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload), encoding="utf-8")


def _require_false_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise DecoderQSelectiveRuntimeFeedbackError(f"{label} must keep {key}=false")


def _reject_true_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is True:
            raise DecoderQSelectiveRuntimeFeedbackError(f"{label} must not set {key}=true")


def _require_local_advisory_input(advisory: dict[str, Any]) -> None:
    """Fail closed if an exact auth payload is routed into local calibration."""

    axis = advisory.get("score_axis")
    if axis not in LOCAL_ADVISORY_SCORE_AXES:
        raise DecoderQSelectiveRuntimeFeedbackError(
            f"advisory score_axis must be one of {sorted(LOCAL_ADVISORY_SCORE_AXES)}"
        )
    evidence_grade = advisory.get("evidence_grade")
    if evidence_grade not in LOCAL_ADVISORY_EVIDENCE_GRADES:
        raise DecoderQSelectiveRuntimeFeedbackError(
            "advisory evidence_grade must be macOS-CPU advisory"
        )
    required_false = (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "rank_or_kill_eligible",
    )
    for key in required_false:
        if advisory.get(key) is not False:
            raise DecoderQSelectiveRuntimeFeedbackError(f"advisory result must keep {key}=false")
    for key in ("ready_for_exact_eval_dispatch", "promotable"):
        if key in advisory and advisory.get(key) is not False:
            raise DecoderQSelectiveRuntimeFeedbackError(f"advisory result must keep {key}=false")


def _as_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQSelectiveRuntimeFeedbackError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise DecoderQSelectiveRuntimeFeedbackError(f"{label} must be finite")
    return result


def _as_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise DecoderQSelectiveRuntimeFeedbackError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQSelectiveRuntimeFeedbackError(f"{label} must be an integer") from exc
    if result != value and not (isinstance(value, str) and str(result) == value):
        raise DecoderQSelectiveRuntimeFeedbackError(f"{label} must be integral")
    return result


def _mutation_key(
    materialization_manifest: dict[str, Any],
    packet_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if packet_plan is None:
        packet_plan = materialization_manifest.get("packet_plan")
    if isinstance(packet_plan, dict):
        mutation = packet_plan.get("mutation")
        if isinstance(mutation, dict):
            return {
                "tensor_name": str(mutation.get("tensor_name")),
                "q_offset": _as_int(mutation.get("q_offset"), label="mutation q_offset"),
                "delta": _as_int(mutation.get("delta"), label="mutation delta"),
            }
    packet_path = materialization_manifest.get("packet_plan_path")
    if packet_path is None:
        packet_path = materialization_manifest.get("plan_path")
    raise DecoderQSelectiveRuntimeFeedbackError(
        "materialization manifest does not embed packet_plan.mutation; "
        f"load packet plan separately before feedback emission ({packet_path!r})"
    )


def _selected_pairs_from_manifest(manifest: dict[str, Any]) -> list[int]:
    dqs1 = manifest.get("dqs1_payload")
    if not isinstance(dqs1, dict):
        raise DecoderQSelectiveRuntimeFeedbackError("materialization dqs1_payload missing")
    pairs = dqs1.get("pair_indices")
    if not isinstance(pairs, list):
        raise DecoderQSelectiveRuntimeFeedbackError("materialization DQS1 pair_indices missing")
    return [_as_int(pair, label="DQS1 pair index") for pair in pairs]


def _selected_frames_from_locality(locality_controls: dict[str, Any]) -> list[int]:
    frames = locality_controls.get("selected_frame_indices")
    if not isinstance(frames, list):
        raise DecoderQSelectiveRuntimeFeedbackError("locality selected_frame_indices missing")
    return [_as_int(frame, label="locality selected frame") for frame in frames]


def _selected_pairs_from_locality(locality_controls: dict[str, Any]) -> list[int]:
    pairs = locality_controls.get("selected_pair_indices")
    if not isinstance(pairs, list):
        raise DecoderQSelectiveRuntimeFeedbackError("locality selected_pair_indices missing")
    return [_as_int(pair, label="locality selected pair") for pair in pairs]


def _affected_frames_from_manifest(manifest: dict[str, Any]) -> list[int]:
    dqs1 = manifest.get("dqs1_payload")
    if not isinstance(dqs1, dict):
        raise DecoderQSelectiveRuntimeFeedbackError("materialization dqs1_payload missing")
    frames = dqs1.get("affected_frame_indices")
    if not isinstance(frames, list):
        raise DecoderQSelectiveRuntimeFeedbackError(
            "materialization DQS1 affected_frame_indices missing"
        )
    return [_as_int(frame, label="DQS1 affected frame") for frame in frames]


def _sum_observed_mlx_gain(bridge_plan: dict[str, Any], selected_pairs: list[int]) -> float:
    selected_set = set(selected_pairs)
    total = 0.0
    for index, unit in enumerate(bridge_plan.get("work_units", [])):
        if not isinstance(unit, dict):
            continue
        window = unit.get("pair_window")
        if not isinstance(window, list) or not window:
            continue
        start = _as_int(window[0], label=f"work unit {index} pair_window[0]")
        if start in selected_set:
            total += _as_float(
                unit.get("observed_mlx_gain", 0.0),
                label=f"work unit {index} observed_mlx_gain",
            )
    return total


def _archive_sha_from_advisory(advisory: dict[str, Any]) -> str:
    provenance = advisory.get("provenance")
    if not isinstance(provenance, dict):
        raise DecoderQSelectiveRuntimeFeedbackError("advisory provenance missing")
    archive_sha = provenance.get("archive_sha256")
    if not isinstance(archive_sha, str) or not archive_sha:
        raise DecoderQSelectiveRuntimeFeedbackError("advisory provenance archive_sha256 missing")
    return archive_sha


def _raw_sha_from_advisory(advisory: dict[str, Any]) -> str | None:
    provenance = advisory.get("provenance")
    manifest = provenance.get("inflated_output_manifest") if isinstance(provenance, dict) else None
    payload = manifest.get("payload") if isinstance(manifest, dict) else None
    files = payload.get("files") if isinstance(payload, dict) else None
    if isinstance(files, list) and files and isinstance(files[0], dict):
        value = files[0].get("sha256")
        return value if isinstance(value, str) else None
    return None


def load_packet_plan_for_materialization(
    materialization_manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Load the packet plan that corresponds to a materialization manifest."""

    embedded = materialization_manifest.get("packet_plan")
    if isinstance(embedded, dict):
        if embedded.get("schema") != PACKET_SCHEMA:
            raise DecoderQSelectiveRuntimeFeedbackError("embedded packet plan schema mismatch")
        return embedded

    candidates: list[Path] = []
    for key in ("packet_plan_path", "plan_path"):
        raw_path = materialization_manifest.get(key)
        if isinstance(raw_path, str) and raw_path:
            candidates.append(Path(raw_path))
    output_dir = materialization_manifest.get("output_submission_dir")
    if isinstance(output_dir, str) and output_dir:
        candidates.append(Path(output_dir) / "decoder_q_selective_runtime_packet_plan.json")
    if manifest_path is not None:
        candidates.append(
            manifest_path.parent / "submission_dir" / "decoder_q_selective_runtime_packet_plan.json"
        )

    seen: set[Path] = set()
    rejected: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            rejected.append(f"{candidate}:missing")
            continue
        payload = load_json_object(candidate)
        if payload.get("schema") == PACKET_SCHEMA:
            return payload
        rejected.append(f"{candidate}:schema={payload.get('schema')!r}")
    raise DecoderQSelectiveRuntimeFeedbackError(
        "materialization packet plan could not be loaded; candidates="
        + ", ".join(rejected)
    )


def build_decoder_q_selective_runtime_feedback(
    *,
    bridge_plan: dict[str, Any],
    materialization_manifest: dict[str, Any],
    locality_controls: dict[str, Any],
    advisory_result: dict[str, Any],
    packet_plan: dict[str, Any] | None = None,
    input_paths: dict[str, str] | None = None,
    local_baseline_score: float,
    min_dispatch_edge: float,
    contest_cpu_frontier_score: float | None = None,
) -> dict[str, Any]:
    """Build fail-closed feedback from a selective DQS1 runtime probe."""

    if bridge_plan.get("schema") != "decoder_q_selective_window_bridge_plan.v1":
        raise DecoderQSelectiveRuntimeFeedbackError("bridge plan schema mismatch")
    if materialization_manifest.get("schema") != "decoder_q_selective_runtime_materialization.v1":
        raise DecoderQSelectiveRuntimeFeedbackError("materialization schema mismatch")
    if packet_plan is not None and packet_plan.get("schema") != PACKET_SCHEMA:
        raise DecoderQSelectiveRuntimeFeedbackError("packet plan schema mismatch")
    if locality_controls.get("schema") != "decoder_q_selective_runtime_locality_controls.v1":
        raise DecoderQSelectiveRuntimeFeedbackError("locality controls schema mismatch")
    _require_false_authority(bridge_plan, label="bridge plan")
    _require_false_authority(materialization_manifest, label="materialization manifest")
    _require_false_authority(locality_controls, label="locality controls")
    _reject_true_authority(advisory_result, label="advisory result")
    _require_local_advisory_input(advisory_result)
    if locality_controls.get("locality_controls_passed") is not True:
        raise DecoderQSelectiveRuntimeFeedbackError("locality controls must pass before feedback")

    materialized = materialization_manifest.get("materialized_archive")
    if not isinstance(materialized, dict):
        raise DecoderQSelectiveRuntimeFeedbackError("materialized archive metadata missing")
    dqs1 = materialization_manifest.get("dqs1_payload")
    if not isinstance(dqs1, dict):
        raise DecoderQSelectiveRuntimeFeedbackError("DQS1 payload metadata missing")
    selected_pairs = _selected_pairs_from_manifest(materialization_manifest)
    locality_pairs = _selected_pairs_from_locality(locality_controls)
    selected_frames = _selected_frames_from_locality(locality_controls)
    manifest_frames = _affected_frames_from_manifest(materialization_manifest)
    if locality_pairs != selected_pairs:
        raise DecoderQSelectiveRuntimeFeedbackError(
            "locality selected_pair_indices mismatch materialization DQS1 pair_indices"
        )
    if selected_frames != manifest_frames:
        raise DecoderQSelectiveRuntimeFeedbackError(
            "locality selected_frame_indices mismatch materialization affected_frame_indices"
        )
    archive_sha = str(materialized.get("zip_sha256"))
    if _archive_sha_from_advisory(advisory_result) != archive_sha:
        raise DecoderQSelectiveRuntimeFeedbackError("advisory archive SHA mismatch")
    locality_selective = locality_controls.get("targets", {}).get("selective")
    if isinstance(locality_selective, dict) and locality_selective.get("archive_sha256") != archive_sha:
        raise DecoderQSelectiveRuntimeFeedbackError("locality selective archive SHA mismatch")
    raw_sha = _raw_sha_from_advisory(advisory_result)
    if raw_sha is None:
        raise DecoderQSelectiveRuntimeFeedbackError("advisory raw SHA missing")
    locality_hashes = locality_controls.get("hashes", {})
    raw_file_hashes = locality_hashes.get("0.raw", {}).get("raw_files") if isinstance(locality_hashes, dict) else None
    if not isinstance(raw_file_hashes, dict) or raw_file_hashes.get("selective") != raw_sha:
        raise DecoderQSelectiveRuntimeFeedbackError("advisory raw SHA mismatch")
    locality_raw_sha = str(raw_file_hashes.get("selective"))

    advisory_score = _as_float(advisory_result.get("canonical_score"), label="advisory score")
    local_baseline_score = _as_float(local_baseline_score, label="local baseline score")
    min_dispatch_edge = _as_float(min_dispatch_edge, label="min dispatch edge")
    score_delta = advisory_score - local_baseline_score
    advisory_improvement = local_baseline_score - advisory_score
    observed_mlx_gain_sum = _sum_observed_mlx_gain(bridge_plan, selected_pairs)
    rate_delta = 25.0 * _as_int(dqs1.get("payload_bytes"), label="DQS1 payload bytes") / 37_545_489
    net_mlx_gain_after_rate = observed_mlx_gain_sum - rate_delta
    transfer_efficiency = (
        advisory_improvement / net_mlx_gain_after_rate
        if net_mlx_gain_after_rate > 0.0
        else None
    )
    naive_contest_delta = None
    if contest_cpu_frontier_score is not None:
        naive_contest_delta = advisory_score - _as_float(
            contest_cpu_frontier_score,
            label="contest CPU frontier score",
        )
    local_spend_triage_positive = (
        advisory_improvement >= min_dispatch_edge
        and (naive_contest_delta is None or naive_contest_delta < 0.0)
    )
    recommended_next_action = "append_local_sign_calibration_label_keep_exact_dispatch_decision_external"

    sign = 0
    if score_delta > 0:
        sign = 1
    elif score_delta < 0:
        sign = -1
    label = {
        "schema": "decoder_q_surface_sign_calibration_label.v1",
        "source_kind": "dqs1_selective_runtime_advisory",
        "archive_sha256": archive_sha,
        "selected_pair_indices": selected_pairs,
        "selected_frame_indices": selected_frames,
        "observed_score_delta": score_delta,
        "observed_score_delta_sign": sign,
        "advisory_score": advisory_score,
        "local_baseline_score": local_baseline_score,
        "score_axis": advisory_result.get("score_axis"),
        "evidence_grade": advisory_result.get("evidence_grade"),
        "atom_mutation_keys": [_mutation_key(materialization_manifest, packet_plan)],
        **FALSE_AUTHORITY,
    }

    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": TOOL,
        "evidence_grade": "macOS-CPU advisory plus inflate-locality-control",
        "archive_sha256": archive_sha,
        "archive_bytes": _as_int(materialized.get("zip_bytes"), label="archive bytes"),
        "dqs1_payload_bytes": _as_int(dqs1.get("payload_bytes"), label="DQS1 payload bytes"),
        "selected_pair_count": len(selected_pairs),
        "selected_frame_count": len(selected_frames),
        "locality_controls_passed": True,
        "mismatch_counts": locality_controls.get("mismatch_counts"),
        "custody": {
            "input_paths": input_paths or {},
            "materialized_archive_sha256": archive_sha,
            "materialized_archive_bytes": _as_int(materialized.get("zip_bytes"), label="archive bytes"),
            "materialized_member_sha256": materialized.get("member_sha256"),
            "materialized_member_bytes": materialized.get("member_bytes"),
            "dqs1_payload_sha256": dqs1.get("payload_sha256"),
            "dqs1_pair_encoding": dqs1.get("pair_encoding"),
            "advisory_archive_sha256": _archive_sha_from_advisory(advisory_result),
            "advisory_raw_sha256": raw_sha,
            "locality_selective_archive_sha256": (
                locality_selective.get("archive_sha256")
                if isinstance(locality_selective, dict)
                else None
            ),
            "locality_selective_raw_sha256": locality_raw_sha,
            "advisory_score_axis": advisory_result.get("score_axis"),
            "advisory_evidence_grade": advisory_result.get("evidence_grade"),
        },
        "advisory": {
            "canonical_score": advisory_score,
            "local_baseline_score": local_baseline_score,
            "score_delta_vs_local_baseline": score_delta,
            "score_improvement_vs_local_baseline": advisory_improvement,
            "min_dispatch_edge": min_dispatch_edge,
            "naive_delta_vs_contest_cpu_frontier": naive_contest_delta,
            "score_axis": advisory_result.get("score_axis"),
            "evidence_grade": advisory_result.get("evidence_grade"),
        },
        "mlx_transfer": {
            "observed_mlx_gain_sum": observed_mlx_gain_sum,
            "rate_delta_from_payload_bytes": rate_delta,
            "net_mlx_gain_after_rate": net_mlx_gain_after_rate,
            "advisory_to_mlx_transfer_efficiency": transfer_efficiency,
        },
        "decision": {
            "dispatch_recommended": False,
            "local_spend_triage_positive": local_spend_triage_positive,
            "exact_dispatch_suppression_allowed": False,
            "recommended_next_action": recommended_next_action,
            "reason": "local macOS advisory is calibration signal only and cannot rank, kill, promote, or suppress exact replay",
        },
        "sign_calibration_label": label,
        **FALSE_AUTHORITY,
    }


def build_sign_calibration_labels(
    feedback_rows: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    labels = []
    for feedback in feedback_rows:
        if feedback.get("schema") != SCHEMA:
            raise DecoderQSelectiveRuntimeFeedbackError("feedback schema mismatch")
        _require_false_authority(feedback, label="feedback")
        label = feedback.get("sign_calibration_label")
        if not isinstance(label, dict):
            raise DecoderQSelectiveRuntimeFeedbackError("feedback sign_calibration_label missing")
        labels.append(label)
    return {
        "schema": SIGN_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": TOOL,
        "source": source,
        "allowed_use": "local_decoder_q_sign_calibration_only",
        "summary": {
            "label_count": len(labels),
            "improvement_label_count": sum(
                1 for label in labels if int(label.get("observed_score_delta_sign") or 0) < 0
            ),
            "regression_label_count": sum(
                1 for label in labels if int(label.get("observed_score_delta_sign") or 0) > 0
            ),
        },
        "labels": labels,
        **FALSE_AUTHORITY,
    }


def render_decoder_q_selective_runtime_feedback_markdown(feedback: dict[str, Any]) -> str:
    advisory = feedback.get("advisory", {})
    mlx = feedback.get("mlx_transfer", {})
    decision = feedback.get("decision", {})
    lines = [
        "# Decoder-Q Selective Runtime Feedback",
        "",
        f"- Archive SHA-256: `{feedback.get('archive_sha256')}`",
        f"- Selected pairs: `{feedback.get('selected_pair_count')}`",
        f"- Selected frames: `{feedback.get('selected_frame_count')}`",
        f"- DQS1 payload bytes: `{feedback.get('dqs1_payload_bytes')}`",
        "- Advisory score "
        f"`[{advisory.get('evidence_grade')}; {advisory.get('score_axis')}; non-authoritative]`: "
        f"`{advisory.get('canonical_score')}`",
        "- Delta vs local baseline "
        f"`[{advisory.get('evidence_grade')}; non-authoritative]`: "
        f"`{advisory.get('score_delta_vs_local_baseline')}`",
        f"- MLX gain sum: `{mlx.get('observed_mlx_gain_sum')}`",
        f"- Net MLX gain after rate: `{mlx.get('net_mlx_gain_after_rate')}`",
        f"- Advisory/MLX transfer efficiency: `{mlx.get('advisory_to_mlx_transfer_efficiency')}`",
        f"- Exact dispatch recommended by this local advisory artifact: `{decision.get('dispatch_recommended')}`",
        f"- Exact dispatch suppression allowed: `{decision.get('exact_dispatch_suppression_allowed')}`",
        f"- Local spend-triage positive: `{decision.get('local_spend_triage_positive')}`",
        f"- Next action: `{decision.get('recommended_next_action')}`",
        "",
        "Authority: `score_claim=false`, `promotion_eligible=false`, "
        "`ready_for_exact_eval_dispatch=false`. This report is local calibration "
        "signal only and cannot rank, kill, promote, or suppress exact replay.",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "FALSE_AUTHORITY",
    "SCHEMA",
    "SIGN_SCHEMA",
    "TOOL",
    "DecoderQSelectiveRuntimeFeedbackError",
    "build_decoder_q_selective_runtime_feedback",
    "build_sign_calibration_labels",
    "dumps_json",
    "load_json_object",
    "load_packet_plan_for_materialization",
    "render_decoder_q_selective_runtime_feedback_markdown",
    "write_json",
]
