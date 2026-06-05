# SPDX-License-Identifier: MIT
"""Queue-owned HiNeRV distortion-stabilization launch gate."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

SCHEMA = "hinerv_distortion_stabilization_queue.v1"
ROW_SCHEMA = "hinerv_distortion_stabilization_dag_node.v1"
DEFAULT_LANE_ID = "lane_hinerv_distortion_stabilization_20260605"
DEFAULT_QUEUE_ID = "hinerv_distortion_stabilization_queue.v1"
DEFAULT_MIN_FREE_BYTES = 1_000_000_000
SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

QUEUE_FALSE_AUTHORITY = {
    **FALSE_AUTHORITY,
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_allowed": False,
    "exact_cpu_cuda_dispatch_allowed": False,
    "local_cpu_replay_allowed": False,
}

_DYNAMIC_RANGE_BLOCKER_FRAGMENTS = (
    "dynamic_range_too_low",
    "scorer_input_out_of_distribution",
    "fit_scale_gate_failed",
    "far_from_reference_fit_gate",
    "argmax_collapsed",
    "argmax_disagreement_too_high",
    "cache_quality_gate_failed",
    "segnet_last_rgb_std_ratio_lt_0_25",
)
_ARCHIVE_IN_LOOP_BLOCKERS = (
    "hi_nerv_archive_in_loop_byte_oracle_missing",
    "hinerv_trained_archive_byte_oracle_feedback_missing",
    "hinerv_candidate_curriculum_recon_pixel_weight_missing",
    "hi_nerv_pr95_staged_curriculum_missing",
)


class HinervDistortionStabilizationQueueError(ValueError):
    """Raised when the HiNeRV distortion-stabilization queue cannot be built."""


def build_hinerv_distortion_stabilization_queue(
    *,
    candidate_feedback_rows: Sequence[Mapping[str, Any]] = (),
    checkpoint_export_reports: Sequence[Mapping[str, Any]] = (),
    waterfill_reports: Sequence[Mapping[str, Any]] = (),
    replay_actuator_reports: Sequence[Mapping[str, Any]] = (),
    output_root: str | Path,
    lane_id: str = DEFAULT_LANE_ID,
    queue_id: str = DEFAULT_QUEUE_ID,
    generated_utc: str | None = None,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    allow_local_output: bool = False,
) -> dict[str, Any]:
    """Build a false-authority DAG gate for HiNeRV distortion work."""

    if not str(lane_id).strip():
        raise HinervDistortionStabilizationQueueError("lane_id must be non-empty")
    if not str(queue_id).strip():
        raise HinervDistortionStabilizationQueueError("queue_id must be non-empty")
    root = Path(output_root)
    storage_preflight = _storage_preflight(
        root,
        min_free_bytes=int(min_free_bytes),
        allow_local_output=bool(allow_local_output),
    )
    generated = generated_utc or datetime.now(UTC).isoformat()
    feedback_state = _feedback_state(candidate_feedback_rows)
    export_state = _checkpoint_export_state(checkpoint_export_reports)
    dynamic_state = _dynamic_range_state(feedback_state, export_state)
    ema_state = _ema_archive_in_loop_state(feedback_state, export_state)
    waterfill_state = _waterfill_state(waterfill_reports, replay_actuator_reports)
    full_video_state = _full_video_prefilter_state(feedback_state, export_state)
    local_replay_state = _local_replay_state(feedback_state, full_video_state)
    nodes = _dag_nodes(
        dynamic_state=dynamic_state,
        export_state=export_state,
        ema_state=ema_state,
        waterfill_state=waterfill_state,
        full_video_state=full_video_state,
        local_replay_state=local_replay_state,
    )
    blocked_nodes = [row for row in nodes if row["blocked"]]
    blockers = _dedupe(
        [
            "hinerv_distortion_stabilization_queue_false_authority",
            *[blocker for row in nodes for blocker in row.get("blockers", ())],
        ]
    )
    return {
        "schema": SCHEMA,
        "queue_id": str(queue_id),
        "lane_id": str(lane_id),
        "generated_utc": generated,
        "axis_tag": "[planning/control:false-authority]",
        "family": "hi_nerv",
        "queue_kind": "distortion_stabilization_launch_gate_not_training_queue",
        "allowed_use": (
            "DAG gating for dynamic-range/scorer-input stabilization, "
            "byte-closed export, receiver proof, archive-in-loop selection, "
            "waterfill proof, full-video MLX prefilter, and local replay"
        ),
        "forbidden_use": (
            "score claim, promotion, rank/kill, local CPU replay launch, or "
            "exact CPU/CUDA dispatch before all prerequisite nodes pass"
        ),
        "storage_preflight": storage_preflight,
        "feedback_evidence": feedback_state,
        "checkpoint_export_evidence": export_state,
        "dynamic_range_evidence": dynamic_state,
        "ema_archive_in_loop_evidence": ema_state,
        "waterfill_recon_pixel_evidence": waterfill_state,
        "full_video_mlx_prefilter_evidence": full_video_state,
        "local_cpu_replay_evidence": local_replay_state,
        "dag_nodes": nodes,
        "dag_node_count": len(nodes),
        "blocked_dag_node_count": len(blocked_nodes),
        "ready_dag_node_count": len(nodes) - len(blocked_nodes),
        "blocking_dag_node_ids": [row["node_id"] for row in blocked_nodes],
        "blockers": blockers,
        **QUEUE_FALSE_AUTHORITY,
    }


def render_hinerv_distortion_stabilization_queue_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing HiNeRV DAG gate."""

    export = report.get("checkpoint_export_evidence")
    export = export if isinstance(export, Mapping) else {}
    dynamic = report.get("dynamic_range_evidence")
    dynamic = dynamic if isinstance(dynamic, Mapping) else {}
    lines = [
        "# HiNeRV Distortion Stabilization Queue",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- lane: `{report.get('lane_id')}`",
        f"- archive bytes: `{export.get('archive_bytes')}`",
        f"- receiver proof ready: `{export.get('receiver_proof_ready')}`",
        f"- dynamic-range/scorer-input stable: `{dynamic.get('dynamic_range_stabilized')}`",
        f"- blocked DAG nodes: `{report.get('blocked_dag_node_count')}`",
        "",
        "## DAG Nodes",
    ]
    for row in report.get("dag_nodes") or ():
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                "",
                f"### `{row.get('node_id')}`",
                f"- status: `{row.get('status')}`",
                f"- blocked: `{row.get('blocked')}`",
                f"- depends on: `{', '.join(row.get('depends_on') or [])}`",
                "- blockers:",
            ]
        )
        blockers = [str(v) for v in row.get("blockers") or ()]
        lines.extend(f"  - `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def _feedback_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("schema") == "nerv_candidate_feedback_row.v1"
        and row.get("family") == "hi_nerv"
    ]
    if not valid:
        return {
            "schema": "hinerv_distortion_feedback_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "source_path": None,
            "source_sha256": None,
            "candidate_id": None,
            "blockers": ["hinerv_candidate_feedback_row_missing"],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        valid,
        key=lambda row: (
            str(row.get("created_utc") or row.get("generated_utc") or ""),
            str(row.get("_source_path") or row.get("source_report_path") or ""),
        ),
    )
    blockers = _string_list(selected.get("blockers"))
    return {
        "schema": "hinerv_distortion_feedback_evidence.v1",
        "artifact_count": len(valid),
        "selected_artifact_schema": selected.get("schema"),
        "selected_artifact_created_utc": selected.get("created_utc"),
        "source_path": selected.get("_source_path") or selected.get("source_report_path"),
        "source_sha256": selected.get("_source_sha256") or selected.get("source_report_sha256"),
        "candidate_id": selected.get("candidate_id"),
        "measured_num_pairs": _nonnegative_int(selected.get("measured_num_pairs")),
        "measured_archive_bytes": _positive_int(
            selected.get("measured_archive_bytes") or selected.get("archive_bytes")
        ),
        "mlx_prefilter_has_full_video": (
            selected.get("mlx_prefilter_has_full_video") is True
        ),
        "local_cpu_replay_gate_has_full_video_mlx_prefilter": (
            selected.get("local_cpu_replay_gate_has_full_video_mlx_prefilter") is True
        ),
        "local_cpu_replay_gate_local_replay_mlx_prefilter_passed": (
            selected.get("local_cpu_replay_gate_local_replay_mlx_prefilter_passed")
            is True
        ),
        "mlx_prefilter_blockers": _string_list(selected.get("mlx_prefilter_blockers")),
        "blockers": blockers,
        **QUEUE_FALSE_AUTHORITY,
    }


def _checkpoint_export_state(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [
        report
        for report in reports
        if isinstance(report, Mapping)
        and report.get("schema") == "hinerv_checkpoint_archive_export.v1"
    ]
    if not valid:
        return {
            "schema": "hinerv_distortion_checkpoint_export_evidence.v1",
            "artifact_count": 0,
            "selected_artifact_schema": None,
            "source_path": None,
            "source_sha256": None,
            "byte_closed_archive_export": False,
            "receiver_proof_ready": False,
            "local_mlx_prefilter_written": False,
            "checkpoint_state_kind": None,
            "blockers": ["hinerv_checkpoint_archive_export_missing"],
            **QUEUE_FALSE_AUTHORITY,
        }
    selected = max(
        valid,
        key=lambda report: (
            int(report.get("checkpoint_epoch") or -1),
            str(report.get("_source_path") or report.get("report_path") or ""),
        ),
    )
    archive_bytes = _positive_int(selected.get("archive_bytes"))
    archive_sha = str(selected.get("archive_sha256") or "")
    archive_path = str(selected.get("archive_path") or "")
    byte_closed = bool(archive_bytes and len(archive_sha) == 64 and archive_path)
    fit_guard = selected.get("receiver_fit_scale_guard")
    fit_guard = fit_guard if isinstance(fit_guard, Mapping) else {}
    prefilter = selected.get("local_mlx_prefilter_profile")
    prefilter = prefilter if isinstance(prefilter, Mapping) else {}
    cache_quality = prefilter.get("cache_quality_gate")
    cache_quality = cache_quality if isinstance(cache_quality, Mapping) else {}
    blockers = []
    if not byte_closed:
        blockers.append("hinerv_byte_closed_archive_export_missing")
    if selected.get("receiver_proof_ready") is not True:
        blockers.append("hi_nerv_receiver_proof_missing")
    blockers.extend(_string_list(selected.get("blockers")))
    return {
        "schema": "hinerv_distortion_checkpoint_export_evidence.v1",
        "artifact_count": len(valid),
        "selected_artifact_schema": selected.get("schema"),
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "source_sha256": selected.get("_source_sha256"),
        "candidate_id": selected.get("candidate_id"),
        "checkpoint_epoch": selected.get("checkpoint_epoch"),
        "checkpoint_state_kind": selected.get("checkpoint_state_kind"),
        "checkpoint_state_path": selected.get("checkpoint_state_path"),
        "report_status": selected.get("report_status"),
        "archive_path": archive_path or None,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha or None,
        "byte_closed_archive_export": byte_closed,
        "receiver_closed": selected.get("receiver_closed") is True,
        "receiver_contract_satisfied": selected.get("receiver_contract_satisfied") is True,
        "receiver_proof_ready": selected.get("receiver_proof_ready") is True,
        "receiver_proof_path": selected.get("receiver_proof_path"),
        "receiver_proof_sha256": selected.get("receiver_proof_sha256"),
        "local_mlx_prefilter_written": selected.get("local_mlx_prefilter_written") is True,
        "local_mlx_prefilter_profile_path": selected.get("local_mlx_prefilter_profile_path")
        or prefilter.get("profile_path"),
        "receiver_fit_scale_guard_passed": fit_guard.get("gate_passed") is True,
        "receiver_fit_scale_guard_blockers": _string_list(fit_guard.get("blockers")),
        "cache_quality_gate_passed": cache_quality.get("fit_gate_passed") is True,
        "cache_quality_gate_verdict": cache_quality.get("verdict"),
        "cache_quality_gate_blockers": _string_list(cache_quality.get("blockers")),
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _dynamic_range_state(
    feedback: Mapping[str, Any],
    export: Mapping[str, Any],
) -> dict[str, Any]:
    source_blockers = [
        *_string_list(feedback.get("blockers")),
        *_string_list(feedback.get("mlx_prefilter_blockers")),
        *_string_list(export.get("receiver_fit_scale_guard_blockers")),
        *_string_list(export.get("cache_quality_gate_blockers")),
    ]
    dynamic_blockers = [
        blocker
        for blocker in source_blockers
        if any(fragment in blocker for fragment in _DYNAMIC_RANGE_BLOCKER_FRAGMENTS)
    ]
    if export.get("receiver_fit_scale_guard_passed") is False:
        dynamic_blockers.append("hinerv_checkpoint_fit_scale_gate_failed")
    if export.get("cache_quality_gate_passed") is False:
        dynamic_blockers.append("hinerv_receiver_cache_quality_gate_failed")
    if feedback.get("schema") and feedback.get("artifact_count") == 0:
        dynamic_blockers.append("hinerv_candidate_feedback_row_missing")
    return {
        "schema": "hinerv_dynamic_range_scorer_input_evidence.v1",
        "dynamic_range_stabilized": not dynamic_blockers,
        "blockers": _dedupe(dynamic_blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _ema_archive_in_loop_state(
    feedback: Mapping[str, Any],
    export: Mapping[str, Any],
) -> dict[str, Any]:
    feedback_blockers = _string_list(feedback.get("blockers"))
    blockers = [
        blocker
        for blocker in feedback_blockers
        if blocker in _ARCHIVE_IN_LOOP_BLOCKERS
    ]
    if export.get("checkpoint_state_kind") != "ema":
        blockers.append("hinerv_ema_checkpoint_export_missing")
    if "hi_nerv_archive_in_loop_byte_oracle_missing" in feedback_blockers:
        blockers.append("hi_nerv_archive_in_loop_byte_oracle_missing")
    return {
        "schema": "hinerv_ema_archive_in_loop_evidence.v1",
        "ema_checkpoint_export_present": export.get("checkpoint_state_kind") == "ema",
        "archive_in_loop_byte_oracle_present": (
            "hi_nerv_archive_in_loop_byte_oracle_missing" not in feedback_blockers
        ),
        "ema_archive_in_loop_selection_ready": not blockers,
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _waterfill_state(
    waterfill_reports: Sequence[Mapping[str, Any]],
    replay_actuator_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    waterfills = [
        report
        for report in waterfill_reports
        if isinstance(report, Mapping)
        and report.get("schema") == "hinerv_archive_ladder_waterfill.v1"
    ]
    replays = [
        report
        for report in replay_actuator_reports
        if isinstance(report, Mapping)
        and report.get("schema") == "hinerv_archive_ladder_replay_actuator.v1"
    ]
    selected = waterfills[-1] if waterfills else {}
    replay = replays[-1] if replays else {}
    receiver_rows = int(replay.get("receiver_proof_ready_row_count") or 0)
    blockers = []
    if not waterfills:
        blockers.append("hinerv_decoder_weight_waterfill_plan_missing")
    else:
        blockers.extend(_string_list(selected.get("blockers")))
    if not replays:
        blockers.append("hinerv_decoder_weight_waterfill_receiver_replay_missing")
    elif receiver_rows < 1:
        blockers.append("hinerv_decoder_weight_waterfill_receiver_proof_not_ready")
    if not _has_recon_pixel_proof(selected):
        blockers.append("hinerv_recon_pixel_weight_proof_missing")
    return {
        "schema": "hinerv_waterfill_recon_pixel_evidence.v1",
        "waterfill_report_count": len(waterfills),
        "replay_actuator_report_count": len(replays),
        "waterfill_ready": bool(waterfills and replays and receiver_rows >= 1)
        and _has_recon_pixel_proof(selected)
        and not blockers,
        "receiver_proof_ready_row_count": receiver_rows,
        "source_path": selected.get("_source_path") or selected.get("report_path"),
        "replay_source_path": replay.get("_source_path") or replay.get("source_report_path"),
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _full_video_prefilter_state(
    feedback: Mapping[str, Any],
    export: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = []
    if export.get("local_mlx_prefilter_written") is not True:
        blockers.append("hi_nerv_full_video_local_prefilter_missing")
    if feedback.get("mlx_prefilter_has_full_video") is not True:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
    blockers.extend(_string_list(feedback.get("mlx_prefilter_blockers")))
    blockers.extend(_string_list(export.get("cache_quality_gate_blockers")))
    return {
        "schema": "hinerv_full_video_mlx_prefilter_evidence.v1",
        "full_video_mlx_prefilter_ready": not blockers,
        "local_mlx_prefilter_written": export.get("local_mlx_prefilter_written") is True,
        "feedback_has_full_video": feedback.get("mlx_prefilter_has_full_video") is True,
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _local_replay_state(
    feedback: Mapping[str, Any],
    full_video: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = []
    if full_video.get("full_video_mlx_prefilter_ready") is not True:
        blockers.append("hi_nerv_full_video_local_prefilter_missing")
    if feedback.get("local_cpu_replay_gate_has_full_video_mlx_prefilter") is not True:
        blockers.append("hi_nerv_local_cpu_replay_gate_missing")
    if feedback.get("local_cpu_replay_gate_local_replay_mlx_prefilter_passed") is not True:
        blockers.append("hi_nerv_local_cpu_replay_not_passed")
    return {
        "schema": "hinerv_local_cpu_replay_evidence.v1",
        "local_cpu_replay_ready": not blockers,
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _dag_nodes(
    *,
    dynamic_state: Mapping[str, Any],
    export_state: Mapping[str, Any],
    ema_state: Mapping[str, Any],
    waterfill_state: Mapping[str, Any],
    full_video_state: Mapping[str, Any],
    local_replay_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _node(
            "dynamic_range_scorer_input_stabilization",
            [],
            dynamic_state.get("dynamic_range_stabilized") is True,
            dynamic_state.get("blockers") or (),
        ),
        _node(
            "byte_closed_archive_export",
            ["dynamic_range_scorer_input_stabilization"],
            export_state.get("byte_closed_archive_export") is True,
            _export_node_blockers(export_state),
        ),
        _node(
            "receiver_archive_replay_proof",
            ["byte_closed_archive_export"],
            export_state.get("receiver_proof_ready") is True,
            [] if export_state.get("receiver_proof_ready") is True else ["hi_nerv_receiver_proof_missing"],
        ),
        _node(
            "ema_archive_in_loop_selection",
            ["receiver_archive_replay_proof"],
            ema_state.get("ema_archive_in_loop_selection_ready") is True,
            ema_state.get("blockers") or (),
        ),
        _node(
            "decoder_weight_waterfill_recon_pixel_proof",
            ["ema_archive_in_loop_selection"],
            waterfill_state.get("waterfill_ready") is True,
            waterfill_state.get("blockers") or (),
        ),
        _node(
            "full_video_mlx_prefilter_gate",
            [
                "dynamic_range_scorer_input_stabilization",
                "decoder_weight_waterfill_recon_pixel_proof",
            ],
            full_video_state.get("full_video_mlx_prefilter_ready") is True,
            full_video_state.get("blockers") or (),
        ),
        _node(
            "local_cpu_replay_gate",
            ["full_video_mlx_prefilter_gate"],
            local_replay_state.get("local_cpu_replay_ready") is True,
            local_replay_state.get("blockers") or (),
        ),
        _node(
            "exact_cpu_cuda_dispatch_gate",
            ["local_cpu_replay_gate"],
            False,
            ["contest_cpu_cuda_exact_eval_blocked_until_local_replay_wins"],
        ),
    ]


def _node(
    node_id: str,
    depends_on: Sequence[str],
    ready: bool,
    blockers: Sequence[Any],
) -> dict[str, Any]:
    clean_blockers = _dedupe(blockers)
    blocked = bool(clean_blockers) or not bool(ready)
    if not clean_blockers and blocked:
        clean_blockers = [f"{node_id}_not_proven"]
    return {
        "schema": ROW_SCHEMA,
        "node_id": node_id,
        "depends_on": list(depends_on),
        "status": "ready_no_authority" if not blocked else "blocked_until_prerequisites",
        "blocked": blocked,
        "blockers": clean_blockers,
        **QUEUE_FALSE_AUTHORITY,
    }


def _export_node_blockers(export: Mapping[str, Any]) -> list[str]:
    blockers = []
    if export.get("byte_closed_archive_export") is not True:
        blockers.append("hinerv_byte_closed_archive_export_missing")
    return _dedupe(blockers)


def _has_recon_pixel_proof(report: Mapping[str, Any]) -> bool:
    if report.get("recon_pixel_weight_proof_passed") is True:
        return True
    if report.get("decoder_weight_recon_pixel_proof_passed") is True:
        return True
    for row in report.get("rows") or ():
        if isinstance(row, Mapping) and row.get("recon_pixel_weight_proof_passed") is True:
            return True
    return False


def _storage_preflight(
    output_root: Path,
    *,
    min_free_bytes: int,
    allow_local_output: bool,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve(strict=False)
    on_ssd = any(_is_relative_to(root, ssd_root) for ssd_root in SSD_ROOTS)
    free = shutil.disk_usage(_nearest_existing_parent(root)).free
    blockers: list[str] = []
    if not on_ssd and not allow_local_output:
        blockers.append("hinerv_distortion_queue_output_root_not_on_ssd_tier")
    if free < int(min_free_bytes):
        blockers.append("hinerv_distortion_queue_output_root_free_space_below_floor")
    if blockers:
        raise HinervDistortionStabilizationQueueError(
            f"{root}: storage preflight blocked: {', '.join(blockers)}"
        )
    return {
        "schema": "hinerv_distortion_stabilization_storage_preflight.v1",
        "output_root": root.as_posix(),
        "ssd_tier": _ssd_tier(root),
        "free_bytes_before": int(free),
        "min_free_bytes": int(min_free_bytes),
        "allow_local_output": bool(allow_local_output),
        "blockers": [],
    }


def load_json_with_source_identity(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise HinervDistortionStabilizationQueueError(
            f"{source_path}: JSON payload must be an object"
        )
    return {
        **dict(payload),
        "_source_path": source_path.as_posix(),
        "_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def _ssd_tier(path: Path) -> str:
    for root in SSD_ROOTS:
        if _is_relative_to(path, root):
            return root.as_posix()
    return "local_or_unknown"


def _nearest_existing_parent(path: Path) -> Path:
    cursor = path
    while not cursor.exists() and cursor.parent != cursor:
        cursor = cursor.parent
    return cursor


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [str(value) for value in values if str(value)]


def _positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _dedupe(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
