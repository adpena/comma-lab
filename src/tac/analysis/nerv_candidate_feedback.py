# SPDX-License-Identifier: MIT
"""Harvestable feedback rows for NeRV candidate curriculum runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates.hprc.mlx_prefilter_coverage import (
    DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY,
    summarize_mlx_prefilter_coverage,
)
from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT

SCHEMA = "nerv_candidate_feedback_row.v1"
LEDGER_SCHEMA = "nerv_candidate_byte_feedback_ledger.v1"
REFRESH_SCHEMA = "nerv_candidate_feedback_refresh.v1"
TELEMETRY_FEEDBACK_SCHEMA = "nerv_training_telemetry_feedback.v1"

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

_MLX_PREFILTER_MISSING_BLOCKERS = {
    "full_video_mlx_scorer_replay_not_attached",
    "local_cpu_replay_waiting_for_full_video_mlx_prefilter",
    "hi_nerv_full_video_local_prefilter_missing",
    "snerv_full_video_local_prefilter_missing",
    "mlx_prefilter_not_full_video",
    "sampled_mlx_prefilter_requires_full_video_rerun",
}

_LOCAL_REPLAY_BLOCKED_BY_MLX_SCORE = (
    "local_cpu_replay_blocked_by_mlx_prefilter_score"
)
_POSE_LOSS_INSTABILITY_THRESHOLD = 1_000.0
_POSE_AXIS_INSTABILITY_THRESHOLD = 1_000.0
_POSE_INSTABILITY_WINDOW_EPOCHS = 32
_POSE_INSTABILITY_BAD_FRACTION = 0.5
_POSE_INSTABILITY_LR_MULTIPLIER = 0.3


def _sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_nerv_candidate_feedback_row(
    *,
    runner_report: Mapping[str, Any],
    source_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one false-authority feedback row from a runner report."""

    selection = dict(runner_report.get("modelsize_candidate_selection") or {})
    curriculum = dict(
        runner_report.get("candidate_curriculum_plan")
        or selection.get("candidate_curriculum_plan")
        or {}
    )
    byte_feedback = dict(curriculum.get("byte_oracle_logging") or {})
    pr95_binding = dict(
        curriculum.get("pr95_stack_binding")
        or selection.get("pr95_stack_binding")
        or {}
    )
    prelaunch_gate = dict(
        curriculum.get("long_campaign_prelaunch_gate")
        or selection.get("long_campaign_prelaunch_gate")
        or {}
    )
    candidate = selection.get("candidate")
    candidate_row = dict(candidate) if isinstance(candidate, Mapping) else {}
    source_path = (
        Path(source_report_path).expanduser().resolve(strict=False)
        if source_report_path
        else None
    )
    local_replay = runner_report.get("local_cpu_replay_summary")
    local_replay_gate = dict(runner_report.get("local_cpu_replay_gate") or {})
    mlx_prefilter = dict(runner_report.get("mlx_prefilter_coverage") or {})
    snerv_profile = dict(runner_report.get("snerv_binary_profile") or {})
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_report_path": source_path.as_posix() if source_path else None,
        "source_report_sha256": _sha256_file(source_path) if source_path else None,
        "mode": runner_report.get("mode"),
        "family": runner_report.get("execute_family"),
        "candidate_id": candidate_row.get("candidate_id") or curriculum.get("candidate_id"),
        "candidate_conditioned": bool(curriculum.get("candidate_conditioned")),
        "candidate_num_pairs": byte_feedback.get("candidate_num_pairs"),
        "measured_num_pairs": byte_feedback.get("measured_num_pairs"),
        "feedback_scope": byte_feedback.get("feedback_scope"),
        "scope_matches_candidate": bool(byte_feedback.get("scope_matches_candidate")),
        "feedback_ready": bool(byte_feedback.get("feedback_ready")),
        "hard_byte_ceiling": byte_feedback.get("hard_byte_ceiling"),
        "nominal_total_payload_bytes": byte_feedback.get("nominal_total_payload_bytes"),
        "measured_payload_bytes": byte_feedback.get("measured_payload_bytes"),
        "measured_archive_bytes": byte_feedback.get("measured_archive_bytes"),
        "measured_minus_nominal_bytes": byte_feedback.get(
            "measured_minus_nominal_bytes"
        ),
        "archive_path": runner_report.get("archive_path"),
        "archive_bytes": runner_report.get("archive_bytes"),
        "archive_sha256": runner_report.get("archive_sha256"),
        "snerv_binary_profile_path": snerv_profile.get("profile_path"),
        "snerv_binary_profile_written": bool(snerv_profile.get("profile_written")),
        "snerv_binary_profile_verdict": snerv_profile.get("verdict"),
        "snerv_binary_profile_charged_archive_bytes": snerv_profile.get(
            "charged_archive_bytes"
        ),
        "snerv_binary_profile_snar1_packet_bytes": snerv_profile.get(
            "snar1_packet_bytes"
        ),
        "snerv_binary_profile_lf_payload_bytes": snerv_profile.get(
            "lf_payload_bytes"
        ),
        "snerv_binary_profile_lf_payload_fraction_of_packet": snerv_profile.get(
            "lf_payload_fraction_of_packet"
        ),
        "snerv_binary_profile_lf_payload_bytes_per_coeff": snerv_profile.get(
            "lf_payload_bytes_per_coeff"
        ),
        "snerv_binary_profile_blockers": list(snerv_profile.get("blockers") or []),
        "pr95_stack_binding_schema": pr95_binding.get("schema"),
        "pr95_stack_binding_satisfied_count": pr95_binding.get("satisfied_count"),
        "pr95_stack_binding_missing_count": pr95_binding.get("missing_count"),
        "pr95_stack_binding_complete": pr95_binding.get("complete"),
        "pr95_stack_binding_blockers": list(pr95_binding.get("blockers") or []),
        "long_campaign_prelaunch_gate_schema": prelaunch_gate.get("schema"),
        "long_campaign_prelaunch_launch_allowed": prelaunch_gate.get(
            "launch_allowed"
        ),
        "long_campaign_prelaunch_blockers": list(prelaunch_gate.get("blockers") or []),
        "receiver_proof_report_paths": list(
            runner_report.get("receiver_proof_report_paths") or []
        ),
        "local_cpu_replay_summary_present": isinstance(local_replay, Mapping),
        "local_cpu_replay_score_estimate": (
            local_replay.get("local_score_estimate")
            if isinstance(local_replay, Mapping)
            else None
        ),
        "local_cpu_replay_gate_requested": local_replay_gate.get("requested"),
        "local_cpu_replay_gate_default_enabled_for_full_coverage": (
            local_replay_gate.get("default_enabled_for_full_coverage")
        ),
        "local_cpu_replay_gate_has_full_video_mlx_prefilter": (
            local_replay_gate.get("has_full_video_mlx_prefilter")
        ),
        "local_cpu_replay_gate_local_replay_mlx_prefilter_passed": (
            local_replay_gate.get("local_replay_mlx_prefilter_passed")
        ),
        "local_cpu_replay_gate_coverage_valid_for_replay": (
            local_replay_gate.get("coverage_valid_for_replay")
        ),
        "local_cpu_replay_gate_executed": local_replay_gate.get("executed"),
        "mlx_prefilter_profile_count": mlx_prefilter.get("profile_count"),
        "mlx_prefilter_has_full_video": mlx_prefilter.get(
            "has_full_video_mlx_prefilter"
        ),
        "mlx_prefilter_local_replay_passed": mlx_prefilter.get(
            "local_replay_mlx_prefilter_passed"
        ),
        "mlx_prefilter_best_full_video_mlx_score": mlx_prefilter.get(
            "best_full_video_mlx_score"
        ),
        "mlx_prefilter_full_video_profile_paths": list(
            mlx_prefilter.get("full_video_profile_paths") or []
        ),
        "mlx_prefilter_local_replay_profile_paths": list(
            mlx_prefilter.get("local_replay_profile_paths") or []
        ),
        "mlx_prefilter_blockers": list(mlx_prefilter.get("blockers") or []),
        "blockers": list(runner_report.get("blockers") or []),
        **FALSE_AUTHORITY,
    }


def build_nerv_training_telemetry_feedback_row(
    *,
    telemetry_path: str | Path,
    family: str,
    candidate_id: str,
    candidate_num_pairs: int,
    source_queue_path: str | Path | None = None,
    stop_reason: str | None = None,
    pose_loss_instability_threshold: float = _POSE_LOSS_INSTABILITY_THRESHOLD,
    pose_axis_instability_threshold: float = _POSE_AXIS_INSTABILITY_THRESHOLD,
    instability_window_epochs: int = _POSE_INSTABILITY_WINDOW_EPOCHS,
    instability_bad_fraction: float = _POSE_INSTABILITY_BAD_FRACTION,
    learning_rate_multiplier: float = _POSE_INSTABILITY_LR_MULTIPLIER,
) -> dict[str, Any]:
    """Build a false-authority candidate-feedback row from training telemetry."""

    telemetry = Path(telemetry_path).expanduser().resolve(strict=False)
    if not telemetry.is_file():
        raise FileNotFoundError(f"telemetry file not found: {telemetry}")
    source_queue = (
        Path(source_queue_path).expanduser().resolve(strict=False)
        if source_queue_path
        else None
    )
    rows = _read_telemetry_rows(telemetry)
    if not rows:
        raise ValueError(f"telemetry file has no JSON rows: {telemetry}")
    health = _summarize_training_telemetry_health(
        rows,
        pose_loss_instability_threshold=float(pose_loss_instability_threshold),
        pose_axis_instability_threshold=float(pose_axis_instability_threshold),
        instability_window_epochs=int(instability_window_epochs),
        instability_bad_fraction=float(instability_bad_fraction),
        learning_rate_multiplier=float(learning_rate_multiplier),
    )
    candidate_pairs = int(candidate_num_pairs)
    measured_pairs = _int_or_none(health.get("num_pairs")) or candidate_pairs
    blockers: list[str] = [
        "hinerv_trained_archive_byte_oracle_feedback_missing",
        "hi_nerv_byte_closed_archive_export_missing",
        "hi_nerv_receiver_proof_missing",
        "hi_nerv_full_video_local_prefilter_missing",
        "hi_nerv_local_cpu_replay_gate_missing",
    ]
    if health["pose_instability_detected"]:
        blockers.append("hi_nerv_pose_instability_telemetry_feedback")
    return {
        "schema": SCHEMA,
        "feedback_kind": "training_telemetry",
        "telemetry_feedback_schema": TELEMETRY_FEEDBACK_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_report_path": telemetry.as_posix(),
        "source_report_sha256": _sha256_file(telemetry),
        "source_queue_path": source_queue.as_posix() if source_queue else None,
        "source_queue_sha256": _sha256_file(source_queue) if source_queue else None,
        "mode": "training_telemetry_harvested",
        "family": _family_key(family),
        "candidate_id": str(candidate_id),
        "candidate_conditioned": True,
        "candidate_num_pairs": candidate_pairs,
        "measured_num_pairs": measured_pairs,
        "feedback_scope": "full600_training_telemetry"
        if measured_pairs >= CONTEST_PAIR_COUNT
        else "partial_training_telemetry",
        "scope_matches_candidate": measured_pairs >= candidate_pairs,
        "feedback_ready": False,
        "hard_byte_ceiling": None,
        "nominal_total_payload_bytes": None,
        "measured_payload_bytes": None,
        "measured_archive_bytes": None,
        "measured_minus_nominal_bytes": None,
        "archive_path": None,
        "archive_bytes": None,
        "archive_sha256": None,
        "training_completed": False,
        "training_stopped": True,
        "training_stop_reason": stop_reason
        or (
            "pose_instability_telemetry"
            if health["pose_instability_detected"]
            else "telemetry_harvest_without_completion_artifact"
        ),
        "training_telemetry": health,
        "pose_instability_detected": bool(health["pose_instability_detected"]),
        "pose_instability_partial_window_detected": bool(
            health.get("pose_instability_partial_window_detected")
        ),
        "pose_instability_first_epoch": health.get("pose_instability_first_epoch"),
        "pose_instability_last_window_bad_fraction": health.get(
            "pose_instability_last_window_bad_fraction"
        ),
        "observed_learning_rate": health.get("observed_learning_rate"),
        "recommended_learning_rate": health.get("recommended_learning_rate"),
        "recommended_learning_rate_multiplier": health.get(
            "recommended_learning_rate_multiplier"
        ),
        "recommended_launch_mutations": list(
            health.get("recommended_launch_mutations") or []
        ),
        "receiver_proof_report_paths": [],
        "local_cpu_replay_summary_present": False,
        "local_cpu_replay_score_estimate": None,
        "local_cpu_replay_gate_requested": None,
        "local_cpu_replay_gate_default_enabled_for_full_coverage": False,
        "local_cpu_replay_gate_has_full_video_mlx_prefilter": False,
        "local_cpu_replay_gate_local_replay_mlx_prefilter_passed": False,
        "local_cpu_replay_gate_coverage_valid_for_replay": False,
        "local_cpu_replay_gate_executed": False,
        "mlx_prefilter_profile_count": 0,
        "mlx_prefilter_has_full_video": False,
        "mlx_prefilter_local_replay_passed": False,
        "mlx_prefilter_best_full_video_mlx_score": None,
        "mlx_prefilter_full_video_profile_paths": [],
        "mlx_prefilter_local_replay_profile_paths": [],
        "mlx_prefilter_blockers": ["full_video_mlx_scorer_replay_not_attached"],
        "blockers": _dedupe_strings(blockers),
        **FALSE_AUTHORITY,
    }


def refresh_nerv_candidate_feedback_report(
    *,
    runner_report: Mapping[str, Any],
    repo_root: str | Path,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    required_pairs: int = CONTEST_PAIR_COUNT,
    max_mlx_score_for_local_replay: float | None = (
        DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a feedback-safe runner report with refreshed MLX gate evidence.

    This is a backfill helper for reports produced before the batched-MLX
    acquisition/replay split existed. It does not rewrite the source report or
    grant authority; it only recomputes typed coverage from file-backed profile
    paths so candidate-feedback ledgers do not lose useful full-video GPU
    acquisition signal.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    selected_profiles = _dedupe_strings(
        [
            *(str(path) for path in mlx_profile_paths),
            *_infer_mlx_profile_paths(runner_report),
        ]
    )
    refreshed = json.loads(json.dumps(dict(runner_report), sort_keys=True, default=str))
    old_blockers = list(refreshed.get("blockers") or [])
    coverage = summarize_mlx_prefilter_coverage(
        tuple(selected_profiles),
        root=root,
        required_pairs=int(required_pairs),
        max_mlx_score_for_local_replay=max_mlx_score_for_local_replay,
    )
    refreshed["mlx_profile_paths"] = [
        str(path) for path in coverage.get("full_video_profile_paths") or selected_profiles
    ]
    refreshed["mlx_prefilter_coverage"] = coverage
    has_full = bool(coverage.get("has_full_video_mlx_prefilter"))
    replay_passed = bool(coverage.get("local_replay_mlx_prefilter_passed"))
    num_pairs = _int_or_none(refreshed.get("num_pairs")) or 0
    coverage_valid_for_replay = int(num_pairs) >= int(required_pairs)
    gate = dict(refreshed.get("local_cpu_replay_gate") or {})
    gate.setdefault("schema", "compact_runner_local_cpu_replay_gate.v1")
    gate["has_full_video_mlx_prefilter"] = has_full
    gate["local_replay_mlx_prefilter_passed"] = replay_passed
    gate["coverage_valid_for_replay"] = coverage_valid_for_replay
    gate["default_enabled_for_full_coverage"] = bool(
        coverage_valid_for_replay and replay_passed
    )
    gate.setdefault("requested", None)
    gate.setdefault(
        "executed",
        isinstance(refreshed.get("local_cpu_replay_summary"), Mapping),
    )
    refreshed["local_cpu_replay_gate"] = gate

    blockers = list(old_blockers)
    removed_blockers: list[str] = []
    if has_full:
        kept: list[str] = []
        for blocker in blockers:
            if blocker in _MLX_PREFILTER_MISSING_BLOCKERS:
                removed_blockers.append(str(blocker))
            else:
                kept.append(str(blocker))
        blockers = kept
    blockers.extend(str(blocker) for blocker in coverage.get("blockers") or [])
    if has_full and coverage_valid_for_replay and not replay_passed:
        blockers.append(_LOCAL_REPLAY_BLOCKED_BY_MLX_SCORE)
    refreshed["blockers"] = _dedupe_strings(blockers)
    nested_removed = (
        _refresh_nested_pr95_stack_binding_blockers(refreshed) if has_full else []
    )

    refresh = {
        "schema": REFRESH_SCHEMA,
        "profile_paths": selected_profiles,
        "required_pairs": int(required_pairs),
        "max_mlx_score_for_local_replay": max_mlx_score_for_local_replay,
        "has_full_video_mlx_prefilter": has_full,
        "local_replay_mlx_prefilter_passed": replay_passed,
        "old_blockers": old_blockers,
        "removed_stale_blockers": removed_blockers,
        "removed_nested_pr95_stack_binding_blockers": nested_removed,
        "new_blockers": refreshed["blockers"],
        "mlx_prefilter_coverage": coverage,
        **FALSE_AUTHORITY,
    }
    return refreshed, refresh


def write_nerv_candidate_feedback_files(
    *,
    runner_report: Mapping[str, Any],
    output_dir: str | Path,
    source_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write JSON and JSONL feedback artifacts under ``output_dir``."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    row = build_nerv_candidate_feedback_row(
        runner_report=runner_report,
        source_report_path=source_report_path,
    )
    row_path = out / "nerv_candidate_byte_feedback_row.json"
    ledger_path = out / "nerv_candidate_byte_feedback.jsonl"
    row_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "schema": LEDGER_SCHEMA,
        "row": row,
        "row_path": row_path.as_posix(),
        "ledger_path": ledger_path.as_posix(),
        "append_only": True,
        **FALSE_AUTHORITY,
    }


def write_refreshed_nerv_candidate_feedback_files(
    *,
    runner_report: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    source_report_path: str | Path | None = None,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    required_pairs: int = CONTEST_PAIR_COUNT,
    max_mlx_score_for_local_replay: float | None = (
        DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY
    ),
) -> dict[str, Any]:
    """Refresh MLX coverage and write a feedback row plus refresh manifest."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    refreshed, refresh = refresh_nerv_candidate_feedback_report(
        runner_report=runner_report,
        repo_root=repo_root,
        mlx_profile_paths=mlx_profile_paths,
        required_pairs=int(required_pairs),
        max_mlx_score_for_local_replay=max_mlx_score_for_local_replay,
    )
    refreshed_report_path = out / "refreshed_runner_report_for_feedback.json"
    refreshed_report_path.write_text(
        json.dumps(refreshed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    feedback = write_nerv_candidate_feedback_files(
        runner_report=refreshed,
        output_dir=out,
        source_report_path=source_report_path,
    )
    refresh_path = out / "nerv_candidate_feedback_refresh.json"
    refresh.update(
        {
            "refreshed_runner_report_path": refreshed_report_path.as_posix(),
            "candidate_feedback_row_path": feedback["row_path"],
            "candidate_feedback_ledger_path": feedback["ledger_path"],
        }
    )
    refresh_path.write_text(
        json.dumps(refresh, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": REFRESH_SCHEMA,
        "refresh": refresh,
        "refresh_path": refresh_path.as_posix(),
        "refreshed_runner_report_path": refreshed_report_path.as_posix(),
        "candidate_feedback": feedback,
        **FALSE_AUTHORITY,
    }


def write_nerv_training_telemetry_feedback_files(
    *,
    telemetry_path: str | Path,
    output_dir: str | Path,
    family: str,
    candidate_id: str,
    candidate_num_pairs: int,
    source_queue_path: str | Path | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Write a telemetry feedback row plus append-only ledger."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry_path,
        family=family,
        candidate_id=candidate_id,
        candidate_num_pairs=int(candidate_num_pairs),
        source_queue_path=source_queue_path,
        stop_reason=stop_reason,
    )
    row_path = out / "nerv_candidate_training_telemetry_feedback_row.json"
    ledger_path = out / "nerv_candidate_training_telemetry_feedback.jsonl"
    row_path.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": TELEMETRY_FEEDBACK_SCHEMA,
        "row": row,
        "row_path": row_path.as_posix(),
        "ledger_path": ledger_path.as_posix(),
        "append_only": True,
        **FALSE_AUTHORITY,
    }
    manifest_path = out / "nerv_training_telemetry_feedback.json"
    manifest.update({"manifest_path": manifest_path.as_posix()})
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _infer_mlx_profile_paths(runner_report: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("mlx_profile_paths", "local_mlx_prefilter_profile_paths"):
        value = runner_report.get(key)
        if isinstance(value, (list, tuple)):
            paths.extend(str(path) for path in value if path)
    auto_path = runner_report.get("auto_mlx_prefilter_profile_path")
    if auto_path:
        paths.append(str(auto_path))
    coverage = runner_report.get("mlx_prefilter_coverage")
    if isinstance(coverage, Mapping):
        for key in ("full_video_profile_paths", "local_replay_profile_paths"):
            value = coverage.get(key)
            if isinstance(value, (list, tuple)):
                paths.extend(str(path) for path in value if path)
    return _dedupe_strings(paths)


def _read_telemetry_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON telemetry row") from exc
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _summarize_training_telemetry_health(
    rows: Sequence[Mapping[str, Any]],
    *,
    pose_loss_instability_threshold: float,
    pose_axis_instability_threshold: float,
    instability_window_epochs: int,
    instability_bad_fraction: float,
    learning_rate_multiplier: float,
) -> dict[str, Any]:
    epochs: list[int] = []
    pose_losses: list[float] = []
    pose_axes: list[float] = []
    seg_axes: list[float] = []
    learning_rates: list[float] = []
    bad_epochs: list[int] = []
    window_size = max(1, int(instability_window_epochs))
    bad_fraction_threshold = min(max(float(instability_bad_fraction), 0.0), 1.0)
    partial_window_min_epochs = max(
        1,
        math.ceil(float(window_size) * float(bad_fraction_threshold)),
    )
    first_bad_window_epoch: int | None = None
    rolling_flags: list[bool] = []
    for row in rows:
        epoch = _int_or_none(row.get("epoch"))
        if epoch is not None:
            epochs.append(epoch)
        lr = _float_or_none(row.get("learning_rate"))
        if lr is not None:
            learning_rates.append(lr)
        loss_components = row.get("loss_components")
        per_axis = row.get("per_axis_decomposition")
        pose_loss = (
            _float_or_none(loss_components.get("loss_part_pose_distill"))
            if isinstance(loss_components, Mapping)
            else None
        )
        pose_axis = (
            _float_or_none(per_axis.get("pose"))
            if isinstance(per_axis, Mapping)
            else None
        )
        seg_axis = (
            _float_or_none(per_axis.get("seg"))
            if isinstance(per_axis, Mapping)
            else None
        )
        if pose_loss is not None:
            pose_losses.append(pose_loss)
        if pose_axis is not None:
            pose_axes.append(pose_axis)
        if seg_axis is not None:
            seg_axes.append(seg_axis)
        bad = bool(
            (pose_loss is not None and pose_loss >= pose_loss_instability_threshold)
            or (pose_axis is not None and pose_axis >= pose_axis_instability_threshold)
        )
        rolling_flags.append(bad)
        if bad and epoch is not None:
            bad_epochs.append(epoch)
        window = rolling_flags[-window_size:]
        if len(window) == window_size and first_bad_window_epoch is None:
            bad_fraction = sum(1 for flag in window if flag) / float(window_size)
            if bad_fraction >= bad_fraction_threshold:
                first_bad_window_epoch = epoch
    last_window = rolling_flags[-window_size:]
    last_bad_fraction = (
        sum(1 for flag in last_window if flag) / float(len(last_window))
        if last_window
        else 0.0
    )
    partial_window_instability = bool(
        first_bad_window_epoch is None
        and len(rolling_flags) < window_size
        and len(rolling_flags) >= partial_window_min_epochs
        and last_bad_fraction >= bad_fraction_threshold
    )
    if partial_window_instability:
        first_bad_window_epoch = epochs[-1] if epochs else None
    observed_lr = learning_rates[-1] if learning_rates else None
    instability = bool(first_bad_window_epoch is not None or partial_window_instability)
    recommended_lr = (
        max(float(observed_lr) * float(learning_rate_multiplier), 1.0e-6)
        if instability and observed_lr is not None
        else None
    )
    mutations: list[str] = []
    if instability:
        mutations.extend(
            [
                "lower_learning_rate_from_pose_instability_telemetry",
                "preserve_pose_instability_guard_for_relaunch",
                "treat_previous_hi_nerv_run_as_fit_failure_not_rate_negative",
            ]
        )
    return {
        "schema": TELEMETRY_FEEDBACK_SCHEMA,
        "row_count": len(rows),
        "num_pairs": CONTEST_PAIR_COUNT,
        "first_epoch": min(epochs) if epochs else None,
        "last_epoch": max(epochs) if epochs else None,
        "observed_learning_rate": observed_lr,
        "pose_loss_instability_threshold": float(pose_loss_instability_threshold),
        "pose_axis_instability_threshold": float(pose_axis_instability_threshold),
        "instability_window_epochs": int(window_size),
        "instability_bad_fraction_threshold": float(bad_fraction_threshold),
        "partial_window_instability_min_epochs": int(partial_window_min_epochs),
        "pose_instability_partial_window_detected": partial_window_instability,
        "pose_bad_epoch_count": len(bad_epochs),
        "pose_bad_epoch_fraction": (
            len(bad_epochs) / float(len(rows)) if rows else 0.0
        ),
        "pose_instability_detected": instability,
        "pose_instability_first_epoch": first_bad_window_epoch,
        "pose_instability_last_window_bad_fraction": last_bad_fraction,
        "max_pose_distill_loss": max(pose_losses) if pose_losses else None,
        "max_pose_axis": max(pose_axes) if pose_axes else None,
        "median_pose_distill_loss": _median(pose_losses),
        "median_pose_axis": _median(pose_axes),
        "median_seg_axis": _median(seg_axes),
        "recommended_learning_rate": recommended_lr,
        "recommended_learning_rate_multiplier": (
            float(learning_rate_multiplier) if instability else None
        ),
        "recommended_launch_mutations": mutations,
        **FALSE_AUTHORITY,
    }


def _refresh_nested_pr95_stack_binding_blockers(report: dict[str, Any]) -> list[str]:
    removed: list[str] = []
    candidate_paths = [
        ("candidate_curriculum_plan", "pr95_stack_binding"),
        (
            "modelsize_candidate_selection",
            "candidate_curriculum_plan",
            "pr95_stack_binding",
        ),
        ("modelsize_candidate_selection", "pr95_stack_binding"),
    ]
    for path in candidate_paths:
        container = report
        for key in path:
            next_value = container.get(key) if isinstance(container, dict) else None
            if not isinstance(next_value, dict):
                container = {}
                break
            container = next_value
        if not container:
            continue
        blockers = list(container.get("blockers") or [])
        kept = [
            str(blocker)
            for blocker in blockers
            if blocker not in _MLX_PREFILTER_MISSING_BLOCKERS
        ]
        path_removed = [
            str(blocker)
            for blocker in blockers
            if blocker in _MLX_PREFILTER_MISSING_BLOCKERS
        ]
        if not path_removed:
            continue
        container["blockers"] = _dedupe_strings(kept)
        container["missing_count"] = len(container["blockers"])
        satisfied = _int_or_none(container.get("satisfied_count"))
        if satisfied is not None:
            container["satisfied_count"] = satisfied + len(path_removed)
        container["complete"] = not container["blockers"]
        removed.extend(path_removed)
    return _dedupe_strings(removed)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    import math

    return number if math.isfinite(number) else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _family_key(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text == "hinerv":
        return "hi_nerv"
    return text


__all__ = [
    "LEDGER_SCHEMA",
    "REFRESH_SCHEMA",
    "SCHEMA",
    "TELEMETRY_FEEDBACK_SCHEMA",
    "build_nerv_candidate_feedback_row",
    "build_nerv_training_telemetry_feedback_row",
    "refresh_nerv_candidate_feedback_report",
    "write_nerv_candidate_feedback_files",
    "write_nerv_training_telemetry_feedback_files",
    "write_refreshed_nerv_candidate_feedback_files",
]
