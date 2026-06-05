#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest HiNeRV smoke/export reports into a planner-consumable comparison."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import read_json, sha256_file, write_json_artifact, write_text_artifact  # noqa: E402

SCHEMA = "hinerv_smoke_comparison_harvest.v1"
FEEDBACK_REFRESH_SCHEMA = "nerv_queue_training_feedback_refresh.v1"
RUNNER_REPORT_NAME = "compact_renderer_mlx_spine_runner_report.json"
ACQUISITION_REPORT_NAME = "hprc_spine_acquisition_report.json"
EXPORT_REPORT_NAME = "hinerv_checkpoint_archive_export.json"
BITSTREAM_REPORT_NAME = "hi_nerv_bitstream_preparation.json"
WATERFILL_REPORT_GLOB = "decoder_weight_waterfill_plan*.json"
DEFAULT_ARTIFACT_ROOT = Path("/Volumes/VertigoDataTier/pact/experiments/results")
FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
_VARIANT_TOKENS = {
    "direct_live": ("direct_live", "directlive"),
    "argmax": ("argmax",),
    "pose_warmup": ("pose_warmup", "posewarm"),
    "spatial_guard": ("spatialguard", "spatial_guard"),
    "bitstream": ("bitstream",),
    "waterfill": ("waterfill",),
    "stage_qat": ("stage_qat", "qat"),
    "section_dual": ("section_dual",),
    "seg_only": ("segonly", "seg_only"),
    "pose_sqrt": ("pose_sqrt",),
    "dynamic_range_guard": ("dynamic_range_guard", "range_guard"),
    "exact_scorer_fit": ("exact_scorer_fit",),
}


def build_hinerv_smoke_comparison(
    *,
    artifact_roots: Sequence[str | Path] = (DEFAULT_ARTIFACT_ROOT,),
    glob_patterns: Sequence[str] = ("hinerv_*",),
    limit: int = 200,
) -> dict[str, Any]:
    """Return a false-authority comparison over HiNeRV SSD smoke artifacts."""

    run_dirs = _discover_run_dirs(
        artifact_roots=artifact_roots,
        glob_patterns=glob_patterns,
        limit=limit,
    )
    rows: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        row = _row_for_run_dir(run_dir)
        if row is None:
            skipped.append(
                {
                    "run_dir": run_dir.as_posix(),
                    "reason": "no_supported_hinerv_report",
                }
            )
            continue
        rows.append(row)
        nested_feedback = row.get("embedded_candidate_feedback_row")
        if isinstance(nested_feedback, Mapping):
            feedback_rows.append(
                {
                    "experiment_id": row["run_id"],
                    "step_id": "embedded_runner_candidate_feedback",
                    "status": "harvested",
                    "family": "hi_nerv",
                    "candidate_id": row.get("candidate_id") or "",
                    "source_report_path": row.get("primary_report_path") or "",
                    "row": dict(nested_feedback),
                }
            )
    rows = sorted(rows, key=_row_sort_key)
    byte_frontier_row = min(
        (row for row in rows if row.get("archive_bytes") is not None),
        key=lambda row: (int(row["archive_bytes"]), str(row.get("run_id") or "")),
        default=None,
    )
    most_ready_row = max(
        rows,
        key=lambda row: (
            int(row.get("readiness_score") or 0),
            -(int(row.get("archive_bytes") or 10**18)),
            str(row.get("run_id") or ""),
        ),
        default=None,
    )
    variant_counts = {
        key: sum(1 for row in rows if bool(row["variant_flags"].get(key)))
        for key in sorted(_VARIANT_TOKENS)
    }
    feedback_refresh = {
        "schema": FEEDBACK_REFRESH_SCHEMA,
        "queue_id": "hinerv_smoke_comparison_harvest",
        "queue_path": None,
        "queue_sha256": None,
        "queue_summary_schema": SCHEMA,
        "included_statuses": ["harvested"],
        "refreshed_row_count": len(feedback_rows),
        "skipped_count": len(skipped),
        "rows": feedback_rows,
        "skipped": skipped,
        **FALSE_AUTHORITY,
    }
    return {
        "schema": SCHEMA,
        "generated_utc": _utc_now(),
        "artifact_roots": [Path(root).expanduser().resolve(strict=False).as_posix() for root in artifact_roots],
        "glob_patterns": list(glob_patterns),
        "discovered_run_dir_count": len(run_dirs),
        "row_count": len(rows),
        "embedded_candidate_feedback_row_count": len(feedback_rows),
        "variant_counts": variant_counts,
        "byte_frontier_row": _public_row(byte_frontier_row),
        "most_ready_row": _public_row(most_ready_row),
        "next_actions": _next_actions(rows),
        "rows": [_public_row(row) for row in rows],
        "feedback_refresh": feedback_refresh,
        **FALSE_AUTHORITY,
    }


def write_hinerv_smoke_comparison(
    *,
    report: Mapping[str, Any],
    output_json: str | Path,
    feedback_refresh_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    """Write the comparison report and optional downstream artifacts."""

    comparison = write_json_artifact(output_json, dict(report))
    out: dict[str, Any] = {
        "schema": "hinerv_smoke_comparison_harvest_write.v1",
        "report_path": comparison.path,
        "report_sha256": comparison.sha256,
        "report_bytes": comparison.bytes_written,
        "row_count": int(report.get("row_count") or 0),
        "embedded_candidate_feedback_row_count": int(
            report.get("embedded_candidate_feedback_row_count") or 0
        ),
        **FALSE_AUTHORITY,
    }
    if feedback_refresh_json is not None:
        refresh = write_json_artifact(
            feedback_refresh_json,
            dict(report.get("feedback_refresh") or {}),
        )
        out["feedback_refresh_path"] = refresh.path
        out["feedback_refresh_sha256"] = refresh.sha256
        out["feedback_refresh_bytes"] = refresh.bytes_written
    if output_md is not None:
        md = write_text_artifact(output_md, render_hinerv_smoke_comparison_markdown(report))
        out["markdown_path"] = md.path
        out["markdown_sha256"] = md.sha256
        out["markdown_bytes"] = md.bytes_written
    return out


def render_hinerv_smoke_comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# HiNeRV Smoke Comparison Harvest",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- rows: {int(report.get('row_count') or 0)}",
        f"- embedded feedback rows: {int(report.get('embedded_candidate_feedback_row_count') or 0)}",
        f"- score claim: {bool(report.get('score_claim'))}",
        f"- promotion eligible: {bool(report.get('promotion_eligible'))}",
        "",
    ]
    byte_frontier = report.get("byte_frontier_row")
    if isinstance(byte_frontier, Mapping):
        lines.extend(
            [
                "## Byte Frontier Row",
                "",
                f"- run: `{byte_frontier.get('run_id')}`",
                f"- archive bytes: {byte_frontier.get('archive_bytes')}",
                f"- readiness score: {byte_frontier.get('readiness_score')}",
                f"- next action: `{byte_frontier.get('next_action')}`",
                "",
            ]
        )
    most_ready = report.get("most_ready_row")
    if isinstance(most_ready, Mapping):
        lines.extend(
            [
                "## Most Ready Row",
                "",
                f"- run: `{most_ready.get('run_id')}`",
                f"- archive bytes: {most_ready.get('archive_bytes')}",
                f"- readiness score: {most_ready.get('readiness_score')}",
                f"- next action: `{most_ready.get('next_action')}`",
                "",
            ]
        )
    lines.extend(["## Top Rows", ""])
    for row in report.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"`{row.get('run_id')}` bytes={row.get('archive_bytes')} "
            f"ready={row.get('readiness_score')} action=`{row.get('next_action')}`"
        )
    lines.append("")
    return "\n".join(lines)


def _discover_run_dirs(
    *,
    artifact_roots: Sequence[str | Path],
    glob_patterns: Sequence[str],
    limit: int,
) -> list[Path]:
    seen: set[Path] = set()
    dirs: list[Path] = []
    for root_value in artifact_roots:
        root = Path(root_value).expanduser().resolve(strict=False)
        if not root.is_dir():
            continue
        for pattern in glob_patterns:
            for path in root.glob(pattern):
                if not path.is_dir():
                    continue
                resolved = path.resolve(strict=False)
                if resolved in seen:
                    continue
                seen.add(resolved)
                dirs.append(resolved)
    dirs.sort(key=lambda path: (_path_mtime_ns(path), path.as_posix()), reverse=True)
    if limit > 0:
        return dirs[:limit]
    return dirs


def _row_for_run_dir(run_dir: Path) -> dict[str, Any] | None:
    reports = _load_reports(run_dir)
    if not reports:
        return None
    bitstream_paths = _nested_artifact_paths(run_dir, BITSTREAM_REPORT_NAME)
    waterfill_paths = _nested_artifact_paths(run_dir, WATERFILL_REPORT_GLOB)
    variant_flags = _variant_flags(run_dir.name)
    variant_flags["bitstream"] = bool(variant_flags["bitstream"] or bitstream_paths)
    variant_flags["waterfill"] = bool(variant_flags["waterfill"] or waterfill_paths)
    runner = reports.get(RUNNER_REPORT_NAME)
    acquisition = reports.get(ACQUISITION_REPORT_NAME)
    export = reports.get(EXPORT_REPORT_NAME)
    embedded_feedback = _embedded_candidate_feedback_row(runner)
    primary_path = _primary_report_path(run_dir, reports)
    archive_bytes = _first_int(
        _mapping_get(runner, "archive_bytes"),
        _mapping_get(embedded_feedback, "archive_bytes"),
        _mapping_get(export, "archive_bytes"),
        _best_acquisition_bytes(acquisition),
    )
    blockers = _dedupe_strings(
        [
            *_string_items(_mapping_get(runner, "blockers")),
            *_string_items(_mapping_get(embedded_feedback, "blockers")),
            *_string_items(_mapping_get(export, "blockers")),
            *_acquisition_blockers(acquisition),
        ]
    )
    readiness = _readiness_score(
        runner=runner,
        export=export,
        embedded_feedback=embedded_feedback,
        blockers=blockers,
    )
    report_paths = [
        (run_dir / name).as_posix()
        for name in (RUNNER_REPORT_NAME, ACQUISITION_REPORT_NAME, EXPORT_REPORT_NAME)
        if name in reports
    ]
    return {
        "run_id": run_dir.name,
        "run_dir": run_dir.as_posix(),
        "primary_report_path": primary_path.as_posix() if primary_path is not None else None,
        "primary_report_sha256": (
            sha256_file(primary_path) if primary_path is not None else None
        ),
        "report_paths": report_paths,
        "report_schemas": {
            name: payload.get("schema") for name, payload in reports.items() if isinstance(payload, Mapping)
        },
        "variant_flags": variant_flags,
        "bitstream_preparation_count": len(bitstream_paths),
        "bitstream_preparation_paths": [path.as_posix() for path in bitstream_paths],
        "decoder_weight_waterfill_plan_count": len(waterfill_paths),
        "decoder_weight_waterfill_plan_paths": [path.as_posix() for path in waterfill_paths],
        "candidate_id": _first_str(
            _mapping_get(embedded_feedback, "candidate_id"),
            _mapping_get(runner, "candidate_id"),
            _mapping_get(export, "candidate_id"),
            _nested_mapping_get(runner, ("modelsize_candidate_selection", "candidate_id")),
            _nested_mapping_get(runner, ("modelsize_candidate_selection", "selected_candidate_id")),
        ),
        "training_executed": _first_bool(_mapping_get(runner, "training_executed")),
        "num_pairs": _first_int(
            _mapping_get(runner, "num_pairs"),
            _mapping_get(embedded_feedback, "measured_num_pairs"),
            _mapping_get(embedded_feedback, "candidate_num_pairs"),
            _nested_mapping_get(acquisition, ("rows", 0, "coverage", "declared_pairs")),
        ),
        "archive_bytes": archive_bytes,
        "hard_byte_ceiling": _first_int(
            _mapping_get(embedded_feedback, "hard_byte_ceiling"),
            _mapping_get(export, "hard_byte_ceiling_requested_by_candidate_or_startup"),
            _first_sequence_int(_mapping_get(runner, "hard_byte_ceilings")),
        ),
        "archive_path": _first_str(
            _mapping_get(runner, "archive_path"),
            _mapping_get(embedded_feedback, "archive_path"),
            _mapping_get(export, "archive_path"),
        ),
        "archive_sha256": _first_str(
            _mapping_get(runner, "archive_sha256"),
            _mapping_get(embedded_feedback, "archive_sha256"),
            _mapping_get(export, "archive_sha256"),
        ),
        "receiver_proof_ready": _first_bool(
            _mapping_get(export, "receiver_proof_ready"),
            _mapping_get(embedded_feedback, "receiver_proof_attached"),
            _nested_mapping_get(runner, ("receiver_proof", "passed")),
        ),
        "local_cpu_replay_ready": _first_bool(
            _mapping_get(embedded_feedback, "local_cpu_replay_gate_attached"),
            _nested_mapping_get(runner, ("local_cpu_replay_gate", "passed")),
            _nested_mapping_get(runner, ("local_cpu_replay_gate", "default_enabled_for_full_coverage")),
        ),
        "full_video_prefilter_ready": _first_bool(
            _mapping_get(embedded_feedback, "full_video_local_prefilter_attached"),
            _nested_mapping_get(runner, ("mlx_prefilter_coverage", "has_full_video_mlx_prefilter")),
            _mapping_get(export, "local_mlx_prefilter_written"),
        ),
        "acquisition_row_count": _first_int(_mapping_get(acquisition, "row_count")),
        "best_acquisition_bytes": _best_acquisition_bytes(acquisition),
        "acquisition_next_actions": _acquisition_next_actions(acquisition),
        "readiness_score": readiness,
        "next_action": _row_next_action(
            archive_bytes=archive_bytes,
            readiness_score=readiness,
            blockers=blockers,
        ),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "embedded_candidate_feedback_row": dict(embedded_feedback) if isinstance(embedded_feedback, Mapping) else None,
        **FALSE_AUTHORITY,
    }


def _load_reports(run_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for name in (RUNNER_REPORT_NAME, ACQUISITION_REPORT_NAME, EXPORT_REPORT_NAME):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        reports[name] = payload
    return reports


def _nested_artifact_paths(run_dir: Path, pattern: str, *, limit: int = 12) -> list[Path]:
    paths: list[Path] = []
    try:
        matches = run_dir.rglob(pattern)
        for path in matches:
            if not path.is_file():
                continue
            paths.append(path.resolve(strict=False))
            if len(paths) >= limit:
                break
    except OSError:
        return []
    return sorted(paths, key=lambda path: path.as_posix())


def _embedded_candidate_feedback_row(runner: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    candidate_feedback = runner.get("candidate_feedback") if isinstance(runner, Mapping) else None
    if isinstance(candidate_feedback, Mapping):
        row = candidate_feedback.get("row")
        if isinstance(row, Mapping) and row.get("schema") == "nerv_candidate_feedback_row.v1":
            return row
    return None


def _primary_report_path(run_dir: Path, reports: Mapping[str, Mapping[str, Any]]) -> Path | None:
    for name in (RUNNER_REPORT_NAME, EXPORT_REPORT_NAME, ACQUISITION_REPORT_NAME):
        if name in reports:
            return run_dir / name
    return None


def _readiness_score(
    *,
    runner: Mapping[str, Any] | None,
    export: Mapping[str, Any] | None,
    embedded_feedback: Mapping[str, Any] | None,
    blockers: Sequence[str],
) -> int:
    score = 0
    if _first_bool(_mapping_get(runner, "training_executed")) is True:
        score += 1
    if _first_bool(_mapping_get(embedded_feedback, "feedback_ready")) is True:
        score += 1
    if _first_bool(
        _mapping_get(export, "receiver_proof_ready"),
        _mapping_get(embedded_feedback, "receiver_proof_attached"),
    ) is True:
        score += 1
    if _first_bool(
        _mapping_get(embedded_feedback, "full_video_local_prefilter_attached"),
        _nested_mapping_get(runner, ("mlx_prefilter_coverage", "has_full_video_mlx_prefilter")),
    ) is True:
        score += 1
    if _first_bool(_mapping_get(embedded_feedback, "local_cpu_replay_gate_attached")) is True:
        score += 1
    if not blockers:
        score += 1
    return score


def _row_next_action(
    *,
    archive_bytes: int | None,
    readiness_score: int,
    blockers: Sequence[str],
) -> str:
    blocker_set = set(blockers)
    if archive_bytes is None:
        return "harvest_archive_bytes"
    if "hi_nerv_receiver_proof_missing" in blocker_set or "hi_nerv_receiver_proof_missing_or_not_passed" in blocker_set:
        return "run_receiver_proof"
    if "hi_nerv_full_video_local_prefilter_missing" in blocker_set or "full_video_mlx_scorer_replay_not_attached" in blocker_set:
        return "run_full_video_mlx_prefilter"
    if "hi_nerv_local_cpu_replay_gate_missing" in blocker_set or "local_cpu_replay_not_run_partial_pair_coverage" in blocker_set:
        return "run_local_cpu_replay_gate"
    if readiness_score < 5:
        return "harvest_missing_readiness_evidence"
    return "queue_exact_eval_review_false_authority"


def _next_actions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("next_action") or "unknown")
        counts[action] = counts.get(action, 0) + 1
    return [
        {"action": action, "row_count": count}
        for action, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _best_acquisition_bytes(acquisition: Mapping[str, Any] | None) -> int | None:
    rows = acquisition.get("rows") if isinstance(acquisition, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    values = [
        value
        for row in rows
        if isinstance(row, Mapping)
        for value in (_first_int(row.get("effective_archive_bytes"), row.get("hprc_projection_bytes")),)
        if value is not None
    ]
    return min(values) if values else None


def _acquisition_next_actions(acquisition: Mapping[str, Any] | None) -> list[str]:
    rows = acquisition.get("rows") if isinstance(acquisition, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return _dedupe_strings(
        str(row.get("recommended_next_action"))
        for row in rows
        if isinstance(row, Mapping) and row.get("recommended_next_action")
    )


def _acquisition_blockers(acquisition: Mapping[str, Any] | None) -> list[str]:
    rows = acquisition.get("rows") if isinstance(acquisition, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    blockers: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("promotable") is False:
            blockers.append("hinerv_acquisition_row_not_promotable")
        for result in row.get("ceiling_results") or []:
            if isinstance(result, Mapping) and result.get("fits") is False:
                blockers.append("hinerv_acquisition_row_over_hard_ceiling")
    return _dedupe_strings(blockers)


def _variant_flags(name: str) -> dict[str, bool]:
    lowered = name.lower()
    return {
        key: any(token in lowered for token in tokens)
        for key, tokens in _VARIANT_TOKENS.items()
    }


def _public_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        key: value
        for key, value in row.items()
        if key != "embedded_candidate_feedback_row"
    }


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    archive_bytes = _first_int(row.get("archive_bytes"))
    return (
        0 if archive_bytes is not None else 1,
        archive_bytes if archive_bytes is not None else 10**18,
        -int(row.get("readiness_score") or 0),
        str(row.get("run_id") or ""),
    )


def _first_int(*values: object) -> int | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_sequence_int(value: object) -> int | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return _first_int(*value)


def _first_str(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def _first_bool(*values: object) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _mapping_get(mapping: Mapping[str, Any] | None, key: str) -> Any:
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def _nested_mapping_get(mapping: Mapping[str, Any] | None, path: Sequence[str | int]) -> Any:
    current: Any = mapping
    for item in path:
        if isinstance(item, int):
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                return None
            if item >= len(current):
                return None
            current = current[item]
            continue
        if not isinstance(current, Mapping):
            return None
        current = current.get(item)
    return current


def _string_items(value: object) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [str(item) for item in value if item is not None]


def _dedupe_strings(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_ARTIFACT_ROOT / f"hinerv_smoke_comparison_harvest_{stamp}_codex"


def _resolve(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (REPO_ROOT / expanded).resolve(strict=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        action="append",
        type=Path,
        default=[],
        help="Root containing hinerv_* result directories. May be repeated.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Directory glob under each artifact root. Default: hinerv_*.",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--feedback-refresh-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = _default_output_dir()
    output_json = _resolve(args.output_json) if args.output_json else output_dir / "hinerv_smoke_comparison_harvest.json"
    feedback_json = (
        _resolve(args.feedback_refresh_json)
        if args.feedback_refresh_json
        else output_json.parent / "hinerv_smoke_comparison_candidate_feedback_refresh.json"
    )
    output_md = _resolve(args.output_md) if args.output_md else output_json.parent / "hinerv_smoke_comparison_harvest.md"
    roots = tuple(args.artifact_root or [DEFAULT_ARTIFACT_ROOT])
    patterns = tuple(args.glob or ["hinerv_*"])
    report = build_hinerv_smoke_comparison(
        artifact_roots=roots,
        glob_patterns=patterns,
        limit=int(args.limit),
    )
    write = write_hinerv_smoke_comparison(
        report=report,
        output_json=output_json,
        feedback_refresh_json=feedback_json,
        output_md=output_md,
    )
    print(json.dumps(write, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
