# SPDX-License-Identifier: MIT
"""Execution preflight for TT5L side-info Lightning paired-axis cells.

This module deliberately does not submit provider work. It converts the
byte-closed Lightning paired-axis dry-run plan into operator-reviewable claim,
execution, and harvest templates so the next action is concrete without
collapsing dry-run custody into a score or promotion claim.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA,
)

L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_SCHEMA = (
    "l5_v2_tt5l_sideinfo_lightning_execution_preflight_v1"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_TOOL_PATH = (
    "tools/build_l5_v2_tt5l_sideinfo_lightning_execution_preflight.py"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_lightning_execution_preflight_20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_REPORT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_lightning_execution_preflight_20260517_codex.md"
)

_FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "dispatch_attempted": False,
}

_TERMINAL_PREFIXES = (
    "completed_",
    "failed_",
    "timed_out",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
    "falsified_",
    "retired_",
    "config_retired_",
    "measured_implementation_retired_",
    "stop_attempt_timeout_duplicate_after_primary_negative",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _lane_id_for_cell(*, variant: str, axis: str) -> str:
    return f"lane_l5_v2_tt5l_sideinfo_effect_curve_{_slug(variant)}_{_slug(axis)}"


def _axis_label(axis: str) -> str:
    return "[contest-CPU]" if axis == "contest_cpu" else "[contest-CUDA]"


def _q(value: str) -> str:
    return shlex.quote(value)


def _is_terminal_status(status: str) -> bool:
    return status.startswith(_TERMINAL_PREFIXES)


def _parse_claim_rows(claims_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in claims_text.splitlines():
        line = raw.strip()
        if (
            not line.startswith("|")
            or "timestamp_utc" in line
            or line.replace("|", "").replace("-", "").strip() == ""
        ):
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
        if len(cells) != 8:
            continue
        rows.append(
            {
                "timestamp_utc": cells[0],
                "agent": cells[1],
                "lane_id": cells[2],
                "platform": cells[3],
                "instance_job_id": cells[4],
                "predicted_eta_utc": cells[5],
                "status": cells[6],
                "notes": cells[7],
            }
        )
    return rows


def _active_claim_conflicts(
    *,
    claims_text: str,
    lane_id: str,
    platform: str,
) -> list[dict[str, str]]:
    conflicts: list[dict[str, str]] = []
    for row in _parse_claim_rows(claims_text):
        if row.get("lane_id") != lane_id:
            continue
        if row.get("platform") != platform:
            continue
        status = row.get("status") or ""
        if not _is_terminal_status(status):
            conflicts.append(row)
    return conflicts


def _plan_cell_rows(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in _mapping_rows(plan.get("cells"))
        if str(row.get("variant") or "") in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        and str(row.get("axis") or "") in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
    ]


def _cell_command_templates(
    *,
    lane_id: str,
    job_name: str,
    variant: str,
    axis: str,
    pair_group_id: str,
    run_id: str,
    archive_sha256: str,
    archive_size_bytes: int | None,
    local_artifact_dir: str,
) -> dict[str, str]:
    notes = (
        "tt5l_sideinfo_effect_curve;"
        f"variant={variant};axis={axis};pair_group_id={pair_group_id};"
        f"run_id={run_id};archive_sha256={archive_sha256};"
        f"archive_bytes={archive_size_bytes if archive_size_bytes is not None else ''};"
        "score_claim=false"
    )
    claim = (
        ".venv/bin/python tools/claim_lane_dispatch.py claim "
        f"--lane-id {_q(lane_id)} --platform lightning "
        f"--instance-job-id {_q(job_name)} "
        "--agent codex:l5_v2_tt5l_sideinfo_execution_preflight "
        "--status active_dispatching "
        f"--notes {_q(notes)}"
    )
    success = (
        ".venv/bin/python tools/claim_lane_dispatch.py claim --force "
        f"--lane-id {_q(lane_id)} --platform lightning "
        f"--instance-job-id {_q(job_name)} "
        "--agent codex:l5_v2_tt5l_sideinfo_execution_preflight "
        "--status completed_lightning_exact_eval_harvested "
        f"--notes {_q(notes + ';result=<contest_auth_eval.json>')}"
    )
    failure = (
        ".venv/bin/python tools/claim_lane_dispatch.py claim --force "
        f"--lane-id {_q(lane_id)} --platform lightning "
        f"--instance-job-id {_q(job_name)} "
        "--agent codex:l5_v2_tt5l_sideinfo_execution_preflight "
        "--status failed_lightning_exact_eval_no_score_claim "
        f"--notes {_q(notes + ';failure_class=<failure_class>')}"
    )
    harvest = (
        "test -f "
        f"{_q(str(Path(local_artifact_dir) / 'contest_auth_eval.json'))} "
        "&& .venv/bin/python "
        "tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py "
        "--lightning-plan-json "
        f"{_q(L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH)} "
        "--output-json "
        ".omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json "
        "--repo-root ."
    )
    return {
        "claim_command": claim,
        "terminal_success_claim_template": success,
        "terminal_failure_claim_template": failure,
        "harvest_probe_command_template": harvest,
    }


def build_l5_v2_tt5l_sideinfo_lightning_execution_preflight(
    *,
    plan: Mapping[str, Any],
    plan_path: str | Path,
    repo_root: str | Path,
    claims_text: str = "",
    current_head_commit: str = "",
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed execution preflight for the TT5L Lightning cells."""

    root = Path(repo_root).resolve()
    resolved_plan_path = _resolve_repo_path(plan_path, root)
    plan_sha256 = _sha256_file(resolved_plan_path) if resolved_plan_path.is_file() else ""
    generated = generated_at_utc or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    blockers: list[str] = []
    if plan.get("schema") != L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA:
        blockers.append("source_lightning_paired_axis_plan_schema_mismatch")
    if plan.get("all_cells_dry_run_ready") is not True:
        blockers.append("source_lightning_paired_axis_plan_cells_not_dry_run_ready")
    if plan.get("ready_for_provider_dispatch") is not False:
        blockers.append("source_lightning_paired_axis_plan_provider_ready_not_false")
    for field in _FALSE_AUTHORITY_FLAGS:
        if plan.get(field) is not False:
            blockers.append(f"source_lightning_paired_axis_plan_{field}_not_false")
    source_plan_commit = str(plan.get("source_commit") or "").strip()
    current_commit = str(current_head_commit or "").strip()
    source_claim_blockers: list[str] = []
    if not source_plan_commit:
        source_claim_blockers.append("source_plan_commit_missing")
    elif not current_commit:
        source_claim_blockers.append("current_head_commit_missing")
    elif source_plan_commit != current_commit:
        source_claim_blockers.append("source_plan_commit_mismatch_current_head")
    blockers.extend(source_claim_blockers)

    cells: list[dict[str, Any]] = []
    cell_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in _plan_cell_rows(plan):
        variant = str(cell.get("variant") or "").strip()
        axis = str(cell.get("axis") or "").strip()
        if variant and axis:
            cell_by_key[(variant, axis)] = cell
    missing_cells = [
        f"{variant}:{axis}"
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        if (variant, axis) not in cell_by_key
    ]
    if missing_cells:
        blockers.append("execution_preflight_missing_plan_cells:" + ",".join(missing_cells))

    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            source_cell = cell_by_key.get((variant, axis), {})
            lane_id = _lane_id_for_cell(variant=variant, axis=axis)
            job_name = str(source_cell.get("job_name") or "").strip()
            pair_group_id = str(source_cell.get("pair_group_id") or "").strip()
            run_id = str(source_cell.get("run_id") or "").strip()
            archive_sha256 = str(source_cell.get("archive_sha256") or "").strip()
            archive_size = source_cell.get("archive_size_bytes")
            archive_size_bytes = archive_size if isinstance(archive_size, int) else None
            local_artifact_dir = str(source_cell.get("local_artifact_dir") or "").strip()
            spec = source_cell.get("spec") if isinstance(source_cell.get("spec"), Mapping) else {}
            command_sha256 = str(source_cell.get("command_sha256") or "").strip()
            cell_blockers: list[str] = list(source_claim_blockers)
            if not source_cell:
                cell_blockers.append("source_cell_missing")
            if source_cell.get("ready_for_operator_dispatch") is not True:
                cell_blockers.append("source_cell_not_ready_for_operator_dispatch")
            if source_cell.get("ready_for_provider_dispatch") is not False:
                cell_blockers.append("source_cell_provider_ready_not_false")
            for field_name, value in {
                "job_name": job_name,
                "pair_group_id": pair_group_id,
                "run_id": run_id,
                "archive_sha256": archive_sha256,
                "local_artifact_dir": local_artifact_dir,
                "command_sha256": command_sha256,
            }.items():
                if not value:
                    cell_blockers.append(f"{field_name}_missing")
            if archive_size_bytes is None:
                cell_blockers.append("archive_size_bytes_missing")
            conflicts = _active_claim_conflicts(
                claims_text=claims_text,
                lane_id=lane_id,
                platform="lightning",
            )
            if conflicts:
                cell_blockers.append("active_lane_claim_conflict")
            commands = _cell_command_templates(
                lane_id=lane_id,
                job_name=job_name or f"l5-v2-tt5l-sideinfo-{variant}-{axis}",
                variant=variant,
                axis=axis,
                pair_group_id=pair_group_id,
                run_id=run_id,
                archive_sha256=archive_sha256,
                archive_size_bytes=archive_size_bytes,
                local_artifact_dir=local_artifact_dir,
            )
            cells.append(
                {
                    **_FALSE_AUTHORITY_FLAGS,
                    "variant": variant,
                    "axis": axis,
                    "axis_label": _axis_label(axis),
                    "lane_id": lane_id,
                    "platform": "lightning",
                    "job_name": job_name,
                    "role": source_cell.get("role"),
                    "required_device": source_cell.get("required_device"),
                    "archive_sha256": archive_sha256,
                    "archive_size_bytes": archive_size_bytes,
                    "pair_group_id": pair_group_id,
                    "run_id": run_id,
                    "local_artifact_dir": local_artifact_dir,
                    "expected_result_json": (
                        str(Path(local_artifact_dir) / "contest_auth_eval.json")
                        if local_artifact_dir
                        else ""
                    ),
                    "expected_adjudicated_json": (
                        str(Path(local_artifact_dir) / "contest_auth_eval.adjudicated.json")
                        if local_artifact_dir
                        else ""
                    ),
                    "source_spec_name": spec.get("name"),
                    "source_spec_role": spec.get("role"),
                    "source_spec_command_sha256": command_sha256,
                    "source_spec_command_is_authoritative": True,
                    "claim_command": commands["claim_command"],
                    "terminal_success_claim_template": commands[
                        "terminal_success_claim_template"
                    ],
                    "terminal_failure_claim_template": commands[
                        "terminal_failure_claim_template"
                    ],
                    "harvest_probe_command_template": commands[
                        "harvest_probe_command_template"
                    ],
                    "active_claim_conflicts": conflicts,
                    "ready_for_operator_claiming": not cell_blockers,
                    "ready_for_provider_dispatch": False,
                    "blockers": list(dict.fromkeys(cell_blockers)),
                }
            )
            blockers.extend(cell_blockers)

    ready_cells = sum(1 for cell in cells if cell["ready_for_operator_claiming"] is True)
    return {
        **_FALSE_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_SCHEMA,
        "tool": L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_TOOL_PATH,
        "generated_at_utc": generated,
        "source_plan": _repo_relative(resolved_plan_path, root),
        "source_plan_sha256": plan_sha256,
        "source_plan_schema": plan.get("schema"),
        "source_plan_generated_at_utc": plan.get("generated_at_utc"),
        "source_plan_commit": source_plan_commit,
        "current_head_commit": current_commit,
        "source_plan_commit_matches_current_head": bool(
            source_plan_commit and current_commit and source_plan_commit == current_commit
        ),
        "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "cell_count": len(cells),
        "expected_cell_count": (
            len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
            * len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
        ),
        "ready_cell_count": ready_cells,
        "cells": cells,
        "ready_for_operator_claiming": ready_cells == len(cells) and not blockers,
        "ready_for_provider_dispatch": False,
        "execution_order": [
            "verify_lightning_identity_and_workspace",
            "stage_source_manifest_to_lightning_workspace",
            "claim_each_per_axis_lane",
            "submit_each_source_plan_spec_command",
            "harvest_contest_auth_eval_json_for_each_cell",
            "write_terminal_claim_for_each_cell",
            "build_harvest_cells_artifact",
            "build_sideinfo_effect_curve_artifact",
            "refresh_l5_v2_architecture_lock_packet",
        ],
        "global_blockers": [
            "requires_lightning_identity_and_workspace_preflight_before_submit",
            "requires_source_manifest_staged_to_lightning_workspace_before_submit",
            "requires_operator_to_submit_source_plan_spec_commands",
            "requires_harvested_contest_cpu_and_contest_cuda_cells_before_sideinfo_effect_claim",
            "score_claim_forbidden_until_effect_curve_artifact_passes",
        ],
        "blockers": list(dict.fromkeys(blockers)),
    }


def l5_v2_tt5l_sideinfo_lightning_execution_preflight_json(
    payload: Mapping[str, Any],
) -> str:
    """Return canonical JSON text for the execution-preflight artifact."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_sideinfo_lightning_execution_preflight_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render a compact operator preflight memo for the TT5L Lightning cells."""

    lines = [
        "# L5 v2 TT5L side-info Lightning execution preflight",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        "",
        "This memo is an execution preflight, not a provider submission. It maps "
        "the byte-closed Lightning paired-axis dry-run plan into per-cell lane "
        "claims, terminal claim templates, and harvest checks for five variants "
        "times `[contest-CPU]` and `[contest-CUDA]`.",
        "",
        "## Status",
        "",
        f"- Source plan: `{payload.get('source_plan')}`",
        f"- Source plan SHA-256: `{payload.get('source_plan_sha256')}`",
        f"- Source plan commit: `{payload.get('source_plan_commit')}`",
        f"- Current head commit: `{payload.get('current_head_commit')}`",
        f"- Cells ready for operator claiming: `{payload.get('ready_cell_count')}`/`{payload.get('cell_count')}`",
        f"- ready_for_operator_claiming: `{payload.get('ready_for_operator_claiming')}`",
        "- ready_for_provider_dispatch: `false`",
        "- dispatch_attempted: `false`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        f"- Blockers: `{payload.get('blockers', [])}`",
        f"- Global blockers: `{payload.get('global_blockers', [])}`",
        "",
        "## Cells",
        "",
        "| variant | axis | lane_id | job_name | ready | archive bytes | archive SHA-256 |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for cell in _mapping_rows(payload.get("cells")):
        lines.append(
            f"| `{cell.get('variant')}` | `{cell.get('axis_label')}` | "
            f"`{cell.get('lane_id')}` | `{cell.get('job_name')}` | "
            f"`{cell.get('ready_for_operator_claiming')}` | "
            f"`{cell.get('archive_size_bytes')}` | `{cell.get('archive_sha256')}` |"
        )
    lines.extend(
        [
            "",
            "## Operator Command Templates",
            "",
            "Run claim commands before any non-dry-run provider submit. Submit the "
            "source plan cell's `spec.command` after Lightning identity and "
            "workspace staging are verified; the command itself remains in the "
            "source plan JSON and is identified here by SHA-256 to avoid duplicating "
            "large shell payloads in prose.",
            "",
        ]
    )
    for cell in _mapping_rows(payload.get("cells")):
        lines.extend(
            [
                f"### `{cell.get('variant')}` `{cell.get('axis_label')}`",
                "",
                f"- source_spec_command_sha256: `{cell.get('source_spec_command_sha256')}`",
                f"- expected_result_json: `{cell.get('expected_result_json')}`",
                f"- claim_command: `{cell.get('claim_command')}`",
                f"- terminal_success_claim_template: `{cell.get('terminal_success_claim_template')}`",
                f"- terminal_failure_claim_template: `{cell.get('terminal_failure_claim_template')}`",
                f"- harvest_probe_command_template: `{cell.get('harvest_probe_command_template')}`",
                f"- blockers: `{cell.get('blockers', [])}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_ARTIFACT_PATH",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_REPORT_PATH",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_SCHEMA",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_TOOL_PATH",
    "build_l5_v2_tt5l_sideinfo_lightning_execution_preflight",
    "l5_v2_tt5l_sideinfo_lightning_execution_preflight_json",
    "render_l5_v2_tt5l_sideinfo_lightning_execution_preflight_markdown",
]
