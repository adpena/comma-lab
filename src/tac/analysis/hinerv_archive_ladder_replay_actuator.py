# SPDX-License-Identifier: MIT
"""Actuate HiNeRV decoder-waterfill archive replay commands."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.analysis.hinerv_archive_ladder_waterfill import (
    HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA,
)
from tac.analysis.hinerv_archive_size_ladder import (
    HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
)
from tac.repo_io import sha256_file
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA = (
    "hinerv_archive_ladder_replay_actuator.v1"
)
HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_AUTHORITY = (
    "false_authority_replay_actuator_no_scorer_claim"
)
HINERV_ARCHIVE_SIZE_LADDER_TOOL = "tools/build_hinerv_archive_size_ladder.py"
DEFAULT_REPLAY_TIMEOUT_SECONDS = 900
DEFAULT_ARTIFACT_DIR = ".omx/research"

ReplayRunner = Callable[[Sequence[str], Path, int | None], Mapping[str, Any]]


class HinervArchiveLadderReplayActuatorError(ValueError):
    """Raised when a replay actuator input is malformed."""


def build_hinerv_archive_ladder_replay_actuator_report(
    waterfill_report: Mapping[str, Any],
    *,
    row_ids: Iterable[str] | None = None,
    execute: bool = False,
    cwd: str | Path = ".",
    timeout_seconds: int | None = DEFAULT_REPLAY_TIMEOUT_SECONDS,
    replay_output_root: str | Path | None = None,
    artifact_tag: str | None = None,
    load_existing: bool = False,
    allow_non_ssd_output: bool = False,
    runner: ReplayRunner | None = None,
) -> dict[str, Any]:
    """Plan or execute HiNeRV archive replay commands embedded in waterfill rows.

    Execution is intentionally still false-authority: the replay command can
    prove receiver-shaped archive bytes and local receiver consumption, but it
    does not attach contest CPU/CUDA scorer distortion.
    """

    if waterfill_report.get("schema") != HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA:
        raise HinervArchiveLadderReplayActuatorError(
            "expected hinerv_archive_ladder_waterfill.v1 report"
        )
    repo_root = Path(cwd).expanduser().resolve(strict=False)
    selected = {str(row_id) for row_id in row_ids} if row_ids is not None else None
    rows_by_id = {
        str(row.get("row_id")): row
        for row in waterfill_report.get("rows") or ()
        if isinstance(row, Mapping) and row.get("row_id") is not None
    }
    missing = sorted(selected - set(rows_by_id)) if selected is not None else []
    rows: list[dict[str, Any]] = []
    for row_id, row in rows_by_id.items():
        if selected is not None and row_id not in selected:
            continue
        rows.append(
            _plan_or_execute_row(
                row,
                row_id=row_id,
                execute=bool(execute),
                cwd=repo_root,
                timeout_seconds=timeout_seconds,
                replay_output_root=replay_output_root,
                artifact_tag=artifact_tag,
                load_existing=load_existing,
                allow_non_ssd_output=allow_non_ssd_output,
                runner=runner or _subprocess_runner,
            )
        )
    blockers = [
        "contest_cpu_cuda_exact_eval_not_executed",
        "hinerv_archive_ladder_replay_false_authority_no_nonrate_score",
    ]
    if missing:
        blockers.append("hinerv_archive_ladder_replay_requested_rows_missing")
    blockers.extend(
        blocker
        for row in rows
        for blocker in row.get("blockers", ())
        if blocker
    )
    return {
        "schema": HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA,
        "source_schema": waterfill_report.get("schema"),
        "authority": HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_AUTHORITY,
        "axis_tag": "[planning/control:false-authority]",
        "family": "hi_nerv",
        "candidate_id": waterfill_report.get("candidate_id"),
        "source_report_path": waterfill_report.get("report_path"),
        "execution_requested": bool(execute),
        "load_existing_requested": bool(load_existing),
        "cwd": repo_root.as_posix(),
        "timeout_seconds": timeout_seconds,
        "replay_output_root": (
            None
            if replay_output_root is None
            else Path(replay_output_root).expanduser().as_posix()
        ),
        "artifact_tag": artifact_tag,
        "allow_non_ssd_output": bool(allow_non_ssd_output),
        "requested_row_ids": sorted(selected) if selected is not None else None,
        "missing_requested_row_ids": missing,
        "row_count": len(rows),
        "execute_ready_row_count": sum(
            1 for row in rows if row.get("execute_ready") is True
        ),
        "executed_row_count": sum(1 for row in rows if row.get("executed") is True),
        "loaded_replay_report_count": sum(
            1 for row in rows if row.get("replay_report_loaded") is True
        ),
        "receiver_proof_ready_row_count": sum(
            1 for row in rows if row.get("receiver_proof_ready") is True
        ),
        "archive_bytes_by_row_id": {
            str(row["row_id"]): int(row["archive_bytes"])
            for row in rows
            if _int_or_none(row.get("archive_bytes")) is not None
        },
        "rows": rows,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def render_hinerv_archive_ladder_replay_actuator_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing replay actuator summary."""

    lines = [
        "# HiNeRV archive ladder replay actuator",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Execution requested: `{report.get('execution_requested')}`",
        "",
        "| row | status | bytes | proof ready | blockers |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report.get("rows") or ():
        lines.append(
            "| {row_id} | {status} | {bytes} | {proof} | {blockers} |".format(
                row_id=row.get("row_id"),
                status=row.get("status"),
                bytes=row.get("archive_bytes") or 0,
                proof=row.get("receiver_proof_ready"),
                blockers=len(row.get("blockers") or ()),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers") or ():
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _plan_or_execute_row(
    row: Mapping[str, Any],
    *,
    row_id: str,
    execute: bool,
    cwd: Path,
    timeout_seconds: int | None,
    replay_output_root: str | Path | None,
    artifact_tag: str | None,
    load_existing: bool,
    allow_non_ssd_output: bool,
    runner: ReplayRunner,
) -> dict[str, Any]:
    command = _string_list(row.get("archive_ladder_replay_command_argv"))
    command = _rewrite_command(
        command,
        row_id=row_id,
        replay_output_root=replay_output_root,
        artifact_tag=artifact_tag,
    )
    output_dir = _flag_value(command, "--output-dir") or str(
        row.get("archive_ladder_replay_output_dir") or ""
    )
    output_json = _flag_value(command, "--output-json")
    output_md = _flag_value(command, "--output-md")
    blockers = _command_blockers(
        command,
        output_dir=output_dir,
        output_json=output_json,
        allow_non_ssd_output=allow_non_ssd_output,
    )
    result: dict[str, Any] = {
        "row_id": row_id,
        "status": "planned" if not execute else "blocked_not_executed",
        "execute_ready": not blockers,
        "executed": False,
        "replay_report_loaded": False,
        "receiver_proof_ready": False,
        "archive_bytes": None,
        "archive_sha256": None,
        "archive_path": None,
        "archive_export_backend_counts": {},
        "command_argv": command,
        "command_hint": " ".join(command) if command else None,
        "output_dir": output_dir or None,
        "output_json": output_json,
        "output_md": output_md,
        "blockers": list(blockers),
        **FALSE_AUTHORITY,
    }
    if blockers:
        return result
    if not execute:
        if load_existing:
            output_path = _resolve_path(output_json, cwd)
            if output_path is not None and output_path.is_file():
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                _bind_replay_payload(
                    result,
                    payload,
                    output_path=output_path,
                    row_id=row_id,
                )
                result["status"] = "existing_report_loaded_false_authority"
            else:
                result["blockers"] = _ordered_unique(
                    [
                        *result["blockers"],
                        "hinerv_archive_ladder_replay_existing_output_json_missing",
                    ]
                )
        return result

    run = dict(runner(command, cwd, timeout_seconds))
    result["executed"] = True
    result["status"] = "executed"
    result["returncode"] = _int_or_none(run.get("returncode"))
    result["stdout_tail"] = run.get("stdout_tail")
    result["stderr_tail"] = run.get("stderr_tail")
    if result["returncode"] != 0:
        result["status"] = "failed"
        result["blockers"] = _ordered_unique(
            [*result["blockers"], "hinerv_archive_ladder_replay_command_failed"]
        )
        return result

    output_path = _resolve_path(output_json, cwd)
    if output_path is None or not output_path.is_file():
        result["status"] = "failed"
        result["blockers"] = _ordered_unique(
            [
                *result["blockers"],
                "hinerv_archive_ladder_replay_output_json_missing",
            ]
        )
        return result

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        result["status"] = "failed"
        result["blockers"] = _ordered_unique(
            [
                *result["blockers"],
                "hinerv_archive_ladder_replay_output_json_unreadable",
            ]
        )
        return result
    _bind_replay_payload(result, payload, output_path=output_path, row_id=row_id)
    return result


def _bind_replay_payload(
    result: dict[str, Any],
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    row_id: str,
) -> None:
    blockers = list(result["blockers"])
    if payload.get("schema") != HINERV_ARCHIVE_SIZE_LADDER_SCHEMA:
        blockers.append("hinerv_archive_ladder_replay_output_schema_unexpected")
    archive_rows = [
        row for row in payload.get("archive_rows") or () if isinstance(row, Mapping)
    ]
    matching = [
        row for row in archive_rows if str(row.get("row_id") or "") == str(row_id)
    ]
    selected = matching[0] if matching else (archive_rows[0] if archive_rows else None)
    if selected is None:
        blockers.append("hinerv_archive_ladder_replay_output_archive_row_missing")
    else:
        result["archive_bytes"] = _int_or_none(selected.get("archive_bytes"))
        result["archive_sha256"] = selected.get("archive_sha256")
        result["archive_path"] = selected.get("archive_path")
        result["submission_dir"] = selected.get("submission_dir")
        result["spine_manifest_path"] = selected.get("spine_manifest_path")
        result["receiver_proof_path"] = selected.get("receiver_proof_path")
        result["decoder_weight_waterfill_plan_path"] = selected.get(
            "decoder_weight_waterfill_plan_path"
        )
        result["receiver_proof_ready"] = (
            selected.get("runtime_consumption_proof_ready") is True
        )
        result["row_report_blockers"] = list(selected.get("blockers") or ())
        blockers.extend(selected.get("blockers") or ())
        if selected.get("runtime_consumption_proof_ready") is not True:
            blockers.append("hinerv_archive_ladder_replay_receiver_proof_not_ready")
    blockers.extend(payload.get("blockers") or ())
    result["status"] = "executed_report_loaded_false_authority"
    result["replay_report_loaded"] = True
    result["replay_report_path"] = output_path.as_posix()
    result["replay_report_sha256"] = sha256_file(output_path)
    result["replay_report_row_count"] = len(archive_rows)
    result["archive_export_backend_counts"] = dict(
        payload.get("archive_export_backend_counts") or {}
    )
    result["blockers"] = _ordered_unique(blockers)


def _command_blockers(
    command: Sequence[str],
    *,
    output_dir: str,
    output_json: str | None,
    allow_non_ssd_output: bool,
) -> list[str]:
    blockers: list[str] = []
    if not command:
        blockers.append("hinerv_archive_ladder_replay_command_missing")
        return blockers
    if not any(part.endswith(HINERV_ARCHIVE_SIZE_LADDER_TOOL) for part in command):
        blockers.append("hinerv_archive_ladder_replay_command_tool_unexpected")
    for flag, blocker in (
        ("--output-dir", "hinerv_archive_ladder_replay_output_dir_missing"),
        ("--output-json", "hinerv_archive_ladder_replay_output_json_arg_missing"),
        ("--row-id", "hinerv_archive_ladder_replay_row_id_arg_missing"),
        ("--decoder-codec", "hinerv_archive_ladder_replay_decoder_codec_missing"),
        (
            "--decoder-weight-saliency-json",
            "hinerv_archive_ladder_replay_saliency_arg_missing",
        ),
    ):
        if flag not in command:
            blockers.append(blocker)
    for flag, blocker in (
        (
            "--emit-receiver-proof",
            "hinerv_archive_ladder_replay_receiver_proof_arg_missing",
        ),
        (
            "--emit-decoder-weight-waterfill-plan",
            "hinerv_archive_ladder_replay_waterfill_plan_arg_missing",
        ),
    ):
        if flag not in command:
            blockers.append(blocker)
    if output_json is None:
        blockers.append("hinerv_archive_ladder_replay_output_json_missing")
    if output_dir and not allow_non_ssd_output and not output_dir.startswith("/Volumes/"):
        blockers.append("hinerv_archive_ladder_replay_output_dir_not_ssd_backed")
    return _ordered_unique(blockers)


def _rewrite_command(
    command: Sequence[str],
    *,
    row_id: str,
    replay_output_root: str | Path | None,
    artifact_tag: str | None,
) -> list[str]:
    out = list(command)
    if replay_output_root is not None:
        replay_root = Path(replay_output_root).expanduser()
        out = _set_flag(out, "--output-dir", (replay_root / _slug(row_id)).as_posix())
    if artifact_tag:
        slug = _slug(row_id)
        tag = _slug(artifact_tag)
        artifact_dir = Path(DEFAULT_ARTIFACT_DIR)
        out = _set_flag(
            out,
            "--output-json",
            (artifact_dir / f"hinerv_archive_size_ladder_replay_{tag}_{slug}_false_authority.json").as_posix(),
        )
        out = _set_flag(
            out,
            "--output-md",
            (artifact_dir / f"hinerv_archive_size_ladder_replay_{tag}_{slug}_false_authority.md").as_posix(),
        )
    return out


def _set_flag(command: Sequence[str], flag: str, value: str) -> list[str]:
    out = list(command)
    if flag in out:
        index = out.index(flag)
        if index + 1 < len(out):
            out[index + 1] = value
            return out
    out.extend([flag, value])
    return out


def _flag_value(command: Sequence[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = list(command).index(flag)
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def _resolve_path(path: str | None, cwd: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve(strict=False)


def _subprocess_runner(
    command: Sequence[str],
    cwd: Path,
    timeout_seconds: int | None,
) -> Mapping[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "returncode": int(completed.returncode),
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _tail(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ordered_unique(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _slug(value: str) -> str:
    text = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value)
    ).strip("_")
    return text or "row"


__all__ = [
    "DEFAULT_REPLAY_TIMEOUT_SECONDS",
    "HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA",
    "HinervArchiveLadderReplayActuatorError",
    "build_hinerv_archive_ladder_replay_actuator_report",
    "render_hinerv_archive_ladder_replay_actuator_markdown",
]
