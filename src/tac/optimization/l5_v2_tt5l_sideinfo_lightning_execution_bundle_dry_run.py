# SPDX-License-Identifier: MIT
"""Dry-run verifier for the TT5L side-info Lightning execution bundle.

The execution bundle is a command generator. This verifier exercises each
generated dry-run command through the real Lightning launch parser and records
the custody invariants that survive into the queued exact-eval spec. It is
deliberately false-authority-safe: passing dry-runs never means provider work
was dispatched or a score can be claimed.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
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
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA,
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_TOOL_PATH,
    T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV,
)

L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_SCHEMA = (
    "l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_v1"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_TOOL_PATH = (
    "tools/verify_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_"
    "20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_REPORT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_"
    "20260517_codex.md"
)

_FALSE_AUTHORITY_FLAGS = {
    "score_claim_valid": False,
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "ready_for_provider_dispatch": False,
    "dispatch_attempted": False,
    "provider_spend_attempted": False,
}
_PLANNING_AUTHORITY_FLAGS = {
    "planning_only": True,
    **_FALSE_AUTHORITY_FLAGS,
}
_SOURCE_FALSE_AUTHORITY_FIELDS = tuple(
    field for field in _FALSE_AUTHORITY_FLAGS if field != "provider_spend_attempted"
)


@dataclass(frozen=True)
class DryRunCommandResult:
    """Captured result from one bundle dry-run command."""

    returncode: int
    stdout: str
    stderr: str
    argv: tuple[str, ...]
    timed_out: bool = False


DryRunRunner = Callable[[str, Path, int], DryRunCommandResult]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


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


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _expected_axis_label(axis: str) -> str:
    return "[contest-CPU]" if axis == "contest_cpu" else "[contest-CUDA]"


def _expected_eval_device(axis: str) -> str:
    if axis == "contest_cpu":
        return "cpu"
    if axis == "contest_cuda":
        return "cuda"
    return ""


def _expected_role(axis: str) -> str:
    device = _expected_eval_device(axis)
    return f"exact_{device}_eval" if device else ""


def _arg_value(argv: Sequence[str], flag: str) -> str:
    try:
        idx = argv.index(flag)
    except ValueError:
        return ""
    if idx + 1 >= len(argv):
        return ""
    return argv[idx + 1]


def _has_arg(argv: Sequence[str], flag: str) -> bool:
    return flag in argv


def _machine_needs_runtime_env(machine: str) -> bool:
    machine_l = machine.lower()
    return "t4" in machine_l or "g4dn" in machine_l


def _missing_runtime_envs(command: str) -> list[str]:
    argv = shlex.split(command)
    machine = _arg_value(argv, "--machine")
    if not _machine_needs_runtime_env(machine):
        return []
    env_values = [
        argv[idx + 1]
        for idx, value in enumerate(argv[:-1])
        if value == "--env"
    ]
    return [
        expected
        for expected in T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV
        if expected not in env_values
    ]


def _inflate_runtime_status(
    *,
    argv: Sequence[str],
    repo_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    raw_path = _arg_value(argv, "--inflate-sh")
    if not raw_path:
        return {
            "path": "",
            "exists": False,
            "executable": False,
        }, ["dry_run_inflate_sh_arg_missing"]
    path = _resolve_repo_path(raw_path, repo_root)
    exists = path.is_file()
    executable = exists and bool(path.stat().st_mode & 0o111)
    if not exists:
        blockers.append("dry_run_inflate_sh_missing")
    elif not executable:
        blockers.append("dry_run_inflate_sh_not_executable")
    return {
        "path": _repo_relative(path, repo_root),
        "exists": exists,
        "executable": executable,
    }, blockers


def _run_dry_run_command(
    command: str,
    repo_root: Path,
    timeout_seconds: int,
) -> DryRunCommandResult:
    argv = tuple(shlex.split(command))
    try:
        proc = subprocess.run(
            argv,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return DryRunCommandResult(
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            argv=argv,
            timed_out=True,
        )
    return DryRunCommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        argv=argv,
    )


def _load_dry_run_record(stdout: str) -> tuple[Mapping[str, Any], list[str]]:
    blockers: list[str] = []
    if not stdout.strip():
        return {}, ["dry_run_stdout_empty"]
    try:
        loaded = json.loads(stdout)
    except json.JSONDecodeError:
        return {}, ["dry_run_stdout_json_invalid"]
    if isinstance(loaded, Mapping):
        return loaded, []
    if isinstance(loaded, list) and len(loaded) == 1 and isinstance(loaded[0], Mapping):
        return loaded[0], []
    blockers.append("dry_run_stdout_not_single_record")
    return {}, blockers


def _load_latest_state_record(
    *,
    argv: Sequence[str],
    repo_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any], list[str]]:
    blockers: list[str] = []
    raw_state_path = _arg_value(argv, "--state-path")
    if not raw_state_path:
        return {}, {"path": "", "exists": False, "checked": False}, [
            "dry_run_state_path_arg_missing"
        ]
    state_path = _resolve_repo_path(raw_state_path, repo_root)
    summary = {
        "path": _repo_relative(state_path, repo_root),
        "exists": state_path.is_file(),
        "checked": True,
        "latest_record_parsed": False,
        "stdout_core_matched": False,
    }
    if not state_path.is_file():
        return {}, summary, ["dry_run_state_file_missing"]
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, summary, ["dry_run_state_file_json_invalid"]
    if not isinstance(loaded, list):
        return {}, summary, ["dry_run_state_file_not_record_list"]
    rows = [row for row in loaded if isinstance(row, Mapping)]
    if not rows:
        return {}, summary, ["dry_run_state_file_no_mapping_records"]
    summary["latest_record_parsed"] = True
    return rows[-1], summary, blockers


def _state_stdout_core_blockers(
    *,
    state_record: Mapping[str, Any],
    stdout_record: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    for field in ("dry_run", "queue", "spec"):
        if state_record.get(field) != stdout_record.get(field):
            blockers.append(f"dry_run_state_stdout_{field}_mismatch")
    return blockers


def _cell_by_key(cells: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in cells:
        variant = str(cell.get("variant") or "").strip()
        axis = str(cell.get("axis") or "").strip()
        if variant and axis:
            out.setdefault((variant, axis), cell)
    return out


def _cell_key_coverage(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = {
        (variant, axis)
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
    }
    counts: dict[tuple[str, str], int] = {}
    key_missing_indices: list[int] = []
    for idx, cell in enumerate(cells):
        variant = str(cell.get("variant") or "").strip()
        axis = str(cell.get("axis") or "").strip()
        if not variant or not axis:
            key_missing_indices.append(idx)
            continue
        key = (variant, axis)
        counts[key] = counts.get(key, 0) + 1

    missing = [
        f"{variant}:{axis}"
        for variant, axis in sorted(required)
        if (variant, axis) not in counts
    ]
    duplicates = [
        f"{variant}:{axis}"
        for (variant, axis), count in sorted(counts.items())
        if (variant, axis) in required and count > 1
    ]
    extras = [
        f"{variant}:{axis}"
        for variant, axis in sorted(counts)
        if (variant, axis) not in required
    ]
    return {
        "missing_cells": missing,
        "duplicate_cells": duplicates,
        "extra_cells": extras,
        "key_missing_indices": key_missing_indices,
    }


def _load_json_mapping(path: Path) -> tuple[Mapping[str, Any], list[str]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ["json_invalid"]
    if not isinstance(loaded, Mapping):
        return {}, ["json_not_object"]
    return loaded, []


def _variant_rows_by_name(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in _mapping_rows(manifest.get("variants")):
        variant = str(row.get("variant") or "").strip()
        if variant:
            out[variant] = row
    return out


def _source_plan_status(
    *,
    bundle: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], Mapping[str, Any]], list[str]]:
    blockers: list[str] = []
    raw_path = str(bundle.get("source_plan") or "").strip()
    expected_sha = str(bundle.get("source_plan_sha256") or "").strip()
    summary: dict[str, Any] = {
        "path": raw_path,
        "exists": False,
        "sha256": "",
        "matches_bundle_sha256": False,
        "schema": "",
        "cell_count": 0,
        "coverage": {
            "missing_cells": [],
            "duplicate_cells": [],
            "extra_cells": [],
            "key_missing_indices": [],
        },
    }
    if not raw_path:
        return summary, {}, ["source_plan_path_missing"]
    path = _resolve_repo_path(raw_path, repo_root)
    summary["path"] = _repo_relative(path, repo_root)
    if not path.is_file():
        return summary, {}, ["source_plan_missing"]

    summary["exists"] = True
    actual_sha = _sha256_file(path)
    summary["sha256"] = actual_sha
    summary["matches_bundle_sha256"] = bool(expected_sha) and actual_sha == expected_sha
    if not expected_sha:
        blockers.append("source_plan_sha256_missing_from_bundle")
    elif actual_sha != expected_sha:
        blockers.append("source_plan_sha256_mismatch_bundle")

    plan, load_blockers = _load_json_mapping(path)
    blockers.extend(f"source_plan:{blocker}" for blocker in load_blockers)
    if load_blockers:
        return summary, {}, blockers
    summary["schema"] = plan.get("schema", "")
    if plan.get("schema") != L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA:
        blockers.append("source_plan_schema_mismatch")

    plan_cells = _mapping_rows(plan.get("cells"))
    summary["cell_count"] = len(plan_cells)
    coverage = _cell_key_coverage(plan_cells)
    summary["coverage"] = coverage
    if coverage["missing_cells"]:
        blockers.append(
            "source_plan_missing_cells:" + ",".join(coverage["missing_cells"])
        )
    if coverage["duplicate_cells"]:
        blockers.append(
            "source_plan_duplicate_cells:" + ",".join(coverage["duplicate_cells"])
        )
    if coverage["extra_cells"]:
        blockers.append("source_plan_extra_cells:" + ",".join(coverage["extra_cells"]))
    if coverage["key_missing_indices"]:
        blockers.append(
            "source_plan_cell_key_missing_indices:"
            + ",".join(str(idx) for idx in coverage["key_missing_indices"])
        )
    if len(plan_cells) != (
        len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
        * len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
    ):
        blockers.append("source_plan_cell_count_mismatch")
    return summary, _cell_by_key(plan_cells), blockers


def _source_plan_cell_summary(
    *,
    cell: Mapping[str, Any],
    source_plan_cell: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not source_plan_cell:
        return {
            "matched": False,
            "command_sha256": "",
        }, ["source_plan_cell_missing"]
    expected_source_sha = str(cell.get("source_spec_command_sha256") or "")
    plan_command_sha = str(source_plan_cell.get("command_sha256") or "").strip()
    if not plan_command_sha:
        blockers.append("source_plan_cell_command_sha256_missing")
    elif expected_source_sha != plan_command_sha:
        blockers.append("source_spec_command_sha256_mismatch_source_plan")
    fields = (
        "archive_sha256",
        "archive_size_bytes",
        "pair_group_id",
        "run_id",
        "local_artifact_dir",
    )
    for field in fields:
        if cell.get(field) != source_plan_cell.get(field):
            blockers.append(f"source_plan_cell_{field}_mismatch")
    return {
        "matched": not blockers,
        "command_sha256": plan_command_sha,
        "source_spec_command_sha256": expected_source_sha,
        "archive_sha256": str(source_plan_cell.get("archive_sha256") or ""),
        "archive_size_bytes": source_plan_cell.get("archive_size_bytes"),
        "pair_group_id": str(source_plan_cell.get("pair_group_id") or ""),
        "run_id": str(source_plan_cell.get("run_id") or ""),
        "local_artifact_dir": str(source_plan_cell.get("local_artifact_dir") or ""),
    }, blockers


def _local_archive_status(
    *,
    cell: Mapping[str, Any],
    variant_row: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    archive_path = str(cell.get("archive_path") or variant_row.get("archive_path") or "").strip()
    expected_sha = str(cell.get("archive_sha256") or "").strip()
    expected_bytes = cell.get("archive_size_bytes")
    manifest_sha = str(variant_row.get("archive_sha256") or "").strip()
    manifest_bytes = variant_row.get("archive_bytes")
    if not variant_row:
        blockers.append("variant_manifest_row_missing")
    if not archive_path:
        blockers.append("local_archive_path_missing")
        return {
            "path": "",
            "exists": False,
            "sha256": "",
            "bytes": None,
            "matches_cell": False,
            "matches_variant_manifest": False,
        }, blockers
    path = _resolve_repo_path(archive_path, repo_root)
    exists = path.is_file()
    actual_sha = _sha256_file(path) if exists else ""
    actual_bytes = path.stat().st_size if exists else None
    if not exists:
        blockers.append("local_archive_missing")
    if exists and actual_sha != expected_sha:
        blockers.append("local_archive_sha_mismatch_cell")
    if exists and actual_bytes != expected_bytes:
        blockers.append("local_archive_bytes_mismatch_cell")
    if manifest_sha and expected_sha and manifest_sha != expected_sha:
        blockers.append("variant_manifest_archive_sha_mismatch_cell")
    if (
        isinstance(manifest_bytes, int)
        and isinstance(expected_bytes, int)
        and manifest_bytes != expected_bytes
    ):
        blockers.append("variant_manifest_archive_bytes_mismatch_cell")
    if exists and manifest_sha and actual_sha != manifest_sha:
        blockers.append("local_archive_sha_mismatch_variant_manifest")
    if exists and isinstance(manifest_bytes, int) and actual_bytes != manifest_bytes:
        blockers.append("local_archive_bytes_mismatch_variant_manifest")
    return {
        "path": archive_path,
        "exists": exists,
        "sha256": actual_sha,
        "bytes": actual_bytes,
        "matches_cell": exists and actual_sha == expected_sha and actual_bytes == expected_bytes,
        "matches_variant_manifest": (
            exists
            and bool(manifest_sha)
            and actual_sha == manifest_sha
            and isinstance(manifest_bytes, int)
            and actual_bytes == manifest_bytes
        ),
    }, blockers


def _paired_axis_blockers(cells: Iterable[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        by_variant.setdefault(str(cell.get("variant") or ""), []).append(cell)
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        rows = {
            str(cell.get("axis") or ""): cell
            for cell in by_variant.get(variant, [])
            if isinstance(cell, Mapping)
        }
        if set(rows) != set(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES):
            blockers.append(f"paired_axis_cells_missing:{variant}")
            continue
        cpu = rows["contest_cpu"]
        cuda = rows["contest_cuda"]
        for field in ("archive_sha256", "archive_size_bytes", "run_id", "pair_group_id"):
            if cpu.get(field) != cuda.get(field):
                blockers.append(f"paired_axis_{field}_mismatch:{variant}")
    return blockers


def _validate_static_cell(cell: Mapping[str, Any], argv: Sequence[str]) -> list[str]:
    blockers: list[str] = []
    variant = str(cell.get("variant") or "").strip()
    axis = str(cell.get("axis") or "").strip()
    expected_device = _expected_eval_device(axis)
    expected_role = _expected_role(axis)
    expected_lane = str(cell.get("lane_id") or "").strip()

    if cell.get("planning_only") is not True:
        blockers.append("cell_not_planning_only")
    for field in _SOURCE_FALSE_AUTHORITY_FIELDS:
        if cell.get(field) is not False:
            blockers.append(f"cell_{field}_not_false")
    if cell.get("axis_label") != _expected_axis_label(axis):
        blockers.append("cell_axis_label_mismatch")
    if cell.get("eval_device") != expected_device:
        blockers.append("cell_eval_device_mismatch")
    if cell.get("ready_for_dry_run_submit") is not True:
        blockers.append("cell_not_ready_for_dry_run_submit")
    if cell.get("ready_for_non_dry_run_submit") is not False:
        blockers.append("cell_non_dry_run_ready")
    if cell.get("ready_for_provider_dispatch") is not False:
        blockers.append("cell_provider_dispatch_ready")
    if not variant:
        blockers.append("cell_variant_missing")
    if axis not in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
        blockers.append("cell_axis_unknown")
    if not expected_lane:
        blockers.append("cell_lane_id_missing")

    command = str(cell.get("dry_run_submit_command") or "")
    if not command:
        blockers.append("dry_run_submit_command_missing")
    if any(token in command for token in ("<lightning-", "<--", "< ")):
        blockers.append("dry_run_submit_command_contains_template_placeholder")
    if len(argv) < 4:
        blockers.append("dry_run_argv_too_short")
    elif argv[:3] != (".venv/bin/python", "scripts/launch_lightning_batch_job.py", "exact-eval"):
        blockers.append("dry_run_argv_not_lightning_exact_eval")
    if not _has_arg(argv, "--dry-run"):
        blockers.append("dry_run_flag_missing")
    if not _has_arg(argv, "--adjudicate"):
        blockers.append("adjudicate_flag_missing")
    if _arg_value(argv, "--eval-device") != expected_device:
        blockers.append("dry_run_eval_device_arg_mismatch")
    if _arg_value(argv, "--dispatch-lane-id") != expected_lane:
        blockers.append("dry_run_dispatch_lane_arg_mismatch")
    if _arg_value(argv, "--expected-archive-sha256") != str(
        cell.get("archive_sha256") or ""
    ):
        blockers.append("dry_run_expected_archive_sha_arg_mismatch")
    if _arg_value(argv, "--expected-archive-size-bytes") != str(
        cell.get("archive_size_bytes") or ""
    ):
        blockers.append("dry_run_expected_archive_bytes_arg_mismatch")
    if _arg_value(argv, "--local-artifact-dir") != str(
        cell.get("local_artifact_dir") or ""
    ):
        blockers.append("dry_run_local_artifact_dir_arg_mismatch")
    if not _arg_value(argv, "--state-path"):
        blockers.append("dry_run_state_path_arg_missing")
    source_manifest = _arg_value(argv, "--source-manifest")
    if not source_manifest or not source_manifest.endswith("/source_manifest.json"):
        blockers.append("dry_run_source_manifest_arg_missing")
    if expected_role and f"--eval-device {expected_device}" not in command:
        blockers.append("dry_run_command_missing_eval_device_text")
    for missing_env in _missing_runtime_envs(command):
        blockers.append(f"dry_run_t4_runtime_env_missing:{missing_env}")
    non_dry = str(cell.get("non_dry_run_submit_command_template") or "")
    if non_dry:
        for missing_env in _missing_runtime_envs(non_dry):
            blockers.append(f"non_dry_run_t4_runtime_env_missing:{missing_env}")
    return blockers


def _validate_queue_record(
    *,
    cell: Mapping[str, Any],
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    axis = str(cell.get("axis") or "").strip()
    expected_device = _expected_eval_device(axis)
    queue = record.get("queue")
    if not isinstance(queue, Mapping):
        return {}, ["dry_run_queue_missing"]
    metadata = queue.get("queue_metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
        blockers.append("dry_run_queue_metadata_missing")
    adjudication = queue.get("adjudication")
    if not isinstance(adjudication, Mapping):
        adjudication = {}
        blockers.append("dry_run_queue_adjudication_missing")
    spec = record.get("spec")
    if not isinstance(spec, Mapping):
        spec = {}
        blockers.append("dry_run_spec_missing")
    command_text = str(spec.get("command") or "")
    computed_command_sha = _sha256_text(command_text) if command_text else ""

    expected_archive_sha = str(cell.get("archive_sha256") or "")
    expected_archive_bytes = cell.get("archive_size_bytes")
    expected_lane = str(cell.get("lane_id") or "")
    expected_pair_group = str(cell.get("pair_group_id") or "")
    expected_run_id = str(cell.get("run_id") or "")
    expected_source_sha = str(cell.get("source_spec_command_sha256") or "")

    if record.get("dry_run") is not True:
        blockers.append("dry_run_record_not_marked_dry_run")
    if queue.get("role") != _expected_role(axis):
        blockers.append("dry_run_queue_role_mismatch")
    if queue.get("expected_archive_sha256") != expected_archive_sha:
        blockers.append("dry_run_queue_archive_sha_mismatch")
    if queue.get("expected_archive_size_bytes") != expected_archive_bytes:
        blockers.append("dry_run_queue_archive_bytes_mismatch")
    if queue.get("local_artifact_dir") != cell.get("local_artifact_dir"):
        blockers.append("dry_run_queue_local_artifact_dir_mismatch")
    if adjudication.get("required_device") != expected_device:
        blockers.append("dry_run_queue_required_device_mismatch")
    if adjudication.get("required_samples") != 600:
        blockers.append("dry_run_queue_required_samples_mismatch")

    expected_metadata = {
        "variant": str(cell.get("variant") or ""),
        "axis": axis,
        "lane_id": expected_lane,
        "pair_group_id": expected_pair_group,
        "run_id": expected_run_id,
        "archive_sha256": expected_archive_sha,
        "source_plan": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
        "source_spec_command_sha256": expected_source_sha,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            blockers.append(f"dry_run_queue_metadata_{key}_mismatch")

    if f"--device {expected_device}" not in command_text:
        blockers.append("dry_run_spec_command_device_missing")
    other_device = "cpu" if expected_device == "cuda" else "cuda"
    if f"--device {other_device}" in command_text:
        blockers.append("dry_run_spec_command_mixes_device_axes")
    if expected_device == "cuda":
        if "export INFLATE_REQUIRE_CUDA=1" not in command_text:
            blockers.append("dry_run_spec_cuda_inflate_requirement_missing")
        for marker in (
            "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK",
            "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK",
        ):
            if marker not in command_text:
                blockers.append(f"dry_run_spec_cuda_marker_missing:{marker}")
    elif expected_device == "cpu":
        if "export INFLATE_REQUIRE_CUDA=1" in command_text:
            blockers.append("dry_run_spec_cpu_requires_cuda_inflate")
        if "LIGHTNING_RUNNER_CPU_PREFLIGHT_OK" not in command_text:
            blockers.append("dry_run_spec_cpu_marker_missing")
        for marker in (
            "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK",
            "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK",
        ):
            if marker in command_text:
                blockers.append(f"dry_run_spec_cpu_forbidden_marker:{marker}")
    if "contest_auth_eval.json" not in command_text:
        blockers.append("dry_run_spec_command_missing_contest_auth_eval_json")
    if "scripts/scan_lightning_supply_chain.py" not in command_text:
        blockers.append("dry_run_spec_command_missing_supply_chain_scan")
    if str(queue.get("command_sha256") or "").strip() == "":
        blockers.append("dry_run_queue_command_sha_missing")
    elif computed_command_sha and queue.get("command_sha256") != computed_command_sha:
        blockers.append("dry_run_queue_command_sha_not_spec_command_sha")

    launcher_sha = str(queue.get("command_sha256") or "").strip()
    return (
        {
            "queue_role": queue.get("role"),
            "required_device": adjudication.get("required_device"),
            "required_samples": adjudication.get("required_samples"),
            "queue_command_sha256": launcher_sha,
            "computed_spec_command_sha256": computed_command_sha,
            "source_spec_command_sha256": expected_source_sha,
            "launcher_command_sha_matches_source_spec": (
                bool(launcher_sha) and launcher_sha == expected_source_sha
            ),
            "command_sha_delta_classification": (
                "identical"
                if launcher_sha and launcher_sha == expected_source_sha
                else "expected_submit_layer_delta"
            ),
            "command_sha_delta_rationale": (
                "The Lightning launcher wraps the source-plan command with "
                "source-manifest, dispatch-lane, adjudication, and queue metadata. "
                "Verifier authority comes from invariant checks, not command-SHA equality."
            ),
            "queue_metadata": dict(metadata),
            "spec_command_sha256": _sha256_text(command_text) if command_text else "",
            "spec_command_contains_device": f"--device {expected_device}" in command_text,
            "spec_command_contains_cuda_requirement": (
                "export INFLATE_REQUIRE_CUDA=1" in command_text
            ),
            "spec_command_contains_supply_chain_scan": (
                "scripts/scan_lightning_supply_chain.py" in command_text
            ),
        },
        blockers,
    )


def build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
    *,
    bundle: Mapping[str, Any],
    bundle_path: str | Path,
    repo_root: str | Path,
    runner: DryRunRunner | None = None,
    timeout_seconds: int = 60,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Execute and verify every dry-run submit command in a TT5L bundle."""

    root = Path(repo_root).resolve()
    bundle_file = _resolve_repo_path(bundle_path, root)
    bundle_sha256 = _sha256_file(bundle_file) if bundle_file.is_file() else ""
    run = runner or _run_dry_run_command
    generated = generated_at_utc or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    expected_cell_count = (
        len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
        * len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
    )
    global_blockers: list[str] = []
    variant_manifest_path = str(bundle.get("source_variant_manifest") or "").strip()
    variant_manifest: Mapping[str, Any] = {}
    variant_rows: dict[str, Mapping[str, Any]] = {}
    if variant_manifest_path:
        variant_manifest_file = _resolve_repo_path(variant_manifest_path, root)
        variant_manifest, manifest_blockers = _load_json_mapping(variant_manifest_file)
        global_blockers.extend(
            f"source_variant_manifest:{blocker}" for blocker in manifest_blockers
        )
        variant_rows = _variant_rows_by_name(variant_manifest)
    else:
        global_blockers.append("source_variant_manifest_path_missing")
    if bundle.get("schema") != L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA:
        global_blockers.append("source_bundle_schema_mismatch")
    if bundle.get("planning_only") is not True:
        global_blockers.append("source_bundle_not_planning_only")
    for field in _SOURCE_FALSE_AUTHORITY_FIELDS:
        if bundle.get(field) is not False:
            global_blockers.append(f"source_bundle_{field}_not_false")
    if bundle.get("ready_for_dry_run_submit") is not True:
        global_blockers.append("source_bundle_not_ready_for_dry_run_submit")
    if bundle.get("ready_for_non_dry_run_submit") is not False:
        global_blockers.append("source_bundle_non_dry_run_ready")
    if bundle.get("ready_for_provider_dispatch") is not False:
        global_blockers.append("source_bundle_provider_dispatch_ready")
    source_plan, source_plan_cells, source_plan_blockers = _source_plan_status(
        bundle=bundle,
        repo_root=root,
    )
    global_blockers.extend(source_plan_blockers)

    cells = _mapping_rows(bundle.get("cells"))
    cell_by_key = _cell_by_key(cells)
    cell_key_coverage = _cell_key_coverage(cells)
    missing_cells = list(cell_key_coverage["missing_cells"])
    if missing_cells:
        global_blockers.append("source_bundle_missing_cells:" + ",".join(missing_cells))
    duplicate_cells = list(cell_key_coverage["duplicate_cells"])
    if duplicate_cells:
        global_blockers.append(
            "source_bundle_duplicate_cells:" + ",".join(duplicate_cells)
        )
    extra_cells = list(cell_key_coverage["extra_cells"])
    if extra_cells:
        global_blockers.append("source_bundle_extra_cells:" + ",".join(extra_cells))
    key_missing_indices = list(cell_key_coverage["key_missing_indices"])
    if key_missing_indices:
        global_blockers.append(
            "source_bundle_cell_key_missing_indices:"
            + ",".join(str(idx) for idx in key_missing_indices)
        )
    if len(cells) != expected_cell_count:
        global_blockers.append("source_bundle_cell_count_mismatch")
    global_blockers.extend(_paired_axis_blockers(cells))

    verified_cells: list[dict[str, Any]] = []
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            cell = cell_by_key.get((variant, axis), {})
            command = str(cell.get("dry_run_submit_command") or "")
            argv = tuple(shlex.split(command)) if command else ()
            cell_blockers = _validate_static_cell(cell, argv) if cell else [
                "source_bundle_cell_missing"
            ]
            result = DryRunCommandResult(
                returncode=2,
                stdout="",
                stderr="",
                argv=argv,
            )
            record: Mapping[str, Any] = {}
            queue_summary: dict[str, Any] = {}
            parse_blockers: list[str] = []
            archive_summary: dict[str, Any] = {}
            state_summary: dict[str, Any] = {}
            inflate_runtime_summary: dict[str, Any] = {}
            source_plan_cell: dict[str, Any] = {}
            if cell:
                source_plan_cell, source_plan_cell_blockers = _source_plan_cell_summary(
                    cell=cell,
                    source_plan_cell=source_plan_cells.get((variant, axis), {}),
                )
                cell_blockers.extend(source_plan_cell_blockers)
                archive_summary, archive_blockers = _local_archive_status(
                    cell=cell,
                    variant_row=variant_rows.get(variant, {}),
                    repo_root=root,
                )
                cell_blockers.extend(archive_blockers)
                inflate_runtime_summary, runtime_blockers = _inflate_runtime_status(
                    argv=argv,
                    repo_root=root,
                )
                cell_blockers.extend(runtime_blockers)
            if command and not cell_blockers:
                result = run(command, root, timeout_seconds)
                if result.timed_out:
                    cell_blockers.append("dry_run_command_timed_out")
                if result.returncode != 0:
                    cell_blockers.append("dry_run_command_returncode_nonzero")
                if result.stderr.strip():
                    cell_blockers.append("dry_run_command_stderr_nonempty")
                record, parse_blockers = _load_dry_run_record(result.stdout)
                cell_blockers.extend(parse_blockers)
                if record:
                    state_record, state_summary, state_blockers = (
                        _load_latest_state_record(argv=result.argv, repo_root=root)
                    )
                    cell_blockers.extend(state_blockers)
                    if state_record:
                        state_core_blockers = _state_stdout_core_blockers(
                            state_record=state_record,
                            stdout_record=record,
                        )
                        if not state_core_blockers:
                            state_summary["stdout_core_matched"] = True
                        cell_blockers.extend(state_core_blockers)
                    queue_summary, queue_blockers = _validate_queue_record(
                        cell=cell,
                        record=record,
                    )
                    cell_blockers.extend(queue_blockers)
            verified_cells.append(
                {
                    **_PLANNING_AUTHORITY_FLAGS,
                    "variant": variant,
                    "axis": axis,
                    "axis_label": _expected_axis_label(axis),
                    "lane_id": str(cell.get("lane_id") or ""),
                    "archive_sha256": str(cell.get("archive_sha256") or ""),
                    "archive_size_bytes": cell.get("archive_size_bytes"),
                    "pair_group_id": str(cell.get("pair_group_id") or ""),
                    "run_id": str(cell.get("run_id") or ""),
                    "local_artifact_dir": str(cell.get("local_artifact_dir") or ""),
                    "local_archive": archive_summary,
                    "inflate_runtime": inflate_runtime_summary,
                    "source_plan_cell": source_plan_cell,
                    "dry_run_state_path": str(cell.get("dry_run_state_path") or ""),
                    "dry_run_state_file": state_summary,
                    "dry_run_command_sha256": _sha256_text(command) if command else "",
                    "dry_run_command_argv": list(result.argv),
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stdout_sha256": _sha256_text(result.stdout) if result.stdout else "",
                    "stderr_sha256": _sha256_text(result.stderr) if result.stderr else "",
                    "stdout_bytes": len(result.stdout.encode("utf-8")),
                    "stderr_bytes": len(result.stderr.encode("utf-8")),
                    "stderr_tail": result.stderr[-500:] if result.stderr else "",
                    "dry_run_record_parsed": bool(record),
                    "verified": not cell_blockers,
                    "queue": queue_summary,
                    "blockers": _dedupe(cell_blockers),
                }
            )

    passed_count = sum(1 for cell in verified_cells if cell["verified"] is True)
    all_blockers = _dedupe(
        [
            *global_blockers,
            *[
                f"{cell['variant']}:{cell['axis']}:{blocker}"
                for cell in verified_cells
                for blocker in cell["blockers"]
            ],
        ]
    )
    all_passed = (
        not all_blockers
        and len(verified_cells) == expected_cell_count
        and passed_count == expected_cell_count
    )
    return {
        **_PLANNING_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_SCHEMA,
        "tool": L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_TOOL_PATH,
        "generated_at_utc": generated,
        "source_bundle": _repo_relative(bundle_file, root),
        "source_bundle_sha256": bundle_sha256,
        "source_bundle_schema": bundle.get("schema", ""),
        "source_bundle_tool": bundle.get(
            "tool",
            L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_TOOL_PATH,
        ),
        "source_plan": source_plan,
        "expected_cell_count": expected_cell_count,
        "cell_count": len(verified_cells),
        "passed_cell_count": passed_count,
        "failed_cell_count": len(verified_cells) - passed_count,
        "dry_run_cells_total": expected_cell_count,
        "dry_run_cells_exercised": len(verified_cells),
        "all_dry_runs_passed": all_passed,
        "ready_for_dry_run_submit": all_passed,
        "ready_for_non_dry_run_submit": False,
        "ready_for_provider_dispatch": False,
        "coverage": {
            "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
            "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
            "covered_variants": sorted({str(cell["variant"]) for cell in verified_cells}),
            "covered_axes": sorted({str(cell["axis"]) for cell in verified_cells}),
            "missing_cells": missing_cells,
            "duplicate_cells": duplicate_cells,
            "extra_cells": extra_cells,
            "key_missing_indices": key_missing_indices,
        },
        "authority_notes": [
            "Dry-run verification proves local parser and queue-spec custody only.",
            "It does not stage source manifests, create active dispatch claims, contact Lightning, run auth eval, or claim score movement.",
            "Source-spec command SHA and launcher queue command SHA are expected to differ because the submit layer adds custody metadata.",
        ],
        "cells": verified_cells,
        "blockers": all_blockers,
    }


def l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_json(
    payload: Mapping[str, Any],
) -> str:
    """Return canonical JSON text for the dry-run verification artifact."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render a compact operator report for the dry-run verification artifact."""

    lines = [
        "# L5 v2 TT5L side-info Lightning dry-run verification",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        "",
        "This report verifies the dry-run submit commands generated by the TT5L "
        "Lightning execution bundle. It is parser and custody evidence only: no "
        "provider work was dispatched and no score movement is claimed.",
        "",
        "## Status",
        "",
        f"- Source bundle: `{payload.get('source_bundle')}`",
        f"- Source bundle SHA-256: `{payload.get('source_bundle_sha256')}`",
        f"- Cells passed: `{payload.get('passed_cell_count')}`/`{payload.get('cell_count')}`",
        f"- all_dry_runs_passed: `{payload.get('all_dry_runs_passed')}`",
        "- ready_for_non_dry_run_submit: `false`",
        "- ready_for_provider_dispatch: `false`",
        "- dispatch_attempted: `false`",
        "- provider_spend_attempted: `false`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        f"- Blockers: `{payload.get('blockers', [])}`",
        "",
        "## Cells",
        "",
        "| variant | axis | lane_id | returncode | verified | queue role | queue command SHA | source command SHA |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for cell in _mapping_rows(payload.get("cells")):
        queue = cell.get("queue") if isinstance(cell.get("queue"), Mapping) else {}
        lines.append(
            f"| `{cell.get('variant')}` | `{cell.get('axis_label')}` | "
            f"`{cell.get('lane_id')}` | `{cell.get('returncode')}` | "
            f"`{cell.get('verified')}` | `{queue.get('queue_role', '')}` | "
            f"`{queue.get('queue_command_sha256', '')}` | "
            f"`{queue.get('source_spec_command_sha256', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Command SHA discipline",
            "",
            "The launcher queue command SHA is not required to match the source "
            "plan command SHA. The submit layer wraps the source command with "
            "source-manifest, dispatch-lane, adjudication, and queue metadata, "
            "so this verifier checks invariant preservation instead of SHA "
            "identity.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_ARTIFACT_PATH",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_REPORT_PATH",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_SCHEMA",
    "L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_TOOL_PATH",
    "DryRunCommandResult",
    "DryRunRunner",
    "build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification",
    "l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_json",
    "render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_markdown",
]
