# SPDX-License-Identifier: MIT
"""Fail-closed non-dry-run gate for TT5L Lightning exact-eval cells.

The TT5L execution bundle is intentionally dry-run/default. This module is the
next guard in the chain: it consumes the bundle, required Lightning doctor
output, staged source manifests, and active dispatch-claim ledger before any
non-dry-run provider submit is treated as allowed.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_tt5l_lightning_doctor_plan import (
    L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_CLAIMS_PATH,
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA,
)

L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_SCHEMA = (
    "l5_v2_tt5l_lightning_non_dry_run_gate_v1"
)
L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_TOOL_PATH = (
    "tools/check_l5_v2_tt5l_lightning_non_dry_run_gate.py"
)
L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.json"
)
L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_REPORT_PATH = (
    ".omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.md"
)

_FALSE_SCORE_FLAGS = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "provider_spend_attempted": False,
}
_PLACEHOLDER_RE = re.compile(r"<[^>]+>")
_TERMINAL_STATUS_PREFIXES = (
    "completed",
    "failed",
    "refused",
    "stale",
    "stopped",
    "cancelled",
    "canceled",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_source_manifest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _load_json_object(path: Path) -> tuple[Mapping[str, Any], list[str]]:
    if not path.is_file():
        return {}, ["json_file_missing"]
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ["json_file_invalid"]
    if not isinstance(loaded, Mapping):
        return {}, ["json_not_object"]
    return loaded, []


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


def _axis_label(axis: str) -> str:
    return "[contest-CPU]" if axis == "contest_cpu" else "[contest-CUDA]"


def _expected_eval_device(axis: str) -> str:
    if axis == "contest_cpu":
        return "cpu"
    if axis == "contest_cuda":
        return "cuda"
    return ""


def _placeholder_tokens(command: str) -> list[str]:
    return _dedupe(_PLACEHOLDER_RE.findall(command))


def _parse_markdown_claim_rows(claims_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers = [
        "timestamp_utc",
        "agent",
        "lane_id",
        "platform",
        "instance/job_id",
        "predicted_eta_utc",
        "status",
        "notes",
    ]
    for line in claims_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("|---"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) != len(headers) or parts[0] in {"timestamp_utc", "---"}:
            continue
        rows.append(dict(zip(headers, parts, strict=True)))
    return rows


def _status_is_active(status: str) -> bool:
    value = status.strip().lower()
    if not value:
        return False
    if value.startswith(_TERMINAL_STATUS_PREFIXES):
        return False
    return "active" in value


def _latest_matching_active_claim(
    *,
    claims_rows: Sequence[Mapping[str, str]],
    lane_id: str,
    job_name: str,
    archive_sha256: str,
    axis: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    for row in claims_rows:
        if row.get("lane_id") != lane_id:
            continue
        if row.get("platform") != "lightning":
            continue
        if row.get("instance/job_id") != job_name:
            continue
        status = str(row.get("status") or "")
        notes = str(row.get("notes") or "")
        active = _status_is_active(status)
        if not active:
            blockers.append(f"latest_matching_claim_not_active:{status}")
            continue
        if archive_sha256 and archive_sha256 not in notes:
            blockers.append("active_claim_archive_sha256_missing")
        if axis and axis not in notes:
            blockers.append("active_claim_axis_missing")
        if "score_claim=false" not in notes:
            blockers.append("active_claim_score_claim_false_missing")
        return {
            "found": True,
            "active": active,
            "row": dict(row),
            "blockers": blockers,
        }, blockers
    return {
        "found": False,
        "active": False,
        "row": {},
        "blockers": ["active_lightning_claim_missing"],
    }, ["active_lightning_claim_missing"]


def _machine_count(machine_inventory: Mapping[str, Any], doctor: Mapping[str, Any]) -> int:
    for source in (machine_inventory, doctor):
        raw = source.get("machine_count")
        if isinstance(raw, int):
            return raw
        for key in ("machines", "available_machines", "inventory"):
            value = source.get(key)
            if isinstance(value, list):
                return len(value)
    return 0


def _validate_doctor(
    *,
    doctor_plan: Mapping[str, Any],
    doctor_output: Mapping[str, Any],
    doctor_output_exists: bool,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if doctor_plan.get("schema") != L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA:
        blockers.append("doctor_plan_schema_mismatch")
    if doctor_plan.get("ready_for_operator_doctor") is not True:
        blockers.append("doctor_plan_not_ready_for_operator_doctor")
    required = doctor_plan.get("doctor_required_checks")
    required_checks: list[str] = []
    required_fields: list[str] = []
    expected_status = "OK"
    if isinstance(required, Mapping):
        expected_status = str(required.get("expected_status") or "OK")
        required_checks = [
            str(item) for item in required.get("required_checks", []) if str(item)
        ]
        required_fields = [
            str(item) for item in required.get("required_json_fields", []) if str(item)
        ]
    if not doctor_output_exists:
        blockers.append("doctor_output_missing")
    if not doctor_output:
        blockers.append("doctor_output_invalid_or_empty")
    for field in required_fields:
        if field not in doctor_output:
            blockers.append(f"doctor_output_field_missing:{field}")
    if doctor_output and doctor_output.get("status") != expected_status:
        blockers.append("doctor_output_status_not_ok")
    failed_checks = doctor_output.get("failed_checks")
    if isinstance(failed_checks, list) and failed_checks:
        blockers.append("doctor_output_failed_checks_not_empty")
    checks = doctor_output.get("checks")
    if not isinstance(checks, Mapping):
        blockers.append("doctor_output_checks_missing")
        checks = {}
    check_status: dict[str, Any] = {}
    for name in required_checks:
        raw = checks.get(name)
        if not isinstance(raw, Mapping):
            blockers.append(f"doctor_check_missing:{name}")
            check_status[name] = {"ok": False, "blockers": [f"doctor_check_missing:{name}"]}
            continue
        ok = raw.get("ok")
        local_ok = ok is not False if name == "local_supply_chain" else ok is True
        local_blockers: list[str] = []
        if not local_ok:
            local_blockers.append(f"doctor_check_not_ok:{name}")
            blockers.append(f"doctor_check_not_ok:{name}")
        if name == "machine_inventory" and _machine_count(raw, doctor_output) <= 0:
            local_blockers.append("doctor_machine_inventory_empty")
            blockers.append("doctor_machine_inventory_empty")
        check_status[name] = {
            "ok": local_ok,
            "raw_ok": ok,
            "machine_count": _machine_count(raw, doctor_output)
            if name == "machine_inventory"
            else None,
            "blockers": local_blockers,
        }
    return {
        "status": doctor_output.get("status", ""),
        "expected_status": expected_status,
        "required_checks": required_checks,
        "checks": check_status,
    }, _dedupe(blockers)


def _validate_source_manifest(
    *,
    manifest: Mapping[str, Any],
    manifest_exists: bool,
    cell: Mapping[str, Any],
    bundle: Mapping[str, Any],
    current_head_commit: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    archive_path = str(cell.get("archive_path") or "")
    archive_sha256 = str(cell.get("archive_sha256") or "")
    archive_size = cell.get("archive_size_bytes")
    job_name = str(cell.get("job_name") or "")
    if not manifest_exists:
        blockers.append("source_manifest_missing")
    if not manifest:
        blockers.append("source_manifest_invalid_or_empty")
    if manifest and manifest.get("schema_version") != 1:
        blockers.append("source_manifest_schema_version_not_1")
    if manifest and manifest.get("tool") != "scripts/lightning_repro_workspace.py":
        blockers.append("source_manifest_tool_mismatch")
    if manifest and manifest.get("run_id") != job_name:
        blockers.append("source_manifest_run_id_mismatch")
    artifact_paths = manifest.get("artifact_paths")
    if isinstance(artifact_paths, list) and archive_path not in artifact_paths:
        blockers.append("source_manifest_artifact_path_missing")
    elif not isinstance(artifact_paths, list):
        blockers.append("source_manifest_artifact_paths_missing")
    git = manifest.get("git") if isinstance(manifest.get("git"), Mapping) else {}
    manifest_head = str(git.get("head") or "")
    bundle_head = str(bundle.get("current_head_commit") or "")
    if bundle_head and manifest_head != bundle_head:
        blockers.append("source_manifest_git_head_mismatch_bundle")
    if current_head_commit and bundle_head and bundle_head != current_head_commit:
        blockers.append("source_bundle_current_head_mismatch")
    if current_head_commit and manifest_head != current_head_commit:
        blockers.append("source_manifest_git_head_mismatch_current")
    files = _mapping_rows(manifest.get("files"))
    artifact_file = next((row for row in files if row.get("path") == archive_path), {})
    if not artifact_file:
        blockers.append("source_manifest_archive_file_missing")
    else:
        if artifact_file.get("role") != "artifact":
            blockers.append("source_manifest_archive_role_not_artifact")
        if artifact_file.get("sha256") != archive_sha256:
            blockers.append("source_manifest_archive_sha256_mismatch")
        if isinstance(archive_size, int) and artifact_file.get("bytes") != archive_size:
            blockers.append("source_manifest_archive_size_mismatch")
    manifest_sha = str(manifest.get("manifest_sha256") or "")
    recomputed_manifest_sha = _sha256_source_manifest(manifest) if manifest else ""
    if manifest_sha and recomputed_manifest_sha and manifest_sha != recomputed_manifest_sha:
        blockers.append("source_manifest_self_sha256_mismatch")
    return {
        "exists": manifest_exists,
        "run_id": manifest.get("run_id", "") if manifest else "",
        "git_head": manifest_head,
        "manifest_sha256": manifest_sha,
        "recomputed_manifest_sha256": recomputed_manifest_sha,
        "file_count": manifest.get("file_count", "") if manifest else "",
        "total_bytes": manifest.get("total_bytes", "") if manifest else "",
        "archive_file": dict(artifact_file) if artifact_file else {},
    }, _dedupe(blockers)


def _validate_stage_receipt(
    *,
    receipt: Mapping[str, Any],
    receipt_exists: bool,
    manifest_status: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage_manifest_path: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not receipt_exists:
        blockers.append("stage_receipt_missing")
    if not receipt:
        blockers.append("stage_receipt_invalid_or_empty")
    if receipt and receipt.get("schema_version") != 1:
        blockers.append("stage_receipt_schema_version_not_1")
    if receipt and receipt.get("tool") != "scripts/lightning_repro_workspace.py":
        blockers.append("stage_receipt_tool_mismatch")
    if receipt and receipt.get("status") != "OK":
        blockers.append("stage_receipt_status_not_ok")
    if receipt and receipt.get("dry_run") is not False:
        blockers.append("stage_receipt_dry_run_true")
    if receipt and receipt.get("remote_sha256_verified") is not True:
        blockers.append("stage_receipt_remote_sha256_not_verified")
    if receipt and receipt.get("manifest") != stage_manifest_path:
        blockers.append("stage_receipt_manifest_path_mismatch")
    manifest_sha = str(manifest_status.get("manifest_sha256") or "")
    if receipt and receipt.get("manifest_sha256") != manifest_sha:
        blockers.append("stage_receipt_manifest_sha256_mismatch")
    job_name = str(cell.get("job_name") or "")
    if receipt and receipt.get("run_id") != job_name:
        blockers.append("stage_receipt_run_id_mismatch")
    if receipt and not str(receipt.get("remote_manifest") or "").strip():
        blockers.append("stage_receipt_remote_manifest_missing")
    expected_file_count = manifest_status.get("file_count")
    if isinstance(expected_file_count, int) and receipt.get("file_count") != expected_file_count:
        blockers.append("stage_receipt_file_count_mismatch")
    expected_total_bytes = manifest_status.get("total_bytes")
    if isinstance(expected_total_bytes, int) and receipt.get("total_bytes") != expected_total_bytes:
        blockers.append("stage_receipt_total_bytes_mismatch")
    return {
        "exists": receipt_exists,
        "status": receipt.get("status", "") if receipt else "",
        "dry_run": receipt.get("dry_run", "") if receipt else "",
        "remote_sha256_verified": (
            receipt.get("remote_sha256_verified", "") if receipt else ""
        ),
        "manifest": receipt.get("manifest", "") if receipt else "",
        "remote_manifest": receipt.get("remote_manifest", "") if receipt else "",
        "run_id": receipt.get("run_id", "") if receipt else "",
        "manifest_sha256": receipt.get("manifest_sha256", "") if receipt else "",
        "file_count": receipt.get("file_count", "") if receipt else "",
        "total_bytes": receipt.get("total_bytes", "") if receipt else "",
    }, _dedupe(blockers)


def _validate_non_dry_run_command(
    *,
    cell: Mapping[str, Any],
    source_manifest_path: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    command = str(cell.get("non_dry_run_submit_command_template") or "")
    placeholders = _placeholder_tokens(command)
    if placeholders:
        blockers.append("non_dry_run_command_placeholders_present")
    try:
        argv = tuple(shlex.split(command))
    except ValueError:
        return {
            "argv": [],
            "placeholders": placeholders,
            "required_args": {},
        }, ["non_dry_run_command_parse_failed", *blockers]
    if _has_arg(argv, "--dry-run"):
        blockers.append("non_dry_run_command_contains_dry_run")
    required_flags = [
        "--job-name",
        "--archive",
        "--repo-dir",
        "--upstream-dir",
        "--output-dir",
        "--machine",
        "--inflate-sh",
        "--expected-archive-sha256",
        "--expected-archive-size-bytes",
        "--local-artifact-dir",
        "--dispatch-lane-id",
        "--dispatch-claims-path",
        "--eval-device",
        "--source-manifest",
        "--adjudicate",
        "--studio",
        "--teamspace",
        "--remote-preflight-ssh-target",
    ]
    required_args: dict[str, str] = {}
    for flag in required_flags:
        if flag == "--adjudicate":
            present = _has_arg(argv, flag)
            required_args[flag] = str(present)
            if not present:
                blockers.append("non_dry_run_command_adjudicate_missing")
            continue
        value = _arg_value(argv, flag)
        required_args[flag] = value
        if not value:
            blockers.append(f"non_dry_run_command_arg_missing:{flag}")
        elif _placeholder_tokens(value):
            blockers.append(f"non_dry_run_command_arg_placeholder:{flag}")
    identity_count = int(_has_arg(argv, "--user")) + int(_has_arg(argv, "--org"))
    if identity_count != 1:
        blockers.append("non_dry_run_command_identity_mode_not_exactly_one")
    expected_device = _expected_eval_device(str(cell.get("axis") or ""))
    if expected_device and _arg_value(argv, "--eval-device") != expected_device:
        blockers.append("non_dry_run_command_eval_device_mismatch")
    if _arg_value(argv, "--dispatch-lane-id") != str(cell.get("lane_id") or ""):
        blockers.append("non_dry_run_command_lane_id_mismatch")
    if _arg_value(argv, "--expected-archive-sha256") != str(
        cell.get("archive_sha256") or ""
    ):
        blockers.append("non_dry_run_command_archive_sha256_mismatch")
    if _arg_value(argv, "--expected-archive-size-bytes") != str(
        cell.get("archive_size_bytes") or ""
    ):
        blockers.append("non_dry_run_command_archive_size_mismatch")
    if _arg_value(argv, "--source-manifest") != source_manifest_path:
        blockers.append("non_dry_run_command_source_manifest_mismatch")
    if _arg_value(argv, "--dispatch-claims-path") != (
        L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_CLAIMS_PATH
    ):
        blockers.append("non_dry_run_command_claims_path_mismatch")
    return {
        "argv": list(argv),
        "placeholders": placeholders,
        "required_args": required_args,
    }, _dedupe(blockers)


def _manifest_out_from_stage_command(cell: Mapping[str, Any]) -> str:
    command = str(cell.get("stage_source_manifest_command_template") or "")
    try:
        argv = shlex.split(command)
    except ValueError:
        return ""
    return _arg_value(argv, "--manifest-out")


def _receipt_out_from_stage_command(cell: Mapping[str, Any]) -> str:
    command = str(cell.get("stage_source_manifest_command_template") or "")
    try:
        argv = shlex.split(command)
    except ValueError:
        return ""
    return _arg_value(argv, "--receipt-out")


def _coverage_blockers(cells: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    expected = {
        (variant, axis)
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
    }
    seen: dict[tuple[str, str], int] = {}
    for cell in cells:
        key = (str(cell.get("variant") or ""), str(cell.get("axis") or ""))
        seen[key] = seen.get(key, 0) + 1
    actual = set(seen)
    missing = sorted(f"{variant}:{axis}" for variant, axis in expected - actual)
    extra = sorted(f"{variant}:{axis}" for variant, axis in actual - expected)
    duplicate = sorted(f"{variant}:{axis}" for (variant, axis), count in seen.items() if count > 1)
    for item in missing:
        blockers.append(f"source_bundle_missing_cell:{item}")
    for item in extra:
        blockers.append(f"source_bundle_extra_cell:{item}")
    for item in duplicate:
        blockers.append(f"source_bundle_duplicate_cell:{item}")
    return {
        "expected_cell_count": len(expected),
        "actual_cell_count": len(cells),
        "missing_cells": missing,
        "extra_cells": extra,
        "duplicate_cells": duplicate,
    }, blockers


def build_l5_v2_tt5l_lightning_non_dry_run_gate(
    *,
    bundle: Mapping[str, Any],
    bundle_path: str | Path,
    doctor_plan: Mapping[str, Any],
    doctor_plan_path: str | Path,
    doctor_output: Mapping[str, Any],
    doctor_output_path: str | Path,
    claims_text: str,
    claims_path: str | Path,
    repo_root: str | Path,
    current_head_commit: str = "",
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Return the TT5L Lightning non-dry-run readiness gate artifact."""

    root = Path(repo_root).resolve()
    bundle_file = _resolve_repo_path(bundle_path, root)
    doctor_plan_file = _resolve_repo_path(doctor_plan_path, root)
    doctor_output_file = _resolve_repo_path(doctor_output_path, root)
    claims_file = _resolve_repo_path(claims_path, root)
    generated = generated_at_utc or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    blockers: list[str] = []
    if bundle.get("schema") != L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA:
        blockers.append("source_bundle_schema_mismatch")
    if bundle.get("ready_for_dry_run_submit") is not True:
        blockers.append("source_bundle_not_dry_run_ready")
    if bundle.get("ready_for_non_dry_run_submit") is not False:
        blockers.append("source_bundle_non_dry_run_flag_not_false")
    if bundle.get("ready_for_provider_dispatch") is not False:
        blockers.append("source_bundle_provider_dispatch_flag_not_false")
    for field, expected in _FALSE_SCORE_FLAGS.items():
        if field in bundle and bundle.get(field) is not expected:
            blockers.append(f"source_bundle_{field}_not_{str(expected).lower()}")

    cells = _mapping_rows(bundle.get("cells"))
    coverage, coverage_blockers = _coverage_blockers(cells)
    blockers.extend(coverage_blockers)
    doctor_output_exists = doctor_output_file.is_file()
    doctor_status, doctor_blockers = _validate_doctor(
        doctor_plan=doctor_plan,
        doctor_output=doctor_output,
        doctor_output_exists=doctor_output_exists,
    )
    blockers.extend(doctor_blockers)
    claims_rows = _parse_markdown_claim_rows(claims_text)
    if not claims_file.is_file():
        blockers.append("claims_ledger_missing")
    if not claims_rows:
        blockers.append("claims_ledger_has_no_rows")

    cell_payloads: list[dict[str, Any]] = []
    for cell in cells:
        cell_blockers: list[str] = []
        variant = str(cell.get("variant") or "")
        axis = str(cell.get("axis") or "")
        stage_manifest_path = _manifest_out_from_stage_command(cell)
        stage_receipt_path = _receipt_out_from_stage_command(cell)
        if not stage_manifest_path:
            cell_blockers.append("stage_source_manifest_path_missing")
        if not stage_receipt_path:
            cell_blockers.append("stage_source_manifest_receipt_path_missing")
        manifest_file = _resolve_repo_path(stage_manifest_path, root) if stage_manifest_path else root
        manifest, manifest_load_blockers = _load_json_object(manifest_file)
        cell_blockers.extend(f"source_manifest:{item}" for item in manifest_load_blockers)
        manifest_status, manifest_blockers = _validate_source_manifest(
            manifest=manifest,
            manifest_exists=manifest_file.is_file() if stage_manifest_path else False,
            cell=cell,
            bundle=bundle,
            current_head_commit=current_head_commit,
        )
        cell_blockers.extend(manifest_blockers)
        receipt_file = _resolve_repo_path(stage_receipt_path, root) if stage_receipt_path else root
        receipt, receipt_load_blockers = _load_json_object(receipt_file)
        receipt_status, receipt_blockers = _validate_stage_receipt(
            receipt=receipt,
            receipt_exists=receipt_file.is_file() if stage_receipt_path else False,
            manifest_status=manifest_status,
            cell=cell,
            stage_manifest_path=stage_manifest_path,
        )
        receipt_status["load_blockers"] = list(receipt_load_blockers)
        cell_blockers.extend(receipt_blockers)
        command_status, command_blockers = _validate_non_dry_run_command(
            cell=cell,
            source_manifest_path=stage_manifest_path,
        )
        cell_blockers.extend(command_blockers)
        claim_status, claim_blockers = _latest_matching_active_claim(
            claims_rows=claims_rows,
            lane_id=str(cell.get("lane_id") or ""),
            job_name=str(cell.get("job_name") or ""),
            archive_sha256=str(cell.get("archive_sha256") or ""),
            axis=axis,
        )
        cell_blockers.extend(claim_blockers)
        ready = not cell_blockers
        cell_payloads.append(
            {
                "variant": variant,
                "axis": axis,
                "axis_label": _axis_label(axis),
                "lane_id": cell.get("lane_id", ""),
                "job_name": cell.get("job_name", ""),
                "archive_path": cell.get("archive_path", ""),
                "archive_sha256": cell.get("archive_sha256", ""),
                "archive_size_bytes": cell.get("archive_size_bytes", ""),
                "source_manifest_path": stage_manifest_path,
                "source_manifest": manifest_status,
                "stage_source_manifest_receipt_path": stage_receipt_path,
                "stage_receipt": receipt_status,
                "non_dry_run_command": command_status,
                "active_claim": claim_status,
                "ready_for_non_dry_run_submit": ready,
                "ready_for_provider_dispatch": ready,
                "blockers": _dedupe(cell_blockers),
            }
        )
        blockers.extend(f"{variant}:{axis}:{item}" for item in _dedupe(cell_blockers))

    ready_cells = sum(
        1 for cell in cell_payloads if cell["ready_for_non_dry_run_submit"] is True
    )
    all_ready = not blockers and ready_cells == coverage["expected_cell_count"]
    return {
        "schema": L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_SCHEMA,
        "tool": L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_TOOL_PATH,
        **_FALSE_SCORE_FLAGS,
        "planning_only": False,
        "operator_execute_required": True,
        "generated_at_utc": generated,
        "current_head_commit": current_head_commit,
        "source_bundle": _repo_relative(bundle_file, root),
        "source_bundle_sha256": _sha256_file(bundle_file) if bundle_file.is_file() else "",
        "source_doctor_plan": _repo_relative(doctor_plan_file, root),
        "source_doctor_plan_sha256": (
            _sha256_file(doctor_plan_file) if doctor_plan_file.is_file() else ""
        ),
        "source_doctor_output": _repo_relative(doctor_output_file, root),
        "source_doctor_output_sha256": (
            _sha256_file(doctor_output_file) if doctor_output_file.is_file() else ""
        ),
        "source_claims_ledger": _repo_relative(claims_file, root),
        "source_claims_ledger_sha256": (
            _sha256_file(claims_file) if claims_file.is_file() else ""
        ),
        "coverage": coverage,
        "doctor_status": doctor_status,
        "cell_count": len(cell_payloads),
        "ready_cell_count": ready_cells,
        "cells": cell_payloads,
        "ready_for_non_dry_run_submit": all_ready,
        "ready_for_provider_dispatch": all_ready,
        "blockers": _dedupe(blockers),
    }


def l5_v2_tt5l_lightning_non_dry_run_gate_json(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON text for the TT5L Lightning non-dry-run gate."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_l5_v2_tt5l_lightning_non_dry_run_gate_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render the TT5L Lightning non-dry-run gate for operators."""

    lines = [
        "# L5 v2 TT5L Lightning non-dry-run gate",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        "",
        "This gate is spend-readiness only. It does not dispatch provider work "
        "and does not claim score movement. It fails closed unless the doctor "
        "output is OK, every per-cell source manifest has a real remote-verified "
        "staging receipt, every cell has an active Lightning lane claim, and all "
        "non-dry-run submit templates are free of placeholders.",
        "",
        "## Status",
        "",
        f"- Source bundle: `{payload.get('source_bundle')}`",
        f"- Doctor plan: `{payload.get('source_doctor_plan')}`",
        f"- Doctor output: `{payload.get('source_doctor_output')}`",
        f"- Claims ledger: `{payload.get('source_claims_ledger')}`",
        f"- ready_for_non_dry_run_submit: `{payload.get('ready_for_non_dry_run_submit')}`",
        f"- ready_for_provider_dispatch: `{payload.get('ready_for_provider_dispatch')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- dispatch_attempted: `false`",
        f"- Ready cells: `{payload.get('ready_cell_count')}`/`{payload.get('cell_count')}`",
        f"- Blocker count: `{len(payload.get('blockers', []))}`",
        "",
        "## Top-Level Blockers",
        "",
    ]
    blockers = [str(blocker) for blocker in payload.get("blockers", [])]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Cells",
            "",
            "| variant | axis | lane_id | job_name | ready | blockers |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cell in _mapping_rows(payload.get("cells")):
        lines.append(
            f"| `{cell.get('variant')}` | `{cell.get('axis_label')}` | "
            f"`{cell.get('lane_id')}` | `{cell.get('job_name')}` | "
            f"`{cell.get('ready_for_non_dry_run_submit')}` | "
            f"`{cell.get('blockers', [])}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_ARTIFACT_PATH",
    "L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_REPORT_PATH",
    "L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_SCHEMA",
    "L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_TOOL_PATH",
    "build_l5_v2_tt5l_lightning_non_dry_run_gate",
    "l5_v2_tt5l_lightning_non_dry_run_gate_json",
    "render_l5_v2_tt5l_lightning_non_dry_run_gate_markdown",
]
