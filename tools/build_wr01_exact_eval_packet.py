#!/usr/bin/env python3
"""Build a deterministic WR01/HNeRV exact-eval operator packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_JOB_NAME = "exact_eval_wr01_apply_pr106x_half_20260506"
DEFAULT_LANE_ID = "wr01_apply_pr106x_half"
DEFAULT_ARCHIVE = Path(
    "experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/"
    "hnerv_wavelet_apply_transform_candidate.zip"
)
DEFAULT_BASELINE_JSON = Path(
    "experiments/results/lightning_batch/"
    "exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_INFLATE_SH = Path(
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/"
    "source/submissions/belt_and_suspenders/inflate.sh"
)
DEFAULT_UPSTREAM_DIR = Path(
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/results/lightning_batch/exact_eval_wr01_apply_pr106x_half_20260506"
)
DEFAULT_RESULT_DIR = Path(
    "experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex"
)
DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
DEFAULT_CLAIM_TTL_HOURS = 24
CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES
TERMINAL_CLAIM_PREFIXES = (
    "completed_",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
)
_CLAIM_SEPARATOR_RE = re.compile(r"^\|\s*-+\s*(\|\s*-+\s*)+\|\s*$")
REQUIRED_ENV = (
    "LIGHTNING_SSH_TARGET",
    "LIGHTNING_REMOTE_PACT",
    "LIGHTNING_UPSTREAM_DIR",
    "LIGHTNING_TEAMSPACE",
    "LIGHTNING_STUDIO",
    "LIGHTNING_SDK_USER",
)


def _format_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _now_utc(args: argparse.Namespace) -> dt.datetime:
    parsed = _parse_utc(args.now_utc)
    if parsed is not None:
        return parsed
    if args.now_utc:
        raise ValueError(f"--now-utc is not ISO-8601 UTC-compatible: {args.now_utc}")
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_repo_path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _q(value: object) -> str:
    text = str(value)
    if text.startswith("$") or text.startswith("${"):
        return text
    return shlex.quote(text)


def _one_liner(cmd: list[str]) -> str:
    return " ".join(_q(item) for item in cmd)


def _artifact_status(path: Path) -> dict[str, Any]:
    full = _repo_path(path)
    return {
        "path": path.as_posix(),
        "exists": full.is_file(),
        "bytes": full.stat().st_size if full.is_file() else None,
    }


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_file(path: Path) -> str | None:
    full = _repo_path(path)
    if not full.is_file():
        return None
    digest = hashlib.sha256()
    with full.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_identity(path: Path) -> dict[str, Any]:
    full = _repo_path(path)
    return {
        "path": path.as_posix(),
        "exists": full.is_file(),
        "sha256": _sha256_file(path),
        "bytes": full.stat().st_size if full.is_file() else None,
    }


def _get_path(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_terminal_claim_status(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_PREFIXES)


def _parse_claim_rows(text: str) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    keys = (
        "timestamp_utc",
        "agent",
        "lane_id",
        "platform",
        "instance_job_id",
        "predicted_eta_utc",
        "status",
        "notes",
    )
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "lane_id" in line and "instance/job_id" in line:
            continue
        if _CLAIM_SEPARATOR_RE.match(line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < len(keys):
            continue
        claims.append(dict(zip(keys, cells[: len(keys)], strict=False)))
    return claims


def _lane_claim_preflight(
    *,
    claims_path: Path,
    lane_id: str,
    job_name: str,
    now_utc: dt.datetime,
    ttl_hours: int,
) -> dict[str, Any]:
    full = _repo_path(claims_path)
    status: dict[str, Any] = {
        "claims_path": claims_path.as_posix(),
        "claims_path_exists": full.is_file(),
        "ttl_hours": ttl_hours,
        "matching_active_claims": [],
        "conflicting_active_claims": [],
        "matching_terminal_claims": [],
        "active_claim_present": False,
        "conflict_present": False,
    }
    if not full.is_file():
        return status
    ttl = dt.timedelta(hours=ttl_hours)
    latest_by_job: dict[str, dict[str, str]] = {}
    for claim in _parse_claim_rows(full.read_text(encoding="utf-8")):
        if claim.get("lane_id") != lane_id:
            continue
        timestamp = _parse_utc(claim.get("timestamp_utc"))
        if timestamp is None or now_utc - timestamp > ttl:
            continue
        job = claim.get("instance_job_id", "")
        previous = latest_by_job.get(job)
        previous_timestamp = _parse_utc(previous.get("timestamp_utc")) if previous else None
        if previous is None or previous_timestamp is None or timestamp > previous_timestamp:
            latest_by_job[job] = claim

    active: list[dict[str, str]] = []
    terminal: list[dict[str, str]] = []
    for claim in latest_by_job.values():
        if _is_terminal_claim_status(claim.get("status", "")):
            terminal.append(claim)
        else:
            active.append(claim)
    status["matching_active_claims"] = [
        claim for claim in active if claim.get("instance_job_id") == job_name
    ]
    status["conflicting_active_claims"] = [
        claim for claim in active if claim.get("instance_job_id") != job_name
    ]
    status["matching_terminal_claims"] = [
        claim for claim in terminal if claim.get("instance_job_id") == job_name
    ]
    status["active_claim_present"] = bool(status["matching_active_claims"])
    status["conflict_present"] = bool(status["conflicting_active_claims"])
    return status


def _failed_compliance_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for check in payload.get("checks", []):
        if not isinstance(check, dict):
            continue
        failed_flag = check.get("passed") is False or check.get("ok") is False
        if failed_flag and check.get("severity") != "warning":
            failed.append(check)
    return failed


def _artifact_flag_violations(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return explicit score/dispatch flags that make readiness unsafe.

    Legacy readiness artifacts do not all carry the same guard fields, so
    missing flags are tolerated. A present truthy flag is never tolerated in a
    packet that is supposed to be custody/readiness-only.
    """

    guarded_flags = (
        "score_claim",
        "promotion_eligible",
        "dispatch_attempted",
        "dispatch_performed",
        "gpu_launched",
    )
    violations: list[dict[str, Any]] = []
    for artifact_name, payload in artifacts.items():
        if not payload:
            continue
        for flag in guarded_flags:
            if payload.get(flag) is True:
                violations.append(
                    {
                        "artifact": artifact_name,
                        "flag": flag,
                        "actual": True,
                        "expected": False,
                    }
                )
    return violations


def _changed_section(payload: dict[str, Any]) -> dict[str, Any] | None:
    changed = [
        section
        for section in payload.get("sections", [])
        if isinstance(section, dict) and section.get("changed") is True
    ]
    if len(changed) != 1:
        return None
    return changed[0]


def _first_archive_member(payload: dict[str, Any]) -> dict[str, Any] | None:
    members = _get_path(payload, "archive", "members")
    if not isinstance(members, list) or not members:
        return None
    return members[0] if isinstance(members[0], dict) else None


def _preflight_section_sha256(payload: dict[str, Any], section_name: str | None) -> Any:
    if section_name is None:
        return None
    member = _first_archive_member(payload)
    if member is None:
        return None
    section = _get_path(member, "decode_smoke", "format", section_name)
    if not isinstance(section, dict):
        return None
    return section.get("sha256")


def _check_equal(
    checks: list[dict[str, Any]],
    blockers: list[str],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    ok = expected is not None and actual == expected
    checks.append({"name": name, "ok": ok, "actual": actual, "expected": expected})
    if not ok:
        blockers.append(name)


def _check_present(
    checks: list[dict[str, Any]],
    blockers: list[str],
    name: str,
    value: Any,
) -> None:
    ok = value is not None
    checks.append({"name": name, "ok": ok, "actual": value, "expected": "present"})
    if not ok:
        blockers.append(name)


def _check_condition(
    checks: list[dict[str, Any]],
    blockers: list[str],
    name: str,
    ok: bool,
    *,
    actual: Any,
    expected: Any,
) -> None:
    checks.append({"name": name, "ok": bool(ok), "actual": actual, "expected": expected})
    if not ok:
        blockers.append(name)


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    now_utc = _now_utc(args)
    result_dir = args.result_dir
    archive = args.archive
    preflight = result_dir / "public_replay_preflight.json"
    compliance = result_dir / "pre_submission_compliance.json"
    dry_run = result_dir / "lightning_exact_eval_dry_run.json"
    payload_diff = result_dir / "payload_section_diff_vs_pr106x.json"
    strength_summary = Path("experiments/results/hnerv_wavelet_apply_transform_wr01_strength_summary_20260506_codex.json")
    manifest = result_dir / "manifest.json"

    missing_env = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    artifacts = [
        archive,
        args.baseline_json,
        manifest,
        preflight,
        compliance,
        payload_diff,
        dry_run,
        strength_summary,
    ]
    artifact_statuses = [_artifact_status(path) for path in artifacts]
    missing_artifacts = [row["path"] for row in artifact_statuses if not row["exists"]]
    manifest_payload = _read_json(manifest) if _repo_path(manifest).is_file() else {}
    preflight_payload = _read_json(preflight) if _repo_path(preflight).is_file() else {}
    compliance_payload = _read_json(compliance) if _repo_path(compliance).is_file() else {}
    payload_diff_payload = _read_json(payload_diff) if _repo_path(payload_diff).is_file() else {}
    dry_run_payload = _read_json(dry_run) if _repo_path(dry_run).is_file() else {}
    artifact_flag_violations = _artifact_flag_violations(
        {
            "manifest": manifest_payload,
            "public_replay_preflight": preflight_payload,
            "pre_submission_compliance": compliance_payload,
            "payload_section_diff": payload_diff_payload,
            "lightning_exact_eval_dry_run": dry_run_payload,
        }
    )
    lane_claim_preflight = _lane_claim_preflight(
        claims_path=args.claims_path,
        lane_id=args.lane_id,
        job_name=args.job_name,
        now_utc=now_utc,
        ttl_hours=args.claim_ttl_hours,
    )

    preflight_ready = (
        preflight_payload.get("ready_for_exact_eval_dispatch") is True
        and not preflight_payload.get("blockers")
    )
    failed_compliance_checks = _failed_compliance_checks(compliance_payload)
    compliance_ok = compliance_payload.get("passed") is True and not failed_compliance_checks
    payload_diff_ready = (
        payload_diff_payload.get("ready_for_archive_preflight") is True
        and payload_diff_payload.get("changed_section_count") == 1
        and not payload_diff_payload.get("blockers")
    )
    dry_run_ready = (dry_run_payload.get("submit_readiness") or {}).get("ok") is True
    archive_identity = _archive_identity(archive)
    consistency_blockers: list[str] = []
    consistency_checks: list[dict[str, Any]] = []
    changed_section = _changed_section(payload_diff_payload)
    changed_section_name = (
        (manifest_payload.get("changed_section") or {}).get("name")
        or manifest_payload.get("changed_section_name")
        or manifest_payload.get("section_name")
    )
    changed_section_source_sha256 = (
        (manifest_payload.get("changed_section") or {}).get("source_sha256")
        or manifest_payload.get("changed_section_source_sha256")
        or manifest_payload.get("source_section_sha256")
    )
    changed_section_sha256 = (
        (manifest_payload.get("changed_section") or {}).get("candidate_sha256")
        or manifest_payload.get("changed_section_sha256")
        or manifest_payload.get("candidate_section_sha256")
    )
    source_archive_path = manifest_payload.get("source_archive_path")
    source_archive_custody_mode = manifest_payload.get("source_archive_custody_mode")
    allowed_source_custody_modes = (
        "verified_source_archive_payload_match",
        "operator_supplied_source_archive_identity",
    )
    source_archive_identity = (
        _archive_identity(Path(source_archive_path))
        if isinstance(source_archive_path, str) and source_archive_path
        else {
            "path": source_archive_path,
            "exists": False,
            "sha256": None,
            "bytes": None,
        }
    )

    _check_equal(
        consistency_checks,
        consistency_blockers,
        "archive_sha256_arg_mismatch",
        archive_identity["sha256"],
        args.archive_sha256,
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "archive_bytes_arg_mismatch",
        archive_identity["bytes"],
        args.archive_bytes,
    )
    _check_present(
        consistency_checks,
        consistency_blockers,
        "manifest_source_archive_sha256_missing",
        manifest_payload.get("source_archive_sha256"),
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_source_archive_sha256_malformed",
        _is_sha256(manifest_payload.get("source_archive_sha256")),
        actual=manifest_payload.get("source_archive_sha256"),
        expected="64-char lowercase hex sha256",
    )
    _check_present(
        consistency_checks,
        consistency_blockers,
        "manifest_source_archive_bytes_missing",
        manifest_payload.get("source_archive_bytes"),
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_source_archive_bytes_not_positive_int",
        _is_positive_int(manifest_payload.get("source_archive_bytes")),
        actual=manifest_payload.get("source_archive_bytes"),
        expected="positive integer bytes",
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_source_archive_custody_mode_invalid",
        source_archive_custody_mode in allowed_source_custody_modes,
        actual=source_archive_custody_mode,
        expected=list(allowed_source_custody_modes),
    )
    if source_archive_custody_mode == "verified_source_archive_payload_match":
        _check_present(
            consistency_checks,
            consistency_blockers,
            "manifest_source_archive_path_missing",
            source_archive_path,
        )
        _check_condition(
            consistency_checks,
            consistency_blockers,
            "manifest_source_archive_path_missing_on_disk",
            bool(source_archive_identity["exists"]),
            actual=source_archive_identity,
            expected="existing source archive path",
        )
        _check_equal(
            consistency_checks,
            consistency_blockers,
            "manifest_source_archive_path_sha256_mismatch",
            source_archive_identity["sha256"],
            manifest_payload.get("source_archive_sha256"),
        )
        _check_equal(
            consistency_checks,
            consistency_blockers,
            "manifest_source_archive_path_bytes_mismatch",
            source_archive_identity["bytes"],
            manifest_payload.get("source_archive_bytes"),
        )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_archive_sha256_mismatch",
        manifest_payload.get("candidate_archive_sha256"),
        archive_identity["sha256"],
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_archive_sha256_malformed",
        _is_sha256(manifest_payload.get("candidate_archive_sha256")),
        actual=manifest_payload.get("candidate_archive_sha256"),
        expected="64-char lowercase hex sha256",
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_archive_bytes_mismatch",
        manifest_payload.get("candidate_archive_bytes"),
        archive_identity["bytes"],
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_archive_bytes_not_positive_int",
        _is_positive_int(manifest_payload.get("candidate_archive_bytes")),
        actual=manifest_payload.get("candidate_archive_bytes"),
        expected="positive integer bytes",
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_archive_sha256_equals_source_archive_sha256_noop",
        not (
            _is_sha256(manifest_payload.get("candidate_archive_sha256"))
            and _is_sha256(manifest_payload.get("source_archive_sha256"))
            and manifest_payload.get("candidate_archive_sha256")
            == manifest_payload.get("source_archive_sha256")
        ),
        actual={
            "source_archive_sha256": manifest_payload.get("source_archive_sha256"),
            "candidate_archive_sha256": manifest_payload.get("candidate_archive_sha256"),
        },
        expected="different source/candidate archive sha256",
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_payload_sha256_malformed",
        _is_sha256(manifest_payload.get("candidate_payload_sha256")),
        actual=manifest_payload.get("candidate_payload_sha256"),
        expected="64-char lowercase hex sha256",
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_source_payload_sha256_malformed",
        _is_sha256(manifest_payload.get("source_payload_sha256")),
        actual=manifest_payload.get("source_payload_sha256"),
        expected="64-char lowercase hex sha256",
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_payload_sha256_equals_source_payload_sha256_noop",
        not (
            _is_sha256(manifest_payload.get("candidate_payload_sha256"))
            and _is_sha256(manifest_payload.get("source_payload_sha256"))
            and manifest_payload.get("candidate_payload_sha256")
            == manifest_payload.get("source_payload_sha256")
        ),
        actual={
            "source_payload_sha256": manifest_payload.get("source_payload_sha256"),
            "candidate_payload_sha256": manifest_payload.get("candidate_payload_sha256"),
        },
        expected="different source/candidate payload sha256",
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "payload_diff_candidate_archive_sha256_mismatch",
        payload_diff_payload.get("candidate_archive_sha256"),
        archive_identity["sha256"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "payload_diff_candidate_archive_bytes_mismatch",
        payload_diff_payload.get("candidate_archive_bytes"),
        archive_identity["bytes"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "payload_diff_source_archive_sha256_mismatch",
        payload_diff_payload.get("source_archive_sha256"),
        manifest_payload.get("source_archive_sha256"),
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "payload_diff_source_archive_bytes_mismatch",
        payload_diff_payload.get("source_archive_bytes"),
        manifest_payload.get("source_archive_bytes"),
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "payload_diff_source_payload_sha256_mismatch",
        payload_diff_payload.get("source_payload_sha256"),
        manifest_payload.get("source_payload_sha256"),
    )
    _check_present(
        consistency_checks,
        consistency_blockers,
        "manifest_changed_section_sha256_missing",
        changed_section_sha256,
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_changed_section_source_sha256_malformed",
        _is_sha256(changed_section_source_sha256),
        actual=changed_section_source_sha256,
        expected="64-char lowercase hex sha256",
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_changed_section_sha256_malformed",
        _is_sha256(changed_section_sha256),
        actual=changed_section_sha256,
        expected="64-char lowercase hex sha256",
    )
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "manifest_changed_section_candidate_sha256_equals_source_sha256_noop",
        not (
            _is_sha256(changed_section_sha256)
            and _is_sha256(changed_section_source_sha256)
            and changed_section_sha256 == changed_section_source_sha256
        ),
        actual={
            "source_section_sha256": changed_section_source_sha256,
            "candidate_section_sha256": changed_section_sha256,
        },
        expected="different source/candidate changed-section sha256",
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "payload_diff_changed_section_name_mismatch",
        changed_section.get("name") if changed_section else None,
        changed_section_name,
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "payload_diff_changed_section_source_sha256_mismatch",
        changed_section.get("source_sha256") if changed_section else None,
        changed_section_source_sha256,
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "payload_diff_changed_section_sha256_mismatch",
        changed_section.get("candidate_sha256") if changed_section else None,
        changed_section_sha256,
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "preflight_archive_sha256_mismatch",
        _get_path(preflight_payload, "archive", "sha256"),
        archive_identity["sha256"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "preflight_archive_bytes_mismatch",
        _get_path(preflight_payload, "archive", "bytes"),
        archive_identity["bytes"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "preflight_changed_section_sha256_mismatch",
        _preflight_section_sha256(preflight_payload, str(changed_section_name) if changed_section_name else None),
        changed_section_sha256,
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "preflight_candidate_payload_sha256_mismatch",
        _get_path(_first_archive_member(preflight_payload) or {}, "decode_smoke", "sha256"),
        manifest_payload.get("candidate_payload_sha256"),
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "compliance_archive_sha256_mismatch",
        _get_path(compliance_payload, "archive", "sha256"),
        archive_identity["sha256"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "compliance_archive_bytes_mismatch",
        _get_path(compliance_payload, "archive", "bytes"),
        archive_identity["bytes"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "compliance_candidate_payload_sha256_mismatch",
        _get_path(_first_archive_member(compliance_payload) or {}, "sha256"),
        manifest_payload.get("candidate_payload_sha256"),
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "compliance_manifest_path_mismatch",
        _get_path(compliance_payload, "archive_manifest", "path"),
        manifest.as_posix(),
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "dry_run_expected_archive_sha256_mismatch",
        _get_path(dry_run_payload, "spec", "expected_archive_sha256"),
        archive_identity["sha256"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "dry_run_expected_archive_bytes_mismatch",
        _get_path(dry_run_payload, "spec", "expected_archive_size_bytes"),
        archive_identity["bytes"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "dry_run_job_name_mismatch",
        _first_present(
            _get_path(dry_run_payload, "spec", "job_name"),
            _get_path(dry_run_payload, "spec", "name"),
        ),
        args.job_name,
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "dry_run_queue_lane_mismatch",
        _get_path(dry_run_payload, "spec", "queue_metadata", "lane"),
        args.lane_id,
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "dry_run_queue_archive_manifest_mismatch",
        _get_path(dry_run_payload, "spec", "queue_metadata", "archive_manifest"),
        manifest.as_posix(),
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "dry_run_queue_public_preflight_mismatch",
        _get_path(dry_run_payload, "spec", "queue_metadata", "public_preflight"),
        preflight.as_posix(),
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "dry_run_queue_payload_section_diff_mismatch",
        _get_path(dry_run_payload, "spec", "queue_metadata", "payload_section_diff"),
        payload_diff.as_posix(),
    )

    notes = (
        f"WR01 half exact CUDA eval; archive_sha256={args.archive_sha256} "
        f"bytes={args.archive_bytes}; static_preflight={preflight.as_posix()}"
    )
    claim_cmd = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        args.lane_id,
        "--platform",
        "lightning",
        "--instance-job-id",
        args.job_name,
        "--agent",
        args.agent,
        "--predicted-eta-utc",
        args.predicted_eta_utc,
        "--status",
        "active_exact_eval",
        "--notes",
        notes,
    ]
    submit_cmd = [
        ".venv/bin/python",
        "scripts/lightning_exact_eval_repro.py",
        "--job-name",
        args.job_name,
        "--stage-workspace",
        "--submit",
        "--archive",
        archive.as_posix(),
        "--baseline-json",
        args.baseline_json.as_posix(),
        "--inflate-sh",
        args.inflate_sh.as_posix(),
        "--upstream-dir",
        "$LIGHTNING_UPSTREAM_DIR",
        "--remote",
        "$LIGHTNING_SSH_TARGET",
        "--remote-pact",
        "$LIGHTNING_REMOTE_PACT",
        "--studio",
        "$LIGHTNING_STUDIO",
        "--teamspace",
        "$LIGHTNING_TEAMSPACE",
        "--sdk-user",
        "$LIGHTNING_SDK_USER",
        "--machine",
        "${LIGHTNING_MACHINE:-T4}",
        "--predicted-band",
        "0.18",
        "0.25",
        "--regression-threshold",
        "0.02",
        "--max-posenet-relative",
        "1.25",
        "--max-segnet-relative",
        "1.10",
        "--max-sane-score",
        "1.0",
        "--component-trace",
        "--queue-metadata",
        f"lane={args.lane_id}",
        "--queue-metadata",
        f"archive_manifest={manifest.as_posix()}",
        "--queue-metadata",
        f"public_preflight={preflight.as_posix()}",
        "--queue-metadata",
        f"payload_section_diff={payload_diff.as_posix()}",
        "--extra-artifact",
        manifest.as_posix(),
        "--extra-artifact",
        preflight.as_posix(),
        "--extra-artifact",
        compliance.as_posix(),
        "--extra-artifact",
        payload_diff.as_posix(),
        "--extra-artifact",
        strength_summary.as_posix(),
    ]
    harvest_cmd = [
        ".venv/bin/python",
        "scripts/launch_lightning_batch_job.py",
        "harvest-ssh",
        "--state-path",
        ".omx/state/lightning_batch_jobs.json",
        "--job-name",
        args.job_name,
        "--ssh-target",
        "$LIGHTNING_SSH_TARGET",
        "--expected-archive-sha256",
        args.archive_sha256,
        "--expected-archive-size-bytes",
        str(args.archive_bytes),
        "--require-adjudication",
    ]
    blockers = []
    if missing_artifacts:
        blockers.append("missing_artifacts")
    if not preflight_ready:
        blockers.append("public_replay_preflight_not_ready")
    if not compliance_ok:
        blockers.append("pre_submission_compliance_failed")
    if not payload_diff_ready:
        blockers.append("payload_section_diff_not_ready")
    if not dry_run_ready:
        blockers.append("lightning_dry_run_not_ready")
    if artifact_flag_violations:
        blockers.append("artifact_score_or_dispatch_flag_violation")
    blockers.extend(consistency_blockers)
    static_blockers = list(blockers)
    static_packet_ready = not static_blockers
    dispatch_gate = (
        "eligible_for_cuda_auth_eval_after_lane_claim"
        if static_packet_ready
        else "blocked_static_packet_ready_until_static_blockers_clear"
    )
    if missing_env:
        blockers.append("missing_lightning_environment")
    if lane_claim_preflight["conflict_present"]:
        blockers.append("active_lane_dispatch_claim_conflict")
    if not lane_claim_preflight["active_claim_present"]:
        blockers.append("missing_active_lane_dispatch_claim")
    if not args.operator_approved_exact_cuda:
        blockers.append("missing_operator_exact_cuda_approval")
    source_archive_bytes = manifest_payload.get("source_archive_bytes")
    archive_byte_delta = (
        args.archive_bytes - int(source_archive_bytes)
        if isinstance(source_archive_bytes, int) and not isinstance(source_archive_bytes, bool)
        else None
    )
    expected_total_score_delta = (
        archive_byte_delta * RATE_SCORE_PER_BYTE
        if archive_byte_delta is not None
        else None
    )
    return {
        "schema_version": 1,
        "schema": "wr01_exact_eval_operator_packet_v1",
        "packet_kind": "wr01_exact_eval_operator_packet",
        "tool": "tools/build_wr01_exact_eval_packet.py",
        "recorded_at_utc": _format_utc(now_utc),
        "candidate_id": args.lane_id,
        "family": "hnerv_wavelet_wr01_apply_transform",
        "family_group": "hnerv_wavelet_wr01_apply",
        "pareto_scope": "hnerv_rate_only_exact_archive",
        "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
        "confidence": 1.0,
        "proxy_row": False,
        "byte_delta": archive_byte_delta,
        "expected_total_score_delta": expected_total_score_delta,
        "expected_seg_dist_delta": 0.0,
        "expected_pose_dist_delta": 0.0,
        "expected_information_gain_nats": 0.0,
        "expected_score_variance": 0.0,
        "interaction_assumptions": [
            "rate-only single-member archive transform; no intentional SegNet or PoseNet distortion change",
            "archive and runtime parity are necessary but exact CUDA auth eval is required before any score claim",
            "no composability assumption with categorical, sensitivity, or pose atoms until stacked archive eval exists",
        ],
        "conflicts_with_families": [],
        "conflicts_with_atoms": [],
        "score_claim": False,
        "dispatch_attempted": False,
        "dispatch_gate": dispatch_gate,
        "dispatch_unlocked": static_packet_ready,
        "ready_for_exact_eval_dispatch_claim": static_packet_ready,
        "candidate_static_preflight_ready": static_packet_ready,
        "ready_for_submit": not blockers,
        "static_packet_ready": static_packet_ready,
        "blockers": blockers,
        "static_blockers": static_blockers,
        "missing_env": missing_env,
        "missing_artifacts": missing_artifacts,
        "operator_approved_exact_cuda": bool(args.operator_approved_exact_cuda),
        "lane_claim_preflight": lane_claim_preflight,
        "lane_id": args.lane_id,
        "job_name": args.job_name,
        "archive_sha256": args.archive_sha256,
        "archive_bytes": args.archive_bytes,
        "archive_identity": archive_identity,
        "source_archive_sha256": manifest_payload.get("source_archive_sha256"),
        "source_archive_bytes": manifest_payload.get("source_archive_bytes"),
        "source_archive_custody_mode": source_archive_custody_mode,
        "source_archive_identity": source_archive_identity,
        "source_payload_sha256": manifest_payload.get("source_payload_sha256"),
        "changed_section_name": changed_section_name,
        "changed_section_sha256": changed_section_sha256,
        "preflight_ready": preflight_ready,
        "compliance_ok": compliance_ok,
        "failed_compliance_checks": failed_compliance_checks,
        "payload_diff_ready": payload_diff_ready,
        "dry_run_ready": dry_run_ready,
        "artifact_flag_violations": artifact_flag_violations,
        "artifact_consistency_ok": not consistency_blockers,
        "artifact_consistency_checks": consistency_checks,
        "artifacts": artifact_statuses,
        "commands": {
            "claim": _one_liner(claim_cmd),
            "submit": _one_liner(submit_cmd),
            "harvest": _one_liner(harvest_cmd),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--inflate-sh", type=Path, default=DEFAULT_INFLATE_SH)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--archive-sha256", default="d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628")
    parser.add_argument("--archive-bytes", type=int, default=186222)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--claim-ttl-hours", type=int, default=DEFAULT_CLAIM_TTL_HOURS)
    parser.add_argument(
        "--operator-approved-exact-cuda",
        action="store_true",
        help=(
            "Explicit operator approval for a non-dry-run exact CUDA submission. "
            "Without this flag the packet stays blocked even when static custody "
            "and lane-claim checks pass."
        ),
    )
    parser.add_argument(
        "--now-utc",
        help="UTC timestamp for deterministic claim-TTL checks, e.g. 2026-05-06T10:00:00Z.",
    )
    parser.add_argument("--agent", default="codex:gpt-5.5")
    parser.add_argument("--predicted-eta-utc", default="2026-05-06T07:30Z")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.claim_ttl_hours <= 0:
        parser.error("--claim-ttl-hours must be positive")
    if args.now_utc and _parse_utc(args.now_utc) is None:
        parser.error("--now-utc must be ISO-8601 UTC-compatible, e.g. 2026-05-06T10:00:00Z")
    payload = build_packet(args)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        _repo_path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _repo_path(args.json_out).write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
