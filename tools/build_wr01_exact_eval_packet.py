#!/usr/bin/env python3
"""Build a deterministic WR01/HNeRV exact-eval operator packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import time
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
REQUIRED_ENV = (
    "LIGHTNING_SSH_TARGET",
    "LIGHTNING_REMOTE_PACT",
    "LIGHTNING_UPSTREAM_DIR",
    "LIGHTNING_TEAMSPACE",
    "LIGHTNING_STUDIO",
    "LIGHTNING_SDK_USER",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
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
    _check_present(
        consistency_checks,
        consistency_blockers,
        "manifest_source_archive_bytes_missing",
        manifest_payload.get("source_archive_bytes"),
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_archive_sha256_mismatch",
        manifest_payload.get("candidate_archive_sha256"),
        archive_identity["sha256"],
    )
    _check_equal(
        consistency_checks,
        consistency_blockers,
        "manifest_candidate_archive_bytes_mismatch",
        manifest_payload.get("candidate_archive_bytes"),
        archive_identity["bytes"],
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
        _get_path(dry_run_payload, "spec", "job_name"),
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
    if missing_env:
        blockers.append("missing_lightning_environment")
    return {
        "schema_version": 1,
        "tool": "tools/build_wr01_exact_eval_packet.py",
        "recorded_at_utc": _utc_now(),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_submit": not blockers,
        "blockers": blockers,
        "missing_env": missing_env,
        "missing_artifacts": missing_artifacts,
        "lane_id": args.lane_id,
        "job_name": args.job_name,
        "archive_sha256": args.archive_sha256,
        "archive_bytes": args.archive_bytes,
        "archive_identity": archive_identity,
        "source_archive_sha256": manifest_payload.get("source_archive_sha256"),
        "source_archive_bytes": manifest_payload.get("source_archive_bytes"),
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
    parser.add_argument("--agent", default="codex:gpt-5.5")
    parser.add_argument("--predicted-eta-utc", default="2026-05-06T07:30Z")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
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
