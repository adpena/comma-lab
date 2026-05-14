#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a deterministic WR01/HNeRV exact-eval operator packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
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
DEFAULT_RATE_ONLY_PRIORITY_PACKET = Path(
    "experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/"
    "hnerv_lowlevel_exact_eval_packet.json"
)
DEFAULT_RELEASE_SURFACE_SUBDIR = "release_surface"
RUNTIME_DECODE_VALIDATION_SCHEMA = "hnerv_wavelet_runtime_decode_validation.v1"
RUNTIME_DECODE_VALIDATION_FILENAME = "hnerv_wavelet_runtime_decode_validation.json"
RUNTIME_DECODE_REVIEW_SCHEMA = "hnerv_wavelet_compress_time_runtime_decode_review.v1"
RUNTIME_DECODE_REVIEW_FILENAME = "hnerv_wavelet_compress_time_runtime_decode_review.json"
RUNTIME_APPLY_SCHEMA = "hnerv_wavelet_runtime_apply.v1"
APPLY_TRANSFORM_TOOL = "tac.hnerv_wavelet_apply_transform.build_wavelet_apply_transform_candidate"
DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
DEFAULT_CLAIM_TTL_HOURS = 24
CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES
RUNTIME_REQUIRED_DISPATCH_BLOCKERS = (
    "requires_archive_manifest_preflight",
    "requires_component_response_or_exact_cuda_eval",
    "requires_lane_dispatch_claim",
    "requires_exact_cuda_auth_eval",
)
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


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _json_line(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _sha256_json_line(payload: Any) -> str:
    return hashlib.sha256(_json_line(payload).encode("utf-8")).hexdigest()


def _manifest_sha256_excluding_self(payload: dict[str, Any]) -> str:
    stripped = {
        key: value for key, value in payload.items() if key != "manifest_sha256_excluding_self"
    }
    return hashlib.sha256(_json_text(stripped).encode("utf-8")).hexdigest()


def _q(value: object) -> str:
    text = str(value)
    if text.startswith("$") or text.startswith("${"):
        return text
    return shlex.quote(text)


def _one_liner(cmd: list[str]) -> str:
    return " ".join(_q(item) for item in cmd)


def _packet_output_path(args: argparse.Namespace) -> Path:
    return args.json_out or args.result_dir / "wr01_exact_eval_packet.json"


def _required_env_check_cmd() -> list[str]:
    code = (
        "import os,sys; "
        "missing=[k for k in sys.argv[1:] if not os.environ.get(k)]; "
        "raise SystemExit(('FATAL: missing Lightning env: '+', '.join(missing)) if missing else 0)"
    )
    return [".venv/bin/python", "-c", code, *REQUIRED_ENV]


def _packet_ready_check_cmd(packet_path: Path) -> list[str]:
    code = (
        "import json,sys; "
        "p=json.load(open(sys.argv[1])); "
        "blockers=p.get('blockers') or []; "
        "ready=p.get('ready_for_submit') is True and not blockers; "
        "raise SystemExit(0 if ready else "
        "'FATAL: WR01 packet not ready_for_submit=true; blockers=' + ','.join(map(str, blockers)))"
    )
    return [".venv/bin/python", "-c", code, packet_path.as_posix()]


def _adversarial_priority_check_cmd(packet_path: Path) -> list[str]:
    code = (
        "import json,sys; "
        "p=json.load(open(sys.argv[1])); "
        "r=p.get('adversarial_priority_review') or {}; "
        "blockers=r.get('blockers') or []; "
        "raise SystemExit(0 if not blockers else "
        "'FATAL: WR01 adversarial priority gate blocked; blockers=' + ','.join(map(str, blockers)))"
    )
    return [".venv/bin/python", "-c", code, packet_path.as_posix()]


def _packet_refresh_cmd(
    args: argparse.Namespace,
    *,
    packet_path: Path,
    operator_approved_exact_cuda: bool,
) -> list[str]:
    cmd = [
        ".venv/bin/python",
        "tools/build_wr01_exact_eval_packet.py",
        "--job-name",
        args.job_name,
        "--lane-id",
        args.lane_id,
        "--archive",
        args.archive.as_posix(),
        "--baseline-json",
        args.baseline_json.as_posix(),
        "--inflate-sh",
        args.inflate_sh.as_posix(),
        "--result-dir",
        args.result_dir.as_posix(),
        "--release-surface-dir",
        args.release_surface_dir.as_posix(),
        "--archive-sha256",
        args.archive_sha256,
        "--archive-bytes",
        str(args.archive_bytes),
        "--claims-path",
        args.claims_path.as_posix(),
        "--claim-ttl-hours",
        str(args.claim_ttl_hours),
        "--agent",
        args.agent,
        "--rate-only-priority-packet",
        args.rate_only_priority_packet.as_posix(),
        "--build-release-surface",
        "--refresh-static-compliance",
        "--json-out",
        packet_path.as_posix(),
    ]
    if args.predicted_eta_utc:
        cmd.extend(["--predicted-eta-utc", args.predicted_eta_utc])
    if operator_approved_exact_cuda:
        cmd.append("--operator-approved-exact-cuda")
    return cmd


def _artifact_status(path: Path) -> dict[str, Any]:
    full = _repo_path(path)
    return {
        "path": path.as_posix(),
        "exists": full.is_file(),
        "bytes": full.stat().st_size if full.is_file() else None,
    }


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_display_path(path: Path) -> str:
    full = _repo_path(path)
    try:
        return full.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return full.as_posix()


def _paths_equivalent(actual: Any, expected: Path) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    try:
        actual_path = _repo_path(Path(actual)).resolve(strict=False)
        expected_path = _repo_path(expected).resolve(strict=False)
    except OSError:
        return False
    return actual_path == expected_path


def _runtime_decode_validation_path(result_dir: Path, manifest_payload: dict[str, Any]) -> Path:
    del manifest_payload
    return result_dir / RUNTIME_DECODE_VALIDATION_FILENAME


def _runtime_decode_review_path(result_dir: Path) -> Path:
    return result_dir / RUNTIME_DECODE_REVIEW_FILENAME


def _release_relative_path(release_dir: Path, target: Path) -> str:
    base = _repo_path(release_dir)
    full = _repo_path(target)
    return os.path.relpath(full, start=base).replace(os.sep, "/")


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


def _release_surface_status(release_dir: Path) -> dict[str, Any]:
    files = {}
    for rel in ("archive.zip", "inflate.sh", "report.txt", "archive_manifest.json"):
        path = release_dir / rel
        full = _repo_path(path)
        files[rel] = {
            "path": _repo_display_path(path),
            "exists": full.is_file(),
            "bytes": full.stat().st_size if full.is_file() else None,
            "sha256": _sha256_file(path) if full.is_file() else None,
        }
    return {
        "schema": "wr01_release_surface_status_v1",
        "path": _repo_display_path(release_dir),
        "exists": _repo_path(release_dir).is_dir(),
        "files": files,
    }


def _inflate_wrapper_text(source_inflate: Path) -> str:
    source_full = _repo_path(source_inflate)
    try:
        source_rel = source_full.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        source_rel = None
    if source_rel is None:
        source_line = f"SOURCE_INFLATE={shlex.quote(source_full.as_posix())}"
    else:
        source_line = f'SOURCE_INFLATE="${{REPO_ROOT}}/{source_rel}"'
    return (
        "#!/usr/bin/env bash\n"
        "# Deterministic WR01 static release-surface wrapper.\n"
        "# Delegates to the audited PR106 runtime without mutating the public-intake tree.\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'if REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"; then\n'
        "  :\n"
        "else\n"
        '  REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"\n'
        "fi\n"
        f"{source_line}\n"
        'if [ ! -x "$SOURCE_INFLATE" ]; then\n'
        '  echo "FATAL: delegated inflate.sh is missing or not executable: $SOURCE_INFLATE" >&2\n'
        "  exit 66\n"
        "fi\n"
        'exec "$SOURCE_INFLATE" "$@"\n'
    )


def _release_surface_report_text(
    *,
    args: argparse.Namespace,
    now_utc: dt.datetime,
    archive_identity: dict[str, Any],
    manifest_payload: dict[str, Any],
) -> str:
    lines = [
        "WR01 static release surface",
        f"recorded_at_utc: {_format_utc(now_utc)}",
        f"candidate_id: {args.lane_id}",
        "family: hnerv_wavelet_wr01_apply_transform",
        "scope: upload-file static surface only",
        "score_claim: false",
        "dispatch_attempted: false",
        "remote_gpu_run: false",
        "evidence_grade: empirical_archive_candidate_until_exact_cuda",
        f"archive_sha256: {archive_identity['sha256']}",
        f"archive_size_bytes: {archive_identity['bytes']}",
        f"archive_member: {manifest_payload.get('candidate_member_name', 'x')}",
        f"source_archive_sha256: {manifest_payload.get('source_archive_sha256')}",
        f"source_archive_size_bytes: {manifest_payload.get('source_archive_bytes')}",
        f"changed_section_name: {manifest_payload.get('changed_section_name')}",
        f"changed_section_sha256: {manifest_payload.get('changed_section_sha256')}",
        f"candidate_manifest: {_repo_display_path(args.result_dir / 'manifest.json')}",
        f"public_replay_preflight: {_repo_display_path(args.result_dir / 'public_replay_preflight.json')}",
        f"payload_section_diff: {_repo_display_path(args.result_dir / 'payload_section_diff_vs_pr106x.json')}",
        "remaining_for_score_or_dispatch: exact CUDA auth eval, adjudication, terminal dispatch claim",
        "notes: This file intentionally records no score and no promotion claim.",
    ]
    return "\n".join(lines) + "\n"


def build_release_surface(args: argparse.Namespace, *, now_utc: dt.datetime) -> dict[str, Any]:
    release_dir = args.release_surface_dir
    release_full = _repo_path(release_dir)
    archive_identity = _archive_identity(args.archive)
    if not archive_identity["exists"]:
        raise FileNotFoundError(f"archive does not exist: {args.archive}")
    if archive_identity["sha256"] != args.archive_sha256:
        raise ValueError(
            f"archive sha256 mismatch: expected={args.archive_sha256} actual={archive_identity['sha256']}"
        )
    if archive_identity["bytes"] != args.archive_bytes:
        raise ValueError(
            f"archive bytes mismatch: expected={args.archive_bytes} actual={archive_identity['bytes']}"
        )
    source_inflate_full = _repo_path(args.inflate_sh)
    if not source_inflate_full.is_file():
        raise FileNotFoundError(f"inflate.sh does not exist: {args.inflate_sh}")

    manifest = args.result_dir / "manifest.json"
    manifest_payload = _read_json(manifest) if _repo_path(manifest).is_file() else {}
    release_full.mkdir(parents=True, exist_ok=True)

    archive_dst = release_full / "archive.zip"
    shutil.copyfile(_repo_path(args.archive), archive_dst)
    archive_dst.chmod(0o644)

    inflate_dst = release_full / "inflate.sh"
    inflate_dst.write_text(_inflate_wrapper_text(args.inflate_sh), encoding="utf-8")
    inflate_dst.chmod(0o755)

    report_dst = release_full / "report.txt"
    report_dst.write_text(
        _release_surface_report_text(
            args=args,
            now_utc=now_utc,
            archive_identity=archive_identity,
            manifest_payload=manifest_payload,
        ),
        encoding="utf-8",
    )
    report_dst.chmod(0o644)

    linked_paths = {
        "candidate_manifest": args.result_dir / "manifest.json",
        "public_replay_preflight": args.result_dir / "public_replay_preflight.json",
        "payload_section_diff": args.result_dir / "payload_section_diff_vs_pr106x.json",
        "runtime_decode_validation": _runtime_decode_validation_path(args.result_dir, manifest_payload),
        "runtime_decode_review": _runtime_decode_review_path(args.result_dir),
        "lightning_exact_eval_dry_run": args.result_dir / "lightning_exact_eval_dry_run.json",
        "wr01_exact_eval_packet": args.result_dir / "wr01_exact_eval_packet.json",
    }
    surface_manifest = {
        "schema": "wr01_release_surface_manifest_v1",
        "schema_version": 1,
        "recorded_at_utc": _format_utc(now_utc),
        "candidate_id": args.lane_id,
        "job_name": args.job_name,
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "release_surface_scope": "static_upload_files_only_no_auth_eval_no_dispatch",
        "candidate_archive_sha256": archive_identity["sha256"],
        "candidate_archive_bytes": archive_identity["bytes"],
        "archive": {
            "path": "archive.zip",
            "source_path": _repo_display_path(args.archive),
            "sha256": archive_identity["sha256"],
            "bytes": archive_identity["bytes"],
        },
        "inflate_sh": {
            "path": "inflate.sh",
            "delegates_to": _repo_display_path(args.inflate_sh),
            "wrapper_sha256": _sha256_file(release_dir / "inflate.sh"),
        },
        "report": {
            "path": "report.txt",
            "sha256": _sha256_file(release_dir / "report.txt"),
        },
        "manifest_links": {
            name: {
                "repo_path": _repo_display_path(path),
                "release_surface_relative_path": _release_relative_path(release_dir, path),
                "exists": _repo_path(path).is_file(),
            }
            for name, path in linked_paths.items()
        },
        "source_archive_sha256": manifest_payload.get("source_archive_sha256"),
        "source_archive_bytes": manifest_payload.get("source_archive_bytes"),
        "source_archive_path": manifest_payload.get("source_archive_path"),
        "candidate_payload_sha256": manifest_payload.get("candidate_payload_sha256"),
        "changed_section_name": manifest_payload.get("changed_section_name"),
        "changed_section_sha256": manifest_payload.get("changed_section_sha256"),
        "remaining_required_for_score_or_dispatch": [
            "exact_cuda_auth_eval",
            "contest_auth_eval_adjudication",
            "terminal_dispatch_claim",
            "operator_score_claim_review",
        ],
    }
    manifest_dst = release_full / "archive_manifest.json"
    manifest_dst.write_text(
        json.dumps(surface_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_dst.chmod(0o644)
    return {
        "schema": "wr01_release_surface_generation_v1",
        "path": _repo_display_path(release_dir),
        "files": _release_surface_status(release_dir)["files"],
    }


def refresh_static_compliance(args: argparse.Namespace) -> dict[str, Any]:
    release_dir = args.release_surface_dir
    output = args.result_dir / "pre_submission_compliance.json"
    cmd = [
        sys.executable,
        "scripts/pre_submission_compliance_check.py",
        "--submission-dir",
        release_dir.as_posix(),
        "--archive",
        (release_dir / "archive.zip").as_posix(),
        "--archive-manifest-json",
        (release_dir / "archive_manifest.json").as_posix(),
        "--expect-single-member",
        "x",
        "--expected-archive-sha256",
        args.archive_sha256,
        "--expected-archive-size-bytes",
        str(args.archive_bytes),
        "--public-scan-path",
        release_dir.as_posix(),
        "--json-out",
        output.as_posix(),
        "--strict",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "static WR01 release-surface compliance refresh failed "
            f"with exit={result.returncode}: {result.stderr[-4000:]}"
        )
    return {
        "schema": "wr01_static_compliance_refresh_v1",
        "command": _one_liner([".venv/bin/python", *cmd[1:]]),
        "json_out": _repo_display_path(output),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
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


def _compliance_failure_summary(failed_checks: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, list[str]] = {
        "release_surface_missing": [],
        "archive_manifest_mismatch": [],
        "auth_eval_or_report_missing": [],
        "archive_integrity": [],
        "other": [],
    }
    for check in failed_checks:
        name = str(check.get("name") or "")
        if name.startswith("required_file_present:"):
            categories["release_surface_missing"].append(name)
        elif name in {"archive_manifest_sha_matches", "archive_manifest_size_matches"}:
            categories["archive_manifest_mismatch"].append(name)
        elif name in {"auth_eval_exists", "report_exists"}:
            categories["auth_eval_or_report_missing"].append(name)
        elif name.startswith("zip_") or name.startswith("archive_") or name.startswith("expected_archive_"):
            categories["archive_integrity"].append(name)
        else:
            categories["other"].append(name)
    return {
        "schema": "wr01_compliance_failure_summary_v1",
        "failed_count": len(failed_checks),
        "failed_check_names": [str(check.get("name") or "") for check in failed_checks],
        "categories": {key: sorted(values) for key, values in categories.items()},
        "release_surface_only": bool(
            failed_checks
            and not categories["archive_integrity"]
            and not categories["other"]
        ),
    }


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


def _queue_metadata_diagnostics(
    dry_run_payload: dict[str, Any],
    *,
    lane_id: str,
    manifest: Path,
    preflight: Path,
    payload_diff: Path,
) -> dict[str, Any]:
    queue = _get_path(dry_run_payload, "spec", "queue_metadata")
    source = "spec.queue_metadata"
    if not isinstance(queue, dict):
        queue = _get_path(dry_run_payload, "queue", "queue_metadata")
        source = "queue.queue_metadata"
    queue = queue if isinstance(queue, dict) else {}
    expected = {
        "lane": lane_id,
        "archive_manifest": manifest.as_posix(),
        "public_preflight": preflight.as_posix(),
        "payload_section_diff": payload_diff.as_posix(),
    }
    actual = {key: queue.get(key) for key in expected}
    mismatches = [
        {
            "key": key,
            "actual": actual[key],
            "expected": expected[key],
            "missing": key not in queue,
        }
        for key in expected
        if actual[key] != expected[key]
    ]
    return {
        "schema": "wr01_dry_run_queue_metadata_diagnostics_v1",
        "source": source,
        "actual": actual,
        "expected": expected,
        "mismatches": mismatches,
        "stale_missing_payload_section_diff_only": (
            len(mismatches) == 1
            and mismatches[0]["key"] == "payload_section_diff"
            and mismatches[0]["missing"] is True
        ),
    }


def _dry_run_submit_readiness(payload: dict[str, Any]) -> dict[str, Any]:
    readiness = payload.get("submit_readiness")
    readiness = readiness if isinstance(readiness, dict) else {}
    raw_blockers = readiness.get("blockers")
    blockers = [str(item) for item in raw_blockers] if isinstance(raw_blockers, list) else []
    remote_preflight_only = bool(blockers) and all(
        item.startswith("missing --remote-preflight-ssh-target") for item in blockers
    )
    ok = readiness.get("ok") is True
    return {
        "schema": "wr01_dry_run_submit_readiness_v1",
        "ok": ok,
        "blockers": blockers,
        "remote_preflight_only": remote_preflight_only,
        "static_ok": ok or remote_preflight_only,
        "note": (
            "remote preflight alias is a submit-time environment blocker, not "
            "a candidate artifact blocker"
            if remote_preflight_only
            else None
        ),
    }


def _rate_only_priority_reference(path: Path) -> dict[str, Any]:
    full = _repo_path(path)
    if not full.is_file():
        return {
            "schema": "wr01_rate_only_priority_reference_v1",
            "path": _repo_display_path(path),
            "exists": False,
            "static_packet_ready": False,
            "byte_delta": None,
            "rate_only": False,
            "blockers": ["rate_only_priority_packet_missing"],
        }
    payload = _read_json(path)
    byte_delta = payload.get("byte_delta")
    static_blockers = payload.get("static_blockers")
    static_blockers = static_blockers if isinstance(static_blockers, list) else []
    pareto_scope = str(payload.get("pareto_scope") or "")
    family = str(payload.get("family") or "")
    rate_only = "rate_only" in pareto_scope or "lowlevel_brotli_repack" in family
    return {
        "schema": "wr01_rate_only_priority_reference_v1",
        "path": _repo_display_path(path),
        "exists": True,
        "candidate_id": payload.get("candidate_id") or payload.get("lane_id"),
        "family": family,
        "pareto_scope": pareto_scope,
        "evidence_grade": payload.get("evidence_grade"),
        "score_claim": payload.get("score_claim"),
        "dispatch_attempted": payload.get("dispatch_attempted"),
        "static_packet_ready": payload.get("static_packet_ready") is True and not static_blockers,
        "ready_for_submit": payload.get("ready_for_submit") is True,
        "byte_delta": byte_delta if isinstance(byte_delta, int) and not isinstance(byte_delta, bool) else None,
        "archive_sha256": payload.get("archive_sha256"),
        "archive_bytes": payload.get("archive_bytes"),
        "source_archive_sha256": payload.get("source_archive_sha256"),
        "source_archive_bytes": payload.get("source_archive_bytes"),
        "rate_only": rate_only,
        "blockers": static_blockers,
    }


def _adversarial_priority_review(
    *,
    rate_only_priority_packet: Path,
    wr01_byte_delta: int | None,
    static_packet_ready: bool,
) -> dict[str, Any]:
    reference = _rate_only_priority_reference(rate_only_priority_packet)
    ref_delta = reference.get("byte_delta")
    rate_only_preempts = (
        static_packet_ready
        and reference.get("exists") is True
        and reference.get("static_packet_ready") is True
        and reference.get("rate_only") is True
        and isinstance(wr01_byte_delta, int)
        and isinstance(ref_delta, int)
        and ref_delta < wr01_byte_delta
    )
    blockers = (
        ["adversarial_priority_review_prioritizes_rate_only_candidate"]
        if rate_only_preempts
        else []
    )
    return {
        "schema": "wr01_adversarial_priority_review_v1",
        "ready": not blockers,
        "blockers": blockers,
        "wr01_candidate": {
            "byte_delta": wr01_byte_delta,
            "static_packet_ready": static_packet_ready,
            "score_family": "scorer_changing_wavelet_apply_transform",
            "component_delta_claim": False,
        },
        "rate_only_reference": reference,
        "fastest_safe_score_lowering_path": (
            "dispatch_hnerv_rate_only_q10_before_wr01_if_lane_claim_and_env_clear"
            if rate_only_preempts
            else "wr01_may_continue_after_operator_env_claim_cuda_gates"
        ),
        "priority_decision": (
            "defer_wr01_behind_hnerv_rate_only_reference"
            if rate_only_preempts
            else "wr01_not_preempted_by_configured_rate_only_reference"
        ),
        "failure_modes": [
            "wr01_changes_decoded_latent_sidecar_bytes_so_segnet_or_posenet_drift_can_overwhelm_the_9_byte_rate_win",
            "local_runtime_decode_validation_is_not_score_evidence",
            "missing_lane_claim_or_lightning_environment_can_create_dispatch_conflicts_or_unharvestable_jobs",
            "rate_only_reference_still_requires_exact_cuda_auth_eval_before_any_score_claim",
        ],
        "adversarial_notes": [
            "Prefer a byte-closed rate-only HNeRV packet with a larger negative byte_delta before scorer-changing WR01.",
            "WR01 remains a byte-custody exact-eval candidate; this gate only orders scarce exact-CUDA wall-clock.",
        ],
    }


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


def _dispatch_blockers(payload: dict[str, Any] | Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("dispatch_blockers")
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _check_required_runtime_dispatch_blockers(
    checks: list[dict[str, Any]],
    blockers: list[str],
    prefix: str,
    payload: dict[str, Any] | Any,
) -> None:
    actual = _dispatch_blockers(payload)
    for required in RUNTIME_REQUIRED_DISPATCH_BLOCKERS:
        _check_condition(
            checks,
            blockers,
            f"{prefix}_missing_dispatch_blocker:{required}",
            required in actual,
            actual=actual,
            expected=f"contains {required}",
        )


def _release_surface_manifest_consistency(
    *,
    release_surface_dir: Path,
    archive_identity: dict[str, Any],
    manifest: Path,
    preflight: Path,
    payload_diff: Path,
    runtime_decode_validation: Path,
    runtime_decode_review: Path,
) -> dict[str, Any]:
    """Validate an existing static release manifest against packet custody.

    The release surface is optional for synthetic packet fixtures, but once an
    archive_manifest.json exists it must not become a stale side record.
    """

    blockers: list[str] = []
    checks: list[dict[str, Any]] = []
    path = release_surface_dir / "archive_manifest.json"
    full = _repo_path(path)
    exists = full.is_file()
    payload = _read_json(path) if exists else {}
    if not exists:
        return {
            "schema": "wr01_release_surface_manifest_consistency_v1",
            "path": _repo_display_path(path),
            "exists": False,
            "ready": True,
            "blockers": [],
            "checks": [],
            "note": "release surface manifest absent; no stale manifest to validate",
        }

    _check_equal(
        checks,
        blockers,
        "release_surface_manifest_score_claim_not_false",
        payload.get("score_claim"),
        False,
    )
    _check_equal(
        checks,
        blockers,
        "release_surface_manifest_dispatch_attempted_not_false",
        payload.get("dispatch_attempted"),
        False,
    )
    _check_equal(
        checks,
        blockers,
        "release_surface_manifest_remote_gpu_run_not_false",
        payload.get("remote_gpu_run"),
        False,
    )
    _check_equal(
        checks,
        blockers,
        "release_surface_manifest_candidate_archive_sha256_mismatch",
        payload.get("candidate_archive_sha256"),
        archive_identity["sha256"],
    )
    _check_equal(
        checks,
        blockers,
        "release_surface_manifest_candidate_archive_bytes_mismatch",
        payload.get("candidate_archive_bytes"),
        archive_identity["bytes"],
    )
    _check_equal(
        checks,
        blockers,
        "release_surface_manifest_archive_sha256_mismatch",
        _get_path(payload, "archive", "sha256"),
        archive_identity["sha256"],
    )
    _check_equal(
        checks,
        blockers,
        "release_surface_manifest_archive_bytes_mismatch",
        _get_path(payload, "archive", "bytes"),
        archive_identity["bytes"],
    )
    required_links = {
        "candidate_manifest": manifest,
        "public_replay_preflight": preflight,
        "payload_section_diff": payload_diff,
        "runtime_decode_validation": runtime_decode_validation,
        "runtime_decode_review": runtime_decode_review,
    }
    links = payload.get("manifest_links")
    links = links if isinstance(links, dict) else {}
    for name, expected_path in required_links.items():
        link = links.get(name)
        link = link if isinstance(link, dict) else {}
        _check_equal(
            checks,
            blockers,
            f"release_surface_manifest_link_missing:{name}",
            link.get("exists"),
            True,
        )
        _check_condition(
            checks,
            blockers,
            f"release_surface_manifest_link_path_mismatch:{name}",
            _paths_equivalent(link.get("repo_path"), expected_path),
            actual=link.get("repo_path"),
            expected=_repo_display_path(expected_path),
        )
    return {
        "schema": "wr01_release_surface_manifest_consistency_v1",
        "path": _repo_display_path(path),
        "exists": True,
        "sha256": _sha256_file(path),
        "bytes": full.stat().st_size,
        "ready": not blockers,
        "blockers": blockers,
        "checks": checks,
    }


def _check_declared_manifest_sha256(
    checks: list[dict[str, Any]],
    blockers: list[str],
    prefix: str,
    payload: dict[str, Any],
) -> str | None:
    declared = payload.get("manifest_sha256_excluding_self")
    _check_condition(
        checks,
        blockers,
        f"{prefix}_manifest_sha256_missing_or_malformed",
        _is_sha256(declared),
        actual=declared,
        expected="64-char lowercase hex sha256",
    )
    computed = _manifest_sha256_excluding_self(payload)
    if _is_sha256(declared):
        _check_equal(
            checks,
            blockers,
            f"{prefix}_manifest_sha256_mismatch",
            declared,
            computed,
        )
    return computed


def _expected_runtime_apply_manifest_hashes(manifest_payload: dict[str, Any]) -> list[str]:
    hashes = {
        _sha256_json_line(manifest_payload),
        _manifest_sha256_excluding_self(manifest_payload),
    }
    declared = manifest_payload.get("manifest_sha256_excluding_self")
    if _is_sha256(declared):
        hashes.add(str(declared))
    return sorted(hashes)


def _runtime_decode_gate_diagnostics(
    *,
    result_dir: Path,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
    runtime_decode_validation_path: Path,
    runtime_decode_validation_payload: dict[str, Any],
    runtime_decode_review_path: Path,
    runtime_decode_review_payload: dict[str, Any],
    archive_identity: dict[str, Any],
    changed_section_name: Any,
    changed_section_source_sha256: Any,
    changed_section_sha256: Any,
) -> dict[str, Any]:
    gate_blockers: list[str] = []
    checks: list[dict[str, Any]] = []
    validation_exists = _repo_path(runtime_decode_validation_path).is_file()
    review_exists = _repo_path(runtime_decode_review_path).is_file()
    expected_runtime_apply_path = (
        Path(str(manifest_payload.get("manifest_path")))
        if isinstance(manifest_payload.get("manifest_path"), str)
        and manifest_payload.get("manifest_path")
        else manifest_path
    )

    _check_condition(
        checks,
        gate_blockers,
        "runtime_decode_validation_file_missing",
        validation_exists,
        actual=_repo_display_path(runtime_decode_validation_path),
        expected="existing WR01 runtime decode validation manifest",
    )
    _check_condition(
        checks,
        gate_blockers,
        "runtime_decode_review_file_missing",
        review_exists,
        actual=_repo_display_path(runtime_decode_review_path),
        expected="existing WR01 runtime apply/decode review manifest",
    )
    _check_present(
        checks,
        gate_blockers,
        "manifest_runtime_decode_validation_manifest_path_missing",
        manifest_payload.get("runtime_decode_validation_manifest_path"),
    )
    _check_condition(
        checks,
        gate_blockers,
        "manifest_runtime_decode_validation_manifest_path_mismatch",
        _paths_equivalent(
            manifest_payload.get("runtime_decode_validation_manifest_path"),
            runtime_decode_validation_path,
        ),
        actual=manifest_payload.get("runtime_decode_validation_manifest_path"),
        expected=_repo_display_path(runtime_decode_validation_path),
    )
    _check_present(
        checks,
        gate_blockers,
        "manifest_runtime_apply_manifest_path_missing",
        manifest_payload.get("manifest_path"),
    )
    if _is_sha256(manifest_payload.get("manifest_sha256_excluding_self")):
        _check_equal(
            checks,
            gate_blockers,
            "manifest_manifest_sha256_mismatch",
            manifest_payload.get("manifest_sha256_excluding_self"),
            _manifest_sha256_excluding_self(manifest_payload),
        )
    _check_equal(
        checks,
        gate_blockers,
        "manifest_tool_mismatch",
        manifest_payload.get("tool"),
        APPLY_TRANSFORM_TOOL,
    )
    _check_equal(
        checks,
        gate_blockers,
        "manifest_score_claim_not_false",
        manifest_payload.get("score_claim"),
        False,
    )
    _check_equal(
        checks,
        gate_blockers,
        "manifest_dispatch_attempted_not_false",
        manifest_payload.get("dispatch_attempted"),
        False,
    )
    _check_equal(
        checks,
        gate_blockers,
        "manifest_ready_for_archive_preflight_not_false",
        manifest_payload.get("ready_for_archive_preflight"),
        False,
    )
    _check_equal(
        checks,
        gate_blockers,
        "manifest_ready_for_exact_eval_dispatch_not_false",
        manifest_payload.get("ready_for_exact_eval_dispatch"),
        False,
    )
    _check_required_runtime_dispatch_blockers(
        checks,
        gate_blockers,
        "manifest",
        manifest_payload,
    )

    runtime_apply = manifest_payload.get("runtime_apply")
    _check_condition(
        checks,
        gate_blockers,
        "runtime_apply_block_missing",
        isinstance(runtime_apply, dict),
        actual=type(runtime_apply).__name__,
        expected="object",
    )
    if isinstance(runtime_apply, dict):
        _check_equal(checks, gate_blockers, "runtime_apply_schema_mismatch", runtime_apply.get("schema"), RUNTIME_APPLY_SCHEMA)
        _check_equal(checks, gate_blockers, "runtime_apply_status_mismatch", runtime_apply.get("status"), "applied")
        _check_equal(
            checks,
            gate_blockers,
            "runtime_apply_ready_for_runtime_apply_review_not_true",
            runtime_apply.get("ready_for_runtime_apply_review"),
            True,
        )
        _check_equal(checks, gate_blockers, "runtime_apply_score_claim_not_false", runtime_apply.get("score_claim"), False)
        _check_equal(
            checks,
            gate_blockers,
            "runtime_apply_dispatch_attempted_not_false",
            runtime_apply.get("dispatch_attempted"),
            False,
        )
        _check_equal(checks, gate_blockers, "runtime_apply_section_name_mismatch", runtime_apply.get("section_name"), changed_section_name)
        _check_equal(
            checks,
            gate_blockers,
            "runtime_apply_source_section_sha256_mismatch",
            runtime_apply.get("source_section_sha256"),
            changed_section_source_sha256,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_apply_candidate_section_sha256_mismatch",
            runtime_apply.get("candidate_section_sha256"),
            changed_section_sha256,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_apply_source_raw_sha256_mismatch",
            runtime_apply.get("source_raw_sha256"),
            manifest_payload.get("source_raw_sha256"),
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_apply_candidate_raw_sha256_mismatch",
            runtime_apply.get("candidate_raw_sha256"),
            manifest_payload.get("candidate_raw_sha256"),
        )
        applied_atom_ids = runtime_apply.get("applied_atom_ids")
        _check_condition(
            checks,
            gate_blockers,
            "runtime_apply_applied_atom_ids_missing",
            isinstance(applied_atom_ids, list) and bool(applied_atom_ids),
            actual=applied_atom_ids,
            expected="nonempty list",
        )
        _check_condition(
            checks,
            gate_blockers,
            "runtime_apply_applied_atom_count_not_positive",
            _is_positive_int(runtime_apply.get("applied_atom_count")),
            actual=runtime_apply.get("applied_atom_count"),
            expected="positive integer",
        )

    validation_sha256 = None
    if validation_exists:
        validation_sha256 = _check_declared_manifest_sha256(
            checks,
            gate_blockers,
            "runtime_decode_validation",
            runtime_decode_validation_payload,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_schema_mismatch",
            runtime_decode_validation_payload.get("schema"),
            RUNTIME_DECODE_VALIDATION_SCHEMA,
        )
        _check_condition(
            checks,
            gate_blockers,
            "runtime_decode_validation_manifest_path_mismatch",
            _paths_equivalent(
                runtime_decode_validation_payload.get("manifest_path"),
                runtime_decode_validation_path,
            ),
            actual=runtime_decode_validation_payload.get("manifest_path"),
            expected=_repo_display_path(runtime_decode_validation_path),
        )
        _check_equal(
            checks,
            gate_blockers,
            "manifest_runtime_decode_validation_manifest_sha256_mismatch",
            manifest_payload.get("runtime_decode_validation_manifest_sha256"),
            validation_sha256,
        )
        embedded_validation = manifest_payload.get("runtime_decode_validation")
        _check_condition(
            checks,
            gate_blockers,
            "manifest_runtime_decode_validation_block_missing",
            isinstance(embedded_validation, dict),
            actual=type(embedded_validation).__name__,
            expected="object",
        )
        if isinstance(embedded_validation, dict):
            _check_equal(
                checks,
                gate_blockers,
                "manifest_runtime_decode_validation_embedded_sha256_mismatch",
                _manifest_sha256_excluding_self(embedded_validation),
                validation_sha256,
            )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_ready_for_runtime_decode_review_not_true",
            runtime_decode_validation_payload.get("ready_for_runtime_decode_review"),
            True,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_ready_for_archive_preflight_not_false",
            runtime_decode_validation_payload.get("ready_for_archive_preflight"),
            False,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_ready_for_exact_eval_dispatch_not_false",
            runtime_decode_validation_payload.get("ready_for_exact_eval_dispatch"),
            False,
        )
        _check_equal(checks, gate_blockers, "runtime_decode_validation_score_claim_not_false", runtime_decode_validation_payload.get("score_claim"), False)
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_dispatch_attempted_not_false",
            runtime_decode_validation_payload.get("dispatch_attempted"),
            False,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_exact_cuda_auth_eval_not_false",
            runtime_decode_validation_payload.get("exact_cuda_auth_eval"),
            False,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_blockers_not_empty",
            runtime_decode_validation_payload.get("blockers") or [],
            [],
        )
        _check_required_runtime_dispatch_blockers(
            checks,
            gate_blockers,
            "runtime_decode_validation",
            runtime_decode_validation_payload,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_source_archive_sha256_mismatch",
            runtime_decode_validation_payload.get("source_archive_sha256"),
            manifest_payload.get("source_archive_sha256"),
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_source_archive_bytes_mismatch",
            runtime_decode_validation_payload.get("source_archive_bytes"),
            manifest_payload.get("source_archive_bytes"),
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_candidate_archive_sha256_mismatch",
            runtime_decode_validation_payload.get("candidate_archive_sha256"),
            archive_identity["sha256"],
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_candidate_archive_bytes_mismatch",
            runtime_decode_validation_payload.get("candidate_archive_bytes"),
            archive_identity["bytes"],
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_source_payload_sha256_mismatch",
            runtime_decode_validation_payload.get("source_payload_sha256"),
            manifest_payload.get("source_payload_sha256"),
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_candidate_payload_sha256_mismatch",
            runtime_decode_validation_payload.get("candidate_payload_sha256"),
            manifest_payload.get("candidate_payload_sha256"),
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_section_name_mismatch",
            runtime_decode_validation_payload.get("section_name"),
            changed_section_name,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_validation_changed_section_only_not_true",
            runtime_decode_validation_payload.get("changed_section_only"),
            True,
        )

    review_sha256 = None
    if review_exists:
        review_sha256 = _check_declared_manifest_sha256(
            checks,
            gate_blockers,
            "runtime_decode_review",
            runtime_decode_review_payload,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_schema_mismatch",
            runtime_decode_review_payload.get("schema"),
            RUNTIME_DECODE_REVIEW_SCHEMA,
        )
        _check_condition(
            checks,
            gate_blockers,
            "runtime_decode_review_manifest_path_mismatch",
            _paths_equivalent(runtime_decode_review_payload.get("manifest_path"), runtime_decode_review_path),
            actual=runtime_decode_review_payload.get("manifest_path"),
            expected=_repo_display_path(runtime_decode_review_path),
        )
        _check_equal(checks, gate_blockers, "runtime_decode_review_status_mismatch", runtime_decode_review_payload.get("status"), "ready")
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_ready_for_runtime_apply_review_not_true",
            runtime_decode_review_payload.get("ready_for_runtime_apply_review"),
            True,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_ready_for_decode_validation_review_not_true",
            runtime_decode_review_payload.get("ready_for_decode_validation_review"),
            True,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_ready_for_archive_preflight_not_false",
            runtime_decode_review_payload.get("ready_for_archive_preflight"),
            False,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_ready_for_exact_eval_dispatch_not_false",
            runtime_decode_review_payload.get("ready_for_exact_eval_dispatch"),
            False,
        )
        _check_equal(checks, gate_blockers, "runtime_decode_review_score_claim_not_false", runtime_decode_review_payload.get("score_claim"), False)
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_dispatch_attempted_not_false",
            runtime_decode_review_payload.get("dispatch_attempted"),
            False,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_blockers_not_empty",
            runtime_decode_review_payload.get("blockers") or [],
            [],
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_runtime_apply_blockers_not_empty",
            runtime_decode_review_payload.get("runtime_apply_blockers") or [],
            [],
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_decode_validation_blockers_not_empty",
            runtime_decode_review_payload.get("decode_validation_blockers") or [],
            [],
        )
        _check_required_runtime_dispatch_blockers(
            checks,
            gate_blockers,
            "runtime_decode_review",
            runtime_decode_review_payload,
        )
        _check_condition(
            checks,
            gate_blockers,
            "runtime_decode_review_apply_manifest_path_mismatch",
            _paths_equivalent(
                runtime_decode_review_payload.get("runtime_apply_manifest_path"),
                expected_runtime_apply_path,
            ),
            actual=runtime_decode_review_payload.get("runtime_apply_manifest_path"),
            expected=_repo_display_path(expected_runtime_apply_path),
        )
        expected_apply_hashes = _expected_runtime_apply_manifest_hashes(manifest_payload)
        _check_condition(
            checks,
            gate_blockers,
            "runtime_decode_review_apply_manifest_sha256_mismatch",
            runtime_decode_review_payload.get("runtime_apply_manifest_sha256")
            in expected_apply_hashes,
            actual=runtime_decode_review_payload.get("runtime_apply_manifest_sha256"),
            expected=expected_apply_hashes,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_runtime_decode_validation_schema_mismatch",
            runtime_decode_review_payload.get("runtime_decode_validation_schema"),
            RUNTIME_DECODE_VALIDATION_SCHEMA,
        )
        _check_condition(
            checks,
            gate_blockers,
            "runtime_decode_review_runtime_decode_validation_path_mismatch",
            _paths_equivalent(
                runtime_decode_review_payload.get("runtime_decode_validation_manifest_path"),
                runtime_decode_validation_path,
            ),
            actual=runtime_decode_review_payload.get("runtime_decode_validation_manifest_path"),
            expected=_repo_display_path(runtime_decode_validation_path),
        )
        if validation_sha256 is not None:
            _check_equal(
                checks,
                gate_blockers,
                "runtime_decode_review_runtime_decode_validation_sha256_mismatch",
                runtime_decode_review_payload.get("runtime_decode_validation_manifest_sha256"),
                validation_sha256,
            )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_runtime_decode_validation_ready_not_true",
            runtime_decode_review_payload.get("runtime_decode_validation_ready"),
            True,
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_candidate_archive_sha256_mismatch",
            runtime_decode_review_payload.get("runtime_apply_candidate_archive_sha256"),
            archive_identity["sha256"],
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_candidate_archive_bytes_mismatch",
            runtime_decode_review_payload.get("runtime_apply_candidate_archive_bytes"),
            archive_identity["bytes"],
        )
        _check_equal(
            checks,
            gate_blockers,
            "runtime_decode_review_changed_section_name_mismatch",
            runtime_decode_review_payload.get("runtime_apply_changed_section_name"),
            changed_section_name,
        )

    return {
        "schema": "wr01_runtime_apply_decode_gate_v1",
        "result_dir": _repo_display_path(result_dir),
        "runtime_decode_validation_path": _repo_display_path(runtime_decode_validation_path),
        "runtime_decode_review_path": _repo_display_path(runtime_decode_review_path),
        "runtime_decode_validation_exists": validation_exists,
        "runtime_decode_review_exists": review_exists,
        "runtime_decode_validation_manifest_sha256": validation_sha256,
        "runtime_decode_review_manifest_sha256": review_sha256,
        "ready": not gate_blockers,
        "blockers": gate_blockers,
        "checks": checks,
    }


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
    manifest_payload = _read_json(manifest) if _repo_path(manifest).is_file() else {}
    runtime_decode_validation = _runtime_decode_validation_path(result_dir, manifest_payload)
    runtime_decode_review = _runtime_decode_review_path(result_dir)

    missing_env = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    artifacts = [
        archive,
        args.baseline_json,
        manifest,
        runtime_decode_validation,
        runtime_decode_review,
        preflight,
        compliance,
        payload_diff,
        dry_run,
        strength_summary,
    ]
    artifact_statuses = [_artifact_status(path) for path in artifacts]
    missing_artifacts = [row["path"] for row in artifact_statuses if not row["exists"]]
    runtime_decode_validation_payload = (
        _read_json(runtime_decode_validation) if _repo_path(runtime_decode_validation).is_file() else {}
    )
    runtime_decode_review_payload = (
        _read_json(runtime_decode_review) if _repo_path(runtime_decode_review).is_file() else {}
    )
    preflight_payload = _read_json(preflight) if _repo_path(preflight).is_file() else {}
    compliance_payload = _read_json(compliance) if _repo_path(compliance).is_file() else {}
    payload_diff_payload = _read_json(payload_diff) if _repo_path(payload_diff).is_file() else {}
    dry_run_payload = _read_json(dry_run) if _repo_path(dry_run).is_file() else {}
    artifact_flag_violations = _artifact_flag_violations(
        {
            "manifest": manifest_payload,
            "runtime_decode_validation": runtime_decode_validation_payload,
            "runtime_decode_review": runtime_decode_review_payload,
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
    compliance_failure_summary = _compliance_failure_summary(failed_compliance_checks)
    compliance_ok = compliance_payload.get("passed") is True and not failed_compliance_checks
    payload_diff_ready = (
        payload_diff_payload.get("ready_for_archive_preflight") is True
        and payload_diff_payload.get("changed_section_count") == 1
        and not payload_diff_payload.get("blockers")
    )
    dry_run_submit_readiness = _dry_run_submit_readiness(dry_run_payload)
    dry_run_ready = dry_run_submit_readiness["static_ok"]
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
    dry_run_queue_metadata = _queue_metadata_diagnostics(
        dry_run_payload,
        lane_id=args.lane_id,
        manifest=manifest,
        preflight=preflight,
        payload_diff=payload_diff,
    )
    release_surface_manifest_consistency = _release_surface_manifest_consistency(
        release_surface_dir=args.release_surface_dir,
        archive_identity=archive_identity,
        manifest=manifest,
        preflight=preflight,
        payload_diff=payload_diff,
        runtime_decode_validation=runtime_decode_validation,
        runtime_decode_review=runtime_decode_review,
    )
    runtime_decode_gate = _runtime_decode_gate_diagnostics(
        result_dir=result_dir,
        manifest_path=manifest,
        manifest_payload=manifest_payload,
        runtime_decode_validation_path=runtime_decode_validation,
        runtime_decode_validation_payload=runtime_decode_validation_payload,
        runtime_decode_review_path=runtime_decode_review,
        runtime_decode_review_payload=runtime_decode_review_payload,
        archive_identity=archive_identity,
        changed_section_name=changed_section_name,
        changed_section_source_sha256=changed_section_source_sha256,
        changed_section_sha256=changed_section_sha256,
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
    compliance_manifest_path = _get_path(compliance_payload, "archive_manifest", "path")
    allowed_compliance_manifest_paths = [
        manifest.as_posix(),
        (args.release_surface_dir / "archive_manifest.json").as_posix(),
    ]
    _check_condition(
        consistency_checks,
        consistency_blockers,
        "compliance_manifest_path_mismatch",
        compliance_manifest_path in allowed_compliance_manifest_paths,
        actual=compliance_manifest_path,
        expected=allowed_compliance_manifest_paths,
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
    packet_path = _packet_output_path(args)
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
        "--status",
        "active_exact_eval",
        "--notes",
        notes,
    ]
    if args.predicted_eta_utc:
        status_index = claim_cmd.index("--status")
        claim_cmd[status_index:status_index] = [
            "--predicted-eta-utc",
            args.predicted_eta_utc,
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
        runtime_decode_validation.as_posix(),
        "--extra-artifact",
        runtime_decode_review.as_posix(),
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
    if not release_surface_manifest_consistency["ready"]:
        blockers.append("release_surface_manifest_not_ready")
    if not runtime_decode_gate["ready"]:
        blockers.append("runtime_decode_gate_not_ready")
    if artifact_flag_violations:
        blockers.append("artifact_score_or_dispatch_flag_violation")
    blockers.extend(consistency_blockers)
    blockers.extend(release_surface_manifest_consistency["blockers"])
    blockers.extend(runtime_decode_gate["blockers"])
    static_blockers = list(blockers)
    static_packet_ready = not static_blockers
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
    byte_custody_exact_eval_candidate_ready = static_packet_ready
    adversarial_priority_review = _adversarial_priority_review(
        rate_only_priority_packet=args.rate_only_priority_packet,
        wr01_byte_delta=archive_byte_delta,
        static_packet_ready=static_packet_ready,
    )
    operator_lane_blockers = []
    if missing_env:
        operator_lane_blockers.append("missing_lightning_environment")
    if lane_claim_preflight["conflict_present"]:
        operator_lane_blockers.append("active_lane_dispatch_claim_conflict")
    if not lane_claim_preflight["active_claim_present"]:
        operator_lane_blockers.append("missing_active_lane_dispatch_claim")
    if not args.operator_approved_exact_cuda:
        operator_lane_blockers.append("missing_operator_exact_cuda_approval")
    operator_lane_blockers.extend(adversarial_priority_review["blockers"])
    blockers.extend(operator_lane_blockers)
    ready_for_exact_eval_dispatch = static_packet_ready and not operator_lane_blockers
    dispatch_gate = (
        "blocked_static_packet_ready_until_static_blockers_clear"
        if not static_packet_ready
        else (
            "eligible_for_cuda_auth_eval_after_lane_claim"
            if ready_for_exact_eval_dispatch
            else "blocked_operator_lane_gates_until_env_claim_approval"
        )
    )
    operator_next_steps = {
        "schema": "wr01_operator_next_steps_v1",
        "copy_safe": True,
        "must_run_in_order": True,
        "first_remote_gpu_step": "submit_exact_cuda",
        "packet_path": packet_path.as_posix(),
        "current_blockers": list(blockers),
        "steps": [
            {
                "order": 1,
                "id": "verify_lightning_env",
                "dispatches_remote_gpu": False,
                "writes_repo_state": False,
                "purpose": "fail loudly until all Lightning identity and path env vars are loaded",
                "copy_safe_command": _one_liner(_required_env_check_cmd()),
            },
            {
                "order": 2,
                "id": "refresh_static_packet_no_dispatch",
                "dispatches_remote_gpu": False,
                "writes_repo_state": True,
                "purpose": "rebuild the static release surface, static compliance JSON, and packet before claiming",
                "copy_safe_command": _one_liner(
                    _packet_refresh_cmd(
                        args,
                        packet_path=packet_path,
                        operator_approved_exact_cuda=False,
                    )
                ),
            },
            {
                "order": 3,
                "id": "review_adversarial_priority",
                "dispatches_remote_gpu": False,
                "writes_repo_state": False,
                "purpose": "fail loudly if a safer byte-closed HNeRV rate-only packet should take the next exact-CUDA slot",
                "copy_safe_command": _one_liner(_adversarial_priority_check_cmd(packet_path)),
            },
            {
                "order": 4,
                "id": "claim_lane_no_dispatch",
                "dispatches_remote_gpu": False,
                "writes_repo_state": True,
                "purpose": "record the Level-2 lane claim; this must fail if a same-lane active claim exists",
                "copy_safe_command": _one_liner(claim_cmd),
            },
            {
                "order": 5,
                "id": "refresh_packet_with_operator_exact_cuda_approval",
                "dispatches_remote_gpu": False,
                "writes_repo_state": True,
                "purpose": "operator's explicit exact-CUDA approval; still no remote job is submitted",
                "copy_safe_command": _one_liner(
                    _packet_refresh_cmd(
                        args,
                        packet_path=packet_path,
                        operator_approved_exact_cuda=True,
                    )
                ),
            },
            {
                "order": 6,
                "id": "assert_packet_ready_for_submit",
                "dispatches_remote_gpu": False,
                "writes_repo_state": False,
                "purpose": "fail loudly if env, claim, approval, or static custody drifted before submit",
                "copy_safe_command": _one_liner(_packet_ready_check_cmd(packet_path)),
            },
            {
                "order": 7,
                "id": "submit_exact_cuda",
                "dispatches_remote_gpu": True,
                "writes_repo_state": True,
                "purpose": "first remote/GPU action; run only after assert_packet_ready_for_submit passes",
                "copy_safe_command": _one_liner(submit_cmd),
            },
            {
                "order": 8,
                "id": "harvest_after_completion",
                "dispatches_remote_gpu": False,
                "writes_repo_state": True,
                "purpose": "harvest only after the Lightning exact CUDA job finishes",
                "copy_safe_command": _one_liner(harvest_cmd),
            },
        ],
    }
    return {
        "schema_version": 1,
        "schema": "wr01_exact_eval_operator_packet_v1",
        "packet_kind": "wr01_exact_eval_operator_packet",
        "tool": "tools/build_wr01_exact_eval_packet.py",
        "recorded_at_utc": _format_utc(now_utc),
        "candidate_id": args.lane_id,
        "family": "hnerv_wavelet_wr01_apply_transform",
        "family_group": "hnerv_wavelet_wr01_apply",
        "pareto_scope": "hnerv_wavelet_apply_transform_exact_archive",
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
            "scorer-visible single-member archive transform; no SegNet or PoseNet delta is claimed before exact CUDA",
            "archive and runtime parity are necessary but exact CUDA auth eval is required before any score claim",
            "no composability assumption with categorical, sensitivity, or pose atoms until stacked archive eval exists",
        ],
        "conflicts_with_families": [],
        "conflicts_with_atoms": [],
        "score_claim": False,
        "dispatch_attempted": False,
        "dispatch_gate": dispatch_gate,
        "dispatch_unlocked": ready_for_exact_eval_dispatch,
        "ready_for_exact_eval_dispatch": ready_for_exact_eval_dispatch,
        "ready_for_exact_eval_dispatch_claim": ready_for_exact_eval_dispatch,
        "byte_custody_exact_eval_candidate_ready": byte_custody_exact_eval_candidate_ready,
        "candidate_static_preflight_ready": static_packet_ready,
        "ready_for_submit": not blockers,
        "static_packet_ready": static_packet_ready,
        "blockers": blockers,
        "static_blockers": static_blockers,
        "operator_lane_blockers": operator_lane_blockers,
        "missing_env": missing_env,
        "missing_artifacts": missing_artifacts,
        "operator_approved_exact_cuda": bool(args.operator_approved_exact_cuda),
        "adversarial_priority_review": adversarial_priority_review,
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
        "compliance_failure_summary": compliance_failure_summary,
        "payload_diff_ready": payload_diff_ready,
        "dry_run_ready": dry_run_ready,
        "dry_run_submit_readiness": dry_run_submit_readiness,
        "dry_run_queue_metadata": dry_run_queue_metadata,
        "release_surface_manifest_consistency": release_surface_manifest_consistency,
        "runtime_decode_gate_ready": runtime_decode_gate["ready"],
        "runtime_decode_gate_blockers": runtime_decode_gate["blockers"],
        "runtime_decode_gate": runtime_decode_gate,
        "runtime_decode_validation_path": _repo_display_path(runtime_decode_validation),
        "runtime_decode_review_path": _repo_display_path(runtime_decode_review),
        "artifact_flag_violations": artifact_flag_violations,
        "artifact_consistency_ok": not consistency_blockers,
        "artifact_consistency_checks": consistency_checks,
        "release_surface": _release_surface_status(args.release_surface_dir),
        "artifacts": artifact_statuses,
        "operator_next_steps": operator_next_steps,
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
    parser.add_argument(
        "--release-surface-dir",
        type=Path,
        help=(
            "Directory for the deterministic WR01 static release surface. "
            "Defaults to <result-dir>/release_surface."
        ),
    )
    parser.add_argument(
        "--build-release-surface",
        action="store_true",
        help="Write archive.zip, inflate.sh, report.txt, and archive_manifest.json under --release-surface-dir.",
    )
    parser.add_argument(
        "--refresh-static-compliance",
        action="store_true",
        help=(
            "After building the release surface, rerun non-dispatch static "
            "pre-submission compliance against it and overwrite "
            "<result-dir>/pre_submission_compliance.json."
        ),
    )
    parser.add_argument("--archive-sha256", default="d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628")
    parser.add_argument("--archive-bytes", type=int, default=186222)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--claim-ttl-hours", type=int, default=DEFAULT_CLAIM_TTL_HOURS)
    parser.add_argument(
        "--rate-only-priority-packet",
        type=Path,
        default=DEFAULT_RATE_ONLY_PRIORITY_PACKET,
        help=(
            "Optional HNeRV rate-only packet used by the adversarial priority "
            "gate. If it is static-ready and has a larger byte win than WR01, "
            "WR01 submit readiness stays blocked until that priority decision is resolved."
        ),
    )
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
    parser.add_argument(
        "--predicted-eta-utc",
        default="",
        help="Optional predicted ETA for the lane-claim row. Omitted from generated claim commands when empty.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.claim_ttl_hours <= 0:
        parser.error("--claim-ttl-hours must be positive")
    if args.now_utc and _parse_utc(args.now_utc) is None:
        parser.error("--now-utc must be ISO-8601 UTC-compatible, e.g. 2026-05-06T10:00:00Z")
    if args.release_surface_dir is None:
        args.release_surface_dir = args.result_dir / DEFAULT_RELEASE_SURFACE_SUBDIR
    generation_payload = None
    compliance_refresh_payload = None
    if args.build_release_surface or args.refresh_static_compliance:
        generation_payload = build_release_surface(args, now_utc=_now_utc(args))
    if args.refresh_static_compliance:
        compliance_refresh_payload = refresh_static_compliance(args)
    payload = build_packet(args)
    if generation_payload is not None:
        payload["release_surface_generation"] = generation_payload
    if compliance_refresh_payload is not None:
        payload["static_compliance_refresh"] = compliance_refresh_payload
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        _repo_path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _repo_path(args.json_out).write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
