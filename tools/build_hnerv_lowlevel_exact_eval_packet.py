#!/usr/bin/env python3
"""Build a static exact-eval packet for HNeRV low-level Brotli candidates.

This is a local custody/readiness tool. It writes deterministic static packet
artifacts and release-surface files, but it never claims a lane, submits a
remote/GPU job, runs CUDA, or claims a score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_lowlevel_packer import (  # noqa: E402
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
)
from tac.repo_io import json_text, read_json, repo_relative, sha256_file  # noqa: E402

DEFAULT_RESULT_DIR = Path(
    "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex"
)
DEFAULT_CANDIDATE_RESULT = DEFAULT_RESULT_DIR / "result.json"
DEFAULT_BASELINE_JSON = Path(
    "experiments/results/lightning_batch/"
    "exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_INFLATE_SH = Path(
    "experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh"
)
DEFAULT_UPSTREAM_DIR = Path("upstream")
DEFAULT_RELEASE_SURFACE_SUBDIR = "release_surface"
DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
DEFAULT_JOB_NAME = "exact_eval_pr106x_lgblock16_1byte_brotli_20260507"
DEFAULT_LANE_ID = "pr106x_lgblock16_1byte_brotli"
CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES
REQUIRED_ENV = (
    "LIGHTNING_SSH_TARGET",
    "LIGHTNING_REMOTE_PACT",
    "LIGHTNING_UPSTREAM_DIR",
    "LIGHTNING_TEAMSPACE",
    "LIGHTNING_STUDIO",
    "LIGHTNING_SDK_USER",
)
TERMINAL_CLAIM_PREFIXES = (
    "completed",
    "failed",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped",
)
BUILD_CODE_PATHS = (
    "tools/build_hnerv_lowlevel_exact_eval_packet.py",
    "src/tac/hnerv_lowlevel_packer.py",
    "src/tac/repo_io.py",
    "experiments/preflight_public_replay_intake.py",
    "experiments/preflight_candidate_manifest_dispatch_readiness.py",
    "scripts/pre_submission_compliance_check.py",
    "scripts/lightning_exact_eval_repro.py",
    "scripts/launch_lightning_batch_job.py",
    "tools/claim_lane_dispatch.py",
)
_CLAIM_SEPARATOR_RE = re.compile(r"^\|\s*-+\s*(\|\s*-+\s*)+\|\s*$")


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _archive_identity(path: Path) -> dict[str, Any]:
    full = _repo_path(path)
    return {
        "path": _repo_rel(path),
        "exists": full.is_file(),
        "sha256": sha256_file(full) if full.is_file() else None,
        "bytes": full.stat().st_size if full.is_file() else None,
    }


def _q(value: object) -> str:
    text = str(value)
    if text.startswith("$") or text.startswith("${"):
        return text
    return shlex.quote(text)


def _one_liner(cmd: list[object]) -> str:
    return " ".join(_q(item) for item in cmd)


def _packet_output_path(args: argparse.Namespace) -> Path:
    return args.json_out or args.result_dir / "hnerv_lowlevel_exact_eval_packet.json"


def _release_surface_status(release_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for rel in ("archive.zip", "inflate.sh", "report.txt", "archive_manifest.json"):
        path = release_dir / rel
        full = _repo_path(path)
        files[rel] = {
            "path": _repo_rel(path),
            "exists": full.is_file(),
            "bytes": full.stat().st_size if full.is_file() else None,
            "sha256": sha256_file(full) if full.is_file() else None,
        }
    return {
        "schema": "hnerv_lowlevel_release_surface_status_v1",
        "path": _repo_rel(release_dir),
        "exists": _repo_path(release_dir).is_dir(),
        "files": files,
    }


def _existing_repo_rel_paths(paths: list[Path]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        full = _repo_path(path)
        if not full.exists():
            continue
        rel = _repo_rel(path)
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def _runtime_module_paths(source_inflate: Path) -> list[str]:
    paths: list[Path] = [source_inflate]
    source_full = _repo_path(source_inflate)
    runtime_dir = source_full.parent
    if runtime_dir.is_dir():
        for child in sorted(runtime_dir.iterdir(), key=lambda item: item.name):
            if child.name.startswith(".") or child.name == "__pycache__":
                continue
            if child.suffix in {".py", ".sh", ".txt"} and child.is_file():
                paths.append(child)
    return _existing_repo_rel_paths(paths)


def _packet_code_paths(args: argparse.Namespace) -> list[str]:
    return _existing_repo_rel_paths([Path(path) for path in BUILD_CODE_PATHS]) + [
        path
        for path in _runtime_module_paths(args.inflate_sh)
        if path not in BUILD_CODE_PATHS
    ]


def _packet_source_paths(
    args: argparse.Namespace,
    *,
    candidate_result: Mapping[str, Any],
    public_preflight_path: Path,
    payload_diff_path: Path,
    compliance_path: Path,
    manifest_path: Path,
    packet_path: Path,
) -> list[str]:
    paths: list[Path] = [
        args.candidate_result,
        args.archive,
        args.baseline_json,
        args.inflate_sh,
        args.upstream_dir / "evaluate.py",
        public_preflight_path,
        payload_diff_path,
        compliance_path,
        manifest_path,
        packet_path,
    ]
    paths.extend(Path(path) for path in _packet_code_paths(args))
    for key in ("source_archive_path", "candidate_archive_path"):
        value = candidate_result.get(key)
        if isinstance(value, str) and value:
            paths.append(Path(value))
    return _existing_repo_rel_paths(paths)


def _remaining_required_for_score_or_dispatch(
    *, operator_approved_exact_cuda: bool
) -> list[str]:
    remaining = [
        "lightning_submit_environment",
        "active_level2_lane_dispatch_claim",
        "exact_cuda_auth_eval",
        "contest_auth_eval_adjudication",
        "operator_score_claim_review",
    ]
    if not operator_approved_exact_cuda:
        remaining.insert(0, "operator_exact_cuda_approval")
    return remaining


def _remaining_required_report_text(*, operator_approved_exact_cuda: bool) -> str:
    labels = {
        "operator_exact_cuda_approval": "operator exact-CUDA approval",
        "lightning_submit_environment": "Lightning submit environment",
        "active_level2_lane_dispatch_claim": "active Level-2 lane claim",
        "exact_cuda_auth_eval": "exact CUDA auth eval",
        "contest_auth_eval_adjudication": "contest auth eval adjudication",
        "operator_score_claim_review": "operator score-claim review",
    }
    return ", ".join(
        labels[item]
        for item in _remaining_required_for_score_or_dispatch(
            operator_approved_exact_cuda=operator_approved_exact_cuda
        )
    )


def _source_line_for_wrapper(source_inflate: Path) -> str:
    source_full = _repo_path(source_inflate)
    try:
        source_rel = source_full.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return f"SOURCE_INFLATE={shlex.quote(source_full.as_posix())}"
    return f'SOURCE_INFLATE="${{REPO_ROOT}}/{source_rel}"'


def _inflate_wrapper_text(source_inflate: Path) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# Deterministic HNeRV low-level static release-surface wrapper.\n"
        "# Delegates to the reviewed PR106 x-member exact-replay adapter.\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'if REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"; then\n'
        "  :\n"
        "else\n"
        '  REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"\n'
        "fi\n"
        f"{_source_line_for_wrapper(source_inflate)}\n"
        'if [ ! -x "$SOURCE_INFLATE" ]; then\n'
        '  echo "FATAL: delegated inflate.sh is missing or not executable: $SOURCE_INFLATE" >&2\n'
        "  exit 66\n"
        "fi\n"
        'exec "$SOURCE_INFLATE" "$@"\n'
    )


def _changed_sections(candidate_result: dict[str, Any]) -> list[dict[str, Any]]:
    audit = candidate_result.get("candidate_diff_audit")
    if not isinstance(audit, dict):
        return []
    sections = audit.get("sections")
    if not isinstance(sections, list):
        return []
    return [row for row in sections if isinstance(row, dict) and row.get("changed") is True]


def _raw_equivalence_by_section(candidate_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = candidate_result.get("brotli_raw_equivalence")
    if not isinstance(rows, list):
        rows = (candidate_result.get("candidate_diff") or {}).get("brotli_raw_equivalence")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("section_name"), str):
                out[row["section_name"]] = row
    return out


def _rate_only_raw_equivalent_kkt_proof(
    *,
    candidate_result: dict[str, Any],
    static_ready: bool,
    byte_delta: int,
    expected_delta: float,
) -> dict[str, Any]:
    """Return a narrow KKT-style descent proof for raw-equivalent rate-only repacks."""

    blockers: list[str] = []
    changed = _changed_sections(candidate_result)
    payload_changes = [row for row in changed if row.get("section_name") != "packed_header_ff_len24"]
    raw_by_section = _raw_equivalence_by_section(candidate_result)
    if not static_ready:
        blockers.append("static_packet_not_ready")
    if byte_delta >= 0:
        blockers.append("byte_delta_not_negative")
    if len(payload_changes) != 1:
        blockers.append("single_payload_change_required")
    for row in payload_changes:
        section_name = str(row.get("section_name") or "")
        raw_row = raw_by_section.get(section_name)
        if not isinstance(raw_row, dict) or raw_row.get("raw_equal") is not True:
            blockers.append(f"raw_equivalence_missing:{section_name or '<unknown>'}")
    official_delta = byte_delta * RATE_SCORE_PER_BYTE
    if abs(expected_delta - official_delta) > 1e-12:
        blockers.append("official_rate_delta_mismatch")

    passed = not blockers
    residual = 0.0 if passed else None
    tolerance = 0.0 if passed else 1e-12
    return {
        "schema": "hnerv_rate_only_raw_equivalent_kkt_proof_v1",
        "status": "passed" if passed else "blocked",
        "proof_class": "discrete_rate_only_raw_equivalent_archive_repack",
        "stationarity_residual": residual,
        "stationarity_tolerance": tolerance,
        "kkt_residual": residual,
        "kkt_tolerance": tolerance,
        "official_rate_score_delta": official_delta,
        "byte_delta": byte_delta,
        "expected_seg_dist_delta": 0.0,
        "expected_pose_dist_delta": 0.0,
        "interaction_assumptions": [
            "raw_equivalent_payload_sections_preserve_scorer_visible_runtime_outputs",
            "only_official_rate_term_changes_before_exact_cuda_confirmation",
        ],
        "blockers": blockers,
    }


def _runtime_tree_from_public_preflight(public_preflight: dict[str, Any] | None) -> str:
    if not isinstance(public_preflight, dict):
        return ""
    runtime = public_preflight.get("runtime")
    if not isinstance(runtime, dict):
        return ""
    runtime_tree = runtime.get("runtime_tree_sha256")
    return runtime_tree if _is_sha256(runtime_tree) else ""


def _validate_candidate_result(
    candidate_result: dict[str, Any],
    *,
    archive: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any, *, severity: str = "blocking") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed and severity == "blocking":
            blockers.append({"code": name, "detail": detail})

    archive_identity = _archive_identity(archive)
    add("candidate_result_score_claim_false", candidate_result.get("score_claim") is False, candidate_result.get("score_claim"))
    add(
        "candidate_result_dispatch_attempted_false",
        candidate_result.get("dispatch_attempted") is False,
        candidate_result.get("dispatch_attempted"),
    )
    add(
        "candidate_result_archive_preflight_ready",
        candidate_result.get("ready_for_archive_preflight") is True,
        candidate_result.get("ready_for_archive_preflight"),
    )
    add("candidate_archive_exists", archive_identity["exists"], archive_identity)
    add(
        "candidate_archive_sha256_matches_result",
        archive_identity["sha256"] == candidate_result.get("candidate_archive_sha256"),
        {
            "actual": archive_identity["sha256"],
            "expected": candidate_result.get("candidate_archive_sha256"),
        },
    )
    add(
        "candidate_archive_bytes_matches_result",
        archive_identity["bytes"] == candidate_result.get("candidate_archive_bytes"),
        {
            "actual": archive_identity["bytes"],
            "expected": candidate_result.get("candidate_archive_bytes"),
        },
    )
    add(
        "candidate_archive_sha256_differs_from_source",
        _is_sha256(candidate_result.get("candidate_archive_sha256"))
        and candidate_result.get("candidate_archive_sha256") != candidate_result.get("source_archive_sha256"),
        {
            "source_archive_sha256": candidate_result.get("source_archive_sha256"),
            "candidate_archive_sha256": candidate_result.get("candidate_archive_sha256"),
        },
    )
    add(
        "candidate_payload_sha256_differs_from_source",
        _is_sha256(candidate_result.get("candidate_payload_sha256"))
        and candidate_result.get("candidate_payload_sha256") != candidate_result.get("source_payload_sha256"),
        {
            "source_payload_sha256": candidate_result.get("source_payload_sha256"),
            "candidate_payload_sha256": candidate_result.get("candidate_payload_sha256"),
        },
    )

    try:
        candidate_archive = read_strict_single_member_zip(_repo_path(archive))
        candidate_payload = candidate_archive.payload
        parse_ff_packed_brotli_hnerv(candidate_payload)
    except Exception as exc:  # pragma: no cover - exact exception classes vary by bad input
        add("candidate_archive_strict_hnerv_parse", False, repr(exc))
        candidate_archive = None
        candidate_payload = b""
    else:
        add(
            "candidate_archive_single_member_matches_result",
            candidate_archive.member_name == candidate_result.get("candidate_member_name"),
            {
                "actual": candidate_archive.member_name,
                "expected": candidate_result.get("candidate_member_name"),
            },
        )
        add(
            "candidate_payload_sha256_matches_archive_member",
            sha256_bytes(candidate_payload) == candidate_result.get("candidate_payload_sha256"),
            {
                "actual": sha256_bytes(candidate_payload),
                "expected": candidate_result.get("candidate_payload_sha256"),
            },
        )

    source_path_value = candidate_result.get("source_archive_path")
    source_path = Path(source_path_value) if isinstance(source_path_value, str) else None
    source_identity = _archive_identity(source_path) if source_path else None
    add("source_archive_path_present", source_path is not None, source_path_value)
    add("source_archive_exists", bool(source_identity and source_identity["exists"]), source_identity)
    if source_identity:
        add(
            "source_archive_sha256_matches_result",
            source_identity["sha256"] == candidate_result.get("source_archive_sha256"),
            {
                "actual": source_identity["sha256"],
                "expected": candidate_result.get("source_archive_sha256"),
            },
        )
        add(
            "source_archive_bytes_matches_result",
            source_identity["bytes"] == candidate_result.get("source_archive_bytes"),
            {
                "actual": source_identity["bytes"],
                "expected": candidate_result.get("source_archive_bytes"),
            },
        )

    audit = candidate_result.get("candidate_diff_audit")
    add("candidate_diff_audit_object_present", isinstance(audit, dict), type(audit).__name__)
    if isinstance(audit, dict):
        add("candidate_diff_audit_blocker_free", not audit.get("blockers"), audit.get("blockers"))
        add(
            "candidate_diff_audit_ready_for_archive_preflight",
            audit.get("ready_for_archive_preflight") is True,
            audit.get("ready_for_archive_preflight"),
        )
        add("candidate_diff_audit_total_byte_delta_negative", int(audit.get("total_byte_delta") or 0) < 0, audit.get("total_byte_delta"))

    changed = _changed_sections(candidate_result)
    header_changes = [row for row in changed if row.get("section_name") == "packed_header_ff_len24"]
    payload_changes = [row for row in changed if row.get("section_name") != "packed_header_ff_len24"]
    add(
        "linked_header_control_change_count",
        len(header_changes) == 1,
        {"count": len(header_changes), "sections": [row.get("section_name") for row in changed]},
    )
    add(
        "single_payload_brotli_section_change",
        len(payload_changes) == 1,
        {"count": len(payload_changes), "sections": [row.get("section_name") for row in changed]},
    )
    if header_changes:
        add(
            "linked_header_change_is_byte_neutral_control",
            header_changes[0].get("byte_delta") == 0
            and header_changes[0].get("optimization_role") == "control_or_metadata",
            header_changes[0],
        )
    if payload_changes:
        payload_change = payload_changes[0]
        section_name = payload_change.get("section_name")
        raw_row = _raw_equivalence_by_section(candidate_result).get(str(section_name))
        add(
            "payload_change_is_rate_positive",
            int(payload_change.get("byte_delta") or 0) < 0,
            payload_change,
        )
        add(
            "payload_change_raw_equivalence_closed",
            isinstance(raw_row, dict) and raw_row.get("raw_equal") is True,
            raw_row,
        )

    return checks, blockers


def _candidate_manifest(
    args: argparse.Namespace,
    *,
    now_utc: dt.datetime,
    candidate_result: dict[str, Any],
    static_ready: bool,
    static_blockers: list[dict[str, Any]],
    public_preflight_path: Path,
    payload_diff_path: Path,
    compliance_path: Path,
    code_paths: list[str],
    source_paths: list[str],
    public_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    byte_delta = int(candidate_result["candidate_archive_bytes"]) - int(candidate_result["source_archive_bytes"])
    expected_delta = byte_delta * RATE_SCORE_PER_BYTE
    changed = _changed_sections(candidate_result)
    raw_equivalence = list(_raw_equivalence_by_section(candidate_result).values())
    kkt_proof = _rate_only_raw_equivalent_kkt_proof(
        candidate_result=candidate_result,
        static_ready=static_ready,
        byte_delta=byte_delta,
        expected_delta=expected_delta,
    )
    runtime_tree = _runtime_tree_from_public_preflight(public_preflight)
    runtime_tree_fields = (
        {
            "runtime_tree_sha256": runtime_tree,
            "runtime_tree_source": _repo_rel(public_preflight_path),
        }
        if runtime_tree
        else {}
    )
    return {
        "schema": "hnerv_lowlevel_exact_eval_candidate_manifest_v1",
        "schema_version": 1,
        "recorded_at_utc": _format_utc(now_utc),
        "candidate_id": args.lane_id,
        "lane_id": args.lane_id,
        "job_name": args.job_name,
        "family": "hnerv_lowlevel_brotli_repack",
        "pareto_scope": "hnerv_rate_only_exact_archive",
        "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
        "score_claim": False,
        "dispatch_attempted": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "operator_approved_exact_cuda": bool(args.operator_approved_exact_cuda),
        "approved_exact_eval_target": bool(static_ready and args.operator_approved_exact_cuda),
        "approval_scope": (
            "operator approval for exact CUDA score-lowering work only; "
            "lane claim, environment, submit, harvest, and score-adjudication gates still apply"
        ),
        "proxy_row": False,
        "code_paths": code_paths,
        "source_paths": source_paths,
        "source_archive_custody_mode": "verified_source_archive_payload_match",
        "source_archive_path": candidate_result.get("source_archive_path"),
        "source_archive_sha256": candidate_result.get("source_archive_sha256"),
        "source_archive_bytes": candidate_result.get("source_archive_bytes"),
        "source_member_name": candidate_result.get("source_member_name"),
        "source_payload_sha256": candidate_result.get("source_payload_sha256"),
        "source_payload_bytes": candidate_result.get("source_payload_bytes"),
        "candidate_archive_path": _repo_rel(args.archive),
        "candidate_archive_sha256": candidate_result.get("candidate_archive_sha256"),
        "candidate_archive_bytes": candidate_result.get("candidate_archive_bytes"),
        "candidate_member_name": candidate_result.get("candidate_member_name"),
        "candidate_payload_sha256": candidate_result.get("candidate_payload_sha256"),
        "candidate_payload_bytes": candidate_result.get("candidate_payload_bytes"),
        "byte_delta": byte_delta,
        "expected_total_score_delta_rate_only": expected_delta,
        "expected_seg_dist_delta": 0.0,
        "expected_pose_dist_delta": 0.0,
        "kkt_proof": kkt_proof,
        "raw_equivalence": raw_equivalence,
        "linked_lowlevel_changes": changed,
        "static_packet_ready": static_ready,
        "candidate_static_preflight_ready": static_ready,
        "static_blockers": static_blockers,
        "dispatch_gate": (
            "eligible_for_cuda_auth_eval_after_lane_claim"
            if static_ready
            else "blocked_static_packet_ready_until_static_blockers_clear"
        ),
        "dispatch_unlocked": static_ready,
        "ready_for_exact_eval_dispatch_claim": static_ready,
        "fixed_runtime_preflight": {
            "ready_for_fixed_runtime_exact_eval": static_ready,
            "remaining_blockers": [row["code"] for row in static_blockers],
            "inflate_sh": _repo_rel(args.inflate_sh),
            "upstream_dir": _repo_rel(args.upstream_dir),
            **runtime_tree_fields,
        },
        "exact_eval_runtime_contract": {
            "ready_for_exact_eval_runtime": static_ready,
            "remaining_blockers": [row["code"] for row in static_blockers],
            "inflate_sh": _repo_rel(args.inflate_sh),
            **runtime_tree_fields,
        },
        "artifact_links": {
            "candidate_result": _repo_rel(args.candidate_result),
            "public_replay_preflight": _repo_rel(public_preflight_path),
            "payload_section_diff": _repo_rel(payload_diff_path),
            "pre_submission_compliance": _repo_rel(compliance_path),
        },
        "score_blockers": [
            "exact_cuda_auth_eval_not_run_for_candidate",
            "contest_auth_eval_adjudication_not_run_for_candidate",
        ],
        "submit_blockers_until_operator_action": [
            "requires_level2_lane_dispatch_claim",
            "requires_lightning_environment",
            *([] if args.operator_approved_exact_cuda else ["requires_operator_exact_cuda_approval"]),
        ],
    }


def _payload_diff(candidate_result: dict[str, Any]) -> dict[str, Any]:
    changed = _changed_sections(candidate_result)
    byte_delta = int(candidate_result["candidate_archive_bytes"]) - int(candidate_result["source_archive_bytes"])
    return {
        "schema": "hnerv_lowlevel_payload_section_diff_v1",
        "schema_version": 1,
        "tool": "tools/build_hnerv_lowlevel_exact_eval_packet.py",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": True,
        "ready_for_exact_eval_dispatch": False,
        "candidate_archive_sha256": candidate_result.get("candidate_archive_sha256"),
        "candidate_archive_bytes": candidate_result.get("candidate_archive_bytes"),
        "source_archive_sha256": candidate_result.get("source_archive_sha256"),
        "source_archive_bytes": candidate_result.get("source_archive_bytes"),
        "source_payload_sha256": candidate_result.get("source_payload_sha256"),
        "candidate_payload_sha256": candidate_result.get("candidate_payload_sha256"),
        "total_byte_delta": byte_delta,
        "expected_total_score_delta_rate_only": byte_delta * RATE_SCORE_PER_BYTE,
        "changed_section_count": len(changed),
        "allowed_linked_change_contract": (
            "one byte-neutral packed_header_ff_len24 control update plus one "
            "rate-positive Brotli payload section whose decompressed raw bytes match"
        ),
        "brotli_raw_equivalence": list(_raw_equivalence_by_section(candidate_result).values()),
        "sections": [
            {
                "name": row.get("section_name"),
                "changed": row.get("changed") is True,
                "optimization_role": row.get("optimization_role"),
                "source_sha256": row.get("source_section_sha256"),
                "candidate_sha256": row.get("candidate_section_sha256"),
                "source_bytes": row.get("source_bytes"),
                "candidate_bytes": row.get("candidate_bytes"),
                "byte_delta": row.get("byte_delta"),
                "score_claim": False,
            }
            for row in changed
        ],
        "blockers": [],
    }


def _release_surface_report(
    *,
    args: argparse.Namespace,
    now_utc: dt.datetime,
    candidate_result: dict[str, Any],
) -> str:
    byte_delta = int(candidate_result["candidate_archive_bytes"]) - int(candidate_result["source_archive_bytes"])
    lines = [
        "HNeRV low-level Brotli static release surface",
        f"recorded_at_utc: {_format_utc(now_utc)}",
        f"candidate_id: {args.lane_id}",
        "family: hnerv_lowlevel_brotli_repack",
        "scope: upload-file static surface only",
        "score_claim: false",
        "dispatch_attempted: false",
        "remote_gpu_run: false",
        "evidence_grade: empirical_archive_candidate_until_exact_cuda",
        f"archive_sha256: {candidate_result['candidate_archive_sha256']}",
        f"archive_size_bytes: {candidate_result['candidate_archive_bytes']}",
        f"archive_member: {candidate_result.get('candidate_member_name', 'x')}",
        f"source_archive_sha256: {candidate_result.get('source_archive_sha256')}",
        f"source_archive_size_bytes: {candidate_result.get('source_archive_bytes')}",
        f"byte_delta_vs_source_archive: {byte_delta}",
        f"candidate_manifest: {_repo_rel(args.result_dir / 'manifest.json')}",
        f"payload_section_diff: {_repo_rel(args.result_dir / 'payload_section_diff_vs_source.json')}",
        f"public_replay_preflight: {_repo_rel(args.result_dir / 'public_replay_preflight.json')}",
        "remaining_for_score_or_dispatch: "
        f"{_remaining_required_report_text(operator_approved_exact_cuda=bool(args.operator_approved_exact_cuda))}",
        "notes: This file intentionally records no score and no promotion claim.",
    ]
    return "\n".join(lines) + "\n"


def build_release_surface(
    args: argparse.Namespace,
    *,
    now_utc: dt.datetime,
    candidate_result: dict[str, Any],
) -> dict[str, Any]:
    release_dir = args.release_surface_dir
    release_full = _repo_path(release_dir)
    release_full.mkdir(parents=True, exist_ok=True)

    archive_identity = _archive_identity(args.archive)
    if archive_identity["sha256"] != candidate_result.get("candidate_archive_sha256"):
        raise ValueError("candidate archive sha256 mismatch before release-surface copy")
    if archive_identity["bytes"] != candidate_result.get("candidate_archive_bytes"):
        raise ValueError("candidate archive byte size mismatch before release-surface copy")
    if not _repo_path(args.inflate_sh).is_file():
        raise FileNotFoundError(f"inflate.sh does not exist: {args.inflate_sh}")

    archive_dst = release_full / "archive.zip"
    shutil.copyfile(_repo_path(args.archive), archive_dst)
    archive_dst.chmod(0o644)

    inflate_dst = release_full / "inflate.sh"
    inflate_dst.write_text(_inflate_wrapper_text(args.inflate_sh), encoding="utf-8")
    inflate_dst.chmod(0o755)

    report_dst = release_full / "report.txt"
    report_dst.write_text(
        _release_surface_report(args=args, now_utc=now_utc, candidate_result=candidate_result),
        encoding="utf-8",
    )
    report_dst.chmod(0o644)

    surface_manifest = {
        "schema": "hnerv_lowlevel_release_surface_manifest_v1",
        "schema_version": 1,
        "recorded_at_utc": _format_utc(now_utc),
        "candidate_id": args.lane_id,
        "job_name": args.job_name,
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "operator_approved_exact_cuda": bool(args.operator_approved_exact_cuda),
        "approved_exact_eval_target": bool(args.operator_approved_exact_cuda),
        "approval_scope": (
            "operator approval recorded for exact CUDA score-lowering work; "
            "this release surface remains static and non-dispatched"
        ),
        "release_surface_scope": "static_upload_files_only_no_auth_eval_no_dispatch",
        "code_paths": _packet_code_paths(args),
        "source_paths": _runtime_module_paths(args.inflate_sh),
        "candidate_archive_sha256": candidate_result.get("candidate_archive_sha256"),
        "candidate_archive_bytes": candidate_result.get("candidate_archive_bytes"),
        "candidate_payload_sha256": candidate_result.get("candidate_payload_sha256"),
        "source_archive_sha256": candidate_result.get("source_archive_sha256"),
        "source_archive_bytes": candidate_result.get("source_archive_bytes"),
        "source_payload_sha256": candidate_result.get("source_payload_sha256"),
        "archive": {
            "path": "archive.zip",
            "source_path": _repo_rel(args.archive),
            "sha256": candidate_result.get("candidate_archive_sha256"),
            "bytes": candidate_result.get("candidate_archive_bytes"),
        },
        "inflate_sh": {
            "path": "inflate.sh",
            "delegates_to": _repo_rel(args.inflate_sh),
            "wrapper_sha256": sha256_file(inflate_dst),
        },
        "report": {
            "path": "report.txt",
            "sha256": sha256_file(report_dst),
        },
        "manifest_links": {
            "candidate_manifest": {
                "repo_path": _repo_rel(args.result_dir / "manifest.json"),
                "exists": _repo_path(args.result_dir / "manifest.json").is_file(),
            },
            "payload_section_diff": {
                "repo_path": _repo_rel(args.result_dir / "payload_section_diff_vs_source.json"),
                "exists": _repo_path(args.result_dir / "payload_section_diff_vs_source.json").is_file(),
            },
            "public_replay_preflight": {
                "repo_path": _repo_rel(args.result_dir / "public_replay_preflight.json"),
                "exists": _repo_path(args.result_dir / "public_replay_preflight.json").is_file(),
            },
            "exact_eval_packet": {
                "repo_path": _repo_rel(_packet_output_path(args)),
                "exists": _repo_path(_packet_output_path(args)).is_file(),
            },
        },
        "remaining_required_for_score_or_dispatch": _remaining_required_for_score_or_dispatch(
            operator_approved_exact_cuda=bool(args.operator_approved_exact_cuda)
        ),
    }
    manifest_dst = release_full / "archive_manifest.json"
    manifest_dst.write_text(json_text(surface_manifest), encoding="utf-8")
    manifest_dst.chmod(0o644)
    return {
        "schema": "hnerv_lowlevel_release_surface_generation_v1",
        "path": _repo_rel(release_dir),
        "files": _release_surface_status(release_dir)["files"],
    }


def _run_json_cmd(cmd: list[str], output_path: Path) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    payload = {
        "command": _one_liner([".venv/bin/python", *cmd[1:]]),
        "json_out": _repo_rel(output_path),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-2000:],
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit={result.returncode}: {_one_liner(cmd)}\n{result.stderr[-4000:]}"
        )
    return payload


def refresh_public_replay_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = args.result_dir / "public_replay_preflight.json"
    cmd = [
        sys.executable,
        "experiments/preflight_public_replay_intake.py",
        "--archive",
        args.archive.as_posix(),
        "--inflate-sh",
        args.inflate_sh.as_posix(),
        "--upstream-dir",
        args.upstream_dir.as_posix(),
        "--expected-archive-sha256",
        args.archive_sha256,
        "--expected-archive-size-bytes",
        str(args.archive_bytes),
        "--json-out",
        output.as_posix(),
        "--fail-if-not-ready",
    ]
    payload = _run_json_cmd(cmd, output)
    payload["schema"] = "hnerv_lowlevel_public_replay_preflight_refresh_v1"
    return payload


def refresh_static_compliance(args: argparse.Namespace) -> dict[str, Any]:
    output = args.result_dir / "pre_submission_compliance.json"
    candidate_result = read_json(_repo_path(args.candidate_result))
    if not isinstance(candidate_result, dict):
        raise ValueError(f"candidate result must be a JSON object: {args.candidate_result}")
    expected_member = str(candidate_result.get("candidate_member_name") or "")
    if not expected_member:
        raise ValueError("candidate result missing candidate_member_name")
    cmd = [
        sys.executable,
        "scripts/pre_submission_compliance_check.py",
        "--submission-dir",
        args.release_surface_dir.as_posix(),
        "--archive",
        (args.release_surface_dir / "archive.zip").as_posix(),
        "--archive-manifest-json",
        (args.release_surface_dir / "archive_manifest.json").as_posix(),
        "--expect-single-member",
        expected_member,
        "--expected-archive-sha256",
        args.archive_sha256,
        "--expected-archive-size-bytes",
        str(args.archive_bytes),
        "--public-scan-path",
        args.release_surface_dir.as_posix(),
        "--json-out",
        output.as_posix(),
        "--strict",
    ]
    payload = _run_json_cmd(cmd, output)
    payload["schema"] = "hnerv_lowlevel_static_compliance_refresh_v1"
    return payload


def refresh_dispatch_readiness(args: argparse.Namespace) -> dict[str, Any]:
    output = args.result_dir / "dispatch_readiness_preflight.json"
    now_utc = _format_utc(_now_utc(args))
    cmd = [
        sys.executable,
        "experiments/preflight_candidate_manifest_dispatch_readiness.py",
        "--manifest",
        (args.result_dir / "manifest.json").as_posix(),
        "--claims-path",
        args.claims_path.as_posix(),
        "--now-utc",
        now_utc,
        "--ttl-hours",
        str(args.claim_ttl_hours),
        "--json-out",
        output.as_posix(),
    ]
    payload = _run_json_cmd(cmd, output)
    stdout_tail = payload.get("stdout_tail")
    if isinstance(stdout_tail, str) and stdout_tail.strip().startswith("{"):
        try:
            payload["underlying_static_readiness_stdout"] = json.loads(stdout_tail)
            payload["stdout_tail"] = ""
            payload["stdout_tail_disposition"] = (
                "parsed_into_underlying_static_readiness_stdout_and_superseded_by_lane_claim_overlay"
            )
        except json.JSONDecodeError:
            payload["stdout_tail_disposition"] = "unparsed_subprocess_stdout_tail"
    claim_report = lane_claim_preflight(args, now_utc=_now_utc(args))
    if not claim_report["active_claim_present"]:
        blockers = payload.setdefault("blockers", [])
        if isinstance(blockers, list):
            blockers.append(
                {
                    "code": "missing_active_lane_dispatch_claim",
                    "severity": "blocking",
                    "detail": "standalone dispatch readiness requires a matching active Level-2 lane claim",
                }
            )
        payload["ready_for_exact_eval_dispatch"] = False
    if claim_report["conflict_present"]:
        blockers = payload.setdefault("blockers", [])
        if isinstance(blockers, list):
            blockers.append(
                {
                    "code": "active_lane_dispatch_claim_conflict",
                    "severity": "blocking",
                    "detail": "active same-lane claim exists for a different job",
                }
            )
        payload["ready_for_exact_eval_dispatch"] = False
    payload["lane_claim"] = claim_report
    payload["schema"] = "hnerv_lowlevel_dispatch_readiness_preflight_refresh_v1"
    output.write_text(json_text(payload), encoding="utf-8")
    return payload


def _failed_compliance_checks(payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for check in payload.get("checks", []):
        if isinstance(check, dict) and check.get("severity") == "error" and check.get("passed") is False:
            failed.append(str(check.get("name")))
    return failed


def _parse_claim_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
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
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "lane_id" in line:
            continue
        if _CLAIM_SEPARATOR_RE.match(line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= len(keys):
            rows.append(dict(zip(keys, cells[: len(keys)], strict=False)))
    return rows


def _is_terminal_claim_status(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_PREFIXES)


def _terminal_claim_audit_row(claim: dict[str, str]) -> dict[str, str]:
    row = dict(claim)
    row["claim_status"] = row.pop("status", "")
    return row


def lane_claim_preflight(args: argparse.Namespace, *, now_utc: dt.datetime) -> dict[str, Any]:
    full = _repo_path(args.claims_path)
    status: dict[str, Any] = {
        "claims_path": _repo_rel(args.claims_path),
        "claims_path_exists": full.is_file(),
        "lane_id": args.lane_id,
        "job_name": args.job_name,
        "ttl_hours": args.claim_ttl_hours,
        "matching_active_claims": [],
        "conflicting_active_claims": [],
        "matching_terminal_claims": [],
        "latest_matching_terminal_claim": None,
        "latest_matching_terminal_status": "",
        "terminal_claim_disposition": "",
        "active_claim_present": False,
        "conflict_present": False,
    }
    if not full.is_file():
        return status
    ttl = dt.timedelta(hours=args.claim_ttl_hours)
    latest_by_job: dict[str, dict[str, str]] = {}
    for claim in _parse_claim_rows(full):
        if claim.get("lane_id") != args.lane_id:
            continue
        timestamp = _parse_utc(claim.get("timestamp_utc"))
        if timestamp is None or now_utc - timestamp > ttl:
            continue
        job = claim.get("instance_job_id", "")
        previous = latest_by_job.get(job)
        previous_timestamp = _parse_utc(previous.get("timestamp_utc")) if previous else None
        if previous is None or previous_timestamp is None or timestamp > previous_timestamp:
            latest_by_job[job] = claim
    active = [
        claim for claim in latest_by_job.values() if not _is_terminal_claim_status(claim.get("status", ""))
    ]
    terminal = [
        claim for claim in latest_by_job.values() if _is_terminal_claim_status(claim.get("status", ""))
    ]
    status["matching_active_claims"] = [
        claim for claim in active if claim.get("instance_job_id") == args.job_name
    ]
    status["conflicting_active_claims"] = [
        claim for claim in active if claim.get("instance_job_id") != args.job_name
    ]
    matching_terminal_raw = [
        claim for claim in terminal if claim.get("instance_job_id") == args.job_name
    ]
    status["matching_terminal_claims"] = [
        _terminal_claim_audit_row(claim) for claim in matching_terminal_raw
    ]
    if matching_terminal_raw:
        latest_terminal_raw = max(
            matching_terminal_raw,
            key=lambda claim: _parse_utc(claim.get("timestamp_utc"))
            or dt.datetime.min.replace(tzinfo=dt.UTC),
        )
        latest_terminal = _terminal_claim_audit_row(latest_terminal_raw)
        status["latest_matching_terminal_claim"] = latest_terminal
        status["latest_matching_terminal_status"] = latest_terminal.get("claim_status", "")
        status["terminal_claim_disposition"] = (
            "terminal claim rows are audit history only; a fresh active claim "
            "is still required before exact-eval dispatch"
        )
    status["active_claim_present"] = bool(status["matching_active_claims"])
    status["conflict_present"] = bool(status["conflicting_active_claims"])
    return status


def _required_env_check_cmd() -> list[str]:
    code = (
        "import os,sys; "
        "missing=[k for k in sys.argv[1:] if not os.environ.get(k)]; "
        "raise SystemExit(('FATAL: missing Lightning env: '+', '.join(missing)) if missing else 0)"
    )
    return [".venv/bin/python", "-c", code, *REQUIRED_ENV]


def _claim_cmd(args: argparse.Namespace) -> list[str]:
    byte_delta = getattr(args, "byte_delta", None)
    source_label = str(getattr(args, "source_label", "") or "").strip()
    member_name = str(getattr(args, "candidate_member_name", "") or "").strip()
    note_parts = [f"{args.lane_id} HNeRV low-level Brotli exact CUDA eval"]
    if isinstance(byte_delta, int) and not isinstance(byte_delta, bool):
        note_parts.append(f"byte_delta={byte_delta}")
    if source_label:
        note_parts.append(f"source={source_label}")
    if member_name:
        note_parts.append(f"member={member_name}")
    notes = (
        "; ".join(note_parts)
        + "; "
        f"archive_sha256={args.archive_sha256} bytes={args.archive_bytes}; "
        f"static_packet={_repo_rel(_packet_output_path(args))}"
    )
    return [
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


def _submit_cmd(args: argparse.Namespace) -> list[str]:
    return [
        ".venv/bin/python",
        "scripts/lightning_exact_eval_repro.py",
        "--job-name",
        args.job_name,
        "--stage-workspace",
        "--submit",
        "--archive",
        args.archive.as_posix(),
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
        "0.01",
        "--max-posenet-relative",
        "1.01",
        "--max-segnet-relative",
        "1.01",
        "--max-sane-score",
        "1.0",
        "--component-trace",
        "--queue-metadata",
        f"lane={args.lane_id}",
        "--queue-metadata",
        f"archive_manifest={(args.result_dir / 'manifest.json').as_posix()}",
        "--queue-metadata",
        f"public_preflight={(args.result_dir / 'public_replay_preflight.json').as_posix()}",
        "--queue-metadata",
        f"payload_section_diff={(args.result_dir / 'payload_section_diff_vs_source.json').as_posix()}",
        "--extra-artifact",
        (args.result_dir / "manifest.json").as_posix(),
        "--extra-artifact",
        (args.result_dir / "public_replay_preflight.json").as_posix(),
        "--extra-artifact",
        (args.result_dir / "pre_submission_compliance.json").as_posix(),
        "--extra-artifact",
        (args.result_dir / "payload_section_diff_vs_source.json").as_posix(),
        "--extra-artifact",
        _packet_output_path(args).as_posix(),
    ]


def _harvest_cmd(args: argparse.Namespace) -> list[str]:
    return [
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


def _submit_blocker_disposition(
    *,
    submit_blockers: list[str],
    missing_env: list[str],
    claim_report: dict[str, Any],
    static_ready: bool,
) -> dict[str, Any]:
    latest_terminal_status = str(claim_report.get("latest_matching_terminal_status") or "")
    return {
        "schema": "hnerv_lowlevel_submit_blocker_disposition_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "method_failure": False,
        "static_packet_ready": static_ready,
        "current_submit_blockers": list(submit_blockers),
        "missing_env": list(missing_env),
        "environment_status": (
            "blocked_missing_lightning_environment" if missing_env else "present"
        ),
        "environment_disposition": (
            "missing Lightning env is an operator/environment blocker, not method "
            "or archive failure"
            if missing_env
            else "required Lightning env vars were present when the packet was built"
        ),
        "lane_claim_status": (
            "active_claim_present"
            if claim_report.get("active_claim_present")
            else "missing_active_claim"
        ),
        "latest_matching_terminal_status": latest_terminal_status,
        "terminal_claim_disposition": (
            claim_report.get("terminal_claim_disposition")
            or "no matching terminal claim row was found in the current TTL window"
        ),
        "exact_remaining_blockers": list(submit_blockers),
    }


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    now_utc = _now_utc(args)
    candidate_result = read_json(_repo_path(args.candidate_result))
    if not isinstance(candidate_result, dict):
        raise ValueError(f"candidate result must be a JSON object: {args.candidate_result}")
    if args.archive is None:
        args.archive = Path(str(candidate_result.get("candidate_archive_path")))
    args.archive_sha256 = args.archive_sha256 or str(candidate_result.get("candidate_archive_sha256") or "")
    args.archive_bytes = args.archive_bytes or int(candidate_result.get("candidate_archive_bytes") or 0)
    args.byte_delta = int(candidate_result["candidate_archive_bytes"]) - int(candidate_result["source_archive_bytes"])
    args.source_label = candidate_result.get("source_label")
    args.candidate_member_name = candidate_result.get("candidate_member_name")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.release_surface_dir is None:
        args.release_surface_dir = args.result_dir / DEFAULT_RELEASE_SURFACE_SUBDIR

    checks, validation_blockers = _validate_candidate_result(candidate_result, archive=args.archive)
    static_blockers = list(validation_blockers)
    public_preflight_refresh = None
    release_surface_generation = None
    static_compliance_refresh = None
    dispatch_readiness_refresh = None
    public_preflight_payload: dict[str, Any] | None = None
    public_preflight_path = args.result_dir / "public_replay_preflight.json"
    payload_diff_path = args.result_dir / "payload_section_diff_vs_source.json"
    compliance_path = args.result_dir / "pre_submission_compliance.json"
    manifest_path = args.result_dir / "manifest.json"
    packet_path = _packet_output_path(args)
    code_paths = _packet_code_paths(args)
    source_paths = _packet_source_paths(
        args,
        candidate_result=candidate_result,
        public_preflight_path=public_preflight_path,
        payload_diff_path=payload_diff_path,
        compliance_path=compliance_path,
        manifest_path=manifest_path,
        packet_path=packet_path,
    )

    if not static_blockers:
        public_preflight_refresh = refresh_public_replay_preflight(args)
        public_preflight = read_json(_repo_path(public_preflight_path))
        public_preflight_payload = public_preflight if isinstance(public_preflight, dict) else None
        if not isinstance(public_preflight, dict) or public_preflight.get("ready_for_exact_eval_dispatch") is not True:
            static_blockers.append(
                {
                    "code": "public_replay_preflight_not_ready",
                    "detail": public_preflight.get("blockers") if isinstance(public_preflight, dict) else None,
                }
            )
    if not static_blockers:
        payload_diff_path.write_text(json_text(_payload_diff(candidate_result)), encoding="utf-8")
        manifest = _candidate_manifest(
            args,
            now_utc=now_utc,
            candidate_result=candidate_result,
            static_ready=True,
            static_blockers=[],
            public_preflight_path=public_preflight_path,
            payload_diff_path=payload_diff_path,
            compliance_path=compliance_path,
            code_paths=code_paths,
            source_paths=source_paths,
            public_preflight=public_preflight_payload,
        )
        manifest_path.write_text(json_text(manifest), encoding="utf-8")
        release_surface_generation = build_release_surface(
            args,
            now_utc=now_utc,
            candidate_result=candidate_result,
        )
        static_compliance_refresh = refresh_static_compliance(args)
        compliance = read_json(_repo_path(compliance_path))
        compliance_failures = _failed_compliance_checks(compliance if isinstance(compliance, dict) else {})
        if compliance_failures:
            static_blockers.append(
                {
                    "code": "pre_submission_compliance_failed",
                    "detail": compliance_failures,
                }
            )
    if not static_blockers:
        dispatch_readiness_refresh = refresh_dispatch_readiness(args)
        dispatch_readiness = read_json(_repo_path(args.result_dir / "dispatch_readiness_preflight.json"))
        if not isinstance(dispatch_readiness, dict) or dispatch_readiness.get("ready_for_exact_eval_dispatch") is not True:
            blockers = dispatch_readiness.get("blockers") if isinstance(dispatch_readiness, dict) else None
            blocker_codes = {
                str(item.get("code"))
                for item in blockers
                if isinstance(item, dict) and item.get("code")
            } if isinstance(blockers, list) else set()
            claim_only_blockers = {
                "missing_active_lane_dispatch_claim",
                "active_lane_dispatch_claim_conflict",
            }
            if not blocker_codes or not blocker_codes.issubset(claim_only_blockers):
                static_blockers.append(
                    {
                        "code": "candidate_manifest_dispatch_readiness_failed",
                        "detail": blockers,
                    }
                )

    static_ready = not static_blockers
    if not static_ready:
        payload_diff_path.write_text(json_text(_payload_diff(candidate_result)), encoding="utf-8")
        manifest = _candidate_manifest(
            args,
            now_utc=now_utc,
            candidate_result=candidate_result,
            static_ready=False,
            static_blockers=static_blockers,
            public_preflight_path=public_preflight_path,
            payload_diff_path=payload_diff_path,
            compliance_path=compliance_path,
            code_paths=code_paths,
            source_paths=source_paths,
            public_preflight=public_preflight_payload,
        )
        manifest_path.write_text(json_text(manifest), encoding="utf-8")

    claim_report = lane_claim_preflight(args, now_utc=now_utc)
    missing_env = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    submit_blockers: list[str] = []
    if not static_ready:
        submit_blockers.append("static_packet_not_ready")
    if missing_env:
        submit_blockers.append("missing_lightning_environment")
    if claim_report["conflict_present"]:
        submit_blockers.append("active_lane_dispatch_claim_conflict")
    if not claim_report["active_claim_present"]:
        submit_blockers.append("missing_active_lane_dispatch_claim")
    if not args.operator_approved_exact_cuda:
        submit_blockers.append("missing_operator_exact_cuda_approval")

    score_blockers = [
        "exact_cuda_auth_eval_not_run_for_candidate",
        "contest_auth_eval_adjudication_not_run_for_candidate",
        "operator_score_claim_review_not_done",
    ]
    byte_delta = int(candidate_result["candidate_archive_bytes"]) - int(candidate_result["source_archive_bytes"])
    expected_delta = byte_delta * RATE_SCORE_PER_BYTE
    kkt_proof = _rate_only_raw_equivalent_kkt_proof(
        candidate_result=candidate_result,
        static_ready=static_ready,
        byte_delta=byte_delta,
        expected_delta=expected_delta,
    )
    packet = {
        "schema": "hnerv_lowlevel_exact_eval_operator_packet_v1",
        "schema_version": 1,
        "packet_kind": "hnerv_lowlevel_exact_eval_operator_packet",
        "tool": "tools/build_hnerv_lowlevel_exact_eval_packet.py",
        "recorded_at_utc": _format_utc(now_utc),
        "candidate_id": args.lane_id,
        "lane_id": args.lane_id,
        "job_name": args.job_name,
        "family": "hnerv_lowlevel_brotli_repack",
        "pareto_scope": "hnerv_rate_only_exact_archive",
        "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_gpu_run": False,
        "code_paths": code_paths,
        "source_paths": source_paths,
        "static_packet_ready": static_ready,
        "candidate_static_preflight_ready": static_ready,
        "operator_approved_exact_cuda": bool(args.operator_approved_exact_cuda),
        "approved_exact_eval_target": bool(static_ready and args.operator_approved_exact_cuda),
        "approval_scope": (
            "operator approval for exact CUDA score-lowering work only; "
            "lane claim, environment, submit, harvest, and score-adjudication gates still apply"
        ),
        "dispatch_gate": (
            "eligible_for_cuda_auth_eval_after_lane_claim"
            if static_ready
            else "blocked_static_packet_ready_until_static_blockers_clear"
        ),
        "dispatch_unlocked": static_ready,
        "ready_for_exact_eval_dispatch_claim": static_ready,
        "ready_for_submit": not submit_blockers,
        "static_blockers": static_blockers,
        "submit_blockers": submit_blockers,
        "submit_blocker_disposition": _submit_blocker_disposition(
            submit_blockers=submit_blockers,
            missing_env=missing_env,
            claim_report=claim_report,
            static_ready=static_ready,
        ),
        "score_blockers": score_blockers,
        "missing_env": missing_env,
        "lane_claim_preflight": claim_report,
        "archive_sha256": args.archive_sha256,
        "archive_bytes": args.archive_bytes,
        "archive_identity": _archive_identity(args.archive),
        "source_archive_sha256": candidate_result.get("source_archive_sha256"),
        "source_archive_bytes": candidate_result.get("source_archive_bytes"),
        "source_payload_sha256": candidate_result.get("source_payload_sha256"),
        "candidate_payload_sha256": candidate_result.get("candidate_payload_sha256"),
        "byte_delta": byte_delta,
        "expected_total_score_delta_rate_only": expected_delta,
        "kkt_proof": kkt_proof,
        "validation_checks": checks,
        "release_surface": _release_surface_status(args.release_surface_dir),
        "refreshes": {
            "public_replay_preflight": public_preflight_refresh,
            "release_surface_generation": release_surface_generation,
            "static_compliance": static_compliance_refresh,
            "dispatch_readiness": dispatch_readiness_refresh,
        },
        "artifacts": {
            "candidate_result": _repo_rel(args.candidate_result),
            "manifest": _repo_rel(manifest_path),
            "payload_section_diff": _repo_rel(payload_diff_path),
            "public_replay_preflight": _repo_rel(public_preflight_path),
            "pre_submission_compliance": _repo_rel(compliance_path),
            "dispatch_readiness_preflight": _repo_rel(args.result_dir / "dispatch_readiness_preflight.json"),
            "release_surface": _repo_rel(args.release_surface_dir),
            "packet": _repo_rel(packet_path),
        },
        "operator_next_steps": {
            "schema": "hnerv_lowlevel_operator_next_steps_v1",
            "copy_safe": True,
            "must_run_in_order": True,
            "first_remote_gpu_step": "submit_exact_cuda",
            "packet_path": _repo_rel(packet_path),
            "current_submit_blockers": submit_blockers,
            "steps": [
                {
                    "order": 1,
                    "id": "verify_lightning_env",
                    "purpose": "fail loudly until all Lightning identity and path env vars are loaded",
                    "dispatches_remote_gpu": False,
                    "writes_repo_state": False,
                    "copy_safe_command": _one_liner(_required_env_check_cmd()),
                },
                {
                    "order": 2,
                    "id": "refresh_static_packet_no_dispatch",
                    "purpose": "refresh static custody, runtime preflight, compliance, and packet without claiming or dispatching",
                    "dispatches_remote_gpu": False,
                    "writes_repo_state": True,
                    "copy_safe_command": _one_liner(_refresh_cmd(args, packet_path=packet_path)),
                },
                {
                    "order": 3,
                    "id": "claim_lane_no_dispatch",
                    "purpose": "record the Level-2 lane claim only after the environment check passes; this does not submit a remote job",
                    "dispatches_remote_gpu": False,
                    "writes_repo_state": True,
                    "copy_safe_command": _one_liner(_claim_cmd(args)),
                },
                {
                    "order": 4,
                    "id": "refresh_with_operator_exact_cuda_approval",
                    "purpose": "record operator exact-CUDA approval after claim/env gates are satisfied; still no remote job",
                    "dispatches_remote_gpu": False,
                    "writes_repo_state": True,
                    "copy_safe_command": _one_liner(
                        _refresh_cmd(args, packet_path=packet_path, operator_approval=True)
                    ),
                },
                {
                    "order": 5,
                    "id": "submit_exact_cuda",
                    "purpose": "first remote/GPU action; run only after ready_for_submit is true",
                    "dispatches_remote_gpu": True,
                    "writes_repo_state": True,
                    "copy_safe_command": _one_liner(_submit_cmd(args)),
                },
                {
                    "order": 6,
                    "id": "harvest_after_completion",
                    "purpose": "harvest and require adjudication only after the Lightning job reaches a terminal state",
                    "dispatches_remote_gpu": False,
                    "writes_repo_state": True,
                    "copy_safe_command": _one_liner(_harvest_cmd(args)),
                },
            ],
        },
        "commands": {
            "claim": _one_liner(_claim_cmd(args)),
            "submit": _one_liner(_submit_cmd(args)),
            "harvest": _one_liner(_harvest_cmd(args)),
        },
    }
    _repo_path(packet_path).parent.mkdir(parents=True, exist_ok=True)
    _repo_path(packet_path).write_text(json_text(packet), encoding="utf-8")
    return packet


def _refresh_cmd(
    args: argparse.Namespace,
    *,
    packet_path: Path,
    operator_approval: bool = False,
) -> list[str]:
    cmd = [
        ".venv/bin/python",
        "tools/build_hnerv_lowlevel_exact_eval_packet.py",
        "--candidate-result",
        args.candidate_result.as_posix(),
        "--archive",
        args.archive.as_posix(),
        "--archive-sha256",
        args.archive_sha256,
        "--archive-bytes",
        str(args.archive_bytes),
        "--baseline-json",
        args.baseline_json.as_posix(),
        "--inflate-sh",
        args.inflate_sh.as_posix(),
        "--upstream-dir",
        args.upstream_dir.as_posix(),
        "--result-dir",
        args.result_dir.as_posix(),
        "--release-surface-dir",
        args.release_surface_dir.as_posix(),
        "--lane-id",
        args.lane_id,
        "--job-name",
        args.job_name,
        "--claims-path",
        args.claims_path.as_posix(),
        "--claim-ttl-hours",
        str(args.claim_ttl_hours),
        "--agent",
        args.agent,
        "--now-utc",
        _format_utc(_now_utc(args)),
        "--json-out",
        packet_path.as_posix(),
    ]
    if operator_approval:
        cmd.append("--operator-approved-exact-cuda")
    return cmd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-result", type=Path, default=DEFAULT_CANDIDATE_RESULT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--archive-bytes", type=int)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--inflate-sh", type=Path, default=DEFAULT_INFLATE_SH)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--release-surface-dir", type=Path)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--claim-ttl-hours", type=int, default=24)
    parser.add_argument("--agent", default="codex:gpt-5.5")
    parser.add_argument("--now-utc")
    parser.add_argument("--operator-approved-exact-cuda", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.claim_ttl_hours <= 0:
        raise SystemExit("--claim-ttl-hours must be positive")
    if args.release_surface_dir is None:
        args.release_surface_dir = args.result_dir / DEFAULT_RELEASE_SURFACE_SUBDIR
    packet = build_packet(args)
    if args.json_out is None:
        print(json_text(packet), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
