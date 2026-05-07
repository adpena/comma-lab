#!/usr/bin/env python3
"""Audit PR102 zero-byte HNeRV tuning custody.

PR102's archive payload is declared to be unchanged from PR100; its score
change comes from runtime constants. This guard catches the exact failure mode
where a generic release-asset scraper downloads an unrelated archive instead of
the HNeRV archive referenced by PR102's reproducibility script.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, repo_relative, sha256_file, write_json  # noqa: E402

SCHEMA_VERSION = 1
TOOL = "tools/audit_pr102_zero_byte_tuning_custody.py"
EXPECTED_PR100_SHA256 = "afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641"
EXPECTED_ARCHIVE_BYTES = 178_981
EXPECTED_MEMBER = "0.bin"
EXPECTED_DELTA_SCALE = 0.0095
EXPECTED_FRAME0_RED_NUDGE = 1.0


class Pr102CustodyAuditError(ValueError):
    """Raised when a PR102 custody audit input is malformed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr102-intake-dir", required=True, type=Path)
    parser.add_argument("--pr100-archive", required=True, type=Path)
    parser.add_argument("--correct-pr102-archive", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-blocked", action="store_true")
    return parser.parse_args(argv)


def build_pr102_zero_byte_tuning_custody(
    *,
    pr102_intake_dir: Path,
    pr100_archive: Path,
    correct_pr102_archive: Path,
    repo_root: Path,
) -> dict[str, Any]:
    source_dir = pr102_intake_dir / "source" / "submissions" / "hnerv_lc_v2_scale095_rplus1"
    if not source_dir.exists():
        raise Pr102CustodyAuditError(f"missing PR102 source dir: {source_dir}")
    pr100_archive_record = _archive_record(pr100_archive, repo_root)
    correct_archive_record = _archive_record(correct_pr102_archive, repo_root)
    existing_archive = pr102_intake_dir / "archive.zip"
    existing_archive_record = _archive_record(existing_archive, repo_root) if existing_archive.exists() else None
    compress_contract = _compress_contract(source_dir / "compress.sh")
    sidecar_contract = _sidecar_contract(source_dir / "sidecar.py")
    inflate_contract = _inflate_contract(source_dir / "inflate.py")

    blockers: list[str] = []
    if compress_contract["expected_sha256"] != EXPECTED_PR100_SHA256:
        blockers.append("pr102_compress_expected_sha_mismatch")
    if pr100_archive_record["sha256"] != EXPECTED_PR100_SHA256:
        blockers.append("pr100_reference_archive_sha_mismatch")
    if correct_archive_record["sha256"] != EXPECTED_PR100_SHA256:
        blockers.append("correct_pr102_archive_sha_mismatch")
    if correct_archive_record["bytes"] != EXPECTED_ARCHIVE_BYTES:
        blockers.append("correct_pr102_archive_size_mismatch")
    if correct_archive_record["members"] != [EXPECTED_MEMBER]:
        blockers.append("correct_pr102_archive_member_mismatch")
    if not _same_file_bytes(pr100_archive, correct_pr102_archive):
        blockers.append("correct_pr102_archive_not_byte_identical_to_pr100")
    if sidecar_contract["delta_scale"] != EXPECTED_DELTA_SCALE:
        blockers.append("pr102_delta_scale_not_0_0095")
    if not inflate_contract["frame0_red_add_one"]:
        blockers.append("pr102_frame0_red_add_one_missing")

    existing_wrong_archive = False
    if existing_archive_record is not None and existing_archive_record["sha256"] != EXPECTED_PR100_SHA256:
        existing_wrong_archive = True
        blockers.append("existing_pr102_intake_archive_is_wrong_release_asset")

    ready_for_source_schema_review = not [
        blocker
        for blocker in blockers
        if blocker
        not in {
            "existing_pr102_intake_archive_is_wrong_release_asset",
        }
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_source_schema_review": ready_for_source_schema_review,
        "ready_for_exact_eval_dispatch": False,
        "pr102_intake_dir": repo_relative(pr102_intake_dir, repo_root),
        "source_dir": repo_relative(source_dir, repo_root),
        "pr100_reference_archive": pr100_archive_record,
        "correct_pr102_archive": correct_archive_record,
        "existing_pr102_intake_archive": existing_archive_record,
        "existing_pr102_intake_archive_wrong": existing_wrong_archive,
        "zero_byte_runtime_contract": {
            "archive_payload_unchanged_from_pr100": _same_file_bytes(pr100_archive, correct_pr102_archive),
            "delta_scale": sidecar_contract["delta_scale"],
            "frame0_red_add_one": inflate_contract["frame0_red_add_one"],
            "archive_byte_delta": correct_archive_record["bytes"] - pr100_archive_record["bytes"],
            "archive_sha256_equal": correct_archive_record["sha256"] == pr100_archive_record["sha256"],
        },
        "compress_contract": compress_contract,
        "sidecar_contract": sidecar_contract,
        "inflate_contract": inflate_contract,
        "readiness_blockers": blockers,
        "dispatch_blockers": [
            *blockers,
            "pr102_exact_cuda_replay_missing",
            "pr102_port_to_current_stack_missing",
            "no_op_control_missing",
        ],
    }


def _archive_record(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.exists():
        raise Pr102CustodyAuditError(f"missing archive: {path}")
    members: list[str] = []
    member_records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            members.append(info.filename)
            member_records.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": info.CRC,
                }
            )
    return {
        "path": repo_relative(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "members": members,
        "member_records": member_records,
    }


def _compress_contract(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    url = _shell_assignment(text, "URL")
    expected_sha = _shell_assignment(text, "EXPECTED_SHA256")
    return {
        "path": repo_relative(path, REPO_ROOT),
        "url": url,
        "expected_sha256": expected_sha,
        "points_to_pr100_release": "BradyMeighan" in url and "hnerv-lc-v2-archive" in url,
    }


def _sidecar_contract(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    delta_scale: float | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DELTA_SCALE" for target in node.targets
        ):
            delta_scale = _literal_float(node.value)
    return {
        "path": repo_relative(path, REPO_ROOT),
        "delta_scale": delta_scale,
        "delta_scale_matches_pr102": delta_scale == EXPECTED_DELTA_SCALE,
    }


def _inflate_contract(path: Path) -> dict[str, Any]:
    source_text = path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(path))
    frame0_red_add_one = False
    add_one_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_":
            continue
        if len(node.args) != 1 or _literal_float(node.args[0]) != EXPECTED_FRAME0_RED_NUDGE:
            continue
        add_one_calls += 1
        if "up[:, 0, 0]" in (ast.get_source_segment(source_text, node) or ""):
            frame0_red_add_one = True
    return {
        "path": repo_relative(path, REPO_ROOT),
        "add_one_call_count": add_one_calls,
        "frame0_red_add_one": frame0_red_add_one,
    }


def _same_file_bytes(left: Path, right: Path) -> bool:
    return left.read_bytes() == right.read_bytes()


def _shell_assignment(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=\"([^\"]+)\"", text, flags=re.MULTILINE)
    if not match:
        raise Pr102CustodyAuditError(f"missing shell assignment: {name}")
    return match.group(1)


def _literal_float(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_pr102_zero_byte_tuning_custody(
            pr102_intake_dir=args.pr102_intake_dir,
            pr100_archive=args.pr100_archive,
            correct_pr102_archive=args.correct_pr102_archive,
            repo_root=REPO_ROOT,
        )
    except (OSError, zipfile.BadZipFile, Pr102CustodyAuditError) as exc:
        print(f"FATAL: PR102 zero-byte custody audit failed: {exc}", file=sys.stderr)
        return 2
    if args.json_out is not None:
        write_json(args.json_out, payload)
    else:
        print(json_text(payload), end="")
    if args.fail_if_blocked and payload["ready_for_source_schema_review"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
