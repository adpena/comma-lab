#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build fail-closed exact-ready bridge inputs from a repair handoff plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    _TOOL_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _TOOL_DIR.parent
    for _path in (str(_REPO_ROOT), str(_TOOL_DIR)):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.repair_family_exact_ready_bridge import (  # noqa: E402
    RepairFamilyExactReadyBridgeError,
    build_repair_family_exact_ready_bridge,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-handoff-plan", required=True, type=Path)
    parser.add_argument("--source-queue-out", required=True, type=Path)
    parser.add_argument("--blocked-exact-ready-queue-out", required=True, type=Path)
    parser.add_argument("--bridge-report-out", required=True, type=Path)
    parser.add_argument("--submission-dir", action="append", default=[], type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairFamilyExactReadyBridgeError(f"{path} must contain a JSON object")
    return payload


def _write(path: Path, payload: object, *, overwrite: bool) -> int:
    target = _resolve(path)
    skipped = False
    expected = None
    if target.exists() and overwrite:
        existing_text = target.read_text(encoding="utf-8")
        next_text = json_text(payload)
        if existing_text == next_text:
            skipped = True
        else:
            expected = sha256_file(target)
    if skipped:
        return 0
    return write_json_artifact(
        target,
        payload,
        allow_overwrite=overwrite,
        expected_existing_sha256=expected,
    ).bytes_written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        handoff_plan_path = _resolve(args.exact_handoff_plan)
        result = build_repair_family_exact_ready_bridge(
            exact_handoff_plan=_load_json(handoff_plan_path),
            exact_handoff_plan_path=args.exact_handoff_plan,
            submission_dirs=tuple(args.submission_dir),
            repo_root=REPO_ROOT,
        )
        source_bytes = _write(
            args.source_queue_out,
            result["source_optimizer_queue"],
            overwrite=bool(args.overwrite),
        )
        blocked_bytes = _write(
            args.blocked_exact_ready_queue_out,
            result["blocked_exact_ready_queue"],
            overwrite=bool(args.overwrite),
        )
        report_bytes = _write(
            args.bridge_report_out,
            result["bridge_report"],
            overwrite=bool(args.overwrite),
        )
    except (
        ArtifactWriteError,
        OSError,
        RepairFamilyExactReadyBridgeError,
        ValueError,
    ) as exc:
        print(f"FATAL: repair exact-ready bridge failed: {exc}", file=sys.stderr)
        return 2
    report = result["bridge_report"]
    print(
        json_text(
            {
                "schema": "repair_family_exact_ready_bridge_cli_result.v1",
                "candidate_count": report["candidate_count"],
                "archive_custody_proven_count": report[
                    "archive_custody_proven_count"
                ],
                "runtime_proof_custody_proven_count": report[
                    "runtime_proof_custody_proven_count"
                ],
                "runtime_content_tree_custody_proven_count": report[
                    "runtime_content_tree_custody_proven_count"
                ],
                "source_queue_out": str(args.source_queue_out),
                "blocked_exact_ready_queue_out": str(args.blocked_exact_ready_queue_out),
                "bridge_report_out": str(args.bridge_report_out),
                "bytes_written": source_bytes + blocked_bytes + report_bytes,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
