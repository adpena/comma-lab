#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan cross-family stack search from repair byte-transform reports."""

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

from tac.optimization.repair_family_stack_search import (  # noqa: E402
    RepairFamilyStackSearchError,
    build_repair_family_exact_handoff_plan,
    plan_repair_family_stack_search,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-report",
        action="append",
        required=True,
        type=Path,
        help="Repair-family byte-transform execution report. Repeatable.",
    )
    parser.add_argument("--stack-plan-out", required=True, type=Path)
    parser.add_argument("--exact-handoff-plan-out", type=Path)
    parser.add_argument("--posterior-path", type=Path)
    parser.add_argument("--byte-credit-budget", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairFamilyStackSearchError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reports = [_load_report(_resolve(path)) for path in args.execution_report]
        plan = plan_repair_family_stack_search(
            execution_reports=reports,
            execution_report_paths=tuple(args.execution_report),
            repo_root=REPO_ROOT,
            posterior_path=args.posterior_path,
            byte_credit_budget=args.byte_credit_budget,
        )
        exact_handoff_plan = build_repair_family_exact_handoff_plan(
            stack_plan=plan,
            stack_plan_path=args.stack_plan_out,
        )
        out = _resolve(args.stack_plan_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if out.exists() and args.overwrite:
            existing_text = out.read_text(encoding="utf-8")
            next_text = json_text(plan)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                out,
                plan,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
        exact_write_result = None
        exact_skipped_identical_existing_artifact = False
        if args.exact_handoff_plan_out is not None:
            exact_out = _resolve(args.exact_handoff_plan_out)
            exact_expected_existing_sha256 = None
            if exact_out.exists() and args.overwrite:
                existing_text = exact_out.read_text(encoding="utf-8")
                next_text = json_text(exact_handoff_plan)
                if existing_text == next_text:
                    exact_skipped_identical_existing_artifact = True
                else:
                    exact_expected_existing_sha256 = sha256_file(exact_out)
            if not exact_skipped_identical_existing_artifact:
                exact_write_result = write_json_artifact(
                    exact_out,
                    exact_handoff_plan,
                    allow_overwrite=bool(args.overwrite),
                    expected_existing_sha256=exact_expected_existing_sha256,
                )
    except (
        ArtifactWriteError,
        OSError,
        RepairFamilyStackSearchError,
        ValueError,
    ) as exc:
        print(f"FATAL: repair family stack search failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_family_stack_search_cli_result.v1",
                "stack_plan_out": str(args.stack_plan_out),
                "execution_report_count": plan["execution_report_count"],
                "candidate_improvement_observed": plan[
                    "candidate_improvement_observed"
                ],
                "planned_family_order": plan["planned_family_order"],
                "exact_eval_handoff_candidate_count": exact_handoff_plan[
                    "candidate_count"
                ],
                "archive_bound_exact_handoff_candidate_count": exact_handoff_plan[
                    "archive_bound_candidate_count"
                ],
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "exact_handoff_plan_out": (
                    None
                    if args.exact_handoff_plan_out is None
                    else str(args.exact_handoff_plan_out)
                ),
                "exact_handoff_bytes_written": (
                    exact_write_result.bytes_written
                    if exact_write_result is not None
                    else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
                "exact_handoff_skipped_identical_existing_artifact": (
                    exact_skipped_identical_existing_artifact
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
