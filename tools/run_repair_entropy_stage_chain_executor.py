#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute selected repair entropy-stage work orders as composed archives."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.repair_entropy_stage_chain_executor import (  # noqa: E402
    REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA,
    RepairEntropyStageChainExecutorError,
    build_repair_entropy_stage_chain_execution_bundle,
)
from tac.repo_io import json_text, sha256_file, write_json_artifact  # noqa: E402


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairEntropyStageChainExecutorError(f"{path} must contain a JSON object")
    return payload


def _expand_reports(paths: list[Path], patterns: list[str]) -> list[Path]:
    out = [_resolve(path) for path in paths]
    for pattern in patterns:
        glob_pattern = pattern if Path(pattern).is_absolute() else str(REPO_ROOT / pattern)
        out.extend(Path(match) for match in glob.glob(glob_pattern, recursive=True))
    unique: dict[str, Path] = {}
    for path in out:
        unique[str(path.resolve(strict=False))] = path
    return list(unique.values())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-report", action="append", default=[], type=Path)
    parser.add_argument("--execution-report-glob", action="append", default=[])
    parser.add_argument("--work-order-bundle", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chain-report-out", required=True, type=Path)
    parser.add_argument(
        "--chain-input-manifest",
        type=Path,
        help="Accepted for replay-bundle argv stability; not used directly.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report_paths = _expand_reports(args.execution_report, args.execution_report_glob)
        reports = [_load_json(path) for path in report_paths]
        work_order_bundle = (
            None if args.work_order_bundle is None else _load_json(_resolve(args.work_order_bundle))
        )
        bundle = build_repair_entropy_stage_chain_execution_bundle(
            execution_reports=reports,
            execution_report_paths=report_paths,
            work_order_bundle=work_order_bundle,
            output_dir=_resolve(args.output_dir),
            repo_root=REPO_ROOT,
            allow_overwrite=bool(args.overwrite),
        )
        output = _resolve(args.chain_report_out)
        write_result = write_json_artifact(
            output,
            bundle,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=sha256_file(output)
            if output.exists() and args.overwrite
            else None,
        )
    except (
        RepairEntropyStageChainExecutorError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: repair entropy-stage chain execution failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_entropy_stage_chain_executor_cli_result.v1",
                "chain_report_out": str(args.chain_report_out),
                "chain_report_schema": REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA,
                "source_execution_report_count": bundle["source_execution_report_count"],
                "chain_count": bundle["chain_count"],
                "materialized_chain_candidate_count": bundle[
                    "materialized_chain_candidate_count"
                ],
                "runtime_consumption_proof_ready_count": bundle[
                    "runtime_consumption_proof_ready_count"
                ],
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "bytes_written": write_result.bytes_written,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
