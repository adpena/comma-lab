#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the materializer exact-readiness bridge for an optimizer source queue."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.materializer_chain_harvest import (  # noqa: E402
    EXACT_READINESS_BRIDGE_SCHEMA,
    run_exact_readiness_bridge_for_harvested_queue,
)
from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    require_no_truthy_authority_fields,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    read_json,
    repo_relative,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", required=True, type=Path)
    parser.add_argument("--exact-readiness-out-dir", required=True, type=Path)
    parser.add_argument("--bridge-report-out", required=True, type=Path)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--allow-source-blocker", action="append", default=[])
    parser.add_argument("--dispatch-claims-path", type=Path)
    parser.add_argument("--claim-ttl-hours", type=float, default=24.0)
    parser.add_argument("--allow-above-active-floor-dispatch", action="store_true")
    parser.add_argument("--operator-override-reason")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Re-run the bridge even when an existing report matches the source queue.",
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _existing_report_matches(
    path: Path,
    *,
    source_queue: Path,
    exact_readiness_out_dir: Path,
    candidate_ids: Sequence[str],
) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    if not isinstance(payload, dict):
        return False
    require_no_truthy_authority_fields(
        payload,
        context=f"materializer_exact_readiness_bridge_existing:{path}",
    )
    if payload.get("schema") != EXACT_READINESS_BRIDGE_SCHEMA:
        return False
    if payload.get("source_queue_path") != repo_relative(source_queue, REPO_ROOT):
        return False
    if payload.get("exact_readiness_out_dir") != repo_relative(
        exact_readiness_out_dir,
        REPO_ROOT,
    ):
        return False
    requested = {str(candidate_id) for candidate_id in candidate_ids if str(candidate_id)}
    if requested:
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return False
        observed = {
            str(row.get("candidate_id") or "")
            for row in rows
            if isinstance(row, dict)
        }
        if observed != requested:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_queue = _resolve(args.source_queue)
    out_dir = _resolve(args.exact_readiness_out_dir)
    bridge_report_out = _resolve(args.bridge_report_out)
    try:
        if bridge_report_out.exists():
            if not args.overwrite:
                raise ArtifactWriteError(
                    f"refusing to overwrite existing bridge report: {bridge_report_out}"
                )
            if (
                not args.force_recompute
                and _existing_report_matches(
                    bridge_report_out,
                    source_queue=source_queue,
                    exact_readiness_out_dir=out_dir,
                    candidate_ids=args.candidate_id,
                )
            ):
                payload = read_json(bridge_report_out)
                print(
                    json_text(
                        {
                            "schema": "materializer_exact_readiness_bridge_cli_result.v1",
                            "bridge_report_out": str(args.bridge_report_out),
                            "candidate_count": payload.get("candidate_count"),
                            "ready_candidate_count": payload.get("ready_candidate_count"),
                            "blocked_candidate_count": payload.get(
                                "blocked_candidate_count"
                            ),
                            "skipped_existing_bridge_report": True,
                            "score_claim": False,
                            "promotion_eligible": False,
                            "rank_or_kill_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        }
                    ),
                    end="",
                )
                return 0
        report = run_exact_readiness_bridge_for_harvested_queue(
            repo_root=REPO_ROOT,
            source_queue_path=source_queue,
            exact_readiness_out_dir=out_dir,
            candidate_ids=tuple(args.candidate_id),
            allow_source_blockers=tuple(args.allow_source_blocker),
            dispatch_claims_path=args.dispatch_claims_path,
            claim_ttl_hours=args.claim_ttl_hours,
            allow_above_active_floor_dispatch=args.allow_above_active_floor_dispatch,
            operator_override_reason=args.operator_override_reason,
        )
        require_no_truthy_authority_fields(
            report,
            context="materializer_exact_readiness_bridge_cli_report",
        )
        write_json_artifact(
            bridge_report_out,
            report,
            allow_overwrite=args.overwrite,
            expected_existing_sha256=(
                sha256_file(bridge_report_out)
                if args.overwrite and bridge_report_out.exists()
                else None
            ),
        )
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: exact-readiness bridge failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "materializer_exact_readiness_bridge_cli_result.v1",
                "bridge_report_out": str(args.bridge_report_out),
                "candidate_count": report.get("candidate_count"),
                "ready_candidate_count": report.get("ready_candidate_count"),
                "blocked_candidate_count": report.get("blocked_candidate_count"),
                "skipped_existing_bridge_report": False,
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
