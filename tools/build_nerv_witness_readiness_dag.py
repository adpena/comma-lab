#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the evaluator-witness readiness DAG for HiNeRV/SNeRV long runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_witness_readiness_dag import (  # noqa: E402
    DEFAULT_QUEUE_ID,
    build_nerv_witness_readiness_dag,
    check_witness_gate_status,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "check-evidence":
        status = check_witness_gate_status(
            node_id=args.node_id,
            source_boundary_audit_report=args.source_boundary_audit_report,
            hinerv_smoke_report=args.hinerv_smoke_report,
            snerv_authority_gate_report=args.snerv_authority_gate_report,
            snerv_long_run_launch_gate_verdict=args.snerv_long_run_launch_gate_verdict,
            pair_local_servo_report=args.pair_local_servo_report,
            pair_local_servo_receipt=args.pair_local_servo_receipt,
            repo_root=REPO_ROOT,
        )
        print(json.dumps(status, sort_keys=True))
        return 0 if status["satisfied"] else 2

    output_root = args.output_root or (
        Path("/Volumes/VertigoDataTier/pact")
        / f"nerv_witness_readiness_dag_{_stamp()}"
    )
    output_json = args.output_json or (
        Path(output_root) / "nerv_witness_readiness_dag.json"
    )
    payload = build_nerv_witness_readiness_dag(
        repo_root=REPO_ROOT,
        output_root=output_root,
        source_boundary_audit_report=args.source_boundary_audit_report,
        hinerv_smoke_report=args.hinerv_smoke_report,
        snerv_authority_gate_report=args.snerv_authority_gate_report,
        snerv_long_run_launch_gate_verdict=args.snerv_long_run_launch_gate_verdict,
        pair_local_servo_report=args.pair_local_servo_report,
        pair_local_servo_receipt=args.pair_local_servo_receipt,
        partner_source_refs=args.partner_source_ref,
        dag_id=args.dag_id,
        max_nodes=args.max_nodes,
    )
    result = write_json_artifact(output_json, payload)
    if args.output_dag:
        write_json_artifact(args.output_dag, payload["dag"])
    if args.output_status_json:
        write_json_artifact(args.output_status_json, payload["status_map"])
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "output_json": result.path,
                "output_json_sha256": result.sha256,
                "hinerv_long_training_approved": payload[
                    "hinerv_long_training_approved"
                ],
                "snerv_long_training_approved": payload[
                    "snerv_long_training_approved"
                ],
                "long_training_approved": payload["long_training_approved"],
                "actionable_blocker_count": len(payload["actionable_blockers"]),
                "next_actions": payload["next_actions"][:8],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    build = sub.add_parser("build", help="write the witness-readiness DAG artifact")
    _add_common(build)
    build.add_argument("--output-root", type=Path)
    build.add_argument("--output-json", type=Path)
    build.add_argument("--output-dag", type=Path)
    build.add_argument("--output-status-json", type=Path)
    build.add_argument("--partner-source-ref", action="append", default=[], type=Path)
    build.add_argument("--dag-id", default=DEFAULT_QUEUE_ID)
    build.add_argument("--max-nodes", type=int, default=8)

    check = sub.add_parser(
        "check-evidence",
        help="exit 0 only when a specific witness-readiness gate is satisfied",
    )
    _add_common(check)
    check.add_argument("--node-id", required=True)

    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] not in {"build", "check-evidence"}:
        argv = ["build", *argv]
    args = parser.parse_args(argv)
    return args


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-boundary-audit-report", type=Path)
    parser.add_argument("--hinerv-smoke-report", type=Path)
    parser.add_argument("--snerv-authority-gate-report", type=Path)
    parser.add_argument("--snerv-long-run-launch-gate-verdict", type=Path)
    parser.add_argument("--pair-local-servo-report", type=Path)
    parser.add_argument("--pair-local-servo-receipt", type=Path)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
