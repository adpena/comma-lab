#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit engineered-correction artifacts for local patch readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import numpy as np  # noqa: E402

from tac.audit_contract import AuditReport, audit_exit_code  # noqa: E402
from tac.engineered_correction_readiness import (  # noqa: E402
    audit_corrections_bin,
    audit_sparse_corrections,
)
from tac.repo_io import json_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrections_bin", nargs="?", type=Path)
    parser.add_argument("--max-packed-bytes", type=int, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--component-trace-plan", type=Path)
    parser.add_argument(
        "--require-component-trace-plan",
        action="store_true",
        help="Require a PR85 scorer-gradient atom plan signed by a cross-checked component trace.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a deterministic in-memory readiness probe for preflight.",
    )
    parser.add_argument("--uncompressed", action="store_true")
    parser.add_argument("--fail-if-not-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        report = audit_sparse_corrections(
            _self_test_sparse_corrections(),
            max_packed_bytes=args.max_packed_bytes,
            require_component_trace_plan=args.require_component_trace_plan,
        )
    else:
        if args.corrections_bin is None:
            raise SystemExit("corrections_bin is required unless --self-test is used")
        report = audit_corrections_bin(
            args.corrections_bin,
            max_packed_bytes=args.max_packed_bytes,
            compressed=not args.uncompressed,
            manifest_path=args.manifest,
            component_trace_plan_path=args.component_trace_plan,
            require_component_trace_plan=args.require_component_trace_plan,
        )
    payload = _audit_report(report).to_dict()
    print(json_text(payload), end="")
    if args.fail_if_not_ready and not report.ready_for_local_patch:
        return audit_exit_code(_audit_report(report))
    return 0


def _audit_report(report) -> AuditReport:
    payload = report.to_dict()
    details = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "blockers",
            "dispatch_attempted",
            "ready_for_local_patch",
            "score_claim",
        }
    }
    return AuditReport(
        audit="engineered_correction_readiness",
        readiness_key="ready_for_local_patch",
        ready=report.ready_for_local_patch,
        blockers=tuple(report.blockers),
        summary={
            "n_kept": report.n_kept,
            "n_total": report.n_total,
            "packed_bytes": report.packed_bytes,
            "quantize_bits": report.quantize_bits,
            "component_trace_signed": report.component_trace_signed,
            "component_trace_atom_count": report.component_trace_atom_count,
            "warning_count": len(report.warnings),
        },
        metadata=details,
    )


def _self_test_sparse_corrections() -> dict:
    return {
        "indices": np.array([1, 7], dtype=np.uint32),
        "n_kept": 2,
        "n_total": 9,
        "quantize_bits": 8,
        "scale": 2.0,
        "shape": [1, 3, 3, 3],
        "top_k_pct": 22.2,
        "values": np.array([[10, -2, 0], [0, 3, -4]], dtype=np.int8),
    }


if __name__ == "__main__":
    raise SystemExit(main())
