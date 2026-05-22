#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fit local macOS-CPU versus contest-CPU drift and emit eureka signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.local_cpu_contest_drift import (  # noqa: E402
    TRUST_REGION_DQS1_FEC6,
    build_eureka_signal,
    fit_drift_calibration,
    paired_anchor_from_json_files,
)


def _write_json(path: str | Path, payload: object) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("LOCAL_JSON", "CONTEST_CPU_JSON"),
        default=[],
        help="same-archive local advisory JSON and contest-CPU JSON",
    )
    parser.add_argument(
        "--trust-region",
        default=TRUST_REGION_DQS1_FEC6,
        help="calibration trust-region label",
    )
    parser.add_argument("--json-out", required=True, help="calibration JSON output")
    parser.add_argument("--eureka-out", default=None, help="optional eureka signal JSON output")
    parser.add_argument("--candidate-id", default="", help="candidate id for eureka signal")
    parser.add_argument("--local-score", type=float, default=None, help="local score for eureka signal")
    parser.add_argument(
        "--auth-frontier-score",
        type=float,
        default=None,
        help="current auth frontier score for eureka signal",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="required positive conservative margin for eureka trigger",
    )
    parser.add_argument("--source-artifact", default="", help="source artifact for eureka signal")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchors = [
        paired_anchor_from_json_files(
            local_path=local,
            contest_path=contest,
            trust_region=args.trust_region,
        )
        for local, contest in args.pair
    ]
    calibration = fit_drift_calibration(anchors, trust_region=args.trust_region)
    payload = calibration.to_dict()
    _write_json(args.json_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    if args.eureka_out:
        if not args.candidate_id or args.local_score is None or args.auth_frontier_score is None:
            raise SystemExit(
                "--eureka-out requires --candidate-id, --local-score, and "
                "--auth-frontier-score"
            )
        signal = build_eureka_signal(
            candidate_id=args.candidate_id,
            local_score=args.local_score,
            auth_frontier_score=args.auth_frontier_score,
            calibration=calibration,
            min_margin=args.min_margin,
            source_artifact=args.source_artifact,
        )
        _write_json(args.eureka_out, signal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
