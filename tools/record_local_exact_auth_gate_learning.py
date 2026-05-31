#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record local exact-auth gate outcomes as planner learning signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.local_exact_auth_gate_learning import (  # noqa: E402
    LocalExactAuthGateLearningError,
    append_local_exact_auth_gate_posterior_signal,
    load_gate_and_build_signal,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-report-json", required=True, type=Path)
    parser.add_argument("--replay-summary-json", type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--family-id", default="unclassified_local_candidate")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--posterior-path", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal = load_gate_and_build_signal(
        gate_report_path=args.gate_report_json,
        repo_root=REPO_ROOT,
        candidate_id=args.candidate_id,
        lane_id=args.lane_id,
        family_id=args.family_id,
        replay_summary_path=args.replay_summary_json,
    )
    write_json_artifact(args.out_json, signal)
    report = append_local_exact_auth_gate_posterior_signal(
        learning_signal=signal,
        learning_signal_path=args.out_json,
        repo_root=REPO_ROOT,
        posterior_path=args.posterior_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except LocalExactAuthGateLearningError as exc:
        print(f"record_local_exact_auth_gate_learning failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except ArtifactWriteError as exc:
        print(f"record_local_exact_auth_gate_learning artifact write failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
