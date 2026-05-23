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

from tac.canonical_frontier_pointer import (  # noqa: E402
    CANONICAL_FRONTIER_POINTER_PATH,
    load_canonical_frontier_pointer_lenient,
)
from tac.optimization.local_cpu_contest_drift import (  # noqa: E402
    TRUST_REGION_DQS1_FEC6,
    build_eureka_signal,
    build_eureka_signal_from_local_json_file,
    fit_drift_calibration,
    load_calibration_json,
    paired_anchor_from_json_files,
    require_eureka_false_authority,
)


def _write_json(path: str | Path, payload: object) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _auth_frontier_score_from_args(args: argparse.Namespace) -> float | None:
    if args.auth_frontier_score is not None:
        return float(args.auth_frontier_score)
    if not args.auth_frontier_score_from_pointer:
        return None

    pointer_path = Path(args.canonical_frontier_pointer)
    if not pointer_path.is_absolute():
        pointer_path = REPO_ROOT / pointer_path
    pointer = load_canonical_frontier_pointer_lenient(
        repo_root=REPO_ROOT,
        path=pointer_path,
    )
    if pointer is None:
        raise SystemExit(
            "--auth-frontier-score-from-pointer could not load "
            f"{args.canonical_frontier_pointer}; run tools/refresh_canonical_frontier.py"
        )
    anchor = pointer.our_local_frontier_contest_cpu
    if anchor is None:
        raise SystemExit(
            "--auth-frontier-score-from-pointer requires "
            "our_local_frontier_contest_cpu in the canonical frontier pointer"
        )
    return float(anchor.score)


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
    parser.add_argument(
        "--calibration-json",
        default=None,
        help="existing calibration JSON to consume instead of fitting --pair rows",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="calibration JSON output; required when fitting --pair rows",
    )
    parser.add_argument("--eureka-out", default=None, help="optional eureka signal JSON output")
    parser.add_argument("--candidate-id", default="", help="candidate id for eureka signal")
    parser.add_argument("--local-score", type=float, default=None, help="local score for eureka signal")
    parser.add_argument(
        "--candidate-local-json",
        default=None,
        help="candidate local advisory JSON to evaluate for eureka trigger",
    )
    parser.add_argument(
        "--candidate-trust-region",
        default=None,
        help="candidate trust-region label; defaults to calibration trust region",
    )
    parser.add_argument(
        "--auth-frontier-score",
        type=float,
        default=None,
        help="current auth frontier score for eureka signal",
    )
    parser.add_argument(
        "--auth-frontier-score-from-pointer",
        action="store_true",
        help=(
            "load the current contest-CPU auth frontier from "
            ".omx/state/canonical_frontier_pointer.json"
        ),
    )
    parser.add_argument(
        "--canonical-frontier-pointer",
        default=str(CANONICAL_FRONTIER_POINTER_PATH),
        help="canonical frontier pointer path used with --auth-frontier-score-from-pointer",
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
    if args.calibration_json:
        calibration = load_calibration_json(args.calibration_json)
    else:
        if not args.json_out:
            raise SystemExit("--json-out is required when fitting calibration from --pair rows")
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
    if args.json_out:
        _write_json(args.json_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    if args.eureka_out:
        auth_frontier_score = _auth_frontier_score_from_args(args)
        if not args.candidate_id or auth_frontier_score is None:
            raise SystemExit(
                "--eureka-out requires --candidate-id and either "
                "--auth-frontier-score or --auth-frontier-score-from-pointer"
            )
        if args.candidate_local_json:
            signal = build_eureka_signal_from_local_json_file(
                candidate_id=args.candidate_id,
                local_path=args.candidate_local_json,
                auth_frontier_score=auth_frontier_score,
                calibration=calibration,
                candidate_trust_region=args.candidate_trust_region,
                min_margin=args.min_margin,
            )
        else:
            if args.local_score is None:
                raise SystemExit(
                    "--eureka-out requires either --candidate-local-json or --local-score"
                )
            signal = build_eureka_signal(
                candidate_id=args.candidate_id,
                local_score=args.local_score,
                auth_frontier_score=auth_frontier_score,
                calibration=calibration,
                min_margin=args.min_margin,
                source_artifact=args.source_artifact,
                candidate_trust_region=args.candidate_trust_region,
            )
        require_eureka_false_authority(
            signal,
            context=f"{args.candidate_id} generated eureka signal",
        )
        _write_json(args.eureka_out, signal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
