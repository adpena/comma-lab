#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build score-claim-false master-gradient trust-region candidate manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.master_gradient_trust_region import (  # noqa: E402
    TRUST_REGION_MODES,
    build_master_gradient_trust_region_candidates,
)
from tac.repo_io import read_json, write_json  # noqa: E402


def _load_probe_outcome(path: Path | None, probe_id: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    selected = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        if probe_id is None or row.get("probe_id") == probe_id:
            selected = row
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-manifest", type=Path, required=True)
    parser.add_argument("--independence-report", type=Path)
    parser.add_argument("--probe-outcome-jsonl", type=Path)
    parser.add_argument("--probe-id")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-rows-per-candidate", type=int, default=8)
    parser.add_argument("--base-mutation-intensity", type=float, default=0.25)
    parser.add_argument("--pair-block-size", type=int)
    parser.add_argument(
        "--mode",
        dest="modes",
        choices=TRUST_REGION_MODES,
        action="append",
        help="Trust-region mode to emit. Repeatable; defaults to all modes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    operator_manifest = read_json(args.operator_manifest)
    operator_manifest.setdefault("manifest_path", args.operator_manifest.as_posix())
    independence_report = (
        read_json(args.independence_report) if args.independence_report is not None else None
    )
    score_response_outcome = _load_probe_outcome(args.probe_outcome_jsonl, args.probe_id)
    payload = build_master_gradient_trust_region_candidates(
        operator_manifest=operator_manifest,
        repo_root=REPO_ROOT,
        independence_report=independence_report,
        score_response_outcome=score_response_outcome,
        modes=tuple(args.modes or TRUST_REGION_MODES),
        max_rows_per_candidate=args.max_rows_per_candidate,
        base_mutation_intensity=args.base_mutation_intensity,
        pair_block_size=args.pair_block_size,
    )
    payload["operator_manifest_path"] = args.operator_manifest.as_posix()
    payload["independence_report_path"] = (
        args.independence_report.as_posix() if args.independence_report is not None else None
    )
    payload["probe_outcome_jsonl_path"] = (
        args.probe_outcome_jsonl.as_posix() if args.probe_outcome_jsonl is not None else None
    )
    payload["output_json"] = args.output_json.as_posix()
    write_json(args.output_json, payload)
    print(json.dumps({"output_json": args.output_json.as_posix(), "candidate_count": payload["candidate_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
