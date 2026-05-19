#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build RATE ACH/assumption process features for Cathedral autopilot.

The output is planning-only. It does not rewrite archives, load scorer code,
dispatch provider work, or claim score. It converts rate-attack candidate rows
into machine-sortable disconfirming-assumption and cheap-probe features.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.contest_exploits.rate_attack_autopilot_features import (  # noqa: E402
    RateAttackCandidateSource,
    build_rate_attack_autopilot_feature_matrix,
    write_rate_attack_autopilot_feature_matrix,
)

DEFAULT_SOURCE_GLOBS = (
    "experiments/results/rate_attack_op1_*/cathedral_autopilot_candidates.jsonl",
    "experiments/results/rate_attack_op2_*/cathedral_autopilot_candidates.jsonl",
    "experiments/results/rate_attack_op3_*/cathedral_autopilot_candidates.jsonl",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help=(
            "Rate-attack source JSON/JSONL carrying autopilot_rows or candidate "
            "rows. Repeat to combine OP1/OP2/OP3 outputs. If omitted, the "
            "canonical experiments/results/rate_attack_op*/ candidates are used."
        ),
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label for each --source. Defaults to source stem.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output feature-matrix JSON. Defaults to "
            "reports/rate_attack_autopilot_feature_matrix_<UTC>.json."
        ),
    )
    parser.add_argument(
        "--rows-jsonl",
        type=Path,
        default=None,
        help="Optional Cathedral candidate JSONL path. Defaults beside output JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_json = args.output_json or (
        REPO_ROOT / "reports" / f"rate_attack_autopilot_feature_matrix_{stamp}.json"
    )
    rows_jsonl = args.rows_jsonl or output_json.with_name(
        f"{output_json.stem}_cathedral_autopilot_candidates.jsonl"
    )
    sources = _candidate_sources(args.source, args.label)
    matrix = build_rate_attack_autopilot_feature_matrix(
        sources,
        generated_at_utc=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    )
    paths = write_rate_attack_autopilot_feature_matrix(
        matrix,
        output_json.parent,
        matrix_path=output_json,
        rows_path=rows_jsonl,
    )
    print(json.dumps({"paths": paths, "matrix": matrix}, indent=2, sort_keys=True))
    return 0


def _candidate_sources(
    explicit_sources: list[Path],
    labels: list[str],
) -> list[RateAttackCandidateSource]:
    sources = explicit_sources or _default_sources()
    if labels and len(labels) != len(sources):
        raise SystemExit("--label must be supplied zero times or once per --source")
    if not sources:
        raise SystemExit("no rate-attack source artifacts found")
    return [
        RateAttackCandidateSource(
            path=source,
            label=labels[index] if labels else source.parent.name,
        )
        for index, source in enumerate(sources)
    ]


def _default_sources() -> list[Path]:
    paths: list[Path] = []
    for pattern in DEFAULT_SOURCE_GLOBS:
        paths.extend(sorted(REPO_ROOT.glob(pattern)))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


if __name__ == "__main__":
    raise SystemExit(main())
