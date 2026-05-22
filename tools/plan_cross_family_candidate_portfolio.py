#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan a false-authority cross-family exact-eval candidate portfolio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.cross_family_candidate_portfolio import (  # noqa: E402
    CrossFamilyCandidatePortfolioError,
    build_cross_family_candidate_portfolio,
    render_cross_family_candidate_portfolio_markdown,
    source_artifacts_from_paths,
    write_json,
)
from tac.repo_io import read_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incumbent-score",
        type=float,
        required=True,
        help="Exact CUDA incumbent score used only as a planning baseline.",
    )
    parser.add_argument(
        "--mlx-selection",
        type=Path,
        action="append",
        default=[],
        help="mlx_effective_spend_triage_candidate_selection.v1 JSON. May repeat.",
    )
    parser.add_argument(
        "--pairset-acquisition",
        type=Path,
        action="append",
        default=[],
        help="decoder_q_pairset_acquisition.v1 JSON. May repeat.",
    )
    parser.add_argument(
        "--hfv2-manifest",
        type=Path,
        action="append",
        default=[],
        help="hfv1_to_hfv2_sparse_sidecar_candidate_v1 JSON. May repeat.",
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        action="append",
        default=[],
        help="Manual candidate JSON object/list, or object with candidates[].",
    )
    parser.add_argument(
        "--family-beliefs",
        type=Path,
        help="Optional family belief JSON object/list overriding weak defaults.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--expected-improvement-weight", type=float, default=1.0)
    parser.add_argument("--information-gain-weight", type=float, default=0.01)
    return parser.parse_args(argv)


def _json_objects(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise CrossFamilyCandidatePortfolioError(f"{path}: expected JSON object")
        out.append(payload)
    return out


def _manual_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            rows = payload["candidates"]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            raise CrossFamilyCandidatePortfolioError(
                f"{path}: expected candidate object/list"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CrossFamilyCandidatePortfolioError(
                    f"{path}: candidate {index} must be object"
                )
            out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_artifacts = source_artifacts_from_paths(
            {
                "mlx_selections": args.mlx_selection,
                "pairset_acquisitions": args.pairset_acquisition,
                "hfv2_manifests": args.hfv2_manifest,
                "manual_candidate_json": args.candidate_json,
                "family_beliefs": args.family_beliefs,
            },
            repo_root=REPO_ROOT,
        )
        family_beliefs = read_json(args.family_beliefs) if args.family_beliefs else None
        portfolio = build_cross_family_candidate_portfolio(
            incumbent_score=args.incumbent_score,
            mlx_selections=_json_objects(args.mlx_selection),
            pairset_acquisitions=_json_objects(args.pairset_acquisition),
            hfv2_manifests=_json_objects(args.hfv2_manifest),
            manual_candidates=_manual_candidates(args.candidate_json),
            family_beliefs=family_beliefs,
            source_artifacts=source_artifacts,
            source_artifact_paths={
                "mlx_selections": [path.as_posix() for path in args.mlx_selection],
                "pairset_acquisitions": [
                    path.as_posix() for path in args.pairset_acquisition
                ],
                "hfv2_manifests": [path.as_posix() for path in args.hfv2_manifest],
            },
            top_k=args.top_k,
            expected_improvement_weight=args.expected_improvement_weight,
            information_gain_weight=args.information_gain_weight,
        )
    except (
        OSError,
        json.JSONDecodeError,
        CrossFamilyCandidatePortfolioError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    write_json(args.json_out, portfolio)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_cross_family_candidate_portfolio_markdown(portfolio),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "ranked_candidate_count": portfolio["portfolio_summary"][
                    "ranked_candidate_count"
                ],
                "candidate_archive_custody_ready_count": portfolio[
                    "portfolio_summary"
                ]["candidate_archive_custody_ready_count"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
