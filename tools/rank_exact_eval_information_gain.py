#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rank exact CUDA candidates by expected improvement and information gain."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.bayesian_experimental_design import rank_exact_eval_candidates  # noqa: E402
from tac.repo_io import json_text, read_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON payload with candidates and beliefs.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--source", help="Override report source label.")
    parser.add_argument("--incumbent-score", type=float, help="Exact CUDA incumbent score.")
    parser.add_argument("--expected-improvement-weight", type=float)
    parser.add_argument("--information-gain-weight", type=float)
    parser.add_argument("--top-k", type=int)
    return parser.parse_args(argv)


def _payload_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        return payload["candidates"]
    raise SystemExit("--input must be a JSON list or an object with a candidates list")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = read_json(args.input)
    if not isinstance(payload, (dict, list)):
        raise SystemExit("--input must contain a JSON object or list")
    payload_dict = payload if isinstance(payload, dict) else {}
    incumbent_score = args.incumbent_score
    if incumbent_score is None:
        incumbent_score = payload_dict.get("incumbent_score")
    if incumbent_score is None:
        raise SystemExit("provide --incumbent-score or input.incumbent_score")

    acquisition_weights = payload_dict.get("acquisition_weights") or {}
    expected_improvement_weight = (
        args.expected_improvement_weight
        if args.expected_improvement_weight is not None
        else float(acquisition_weights.get("expected_improvement", 1.0))
    )
    information_gain_weight = (
        args.information_gain_weight
        if args.information_gain_weight is not None
        else float(acquisition_weights.get("expected_information_gain", 1.0))
    )
    report = rank_exact_eval_candidates(
        _payload_candidates(payload),
        incumbent_score=float(incumbent_score),
        family_beliefs=payload_dict.get("family_beliefs"),
        source=args.source or str(payload_dict.get("source") or args.input),
        expected_improvement_weight=expected_improvement_weight,
        information_gain_weight=information_gain_weight,
        top_k=args.top_k,
    )
    text = json_text(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
