#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate a non-authoritative scorer-response dataset for LL held-out use."""

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

from tac.optimization.scorer_response_dataset import (  # noqa: E402
    DEFAULT_RESPONSE_PREDICTION_FIELDS,
    ScorerResponseDatasetError,
    build_scorer_response_validation_gate,
    render_validation_gate_markdown,
)


def _parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        out.append(int(text))
    if not out:
        raise argparse.ArgumentTypeError("at least one fold is required")
    return out


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--min-families", type=int, default=2)
    parser.add_argument("--required-folds", type=_parse_csv_ints, default=[0, 1, 2, 3, 4])
    parser.add_argument("--target", default="delta_vs_baseline_score")
    parser.add_argument(
        "--prediction-fields",
        type=_parse_csv_strings,
        default=list(DEFAULT_RESPONSE_PREDICTION_FIELDS),
        help="Comma-separated prediction fields to test against --target.",
    )
    parser.add_argument("--min-prediction-pairs-per-fold", type=int, default=3)
    parser.add_argument("--min-pearson-r", type=float, default=0.2)
    parser.add_argument(
        "--required-spend-triage-family",
        action="append",
        default=[],
        help=(
            "Require this exact response family to strict-pass its own OOF "
            "spend-triage gate. May repeat."
        ),
    )
    parser.add_argument(
        "--allow-mixed-axis-targets",
        action="store_true",
        help=(
            "Disable the default fail-closed requirement that all rows share "
            "one score axis/target scale."
        ),
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help=(
            "After writing validation outputs, exit nonzero unless the validation "
            "gate passed. Use this for CI/automation; omit it for report-only runs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        gate = build_scorer_response_validation_gate(
            dataset,
            min_rows=args.min_rows,
            min_families=args.min_families,
            required_folds=args.required_folds,
            target=args.target,
            prediction_fields=args.prediction_fields,
            min_prediction_pairs_per_fold=args.min_prediction_pairs_per_fold,
            min_pearson_r=args.min_pearson_r,
            required_spend_triage_families=args.required_spend_triage_family,
            require_single_axis=not args.allow_mixed_axis_targets,
        )
    except (OSError, ValueError, ScorerResponseDatasetError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_validation_gate_markdown(gate), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "status": gate["status"],
                "passed": gate["passed"],
                "prediction_spend_triage_usable": gate[
                    "prediction_spend_triage_usable"
                ],
                "required_family_spend_triage_passed": gate[
                    "required_family_spend_triage_passed"
                ],
                "required_family_spend_triage_blockers": gate[
                    "required_family_spend_triage_blockers"
                ],
                "blockers": gate["blockers"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_pass and gate.get("passed") is not True:
        return 2
    if (
        args.require_pass
        and args.required_spend_triage_family
        and gate.get("required_family_spend_triage_passed") is not True
    ):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
