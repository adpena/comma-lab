#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Attach out-of-fold linear predictions to a scorer-response dataset."""

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
    ScorerResponseDatasetError,
    render_markdown,
)
from tac.optimization.scorer_response_prediction import (  # noqa: E402
    DEFAULT_PREDICTION_FIELD,
    attach_out_of_fold_linear_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--target", default="delta_vs_baseline_score")
    parser.add_argument("--prediction-field", default=DEFAULT_PREDICTION_FIELD)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        if not isinstance(dataset, dict):
            raise ScorerResponseDatasetError("dataset must be a JSON object")
        predicted = attach_out_of_fold_linear_predictions(
            dataset,
            target=args.target,
            prediction_field=args.prediction_field,
            ridge_lambda=args.ridge_lambda,
        )
    except (OSError, ValueError, ScorerResponseDatasetError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(predicted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(predicted), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "prediction_fit": predicted["prediction_fit"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
