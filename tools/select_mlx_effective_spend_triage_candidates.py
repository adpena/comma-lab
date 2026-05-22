#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select candidate-generation rows from a strict effective MLX gate."""

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

from tac.optimization.mlx_effective_spend_triage_selection import (  # noqa: E402
    DEFAULT_PREDICTION_FIELD,
    MLXEffectiveSpendTriageSelectionError,
    build_mlx_effective_spend_triage_selection,
    dumps_json,
    load_json_object,
    render_mlx_effective_spend_triage_selection_markdown,
    source_artifact_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help="Family allowlist. May repeat. Defaults to all MLX rows.",
    )
    parser.add_argument(
        "--min-observed-gain",
        type=float,
        help=(
            "Override the score-calibration recommended minimum MLX gap. "
            "Default reads the strict score-calibration gate."
        ),
    )
    parser.add_argument("--prediction-field", default=DEFAULT_PREDICTION_FIELD)
    parser.add_argument(
        "--require-prediction-negative",
        action="store_true",
        help=(
            "Require the selected OOF prediction to also be negative. Omit this "
            "when the observed strict-gated MLX response is the selection basis."
        ),
    )
    parser.add_argument(
        "--allow-non-singleton-windows",
        action="store_true",
        help=(
            "Allow non-singleton rows. Current full-600 MLX authority should not "
            "use this until batch/window invariance has its own contract."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = load_json_object(args.dataset)
        plan = load_json_object(args.plan)
        selection = build_mlx_effective_spend_triage_selection(
            dataset,
            plan,
            top_k=args.top_k,
            families=args.families,
            min_observed_gain=args.min_observed_gain,
            prediction_field=args.prediction_field,
            require_prediction_negative=args.require_prediction_negative,
            require_singleton_windows=not args.allow_non_singleton_windows,
            source_artifacts=source_artifact_metadata(
                {"dataset": args.dataset, "plan": args.plan}
            ),
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        MLXEffectiveSpendTriageSelectionError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(dumps_json(selection), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_mlx_effective_spend_triage_selection_markdown(selection),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "selected_count": selection["summary"]["selected_count"],
                "eligible_row_count": selection["summary"]["eligible_row_count"],
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
