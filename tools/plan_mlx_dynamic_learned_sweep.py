#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan learned local/MLX sweep configs without score or dispatch authority."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.mlx_dynamic_learned_sweep import (  # noqa: E402
    MLXDynamicLearnedSweepError,
    build_mlx_dynamic_learned_sweep_plan,
    dumps_json,
    file_sha256,
    load_json_object,
    render_mlx_dynamic_learned_sweep_markdown,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-score", type=float, required=True)
    parser.add_argument(
        "--selector-pareto",
        type=Path,
        help="decoder_q_selective_selector_pareto.v1 JSON to rank as configs.",
    )
    parser.add_argument(
        "--candidate-payload",
        type=Path,
        action="append",
        default=[],
        help="Generic JSON object with candidates[]. May repeat.",
    )
    parser.add_argument(
        "--execution-configs",
        type=Path,
        help=(
            "Optional JSON list or object.execution_configs defining substrate/cost/"
            "quality rows. Defaults cover MLX, macOS CPU advisory, contest CPU, "
            "and contest CUDA diagnostic."
        ),
    )
    parser.add_argument(
        "--optimization-passes",
        type=Path,
        help=(
            "Optional JSON list or object.optimization_passes defining recursive "
            "smoke/micro/intermediate/macro sweep passes."
        ),
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--per-pass-top-k",
        type=int,
        help=(
            "When set, keep this many rows per optimization pass before the final "
            "top-k cap, so smoke/micro/intermediate/macro rows all remain visible."
        ),
    )
    parser.add_argument("--default-score-variance", type=float)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--expected-improvement-weight", type=float, default=1.0)
    parser.add_argument("--exploration-weight", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def _load_execution_configs(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("execution_configs"), list):
        return [item for item in payload["execution_configs"] if isinstance(item, dict)]
    raise MLXDynamicLearnedSweepError(
        "--execution-configs must be a JSON list or object.execution_configs"
    )


def _load_optimization_passes(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("optimization_passes"), list):
        return [item for item in payload["optimization_passes"] if isinstance(item, dict)]
    raise MLXDynamicLearnedSweepError(
        "--optimization-passes must be a JSON list or object.optimization_passes"
    )


def _source_artifacts(paths: dict[str, Path | list[Path] | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, value in paths.items():
        if value is None:
            continue
        if isinstance(value, list):
            out[label] = [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in value
            ]
        else:
            out[label] = {
                "path": str(value),
                "sha256": file_sha256(value),
                "bytes": value.stat().st_size,
            }
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selector_pareto = (
            load_json_object(args.selector_pareto) if args.selector_pareto else None
        )
        candidate_payloads = [load_json_object(path) for path in args.candidate_payload]
        kwargs: dict[str, Any] = {}
        if args.default_score_variance is not None:
            kwargs["default_score_variance"] = args.default_score_variance
        plan = build_mlx_dynamic_learned_sweep_plan(
            incumbent_score=args.incumbent_score,
            selector_pareto=selector_pareto,
            candidate_payloads=candidate_payloads,
            execution_configs=_load_execution_configs(args.execution_configs),
            optimization_passes=_load_optimization_passes(args.optimization_passes),
            top_k=args.top_k,
            per_pass_top_k=args.per_pass_top_k,
            lcb_z=args.lcb_z,
            expected_improvement_weight=args.expected_improvement_weight,
            exploration_weight=args.exploration_weight,
            source_artifacts=_source_artifacts(
                {
                    "selector_pareto": args.selector_pareto,
                    "candidate_payloads": args.candidate_payload,
                    "execution_configs": args.execution_configs,
                    "optimization_passes": args.optimization_passes,
                }
            ),
            **kwargs,
        )
    except (
        OSError,
        json.JSONDecodeError,
        MLXDynamicLearnedSweepError,
        ValueError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    write_json(args.json_out, plan)
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_mlx_dynamic_learned_sweep_markdown(plan),
            encoding="utf-8",
        )
    print(
        dumps_json(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "ranked_row_count": plan["summary"]["ranked_row_count"],
                "local_ready_row_count": plan["summary"]["local_ready_row_count"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
