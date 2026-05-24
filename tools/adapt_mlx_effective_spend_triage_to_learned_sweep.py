#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Adapt strict MLX effective-spend-triage selections for learned sweep."""

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

from tac.optimization.mlx_effective_spend_triage_learned_sweep_adapter import (  # noqa: E402
    DEFAULT_VARIANCE_FLOOR,
    MLXEffectiveSpendTriageLearnedSweepAdapterError,
    build_mlx_effective_spend_triage_learned_sweep_candidates,
    build_refusal_payload,
    dumps_json,
    load_json_object,
    source_artifact_metadata,
    write_json,
)
from tac.repo_io import ArtifactWriteError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--incumbent-score", type=float, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=DEFAULT_VARIANCE_FLOOR,
    )
    parser.add_argument(
        "--failure-json-out",
        type=Path,
        help="Optional fail-closed refusal artifact written when adaptation fails.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--expected-failure-output-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_artifacts: dict[str, Any] = {}
    try:
        source_artifacts = source_artifact_metadata({"selection": args.selection})
        selection = load_json_object(args.selection)
        payload = build_mlx_effective_spend_triage_learned_sweep_candidates(
            selection,
            incumbent_score=args.incumbent_score,
            top_k=args.top_k,
            source_artifacts=source_artifacts,
            variance_floor=args.variance_floor,
        )
        write_json(
            args.json_out,
            payload,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
    except (
        ArtifactWriteError,
        OSError,
        json.JSONDecodeError,
        MLXEffectiveSpendTriageLearnedSweepAdapterError,
        ValueError,
    ) as exc:
        if args.failure_json_out is not None:
            try:
                write_json(
                    args.failure_json_out,
                    build_refusal_payload(
                        error=str(exc),
                        source_artifacts=source_artifacts,
                    ),
                    allow_overwrite=args.allow_overwrite,
                    expected_existing_sha256=args.expected_failure_output_sha256,
                )
            except ArtifactWriteError as refusal_exc:
                print(
                    f"FATAL: {exc}; additionally failed to write refusal artifact: "
                    f"{refusal_exc}",
                    file=sys.stderr,
                )
                return 2
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        dumps_json(
            {
                "json_out": str(args.json_out),
                "adapted_candidate_count": payload["summary"][
                    "adapted_candidate_count"
                ],
                "best_predicted_score_mean": payload["summary"][
                    "best_predicted_score_mean"
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
