#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest MLX drift-scope summaries into a reusable recommendation artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from comma_lab.scheduler.mlx_drift_scope_harvest import (  # noqa: E402
    SELECTION_POLICIES,
    build_mlx_drift_scope_recommendation_batch,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--selection-policy",
        choices=tuple(sorted(SELECTION_POLICIES)),
        default="minimal_no_cliff_then_best_delta",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batch = build_mlx_drift_scope_recommendation_batch(
        [_load_json(path) for path in args.summary],
        summary_paths=args.summary,
        repo_root=args.repo_root,
        selection_policy=args.selection_policy,
    )
    try:
        artifact = write_json_artifact(
            args.output,
            batch,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc
    primary = batch["primary_recommendation"]
    print(
        json.dumps(
            {
                "schema": batch["schema"],
                "output": artifact.path,
                "output_sha256": artifact.sha256,
                "primary_selected_candidate_id": primary["selected_candidate_id"],
                "primary_recommended_conv2d_override_preset": primary[
                    "recommended_conv2d_override_preset"
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
