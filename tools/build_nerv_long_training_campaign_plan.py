#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the false-authority HiNeRV/SNeRV long-training campaign plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_long_training_campaign_plan import (  # noqa: E402
    DEFAULT_BATCH_PAIRS,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPTIMIZER_KINDS,
    build_nerv_long_training_campaign_plan,
    render_nerv_long_training_campaign_plan_markdown,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hinerv-modelsize-budget", type=Path, required=True)
    parser.add_argument("--snerv-modelsize-budget", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-queue", type=Path)
    parser.add_argument("--expected-output-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    parser.add_argument("--expected-output-queue-sha256")
    parser.add_argument("--optimizer-kind", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-pairs", type=int, default=DEFAULT_BATCH_PAIRS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--joint-recon-weight-manifest",
        action="append",
        default=[],
        type=Path,
        help=(
            "Verified joint P18/P19 recon-pixel-weight manifest to pin in "
            "HiNeRV campaign rows. Repeatable for multiple pair counts."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns",
    )
    parser.add_argument("--max-candidates-per-family", type=int, default=3)
    args = parser.parse_args(argv)

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_load(args.hinerv_modelsize_budget),
        snerv_modelsize_budget=_load(args.snerv_modelsize_budget),
        optimizer_kinds=tuple(args.optimizer_kind or DEFAULT_OPTIMIZER_KINDS),
        epochs=args.epochs,
        batch_pairs=args.batch_pairs,
        learning_rate=args.learning_rate,
        output_root=args.output_root,
        max_candidates_per_family=args.max_candidates_per_family,
        joint_recon_weight_manifest_paths=tuple(args.joint_recon_weight_manifest),
    )
    write_json_artifact(
        args.output_json,
        report,
        allow_overwrite=args.expected_output_json_sha256 is not None,
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    if args.output_queue:
        write_json_artifact(
            args.output_queue,
            report["experiment_queue"],
            allow_overwrite=args.expected_output_queue_sha256 is not None,
            expected_existing_sha256=args.expected_output_queue_sha256,
        )
    if args.output_md:
        write_text_artifact(
            args.output_md,
            render_nerv_long_training_campaign_plan_markdown(report),
            allow_overwrite=args.expected_output_md_sha256 is not None,
            expected_existing_sha256=args.expected_output_md_sha256,
        )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "campaign_row_count": report["campaign_row_count"],
                "launchable_local_row_count": report[
                    "launchable_local_row_count"
                ],
                "blocked_row_count": report["blocked_row_count"],
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
                "output_json": args.output_json.as_posix(),
                "output_queue": (
                    None
                    if args.output_queue is None
                    else args.output_queue.as_posix()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object")
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
