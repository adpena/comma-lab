#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Append one fail-closed MLX dynamic sweep observation row."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.mlx_dynamic_sweep_observations import (  # noqa: E402
    EVIDENCE_TAG_MLX,
    MLXDynamicSweepObservationError,
    append_observation_row,
    build_observation_row,
    file_sha256,
    json_text,
    summarize_observation_file,
)


def _parse_component_delta(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("component deltas must use KEY=FLOAT")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("component delta key must be non-empty")
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{key} value must be numeric") from exc
    return key, parsed


def _parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for raw in value.split(","):
        text = raw.strip()
        if not text:
            continue
        try:
            out.append(int(text))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--selected-pair-indices must be a comma-separated integer list"
            ) from exc
    if not out:
        raise argparse.ArgumentTypeError(
            "--selected-pair-indices must include at least one integer"
        )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True, help="Observation JSONL to append.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--sweep-config-id", required=True)
    parser.add_argument("--optimization-pass-id", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--observed-axis", required=True)
    parser.add_argument("--evidence-grade")
    parser.add_argument("--evidence-tag", default=EVIDENCE_TAG_MLX)
    parser.add_argument("--observed-score-or-delta", type=float, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--raw-output-or-cache-sha256", required=True)
    parser.add_argument("--segnet-delta", type=float, required=True)
    parser.add_argument("--posenet-delta", type=float, required=True)
    parser.add_argument("--rate-delta", type=float, required=True)
    parser.add_argument(
        "--component-delta",
        action="append",
        default=[],
        type=_parse_component_delta,
        metavar="KEY=FLOAT",
        help="Optional additional component delta. May repeat.",
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        help="Local source artifact path; SHA-256 is computed automatically.",
    )
    parser.add_argument(
        "--source-artifact-path",
        help="Source artifact path when the file is not locally readable.",
    )
    parser.add_argument(
        "--source-artifact-sha256",
        help="SHA-256 for --source-artifact-path.",
    )
    parser.add_argument("--observed-at-utc")
    parser.add_argument("--run-id")
    parser.add_argument("--notes")
    parser.add_argument(
        "--selected-pair-indices",
        type=_parse_int_csv,
        help="Comma-separated pair indices for pairset identity/custody matching.",
    )
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument(
        "--allow-duplicate-observation",
        action="store_true",
        help="Intentionally append a duplicate candidate/axis/archive/raw/source observation.",
    )
    return parser.parse_args(argv)


def _source_artifact_args(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.source_artifact and args.source_artifact_path:
        raise MLXDynamicSweepObservationError(
            "use --source-artifact or --source-artifact-path, not both"
        )
    if args.source_artifact:
        return str(args.source_artifact), file_sha256(args.source_artifact)
    if args.source_artifact_path or args.source_artifact_sha256:
        if not args.source_artifact_path or not args.source_artifact_sha256:
            raise MLXDynamicSweepObservationError(
                "--source-artifact-path and --source-artifact-sha256 must be supplied together"
            )
        return args.source_artifact_path, args.source_artifact_sha256
    return None, None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_path, source_sha = _source_artifact_args(args)
        component_deltas: dict[str, Any] = {
            "segnet_delta": args.segnet_delta,
            "posenet_delta": args.posenet_delta,
            "rate_delta": args.rate_delta,
        }
        component_deltas.update(dict(args.component_delta))
        extra = {
            key: value
            for key, value in {
                "run_id": args.run_id,
                "notes": args.notes,
                "evidence_grade": args.evidence_grade,
                "selected_pair_indices": args.selected_pair_indices,
            }.items()
            if value is not None
        }
        row = build_observation_row(
            candidate_id=args.candidate_id,
            sweep_config_id=args.sweep_config_id,
            optimization_pass_id=args.optimization_pass_id,
            family=args.family,
            observed_axis=args.observed_axis,
            evidence_tag=args.evidence_tag,
            observed_score_or_delta=args.observed_score_or_delta,
            archive_sha256=args.archive_sha256,
            runtime_sha256=args.runtime_sha256,
            raw_output_or_cache_sha256=args.raw_output_or_cache_sha256,
            component_deltas=component_deltas,
            source_artifact_path=source_path,
            source_artifact_sha256=source_sha,
            observed_at_utc=args.observed_at_utc,
            extra=extra,
        )
        appended = append_observation_row(
            row,
            output_path=args.jsonl,
            allow_duplicate_observation=args.allow_duplicate_observation,
        )
        payload = summarize_observation_file(args.jsonl) if args.print_summary else appended
    except (OSError, MLXDynamicSweepObservationError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json_text(payload), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
