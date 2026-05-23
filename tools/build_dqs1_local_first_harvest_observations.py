#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build planning-only observation rows from DQS1 local-first harvests."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.dqs1_local_first_harvest_observations import (  # noqa: E402
    DQS1LocalHarvestObservationError,
    build_harvest_observation_summary,
    build_observation_rows_from_harvests,
    json_text,
    render_markdown_summary,
    write_observation_jsonl,
)


def _path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _expand_harvest_inputs(values: list[str]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        is_glob = glob.has_magic(value)
        matches = sorted(glob.glob(value))
        if not matches:
            matches = sorted(glob.glob(str(REPO_ROOT / value)))
        if matches:
            for match in matches:
                path = Path(match)
                if is_glob and path.name.startswith("dqs1_local_first_harvest_observations_"):
                    continue
                out.append(path)
        else:
            out.append(_path_arg(value))
    unique: dict[str, Path] = {}
    for path in out:
        unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harvest",
        action="append",
        default=[],
        help="Harvest JSON path or glob. May repeat.",
    )
    parser.add_argument(
        "--pairset-acquisition",
        type=_path_arg,
        required=True,
        help="DQS1 pairset acquisition JSON.",
    )
    parser.add_argument(
        "--baseline-advisory",
        type=_path_arg,
        required=True,
        help="Local top32 advisory JSON used for SegNet/PoseNet baseline components.",
    )
    parser.add_argument(
        "--baseline-archive-size-bytes",
        type=int,
        required=True,
        help="Archive byte count for the matching compact baseline packet.",
    )
    parser.add_argument(
        "--baseline-candidate-id",
        default="dqs1_top32_gap_uleb",
        help="Baseline id recorded in output rows.",
    )
    parser.add_argument("--jsonl-out", type=_path_arg, required=True)
    parser.add_argument("--summary-json-out", type=_path_arg)
    parser.add_argument("--md-out", type=_path_arg)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite --jsonl-out/summary/markdown if they already exist.",
    )
    return parser.parse_args(argv)


def _write_text(path: Path, text: str, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise DQS1LocalHarvestObservationError(
            f"{path} already exists; pass --replace to overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        harvest_paths = _expand_harvest_inputs(args.harvest)
        if not harvest_paths:
            raise DQS1LocalHarvestObservationError("at least one --harvest is required")
        missing = [str(path) for path in harvest_paths if not path.is_file()]
        if missing:
            raise DQS1LocalHarvestObservationError(
                "missing harvest JSON: " + ", ".join(missing)
            )
        rows = build_observation_rows_from_harvests(
            harvest_paths,
            repo_root=REPO_ROOT,
            pairset_acquisition_path=args.pairset_acquisition,
            baseline_advisory_path=args.baseline_advisory,
            baseline_archive_size_bytes=args.baseline_archive_size_bytes,
            baseline_candidate_id=args.baseline_candidate_id,
        )
        write_observation_jsonl(rows, output_path=args.jsonl_out, replace=args.replace)
        summary = build_harvest_observation_summary(
            rows,
            jsonl_path=args.jsonl_out,
            repo_root=REPO_ROOT,
        )
        if args.summary_json_out is not None:
            _write_text(args.summary_json_out, json_text(summary), replace=args.replace)
        if args.md_out is not None:
            _write_text(
                args.md_out,
                render_markdown_summary(summary),
                replace=args.replace,
            )
    except (OSError, DQS1LocalHarvestObservationError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json_text(summary), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
