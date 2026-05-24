#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stamp validated macOS-CPU advisory artifacts onto learned-sweep selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.mlx_learned_sweep_advisory_handoff import (  # noqa: E402
    MLXLearnedSweepAdvisoryHandoffError,
    stamp_macos_cpu_advisory_paths,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    read_json,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--path-map", required=True, type=Path)
    parser.add_argument("--output-selection", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--source-artifact-root", type=Path)
    parser.add_argument("--require-all-selected", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-existing-selection-sha256")
    parser.add_argument("--expected-existing-report-sha256")
    return parser.parse_args(argv)


def _resolve(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    selection_path = _resolve(args.selection, repo_root=repo_root)
    path_map_path = _resolve(args.path_map, repo_root=repo_root)
    output_selection_path = _resolve(args.output_selection, repo_root=repo_root)
    report_output_path = _resolve(args.report_output, repo_root=repo_root)
    source_artifact_root = (
        path_map_path.parent
        if args.source_artifact_root is None
        else _resolve(args.source_artifact_root, repo_root=repo_root)
    )

    stamped_selection, report = stamp_macos_cpu_advisory_paths(
        read_json(selection_path),
        read_json(path_map_path),
        source_artifact_root=source_artifact_root,
        require_all_selected=args.require_all_selected,
    )
    selection_result = write_json_artifact(
        output_selection_path,
        stamped_selection,
        allow_overwrite=args.allow_overwrite,
        expected_existing_sha256=args.expected_existing_selection_sha256,
    )
    report_result = write_json_artifact(
        report_output_path,
        report,
        allow_overwrite=args.allow_overwrite,
        expected_existing_sha256=args.expected_existing_report_sha256,
    )
    print(
        json.dumps(
            {
                "schema": "mlx_learned_sweep_macos_cpu_advisory_handoff_cli.v1",
                "selection": selection_result.__dict__,
                "report": report_result.__dict__,
                "stamped_row_count": report["stamped_row_count"],
                "ready_for_macos_cpu_advisory_queue": report[
                    "ready_for_macos_cpu_advisory_queue"
                ],
                "missing_selection_ids": report["missing_selection_ids"],
                "unused_mapping_ids": report["unused_mapping_ids"],
                "authority": report["authority_boundary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, MLXLearnedSweepAdvisoryHandoffError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
