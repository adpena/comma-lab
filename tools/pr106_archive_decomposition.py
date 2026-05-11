#!/usr/bin/env python3
"""Inspect PR101/PR106 frontier archive physical and parser-proven layout.

Legacy filename retained because task notes referred to "PR106 archive
decomposition". The tool is deliberately stricter than filename heuristics:
single-member frontier archives are reported as monolithic packets, and mask or
pose budgets are only available if an internal parser proves a logical section.

This is CPU-only archive custody evidence. It emits no score claim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.frontier_archive_layout import (  # noqa: E402
    dumps_manifest,
    inspect_frontier_archive_layout,
    render_frontier_archive_layout_summary,
)
from tac.repo_io import json_text  # noqa: E402

DEFAULT_ARCHIVES = (
    REPO_ROOT / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
    REPO_ROOT / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        action="append",
        default=None,
        help="Archive zip to inspect. May be repeated. Defaults to local PR101 and PR106 frontier archives.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "reports/frontier_monolithic_archive_layout_20260508.json",
        help="Path for the JSON manifest.",
    )
    parser.add_argument(
        "--summary-text",
        action="store_true",
        help="Print human-readable summaries after writing JSON.",
    )
    args = parser.parse_args(argv)

    archives = tuple(args.archive) if args.archive else DEFAULT_ARCHIVES
    manifests = [inspect_frontier_archive_layout(path) for path in archives]
    payload = {
        "schema": "tac_frontier_archive_layout_batch_v1",
        "score_claim": False,
        "evidence_grade": "empirical_archive_layout_cpu_no_score",
        "runs": manifests,
        "global_implication": (
            "frontier HNeRV archives are monolithic physical packets; "
            "component budgets must be internal-parser-proven logical slices, "
            "not ZIP member-name categories"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(dumps_manifest(payload), encoding="utf-8")

    if args.summary_text:
        for manifest in manifests:
            print(render_frontier_archive_layout_summary(manifest))
            print()
        print(f"manifest: {args.output_json}")
    else:
        sys.stdout.write(json_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
