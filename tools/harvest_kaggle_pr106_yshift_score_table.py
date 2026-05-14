#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest the PR106 y-shift score-table Kaggle kernel through canonical ingest."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.deploy.kaggle.kaggle_output_ingest import (  # noqa: E402
    DEFAULT_EXCLUDE_OUTPUT_PATTERNS,
    download_kernel_outputs,
    ingest_downloaded_outputs,
)
from tac.repo_io import json_text, write_json  # noqa: E402

DEFAULT_KERNEL_REF = "adpena/comma-lab-pr106-yshift-score-table"
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "reports/raw/kaggle_pr106_yshift_score_table_latest"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/raw/kaggle_ingested"
DEFAULT_INCLUDE_PATTERNS = (
    r"^pr106_yshift_score_table/",
    r"^pact_pr106_yshift_workspace/inputs/pr106_archive\.zip$",
)


def default_run_id() -> str:
    return "kaggle_pr106_yshift_score_table_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_ingest_manifest(
    *,
    manifest_path: Path,
    run_id: str,
    kernel_ref: str,
) -> dict[str, object]:
    payload = {
        "schema": "kaggle_pr106_yshift_score_table_harvest_manifest_v1",
        "run_id": run_id,
        "slug": "pr106_yshift_score_table",
        "kernel_ref": kernel_ref,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotion_requires": "byte_closed_packet_and_contest_cuda_adjudication",
    }
    write_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-ref", default=DEFAULT_KERNEL_REF)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--include-output-pattern",
        action="append",
        default=list(DEFAULT_INCLUDE_PATTERNS),
        help="Regex for Kaggle output paths to keep. Defaults to the y-shift run prefix.",
    )
    parser.add_argument(
        "--exclude-output-pattern",
        action="append",
        default=list(DEFAULT_EXCLUDE_OUTPUT_PATTERNS),
        help="Regex for Kaggle output paths to skip.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or default_run_id()
    manifest_path = args.manifest_output or args.download_dir / "kaggle_pr106_yshift_score_table_manifest.json"
    manifest = write_ingest_manifest(
        manifest_path=manifest_path,
        run_id=run_id,
        kernel_ref=args.kernel_ref,
    )
    if not args.no_download:
        download_kernel_outputs(
            kernel_ref=args.kernel_ref,
            download_dir=args.download_dir,
            include_patterns=args.include_output_pattern,
            exclude_patterns=args.exclude_output_pattern,
        )
    summary = ingest_downloaded_outputs(
        manifest_path=manifest_path,
        download_dir=args.download_dir,
        output_root=args.output_root,
    )
    print(json_text({"manifest": manifest, "ingest_summary": summary}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
