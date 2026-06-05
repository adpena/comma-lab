#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the SNeRV official TUB LF/HF replacement authority gate."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_official_tub_lf_hf_replacement_authority_gate import (  # noqa: E402
    DEFAULT_LANE_ID,
    DEFAULT_MIN_FREE_BYTES,
    build_snerv_official_tub_lf_hf_replacement_authority_gate,
    load_json_with_source_identity,
    render_snerv_official_tub_lf_hf_replacement_authority_gate_markdown,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-forward-artifact",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_official_mfu_hfr_tub_forward_parity.v1 JSON. Repeatable; "
            "the freshest/fullest source-forward artifact is selected."
        ),
    )
    parser.add_argument(
        "--checkpoint-export-report",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_checkpoint_archive_export.v1 JSON with "
            "official_checkpoint_export_binding evidence. Repeatable."
        ),
    )
    parser.add_argument(
        "--tub-source-forward-artifact",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_official_tub_source_forward_replay.v1 JSON. Repeatable; "
            "records fixture TUB replay separately from trained full replay."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument(
        "--allow-local-output",
        action="store_true",
        help="Allow non-SSD output root. Intended only for tests.",
    )
    args = parser.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_root
        or Path("/Volumes/VertigoDataTier/pact")
        / f"snerv_official_tub_lf_hf_replacement_authority_gate_{stamp}"
    )
    output_json = (
        args.output_json
        or output_root / "snerv_official_tub_lf_hf_replacement_authority_gate.json"
    )
    output_md = (
        args.output_md
        or output_root / "snerv_official_tub_lf_hf_replacement_authority_gate.md"
    )
    source_forward_artifacts = [
        load_json_with_source_identity(path) for path in args.source_forward_artifact
    ]
    checkpoint_export_reports = [
        load_json_with_source_identity(path) for path in args.checkpoint_export_report
    ]
    tub_source_forward_artifacts = [
        load_json_with_source_identity(path)
        for path in args.tub_source_forward_artifact
    ]
    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=source_forward_artifacts,
        checkpoint_export_reports=checkpoint_export_reports,
        tub_source_forward_artifacts=tub_source_forward_artifacts,
        output_root=output_root,
        lane_id=str(args.lane_id),
        min_free_bytes=int(args.min_free_bytes),
        allow_local_output=bool(args.allow_local_output),
    )
    json_result = write_json_artifact(output_json, report)
    md_result = write_text_artifact(
        output_md,
        render_snerv_official_tub_lf_hf_replacement_authority_gate_markdown(report),
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "lane_id": report["lane_id"],
                "output_json": json_result.path,
                "output_json_sha256": json_result.sha256,
                "output_md": md_result.path,
                "output_md_sha256": md_result.sha256,
                "official_tub_lf_hf_decoder_replacement_ready": report[
                    "official_tub_lf_hf_decoder_replacement_ready"
                ],
                "blocked_gate_row_count": report["blocked_gate_row_count"],
                "queue_blockers": report["queue_blockers"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
