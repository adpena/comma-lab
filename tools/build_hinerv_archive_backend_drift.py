#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare HiNeRV replay archive bytes across two local backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hinerv_archive_backend_drift import (  # noqa: E402
    DEFAULT_MAX_ABS_BYTE_DELTA,
    HINERV_ARCHIVE_BACKEND_DRIFT_SCHEMA,
    build_hinerv_archive_backend_drift_report,
    render_hinerv_archive_backend_drift_markdown,
)
from tac.repo_io import sha256_file, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", required=True, type=Path)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--max-abs-byte-delta",
        type=int,
        default=DEFAULT_MAX_ABS_BYTE_DELTA,
    )
    args = parser.parse_args(argv)

    reference_path = args.reference_json.expanduser().resolve(strict=False)
    candidate_path = args.candidate_json.expanduser().resolve(strict=False)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    report = build_hinerv_archive_backend_drift_report(
        reference,
        candidate,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        max_abs_byte_delta=int(args.max_abs_byte_delta),
    )
    report["reference_json_path"] = reference_path.as_posix()
    report["candidate_json_path"] = candidate_path.as_posix()
    report["reference_json_sha256"] = sha256_file(reference_path)
    report["candidate_json_sha256"] = sha256_file(candidate_path)

    output = args.output_json.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(
            render_hinerv_archive_backend_drift_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": HINERV_ARCHIVE_BACKEND_DRIFT_SCHEMA,
        "report_path": report.get("report_path"),
        "reference_label": report["reference_label"],
        "candidate_label": report["candidate_label"],
        "row_count": report["row_count"],
        "local_dev_velocity_ready": report["local_dev_velocity_ready"],
        "max_abs_byte_delta_observed": report["max_abs_byte_delta_observed"],
        "sum_byte_delta_candidate_minus_reference": (
            report["sum_byte_delta_candidate_minus_reference"]
        ),
        "sum_rate_score_delta_candidate_minus_reference": (
            report["sum_rate_score_delta_candidate_minus_reference"]
        ),
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
        "blockers": report["blockers"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
