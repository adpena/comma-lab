#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority PR95 baseline identity packet."""

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

from tac.analysis.pr95_baseline_identity import (  # noqa: E402
    build_pr95_baseline_identity,
    render_pr95_baseline_identity_markdown,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-artifact",
        action="append",
        required=True,
        type=Path,
        help=(
            "PR95 Stage-8 report, receiver/runtime proof, auth-eval JSON, or "
            "archive ZIP. Repeatable."
        ),
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/pr95_baseline_identity"),
        help="Output root embedded in the paired exact-eval work order.",
    )
    parser.add_argument("--expected-output-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    args = parser.parse_args(argv)

    report = build_pr95_baseline_identity(
        source_artifacts=tuple(args.source_artifact),
        output_root=args.output_root,
    )
    write_json_artifact(
        args.output_json,
        report,
        allow_overwrite=args.expected_output_json_sha256 is not None,
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    if args.output_md is not None:
        write_text_artifact(
            args.output_md,
            render_pr95_baseline_identity_markdown(report),
            allow_overwrite=args.expected_output_md_sha256 is not None,
            expected_existing_sha256=args.expected_output_md_sha256,
        )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "baseline_identity_reusable": report[
                    "baseline_identity_reusable"
                ],
                "candidate_archive_count": report["candidate_archive_count"],
                "selected_archive_sha256": (
                    report.get("selected_reusable_candidate_archive") or {}
                ).get("sha256"),
                "local_cpu_mlx_ready": report["local_cpu_mlx_work_order"][
                    "ready"
                ],
                "modal_dispatch_allowed": report["modal_dispatch_policy"][
                    "modal_dispatch_allowed"
                ],
                "blockers": report["blockers"],
                "output_json": args.output_json.as_posix(),
                "output_md": (
                    None if args.output_md is None else args.output_md.as_posix()
                ),
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
