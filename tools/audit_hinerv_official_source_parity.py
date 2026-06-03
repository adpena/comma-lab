#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit official HiNeRV OSS controls against local receiver-visible bindings."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hinerv_official_source_parity_audit import (  # noqa: E402
    build_hinerv_official_forward_parity_artifact,
    build_hinerv_official_source_parity_audit,
    render_hinerv_official_source_parity_markdown,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def _default_output_json() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/hinerv_official_source_parity_audit_{stamp}.json"


def _default_forward_parity_artifact_json() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/hinerv_official_forward_parity_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--official-repo-dir",
        type=Path,
        required=True,
        help="SSD-backed checkout of https://github.com/hmkx/HiNeRV.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--official-forward-parity-artifact",
        type=Path,
        help=(
            "Optional JSON proof artifact with schema "
            "hinerv_official_forward_parity.v1. Marker coverage alone never "
            "proves official HiNeRV source-forward parity."
        ),
    )
    parser.add_argument(
        "--output-forward-parity-artifact",
        type=Path,
        nargs="?",
        const=_default_forward_parity_artifact_json(),
        help=(
            "Write a source-backed HiNeRV proof-or-falsification artifact and "
            "consume it in the audit report. Provide a path or omit the value "
            "to use .omx/research."
        ),
    )
    parser.add_argument("--expected-output-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    parser.add_argument("--expected-output-forward-parity-artifact-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.official_forward_parity_artifact and args.output_forward_parity_artifact:
        parser.error("--official-forward-parity-artifact and --output-forward-parity-artifact are mutually exclusive")
    output_json = args.output_json or _default_output_json()
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    output_md = args.output_md
    if output_md is not None and not output_md.is_absolute():
        output_md = REPO_ROOT / output_md
    output_forward_parity_artifact = args.output_forward_parity_artifact
    if output_forward_parity_artifact is not None and not output_forward_parity_artifact.is_absolute():
        output_forward_parity_artifact = REPO_ROOT / output_forward_parity_artifact

    forward_artifact_result = None
    forward_artifact_path = args.official_forward_parity_artifact
    if output_forward_parity_artifact is not None:
        forward_artifact = build_hinerv_official_forward_parity_artifact(
            official_repo_dir=args.official_repo_dir,
            repo_root=args.repo_root,
        )
        forward_artifact_result = write_json_artifact(
            output_forward_parity_artifact,
            forward_artifact,
            allow_overwrite=args.expected_output_forward_parity_artifact_sha256 is not None,
            expected_existing_sha256=args.expected_output_forward_parity_artifact_sha256,
        )
        forward_artifact_path = Path(forward_artifact_result.path)

    report = build_hinerv_official_source_parity_audit(
        official_repo_dir=args.official_repo_dir,
        repo_root=args.repo_root,
        official_forward_parity_artifact_path=forward_artifact_path,
    )
    json_result = write_json_artifact(
        output_json,
        report,
        allow_overwrite=args.expected_output_json_sha256 is not None,
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    md_result = None
    if output_md is not None:
        md_result = write_text_artifact(
            output_md,
            render_hinerv_official_source_parity_markdown(report),
            allow_overwrite=args.expected_output_md_sha256 is not None,
            expected_existing_sha256=args.expected_output_md_sha256,
        )

    print(
        json.dumps(
            {
                "schema": report["schema"],
                "authority": report["authority"],
                "official_source_markers_present": report[
                    "official_source_markers_present"
                ],
                "local_receiver_bindings_present": report[
                    "local_receiver_bindings_present"
                ],
                "official_forward_parity_proven": report[
                    "official_forward_parity_proven"
                ],
                "blocker_count": len(report["blockers"]),
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
                "output_json": json_result.path,
                "output_json_sha256": json_result.sha256,
                "output_md": None if md_result is None else md_result.path,
                "output_md_sha256": None if md_result is None else md_result.sha256,
                "output_forward_parity_artifact": (
                    None if forward_artifact_result is None else forward_artifact_result.path
                ),
                "output_forward_parity_artifact_sha256": (
                    None if forward_artifact_result is None else forward_artifact_result.sha256
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
