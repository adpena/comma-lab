#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build the fail-closed top-priority NeRV stack orchestration artifact."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_top_priority_stack_seam import (  # noqa: E402
    DEFAULT_LANE_ID,
    build_nerv_top_priority_stack_seam,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/nerv_top_priority_stack_seam_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--upstream-repo-dir",
        type=Path,
        default=REPO_ROOT / "upstream",
    )
    parser.add_argument(
        "--pr95-intake-root",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/results/public_pr_archive_release_view/"
            "public_pr95_intake_20260505_auto"
        ),
    )
    parser.add_argument(
        "--active-claims-path",
        type=Path,
        default=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md",
    )
    parser.add_argument(
        "--pr95-pr-metadata-json",
        type=Path,
        help="Optional gh-pr-view JSON payload for PR95.",
    )
    parser.add_argument("--pr95-pr-url")
    parser.add_argument("--pr95-pr-title")
    parser.add_argument("--pr95-pr-state")
    parser.add_argument("--pr95-head-sha")
    parser.add_argument("--pr95-head-ref")
    parser.add_argument("--snerv-oss-head-sha")
    parser.add_argument("--hinerv-oss-head-sha")
    parser.add_argument("--hnerv-oss-head-sha")
    parser.add_argument(
        "--oss-audit-root",
        type=Path,
        help="SSD-backed source-audit root containing official NeRV repo snapshots.",
    )
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing --out only with --expected-existing-sha256.",
    )
    parser.add_argument("--expected-existing-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pr95_pr_metadata = None
    if args.pr95_pr_metadata_json is not None:
        pr95_pr_metadata = json.loads(
            args.pr95_pr_metadata_json.read_text(encoding="utf-8")
        )
    flag_metadata = {
        "url": args.pr95_pr_url,
        "title": args.pr95_pr_title,
        "state": args.pr95_pr_state,
        "headRefOid": args.pr95_head_sha,
        "headRefName": args.pr95_head_ref,
    }
    flag_metadata = {key: value for key, value in flag_metadata.items() if value}
    if flag_metadata:
        pr95_pr_metadata = {**(pr95_pr_metadata or {}), **flag_metadata}

    oss_source_metadata = {}
    oss_audit_root = args.oss_audit_root.resolve() if args.oss_audit_root else None
    for stack_id, head_sha, repo_name, repo_url in (
        ("snerv", args.snerv_oss_head_sha, "SNeRV", "https://github.com/qwertja/SNeRV.git"),
        ("hinerv", args.hinerv_oss_head_sha, "HiNeRV", "https://github.com/hmkx/HiNeRV.git"),
        (
            "hnerv_pr95_control",
            args.hnerv_oss_head_sha,
            "HNeRV",
            "https://github.com/haochen-rye/HNeRV.git",
        ),
    ):
        row = {"repo_url": repo_url}
        if head_sha:
            row["head_sha"] = head_sha
        if oss_audit_root is not None:
            row["audit_root"] = (oss_audit_root / "repos" / repo_name).as_posix()
        if len(row) > 1:
            oss_source_metadata[stack_id] = row

    payload = build_nerv_top_priority_stack_seam(
        repo_root=args.repo_root,
        upstream_repo_dir=args.upstream_repo_dir,
        pr95_intake_root=args.pr95_intake_root,
        active_claims_path=args.active_claims_path,
        pr95_pr_metadata=pr95_pr_metadata,
        oss_source_metadata=oss_source_metadata,
        lane_id=args.lane_id,
    )
    out_path = args.out or _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    result = write_json_artifact(
        out_path,
        payload,
        allow_overwrite=args.allow_overwrite,
        expected_existing_sha256=args.expected_existing_sha256,
    )

    print("[NeRV top-priority stack seam] false-authority")
    print(f"  verdict: {payload['go_no_go_verdict']}")
    print(f"  carriers: {payload['top_priority_carriers']}")
    print(f"  baseline: {payload['baseline_to_beat']}")
    print(f"  blocked_dispatch: {payload['blocked_dispatch']}")
    print(f"  dispatch_blockers: {payload['dispatch_blockers']}")
    print(f"  blockers: {len(payload['blockers'])}")
    print(f"  wrote {result.path} ({result.bytes_written} bytes sha256={result.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
