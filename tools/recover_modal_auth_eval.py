#!/usr/bin/env python3
"""Recover artifacts from a detached Modal auth-eval dispatch.

The CUDA and CPU Modal auth-eval wrappers can launch with ``--detach`` so the
local process is not held open for a long scorer run. This tool is the single
recovery path for both axes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.deploy.modal.auth_eval import (  # noqa: E402
    ClaimSpec,
    read_spawn_metadata,
    recover_modal_auth_eval,
    terminal_modal_auth_eval_claim,
)


def _terminal_status(summary: dict, metadata: dict) -> str:
    if summary.get("status") == "pending":
        return ""
    axis = str(metadata.get("axis") or "")
    if summary.get("passed"):
        return (
            "completed_modal_cpu_auth_eval_recovered"
            if axis == "contest_cpu"
            else "completed_modal_auth_eval_recovered"
        )
    return (
        "failed_modal_cpu_auth_eval_no_score_claim"
        if axis == "contest_cpu"
        else "failed_modal_auth_eval_no_score_claim"
    )


def _claim_spec_from_metadata(metadata: dict) -> ClaimSpec | None:
    lane_id = metadata.get("lane_id")
    instance_job_id = metadata.get("instance_job_id")
    agent = metadata.get("claim_agent") or "codex:recover_modal_auth_eval"
    platform = metadata.get("claim_platform") or "modal"
    if not isinstance(lane_id, str) or not lane_id:
        return None
    if not isinstance(instance_job_id, str) or not instance_job_id:
        return None
    if not isinstance(agent, str) or not agent:
        agent = "codex:recover_modal_auth_eval"
    if not isinstance(platform, str) or not platform:
        platform = "modal"
    return ClaimSpec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=agent,
        platform=platform,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--call-id", default="", help="Override metadata call_id.")
    parser.add_argument("--timeout-s", type=float, default=0.0)
    parser.add_argument(
        "--no-close-claim",
        action="store_true",
        help="Harvest artifacts but do not write a terminal dispatch-claim row.",
    )
    args = parser.parse_args(argv)

    out_dir = args.output_dir.resolve()
    metadata = read_spawn_metadata(out_dir)
    summary = recover_modal_auth_eval(
        out_dir=out_dir,
        call_id=args.call_id or None,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("status") == "pending":
        return 4

    if not args.no_close_claim:
        claim_spec = _claim_spec_from_metadata(metadata)
        status = _terminal_status(summary, metadata)
        if claim_spec is not None and status:
            terminal_modal_auth_eval_claim(
                repo_root=REPO_ROOT,
                spec=claim_spec,
                status=status,
                notes=(
                    f"Modal auth eval recovered; passed={summary.get('passed')}; "
                    f"result_json={summary.get('result_json')}"
                ),
            )

    return 0 if summary.get("passed") else int(summary.get("returncode") or 1)


if __name__ == "__main__":
    raise SystemExit(main())
