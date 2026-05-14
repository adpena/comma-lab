#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recover artifacts from a detached Modal auth-eval dispatch.

The CUDA and CPU Modal auth-eval wrappers can launch with ``--detach`` so the
local process is not held open for a long scorer run. This tool is the single
recovery path for both axes.
"""
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

from tac.deploy.modal.auth_eval import (  # noqa: E402
    ClaimSpec,
    read_spawn_metadata,
    recover_modal_auth_eval,
    terminal_modal_auth_eval_claim,
)
from tac.continual_learning import (  # noqa: E402
    contest_result_from_auth_eval_payload,
    posterior_update_locked_from_auth_eval_json,
)


def _terminal_status(summary: dict, metadata: dict) -> str:
    if summary.get("status") == "pending":
        return ""
    axis = str(metadata.get("axis") or "")
    if summary.get("passed"):
        if axis == "contest_cuda" and summary.get("score_claim") is True:
            return "completed_contest_cuda_modal_auth_eval_recovered"
        if axis == "contest_cpu":
            return "completed_contest_cpu_modal_auth_eval_recovered"
        return "completed_modal_auth_eval_recovered_no_score_claim"
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


def _auth_eval_artifact_path(summary: dict) -> Path | None:
    out_dir = Path(str(summary.get("output_dir") or ""))
    for name in ("contest_auth_eval.json", "contest_auth_eval.adjudicated.json"):
        candidate = out_dir / name
        if candidate.is_file():
            return candidate
    result_json = summary.get("result_json")
    if isinstance(result_json, str) and result_json:
        candidate = Path(result_json)
        if candidate.is_file():
            return candidate
    return None


def _terminal_notes(summary: dict, metadata: dict, posterior_note: str = "") -> str:
    parts = [
        "Modal auth eval recovered",
        f"passed={summary.get('passed')}",
        f"result_json={summary.get('result_json')}",
    ]
    artifact = _auth_eval_artifact_path(summary)
    if artifact is not None:
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            result = contest_result_from_auth_eval_payload(
                payload,
                architecture_class=str(metadata.get("lane_id") or "unknown"),
                source_path=artifact,
            )
            parts.extend([
                f"archive_sha={result.archive_sha256}",
                f"archive_bytes={result.archive_bytes}",
                f"score_recomputed={result.score_value:.17g}",
                f"axis={result.axis}",
                f"hardware_substrate={result.hardware_substrate}",
            ])
        except Exception as exc:
            parts.append(f"auth_eval_bridge_error={type(exc).__name__}:{exc}")
    if posterior_note:
        parts.append(posterior_note)
    return "; ".join(parts)


def _maybe_update_posterior(summary: dict, metadata: dict) -> str:
    if summary.get("passed") is not True:
        return "posterior_update=skipped_not_passed"
    artifact = _auth_eval_artifact_path(summary)
    if artifact is None:
        return "posterior_update=skipped_missing_auth_eval_json"
    lane_id = metadata.get("lane_id")
    if not isinstance(lane_id, str) or not lane_id:
        return "posterior_update=skipped_missing_lane_id"
    update = posterior_update_locked_from_auth_eval_json(
        artifact,
        architecture_class=lane_id,
        notes="recover_modal_auth_eval",
    )
    status = "accepted" if update.accepted else "refused"
    return (
        f"posterior_update={status}; posterior_n={update.posterior_n_anchors_after}; "
        f"posterior_reason={update.refusal_reason or 'accepted'}"
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
    parser.add_argument(
        "--no-posterior-update",
        action="store_true",
        help="Recover artifacts but do not update the continual-learning posterior.",
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

    posterior_note = (
        "posterior_update=skipped_by_flag"
        if args.no_posterior_update
        else _maybe_update_posterior(summary, metadata)
    )

    if not args.no_close_claim:
        claim_spec = _claim_spec_from_metadata(metadata)
        status = _terminal_status(summary, metadata)
        if claim_spec is not None and status:
            terminal_modal_auth_eval_claim(
                repo_root=REPO_ROOT,
                spec=claim_spec,
                status=status,
                notes=_terminal_notes(summary, metadata, posterior_note),
            )

    if summary.get("passed"):
        return 0
    if summary.get("status") in {"remote_error", "invalid_result"}:
        return 5
    return int(summary.get("returncode") or 1)


if __name__ == "__main__":
    raise SystemExit(main())
