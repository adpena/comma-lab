#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest the PR106 latent score-table Kaggle kernel through canonical ingest."""
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

from tac.deploy.claims import DispatchClaimSpec, terminal_dispatch_claim  # noqa: E402
from tac.deploy.kaggle.kaggle_output_ingest import (  # noqa: E402
    DEFAULT_EXCLUDE_OUTPUT_PATTERNS,
    download_kernel_outputs,
    ingest_downloaded_outputs,
)
from tac.repo_io import json_text, write_json  # noqa: E402

DEFAULT_KERNEL_REF = "adpena/comma-lab-pr106-latent-score-table"
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "reports/raw/kaggle_pr106_latent_score_table_latest"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/raw/kaggle_ingested"
DEFAULT_LANE_ID = "lane_pr106_latent_sidecar"
DEFAULT_AGENT = "codex:gpt-5.5"
DEFAULT_INCLUDE_PATTERNS = (
    r"^pr106_latent_score_table/",
    r"^pact_pr106_latent_workspace/inputs/pr106_archive\.zip$",
)


def default_run_id() -> str:
    return "kaggle_pr106_latent_score_table_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_ingest_manifest(
    *,
    manifest_path: Path,
    run_id: str,
    kernel_ref: str,
    lane_id: str,
    instance_job_id: str,
    terminal_claim_on_harvest: bool,
    terminal_claim_suppression_reason: str = "",
) -> dict[str, object]:
    payload = {
        "schema": "kaggle_pr106_latent_score_table_harvest_manifest_v1",
        "run_id": run_id,
        "slug": "pr106_latent_score_table",
        "kernel_ref": kernel_ref,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "terminal_claim_on_harvest": terminal_claim_on_harvest,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotion_requires": "byte_closed_packet_and_contest_cuda_adjudication",
    }
    if terminal_claim_suppression_reason:
        payload["terminal_claim_suppression_reason"] = terminal_claim_suppression_reason
    write_json(manifest_path, payload)
    return payload


def terminal_status_from_ingest_summary(summary: dict[str, object]) -> str:
    """Classify a PR106 latent score-table harvest into a terminal claim status."""

    if isinstance(summary.get("latest_failure"), dict):
        return "failed_kaggle_kernel_error"
    if isinstance(summary.get("latest_score_table"), dict):
        return "completed_kaggle_score_table_harvested_no_score_claim"
    return "failed_kaggle_no_score_table"


def close_terminal_claim_from_summary(
    *,
    summary: dict[str, object],
    lane_id: str,
    instance_job_id: str,
    agent: str,
) -> dict[str, str]:
    """Append the terminal dispatch claim row for a harvested Kaggle run."""

    status = terminal_status_from_ingest_summary(summary)
    evidence_dir = str(summary.get("evidence_dir") or "")
    failure = summary.get("latest_failure")
    if isinstance(failure, dict):
        detail = f"failure={failure.get('error_type', 'UnknownError')}:{failure.get('message', '')}"
    else:
        score_table = summary.get("latest_score_table")
        if isinstance(score_table, dict):
            detail = (
                "score_table="
                f"{score_table.get('manifest_path', '')}; "
                f"ready_for_builder={score_table.get('ready_for_builder')}"
            )
        else:
            detail = "no failure object and no score_table_manifest found"
    notes = (
        "Kaggle PR106 latent score-table harvest; "
        "score_claim=false; promotion_eligible=false; "
        f"evidence={evidence_dir}; {detail}"
    )
    terminal_dispatch_claim(
        repo_root=REPO_ROOT,
        spec=DispatchClaimSpec(
            lane_id=lane_id,
            instance_job_id=instance_job_id,
            agent=agent,
            platform="kaggle",
        ),
        status=status,
        notes=notes,
    )
    return {"status": status, "notes": notes}


def has_terminal_ingest_summary(summary: dict[str, object]) -> bool:
    return isinstance(summary.get("latest_failure"), dict) or isinstance(
        summary.get("latest_score_table"), dict
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-ref", default=DEFAULT_KERNEL_REF)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--instance-job-id", default="")
    parser.add_argument("--claim-agent", default=DEFAULT_AGENT)
    parser.add_argument(
        "--close-claim",
        dest="close_claim",
        action="store_true",
        default=True,
        help="Close a terminal dispatch claim after a terminal ingest summary is found (default).",
    )
    parser.add_argument(
        "--no-close-claim",
        dest="close_claim",
        action="store_false",
        help="Suppress terminal dispatch-claim closure. Requires --skip-close-claim-reason.",
    )
    parser.add_argument(
        "--skip-close-claim-reason",
        default="",
        help="Auditable reason for --no-close-claim.",
    )
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--include-output-pattern",
        action="append",
        default=list(DEFAULT_INCLUDE_PATTERNS),
        help="Regex for Kaggle output paths to keep. Defaults to the latent run prefix.",
    )
    parser.add_argument(
        "--exclude-output-pattern",
        action="append",
        default=list(DEFAULT_EXCLUDE_OUTPUT_PATTERNS),
        help="Regex for Kaggle output paths to skip.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    skip_close_reason = str(args.skip_close_claim_reason or "").strip()
    if not args.close_claim and not skip_close_reason:
        parser.error("--no-close-claim requires --skip-close-claim-reason")
    if args.close_claim and skip_close_reason:
        parser.error("--skip-close-claim-reason is only valid with --no-close-claim")
    run_id = args.run_id or default_run_id()
    instance_job_id = args.instance_job_id or run_id
    manifest_path = args.manifest_output or args.download_dir / "kaggle_pr106_latent_score_table_manifest.json"
    manifest = write_ingest_manifest(
        manifest_path=manifest_path,
        run_id=run_id,
        kernel_ref=args.kernel_ref,
        lane_id=args.lane_id,
        instance_job_id=instance_job_id,
        terminal_claim_on_harvest=args.close_claim,
        terminal_claim_suppression_reason=skip_close_reason,
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
    terminal_claim: dict[str, str] | None = None
    terminal_summary_exists = has_terminal_ingest_summary(summary)
    if args.close_claim and terminal_summary_exists:
        terminal_claim = close_terminal_claim_from_summary(
            summary=summary,
            lane_id=args.lane_id,
            instance_job_id=instance_job_id,
            agent=args.claim_agent,
        )
    elif not args.close_claim and terminal_summary_exists:
        terminal_claim = {
            "status": "skipped_no_close_claim",
            "notes": f"terminal dispatch-claim closure suppressed: {skip_close_reason}",
        }
    print(
        json_text(
            {
                "manifest": manifest,
                "ingest_summary": summary,
                "terminal_claim": terminal_claim,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
