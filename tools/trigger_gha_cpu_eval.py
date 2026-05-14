#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Trigger a [contest-CPU] GHA workflow_dispatch on the fork eval.yml workflow.

This is the QUEUE-INFRASTRUCTURE TRIGGER PHASE. It builds on (does NOT
duplicate) ``tools/dispatch_cpu_eval_via_github_actions.py``, which is the
canonical end-to-end dispatcher. This wrapper adds the discipline that was
missing from the canonical dispatcher's contract:

  1. **Lane-claim integration** (CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION"
     non-negotiable): claim the lane via ``tools/claim_lane_dispatch.py``
     BEFORE triggering, so a second concurrent dispatcher refuses to fire.
  2. **Standardized dispatch_metadata.json**: writes a queue-aware metadata
     file that ``tools/harvest_gha_cpu_eval.py`` can read to harvest the run.
  3. **Workflow existence verification**: confirms ``eval.yml`` is registered
     at the target repo BEFORE submitting any GH API call.

What it does NOT do:
  - Upload the archive (caller is expected to publish the release-asset URL,
    OR delegate to ``dispatch_cpu_eval_via_github_actions.py`` for the full
    create-release + upload + trigger flow).
  - Poll for completion (use the harvest counterpart).
  - Parse report.txt (use the harvest counterpart).

Per CLAUDE.md "NEVER invent CLI flags": every gh subcommand emitted here is
verified against the actual gh CLI surface. The eval.yml inputs come from the
real workflow file (``upstream/.github/workflows/eval.yml``).

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE": every triggered run defaults to
``runner=ubuntu-latest`` (Linux x86_64). The result is ``[contest-CPU]``
ONLY after the harvest tool actually parses a score; until then the
dispatch_metadata.json carries ``score_claim_valid=False``.

Per CLAUDE.md "Public Disclosure Hygiene": the asset URL must be a public
GH release-asset URL on a public repo (no credentials, no private infra).

Usage:
  python tools/trigger_gha_cpu_eval.py \\
    --archive-url https://github.com/<owner>/<repo>/releases/download/<tag>/archive.zip \\
    --archive-sha256 <hex64> \\
    --archive-size-bytes <int> \\
    --label t1_balle_<utc-stamp> \\
    --submission-name t1_balle_<utc-stamp> \\
    --pr-number <optional fork PR number>

Exit codes:
  0 — workflow_dispatch accepted; run id captured in dispatch_metadata.json
  2 — validation error (URL format, sha256 hex, etc.)
  3 — workflow not registered at target repo
  4 — gh CLI error / GitHub API error
  5 — lane-claim refused (active conflict)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FORK_REPO = "adpena/comma_video_compression_challenge"
DEFAULT_WORKFLOW_FILE = "eval.yml"
DEFAULT_RUNNER = "ubuntu-latest"
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_RELEASE_ASSET_URL_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/releases/download/[A-Za-z0-9._/-]+/[A-Za-z0-9._-]+$"
)


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_archive_url(url: str) -> None:
    """Refuse any URL that isn't a public GH release asset URL.

    Per CLAUDE.md "Public Disclosure Hygiene": GHA workflow_dispatch must use
    only public URLs. Refuses non-https schemes, private hosts, and obvious
    non-release patterns (like raw blob URLs that may be private).
    """
    if not isinstance(url, str) or not url.strip():
        raise SystemExit("VALIDATION_ERROR: --archive-url must be a non-empty string")
    if not url.startswith("https://github.com/"):
        raise SystemExit(
            "VALIDATION_ERROR: --archive-url must be a public github.com release-asset URL; "
            f"got {url!r}"
        )
    if "/releases/download/" not in url:
        raise SystemExit(
            "VALIDATION_ERROR: --archive-url must point at a GH releases/download/... asset; "
            f"got {url!r}"
        )
    if not _RELEASE_ASSET_URL_RE.match(url):
        raise SystemExit(
            "VALIDATION_ERROR: --archive-url did not match the GH release-asset URL pattern; "
            f"got {url!r}"
        )


def validate_archive_sha256(sha: str) -> None:
    if not isinstance(sha, str) or not _SHA256_HEX_RE.match(sha.strip()):
        raise SystemExit(
            "VALIDATION_ERROR: --archive-sha256 must be 64 lowercase hex chars; "
            f"got {sha!r}"
        )


def validate_archive_size_bytes(size: int) -> None:
    if not isinstance(size, int) or size <= 0:
        raise SystemExit(
            f"VALIDATION_ERROR: --archive-size-bytes must be a positive int; got {size!r}"
        )


def _parse_run_ids(stdout: str) -> set[int]:
    """Parse ``gh run list --json databaseId`` output.

    Run binding is custody-critical: if the pre/post run-list cannot be parsed,
    callers must fail closed instead of guessing which workflow run belongs to
    this archive.
    """
    rows = json.loads(stdout or "[]")
    if not isinstance(rows, list):
        raise ValueError("run list JSON must be an array")
    run_ids: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or "databaseId" not in row:
            raise ValueError(f"run list row missing databaseId: {row!r}")
        run_ids.add(int(row["databaseId"]))
    return run_ids


def validate_label(label: str) -> None:
    if not isinstance(label, str) or not label.strip():
        raise SystemExit("VALIDATION_ERROR: --label must be a non-empty string")
    if re.search(r"[^A-Za-z0-9_.-]", label):
        raise SystemExit(
            "VALIDATION_ERROR: --label may only contain A-Za-z0-9_.-; got "
            f"{label!r}"
        )


def run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a gh CLI command; capture stdout+stderr.

    Caller checks ``returncode``; this helper does not exit on failure to
    allow callers to inspect the error and exit with a precise code.
    """
    return subprocess.run(["gh", *args], check=False, capture_output=True, text=True)


def workflow_registered(repo: str, workflow_file: str) -> bool:
    """Return True iff the target repo has ``workflow_file`` registered."""
    result = run_gh(["workflow", "list", "-R", repo, "--json", "path,name,state"])
    if result.returncode != 0:
        sys.stderr.write(
            f"[gh-error] workflow list failed: rc={result.returncode}\n"
            f"  stderr: {result.stderr!r}\n"
        )
        return False
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    for row in rows:
        path = row.get("path", "")
        # path looks like ".github/workflows/eval.yml"
        if path.endswith(f"/{workflow_file}"):
            return True
    return False


def claim_lane(
    *,
    lane_id: str,
    instance_job_id: str,
    agent: str,
    notes: str,
    claims_path: Path | None = None,
) -> int:
    """Delegate to ``tools/claim_lane_dispatch.py claim``.

    Returns the helper's exit code; caller maps it to the trigger's exit code
    (0 = claimed; 3 = REFUSING_DISPATCH; 2 = validation).
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id",
        lane_id,
        "--platform",
        "github",
        "--instance-job-id",
        instance_job_id,
        "--agent",
        agent,
        "--status",
        "active_dispatch",
        "--notes",
        notes,
    ]
    if claims_path is not None:
        cmd.extend(["--claims-path", str(claims_path)])
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def close_lane_claim(
    *,
    lane_id: str,
    instance_job_id: str,
    agent: str,
    status: str,
    notes: str,
    claims_path: Path | None = None,
) -> int:
    """Append a terminal claim row for a trigger attempt that created no run."""

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--force",
        "--lane-id",
        lane_id,
        "--platform",
        "github",
        "--instance-job-id",
        instance_job_id,
        "--agent",
        agent,
        "--status",
        status,
        "--notes",
        notes,
    ]
    if claims_path is not None:
        cmd.extend(["--claims-path", str(claims_path)])
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def trigger_workflow_dispatch(
    *,
    repo: str,
    workflow_file: str,
    submission_name: str,
    submission_url: str,
    runner: str,
    pr_number: str | None,
) -> tuple[int | None, str]:
    """Submit workflow_dispatch and return ``(run_id, message)``.

    Returns ``(None, error_message)`` on failure. The run id is discovered by
    comparing the workflow's run list before and after dispatch.
    """
    # Capture existing run ids so we can find the new one after dispatch.
    pre_runs = run_gh(
        [
            "run",
            "list",
            "-R",
            repo,
            "-w",
            workflow_file,
            "-L",
            "20",
            "--json",
            "databaseId",
        ]
    )
    if pre_runs.returncode != 0:
        return None, f"could not capture pre-dispatch run list: stderr={pre_runs.stderr!r}"
    try:
        pre_ids = _parse_run_ids(pre_runs.stdout)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, f"could not parse pre-dispatch run list: {exc}"

    dispatch_args = [
        "workflow",
        "run",
        workflow_file,
        "-R",
        repo,
        "-f",
        f"submission_name={submission_name}",
        "-f",
        f"submission_url={submission_url}",
        "-f",
        f"runner={runner}",
    ]
    if pr_number:
        dispatch_args.extend(["-f", f"pr_number={pr_number}"])
    dispatch = run_gh(dispatch_args)
    if dispatch.returncode != 0:
        return None, f"gh workflow run failed: stderr={dispatch.stderr!r}"

    # Poll briefly for the new run id. workflow_dispatch returns no run id;
    # rely on the run-list delta. 60s is enough for the run to register.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        runs_q = run_gh(
            [
                "run",
                "list",
                "-R",
                repo,
                "-w",
                workflow_file,
                "-L",
                "20",
                "--json",
                "databaseId,status,createdAt",
            ]
        )
        if runs_q.returncode == 0 and runs_q.stdout.strip():
            try:
                current_ids = _parse_run_ids(runs_q.stdout)
            except (ValueError, TypeError, json.JSONDecodeError):
                current_ids = set()
            new_ids = sorted(current_ids - pre_ids)
            if len(new_ids) == 1:
                return new_ids[0], "ok"
            if len(new_ids) > 1:
                # Ambiguous: another agent may have dispatched concurrently.
                # Fail closed instead of binding dispatch_metadata.json to a
                # run that might carry a different archive with the same label.
                return None, f"ambiguous_concurrent_dispatch_new_run_ids={new_ids}"
        time.sleep(3)
    return None, "could not identify new run id within 60s"


def write_dispatch_metadata(
    output_dir: Path,
    *,
    label: str,
    submission_name: str,
    archive_url: str,
    archive_sha256: str,
    archive_size_bytes: int,
    repo: str,
    workflow_file: str,
    runner: str,
    pr_number: str | None,
    run_id: int | None,
    run_url: str | None,
    dispatched_at_utc: str,
    instance_job_id: str,
    lane_id: str,
    trigger_status: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "pact.gha_cpu_eval_dispatch_metadata.v1",
        "label": label,
        "submission_name": submission_name,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "archive_url": archive_url,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "fork_repo": repo,
        "workflow_file": workflow_file,
        "runner": runner,
        "pr_number": pr_number,
        "workflow_run_id": run_id,
        "workflow_run_url": run_url,
        "dispatched_at_utc": dispatched_at_utc,
        "trigger_status": trigger_status,
        "score_claim_valid": False,
        "evidence_grade": "pending_harvest",
        "lane_tag": "pending_harvest",
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "notes": [
            "This metadata is written by tools/trigger_gha_cpu_eval.py BEFORE the workflow completes.",
            "Run tools/harvest_gha_cpu_eval.py with this dispatch_metadata.json to harvest the score.",
            "[contest-CPU] tagging is forbidden until the harvest writes a score from report.txt.",
            "Hardware label after harvest will be 'github-actions-ubuntu-latest-x86_64' when runner=ubuntu-latest.",
        ],
    }
    out_path = output_dir / "dispatch_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def build_run_url(repo: str, run_id: int) -> str:
    return f"https://github.com/{repo}/actions/runs/{run_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trigger a [contest-CPU] GHA eval.yml workflow_dispatch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--archive-url", required=True, help="Public GH release-asset URL")
    parser.add_argument("--archive-sha256", required=True, help="64 hex chars")
    parser.add_argument("--archive-size-bytes", required=True, type=int)
    parser.add_argument(
        "--label",
        required=True,
        help="Unique dispatch label (A-Za-z0-9_.-); used to scope output_dir + instance_job_id",
    )
    parser.add_argument(
        "--submission-name",
        default=None,
        help="GHA workflow submission_name input; defaults to --label",
    )
    parser.add_argument(
        "--lane-id",
        default=None,
        help="Lane id for cross-agent dispatch claim; defaults to 'gha_cpu_eval_<label>'",
    )
    parser.add_argument("--repo", default=DEFAULT_FORK_REPO, help=f"default: {DEFAULT_FORK_REPO}")
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE)
    parser.add_argument("--runner", default=DEFAULT_RUNNER, choices=["ubuntu-latest", "linux-nvidia-t4"])
    parser.add_argument(
        "--pr-number",
        default=None,
        help="Optional fork PR number; required for non-baseline submissions whose "
        "submissions/<name>/inflate.sh isn't yet on master",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write dispatch_metadata.json (default: experiments/results/gha_cpu_eval_<label>_<utc>/)",
    )
    parser.add_argument(
        "--agent",
        default="claude:trigger_gha_cpu_eval",
        help="Agent name recorded in the dispatch claim ledger",
    )
    parser.add_argument(
        "--skip-claim",
        action="store_true",
        help="(diagnostic) skip the lane-claim step; not for production",
    )
    parser.add_argument(
        "--skip-trigger",
        action="store_true",
        help="(diagnostic) validate + claim only; do not call workflow_dispatch",
    )
    args = parser.parse_args(argv)

    validate_archive_url(args.archive_url)
    validate_archive_sha256(args.archive_sha256)
    validate_archive_size_bytes(args.archive_size_bytes)
    validate_label(args.label)

    submission_name = (args.submission_name or args.label).strip()
    validate_label(submission_name)
    lane_id = (args.lane_id or f"gha_cpu_eval_{args.label}").strip()
    instance_job_id = f"gha_cpu_eval_{args.label}_{_utc_now_compact()}"
    dispatched_at_utc = _utc_now_iso()

    output_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"gha_cpu_eval_{args.label}_{_utc_now_compact()}"
    )

    # Verify the workflow exists at the target repo BEFORE doing anything else.
    if not workflow_registered(args.repo, args.workflow_file):
        sys.stderr.write(
            f"[fatal] workflow {args.workflow_file!r} not registered at {args.repo!r}; "
            "verify the fork has the workflow checked in to .github/workflows/\n"
        )
        return 3

    # Claim the lane BEFORE dispatching.
    if not args.skip_claim:
        rc = claim_lane(
            lane_id=lane_id,
            instance_job_id=instance_job_id,
            agent=args.agent,
            notes=(
                f"GHA CPU eval trigger for submission_name={submission_name} "
                f"on {args.repo} runner={args.runner}; archive_sha={args.archive_sha256[:12]}..."
            ),
        )
        if rc == 3:
            sys.stderr.write(
                f"[refused] active dispatch claim conflict on lane_id={lane_id!r}; "
                "see .omx/state/active_lane_dispatch_claims.md\n"
            )
            return 5
        if rc != 0:
            sys.stderr.write(f"[fatal] claim_lane_dispatch.py exited rc={rc}\n")
            return 5

    if args.skip_trigger:
        out_path = write_dispatch_metadata(
            output_dir,
            label=args.label,
            submission_name=submission_name,
            archive_url=args.archive_url,
            archive_sha256=args.archive_sha256,
            archive_size_bytes=args.archive_size_bytes,
            repo=args.repo,
            workflow_file=args.workflow_file,
            runner=args.runner,
            pr_number=args.pr_number,
            run_id=None,
            run_url=None,
            dispatched_at_utc=dispatched_at_utc,
            instance_job_id=instance_job_id,
            lane_id=lane_id,
            trigger_status="skipped_trigger_diagnostic",
        )
        print(
            f"[trigger-skip] claim recorded; metadata at {out_path}\n"
            "  (re-run without --skip-trigger to fire the workflow)",
            flush=True,
        )
        if not args.skip_claim:
            close_rc = close_lane_claim(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=args.agent,
                status="refused_dispatch_skip_trigger_diagnostic",
                notes=(
                    "Diagnostic --skip-trigger path did not create a workflow run; "
                    f"metadata={out_path}; archive_sha={args.archive_sha256}"
                ),
            )
            if close_rc != 0:
                sys.stderr.write(
                    "[fatal] failed to append terminal claim row for --skip-trigger "
                    f"diagnostic path; rc={close_rc}; metadata at {out_path}\n"
                )
                return 5
        return 0

    run_id, msg = trigger_workflow_dispatch(
        repo=args.repo,
        workflow_file=args.workflow_file,
        submission_name=submission_name,
        submission_url=args.archive_url,
        runner=args.runner,
        pr_number=args.pr_number,
    )
    trigger_status = "ok" if run_id is not None and msg == "ok" else msg
    run_url = build_run_url(args.repo, run_id) if run_id is not None else None

    out_path = write_dispatch_metadata(
        output_dir,
        label=args.label,
        submission_name=submission_name,
        archive_url=args.archive_url,
        archive_sha256=args.archive_sha256,
        archive_size_bytes=args.archive_size_bytes,
        repo=args.repo,
        workflow_file=args.workflow_file,
        runner=args.runner,
        pr_number=args.pr_number,
        run_id=run_id,
        run_url=run_url,
        dispatched_at_utc=dispatched_at_utc,
        instance_job_id=instance_job_id,
        lane_id=lane_id,
        trigger_status=trigger_status,
    )

    if run_id is None:
        if not args.skip_claim:
            close_rc = close_lane_claim(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=args.agent,
                status="failed_gha_cpu_eval_trigger_no_run_id",
                notes=(
                    f"workflow_dispatch did not produce a workflow run; reason={msg!r}; "
                    f"metadata={out_path}; archive_sha={args.archive_sha256}"
                ),
            )
            if close_rc != 0:
                sys.stderr.write(
                    "[fatal] workflow_dispatch failed and terminal claim append also "
                    f"failed rc={close_rc}; reason={msg!r}; metadata at {out_path}\n"
                )
                return 5
        sys.stderr.write(
            f"[fatal] workflow_dispatch failed: {msg!r}; metadata at {out_path}\n"
        )
        return 4

    print(
        f"[ok] workflow_dispatch fired\n"
        f"  run_id:   {run_id}\n"
        f"  run_url:  {run_url}\n"
        f"  metadata: {out_path}\n"
        f"  next:     python tools/harvest_gha_cpu_eval.py --dispatch-metadata {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
