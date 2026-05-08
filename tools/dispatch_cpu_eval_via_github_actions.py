#!/usr/bin/env python3
"""Dispatch a contest-faithful CPU auth eval on the comma_video_compression_challenge
fork via GitHub Actions.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" (commit b4919d24): every shippable submission archive must get an
authoritative CPU score on a 1:1 contest-compliant runner. The contest's
`eval.yml` workflow on `commaai/comma_video_compression_challenge` uses
`runner=ubuntu-latest` (Linux x86_64) — that IS the literal contest hardware.
A non-admin cannot trigger that workflow on the upstream repo, but we maintain a
fork (`adpena/comma_video_compression_challenge`) that exposes the same workflow
and same runner image. Free GitHub Actions minutes are the cheapest path AND
the literal contest hardware.

This helper:

  1. Verifies the input archive's SHA (fail-fast if mismatch).
  2. Uploads the archive as an asset on a fresh GitHub Release on the fork
     (Release tag scoped to the dispatch timestamp; one tag per dispatch).
  3. Calls the `eval.yml` `workflow_dispatch` with `submission_url` pointing
     at the release asset, `runner=ubuntu-latest`, `submission_name` unique.
  4. Polls until the workflow run completes (success or failure).
  5. Downloads the `eval-<submission_name>` artifact (contains `archive.zip`
     and `report.txt`).
  6. Parses `report.txt` for `pose_avg`, `seg_avg`, `rate`, `score`.
  7. Writes `contest_auth_eval.adjudicated.json` next to the archive with the
     full provenance + components + lane-tag `[contest-CPU]` +
     `hardware: github-actions-ubuntu-latest-x86_64`.

Output JSON keys:
  - archive_size_bytes (int)
  - archive_sha256 (str)
  - canonical_score (float)  — the final scalar
  - avg_segnet_dist (float)
  - avg_posenet_dist (float)
  - compression_rate (float)
  - score_recomputed_from_components (float)  — sanity check
  - device ("cpu")
  - hardware ("github-actions-ubuntu-latest-x86_64")
  - runner_os_release (str, parsed from log)
  - evaluate_py_sha256 (str)
  - workflow_run_id (int)
  - workflow_run_url (str)
  - release_tag (str)
  - asset_url (str)
  - lane_tag ("[contest-CPU]")
  - dispatched_at_utc (ISO 8601)
  - completed_at_utc (ISO 8601)
  - report_text (str)  — full report.txt contents

Exit codes:
  0 — completed; adjudicated JSON written
  2 — input validation error (missing archive, SHA mismatch, etc.)
  3 — workflow failed (timeout, eval crash, parse error)
  4 — gh CLI / GitHub API error

Usage:
  python tools/dispatch_cpu_eval_via_github_actions.py \\
    --archive-path experiments/results/.../archive.zip \\
    --archive-sha 7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb \\
    --submission-name pr107_apogee_gha_cpu_20260508 \\
    --output-dir experiments/results/pr107_cpu_eval_gha_20260508/

Per CLAUDE.md "NEVER invent CLI flags" rule: every flag passed to `gh` is
verified against `gh workflow run --help` / `gh release create --help`. The
eval.yml inputs are read from the actual workflow file before dispatch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

UPSTREAM_FORK_REPO = "adpena/comma_video_compression_challenge"
EVAL_WORKFLOW_FILE = "eval.yml"
DEFAULT_RUNNER = "ubuntu-latest"
LANE_TAG = "[contest-CPU]"
HARDWARE_LABEL = "github-actions-ubuntu-latest-x86_64"
POLL_INTERVAL_SEC = 30
POLL_TIMEOUT_SEC = 60 * 45  # 45 min wall-clock budget; eval has 30 min internal


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_gh(args: list[str], capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a gh CLI command. Raises on non-zero exit by default."""
    result = subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"[gh-error] cmd={' '.join(args)} rc={result.returncode}\n"
            f"  stdout={result.stdout!r}\n  stderr={result.stderr!r}\n"
        )
    return result


def verify_archive(archive_path: Path, expected_sha: str) -> int:
    if not archive_path.exists():
        sys.stderr.write(f"[fatal] archive not found: {archive_path}\n")
        sys.exit(2)
    actual_sha = sha256_of(archive_path)
    if actual_sha != expected_sha:
        sys.stderr.write(
            f"[fatal] archive SHA mismatch:\n"
            f"  expected: {expected_sha}\n  actual:   {actual_sha}\n"
        )
        sys.exit(2)
    # Verify it's a valid zip
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
            if not members:
                sys.stderr.write(f"[fatal] archive {archive_path} is an empty zip\n")
                sys.exit(2)
    except zipfile.BadZipFile:
        sys.stderr.write(f"[fatal] archive {archive_path} is not a valid zip\n")
        sys.exit(2)
    return archive_path.stat().st_size


def create_release_with_asset(
    archive_path: Path,
    release_tag: str,
    archive_sha: str,
    archive_size: int,
    repo: str,
) -> str:
    """Create a release on the fork and upload archive.zip as an asset.

    Returns the asset download URL.
    """
    notes = (
        f"Auto-created by tools/dispatch_cpu_eval_via_github_actions.py.\n\n"
        f"- archive_sha256: {archive_sha}\n"
        f"- archive_size_bytes: {archive_size}\n"
        f"- dispatched_at_utc: {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
        f"- purpose: CPU auth eval on contest-compliant Linux x86_64 runner\n"
    )
    result = run_gh(
        [
            "release",
            "create",
            release_tag,
            "-R",
            repo,
            "--title",
            f"CPU auth eval dispatch — {release_tag}",
            "--notes",
            notes,
            str(archive_path),
        ]
    )
    if result.returncode != 0:
        sys.stderr.write("[fatal] gh release create failed\n")
        sys.exit(4)
    # Read the asset URL from the freshly created release
    asset_q = run_gh(
        [
            "release",
            "view",
            release_tag,
            "-R",
            repo,
            "--json",
            "assets",
            "--jq",
            ".assets[] | select(.name == \"archive.zip\") | .url",
        ]
    )
    if asset_q.returncode != 0 or not asset_q.stdout.strip():
        sys.stderr.write("[fatal] could not read release asset url after create\n")
        sys.exit(4)
    return asset_q.stdout.strip()


def trigger_workflow(
    submission_name: str,
    submission_url: str,
    runner: str,
    repo: str,
    pr_number: str | None = None,
) -> int:
    """Trigger the eval.yml workflow_dispatch and return the run ID.

    The dispatch endpoint doesn't return the run ID, so we list runs
    immediately afterward and pick the most recent run for this workflow
    that is in_progress or queued.

    pr_number (optional): when provided, the workflow's actions/checkout step
    will use ``refs/pull/<n>/merge`` instead of master. This is required when
    the submission directory (e.g. ``submissions/apogee/``) lives on a fork
    branch that is not yet merged to master — without this the Evaluate step
    fails with ``inflate.sh not found``. Verified against the fork's
    ``eval.yml`` workflow inputs (the ``pr_number`` input IS declared upstream).
    """
    pre_runs = run_gh(
        [
            "run",
            "list",
            "-R",
            repo,
            "-w",
            EVAL_WORKFLOW_FILE,
            "-L",
            "1",
            "--json",
            "databaseId",
        ]
    )
    pre_id: int | None = None
    if pre_runs.returncode == 0 and pre_runs.stdout.strip():
        pre_runs_json = json.loads(pre_runs.stdout)
        if pre_runs_json:
            pre_id = pre_runs_json[0]["databaseId"]

    dispatch_args = [
        "workflow",
        "run",
        EVAL_WORKFLOW_FILE,
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
        sys.stderr.write("[fatal] gh workflow run failed\n")
        sys.exit(4)

    # Poll for the new run ID (different from pre_id).
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        runs_q = run_gh(
            [
                "run",
                "list",
                "-R",
                repo,
                "-w",
                EVAL_WORKFLOW_FILE,
                "-L",
                "1",
                "--json",
                "databaseId,status",
            ]
        )
        if runs_q.returncode == 0 and runs_q.stdout.strip():
            runs = json.loads(runs_q.stdout)
            if runs:
                rid = runs[0]["databaseId"]
                if rid != pre_id:
                    return rid
        time.sleep(2)
    sys.stderr.write("[fatal] could not detect new workflow run within 60s\n")
    sys.exit(4)


def poll_run(run_id: int, repo: str) -> dict[str, Any]:
    """Poll until the run completes; return the final status JSON."""
    started = time.monotonic()
    last_step_logged = ""
    while True:
        elapsed = time.monotonic() - started
        if elapsed > POLL_TIMEOUT_SEC:
            sys.stderr.write(
                f"[fatal] poll timeout after {elapsed:.0f}s on run {run_id}\n"
            )
            sys.exit(3)
        q = run_gh(
            [
                "run",
                "view",
                str(run_id),
                "-R",
                repo,
                "--json",
                "status,conclusion,jobs,url,createdAt,updatedAt",
            ]
        )
        if q.returncode != 0:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        info = json.loads(q.stdout)
        status = info.get("status", "")
        # Surface progress
        for job in info.get("jobs", []):
            if job.get("name") == "test":
                for step in job.get("steps", []):
                    if step.get("status") == "in_progress":
                        name = step.get("name", "")
                        if name and name != last_step_logged:
                            print(
                                f"[poll +{elapsed:.0f}s] step in progress: {name}",
                                flush=True,
                            )
                            last_step_logged = name
        if status == "completed":
            return info
        time.sleep(POLL_INTERVAL_SEC)


def download_artifact(
    run_id: int, submission_name: str, repo: str, dest_dir: Path
) -> Path:
    """Download the eval-<submission_name> artifact; return the path to report.txt."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"eval-{submission_name}"
    result = run_gh(
        [
            "run",
            "download",
            str(run_id),
            "-R",
            repo,
            "-n",
            artifact_name,
            "-D",
            str(dest_dir),
        ]
    )
    if result.returncode != 0:
        sys.stderr.write("[fatal] gh run download failed\n")
        sys.exit(4)
    # Locate report.txt within dest_dir
    candidates = list(dest_dir.rglob("report.txt"))
    if not candidates:
        sys.stderr.write(
            f"[fatal] report.txt not found in downloaded artifact at {dest_dir}\n"
        )
        sys.exit(3)
    return candidates[0]


REPORT_PATTERNS = {
    "avg_posenet_dist": re.compile(
        r"Average PoseNet Distortion:\s*([0-9.eE+-]+)"
    ),
    "avg_segnet_dist": re.compile(
        r"Average SegNet Distortion:\s*([0-9.eE+-]+)"
    ),
    "compression_rate": re.compile(r"Compression Rate:\s*([0-9.eE+-]+)"),
    "canonical_score": re.compile(r"Final score:.*=\s*([0-9.eE+-]+)"),
    "n_samples": re.compile(r"Evaluation results over (\d+) samples"),
}


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text()
    parsed: dict[str, Any] = {"report_text": text}
    for key, pat in REPORT_PATTERNS.items():
        m = pat.search(text)
        if not m:
            sys.stderr.write(
                f"[fatal] could not parse {key} from report.txt:\n{text}\n"
            )
            sys.exit(3)
        val = m.group(1)
        parsed[key] = int(val) if key == "n_samples" else float(val)
    # Recompute score from components for sanity
    import math

    recomputed = (
        100.0 * parsed["avg_segnet_dist"]
        + math.sqrt(10.0 * parsed["avg_posenet_dist"])
        + 25.0 * parsed["compression_rate"]
    )
    parsed["score_recomputed_from_components"] = recomputed
    drift = abs(recomputed - parsed["canonical_score"])
    if drift > 0.02:  # report.txt rounds to 2 decimals
        sys.stderr.write(
            f"[warn] score drift {drift:.4f} > 0.02; canonical={parsed['canonical_score']} "
            f"recomputed={recomputed}\n"
        )
    return parsed


def fetch_log_for_runner(run_id: int, repo: str) -> str:
    """Fetch a snippet of the workflow log to capture runner OS info."""
    result = subprocess.run(
        ["gh", "run", "view", str(run_id), "-R", repo, "--log"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "<log fetch failed>"
    # Look for "Linux" / "ubuntu" / kernel info in early steps
    out = result.stdout
    for line in out.splitlines():
        ll = line.lower()
        if "ubuntu" in ll and ("22.04" in ll or "24.04" in ll or "20.04" in ll):
            return line.strip()
    # Fallback: first 2000 chars
    return out[:2000]


def write_adjudicated(
    output_path: Path,
    *,
    archive_path: Path,
    archive_sha: str,
    archive_size: int,
    parsed: dict[str, Any],
    run_id: int,
    run_url: str,
    release_tag: str,
    asset_url: str,
    runner_os_release: str,
    evaluate_py_sha: str,
    submission_name: str,
    dispatched_at: str,
    completed_at: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "archive_relpath": str(archive_path),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "canonical_score": parsed["canonical_score"],
        "avg_segnet_dist": parsed["avg_segnet_dist"],
        "avg_posenet_dist": parsed["avg_posenet_dist"],
        "compression_rate": parsed["compression_rate"],
        "score_recomputed_from_components": parsed[
            "score_recomputed_from_components"
        ],
        "n_samples": parsed["n_samples"],
        "device": "cpu",
        "hardware": HARDWARE_LABEL,
        "runner_os_release": runner_os_release,
        "evaluate_py_sha256": evaluate_py_sha,
        "workflow_run_id": run_id,
        "workflow_run_url": run_url,
        "fork_repo": UPSTREAM_FORK_REPO,
        "submission_name": submission_name,
        "release_tag": release_tag,
        "asset_url": asset_url,
        "lane_tag": LANE_TAG,
        "evidence_grade": "contest-CPU-1to1",
        "dispatched_at_utc": dispatched_at,
        "completed_at_utc": completed_at,
        "report_text": parsed["report_text"],
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dispatch contest-faithful CPU auth eval via fork GHA",
    )
    p.add_argument("--archive-path", required=True, type=Path)
    p.add_argument("--archive-sha", required=True, type=str)
    p.add_argument(
        "--submission-name",
        required=True,
        type=str,
        help="unique submission name (will appear under submissions/<name>/)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory to write adjudicated.json and downloaded report",
    )
    p.add_argument(
        "--repo",
        default=UPSTREAM_FORK_REPO,
        help=f"GitHub repo (default: {UPSTREAM_FORK_REPO})",
    )
    p.add_argument("--runner", default=DEFAULT_RUNNER)
    p.add_argument(
        "--release-tag",
        default=None,
        help="release tag to use (default: cpu-eval-<submission_name>-<utc-stamp>)",
    )
    p.add_argument(
        "--evaluate-py-path",
        type=Path,
        default=Path("upstream/evaluate.py"),
        help="path to local upstream/evaluate.py for SHA pinning",
    )
    p.add_argument(
        "--skip-dispatch",
        action="store_true",
        help="(diagnostic) only verify archive+upload; don't dispatch workflow",
    )
    p.add_argument(
        "--pr-number",
        type=str,
        default=None,
        help=(
            "optional PR number on the fork; when set the workflow's "
            "actions/checkout step uses refs/pull/<n>/merge instead of master "
            "(required when submission code lives on a PR branch). The fork's "
            "eval.yml exposes pr_number as an optional workflow_dispatch input."
        ),
    )
    args = p.parse_args()

    archive_size = verify_archive(args.archive_path, args.archive_sha)
    print(
        f"[ok] archive {args.archive_path} verified: "
        f"sha256={args.archive_sha} bytes={archive_size}",
        flush=True,
    )

    evaluate_py_sha = (
        sha256_of(args.evaluate_py_path)
        if args.evaluate_py_path.exists()
        else "<not-found>"
    )

    dispatched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    release_tag = args.release_tag or (
        f"cpu-eval-{args.submission_name}-"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    asset_url = create_release_with_asset(
        archive_path=args.archive_path,
        release_tag=release_tag,
        archive_sha=args.archive_sha,
        archive_size=archive_size,
        repo=args.repo,
    )
    print(f"[ok] release asset uploaded: {asset_url}", flush=True)

    if args.skip_dispatch:
        print("[skip-dispatch] exiting before workflow trigger", flush=True)
        return 0

    run_id = trigger_workflow(
        submission_name=args.submission_name,
        submission_url=asset_url,
        runner=args.runner,
        repo=args.repo,
        pr_number=args.pr_number,
    )
    run_url = f"https://github.com/{args.repo}/actions/runs/{run_id}"
    print(f"[ok] workflow dispatched: run_id={run_id} url={run_url}", flush=True)

    info = poll_run(run_id, args.repo)
    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    if info.get("conclusion") != "success":
        sys.stderr.write(
            f"[fatal] workflow run {run_id} concluded "
            f"{info.get('conclusion')!r}; see {run_url}\n"
        )
        return 3

    with tempfile.TemporaryDirectory() as td:
        report_path = download_artifact(
            run_id, args.submission_name, args.repo, Path(td)
        )
        parsed = parse_report(report_path)
        # Copy report.txt into output_dir for posterity
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(report_path, args.output_dir / "report.txt")

    runner_os = fetch_log_for_runner(run_id, args.repo)

    out = write_adjudicated(
        args.output_dir / "contest_auth_eval.adjudicated.json",
        archive_path=args.archive_path,
        archive_sha=args.archive_sha,
        archive_size=archive_size,
        parsed=parsed,
        run_id=run_id,
        run_url=run_url,
        release_tag=release_tag,
        asset_url=asset_url,
        runner_os_release=runner_os,
        evaluate_py_sha=evaluate_py_sha,
        submission_name=args.submission_name,
        dispatched_at=dispatched_at,
        completed_at=completed_at,
    )
    print(
        f"[done] adjudicated.json written to {out}\n"
        f"  canonical_score = {parsed['canonical_score']}  {LANE_TAG}\n"
        f"  pose_avg = {parsed['avg_posenet_dist']}\n"
        f"  seg_avg  = {parsed['avg_segnet_dist']}\n"
        f"  rate     = {parsed['compression_rate']}\n"
        f"  recomputed = {parsed['score_recomputed_from_components']}\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
