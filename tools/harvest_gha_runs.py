#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Poll one or more GHA workflow runs to completion and harvest the result.

Used to recover from a disowned-subprocess case where the dispatcher's poll
loop exited before harvest. Re-creates the same adjudicated JSON files that
``tools/dispatch_cpu_eval_via_github_actions.py`` would have written.

Usage:
  python tools/harvest_gha_runs.py \\
    --run-id 25557780130 \\
    --archive-path .../b080/archive.zip \\
    --archive-sha abeb4fc7... \\
    --release-tag pr107-stack-b080-cpu-eval-20260508T132345Z \\
    --asset-url https://github.com/.../archive.zip \\
    --submission-name apogee \\
    --output-dir experiments/results/pr107_apogee_stack_b080_cpu_eval_gha_20260508
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_DEFAULT = "adpena/comma_video_compression_challenge"
POLL_INTERVAL_SEC = 30
POLL_TIMEOUT_SEC = 60 * 45


def run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", *args], check=False, capture_output=True, text=True)


def poll_run(run_id: int, repo: str) -> dict[str, Any]:
    started = time.monotonic()
    last_status = ""
    while True:
        if time.monotonic() - started > POLL_TIMEOUT_SEC:
            sys.stderr.write(f"[fatal] poll timeout on run {run_id}\n")
            sys.exit(3)
        q = run_gh(
            [
                "run",
                "view",
                str(run_id),
                "-R",
                repo,
                "--json",
                "status,conclusion,jobs",
            ]
        )
        if q.returncode != 0:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        info = json.loads(q.stdout)
        status = info.get("status", "")
        if status == "completed":
            return info
        # Show the in-progress step
        try:
            jobs = info.get("jobs", [])
            if jobs:
                job = jobs[0]
                steps = job.get("steps", [])
                cur_step = next(
                    (s for s in steps if s.get("status") == "in_progress"),
                    None,
                )
                if cur_step:
                    msg = cur_step.get("name", "?")
                    if msg != last_status:
                        elapsed = int(time.monotonic() - started)
                        print(f"[poll +{elapsed:>3}s] run {run_id} step: {msg}", flush=True)
                        last_status = msg
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_SEC)


def download_artifact(run_id: int, submission_name: str, repo: str, dest_dir: Path) -> Path:
    artifact_name = f"eval-{submission_name}"
    res = run_gh(
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
    if res.returncode != 0:
        sys.stderr.write(f"[fatal] artifact download failed: {res.stderr}\n")
        sys.exit(3)
    report_path = dest_dir / "report.txt"
    if not report_path.exists():
        sys.stderr.write("[fatal] artifact missing report.txt\n")
        sys.exit(3)
    return report_path


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text()
    patterns = {
        "avg_posenet_dist": r"Average PoseNet Distortion:\s*([0-9.eE+,-]+)",
        "avg_segnet_dist": r"Average SegNet Distortion:\s*([0-9.eE+,-]+)",
        "compression_rate": r"Compression Rate:\s*([0-9.eE+,-]+)",
        "reported_score_display": r"Final score:.*=\s*([0-9.eE+,-]+)",
        "n_samples": r"Evaluation results over (\d+) samples",
    }
    parsed: dict[str, Any] = {"report_text": text}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            sys.stderr.write(f"[fatal] could not parse {key} from report.txt:\n{text}\n")
            sys.exit(3)
        raw = match.group(1).replace(",", "")
        parsed[key] = int(raw) if key == "n_samples" else float(raw)

    recomputed = (
        100.0 * parsed["avg_segnet_dist"]
        + math.sqrt(10.0 * parsed["avg_posenet_dist"])
        + 25.0 * parsed["compression_rate"]
    )
    parsed["canonical_score"] = recomputed
    parsed["canonical_score_recomputed"] = recomputed
    parsed["score_recomputed_from_components"] = recomputed
    return parsed


def fetch_log_for_runner(run_id: int, repo: str) -> str:
    """Fetch the runner OS info."""
    res = run_gh(["run", "view", str(run_id), "-R", repo, "--log"])
    if res.returncode != 0:
        return ""
    for line in res.stdout.splitlines():
        if "Image: ubuntu-" in line or "Operating System" in line:
            return line.strip()
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, type=int)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--archive-path", required=True, type=Path)
    ap.add_argument("--archive-sha", required=True)
    ap.add_argument("--release-tag", required=True)
    ap.add_argument("--asset-url", required=True)
    ap.add_argument("--submission-name", required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[harvest] polling run {args.run_id} ...", flush=True)
    info = poll_run(args.run_id, args.repo)
    conclusion = info.get("conclusion")
    if conclusion != "success":
        sys.stderr.write(f"[fatal] run {args.run_id} concluded {conclusion!r}\n")
        sys.exit(3)
    print(f"[harvest] run {args.run_id} completed conclusion=success", flush=True)

    with tempfile.TemporaryDirectory() as td:
        report_path = download_artifact(args.run_id, args.submission_name, args.repo, Path(td))
        parsed = parse_report(report_path)
        shutil.copy(report_path, args.output_dir / "report.txt")

    runner_os = fetch_log_for_runner(args.run_id, args.repo)

    # Compute archive SHA + size
    archive_size = args.archive_path.stat().st_size
    import hashlib
    h = hashlib.sha256()
    with args.archive_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != args.archive_sha:
        sys.stderr.write(f"[fatal] archive SHA drift: expected {args.archive_sha} got {actual_sha}\n")
        sys.exit(2)

    # Verify archive member SHA
    import zipfile
    with zipfile.ZipFile(args.archive_path) as zf:
        member_name = zf.namelist()[0]
        member_bytes = zf.read(member_name)
    member_sha = hashlib.sha256(member_bytes).hexdigest()

    out_path = args.output_dir / "contest_auth_eval.adjudicated.json"
    out = {
        "archive_relpath": str(args.archive_path),
        "archive_size_bytes": archive_size,
        "archive_sha256": args.archive_sha,
        "archive_member": member_name,
        "archive_member_sha256": member_sha,
        "canonical_score": parsed["canonical_score"],
        "canonical_score_recomputed": parsed["canonical_score_recomputed"],
        "score_recomputed_from_components": parsed["score_recomputed_from_components"],
        "reported_score_display": parsed["reported_score_display"],
        "avg_segnet_dist": parsed["avg_segnet_dist"],
        "avg_posenet_dist": parsed["avg_posenet_dist"],
        "compression_rate": parsed["compression_rate"],
        "n_samples": parsed["n_samples"],
        "device": "cpu",
        "hardware": "github-actions-ubuntu-latest-x86_64",
        "runner_os_release": runner_os,
        "workflow_run_id": args.run_id,
        "workflow_run_url": f"https://github.com/{args.repo}/actions/runs/{args.run_id}",
        "release_tag": args.release_tag,
        "asset_url": args.asset_url,
        "submission_name": args.submission_name,
        "lane_tag": "[contest-CPU]",
        "evidence_grade": "contest-CPU-1to1",
        "is_contest_compliant": True,
        "report_text": (args.output_dir / "report.txt").read_text(),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[harvest] wrote {out_path}")
    print(f"[harvest] score: {out['canonical_score']}")
    print(f"  pose_avg: {out['avg_posenet_dist']}")
    print(f"  seg_avg:  {out['avg_segnet_dist']}")
    print(f"  rate:     {out['compression_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
