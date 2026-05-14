#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check Modal TTO experiment progress.

Downloads the incremental stdout log and shows latest batch progress.

Usage:
    python scripts/modal_check.py
    python scripts/modal_check.py --tag asym_v5_lagrangian_fixed
    python scripts/modal_check.py --volume tac-asymmetric-results
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check Modal TTO experiment progress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tag", type=str, default="asym_v5_lagrangian_fixed",
                   help="Experiment tag (subdirectory in the volume)")
    p.add_argument("--volume", type=str, default="tac-asymmetric-results",
                   help="Modal volume name")
    p.add_argument("--stdout-log", type=str, default="tto_results/tto_stdout.log",
                   help="Path to stdout log relative to tag dir")
    p.add_argument("--run-log", type=str, default="tto_results/tto_run.log",
                   help="Path to run status log relative to tag dir")
    return p.parse_args()


def modal_bin() -> str:
    """Find the modal CLI binary — checks venv first, then PATH."""
    import shutil
    from pathlib import Path
    venv_modal = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "modal"
    if venv_modal.exists():
        return str(venv_modal)
    found = shutil.which("modal")
    if found:
        return found
    raise FileNotFoundError("modal CLI not found. Install with: uv pip install modal")


def check_running_apps(modal: str) -> list[str]:
    """List running/ephemeral Modal apps.

    The ``modal app list`` table uses Unicode box-drawing characters and
    multi-line rows.  Instead of fragile regex parsing we simply look for
    lines containing ``ephemeral`` or ``running`` (the two states that
    indicate a live app).
    """
    result = subprocess.run(
        [modal, "app", "list"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    running: list[str] = []
    for line in result.stdout.splitlines():
        low = line.lower()
        # Skip header/border rows — look for data rows with active states
        if "ephemeral" in low or "running" in low:
            # Strip box-drawing characters and extra whitespace for display
            clean = line.replace("┃", "|").replace("│", "|").strip()
            if clean and clean not in running:
                running.append(clean)
    return running


def download_log(modal: str, volume: str, remote_path: str) -> str | None:
    """Download a file from a Modal volume and return its contents.

    Uses ``--force`` so that pre-existing local files don't cause an error.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        tmp_path = tmp.name

    result = subprocess.run(
        [modal, "volume", "get", volume, remote_path, tmp_path, "--force"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        Path(tmp_path).unlink(missing_ok=True)
        return None

    content = Path(tmp_path).read_text(errors="replace")
    Path(tmp_path).unlink(missing_ok=True)
    return content


def parse_batch_progress(log_content: str) -> dict:
    """Parse TTO batch progress from log content.

    Looks for lines like:
        [tto] Batch 5/60: ...
        [tto] Batch 5/60 done in 42.3s (0.42s/frame)

    Returns dict with latest_batch, total_batches, completed batches,
    and per-batch timing/pose snapshots.
    """
    batch_pattern = re.compile(
        r"\[tto\] Batch (\d+)/(\d+)"
    )
    done_pattern = re.compile(
        r"\[tto\] Batch (\d+)/(\d+) done in ([\d.]+)s"
    )
    # Match the coupled-4dvar log format: "pose=0.1234" or "best_pose=0.1234"
    # but NOT config lines like "pose_weight=10.0"
    pose_pattern = re.compile(
        r"\[coupled-4dvar\].*pose=([\d.]+(?:e[+-]?\d+)?)",
    )
    resumed_pattern = re.compile(
        r"\[tto\] Batch (\d+)/(\d+): RESUMED"
    )

    latest_batch = 0
    total_batches = 0
    completed = []
    resumed = []
    batch_times: list[float] = []
    pose_snapshots: list[str] = []

    for line in log_content.splitlines():
        m = batch_pattern.search(line)
        if m:
            b, t = int(m.group(1)), int(m.group(2))
            if b > latest_batch:
                latest_batch = b
            total_batches = max(total_batches, t)

        m = done_pattern.search(line)
        if m:
            b = int(m.group(1))
            dt = float(m.group(3))
            completed.append(b)
            batch_times.append(dt)

        m = resumed_pattern.search(line)
        if m:
            resumed.append(int(m.group(1)))

        m = pose_pattern.search(line)
        if m:
            pose_snapshots.append(line.strip())

    return {
        "latest_batch": latest_batch,
        "total_batches": total_batches,
        "completed": completed,
        "resumed": resumed,
        "batch_times": batch_times,
        "pose_snapshots": pose_snapshots[-5:],  # last 5
    }


def main() -> int:
    args = parse_args()

    try:
        modal = modal_bin()
    except FileNotFoundError as e:
        print(f"{RED}{e}{RESET}")
        return 1

    print(f"\n{BOLD}Modal TTO Progress Check{RESET}")
    print(f"{'=' * 60}")
    print(f"  Volume: {args.volume}")
    print(f"  Tag:    {args.tag}\n")

    # Check running apps
    running = check_running_apps(modal)
    if running:
        print(f"  {GREEN}{BOLD}Running Modal apps:{RESET}")
        for app in running:
            print(f"    {app}")
    else:
        print(f"  {YELLOW}No running Modal apps detected.{RESET}")
    print()

    # Download and parse run status log
    run_log_path = f"{args.tag}/{args.run_log}"
    run_content = download_log(modal, args.volume, run_log_path)
    if run_content is not None:
        last_line = run_content.strip().splitlines()[-1] if run_content.strip() else "(empty)"
        if "ok" in last_line.lower() or "complete" in last_line.lower():
            status_color = GREEN
        elif "fail" in last_line.lower() or "error" in last_line.lower():
            status_color = RED
        elif "running" in last_line.lower():
            status_color = BLUE
        else:
            status_color = YELLOW
        print(f"  Run status ({args.run_log}): {status_color}{BOLD}{last_line}{RESET}")
    else:
        print(f"  {YELLOW}Run log not found: {run_log_path}{RESET}")
    print()

    # Download and parse stdout log
    stdout_path = f"{args.tag}/{args.stdout_log}"
    stdout_content = download_log(modal, args.volume, stdout_path)
    if stdout_content is None:
        print(f"  {YELLOW}Stdout log not found: {stdout_path}{RESET}")
        print(f"  Try: modal volume ls {args.volume} {args.tag}/")
        return 1

    progress = parse_batch_progress(stdout_content)

    if progress["total_batches"] == 0:
        print(f"  {YELLOW}No batch progress lines found in log.{RESET}")
        # Show last 5 lines of log as fallback
        tail = stdout_content.strip().splitlines()[-5:]
        for line in tail:
            print(f"    {line}")
        return 0

    n_computed = len(progress["completed"])
    n_resumed = len(progress["resumed"])
    n_done = n_computed + n_resumed  # both computed and resumed batches are done
    total = progress["total_batches"]
    pct = 100.0 * n_done / total if total > 0 else 0.0

    print(f"  {BOLD}Batch progress:{RESET} {n_done}/{total} completed ({pct:.0f}%)")
    if n_resumed:
        print(f"  {BLUE}  ({n_computed} computed + {n_resumed} resumed from checkpoint){RESET}")
    print(f"  Latest batch: {progress['latest_batch']}/{total}")

    # ETA calculation (only newly computed batches count for timing)
    if progress["batch_times"]:
        avg_time = sum(progress["batch_times"]) / len(progress["batch_times"])
        remaining = total - n_done
        eta_s = remaining * avg_time
        eta_min = eta_s / 60.0
        eta_hr = eta_min / 60.0
        print(f"\n  {BOLD}Timing:{RESET}")
        print(f"    Avg batch time: {avg_time:.1f}s")
        if eta_hr > 1:
            print(f"    ETA: {eta_hr:.1f}h ({remaining} batches remaining)")
        else:
            print(f"    ETA: {eta_min:.1f}min ({remaining} batches remaining)")

    # PoseNet snapshots
    if progress["pose_snapshots"]:
        print(f"\n  {BOLD}Recent PoseNet snapshots:{RESET}")
        for snap in progress["pose_snapshots"]:
            print(f"    {snap}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
