#!/usr/bin/env python3
"""Experiment runner — automated training loop with job queue.

Reads a YAML-like job queue, launches one job at a time on MPS,
monitors progress, auto-restarts on crash, and promotes candidates
that cross the proxy threshold.

Usage:
    .venv/bin/python tools/experiment_runner.py

The runner:
  1. Reads experiments/job_queue.json for the next QUEUED job
  2. Launches it on MPS (one at a time to avoid OOM)
  3. Monitors checkpoint files every 30s
  4. Sends notifications on new best, crash, completion
  5. When a job finishes or crosses the scorer threshold, moves to next
  6. Logs everything to .ralph/run_log.md
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent.parent
VENV_PYTHON = str(REPO / ".venv" / "bin" / "python")
WEIGHTS_DIR = REPO / "experiments" / "postfilter_weights"
QUEUE_PATH = REPO / "experiments" / "job_queue.json"
RUN_LOG = REPO / ".ralph" / "run_log.md"
POLL_INTERVAL = 30  # seconds
PROXY_THRESHOLD = 3.55  # scorer below this → proxy-score candidate
MPS_WATERMARK = "0.0"

# Default queue if none exists
DEFAULT_QUEUE = [
    {
        "tag": "standard_h64_long2500",
        "cmd": "experiments/train_postfilter_qat_ema.py --hidden 64 --epochs 2500 --alpha 20 --tag standard_h64_long2500",
        "platform": "mps",
        "status": "running",
        "priority": 1,
    },
    {
        "tag": "h96_modal",
        "cmd": "modal run experiments/modal_h96_deploy.py",
        "platform": "modal",
        "status": "running",
        "priority": 1,
    },
    {
        "tag": "standard_h128_modal",
        "cmd": "modal run experiments/modal_h96_deploy.py",  # TODO: h128 config
        "platform": "modal",
        "status": "queued",
        "priority": 2,
        "depends_on": "h96_modal",
    },
]


def notify(title: str, body: str):
    subprocess.run(
        ["/usr/bin/osascript", "-e",
         f'display notification "{body}" with title "{title}" sound name "Glass"'],
        capture_output=True,
    )


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def read_queue() -> list[dict]:
    if QUEUE_PATH.exists():
        return json.loads(QUEUE_PATH.read_text())
    QUEUE_PATH.write_text(json.dumps(DEFAULT_QUEUE, indent=2))
    return DEFAULT_QUEUE


def write_queue(queue: list[dict]):
    QUEUE_PATH.write_text(json.dumps(queue, indent=2))


def get_checkpoint(tag: str) -> dict | None:
    path = WEIGHTS_DIR / f"postfilter_{tag}_best_meta.json"
    if path.exists():
        try:
            d = json.loads(path.read_text())
            d["age_s"] = int(time.time() - path.stat().st_mtime)
            return d
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def is_process_alive(tag: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", f"--tag {tag}"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def launch_mps_job(job: dict) -> subprocess.Popen | None:
    cmd = f"{VENV_PYTHON} -u {job['cmd']}"
    log_path = f"/tmp/{job['tag']}.log"
    env = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": MPS_WATERMARK}

    log(f"Launching: {cmd}")
    log(f"Log: {log_path}")

    proc = subprocess.Popen(
        cmd.split(),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(REPO),
    )
    return proc


def append_run_log(entry: str):
    with open(RUN_LOG, "a") as f:
        f.write(f"\n{entry}\n")


def main():
    log("Experiment runner started")
    notify("Experiment Runner", "Automated training loop started")

    queue = read_queue()
    active_proc: subprocess.Popen | None = None
    active_tag: str | None = None
    last_best: dict[str, float] = {}  # tag -> best scorer

    while True:
        queue = read_queue()

        # Find active MPS job
        mps_jobs = [j for j in queue if j["platform"] == "mps"]
        running_mps = [j for j in mps_jobs if j["status"] == "running"]
        queued_mps = [j for j in mps_jobs if j["status"] == "queued"]

        # Check if active process is alive
        if active_proc and active_tag:
            if active_proc.poll() is not None:
                # Process exited
                exit_code = active_proc.returncode
                log(f"Job {active_tag} exited with code {exit_code}")
                notify(f"Job exited: {active_tag}", f"Exit code {exit_code}")

                ckpt = get_checkpoint(active_tag)
                if ckpt:
                    log(f"  Final: ep {ckpt['epoch']}, scorer {ckpt['scorer']:.4f}")

                # Check if it should restart or move to next
                for j in queue:
                    if j["tag"] == active_tag:
                        if exit_code != 0 and ckpt and ckpt.get("epoch", 0) < 100:
                            # Crashed early — restart
                            log(f"  Restarting (crashed at ep {ckpt.get('epoch', 0)})")
                            j["status"] = "running"
                            active_proc = launch_mps_job(j)
                        else:
                            j["status"] = "completed"
                            active_proc = None
                            active_tag = None
                write_queue(queue)

        # Launch next queued job if no active MPS job
        if active_proc is None and queued_mps:
            # Check dependencies
            for j in sorted(queued_mps, key=lambda x: x.get("priority", 99)):
                dep = j.get("depends_on")
                if dep:
                    dep_job = next((q for q in queue if q["tag"] == dep), None)
                    if dep_job and dep_job["status"] != "completed":
                        continue  # dependency not met
                j["status"] = "running"
                active_tag = j["tag"]
                active_proc = launch_mps_job(j)
                write_queue(queue)
                notify(f"Started: {active_tag}", j["cmd"][:60])
                break

        # Monitor all checkpoints
        for j in queue:
            if j["status"] not in ("running", "completed"):
                continue
            ckpt = get_checkpoint(j["tag"])
            if not ckpt:
                continue

            scorer = ckpt.get("scorer", 999)
            prev = last_best.get(j["tag"], 999)

            if scorer < prev - 0.001:
                last_best[j["tag"]] = scorer
                log(f"New best {j['tag']}: ep {ckpt['epoch']}, scorer {scorer:.4f}")
                notify(f"📉 {scorer:.4f}", f"{j['tag']} ep {ckpt['epoch']}")

            # Check proxy threshold
            if scorer < PROXY_THRESHOLD and prev >= PROXY_THRESHOLD:
                log(f"🎯 {j['tag']} crossed proxy threshold! scorer {scorer:.4f}")
                notify(f"🎯 PROXY READY: {scorer:.4f}", f"{j['tag']} — run proxy scorer")
                append_run_log(
                    f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"🎯 {j['tag']} crossed proxy threshold at scorer {scorer:.4f}, ep {ckpt['epoch']}"
                )

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Experiment runner stopped")
