#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""tac monitor — live training dashboard from telemetry JSONL files.

Reads telemetry_*.jsonl files from experiment output directories and
displays a live-updating table of all running experiments.

Usage:
    .venv/bin/python -m tac.experiments.monitor
    .venv/bin/python -m tac.experiments.monitor --dir /path/to/weights
    .venv/bin/python -m tac.experiments.monitor --logs
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def read_telemetry(path: Path) -> dict | None:
    """Read the last entry from a telemetry JSONL file."""
    try:
        lines = path.read_text().strip().split("\n")
        if not lines or not lines[-1].strip():
            return None
        return json.loads(lines[-1])
    except (json.JSONDecodeError, FileNotFoundError, IndexError):
        return None


def read_log_tail(path: Path) -> dict | None:
    """Parse the last epoch line from a training log file."""
    try:
        lines = path.read_text().strip().split("\n")
        for line in reversed(lines):
            if "[ep " in line and "scorer=" in line:
                parts = {}
                # Parse: [ep  103] loss=4.60 pose=0.090 seg=0.127 scorer=1.43 best=1.43 lr=0.0005
                for token in line.split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        try:
                            parts[k] = float(v)
                        except ValueError:
                            parts[k] = v
                # Extract epoch
                if "[ep" in line:
                    ep_str = line.split("]")[0].split("[ep")[1].strip().rstrip("]*")
                    parts["epoch"] = int(ep_str)
                return parts
        return None
    except (FileNotFoundError, IndexError):
        return None


def discover_experiments(telemetry_dir: Path, log_dir: Path | None = None) -> list[dict]:
    """Find all experiments from telemetry files and log files."""
    experiments = []

    # Telemetry JSONL files
    for f in sorted(telemetry_dir.rglob("telemetry_*.jsonl")):
        entry = read_telemetry(f)
        if entry:
            tag = f.stem.replace("telemetry_", "")
            entry["_tag"] = tag
            entry["_source"] = "telemetry"
            experiments.append(entry)

    # Log files (fallback for experiments without telemetry)
    if log_dir:
        for f in sorted(log_dir.glob("exp_*.log")):
            entry = read_log_tail(f)
            if entry:
                tag = f.stem.replace("exp_", "")
                # Don't duplicate if we already have telemetry
                if not any(e["_tag"] == tag for e in experiments):
                    entry["_tag"] = tag
                    entry["_source"] = "log"
                    experiments.append(entry)

    return experiments


def format_table(experiments: list[dict]) -> str:
    """Format experiments as a readable table."""
    if not experiments:
        return "No experiments found."

    header = f"{'Tag':<30} {'Ep':>5} {'Scorer':>8} {'Pose':>10} {'Seg':>10} {'LR':>10} {'Conf':>5} {'Src':>4}"
    sep = "-" * len(header)
    rows = [header, sep]

    # Sort by scorer (best first)
    experiments.sort(key=lambda e: e.get("scorer", e.get("best_scorer", 999)))

    for e in experiments:
        tag = e.get("_tag", "?")[:30]
        ep = e.get("epoch", "?")
        scorer = e.get("scorer", e.get("best_scorer", "?"))
        pose = e.get("eval_pose", e.get("pose", "?"))
        seg = e.get("eval_seg", e.get("seg", "?"))
        lr = e.get("lr", "?")
        conf = e.get("proxy_confidence", "")
        src = e.get("_source", "?")[:4]

        scorer_str = f"{scorer:.4f}" if isinstance(scorer, float) else str(scorer)
        pose_str = f"{pose:.6f}" if isinstance(pose, float) else str(pose)
        seg_str = f"{seg:.6f}" if isinstance(seg, float) else str(seg)
        lr_str = f"{lr:.6f}" if isinstance(lr, float) else str(lr)
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else ""

        rows.append(f"{tag:<30} {ep:>5} {scorer_str:>8} {pose_str:>10} {seg_str:>10} {lr_str:>10} {conf_str:>5} {src:>4}")

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="tac monitor — live training dashboard")
    parser.add_argument("--dir", default="experiments/postfilter_weights",
                        help="Directory containing telemetry_*.jsonl files")
    parser.add_argument("--logs", action="store_true",
                        help="Also read /tmp/exp_*.log files for experiments without telemetry")
    parser.add_argument("--watch", type=int, default=0,
                        help="Refresh every N seconds (0 = print once and exit)")
    args = parser.parse_args()

    telemetry_dir = Path(args.dir)
    log_dir = Path("/tmp") if args.logs else None

    while True:
        os.system("clear" if os.name != "nt" else "cls") if args.watch else None
        print(f"=== tac monitor — {time.strftime('%H:%M:%S')} ===")
        print(f"Telemetry: {telemetry_dir}")
        if log_dir:
            print(f"Logs: {log_dir}/exp_*.log")
        print()

        experiments = discover_experiments(telemetry_dir, log_dir)
        print(format_table(experiments))
        print()
        print(f"{len(experiments)} experiment(s) found")

        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
