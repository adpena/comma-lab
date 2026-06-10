#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sequential task-#74 distill-student SIZE SWEEP as a durable double-fork daemon.

Runs the trainer across the size ladder {40,60,80,100,120}kb sequentially (CPU-bound; sequential
avoids the N-way CPU contention that parallel daemons cause), writing each run's train_result.json to
its own dir + a rolling sweep_manifest.json. Fully detached (os.setsid double-fork) so it survives the
agent's bash-session boundary.

Usage:
  python tools/_run_distill_sweep_daemon.py <n_pairs> <epochs> <lr> <sweep_root> <log_path>
"""
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

REPO = "/Users/adpena/Projects/pact"
SIZES = ("40kb", "60kb", "80kb", "100kb", "120kb")


def _utc():
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    n_pairs, epochs, lr, sweep_root, log_path = sys.argv[1:6]
    if os.fork() != 0:
        os._exit(0)
    os.setsid()
    if os.fork() != 0:
        os._exit(0)

    os.makedirs(sweep_root, exist_ok=True)
    manifest_path = os.path.join(sweep_root, "sweep_manifest.json")
    manifest = {"started_utc": _utc(), "n_pairs": int(n_pairs), "epochs": int(epochs),
                "lr": float(lr), "runs": {}}

    with open(log_path, "w") as f:
        f.write(f"SWEEP_PID={os.getpid()} sizes={SIZES} n_pairs={n_pairs} epochs={epochs}\n")
        f.flush()
        for size in SIZES:
            run_dir = os.path.join(sweep_root, f"{size}_{_utc()}")
            f.write(f"[{_utc()}] START {size} -> {run_dir}\n")
            f.flush()
            t0 = time.time()
            rc = subprocess.run(
                [".venv/bin/python", "tools/distill_smaller_student_from_frontier_teacher.py",
                 "--size", size, "--n-pairs", n_pairs, "--epochs", epochs, "--eval-every", "40",
                 "--lr", lr, "--out-dir", run_dir],
                stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=REPO,
            ).returncode
            result_path = os.path.join(run_dir, "train_result.json")
            entry = {"run_dir": run_dir, "rc": rc, "wall_s": round(time.time() - t0, 1),
                     "result_path": result_path if os.path.exists(result_path) else None}
            if os.path.exists(result_path):
                with open(result_path) as rf:
                    r = json.load(rf)
                entry.update({
                    "total_bytes": r["byte_account"]["total_bytes"],
                    "exact_mean_d_seg": r["exact_mean_d_seg"],
                    "exact_mean_d_pose": r["exact_mean_d_pose"],
                    "advisory_score_student_only": r["advisory_score_student_only"],
                    "parity_pass": r["portability_parity"]["parity_pass"],
                    "constant_control": r.get("constant_frame_control"),
                })
            manifest["runs"][size] = entry
            manifest["updated_utc"] = _utc()
            with open(manifest_path, "w") as mf:
                json.dump(manifest, mf, indent=2)
            f.write(f"[{_utc()}] DONE {size} rc={rc} "
                    f"bytes={entry.get('total_bytes')} d_seg={entry.get('exact_mean_d_seg')} "
                    f"d_pose={entry.get('exact_mean_d_pose')} S={entry.get('advisory_score_student_only')}\n")
            f.flush()
        manifest["finished_utc"] = _utc()
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)
        f.write(f"SWEEP_DONE_RC=0 manifest={manifest_path}\n")
        f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
