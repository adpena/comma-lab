#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Double-fork daemon launcher for the task-#74 distill-student sweep (durable; survives the parent
shell session). macOS has no ``setsid`` binary, so this uses the os.fork()/os.setsid() double-fork
idiom + subprocess.Popen so the training process is fully detached and keeps running across the
agent's bash-session boundary (the SIGURG-144 / session-end trap per CLAUDE.md daemon discipline).

Usage:
  python tools/_launch_distill_daemon.py <size> <n_pairs> <epochs> <lr> <run_dir> <log_path>
"""
import os
import subprocess
import sys

REPO = "/Users/adpena/Projects/pact"


def main() -> int:
    size, n_pairs, epochs, lr, run_dir, log_path = sys.argv[1:7]
    if os.fork() != 0:
        os._exit(0)
    os.setsid()
    if os.fork() != 0:
        os._exit(0)
    with open(log_path, "w") as f:
        p = subprocess.Popen(
            [".venv/bin/python", "tools/distill_smaller_student_from_frontier_teacher.py",
             "--size", size, "--n-pairs", n_pairs, "--epochs", epochs, "--eval-every", "40",
             "--lr", lr, "--out-dir", run_dir],
            stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=REPO,
        )
        f.write(f"PID={p.pid} size={size} run_dir={run_dir}\n")
        f.flush()
        p.wait()
        f.write(f"DONE_RC={p.returncode}\n")
        f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
