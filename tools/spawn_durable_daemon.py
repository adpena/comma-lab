#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Spawn a command as a TRULY detached daemon that survives the parent process
(e.g. a Claude Code process exit / context-compaction / harness restart).

WHY: nohup+disown is NOT enough on macOS — the child stays in the launcher's
process GROUP and dies when that group is torn down (which is what killed the
basin + dashboard + codex together when the Claude process exited). This uses
``subprocess.Popen(start_new_session=True)`` so the child calls ``setsid()`` and
gets its OWN session + process group, fully detached from the launcher. A signal
sent to the launcher's process group can no longer reach it.

Usage:
    .venv/bin/python tools/spawn_durable_daemon.py --log <path> -- <cmd> [args...]
Prints the daemon PID and exits 0 immediately. stdin=/dev/null, stdout+stderr->log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="append-only log for the daemon's stdout+stderr")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command> [args...]")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("error: no command (use: --log L -- cmd args)", file=sys.stderr)
        return 2

    log = open(a.log, "ab", buffering=0)  # noqa: SIM115 — kept open for the child
    devnull = open(os.devnull, "rb")  # noqa: SIM115
    log.write(f"[durable-daemon] launching: {' '.join(cmd)}\n".encode())
    proc = subprocess.Popen(
        cmd,
        stdin=devnull,
        stdout=log,
        stderr=log,
        start_new_session=True,  # setsid() in the child -> new session/pgroup -> survives parent
        close_fds=True,
        cwd=os.getcwd(),
    )
    print(f"[durable-daemon] pid={proc.pid} (detached session) log={a.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
