#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Launch a long local command as a real detached session with provenance.

This is intentionally small: macOS shells launched from agent/tool contexts can
clean up background children even when ``nohup`` is used.  ``start_new_session``
hands the child to PID 1, records the argv, and leaves a deterministic run
directory that can be polled or harvested later.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "detached_local_process_launch.v1"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a command detached from the current shell/process group."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Durable run directory for manifest, pid, stdout/stderr log, and exit placeholder.",
    )
    parser.add_argument(
        "--cwd",
        default=".",
        type=Path,
        help="Working directory for the child command.",
    )
    parser.add_argument(
        "--purpose",
        default="detached local long run",
        help="Human-readable reason stored in launch_manifest.json.",
    )
    parser.add_argument(
        "--authority",
        default="local detached execution; downstream artifacts decide authority",
        help="Authority/provenance note stored in launch_manifest.json.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override for the child. May be repeated.",
    )
    parser.add_argument(
        "--done-receipt",
        default=None,
        metavar="NAME",
        help=(
            "Write .omx/tmp/codex_runs/<NAME>.done (rc=<rc> elapsed=<s>) when the "
            "child exits — the fleet watcher (tools/codex_arm_watch.py) turns it "
            "into a MAIN notification, same channel as codex-arm completions."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifest without launching the child process.",
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("command argv is required after --")
    for item in args.env:
        if "=" not in item:
            parser.error(f"--env must be KEY=VALUE, got {item!r}")
    return args


def main() -> int:
    args = parse_args()
    out = args.output_dir.expanduser().resolve(strict=False)
    cwd = args.cwd.expanduser().resolve(strict=False)
    # Fail fast on unlaunchable argv: a detached child dies silently in run.log otherwise
    # (2026-07-20 launch_003: script existed only on an unmerged arm branch, not at cwd).
    # Absolute executables must exist; a relative script path in argv must exist at cwd.
    exe = args.cmd[0]
    if "/" in exe and not Path(exe).expanduser().exists():
        print(json.dumps({"error": f"executable not found: {exe}"}), file=sys.stderr)
        return 2
    for part in args.cmd[1:]:
        if part.endswith(".py"):
            cand = Path(part) if Path(part).is_absolute() else (cwd / part)
            if not cand.exists():
                print(json.dumps({"error": f"script not found at cwd: {cand.as_posix()}"}),
                      file=sys.stderr)
                return 2
            break  # only the first script arg is the entry point
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(dict(item.split("=", 1) for item in args.env))
    log_path = out / "run.log"
    pid_path = out / "run.pid"
    manifest_path = out / "launch_manifest.json"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": _utc_now(),
        "cwd": cwd.as_posix(),
        "git_sha": _git_sha(cwd),
        "purpose": str(args.purpose),
        "authority": str(args.authority),
        "detach_method": "subprocess.Popen(start_new_session=True)",
        "output_dir": out.as_posix(),
        "pid_path": pid_path.as_posix(),
        "log_path": log_path.as_posix(),
        "argv": [str(part) for part in args.cmd],
        "dry_run": bool(args.dry_run),
    }
    launch_argv = [str(part) for part in args.cmd]
    if args.done_receipt:
        # Wrap in a detached supervisor that writes the watcher-visible exit
        # receipt (same format as codex-arm keepers). The fleet watcher
        # (tools/codex_arm_watch.py) turns the receipt into a MAIN
        # notification — no polling, ever (operator permanent-fix 2026-08-04).
        runs_dir = Path.cwd() / ".omx" / "tmp" / "codex_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        done_path = runs_dir / f"{args.done_receipt}.done"
        supervisor_src = (
            "import signal,subprocess,sys,time,pathlib\n"
            "for _s in ('SIGURG','SIGPIPE'):\n"
            "    try: signal.signal(getattr(signal,_s), signal.SIG_IGN)\n"
            "    except Exception: pass\n"
            "done=pathlib.Path(sys.argv[1]); argv=sys.argv[2:]\n"
            "t0=time.time(); rc=subprocess.call(argv)\n"
            "tmp=done.with_suffix('.done.tmp')\n"
            "tmp.write_text('rc=%d elapsed=%d detached-job\\n' % (rc, int(time.time()-t0)))\n"
            "tmp.replace(done)\n"
            "sys.exit(rc)\n"
        )
        launch_argv = [sys.executable, "-c", supervisor_src, str(done_path), *launch_argv]
        payload["done_receipt_path"] = done_path.as_posix()
    if args.dry_run:
        _write_json(manifest_path, payload)
        print(json.dumps({"dry_run": True, "manifest_path": manifest_path.as_posix()}))
        return 0
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(
            launch_argv,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    payload["pid"] = int(proc.pid)
    _write_json(manifest_path, payload)
    pid_path.write_text(f"{proc.pid}\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "pid": int(proc.pid),
                "output_dir": out.as_posix(),
                "manifest_path": manifest_path.as_posix(),
                "log_path": log_path.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
