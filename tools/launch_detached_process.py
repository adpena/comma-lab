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
    if args.dry_run:
        _write_json(manifest_path, payload)
        print(json.dumps({"dry_run": True, "manifest_path": manifest_path.as_posix()}))
        return 0
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(
            [str(part) for part in args.cmd],
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
