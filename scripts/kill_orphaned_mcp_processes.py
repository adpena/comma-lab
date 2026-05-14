#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Terminate orphaned MCP helper processes for contest/eval work.

This is intentionally narrow. It targets only known MCP helper command tokens
that this project has disabled, and it skips its own process. Use --dry-run to
inspect matches.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass


MCP_PROCESS_TOKENS = (
    "chrome-devtools-mcp",
    "model.context",
    "rbx-studio-mcp",
    "roblox_studio_mcp",
)
_SHELL_BASENAMES = {"bash", "dash", "sh", "zsh"}
_INSPECTION_BASENAMES = {
    "awk",
    "egrep",
    "fgrep",
    "find",
    "grep",
    "head",
    "ps",
    "rg",
    "sed",
    "tail",
    "xargs",
}
_PACKAGE_LAUNCHER_BASENAMES = {
    "bun",
    "npx",
    "pnpm",
    "uvx",
    "yarn",
}
_PYTHON_BASENAME_PREFIXES = ("python",)


@dataclass(frozen=True)
class ProcessMatch:
    pid: int
    command: str
    token: str


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _basename(arg: str) -> str:
    return os.path.basename(arg.rstrip("/"))


def _arg_matches_process_token(arg: str, token: str) -> bool:
    base = _basename(arg)
    if base == token or base.startswith(f"{token}@"):
        return True
    parts = [part for part in arg.replace("\\", "/").split("/") if part]
    return any(part == token or part.startswith(f"{token}@") for part in parts)


def _classify_mcp_helper_command(command: str, *, shell_depth: int = 0) -> str | None:
    argv = _split_command(command)
    if not argv:
        return None
    base = _basename(argv[0])

    if base in _SHELL_BASENAMES and shell_depth < 2 and "-c" in argv:
        index = argv.index("-c")
        if index + 1 < len(argv):
            return _classify_mcp_helper_command(argv[index + 1], shell_depth=shell_depth + 1)
        return None

    if base in {"command", "exec"} and len(argv) > 1:
        return _classify_mcp_helper_command(" ".join(shlex.quote(part) for part in argv[1:]), shell_depth=shell_depth)

    if base in _INSPECTION_BASENAMES:
        return None

    for token in MCP_PROCESS_TOKENS:
        if _arg_matches_process_token(argv[0], token):
            return token

    if base == "npm":
        launch_indices = [i for i, arg in enumerate(argv[1:], start=1) if arg in {"exec", "x"}]
        search_from = (launch_indices[0] + 1) if launch_indices else len(argv)
        for arg in argv[search_from:]:
            for token in MCP_PROCESS_TOKENS:
                if _arg_matches_process_token(arg, token):
                    return token
        return None

    if base in _PACKAGE_LAUNCHER_BASENAMES:
        for arg in argv[1:]:
            if arg.startswith("-"):
                continue
            for token in MCP_PROCESS_TOKENS:
                if _arg_matches_process_token(arg, token):
                    return token
        return None

    if base.startswith(_PYTHON_BASENAME_PREFIXES):
        for index, arg in enumerate(argv[:-1]):
            if arg == "-m":
                module = argv[index + 1]
                for token in MCP_PROCESS_TOKENS:
                    if module == token:
                        return token
        return None

    for arg in argv[1:]:
        for token in MCP_PROCESS_TOKENS:
            if _arg_matches_process_token(arg, token):
                return token
    return None


def parse_ps_rows(rows: list[str], *, current_pid: int | None = None) -> list[ProcessMatch]:
    matches: list[ProcessMatch] = []
    for raw in rows:
        line = raw.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if current_pid is not None and pid == current_pid:
            continue
        token = _classify_mcp_helper_command(command)
        if token is not None:
            matches.append(ProcessMatch(pid=pid, command=command, token=token))
    return matches


def ps_rows() -> list[str]:
    proc = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ps failed")
    return proc.stdout.splitlines()


def remaining_pids(pids: set[int]) -> set[int]:
    if not pids:
        return set()
    alive: set[int] = set()
    for match in parse_ps_rows(ps_rows(), current_pid=os.getpid()):
        if match.pid in pids:
            alive.add(match.pid)
    return alive


def terminate(matches: list[ProcessMatch], *, sig: int, dry_run: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for match in matches:
        record: dict[str, object] = asdict(match)
        record["signal"] = sig
        record["dry_run"] = dry_run
        if dry_run:
            record["status"] = "matched"
        else:
            try:
                os.kill(match.pid, sig)
            except ProcessLookupError:
                record["status"] = "already_exited"
            except PermissionError as exc:
                record["status"] = "permission_denied"
                record["error"] = str(exc)
            else:
                record["status"] = "signaled"
        records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print matches without signaling them.")
    parser.add_argument(
        "--signal",
        choices=("TERM", "KILL"),
        default="TERM",
        help="Signal to send to matched MCP helper processes.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait before checking for remaining matched processes.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if matched processes remain after signaling.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sig = signal.SIGTERM if args.signal == "TERM" else signal.SIGKILL
    matches = parse_ps_rows(ps_rows(), current_pid=os.getpid())
    records = terminate(matches, sig=sig, dry_run=args.dry_run)
    if not args.dry_run and matches and args.wait_seconds > 0:
        time.sleep(args.wait_seconds)
    remaining = remaining_pids({match.pid for match in matches}) if not args.dry_run else set()
    payload = {
        "schema_version": 1,
        "tool": "scripts/kill_orphaned_mcp_processes.py",
        "matched_count": len(matches),
        "remaining_count": len(remaining),
        "remaining_pids": sorted(remaining),
        "records": records,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if not records:
            print("MCP_CLEANUP_OK no orphaned MCP helper processes matched")
        for record in records:
            print(
                f"MCP_CLEANUP {record['status']} pid={record['pid']} "
                f"token={record['token']} command={record['command']}"
            )
        if remaining:
            print(f"MCP_CLEANUP_REMAINING pids={sorted(remaining)}", file=sys.stderr)
        elif records and not args.dry_run:
            print("MCP_CLEANUP_OK no matched MCP helper processes remain")
    return 2 if args.strict and remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
