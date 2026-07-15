#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evidence-grade local process-tree liveness check.

Never infer death from log silence, a wrapper-only PID check, or a grep
pipeline.  This tool samples the full process table once, walks every PPID
descendant of the custody root, and optionally searches a command token when
the wrapper PID has already disappeared.  A token-only match is explicitly
reported as lacking root custody; absence is ``NOT_PRESENT_AT_SAMPLE``, not a
claim about why or when the process exited.

This is the class fix for
``false_dead_diagnosis_incomplete_process_tree_walk``: the owed16v2 trainer
was alive as a Python child at 99% CPU while a wrapper-only/grep diagnosis
declared it dead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.memory_guard import ProcessSample, descendant_pids, sample_processes  # noqa: E402


def build_liveness_report(
    samples: Mapping[int, ProcessSample],
    *,
    root_pid: int,
    command_token: str | None = None,
) -> dict:
    """Build a deterministic verdict from one process-table snapshot."""
    root_present = root_pid in samples
    tree_ids = descendant_pids(samples, root_pid) if root_present else set()
    token = (command_token or "").strip()
    token_ids = {
        pid for pid, sample in samples.items() if token and token in sample.command
    }

    if tree_ids:
        status = "TREE_PRESENT"
        authority = "ROOT_CUSTODIED_PROCESS_TREE"
    elif token_ids:
        status = "TOKEN_MATCH_WITHOUT_ROOT"
        authority = "TOKEN_FALLBACK_NOT_ROOT_CUSTODY"
    else:
        status = "NOT_PRESENT_AT_SAMPLE"
        authority = "SINGLE_PROCESS_TABLE_SAMPLE"

    observed_ids = sorted(tree_ids | token_ids)
    rows = [
        {
            "pid": pid,
            "ppid": samples[pid].ppid,
            "pgid": samples[pid].pgid,
            "rss_kb": samples[pid].rss_kb,
            "command": samples[pid].command,
            "in_root_tree": pid in tree_ids,
            "matched_token": pid in token_ids,
        }
        for pid in observed_ids
    ]
    return {
        "schema": "process_tree_liveness.v1",
        "root_pid": root_pid,
        "root_present": root_present,
        "command_token": token or None,
        "status": status,
        "authority": authority,
        "alive": bool(tree_ids or token_ids),
        "tree_pids": sorted(tree_ids),
        "token_match_pids": sorted(token_ids),
        "processes": rows,
        "verdict_scope": "one_local_process_table_sample",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", required=True, type=int)
    parser.add_argument(
        "--command-token",
        help="Fallback literal command substring when the wrapper PID may be gone.",
    )
    parser.add_argument("--output", type=Path, help="Optional durable JSON receipt path.")
    args = parser.parse_args(argv)
    if args.root_pid <= 0:
        parser.error("--root-pid must be positive")

    report = build_liveness_report(
        sample_processes(),
        root_pid=args.root_pid,
        command_token=args.command_token,
    )
    text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, args.output)
    sys.stdout.write(text)
    return 0 if report["alive"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
