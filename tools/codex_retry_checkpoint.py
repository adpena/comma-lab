#!/usr/bin/env python3
"""Fail closed unless a delegated Codex arm has a resumable checkpoint.

Transient retries must continue from durable custody, never replay a long arm's
original invocation from zero. The delegation key is exact so one arm cannot
borrow an unrelated checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROGRESS = REPO / ".omx" / "state" / "subagent_progress.jsonl"
REFUSED_RC = 20


def latest_resumable_checkpoint(path: Path, delegation_key: str) -> dict | None:
    """Return the newest valid in-progress checkpoint for ``delegation_key``."""
    if not path.is_file():
        return None
    latest: dict | None = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("parent_id_or_session") != delegation_key:
            continue
        step = row.get("step")
        if (
            row.get("status") != "in_progress"
            or not isinstance(step, int)
            or isinstance(step, bool)
            or step < 1
            or not str(row.get("next_action") or "").strip()
        ):
            continue
        if latest is None or step >= int(latest["step"]):
            latest = row
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delegation-key", required=True)
    parser.add_argument("--progress-file", type=Path, default=DEFAULT_PROGRESS)
    args = parser.parse_args(argv)
    row = latest_resumable_checkpoint(args.progress_file, args.delegation_key)
    if row is None:
        print(
            f"RETRY-REFUSED-NO-CHECKPOINT delegation_key={args.delegation_key} "
            f"progress_file={args.progress_file}"
        )
        return REFUSED_RC
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
