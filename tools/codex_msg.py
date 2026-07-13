#!/usr/bin/env python3
"""codex_msg.py — push a live directive into a running codex arm's watched inbox.

WHY: codex arms launched via tools/codex_delegate.py read their prompt ONCE at
launch — the main loop could not amend a running arm mid-flight (the "codex reads
its prompt once" gap). This closes it: every delegated arm is given an INBOX
contract (see codex_delegate._INBOX_CONTRACT) telling it to poll a per-label
inbox file at every checkpoint and CONSUME new directives (which SUPERSEDE
conflicting prompt instructions, being later information). This tool is the
producer side — it appends a directive the arm consumes on its next poll.

The channel is append-only fcntl-locked JSONL, one directive per line:
    {"utc": "...Z", "from": "operator|main", "priority": "normal|high|stop",
     "directive": "<text>"}

USAGE:
    # push to ONE running arm:
    .venv/bin/python tools/codex_msg.py --label throughput_fresh_eyes \
        --message "You now have full any-configuration build/adapt authority; drop the ANE-defer."

    # broadcast to EVERY arm (writes the shared _broadcast.jsonl every arm also polls):
    .venv/bin/python tools/codex_msg.py --broadcast \
        --message "New coordination: arm X owns the sparse kernel; hand it any backward finding."

    # a clean-exit directive (the arm checkpoints + exits on its next poll):
    .venv/bin/python tools/codex_msg.py --label foo --priority stop \
        --message "Superseded by a newer arm; checkpoint your state and exit."

    # inspect an inbox:
    .venv/bin/python tools/codex_msg.py --label foo --show

NOTE: only arms LAUNCHED WITH the inbox contract (codex_delegate after this
landing) poll an inbox — pre-existing running arms never learned to. `--show`
and the ledger make it obvious which arms are inbox-aware.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INBOX_DIR = REPO / ".omx" / "tmp" / "codex_inbox"
BROADCAST = INBOX_DIR / "_broadcast.jsonl"
_PRIORITIES = ("normal", "high", "stop")


def inbox_path(label: str) -> Path:
    """Canonical per-label inbox file. Kept in one place so the delegate launcher,
    the arm's poll instruction, and this producer all agree on the location."""
    return INBOX_DIR / f"{label}.jsonl"


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _show(path: Path) -> int:
    if not path.exists():
        print(f"(no inbox yet at {path})")
        return 0
    print(f"=== {path} ===")
    for line in path.read_text(encoding="utf-8").splitlines():
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Push a live directive into a running codex arm's watched inbox.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--label", help="target one running arm's inbox by its delegation label")
    g.add_argument("--broadcast", action="store_true", help="write to the fleet-wide inbox every arm polls")
    ap.add_argument("--message", help="the directive text the arm will consume")
    ap.add_argument("--from", dest="sender", default="main", choices=["operator", "main"],
                    help="who issued the directive (arms treat operator as top authority)")
    ap.add_argument("--priority", default="normal", choices=_PRIORITIES,
                    help="stop = arm should checkpoint + exit cleanly on next poll")
    ap.add_argument("--show", action="store_true", help="print the target inbox instead of writing")
    args = ap.parse_args(argv)

    target = BROADCAST if args.broadcast else inbox_path(args.label)

    if args.show:
        return _show(target)

    if not args.message:
        ap.error("--message is required unless --show")

    row = {
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": args.sender,
        "priority": args.priority,
        "directive": args.message,
    }
    _append(target, row)
    scope = "BROADCAST (all arms)" if args.broadcast else f"arm '{args.label}'"
    print(json.dumps({"pushed_to": scope, "inbox": str(target), "row": row}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
