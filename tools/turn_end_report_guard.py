#!/usr/bin/env python3
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Stop hook: refuse to end a turn whose last assistant output is not plain text.

The swallowed-output defect (operator x5, 2026-08-12): a turn that ends on a
tool call (typically ScheduleWakeup) renders as collapsed command rows on
mobile -- the operator sees NOTHING. The binding turn law says every turn ends
with a plain-text report as its LAST element. Memory-file discipline failed 5
times; this hook makes the law structural: it BLOCKS the Stop and tells the
model to write the report.

Contract (Claude Code Stop hook):
  stdin  : JSON {session_id, transcript_path, stop_hook_active, ...}
  block  : print JSON {"decision": "block", "reason": ...} to stdout, exit 0
  allow  : exit 0 with no decision output
Loop safety: if stop_hook_active is true (we already blocked once this stop),
always allow -- one jolt per turn, never an infinite loop.
Fail-open: any parse/read error allows the stop (a guard must never wedge the
session -- the control-plane-safety rule).

Testable core: evaluate_transcript_lines(lines, stop_hook_active) -> reason|None.
"""
from __future__ import annotations

import json
import sys

MIN_REPORT_CHARS = 40  # a real report, not a fragment

BLOCK_REASON = (
    "TURN-END GUARD: the last assistant output is a tool call with no trailing "
    "plain-text report. The operator sees NOTHING on mobile (5th recurrence "
    "2026-08-12). Write the full plain-text report NOW as the final message -- "
    "status, findings, frontier line. ScheduleWakeup's 'nothing more to do' is "
    "a scheduling receipt, never permission to end silently (memory: "
    "never-end-turn-on-schedulewakeup-report-is-owed)."
)


def _last_assistant_blocks(lines):
    """Return the content blocks of the final assistant record, else None."""
    last = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if rec.get("type") == "assistant":
            last = rec
    if last is None:
        return None
    msg = last.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return None


def evaluate_transcript_lines(lines, stop_hook_active):
    """Return a block reason, or None to allow the stop."""
    if stop_hook_active:
        return None  # already jolted once this stop; never loop
    blocks = _last_assistant_blocks(lines)
    if blocks is None:
        return None  # fail-open: nothing parseable
    # The final block of the final assistant record must be substantive text.
    final = blocks[-1] if blocks else None
    if (
        isinstance(final, dict)
        and final.get("type") == "text"
        and len(str(final.get("text", "")).strip()) >= MIN_REPORT_CHARS
    ):
        return None
    return BLOCK_REASON


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # fail-open
    transcript_path = data.get("transcript_path")
    if not transcript_path:
        return 0
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return 0  # fail-open
    reason = evaluate_transcript_lines(lines, bool(data.get("stop_hook_active")))
    if reason:
        print(json.dumps({"decision": "block", "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
