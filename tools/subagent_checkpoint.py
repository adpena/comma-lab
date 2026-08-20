#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Subagent crash-resume checkpoint tool.

Background — the bug class this prevents
─────────────────────────────────────────
2026-05-14 operator directive: "why did it die? need to investigate and fix
permanently". Empirical anchor: the Anthropic API returned ``Internal server
error`` mid-subagent-session for a Wyner-Ziv research subagent (id
``a1362a24d986029c3``) that had completed 17 minutes of work / 58 tool uses /
1704 tokens. All in-flight progress was lost; the parent had to re-spawn from
scratch with no resume signal. A second incident this session
(WAVE-3-HNERV-C-RETRY pattern for DSNeRV + HiNeRV trainers) had the same
failure class and was only recoverable because the subagent had already
committed intermediate progress that the successor could grep for.

The bug class: long-running subagents accumulate non-trivial work in-context
(file edits, research synthesis, dispatch plans) that is invisible to the
parent until either (a) a commit lands, or (b) the subagent reports back.
When the API crashes mid-session, every uncommitted-and-unreported byte is
lost. The parent has no canonical place to look for "did predecessor X get
anywhere before crashing?".

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable: structural extinction requires (1) a canonical place to
checkpoint subagent progress + (2) a discipline for every long-running
subagent to write checkpoints + (3) a STRICT preflight gate that refuses
subagent commits without checkpoint traces.

This tool is layer 1: the canonical checkpoint store.

Schema
──────
Records are appended JSONL to ``.omx/state/subagent_progress.jsonl``. Each
record is one line of JSON with these fields::

    {
        "subagent_id": "<freeform string; conventionally the subagent's name>",
        "parent_id_or_session": "<optional parent session id>",
        "step": <integer 1..N or string 'complete'>,
        "status": <"in_progress" | "blocked" | "complete">,
        "files_touched": ["<repo-relative path>", ...],
        "next_action": "<one-line description of what comes next>",
        "findings": ["<one-line thing LEARNED at this step>", ...],
        "notes": "<freeform>",
        "written_at_utc": "<ISO-8601 timestamp>",
        "pid": <integer>,
        "host": "<hostname>",
    }

Per CLAUDE.md Catalog #131 (``check_no_bare_writes_to_shared_state``):
every write acquires ``fcntl.flock(LOCK_EX)`` on ``.omx/state/.subagent_progress.lock``
so concurrent appends from sibling subagents serialize without lost rows.

Usage
─────
Write a checkpoint::

    .venv/bin/python tools/subagent_checkpoint.py \
        --subagent-id WAVE-7-FOO-SUBAGENT \
        --step 3 \
        --status in_progress \
        --files-touched src/tac/foo.py,src/tac/tests/test_foo.py \
        --next-action "wire foo() into preflight_all() and add 5 more tests" \
        --finding "foo() is called from 3 sites, not 1 - the docstring is stale" \
        --notes "completed 12 of estimated 25 tool uses"

Read latest checkpoints for a subagent::

    .venv/bin/python tools/subagent_checkpoint.py read \
        --subagent-id WAVE-7-FOO-SUBAGENT

Read the running knowledge log — what was LEARNED, not where to resume::

    .venv/bin/python tools/subagent_checkpoint.py read --findings
    .venv/bin/python tools/subagent_checkpoint.py read --findings \
        --subagent-id WAVE-7-FOO-SUBAGENT

Read raises ``SystemExit(2)`` if no records exist for the subagent (so
predecessor-resume scripts can branch on the rc).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / ".omx" / "state"
JSONL_PATH = STATE_DIR / "subagent_progress.jsonl"
LOCK_PATH = STATE_DIR / ".subagent_progress.lock"

# Lock acquisition timeout in seconds. A single append is fast (<10ms) so 30s
# is generous even under heavy fan-out contention.
LOCK_TIMEOUT_SECONDS = 30

VALID_STATUSES = ("in_progress", "blocked", "complete")


def default_session_anchor(env: dict[str, str] | None = None) -> str | None:
    """Return the strongest local agent-session identity available."""
    source = os.environ if env is None else env
    for key in (
        "CODEX_THREAD_ID",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_BRIDGE_SESSION_ID",
    ):
        value = source.get(key, "").strip()
        if value:
            return value
    return None


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _acquire_lock(timeout_seconds: int):
    """Open the lock file and acquire fcntl LOCK_EX with timeout.

    Returns an open file handle that the caller must close to release.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)
    fh = open(LOCK_PATH, "r+")  # noqa: SIM115 - caller owns lock lifetime
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(
                    f"could not acquire {LOCK_PATH} within {timeout_seconds}s"
                ) from None
            time.sleep(0.05)


def _release_lock(fh) -> None:
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def _validate_record(record: dict) -> None:
    """Sanity-check a record before append. Raises ValueError on bad input."""
    sid = record.get("subagent_id")
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError("subagent_id must be a non-empty string")
    if any(c in sid for c in ("\n", "\t", "\x1f")):
        raise ValueError("subagent_id must not contain newlines/tabs/0x1f")

    status = record.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {VALID_STATUSES!r}, got {status!r}"
        )

    step = record.get("step")
    if not (isinstance(step, int) or step == "complete"):
        raise ValueError(
            f"step must be int or the literal string 'complete', got {step!r}"
        )

    files = record.get("files_touched", [])
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise ValueError("files_touched must be a list of strings")

    next_action = record.get("next_action", "")
    if not isinstance(next_action, str):
        raise ValueError("next_action must be a string")

    notes = record.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")

    findings = record.get("findings")
    if findings is not None and (
        not isinstance(findings, list)
        or not all(isinstance(f, str) for f in findings)
    ):
        raise ValueError("findings must be None or a list of strings")


def append_checkpoint(
    *,
    subagent_id: str,
    step: int | str,
    status: str,
    files_touched: list[str],
    next_action: str,
    notes: str = "",
    parent_id_or_session: str | None = None,
    lane_id: str | None = None,
    respawn_context: str | None = None,
    expected_outputs: list[str] | None = None,
    findings: list[str] | None = None,
) -> dict:
    """Append a single checkpoint record under the fcntl lock.

    Returns the record as-written (including server-side fields like
    ``written_at_utc`` / ``pid`` / ``host``).

    ``lane_id`` (Codex finding #3, 2026-05-14) is a structured field that
    enables resume-lookup via ``read_checkpoints_by_lane``. Older checkpoint
    records that pre-date this field still satisfy the lane query via
    notes-substring fallback.

    ``respawn_context`` + ``expected_outputs`` (task #388, 2026-07-09) are
    OPTIONAL, additive, legacy-compatible fields consumed by
    ``tac.session_bus.recovery_manifest``. ``respawn_context`` is a
    pointer-rich (<=2KB) string a successor can paste to resume a crashed
    predecessor (task#, spec paths, protocol); ``expected_outputs`` names the
    files/artifacts the in-flight agent was going to produce. Records written
    before these fields still load — readers use ``.get(...)``.

    ``findings`` (`ddm_rs2`, 2026-08-03) is an OPTIONAL, additive,
    legacy-compatible list of one-line things the agent LEARNED at this step.
    It closes the store's original blind spot: every prior field answers
    *"where do I resume"* and none answers *"what did we learn"*. When four
    arms were killed mid-flight by a provider usage limit on 2026-08-03,
    ``ddm_gd2``'s checkpoint recorded only a next action ("verify mt1 rate
    ladder; determine whether a no-train ds=32 archive yields a meaningful
    d_seg") and not the structural blocker it had already found. The cost is
    measured and it is not "the finding was lost": a sister arm, ``ddm_gd3``,
    spent a whole unit RE-DERIVING it (commit ``db3abc5b4a``). The bug class
    this field extincts is therefore PAID REDISCOVERY, not permanent loss —
    which is the cheaper claim and the true one. Findings are per-step and APPEND-ONLY: a
    later step never rewrites an earlier step's finding, so the JSONL is a
    running knowledge log a successor (or a harvester) can read end-to-end
    without replaying the transcript. Records written before this field still
    load — readers use ``.get("findings") or []``.
    """
    # Validate BEFORE the list() coercion below so callers passing a string
    # (or other non-list) for ``files_touched`` get a clear error rather than
    # silently being coerced to a list-of-characters.
    if not isinstance(files_touched, list) or not all(
        isinstance(f, str) for f in files_touched
    ):
        raise ValueError("files_touched must be a list of strings")
    if expected_outputs is not None and (
        not isinstance(expected_outputs, list)
        or not all(isinstance(f, str) for f in expected_outputs)
    ):
        raise ValueError("expected_outputs must be None or a list of strings")
    if respawn_context is not None and not isinstance(respawn_context, str):
        raise ValueError("respawn_context must be None or a string")
    if findings is not None and (
        not isinstance(findings, list)
        or not all(isinstance(f, str) for f in findings)
    ):
        raise ValueError("findings must be None or a list of strings")
    record = {
        "subagent_id": subagent_id,
        "parent_id_or_session": parent_id_or_session,
        "lane_id": lane_id,
        "step": step,
        "status": status,
        "files_touched": list(files_touched),
        "next_action": next_action,
        "findings": list(findings) if findings is not None else None,
        "notes": notes,
        "respawn_context": respawn_context,
        "expected_outputs": (
            list(expected_outputs) if expected_outputs is not None else None
        ),
        "written_at_utc": _now_iso(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    _validate_record(record)

    fh = _acquire_lock(LOCK_TIMEOUT_SECONDS)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        # Append-only: open in 'a' mode under the lock so multiple appenders
        # don't truncate each other's content. The lock serializes appends so
        # the kernel's append-atomicity is irrelevant; this is belt + suspenders.
        with open(JSONL_PATH, "a") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
    finally:
        _release_lock(fh)
    return record


def read_checkpoints(subagent_id: str | None = None) -> list[dict]:
    """Read all checkpoint records, optionally filtered to one subagent.

    Returns the records in the order they appear in the JSONL file (which is
    the order they were written under the lock).
    """
    if not JSONL_PATH.exists():
        return []
    rows: list[dict] = []
    with open(JSONL_PATH) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if subagent_id is not None and rec.get("subagent_id") != subagent_id:
                continue
            rows.append(rec)
    return rows


def latest_checkpoint(subagent_id: str) -> dict | None:
    """Return the most recent record for ``subagent_id`` or None."""
    rows = read_checkpoints(subagent_id)
    if not rows:
        return None
    return rows[-1]


# ─── Predecessor-resume query paths (Codex finding #3, 2026-05-14) ───────
#
# Anchor: when a long-running subagent crashes (Anthropic API "Internal
# server error" mid-session) the parent typically respawns a SUCCESSOR with
# a NEW subagent_id. The original ``read --subagent-id`` flow requires an
# exact match, so the documented startup query returned no rows even when
# the predecessor had checkpointed — defeating the core crash-resume
# scenario.
#
# The fix adds three additional query modes:
#   * ``read --parent-id-or-session <id>`` — resolves by predecessor's
#     parent_id_or_session field, supporting "spawn under the same parent
#     session" semantics.
#   * ``read --latest-incomplete`` — returns the most-recent non-complete
#     record so a successor can grep for whatever's in-progress globally.
#   * ``read --lane-id <lane>`` — resolves by lane id (matches structured
#     ``lane_id`` field on the record OR substring inside ``notes`` for
#     backward compatibility with older checkpoints).
#
# Memory: feedback_codex_3_findings_fix_landed_20260514.md.


def read_checkpoints_by_parent(parent_id_or_session: str) -> list[dict]:
    """Return all checkpoints with matching ``parent_id_or_session`` field.

    Used by a SUCCESSOR subagent that has a different ``subagent_id`` from
    the crashed predecessor but inherited the parent session anchor. Records
    are ordered by write time (the JSONL append order, which is also the
    write order under the lock).
    """
    if not parent_id_or_session:
        raise ValueError("parent_id_or_session must be a non-empty string")
    rows: list[dict] = []
    for rec in read_checkpoints():
        if rec.get("parent_id_or_session") == parent_id_or_session:
            rows.append(rec)
    return rows


def read_checkpoints_by_lane(lane_id: str) -> list[dict]:
    """Return all checkpoints associated with ``lane_id``.

    Resolution order:
      1. records that carry a structured ``lane_id`` field equal to ``lane_id``;
      2. records whose ``notes`` field contains ``lane_id`` as a substring
         (backward-compat for checkpoints written before the structured
         field landed).

    Returns the union of (1) and (2) preserving JSONL append order.
    """
    if not lane_id:
        raise ValueError("lane_id must be a non-empty string")
    rows: list[dict] = []
    for rec in read_checkpoints():
        if rec.get("lane_id") == lane_id:
            rows.append(rec)
            continue
        notes = rec.get("notes", "")
        if isinstance(notes, str) and lane_id in notes:
            rows.append(rec)
    return rows


def latest_incomplete_checkpoint() -> dict | None:
    """Return the MOST-RECENT checkpoint whose status is not ``complete``.

    Used by a successor that has no predecessor id to start from: the
    most-recent in-progress (or blocked) record in the JSONL is the most
    plausible predecessor-resume candidate.
    """
    rows = read_checkpoints()
    for rec in reversed(rows):
        if rec.get("status") and rec["status"] != "complete":
            return rec
    return None


def latest_incomplete_for_parent(parent_id_or_session: str) -> dict | None:
    """Return the latest non-complete record for a given parent session."""
    rows = read_checkpoints_by_parent(parent_id_or_session)
    for rec in reversed(rows):
        if rec.get("status") and rec["status"] != "complete":
            return rec
    return None


def latest_incomplete_for_lane(lane_id: str) -> dict | None:
    """Return the latest non-complete record for a given lane id."""
    rows = read_checkpoints_by_lane(lane_id)
    for rec in reversed(rows):
        if rec.get("status") and rec["status"] != "complete":
            return rec
    return None


def read_findings(subagent_id: str | None = None) -> list[dict]:
    """Return the running knowledge log: every recorded finding, in order.

    ``ddm_rs2``, 2026-08-03. This is the query the store could not answer
    before the ``findings`` field existed. Each returned row is::

        {"subagent_id": ..., "step": ..., "written_at_utc": ..., "finding": "<one line>"}

    one row PER FINDING (a checkpoint carrying three findings yields three
    rows), so a successor or harvester can read what was learned without
    replaying any transcript. Records that pre-date the field contribute
    nothing rather than raising — the log is simply shorter for them, which
    is the honest representation of the fact that their findings were never
    captured.
    """
    out: list[dict] = []
    for rec in read_checkpoints(subagent_id):
        for finding in rec.get("findings") or []:
            out.append(
                {
                    "subagent_id": rec.get("subagent_id"),
                    "step": rec.get("step"),
                    "written_at_utc": rec.get("written_at_utc"),
                    "finding": finding,
                }
            )
    return out


def _parse_files_touched(raw: str | None) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_step(raw: str) -> int | str:
    if raw == "complete":
        return "complete"
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(
            f"--step must be an integer or the literal string 'complete', "
            f"got {raw!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Subagent crash-resume checkpoint tool. Writes JSONL to "
            ".omx/state/subagent_progress.jsonl under fcntl lock per "
            "Catalog #131 bare-write discipline."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # 'read' subcommand
    read_p = subparsers.add_parser(
        "read",
        help=(
            "Read latest checkpoint(s) for a subagent. For crash-resume of "
            "a SUCCESSOR with a different subagent_id, use "
            "--parent-id-or-session, --latest-incomplete, or --lane-id "
            "(Codex finding #3 fix, 2026-05-14)."
        ),
    )
    # ONE of {--subagent-id, --parent-id-or-session, --lane-id,
    # --latest-incomplete} is required. We don't use ``required=True`` on any
    # single argument because the four are mutually-substitutable query
    # paths; the validator at the end of main() enforces "at least one".
    read_p.add_argument(
        "--subagent-id",
        default=None,
        help="Subagent id to look up (exact match).",
    )
    read_p.add_argument(
        "--parent-id-or-session",
        default=None,
        help=(
            "Predecessor's parent_id_or_session value. Returns checkpoints "
            "for any subagent_id that shared this parent. Recommended for "
            "successor subagents respawned after a crash."
        ),
    )
    read_p.add_argument(
        "--lane-id",
        default=None,
        help=(
            "Lane id (matches structured lane_id field on a record OR "
            "substring inside notes for backward compatibility)."
        ),
    )
    read_p.add_argument(
        "--latest-incomplete",
        action="store_true",
        help=(
            "Return only the most-recent record whose status is not "
            "'complete'. Use this when neither the predecessor subagent id "
            "nor the parent session is known."
        ),
    )
    read_p.add_argument(
        "--latest-only",
        action="store_true",
        help="Print only the most recent record (default: all records).",
    )
    read_p.add_argument(
        "--findings",
        action="store_true",
        help=(
            "Print the running knowledge log (one row per recorded finding) "
            "instead of full records. Answers 'what did we learn', which the "
            "resume fields cannot. May be used with --subagent-id, or with NO "
            "query mode at all to read the whole fleet's findings."
        ),
    )

    # Default (write) flags directly on the top-level parser
    parser.add_argument(
        "--subagent-id",
        help="Subagent id (required for writes).",
    )
    parser.add_argument(
        "--step",
        help="Step number (integer >=1) or the literal string 'complete'.",
    )
    parser.add_argument(
        "--status",
        choices=VALID_STATUSES,
        help=f"One of {VALID_STATUSES}.",
    )
    parser.add_argument(
        "--files-touched",
        default="",
        help="Comma-separated repo-relative file paths.",
    )
    parser.add_argument(
        "--next-action",
        default="",
        help="One-line description of the next planned action.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Freeform notes (multi-line allowed; will be JSON-escaped).",
    )
    parser.add_argument(
        "--parent-id-or-session",
        default=default_session_anchor(),
        help="Parent/session anchor (default: CODEX_THREAD_ID, then Claude session id).",
    )
    parser.add_argument(
        "--lane-id",
        default=None,
        help=(
            "Optional structured lane id (e.g. lane_codex_3_findings_fix_"
            "20260514). Codex finding #3 fix 2026-05-14: enables successor "
            "subagents to resume-lookup by lane without needing the "
            "predecessor's subagent id."
        ),
    )
    parser.add_argument(
        "--finding",
        action="append",
        default=None,
        dest="finding",
        help=(
            "One-line thing you LEARNED at this step. Repeat the flag for "
            "multiple findings. Deliberately NOT comma-split: findings are "
            "prose and prose contains commas. Required in spirit by the "
            "standard subagent contract — every checkpoint should carry at "
            "least one."
        ),
    )
    parser.add_argument(
        "--respawn-context",
        default=None,
        help=(
            "Optional pointer-rich (<=2KB) string a successor can paste to "
            "resume this agent after a crash (task#, spec paths, protocol). "
            "Consumed by tac.session_bus.recovery_manifest (task #388)."
        ),
    )
    parser.add_argument(
        "--expected-outputs",
        default="",
        help=(
            "Optional comma-separated files/artifacts this agent was going to "
            "produce (consumed by tac.session_bus.recovery_manifest)."
        ),
    )

    args = parser.parse_args(argv)

    if args.subcommand == "read":
        # Validate: exactly one query path must be present.
        query_modes = sum(
            1
            for v in (
                args.subagent_id,
                args.parent_id_or_session,
                args.lane_id,
                args.latest_incomplete,
            )
            if v
        )
        if query_modes == 0 and not args.findings:
            parser.error(
                "'read' requires one of: --subagent-id, "
                "--parent-id-or-session, --lane-id, --latest-incomplete "
                "(or --findings alone for the whole-fleet knowledge log)"
            )
        if query_modes > 1:
            parser.error(
                "'read' accepts exactly ONE query mode at a time; "
                "got multiple: --subagent-id / --parent-id-or-session / "
                "--lane-id / --latest-incomplete"
            )

        if args.findings:
            # The knowledge log. Works with --subagent-id or with no query
            # mode at all (whole fleet). Other query modes are not supported
            # here because a finding belongs to an agent, not to a lane.
            if args.parent_id_or_session or args.lane_id or args.latest_incomplete:
                parser.error(
                    "--findings supports --subagent-id or no query mode; "
                    "it does not compose with --parent-id-or-session / "
                    "--lane-id / --latest-incomplete"
                )
            rows = read_findings(args.subagent_id)
            if not rows:
                label = (
                    f"--subagent-id={args.subagent_id!r}"
                    if args.subagent_id
                    else "the whole fleet"
                )
                print(
                    f"[subagent-checkpoint] no findings recorded for {label}",
                    file=sys.stderr,
                )
                return 2
            for row in rows[-1:] if args.latest_only else rows:
                print(json.dumps(row, sort_keys=True))
            return 0

        records: list[dict]
        query_label: str
        if args.latest_incomplete:
            rec = latest_incomplete_checkpoint()
            records = [rec] if rec is not None else []
            query_label = "--latest-incomplete"
        elif args.parent_id_or_session:
            records = read_checkpoints_by_parent(args.parent_id_or_session)
            query_label = (
                f"--parent-id-or-session={args.parent_id_or_session!r}"
            )
        elif args.lane_id:
            records = read_checkpoints_by_lane(args.lane_id)
            query_label = f"--lane-id={args.lane_id!r}"
        else:
            records = read_checkpoints(args.subagent_id)
            query_label = f"--subagent-id={args.subagent_id!r}"

        if not records:
            print(
                f"[subagent-checkpoint] no records for {query_label}",
                file=sys.stderr,
            )
            return 2
        out = records[-1:] if args.latest_only else records
        for rec in out:
            print(json.dumps(rec, sort_keys=True))
        return 0

    # Default = write
    if not args.subagent_id:
        parser.error("--subagent-id is required for writes")
    if args.step is None:
        parser.error("--step is required for writes")
    if args.status is None:
        parser.error("--status is required for writes")

    step_val = _parse_step(args.step)
    files = _parse_files_touched(args.files_touched)
    expected_outputs = _parse_files_touched(args.expected_outputs) or None

    record = append_checkpoint(
        subagent_id=args.subagent_id,
        step=step_val,
        status=args.status,
        files_touched=files,
        next_action=args.next_action,
        notes=args.notes,
        parent_id_or_session=args.parent_id_or_session,
        lane_id=args.lane_id,
        respawn_context=args.respawn_context,
        expected_outputs=expected_outputs,
        findings=args.finding,
    )
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
