#!/usr/bin/env python3
"""Operator-P0 ledger + digest — the anti-abandonment memory surface.

Operator binding 2026-07-15 (verbatim): "There have been multiple things I have
designated as p0 you have abandoned because you forgot and silently got
distracted and moved on to other things." + "Do we need a hook or gate or
something to remind you if p0 to survive compaction? That also demands update
when complete or when new p0 designated."

This module is BOTH:
  * the canonical fcntl-locked append-only ledger library for
    ``.omx/state/operator_p0_ledger.jsonl`` (latest-row-wins per ``p0_id``,
    modeled on the tac.deploy.modal.call_id_ledger 4-layer pattern), AND
  * the COMPACTION-SURVIVAL digest CLI: wired into the ``SessionStart`` hook
    chain (all sources, INCLUDING ``compact``) alongside tools/costate_digest.py,
    so every fresh or compacted context window re-injects the open operator-P0s.
    The apparatus remembers; the operator never has to.

Sister surface: tools/operator_p0_stop_hook.py (the Stop-hook demand-update
nag — landing 2 of the two-landing discipline). Recovery seed data:
.omx/research/operator_p0_abandonment_recovery_20260715.md.

Design invariants:
  * FAIL-OPEN in hook mode (``--session-start``): any error prints what it can
    and exits 0. A SessionStart hook must never wedge a session.
  * APPEND-ONLY + latest-row-wins: updates are new rows keyed by ``p0_id``;
    history is preserved (HISTORICAL_PROVENANCE, Catalog #110/#113 discipline).
  * fcntl LOCK_EX on a sibling ``.lock`` file for every append (Catalog #131).
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import sys

LEDGER_REL = ".omx/state/operator_p0_ledger.jsonl"
LOCK_REL = ".omx/state/.operator_p0_ledger.jsonl.lock"

STATUSES = ("open", "in_progress", "complete", "superseded")

# Required schema fields for every row (extras allowed, e.g. watch_paths/task_ids).
REQUIRED_FIELDS = (
    "p0_id",
    "designated_date",
    "verbatim_ask",
    "status",
    "evidence",
    "next_action",
    "last_verified_utc",
    "source",
)


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root(explicit: str | None = None) -> str:
    """Resolve the repo root: explicit arg → $CLAUDE_PROJECT_DIR → cwd-walk for .git."""
    if explicit:
        return explicit
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and os.path.isdir(env):
        return env
    d = os.getcwd()
    while d and d != "/":
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


def ledger_path(root: str) -> str:
    return os.path.join(root, LEDGER_REL)


def read_rows(root: str) -> list[dict]:
    """All rows, file order, malformed lines skipped (defensive, never raises)."""
    rows: list[dict] = []
    path = ledger_path(root)
    if not os.path.exists(path):
        return rows
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and row.get("p0_id"):
                    rows.append(row)
    except Exception:
        pass
    return rows


def read_ledger(root: str) -> dict[str, dict]:
    """Latest-row-wins view keyed by p0_id (file order == append order)."""
    latest: dict[str, dict] = {}
    for row in read_rows(root):
        latest[str(row["p0_id"])] = row
    return latest


def max_written_utc(root: str) -> str:
    """The max ``written_at_utc`` across all rows ('' when empty) — the Stop hook's
    cheap "did any ledger row land this window?" signal."""
    best = ""
    for row in read_rows(root):
        w = str(row.get("written_at_utc") or "")
        if w > best:
            best = w
    return best


def validate_row(row: dict) -> list[str]:
    """Schema problems (empty == valid)."""
    problems = []
    for f in REQUIRED_FIELDS:
        if not str(row.get(f) or "").strip():
            problems.append(f"missing/empty required field: {f}")
    status = str(row.get("status") or "")
    if status and status not in STATUSES:
        problems.append(f"status {status!r} not in {STATUSES}")
    return problems


def append_row(root: str, row: dict) -> dict:
    """Append one row under fcntl LOCK_EX; stamps written_at_utc; validates schema.

    Raises ValueError on schema problems (write path is fail-CLOSED; only the
    read/digest path is fail-open)."""
    row = dict(row)
    row.setdefault("last_verified_utc", _now())
    row["written_at_utc"] = _now()
    problems = validate_row(row)
    if problems:
        raise ValueError("operator_p0_ledger row invalid: " + "; ".join(problems))
    path = ledger_path(root)
    lock = os.path.join(root, LOCK_REL)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(lock, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return row


def open_rows(root: str) -> list[dict]:
    """Latest rows with status ∈ {open, in_progress}, open first, then by p0_id."""
    latest = read_ledger(root)
    rows = [r for r in latest.values() if r.get("status") in ("open", "in_progress")]
    rows.sort(key=lambda r: ({"open": 0, "in_progress": 1}.get(str(r.get("status")), 2),
                             str(r.get("p0_id"))))
    return rows


# --- SessionStart line budget (ddm_gh2, 2026-07-31) -------------------------
# MEASURED before this change: 7,315 B (~1,830 tokens) for 22 rows, paid on EVERY
# session start AND every compaction, at 2 lines/row (~362 B/row).
#
# The anti-abandonment invariant is ABSOLUTE and is NOT traded for tokens: EVERY
# open/in_progress row still prints, always.  What the hook rendering drops is the
# per-row ``verbatim_ask``, because (a) for nearly every row the p0_id already
# states it (``p0_ema_calibration_20260717`` vs ask "EMA calibration is p0 too"),
# (b) it was truncated mid-word at 140 chars, so the tail was an unparseable
# fragment rather than signal, and (c) the full text is preserved in the ledger
# and one ``--verbose`` away.  ``next_action`` — the actionable cue — is kept.
_COMPACT_NEXT_CHARS = 120
_STATUS_ABBREV = {"open": "O ", "in_progress": "IP"}


def format_digest(
    rows: list[dict], header: bool = True, verbose: bool = False, compact: bool = False
) -> str:
    """Human-readable digest of ledger rows.

    ``compact`` is the SessionStart hook rendering: one line per row (id + status +
    next-action cue), never fewer rows.  Default/``verbose`` renderings are
    unchanged, so an interactive call still shows the verbatim asks."""
    out: list[str] = []
    if header:
        n_open = sum(1 for r in rows if r.get("status") == "open")
        n_prog = sum(1 for r in rows if r.get("status") == "in_progress")
        tail = (
            "full text: tools/operator_p0_digest.py --verbose"
            if compact
            else "update via tools/operator_p0_digest.py --update"
        )
        out.append(
            f"OPERATOR-P0 LEDGER — {n_open} open / {n_prog} in_progress "
            f"(operator-designated P0s; NONE may be silently dropped — {tail})"
        )
    for r in rows:
        raw_status = str(r.get("status") or "?")
        nxt = str(r.get("next_action") or "").strip().replace("\n", " ")
        if compact:
            # One line per row: the handle, the state, and a recognition cue.
            cue = nxt[:_COMPACT_NEXT_CHARS] or "(no next_action recorded)"
            out.append(f"  [{_STATUS_ABBREV.get(raw_status, '? ')}] {r.get('p0_id')} · {cue}")
            continue
        ask = str(r.get("verbatim_ask") or "").strip().replace("\n", " ")
        if not verbose:
            ask = ask[:140]
            nxt = nxt[:160]
        out.append(f"  [{raw_status.upper()}] {r.get('p0_id')} ({r.get('designated_date')}): {ask}")
        if nxt:
            out.append(f"      NEXT: {nxt}")
        if verbose and r.get("evidence"):
            out.append(f"      EVIDENCE: {str(r.get('evidence')).strip()[:300]}")
    if not rows:
        out.append("  (no open operator-P0s — ledger clean)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="repo root (default: auto)")
    ap.add_argument("--session-start", action="store_true",
                    help="hook mode: print open/in_progress digest, ALWAYS exit 0")
    ap.add_argument("--all", action="store_true", help="include complete/superseded rows")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--verbose", action="store_true", help="full text + evidence")
    # --update appends a latest-wins row for an existing or new p0_id.
    ap.add_argument("--update", metavar="P0_ID", default=None)
    ap.add_argument("--status", choices=STATUSES, default=None)
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--next-action", default=None)
    ap.add_argument("--verbatim-ask", default=None)
    ap.add_argument("--designated-date", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--watch-path", action="append", default=None,
                    help="repo-relative path prefix tied to this P0 (repeatable; "
                         "the Stop hook uses it to demand ledger updates)")
    ap.add_argument("--task-id", action="append", default=None,
                    help="harness task number tied to this P0 (repeatable)")
    args = ap.parse_args(argv)

    if args.session_start:
        # FAIL-OPEN hook mode: never a nonzero exit, never a traceback.
        try:
            root = repo_root(args.root)
            rows = open_rows(root)
            if rows:
                print(format_digest(rows, compact=not args.verbose))
        except Exception:
            pass
        return 0

    root = repo_root(args.root)

    if args.update:
        prior = read_ledger(root).get(args.update, {})
        row = {
            "p0_id": args.update,
            "designated_date": args.designated_date or prior.get("designated_date") or _now()[:10],
            "verbatim_ask": args.verbatim_ask or prior.get("verbatim_ask") or "",
            "status": args.status or prior.get("status") or "open",
            "evidence": args.evidence or prior.get("evidence") or "",
            "next_action": (args.next_action if args.next_action is not None
                            else prior.get("next_action") or ""),
            "source": args.source or prior.get("source") or "",
            "last_verified_utc": _now(),
        }
        if args.watch_path or prior.get("watch_paths"):
            row["watch_paths"] = args.watch_path or prior.get("watch_paths")
        if args.task_id or prior.get("task_ids"):
            row["task_ids"] = args.task_id or prior.get("task_ids")
        try:
            append_row(root, row)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(f"appended: {args.update} status={row['status']}")
        return 0

    latest = read_ledger(root)
    rows = (sorted(latest.values(), key=lambda r: str(r.get("p0_id")))
            if args.all else open_rows(root))
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(format_digest(rows, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
