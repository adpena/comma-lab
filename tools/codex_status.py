#!/usr/bin/env python3
"""codex_status.py — show the status of codex delegations launched via codex_delegate.py.

Reads the ledger (.omx/state/codex_delegations.jsonl), checks each run's .done marker
and whether its codex process is still alive, and prints a table. Marks rows done when
their marker appears. --json for machine-readable.

With --classify, each DONE run's final message (`.last.txt`) is classified SEMANTICALLY
via fmtools.local_extract (our on-device Apple FM, structured generation against a closed
schema) into {landed_result, landed_with_blocker, stalled_no_result, errored} + a one-line
reason + did-it-commit — instead of brittle regex over the log. fmtools lives in its own
venv (~/Projects/fmtools/.venv/bin/python); run --classify from there, or the base venv
falls back to an honest "fm-unavailable" outcome (never a faked classification).

USAGE:  .venv/bin/python tools/codex_status.py [--json]
        ~/Projects/fmtools/.venv/bin/python tools/codex_status.py --classify [--json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / ".omx" / "state" / "codex_delegations.jsonl"
LANDING_LEDGER = REPO / ".omx" / "state" / "codex_landing_ledger.jsonl"
INBOX_DIR = REPO / ".omx" / "tmp" / "codex_inbox"

# A landed arm is DISPOSITIONED only when the landing review gate records a
# TERMINAL state (mirrors tools/codex_landing_review_gate.py). Anything else
# (undispositioned, or held_entangled) is un-reviewed = the drift/signal-loss
# the operator flagged: landed but not yet reviewed/respawned/closed.
_TERMINAL_DISPOSITIONS = {"reviewed_committed", "respawned", "closed"}
# A no-proc/no-marker arm launched within this window DIED (actionable); older =
# STALE noise (ancient self-tests) suppressed unless --all.
_STALE_AFTER_HOURS = 6.0
# An ALIVE arm with NO PROGRESS in this window is STALLED (hung stream, 0% CPU)
# — process-existence is NOT progress. Empirical anchor: the
# curvelet_optimal_form_crux_20260715 arm sat "RUNNING" for 33h at 0.0% CPU with
# its log frozen mid-line; the operator caught it by wall-clock intuition because
# this surface only checked pgrep.
#
# PROGRESS is a COMPOSITE signal (operator design 2026-07-16): log growth OR
# worktree file writes OR commits within the window. MEASURED basis:
#   (a) healthy codex logs are continuous heartbeats — both live arms sampled at
#       log_age=0m; all 10 most-recent completed arms have log-final-mtime ==
#       done-marker time to the minute (log written to the very end);
#   (b) the ONLY legitimate log silence is one long tool call that streams
#       nothing AND writes nothing — the fleet's MEASURED worst case for that
#       class is the one-time n600 gt-load ≈ 26 min (c2 bench boot, 2026-07-16)
#       / CPU-torch n600 verdict ≈ 23 min (#495);
#   (c) long calls that DO write output files are rescued by the worktree-mtime
#       leg of the composite, so (b) bounds the all-silent case.
# DERIVED threshold: 2 × the 26-min measured ceiling (safety factor 2 on the
# slowest legitimate all-silent step). Not a hand-picked constant — re-derive if
# a slower legitimate no-write compute step enters the fleet's workload.
_MAX_LEGIT_QUIET_MIN = 26.0  # MEASURED: n600 gt-load, the slowest no-emit no-write step
_STALL_SAFETY_FACTOR = 2.0
_STALL_AFTER_HOURS = (_STALL_SAFETY_FACTOR * _MAX_LEGIT_QUIET_MIN) / 60.0  # ≈ 0.87h


def _alive(label: str, stamp: str) -> bool:
    # a codex whose -o path contains "<label>_<stamp>" is this run's process.
    # NOTE: match on the label_stamp TOKEN, never `pgrep -fl | head` — the full
    # prompt is inlined into argv, so a line-count clip silently undercounts live
    # arms (the 1-of-8 misread this hardening extincts). Use this fn, not ad-hoc pgrep.
    r = subprocess.run(["pgrep", "-f", f"{label}_{stamp}"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _latest_dispositions() -> dict[tuple[str, str], str]:
    """(label, stamp) -> latest landing-gate disposition status (append-only ⇒ last wins)."""
    out: dict[tuple[str, str], str] = {}
    if not LANDING_LEDGER.is_file():
        return out
    for line in LANDING_LEDGER.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        lbl, stmp, st = d.get("label"), d.get("stamp"), d.get("status")
        if lbl and stmp and st:
            out[(lbl, stmp)] = st
    return out


def _inbox_lines(label: str) -> int:
    """Count directive lines queued in this arm's inbox (unconsumed cursor is
    arm-internal; a non-empty inbox on a RUNNING arm flags 'directive in flight')."""
    p = INBOX_DIR / f"{label}.jsonl"
    if not p.is_file():
        return 0
    return sum(1 for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())


def _log_stale_hours(log_path: str | None) -> float | None:
    """Hours since the run's log file last grew (None if no/unreadable log).

    The mtime is the progress signal — never the process table. A live codex
    writes its event stream continuously; a frozen mtime on an alive process is
    the stalled-stream hang signature this detector exists for.
    """
    if not log_path:
        return None
    p = Path(log_path)
    try:
        if not p.is_file():
            return None
        import time
        return max(0.0, (time.time() - p.stat().st_mtime) / 3600.0)
    except OSError:
        return None


def _worktree_progress_within(worktree: str | None, hours: float) -> bool:
    """True if the arm's isolated worktree shows ANY progress within `hours`:
    a commit newer than the cutoff, or any non-.git file written after it.

    This is the composite-progress rescue (operator design 2026-07-16): a long
    tool call that streams nothing to the log but WRITES artifacts is working,
    not hung. The file scan stops at the first fresh file (-print -quit); the
    full-tree worst case only happens on arms that are probably hung anyway.
    """
    if not worktree:
        return False
    wt = Path(worktree)
    if not wt.is_dir():
        return False
    import subprocess
    import time
    cutoff = time.time() - hours * 3600.0
    try:  # commit leg (cheap)
        r = subprocess.run(["git", "-C", str(wt), "log", "-1", "--format=%ct"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip() and float(r.stdout.strip()) >= cutoff:
            return True
    except Exception:
        pass
    try:  # file-write leg (first hit short-circuits)
        mins = max(1, int(hours * 60))
        r = subprocess.run(  # subprocess-no-check-OK: recent-write probe; failure/no-match reads False (advisory status)
            ["find", str(wt), "-path", str(wt / ".git"), "-prune", "-o",
             "-type", "f", "-newermt", f"-{mins} minutes", "-print", "-quit"],
            capture_output=True, text=True, timeout=60)
        return bool(r.stdout.strip())
    except Exception:
        return False


def _age_hours(launched_utc: str | None) -> float | None:
    if not launched_utc:
        return None
    try:
        from datetime import UTC, datetime
        t = datetime.fromisoformat(launched_utc)
        return (datetime.now(UTC) - t).total_seconds() / 3600.0
    except Exception:
        return None


def _is_strand_doomed(delegation: dict) -> bool:
    """Legacy live writers without an isolated worktree cannot land safely."""
    return delegation.get("sandbox", "workspace-write") != "read-only" and (
        delegation.get("isolate") is not True
    )


def _bucket(
    alive: bool,
    done: dict | None,
    disp: str | None,
    age_h: float | None,
    *,
    strand_doomed: bool = False,
    log_stale_h: float | None = None,
    worktree_progress: bool = False,
) -> str:
    """RUNNING · STALLED (alive, NO composite progress — log frozen past
    _STALL_AFTER_HOURS AND no worktree write/commit in that window) ·
    NEEDS_REVIEW (landed, no terminal disposition) · REVIEWED · DIED (recent
    no-proc/no-marker) · STALE (ancient, suppressed by default)."""
    if done:
        return "REVIEWED" if disp in _TERMINAL_DISPOSITIONS else "NEEDS_REVIEW"
    if alive:
        if strand_doomed:
            return "STRAND_DOOMED"
        if (log_stale_h is not None and log_stale_h >= _STALL_AFTER_HOURS
                and not worktree_progress):
            return "STALLED"
        return "RUNNING"
    if age_h is not None and age_h <= _STALE_AFTER_HOURS:
        return "DIED"
    return "STALE"


def _read_done(marker: str) -> dict | None:
    p = Path(marker)
    if not p.is_file():
        return None
    out: dict = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _classify_outcome(last_text: str) -> dict:
    """Classify one codex agent's FINAL MESSAGE into a structured outcome via the
    on-device Apple FM (fmtools.local_extract, structured generation against a closed
    schema). Returns {"ok": True, "outcome", "reason", "committed"} on success, or an
    honest {"ok": False, "why": ...} when fmtools/apple_fm_sdk is unavailable (base
    venv) or the model errors — NEVER a faked/regex label. Mirrors the
    tools/dashboard_fm_events.py pattern (lazy import, guardrail-tolerant prose input)."""
    try:
        import asyncio

        import apple_fm_sdk as fm
        from fmtools import local_extract
    except Exception as exc:  # base venv has no fmtools — honest skip, not a fake label
        return {"ok": False, "why": f"fm-unavailable ({type(exc).__name__})"}

    @fm.generable()
    class CodexOutcome:
        outcome: str = fm.guide(anyOf=[
            "landed_result", "landed_with_blocker", "stalled_no_result", "errored"])
        reason: str = fm.guide(
            description="One short plain sentence: what the agent accomplished, or "
                        "why it stopped. Use only facts present in the message; never "
                        "invent numbers or outcomes.")
        committed: str = fm.guide(anyOf=["yes", "no", "unknown"])

    _instructions = (
        "You label the FINAL MESSAGE of a coding agent that was delegated a task. "
        "outcome: 'landed_result' if it reports finishing and landing/committing real "
        "work; 'landed_with_blocker' if it did partial work but names an unresolved "
        "blocker; 'stalled_no_result' if it ran but produced no usable landing; "
        "'errored' if the message itself reports a crash/error/refusal. reason: one "
        "short plain sentence using ONLY facts in the message. committed: 'yes' if the "
        "message clearly states it committed/landed code, 'no' if it clearly did not, "
        "else 'unknown'.")

    @local_extract(CodexOutcome, retries=2, instructions=_instructions)
    async def _classify(msg: str) -> CodexOutcome:
        """(instructions provided explicitly above)"""

    # prose-frame + cap: the tail carries the verdict; a dense head can trip the guardrail
    body = last_text.strip()[-1800:] or "(empty final message)"
    try:
        r = asyncio.run(_classify(f"The delegated coding agent's final message was: {body}"))
    except Exception as exc:
        return {"ok": False, "why": f"fm-error ({type(exc).__name__})"}
    return {
        "ok": True,
        "outcome": str(getattr(r, "outcome", "stalled_no_result")),
        "reason": str(getattr(r, "reason", ""))[:200],
        "committed": str(getattr(r, "committed", "unknown")),
        "classifier": "apple-fm-on-device",
    }


def status_rows(*, classify: bool = False) -> list[dict]:
    """Return the canonical fleet snapshot consumed by status and drain tools.

    Keeping liveness reconciliation here prevents drain monitors from growing a
    second, subtly different ``pgrep`` implementation.  Additional custody
    paths are carried through for liveness evidence but do not change buckets.
    """
    if not LEDGER.is_file():
        return []

    dispositions = _latest_dispositions()
    seen: dict[str, dict] = {}
    for line in LEDGER.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen[f"{row.get('label')}_{row.get('stamp')}"] = row

    rows: list[dict] = []
    for delegation in seen.values():
        label, stamp = delegation.get("label", ""), delegation.get("stamp", "")
        done = _read_done(delegation.get("done_marker", ""))
        alive = _alive(label, stamp)
        disp = dispositions.get((label, stamp))
        age_h = _age_hours(delegation.get("launched_utc"))
        strand_doomed = _is_strand_doomed(delegation)
        log_stale_h = _log_stale_hours(delegation.get("log")) if (alive and not done) else None
        # composite-progress rescue: only consulted when the log leg already
        # looks stalled (keeps the common healthy path to one stat() call).
        wt_progress = False
        if log_stale_h is not None and log_stale_h >= _STALL_AFTER_HOURS:
            wt_progress = _worktree_progress_within(
                delegation.get("worktree"), _STALL_AFTER_HOURS)
        bucket = _bucket(alive, done, disp, age_h, strand_doomed=strand_doomed,
                         log_stale_h=log_stale_h, worktree_progress=wt_progress)
        row = {
            "label": label,
            "stamp": stamp,
            "model": delegation.get("model"),
            "effort": delegation.get("effort"),
            "sandbox": delegation.get("sandbox"),
            "isolate": delegation.get("isolate"),
            "worktree": delegation.get("worktree"),
            "progress_path": delegation.get("progress_path"),
            "status": bucket,
            "contract_violation": "nonisolated_writer" if strand_doomed else None,
            "disposition": disp,
            "log_stale_hours": round(log_stale_h, 2) if log_stale_h is not None else None,
            "inbox_pending": _inbox_lines(label) if bucket in {"RUNNING", "STRAND_DOOMED", "STALLED"} else 0,
            "rc": (done or {}).get("rc"),
            "finished_utc": (done or {}).get("finished_utc"),
            "launched_utc": delegation.get("launched_utc"),
            "log": delegation.get("log"),
        }
        if classify and done:
            last_path = Path(delegation.get("last", "") or "")
            if last_path.is_file():
                row["outcome"] = _classify_outcome(
                    last_path.read_text(encoding="utf-8", errors="ignore")
                )
            else:
                row["outcome"] = {"ok": False, "why": "no-last-txt"}
        rows.append(row)
    rows.sort(key=lambda row: row.get("launched_utc") or "")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="show every row incl. REVIEWED + STALE (default: only actionable "
                         "RUNNING / NEEDS_REVIEW / DIED)")
    ap.add_argument("--classify", action="store_true",
                    help="classify each DONE run's final message via fmtools on-device FM "
                         "(run from the fmtools venv; base venv reports fm-unavailable)")
    args = ap.parse_args(argv)

    if not LEDGER.is_file():
        print("(no codex delegations yet)")
        return 0

    rows = status_rows(classify=args.classify)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("(no codex delegations yet)")
        return 0

    # high-signal digest: one summary line, then only ACTIONABLE rows by default.
    from collections import Counter
    counts = Counter(r["status"] for r in rows)
    summary = " · ".join(f"{counts[b]} {b}" for b in
                         ("STALLED", "STRAND_DOOMED", "RUNNING", "NEEDS_REVIEW", "DIED", "REVIEWED", "STALE") if counts.get(b))
    print(f"codex fleet: {summary or '(none)'}"
          + ("" if args.all else "   [--all for REVIEWED+STALE]"))
    actionable = {"STALLED", "STRAND_DOOMED", "RUNNING", "NEEDS_REVIEW", "DIED"}
    shown = [r for r in rows if args.all or r["status"] in actionable]
    for r in shown:
        me = f"{(r.get('model') or '?').split('-')[-1]}/{r.get('effort') or '?'}"
        flag = ""
        if r["status"] == "NEEDS_REVIEW":
            flag = f"  ⚠ disposition={r.get('disposition') or 'none'} → codex_landing_review_gate"
        elif r["status"] == "STRAND_DOOMED":
            flag = "  ⚠ live non-isolated writer cannot land safely — drain/harvest; do not retry"
        elif r["status"] == "STALLED":
            flag = (f"  ⚠ alive, NO progress {r.get('log_stale_hours') or '?'}h (log+worktree+commits "
                    "all silent) — hung stream: killpg the pgid, harvest its worktree, relaunch "
                    "(2026-07-16 curvelet class)")
        elif r["status"] == "DIED":
            flag = "  ⚠ no proc, no DONE marker — investigate/relaunch"
        elif r["status"] == "RUNNING" and r.get("inbox_pending"):
            flag = f"  ✉ {r['inbox_pending']} inbox directive(s)"
        print(f"  {r['status']:<13} {(r['label'] or '?'):<40} {me:<12} rc={r.get('rc') or '-'!s}{flag}")
        oc = r.get("outcome")  # present only under --classify on DONE runs
        if oc:
            print(f"      ↳ {oc['outcome']} (committed={oc.get('committed', '?')}) — {oc.get('reason', '')}"
                  if oc.get("ok") else f"      ↳ [not classified: {oc.get('why', '?')}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
