#!/usr/bin/env python3
"""Codex landing review gate — a Claude Code ``Stop`` hook + CLI.

THE FAILURE THIS EXTINCTS (operator 2026-07-14): a codex/sol subagent that has
LANDED (its ``.done`` marker / ``.last.txt`` is present and its process is gone)
but whose work has NOT been reviewed + committed + given a follow-on-respawn
disposition is a **dangerous source of drift, signal loss, confounding, and
un-reviewed recovered state**. Landed-but-un-dispositioned codex work must NOT
sit: the moment an arm settles, main MUST review it and record a disposition —
commit it, respawn its follow-on, close it, or explicitly hold-it-entangled with
the NAMED live arms blocking the commit (a tracked queue state, never a silent
grave; sister of "off is a tracked queue" + "recovery-written = UNREVIEWED").

This module is BOTH:
  * the **immediate** enforcement — a ``Stop`` hook that fires at the end of each
    main-agent turn and injects ONE firm nudge naming every landed arm with no
    disposition (loop-safe, fail-open — a Stop hook must NEVER wedge a session);
  * the **structural** enforcement's data layer — the disposition ledger that the
    STRICT preflight guard ``check_codex_landing_dispositioned`` reads to REFUSE
    a stale un-dispositioned landing at commit time (the two-landing pattern:
    "Bugs must be permanently fixed AND self-protected against").

Disposition states (terminal unless noted):
  * ``reviewed_committed`` — reviewed + landed via the serializer (commit sha).
                             REQUIRES ``--consumed-by`` (see below).
  * ``respawned``          — a follow-on arm was launched (respawn label).
  * ``closed``             — reviewed + intentionally not committed (reason).
                             REQUIRES ``--consumed-by`` (see below).

CONSUMPTION POINTER (operator 2026-07-17 — "What other findings memos were
landed that nobody consumed? That is a super poisonous bug class"): a
``reviewed_committed`` / ``closed`` disposition is a CUSTODY state (the arm's
working-tree bytes were handled) — it says NOTHING about whether the arm's
FINDINGS were routed into a decision surface. The proven orphan class: 4 basis
arms rc=0 REVIEWED with findings unconsumed until operator memory caught it.
Therefore terminal review dispositions REQUIRE ``--consumed-by
<artifact-path-or-task#>`` — the DAG FEED memo / task# / P0 row / spec § /
DSL lever / audit memo that RECORDS A DECISION about the findings (routing,
absorption, or a named-reason rejection all count; a bare filename mention
does not). ``none:<reason>`` is the explicit nothing-to-consume escape for
pure-mechanical arms (bare ``none``/``n/a``/``tbd`` refused). ``held_entangled``
/ ``respawned`` are exempt (non-terminal / the respawn IS the route). Old
ledger rows without the field are unchanged (backwards-compatible; enforcement
is at the CLI write path only). Sister warn-only preflight surface:
``tac.preflight.check_codex_findings_memos_consumed`` flags fresh
``codex_findings_*.md`` memos with no consumer surface at all.
  * ``held_entangled``     — NOT terminal: the commit is blocked because named
                             LIVE arms are still editing the shared files; carries
                             ``blocked_by`` arm labels. AUTO-RE-SURFACES to pending
                             the moment all blocking arms are dead (held ≠ grave).

Design invariants (a Stop hook is load-bearing infrastructure):
  * FAIL-OPEN — any error / missing state / non-git ⇒ allow the stop (exit 0).
  * LOOP-SAFE — one firm nudge per distinct pending-set signature, guarded by
    BOTH Claude Code's ``stop_hook_active`` flag AND a persisted signature, so it
    cannot loop even when ``stop_hook_active`` is absent. It NAGS, never WEDGES —
    the hard refusal lives in the preflight guard, not here.
  * READ-ONLY on the fleet — never kills a process, never edits an arm's files.

Not placed in ``tac`` — this is Claude-workflow apparatus, not contest/codec
logic (per CLAUDE.md ``tac`` cleanliness rule). Pure logic lives in module-level
functions so it is unit-testable without a live fleet.

USAGE:
  # Stop-hook mode (no args; JSON on stdin) — wired in .claude/settings.json:
  .venv/bin/python tools/codex_landing_review_gate.py

  # Human/agent CLI:
  .venv/bin/python tools/codex_landing_review_gate.py status
  .venv/bin/python tools/codex_landing_review_gate.py disposition \
      --label optimal_metric_p0_build_surrogate_followons --stamp 20260714T123447Z \
      --status reviewed_committed --commit <sha> --reason "harvested 2 rows" \
      --consumed-by ".omx/research/optimal_metric_p0_surrogate_followons_DAG_FEED_20260714.md"
  .venv/bin/python tools/codex_landing_review_gate.py disposition \
      --label some_arm --stamp <stamp> --status held_entangled \
      --blocked-by arm_a,arm_b --reason "shares trainer.py with live arms"
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DELEGATIONS = REPO / ".omx" / "state" / "codex_delegations.jsonl"
LEDGER = REPO / ".omx" / "state" / "codex_landing_ledger.jsonl"
MARKER = REPO / ".omx" / "state" / "codex_landing_gate_marker.json"
LOG = REPO / ".omx" / "state" / "codex_landing_review_gate.log"

TERMINAL_STATES = {"reviewed_committed", "respawned", "closed"}
NONTERMINAL_STATES = {"held_entangled"}
VALID_STATES = TERMINAL_STATES | NONTERMINAL_STATES

# A landing younger than this (its .last.txt mtime) is still "fresh" — the hook
# nudges, but the STRICT preflight guard gives this grace before it REFUSES, so a
# just-landed arm main is actively harvesting does not trip a hard block.
STALE_GRACE_SECONDS = 900  # 15 minutes


def _utcnow() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _log(msg: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"{_utcnow()} {msg}\n")
    except Exception:
        pass


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping malformed lines. Missing ⇒ []."""
    rows: list[dict] = []
    try:
        if not path.is_file():
            return rows
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return rows
    return rows


def _arm_alive(label: str, stamp: str) -> bool:
    """A codex whose -o path contains ``<label>_<stamp>`` is this run's process.
    Mirrors tools/codex_status.py::_alive. pgrep-missing ⇒ treat as dead."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", f"{label}_{stamp}"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _landing_done(deleg: dict) -> bool:
    """A delegation has LANDED if its .done marker OR .last.txt final message exists."""
    for key in ("done_marker", "last"):
        p = deleg.get(key)
        if p and Path(p).is_file():
            return True
    return False


def _landing_mtime(deleg: dict) -> float | None:
    for key in ("last", "done_marker"):
        p = deleg.get(key)
        try:
            if p and Path(p).is_file():
                return Path(p).stat().st_mtime
        except Exception:
            continue
    return None


def latest_dispositions(ledger_rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Latest-row-wins disposition per (label, stamp)."""
    out: dict[tuple[str, str], dict] = {}
    for row in ledger_rows:
        label = row.get("label")
        stamp = row.get("stamp")
        if not label or not stamp:
            continue
        out[(label, stamp)] = row  # append-only ⇒ later rows overwrite
    return out


def compute_pending(
    delegations: list[dict],
    dispositions: dict[tuple[str, str], dict],
    alive_fn=_arm_alive,
) -> list[dict]:
    """Return the list of LANDED + DEAD delegations that lack a terminal
    disposition. A ``held_entangled`` disposition whose ``blocked_by`` arms are
    ALL now dead is RE-SURFACED as pending (held is a tracked queue, not a grave).

    Each returned dict: {label, stamp, mtime, reason}. ``reason`` ∈
    {"undispositioned", "held_entangled_unblocked"}.
    """
    pending: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for deleg in delegations:
        label = deleg.get("label")
        stamp = deleg.get("stamp")
        if not label or not stamp:
            continue
        key = (label, stamp)
        if key in seen:
            continue
        seen.add(key)
        if not _landing_done(deleg):
            continue  # still running / never launched
        if alive_fn(label, stamp):
            continue  # process still alive ⇒ held-by-design, not landed
        disp = dispositions.get(key)
        if disp is None:
            pending.append({
                "label": label, "stamp": stamp,
                "mtime": _landing_mtime(deleg), "reason": "undispositioned",
            })
            continue
        status = disp.get("status")
        if status in TERMINAL_STATES:
            continue  # done + reviewed
        if status == "held_entangled":
            blockers = disp.get("blocked_by") or []
            # re-surface if EVERY named blocker is now dead
            still_blocked = any(
                _blocker_alive(b, delegations, alive_fn) for b in blockers
            )
            if not still_blocked:
                pending.append({
                    "label": label, "stamp": stamp,
                    "mtime": _landing_mtime(deleg),
                    "reason": "held_entangled_unblocked",
                })
            continue
        # unknown status ⇒ treat as pending (fail-safe toward review)
        pending.append({
            "label": label, "stamp": stamp,
            "mtime": _landing_mtime(deleg), "reason": "undispositioned",
        })
    return pending


def _blocker_alive(blocker_label: str, delegations: list[dict], alive_fn) -> bool:
    """A named blocker arm is 'alive' if ANY delegation with that label has a live
    process. Blockers may be named by label only (latest stamp resolved here)."""
    for deleg in delegations:
        if deleg.get("label") == blocker_label:
            stamp = deleg.get("stamp", "")
            if stamp and alive_fn(blocker_label, stamp):
                return True
    return False


def pending_signature(pending: list[dict]) -> str:
    keys = sorted(f"{p['label']}::{p['stamp']}" for p in pending)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def append_disposition(row: dict) -> None:
    """fcntl-locked append to the disposition ledger (atomic, sister of the
    canonical JSONL-store discipline)."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("written_utc", _utcnow())
    with LEDGER.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------
def _cmd_disposition(args: argparse.Namespace) -> int:
    if args.status not in VALID_STATES:
        print(f"ERROR: --status must be one of {sorted(VALID_STATES)}", file=sys.stderr)
        return 2
    blocked_by = [b.strip() for b in (args.blocked_by or "").split(",") if b.strip()]
    if args.status == "held_entangled" and not blocked_by:
        print("ERROR: held_entangled requires --blocked-by <live-arm[,live-arm...]>",
              file=sys.stderr)
        return 2
    if args.status in {"closed"} and not args.reason:
        print("ERROR: closed requires --reason", file=sys.stderr)
        return 2
    # DISPOSITION != CONSUMPTION (operator 2026-07-17, memory
    # codex_findings_disposition_is_not_consumption_bug_class_20260717): a
    # "reviewed"/"closed" stamp without a NAMED CONSUMER let findings rot
    # unrouted (proven: the 4-arm curvelet/Fourier basis cluster). Terminal
    # dispositions now REQUIRE --consumed-by naming the artifact that absorbed
    # (or knowingly rejected, with a recorded decision) the arm's findings:
    # a task id (#NNN), P0 ledger row id, DAG FEED tag, spec section, DSL
    # lever, memo path, or commit sha. `none:<reason>` is the explicit
    # nothing-to-consume escape (pure-mechanical arms) — it must carry a real
    # reason, so silence is impossible. respawned/held_entangled are exempt
    # (custody continues elsewhere).
    consumed_by = getattr(args, "consumed_by", None)
    if args.status in {"reviewed_committed", "closed"} and not consumed_by:
        print(
            "ERROR: reviewed_committed/closed require --consumed-by "
            "<task#|p0-row|DAG-FEED|spec-section|lever|memo-path|commit-sha|none:<reason>> "
            "— a disposition stamp is custody, not consumption; name where the "
            "findings were routed (or none:<reason> if genuinely nothing to route).",
            file=sys.stderr)
        return 2
    if consumed_by and consumed_by.strip().lower() in {"none", "none:", "n/a", "tbd", "<reason>"}:
        print("ERROR: --consumed-by none requires a reason: none:<why nothing to route>",
              file=sys.stderr)
        return 2
    row = {
        "label": args.label, "stamp": args.stamp, "status": args.status,
        "commit": args.commit, "respawn": args.respawn,
        "reason": args.reason, "blocked_by": blocked_by,
        "consumed_by": consumed_by,
        "by": "cli",
    }
    append_disposition(row)
    print(f"OK dispositioned {args.label}_{args.stamp} -> {args.status}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    delegations = _read_jsonl(DELEGATIONS)
    dispositions = latest_dispositions(_read_jsonl(LEDGER))
    pending = compute_pending(delegations, dispositions)
    print(f"delegations={len(delegations)} dispositioned={len(dispositions)} "
          f"pending_review={len(pending)}")
    for p in pending:
        age = ""
        if p.get("mtime"):
            secs = datetime.datetime.now().timestamp() - p["mtime"]
            age = f" ({int(secs//60)}m ago)"
        flag = "!STALE" if _is_stale(p) else ""
        print(f"  PENDING-REVIEW {p['label']}_{p['stamp']} [{p['reason']}]{age} {flag}")
    if not pending:
        print("  (all landed codex arms dispositioned — no drift)")
    return 0


def _is_stale(p: dict) -> bool:
    mt = p.get("mtime")
    if not mt:
        return False
    return (datetime.datetime.now().timestamp() - mt) > STALE_GRACE_SECONDS


def _cmd_seed_baseline(args: argparse.Namespace) -> int:
    """Seed the ledger at gate-install time: mark every ALREADY-landed+dead arm
    older than ``--before`` as ``closed`` with a baseline reason. Landings that
    predate the gate are outside its jurisdiction (they were harvested in prior
    sessions per the commit log); this declares that HONESTLY rather than nagging
    about 100+ historical arms. Go-forward landings are tracked from scratch.
    Idempotent: already-dispositioned (label,stamp) rows are skipped."""
    try:
        cutoff = float(args.before)
    except (TypeError, ValueError):
        cutoff = datetime.datetime.fromisoformat(args.before).timestamp()
    delegations = _read_jsonl(DELEGATIONS)
    already = latest_dispositions(_read_jsonl(LEDGER))
    seeded = 0
    skipped_recent = 0
    for deleg in delegations:
        label, stamp = deleg.get("label"), deleg.get("stamp")
        if not label or not stamp:
            continue
        if (label, stamp) in already:
            continue
        if not _landing_done(deleg):
            continue
        if _arm_alive(label, stamp):
            continue
        mt = _landing_mtime(deleg)
        if mt is not None and mt >= cutoff:
            skipped_recent += 1
            continue  # inside the go-forward window ⇒ leave PENDING (real review)
        append_disposition({
            "label": label, "stamp": stamp, "status": "closed",
            "commit": None, "respawn": None,
            "reason": args.reason, "blocked_by": [], "by": "seed-baseline",
        })
        seeded += 1
    print(f"OK seed-baseline: closed {seeded} pre-gate landing(s); "
          f"left {skipped_recent} recent landing(s) PENDING for real review.")
    return 0


# ---------------------------------------------------------------------------
# Stop-hook mode
# ---------------------------------------------------------------------------
def _run_stop_hook() -> int:
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        inp = json.loads(raw) if raw.strip() else {}
    except Exception:
        inp = {}
    stop_hook_active = bool(inp.get("stop_hook_active"))

    try:
        delegations = _read_jsonl(DELEGATIONS)
        dispositions = latest_dispositions(_read_jsonl(LEDGER))
        pending = compute_pending(delegations, dispositions)
    except Exception as exc:  # FAIL-OPEN
        _log(f"allow(error) {type(exc).__name__}: {exc}")
        sys.exit(0)

    if not pending:
        # nothing landed-but-un-dispositioned ⇒ clear the marker + allow
        try:
            if MARKER.is_file():
                MARKER.unlink()
        except Exception:
            pass
        sys.exit(0)

    sig = pending_signature(pending)
    last_sig = None
    try:
        if MARKER.is_file():
            last_sig = json.loads(MARKER.read_text()).get("signature")
    except Exception:
        last_sig = None

    # LOOP-SAFE: one firm nudge per distinct pending-set. If Claude already
    # re-entered the stop (stop_hook_active) OR we already nudged THIS exact set,
    # allow the stop (the preflight guard is the hard enforcement, not this hook).
    if stop_hook_active or (last_sig and last_sig == sig):
        _log(f"allow(already-nudged) sig={sig} active={stop_hook_active} pending={len(pending)}")
        sys.exit(0)

    try:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(json.dumps({"signature": sig, "utc": _utcnow(),
                                      "pending": [f"{p['label']}_{p['stamp']}" for p in pending]}))
    except Exception:
        pass

    names = "\n".join(
        f"  • {p['label']}_{p['stamp']}"
        + (" [held-entangled arms now DEAD → review now]" if p["reason"] == "held_entangled_unblocked" else "")
        + ("  ⚠STALE >15m" if _is_stale(p) else "")
        for p in pending
    )
    reason = (
        f"CODEX LANDING REVIEW GATE: {len(pending)} landed codex arm(s) have NO "
        f"review disposition — this is a tracked source of drift / signal-loss / "
        f"confounding (operator 2026-07-14). Before ending the turn, for EACH:\n"
        f"{names}\n"
        f"REVIEW its landing (findings + working-tree edits) and record a disposition "
        f"via `tools/codex_landing_review_gate.py disposition --label <L> --stamp <S> "
        f"--status <...>`:\n"
        f"  reviewed_committed (--commit <sha>) · respawned (--respawn <label>) · "
        f"closed (--reason ...) · held_entangled (--blocked-by <live-arm,..> --reason ...).\n"
        f"held_entangled is the CORRECT state when a live arm still edits the shared "
        f"files — it auto-re-surfaces when those arms die (never a silent grave). "
        f"Immediate review + follow-on respawn is the policy; do NOT let a landing sit."
    )
    print(json.dumps({"decision": "block", "reason": reason}))
    _log(f"block sig={sig} pending={len(pending)}")
    sys.exit(0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"disposition", "status", "seed-baseline"}:
        # Stop-hook mode (Claude Code invokes with no args, JSON on stdin).
        _run_stop_hook()
        return 0

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("disposition", help="record a review disposition for a landed arm")
    d.add_argument("--label", required=True)
    d.add_argument("--stamp", required=True)
    d.add_argument("--status", required=True, choices=sorted(VALID_STATES))
    d.add_argument("--commit", default=None, help="commit sha (reviewed_committed)")
    d.add_argument("--respawn", default=None, help="follow-on arm label (respawned)")
    d.add_argument("--reason", default=None)
    d.add_argument("--blocked-by", default=None,
                   help="comma-separated LIVE arm labels (held_entangled)")
    d.add_argument("--consumed-by", default=None, dest="consumed_by",
                   help="REQUIRED for reviewed_committed/closed: the artifact that "
                        "absorbed the arm's findings (task#/P0-row/DAG-FEED/spec-section/"
                        "lever/memo-path/commit-sha) or none:<reason>. Disposition is "
                        "custody; this field is CONSUMPTION.")

    sub.add_parser("status", help="print landed-vs-dispositioned status")

    sb = sub.add_parser("seed-baseline",
                        help="close all pre-gate landings older than --before (install seed)")
    sb.add_argument("--before", required=True,
                    help="ISO utc or epoch seconds; landings older than this are baseline-closed")
    sb.add_argument("--reason",
                    default="pre-gate-install baseline; harvested in prior session(s) per commit log",
                    help="baseline closure reason")

    args = ap.parse_args(argv)
    if args.cmd == "disposition":
        return _cmd_disposition(args)
    if args.cmd == "status":
        return _cmd_status(args)
    if args.cmd == "seed-baseline":
        return _cmd_seed_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
