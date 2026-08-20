#!/usr/bin/env python3
"""consolidation_debt.py — proactive consolidation-debt monitor (operator 2026-07-15:
"We need regular and proactive consolidation to prevent that from happening again").

WHY: the 311-file drained-fleet pile + chat-only signal decay happened because consolidation
was REACTIVE (only when the pile exploded or the operator asked). This makes the debt VISIBLE
on a cadence so consolidation stays CONTINUOUS — the "commit early + often" discipline lifted
from single commits to the fleet + signal level. It READS existing durable state only (git +
the landing-review ledger); it is score-neutral observability (per the default-off/orphan rule:
read-only telemetry defaults ON, no safety reason to gate it). It NEVER auto-commits or launches
anything — it emits a debt report + a verdict; main/operator decide to consolidate.

DEBT COMPONENTS (each with a threshold; the verdict is the max severity):
  1. CODE PILE      — uncommitted files/lines in the working tree (the 311-file failure mode).
  2. LANDINGS       — codex arms that landed with NO review disposition (drift/confound source).
  3. STALE-SINCE    — commits since the last consolidation/reconciliation-tagged commit.
  4. SIGNAL RATIO   — recent research memos vs system-intelligence commits (canonical_equations
                      / witness_dsl), a COARSE heuristic for un-routed findings (labeled as such).
  5. SSD-ONLY CODE  — authored sources that exist on the SSD tier and NOWHERE in reachable git
                      history, so one disk loses them (ddm_oc2 measured 1,071 such blobs on
                      2026-08-20). Read from the cache `tools/audit_ssd_authored_signal.py
                      --write-cache` writes; a full byte-identity sweep is minutes of I/O and
                      must not run on every session boundary. A STALE cache is reported as its
                      own state — an old clean number is not a current clean number.

USAGE:
  .venv/bin/python tools/consolidation_debt.py            # human report + verdict
  .venv/bin/python tools/consolidation_debt.py --json     # machine-readable
  .venv/bin/python tools/consolidation_debt.py --quiet-ok # print nothing if verdict==OK (hook use)
Exit code: 0 always (observability, never blocks) unless --strict (then rc=2 on CONSOLIDATE-NOW).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LANDING_LEDGER = REPO / ".omx" / "state" / "codex_landing_ledger.jsonl"
TERMINAL_LANDING_STATES = frozenset({"reviewed_committed", "respawned", "closed"})

# thresholds (soon / now) — deliberately conservative so debt is caught EARLY, not at crisis.
TH = {
    "pile_files": (8, 25),        # uncommitted files
    "pile_lines": (400, 2000),    # uncommitted line churn
    "landings": (1, 3),           # undispositioned landed arms
    "stale_commits": (25, 60),    # commits since last consolidation-tagged commit
    "signal_ratio": (12.0, 30.0), # memos-per-system-intelligence-commit (24h) — HEURISTIC
    "ssd_only_code": (1, 50),     # authored blobs living on the SSD alone (0 is the target)
}
# How old the SSD sweep may be before its number stops counting as current. Chosen to match the
# cadence at which arms create SSD workspaces (daily), not an arbitrary round number.
SSD_CACHE_MAX_AGE_H = 72
SSD_CACHE = REPO / ".omx" / "state" / "ssd_authored_signal_debt.json"
_CONSOLIDATION_RE = r"consolidat|signal.reconciliation|reconcile|drain|signal-loss"


# Every git call is bounded (ddm_gh2, 2026-07-31). This hook is wired into BOTH
# SessionStart and Stop, and `git status` can block on `.git/index.lock` — which is
# a live, recurring event in this repo (concurrent arms, the auto_push_main Stop
# hook, a running trainer). An unbounded subprocess in a SessionStart hook is a
# session-hang hazard, so the call is bounded here and the hook registration
# declares an outer timeout too (belt and suspenders).
_GIT_TIMEOUT_S = 15
# Set when any git call fails or times out. Previously `_sh` swallowed rc and
# stderr, so a failed `git status` returned "" -> pile_files=0 -> verdict "OK":
# the monitor reported CLEAN precisely when it had gone BLIND. Per the CLAUDE.md
# confound discipline (L1: make silent artifacts LOUD), a blind monitor must say
# so rather than emit a false all-clear.
_DEGRADED: list[str] = []


def _sh(args: list[str]) -> str:
    """Run a git command, bounded. On ANY failure record the degradation and return
    "" — callers keep their shape, but the verdict is no longer allowed to read
    clean (see `compute`)."""
    try:
        r = subprocess.run(
            args, cwd=REPO, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        _DEGRADED.append(f"{' '.join(args[:3])}: timeout after {_GIT_TIMEOUT_S}s")
        return ""
    except OSError as exc:  # git missing / exec failure
        _DEGRADED.append(f"{' '.join(args[:3])}: {type(exc).__name__}")
        return ""
    if r.returncode != 0:
        _DEGRADED.append(f"{' '.join(args[:3])}: rc={r.returncode} {r.stderr.strip()[:80]}")
        return ""
    return r.stdout


def _pile() -> tuple[int, int]:
    files = [ln for ln in _sh(["git", "status", "--short"]).splitlines() if ln.strip()]
    churn = _sh(["git", "diff", "--shortstat"]) + _sh(["git", "diff", "--cached", "--shortstat"])
    lines = 0
    for tok in churn.replace(",", " ").split():
        if tok.isdigit():
            lines += int(tok)
    return len(files), lines


def _landing_rows(path: Path = LANDING_LEDGER) -> list[dict]:
    """Read the authoritative append-only landing ledger; malformed rows are ignored."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    rows: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _undispositioned_landings(path: Path = LANDING_LEDGER) -> int:
    """Count latest ledger entries without a terminal review disposition.

    The landing ledger is append-only and the landing-review gate uses latest-row-wins
    semantics per ``(label, stamp)``. Reading it directly avoids parsing human-facing
    Stop-hook output, whose formatting and loop-suppression are intentionally advisory.
    """
    latest: dict[tuple[object, object], dict] = {}
    for row in _landing_rows(path):
        label = row.get("label")
        stamp = row.get("stamp")
        if not label or not stamp:
            continue
        latest[(label, stamp)] = row
    return sum(
        row.get("status") not in TERMINAL_LANDING_STATES
        for row in latest.values()
    )


def _stale_commits() -> int:
    """Commits since the last consolidation/reconciliation-tagged commit (0 = just consolidated)."""
    log = _sh(["git", "log", "-200", "--format=%H %s"]).splitlines()
    import re
    for i, ln in enumerate(log):
        if re.search(_CONSOLIDATION_RE, ln, re.IGNORECASE):
            return i
    return len(log)


def _signal_ratio() -> tuple[int, int, float]:
    memos = _sh(["git", "log", "--since=24 hours ago", "--name-only", "--pretty=format:",
                 "--", ".omx/research/"]).splitlines()
    n_memos = len({m for m in memos if m.endswith(".md")})
    si = _sh(["git", "log", "--since=24 hours ago", "--oneline",
              "--", "src/tac/canonical_equations/", "src/tac/witness_dsl/"]).splitlines()
    n_si = len([x for x in si if x.strip()])
    ratio = (n_memos / n_si) if n_si else float(n_memos)
    return n_memos, n_si, ratio


def _ssd_only_code(path: Path | None = None) -> tuple[int, dict]:
    """Authored blobs living on the SSD alone, read from the sweep cache.

    Returns (count, detail). A missing or stale cache returns a count of 0 with a detail that says
    so, and `compute` records the degradation — because an absent measurement and a measured zero
    are not the same answer, and laundering the first into the second is the false all-clear this
    monitor already refuses on the git side.

    The cache location is resolved at CALL time; an early-bound default would ignore a redirected
    `SSD_CACHE` and silently report "no cache" against the wrong path.
    """
    path = path or SSD_CACHE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _DEGRADED.append(f"ssd sweep cache unreadable: {type(exc).__name__}")
        return 0, {"state": "no_cache",
                   "hint": "run tools/audit_ssd_authored_signal.py --write-cache"}
    stamp = str(raw.get("date_utc", ""))
    age_h = None
    try:
        from datetime import UTC, datetime

        age_h = (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds() / 3600.0
    except (TypeError, ValueError):
        pass
    detail = {
        "state": "fresh",
        "measured_at_utc": stamp,
        "age_hours": round(age_h, 1) if age_h is not None else None,
        "scanned": raw.get("scanned"),
        "gc_eligible": raw.get("owed_of_which_gc_eligible"),
        "certified_in_place": raw.get("certified_in_place"),
        "roots_absent": raw.get("roots_absent") or [],
    }
    if age_h is None or age_h > SSD_CACHE_MAX_AGE_H:
        detail["state"] = "stale"
        _DEGRADED.append(
            f"ssd sweep cache stale ({detail['age_hours']}h > {SSD_CACHE_MAX_AGE_H}h)")
    if detail["roots_absent"]:
        detail["state"] = "partial"
        _DEGRADED.append(f"ssd sweep was PARTIAL — unmounted: {detail['roots_absent']}")
    return int(raw.get("owed_authored_blobs") or 0), detail


def _sev(value: float, soon: float, now: float) -> int:
    return 2 if value >= now else 1 if value >= soon else 0


def compute() -> dict:
    _DEGRADED.clear()
    pf, pl = _pile()
    land = _undispositioned_landings()
    stale = _stale_commits()
    n_memos, n_si, ratio = _signal_ratio()
    ssd_only, ssd_detail = _ssd_only_code()
    comps = {
        "pile_files": (pf, _sev(pf, *TH["pile_files"])),
        "pile_lines": (pl, _sev(pl, *TH["pile_lines"])),
        "landings": (land, _sev(land, *TH["landings"])),
        "stale_commits": (stale, _sev(stale, *TH["stale_commits"])),
        "signal_ratio": (round(ratio, 1), _sev(ratio, *TH["signal_ratio"])),
        "ssd_only_code": (ssd_only, _sev(ssd_only, *TH["ssd_only_code"])),
    }
    sev = max(s for _, s in comps.values())
    verdict = ("CONSOLIDATE-NOW" if sev == 2 else "CONSOLIDATE-SOON" if sev == 1 else "OK")
    degraded = list(_DEGRADED)
    if degraded and verdict == "OK":
        # A clean read and a blind read are NOT the same answer. Never launder the
        # second into the first: an OK produced from missing inputs is a false
        # all-clear, which is worse than no report.
        verdict = "DEGRADED"
    return {"verdict": verdict, "severity": sev, "components": comps, "degraded": degraded,
            "signal_detail": {"memos_24h": n_memos, "system_intelligence_commits_24h": n_si},
            "ssd_detail": ssd_detail}


def _fmt(r: dict) -> str:
    icon = {0: "✓", 1: "△", 2: "✕"}
    lines = [f"[consolidation-debt] {r['verdict']}  (proactive monitor — read-only, never blocks)"]
    if r.get("degraded"):
        lines.append(f"  ! BLIND on {len(r['degraded'])} git read(s) — components below "
                     f"may under-report: {'; '.join(r['degraded'][:2])}")
    for k, (v, s) in r["components"].items():
        if s or k in ("pile_files", "landings", "ssd_only_code"):
            soon, now = TH[k]
            lines.append(f"  {icon[s]} {k}: {v}  (soon>={soon} now>={now})")
    d = r.get("ssd_detail")
    ssd = r["components"].get("ssd_only_code")
    if d and ssd:
        if d.get("state") != "fresh":
            lines.append(f"    ssd sweep {d.get('state', '?')} "
                         f"({d.get('measured_at_utc') or 'never'}) — the SSD number above is NOT "
                         f"current; refresh with tools/audit_ssd_authored_signal.py --write-cache")
        elif ssd[1]:
            lines.append(f"    authored code living on the SSD alone: {ssd[0]} blobs "
                         f"({d.get('gc_eligible')} gc-eligible) measured {d.get('age_hours')}h ago "
                         f"over {d.get('scanned'):,} files — commit them or certify with a rationale.")
    # `.get`, not `[...]`: a caller (or an older cached report) may carry a partial component set,
    # and a formatter that raises turns an advisory monitor into a session-boundary crash.
    if (r["components"].get("signal_ratio") or (0, 0))[1]:
        d = r["signal_detail"]
        lines.append(f"    signal-ratio HEURISTIC: {d['memos_24h']} research memos vs "
                     f"{d['system_intelligence_commits_24h']} canonical_equations/DSL commits (24h) "
                     f"— possible un-routed findings; verify via signal-loss audit, do not assume loss.")
    if r["severity"]:
        lines.append("  → CONSOLIDATE: commit clean artifacts · disposition landed arms · "
                     "route findings to canonical_equations/DSL/tasks/memory · then re-run.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Proactive consolidation-debt monitor (read-only).")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet-ok", action="store_true", help="print nothing if verdict==OK")
    ap.add_argument("--strict", action="store_true", help="rc=2 when verdict==CONSOLIDATE-NOW")
    args = ap.parse_args(argv)
    try:
        r = compute()
    except Exception as exc:
        # FAIL-OPEN (ddm_gh2, 2026-07-31). This hook is wired into SessionStart AND
        # Stop; before this guard ANY exception (git absent, decode error, ledger
        # corruption) escaped as a traceback on a nonzero exit at a session boundary.
        # An observability monitor must never wedge or noise-up a session — but it
        # must still SAY it failed rather than pretend it passed.
        print(f"[consolidation-debt] DEGRADED — monitor unavailable "
              f"({type(exc).__name__}: {str(exc)[:120]}); treat debt as UNKNOWN, not OK")
        return 0
    if args.json:
        print(json.dumps(r, indent=2))
    elif not (args.quiet_ok and r["verdict"] == "OK"):
        print(_fmt(r))
    return 2 if (args.strict and r["severity"] == 2) else 0


if __name__ == "__main__":
    raise SystemExit(main())
