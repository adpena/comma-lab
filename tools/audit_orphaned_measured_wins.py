#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit `.omx/research/*.md` findings memos for ORPHANED MEASURED WINS.

The bug class (operator 2026-07-02, "not wiring Track B's store-nothing
win was CLASSIC ORPHANED SIGNAL and a related meta-bug"): a MEASURED
n600 mechanism-level win — a real result that would lower the contest
score S, OR a better mechanism than the current launch config uses —
that lands as a ledger/research memo but is NOT (a) wired into the
launchable vehicle (trainer mode + launch config) NOR (b) propagated to
all three triality legs (DSL gauge + canonical_equation + DAG) NOR
(c) explicitly tagged `research_only` + `reactivation_criteria`, is
ORPHANED SIGNAL. The launch config then silently ships an inferior
un-integrated mechanism.

This tool is the OPERATOR-FACING retroactive sweep (task #225) + the
burndown surface for the sister STRICT preflight gate Catalog #396
(`check_measured_win_findings_are_wired_or_research_only`). It uses the
ONE canonical classifier `tac.preflight.classify_findings_memo_orphan_
status` so the audit and the gate never drift.

Verdicts (per memo):
- WIRED         -- canonical_equation reference AND a launch-config/DSL/
  wiring pointer both present (formalized AND integrated).
- RESEARCH_ONLY -- `research_only` AND `reactivation` criteria present
  (honestly parked with a reactivation path).
- WAIVED        -- a valid `# ORPHAN_WIN_WAIVED:<rationale>` waiver.
- ORPHAN        -- a MEASURED mechanism win with none of the above.
  These are the burndown targets: wire into vehicle+config + all 3
  triality legs, OR tag research_only+reactivation.
- (NOT_A_WIN memos are silently skipped — no measured mechanism-win
  claim detected.)

The gate (Catalog #396) is forward-looking (memos dated >= 2026-07-02,
the naming day); this tool scans a WIDER window (default >= 2026-06-01)
so the operator sees the recent backlog for burndown.

The magnitude/ease ranking is a HEURISTIC prioritization aid (count of
scored-term win mentions + explicit numeric deltas vs a landed-tool
proxy for ease); it is NOT a measured ΔS. Rank ORPHANs manually against
the ledger for the load-bearing prioritization.

Usage:
    .venv/bin/python tools/audit_orphaned_measured_wins.py
    .venv/bin/python tools/audit_orphaned_measured_wins.py --json
    .venv/bin/python tools/audit_orphaned_measured_wins.py --since 20260601
    .venv/bin/python tools/audit_orphaned_measured_wins.py --orphans-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Canonical classifier + verdict constants — ONE source of truth (the gate
# and this audit tool share them so they never drift).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.preflight import (  # noqa: E402
    ORPHAN_WIN_STATUS_NOT_A_WIN,
    ORPHAN_WIN_STATUS_ORPHAN,
    classify_findings_memo_orphan_status,
)

_RESEARCH_RELPATH = ".omx/research"
_DEFAULT_SINCE = 20260601  # wider than the gate cutoff (20260702) for burndown
_DESIGN_FILENAME_RE = re.compile(r".*_(\d{8})(?:T\d{6}Z)?(?:_codex)?\.md$")

# Heuristic magnitude signals (proxy for |measured ΔS|). Keyed off
# win-signal DENSITY (explicit reductions/deltas + strong-win phrases),
# NOT generic scored-term mentions (which saturate on long memos).
_STRONG_WIN_PHRASES = ("pareto-domin", "beats ", "better than", " wins", "huge win")
_PERCENT_RE = re.compile(r"[-−]\s?\d+(?:\.\d+)?\s?%")
_NUM_DELTA_RE = re.compile(r"[-−]?\d+\.\d{2,}")

# Heuristic ease-of-wire signals: reference to a LANDED tool/mode/plumbing
# lowers the effort; "new build" / "stub" / "NotImplemented" raises it.
_LANDED_TOKENS = (
    "landed",
    "default-on",
    "default on",
    "bit-exact",
    "byte-close",
    "tools/levelset_byte_close_and_eval",
    "compose_witness_archive",
    "wire_in",
)
_NEW_BUILD_TOKENS = (
    "new build",
    "notimplemented",
    "stub",
    "pending",
    "needs the #205",
    "needs a wiring build",
)


@dataclass
class OrphanAuditRow:
    path: str
    date_int: int
    status: str
    magnitude_score: int
    ease_score: int
    priority_score: float
    win_snippet: str


def _magnitude_score(text: str) -> int:
    """Heuristic |ΔS| proxy: win-signal density = strong-win phrases +
    explicit percent reductions + explicit multi-decimal deltas (capped).
    NOT a measured ΔS — a prioritization aid only."""
    low = text.lower()
    strong = sum(low.count(p) for p in _STRONG_WIN_PHRASES)
    pct = len(_PERCENT_RE.findall(text))
    deltas = len(_NUM_DELTA_RE.findall(text))
    return min(3 * strong + 4 * pct + deltas, 100)


def _ease_score(text: str) -> int:
    """Heuristic ease-of-wire: higher = easier to wire NOW. Landed plumbing
    raises it; new-build/stub signals lower it. Range ~[1, 10]."""
    low = text.lower()
    score = 5
    score += sum(2 for tok in _LANDED_TOKENS if tok in low)
    score -= sum(2 for tok in _NEW_BUILD_TOKENS if tok in low)
    return max(1, min(score, 10))


def _win_snippet(text: str) -> str:
    """First line containing a win/measured signal, trimmed."""
    for line in text.splitlines():
        low = line.lower()
        if ("wins" in low or "better than" in low or "beats" in low
                or _PERCENT_RE.search(line)) and (
                "measured" in low or "n600" in low or "d_pose" in low
                or "d_seg" in low or "rate" in low):
            return line.strip()[:200]
    for line in text.splitlines():
        low = line.lower()
        if "wins" in low or "better than" in low or "beats" in low:
            return line.strip()[:200]
    return ""


def scan(repo_root: Path, since: int) -> list[OrphanAuditRow]:
    research = repo_root / _RESEARCH_RELPATH
    rows: list[OrphanAuditRow] = []
    if not research.is_dir():
        return rows
    for entry in sorted(research.iterdir()):
        if not entry.is_file() or not entry.name.endswith(".md"):
            continue
        m = _DESIGN_FILENAME_RE.match(entry.name)
        if not m:
            continue
        try:
            date_int = int(m.group(1))
        except (TypeError, ValueError):
            continue
        if date_int < since:
            continue
        try:
            text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        status = classify_findings_memo_orphan_status(text)
        if status == ORPHAN_WIN_STATUS_NOT_A_WIN:
            continue
        mag = _magnitude_score(text)
        ease = _ease_score(text)
        rows.append(
            OrphanAuditRow(
                path=str(entry.relative_to(repo_root)),
                date_int=date_int,
                status=status,
                magnitude_score=mag,
                ease_score=ease,
                priority_score=round(mag * ease / 10.0, 2),
                win_snippet=_win_snippet(text),
            )
        )
    # Orphans first, then by priority (|ΔS| proxy * ease), then recency.
    rows.sort(
        key=lambda r: (
            r.status != ORPHAN_WIN_STATUS_ORPHAN,
            -r.priority_score,
            -r.date_int,
        )
    )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(_REPO_ROOT))
    ap.add_argument(
        "--since", type=int, default=_DEFAULT_SINCE,
        help="only scan memos dated >= this YYYYMMDD (default 20260601)",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--orphans-only", action="store_true",
        help="only print ORPHAN-verdict rows (the burndown list)",
    )
    args = ap.parse_args()

    rows = scan(Path(args.repo_root).resolve(), args.since)
    if args.orphans_only:
        rows = [r for r in rows if r.status == ORPHAN_WIN_STATUS_ORPHAN]

    if args.json:
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.status] = counts.get(r.status, 0) + 1
        print(json.dumps({
            "since": args.since,
            "total": len(rows),
            "counts": counts,
            "rows": [asdict(r) for r in rows],
        }, indent=2))
        return 0

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"Orphaned-measured-win audit (memos dated >= {args.since})")
    print(f"  measured-win memos: {len(rows)}  |  " + "  ".join(
        f"{k}={v}" for k, v in sorted(counts.items())))
    print(
        "  ranking is a HEURISTIC prioritization aid (|ΔS| proxy * ease), "
        "NOT a measured ΔS.\n"
    )
    for r in rows:
        flag = "  ORPHAN ->" if r.status == ORPHAN_WIN_STATUS_ORPHAN else "         "
        print(f"{flag} [{r.status:<13}] prio={r.priority_score:<6} "
              f"(mag={r.magnitude_score} ease={r.ease_score}) {r.path}")
        if r.win_snippet:
            print(f"           win: {r.win_snippet}")
    if not rows:
        print("  (no measured-win memos in window)")
    n_orphan = counts.get(ORPHAN_WIN_STATUS_ORPHAN, 0)
    print(
        f"\n  {n_orphan} ORPHAN(s) to burn down: wire into vehicle+config + "
        "all 3 triality legs (DSL gauge + canonical_equation + DAG), OR tag "
        "research_only + reactivation_criteria. Sister gate: Catalog #396."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
