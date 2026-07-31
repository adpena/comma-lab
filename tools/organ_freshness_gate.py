#!/usr/bin/env python3
"""organ_freshness_gate.py — event-driven costate-organ rounds (co8 rung 6, operator
2026-07-28: "Perhaps we need to add a gate or hook so that we update the costate organ
whenever necessary or optimal").

WHY: eight consecutive organ rounds (co2..co8) were OPERATOR-driven — the recurring
"Continue enhancing the organ" directive is the default-off/orphan pattern applied to the
organ's own freshness (the apparatus, not the operator, should hold this memory). This
detector makes organ rounds EVENT-DRIVEN: it diffs MERGED campaign landings on main
(`.omx/research/ddm_*.md` memos, by git first-add time) against the organ's
consumed-evidence registry (derived LIVE from the module constants the organ actually
reads — `tac.ddm_costate_organ.consumed_evidence_registry()`, never a hand-maintained
list; NO-FAKE) and surfaces "ORGAN ROUND DUE: [named rows]" when either:

  1. GENERIC BACKLOG — >= N unconsumed landings since the last organ round, where N is
     DERIVED from the measured per-round consumption (half the mean landings per
     inter-round window, floored at 2 — conservative/early per the consolidation_debt
     precedent), never hardcoded; OR
  2. RE-ROUTING CLASS — ANY unconsumed landing carrying re-routing markers
     (VOI / precondition-trigger / reactivation / frontmatter verdict) sits unconsumed
     at all (threshold 1: such a landing can re-route the sealed burn).

WARN-ONLY Stop-hook (triality-drift-detector precedent): fail-open, never blocking —
exit 0 always (rc=2 only under explicit --strict, for tests). Read-only observability;
score-neutral; actuation NONE.

USAGE:
  .venv/bin/python tools/organ_freshness_gate.py            # human report
  .venv/bin/python tools/organ_freshness_gate.py --json     # machine-readable
  .venv/bin/python tools/organ_freshness_gate.py --quiet-ok # silent unless DUE (hook)
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RESEARCH_PREFIX = ".omx/research/"
ROUND_RE = re.compile(r"^ddm_co(\d+)[_.]")
# Content markers for the re-routing class (operator: "VOI/precondition-trigger/verdict").
REROUTING_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("frontmatter_verdict", re.compile(r"^verdict\s*:", re.MULTILINE)),
    ("voi", re.compile(r"\bVOI\b")),
    ("reactivation", re.compile(r"reactivat", re.IGNORECASE)),
    ("precondition_trigger", re.compile(r"precondition", re.IGNORECASE)),
    ("rerouting", re.compile(r"re-?rout", re.IGNORECASE)),
)
THRESHOLD_FLOOR = 2  # minimum N; the derived value never drops below this
# Stop-hook output budget (ddm_gh2, 2026-07-31). MEASURED before this change: the
# DUE branch emitted ONE 3,471-byte line — a raw join of every unconsumed memo path
# — on EVERY Stop, i.e. every turn boundary, unchanged until the debt is consumed.
# Naming the rows is the point of the gate, so rows are still NAMED; what is bounded
# is HOW MANY are named inline. The count is always exact and the residue is always
# disclosed (never a silent truncation), with --json carrying the complete list.
_NAMED_ROWS_IN_HOOK = 4


def _consumed_registry() -> dict[str, Any]:
    """Import the organ's LIVE consumed-evidence registry (module-constant derived)."""

    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tac.ddm_costate_organ import consumed_evidence_registry

    return consumed_evidence_registry()


def _git_add_times(repo: Path = REPO) -> dict[str, int]:
    """First-add (oldest --diff-filter=A) commit time per .omx/research file, ONE git call."""

    out = subprocess.run(
        [
            "git",
            "log",
            "--diff-filter=A",
            "--format=C %ct",
            "--name-only",
            "--",
            ".omx/research",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git log failed: {out.stderr.strip()[:200]}")
    add_ct: dict[str, int] = {}
    current = 0
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("C ") and line[2:].isdigit():
            current = int(line[2:])
            continue
        # newest-first: keep overwriting so the OLDEST (true add) commit wins.
        add_ct[line] = current
    return add_ct


def _is_candidate_landing(rel: str) -> bool:
    """A top-level ddm_* markdown memo that is not an organ round or a DAG FEED rider."""

    if not rel.startswith(RESEARCH_PREFIX):
        return False
    name = rel[len(RESEARCH_PREFIX) :]
    if "/" in name or not name.endswith(".md") or not name.startswith("ddm_"):
        return False
    if ROUND_RE.match(name):
        return False  # the organ's own round memos are not landings to consume
    # DAG FEED files ride their parent memo — the parent is the consumable landing.
    return "_DAG_FEED_" not in name


def _is_consumed(rel: str, registry: dict[str, Any]) -> bool:
    if rel in set(registry["paths"]):
        return True
    return any(fnmatch.fnmatch(rel, pattern) for pattern in registry["globs"])


def _rerouting_markers(path: Path) -> list[str]:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    except OSError:
        return []
    return [name for name, pattern in REROUTING_MARKERS if pattern.search(head)]


def derive_threshold(round_cts: list[int], candidate_cts: list[int]) -> dict[str, Any]:
    """N derived from MEASURED per-round consumption: half the mean landings per
    inter-round window, floored at THRESHOLD_FLOOR (early-catch, conservative)."""

    # A round lands several marker files in one commit cluster; dedupe commit times so
    # zero-width windows do not drag the measured per-round mean toward zero.
    rounds = sorted(set(round_cts))
    windows: list[int] = []
    for lo, hi in pairwise(rounds):
        windows.append(sum(1 for ct in candidate_cts if lo < ct <= hi))
    if not windows:
        return {
            "n": THRESHOLD_FLOOR,
            "basis": "FLOOR_DEFAULT_INSUFFICIENT_ROUND_HISTORY",
            "mean_landings_per_round": None,
            "windows": 0,
        }
    mean = sum(windows) / len(windows)
    return {
        "n": max(THRESHOLD_FLOOR, math.ceil(mean / 2.0)),
        "basis": (
            "half the MEASURED mean landings per inter-round window (catch early), "
            f"floored at {THRESHOLD_FLOOR}"
        ),
        "mean_landings_per_round": round(mean, 2),
        "windows": len(windows),
    }


def evaluate(repo: Path = REPO) -> dict[str, Any]:
    registry = _consumed_registry()
    add_ct = _git_add_times(repo)
    rounds = {
        rel: ct
        for rel, ct in add_ct.items()
        if rel.startswith(RESEARCH_PREFIX) and ROUND_RE.match(rel[len(RESEARCH_PREFIX) :])
    }
    candidates = {rel: ct for rel, ct in add_ct.items() if _is_candidate_landing(rel)}
    if rounds:
        last_round_rel, last_round_ct = max(rounds.items(), key=lambda kv: kv[1])
    else:
        last_round_rel, last_round_ct = None, 0
    unconsumed = sorted(
        rel
        for rel, ct in candidates.items()
        if ct > last_round_ct and not _is_consumed(rel, registry)
    )
    rerouting = {
        rel: markers
        for rel in unconsumed
        if (markers := _rerouting_markers(repo / rel))
    }
    threshold = derive_threshold(list(rounds.values()), list(candidates.values()))
    due_generic = len(unconsumed) >= threshold["n"]
    due_rerouting = bool(rerouting)
    reasons: list[str] = []
    if due_rerouting:
        # COUNT only. This used to interpolate `sorted(rerouting)` — a raw list repr
        # of full paths — straight into the reason string, which is what made the
        # Stop-hook line 3,471 B (MEASURED 2026-07-31). The names are not lost: they
        # are carried structurally in the "rerouting" field below (so --json is
        # complete) and rendered, capped, by the human formatter.
        reasons.append(
            f"{len(rerouting)} re-routing-class landing(s) unconsumed (threshold 1)"
        )
    if due_generic:
        reasons.append(
            f"{len(unconsumed)} unconsumed landings >= derived N={threshold['n']}"
        )
    return {
        "schema": "ddm_organ_freshness_gate.v1",
        "due": due_generic or due_rerouting,
        "reasons": reasons,
        "rerouting": sorted(rerouting),
        "unconsumed": unconsumed,
        "rerouting_class": rerouting,
        "threshold": threshold,
        "last_round": {"path": last_round_rel, "ct": last_round_ct},
        "consumed_registry": {
            "paths": len(registry["paths"]),
            "globs": len(registry["globs"]),
        },
        "actuation": "NONE",
        "score_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet-ok", action="store_true", help="print nothing unless DUE (hook)")
    ap.add_argument("--strict", action="store_true", help="rc=2 when DUE (tests only)")
    args = ap.parse_args(argv)
    try:
        report = evaluate()
    except Exception as exc:  # fail-open: a broken detector must never block a Stop
        if not args.quiet_ok:
            print(f"organ-freshness-gate: fail-open ({type(exc).__name__}: {exc})")
        return 0
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["due"]:
        rows = list(report["unconsumed"])
        # Paths are long and repetitive (".omx/research/ddm_*.md"); the basename is
        # the identifying part, so name basenames and keep the directory implicit.
        shown = [Path(p).name for p in rows[:_NAMED_ROWS_IN_HOOK]]
        named = ", ".join(shown) or "(none named)"
        if len(rows) > len(shown):
            named += f", +{len(rows) - len(shown)} more (full list: --json)"
        print(
            "ORGAN ROUND DUE: "
            + "; ".join(report["reasons"])
            + f" — {len(rows)} unconsumed in .omx/research/: {named}"
            " (WARN-only; advisory; actuation NONE)"
        )
    elif not args.quiet_ok:
        th = report["threshold"]
        print(
            f"organ-freshness: OK — unconsumed={len(report['unconsumed'])} < N={th['n']} "
            f"(basis: {th['basis']}); last round={report['last_round']['path']}"
        )
    if args.strict and report["due"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
