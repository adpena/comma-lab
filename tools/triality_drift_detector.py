#!/usr/bin/env python3
"""Triality drift detector — a Claude Code ``Stop`` hook.

Fires at the end of each main-agent turn. Detects the **velocity-orphaning**
failure mode (our own "deepest signal-loss meta-bug"): a SUBSTANTIVE commit
landed this turn — a measured row / launch / build / lever / byte-close /
d_seg-d_pose / pointer move — WITHOUT a corresponding touch to the triality
(a DAG FEED / the DSL / the canonical equations / the research index). When it
finds that, it injects ONE firm nudge so the trajectory point gets recorded
before the turn truly ends.

This is the STRUCTURAL BACKSTOP for the triality-consistency discipline; it does
not replace proactive recording. It should rarely fire — a turn that already
touched a triality leg, or that landed nothing substantive, passes silently.

Design invariants (all NON-NEGOTIABLE for a Stop hook):
  * FAIL-OPEN — any error, non-git dir, or timeout ⇒ allow the stop (exit 0).
    A Stop hook must NEVER wedge a session.
  * LOOP-SAFE — one firm nudge per drift state, guarded by BOTH Claude Code's
    own ``stop_hook_active`` flag AND a persisted ``last_block_head`` backstop,
    so it cannot loop even if ``stop_hook_active`` is absent.
  * EVENT-TRIGGERED, not per-turn — silent when there are no new commits since
    the last stop, and silent when this window already touched a triality leg.
  * ESCAPE VALVE ("never binary") — a commit whose message contains
    ``[no-triality]`` (or ``[skip-drift]``) is a deliberate chore/apparatus
    commit and does not count as drift. Not every substantive keyword forces a
    DAG feed.

Wired via ``.claude/settings.json`` ``hooks.Stop``. Pure logic lives in
module-level functions (``classify`` et al.) so it is unit-testable without a
crafted git HEAD. Not placed in ``tac`` — this is Claude-workflow apparatus, not
contest/codec logic (per CLAUDE.md ``tac`` cleanliness rule).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys

# --- state files (both gitignored under .omx/state/*) ---
MARKER = ".omx/state/triality_drift_marker.json"
LOG = ".omx/state/triality_drift_detector.log"

# Commit-subject signatures of a score/trajectory/build event that WARRANTS a
# triality touch. Inclusive-but-focused; the [no-triality] escape valve + the
# triality-touch check keep false positives cheap (worst case: add a DAG line).
# Stems end in \w* (not a trailing \b) so prefix-stems like "clos" match the
# whole word ("byte-close"); the leading \b + \w* keep matches word-anchored
# (e.g. "\blever\w*" hits "lever/levers" but NOT "clever"/"level").
SUBSTANTIVE = re.compile(
    r"\b(?:"
    r"measur\w*|byte.?clos\w*|exact.?eval\w*|exact\s+row|d_seg|d_pose|pointer\w*|"
    r"launch\w*|lever\w*|verdict\w*|carrier\w*|coder\w*|kernel\w*|attribution\w*|"
    r"witness\w*|wire.?in\w*|integrat\w*|rate.?term\w*|score.?row\w*|scored|"
    r"probe\w*|erasure\w*|island\w*|n600|frontier\w*|curriculum\w*"
    r")",
    re.IGNORECASE,
)

# PER-LEG requirement (the strengthening, 2026-07-06): the old gate accepted ANY
# triality leg, so a lever commit touching ONLY the DAG passed while the DSL +
# equations silently drifted — the exact failure mode ("you've never touched the
# DSL or equations unforced"). Now a change of a given TYPE must touch the leg
# that TYPE lives in. Calibrated firm-not-noisy: broad stems (bare d_seg / witness
# / probe) stay in SUBSTANTIVE (the any-leg fallback) but do NOT force a specific
# leg; only the control/law signatures below do.
#   CONTROL change  → must touch the DSL   (the config-generator; a new/changed lever)
DSL_REQUIRING = re.compile(
    r"\b(?:lever\w*|wire.?in\w*|integrat\w*|launch\w*|curriculum\w*|carrier\w*|gauge\w*)",
    re.IGNORECASE,
)
#   LAW / measured finding → must register/refine a canonical equation
EQUATION_REQUIRING = re.compile(
    r"\b(?:measur\w*|byte.?clos\w*|exact.?eval\w*|exact\s+row|verdict\w*|pointer\w*|"
    r"scored|attribution\w*|erasure\w*|refut\w*|ratif\w*|\bfloor\b|\blaw\b)",
    re.IGNORECASE,
)

# In-repo surfaces whose modification means "a triality leg was updated this
# window" (DAG = trajectory · DSL = control · equations = law · index = memory
# proxy). MEMORY.md lives outside the repo (~/.claude/...) so it cannot be
# git-detected; the DAG is the primary in-repo trajectory leg and is sufficient.
TRIALITY_PREFIXES = (
    ".omx/research/sub015_DAG_",
    "src/tac/witness_dsl/",
    "src/tac/canonical_equations",
    "docs/triality_dag_dsl_equations",
    ".omx/research/CANONICAL_RESEARCH_INDEX",
)

SKIP_TOKEN = re.compile(r"\[(no-triality|skip-drift)\]", re.IGNORECASE)


# ----------------------------- pure logic (tested) -----------------------------
def is_substantive(subjects: list[str]) -> bool:
    """True if any commit subject names a score/trajectory/build event."""
    return any(SUBSTANTIVE.search(s or "") for s in subjects)


def is_opted_out(subjects: list[str]) -> bool:
    """True if any commit in the window opts out via [no-triality]/[skip-drift]."""
    return any(SKIP_TOKEN.search(s or "") for s in subjects)


def has_triality_touch(files: list[str]) -> bool:
    """True if any changed file is a triality leg (DAG/DSL/equations/index)."""
    return any((f or "").startswith(TRIALITY_PREFIXES) for f in files)


def touched_dsl(files: list[str]) -> bool:
    """True if the DSL leg (the config-generator) was updated this window."""
    return any((f or "").startswith("src/tac/witness_dsl/") for f in files)


def touched_equations(files: list[str]) -> bool:
    """True if the canonical-equations leg (the law) was updated this window."""
    return any((f or "").startswith("src/tac/canonical_equations") for f in files)


def missing_legs(subjects: list[str], files: list[str]) -> list[str]:
    """The SPECIFIC triality legs a change of this type REQUIRES but did not touch.

    A control change (lever/wire-in/launch/curriculum/carrier/gauge) must touch the
    DSL; a measured-law change (measure/byte-close/verdict/exact-row/…) must touch the
    canonical equations. Returns the leg names that are required-but-absent (empty ==
    every required leg was touched)."""
    joined = " ".join(s or "" for s in subjects)
    miss: list[str] = []
    if DSL_REQUIRING.search(joined) and not touched_dsl(files):
        miss.append("DSL")
    if EQUATION_REQUIRING.search(joined) and not touched_equations(files):
        miss.append("equations")
    return miss


def classify(subjects: list[str], files: list[str]) -> str:
    """Return ``"drift"`` iff a required leg is absent (per-leg) OR a substantive commit
    touched NO leg at all; else ``"clean"``. Opt-out via [no-triality]/[skip-drift]."""
    if not subjects:
        return "clean"
    if is_opted_out(subjects):
        return "clean"
    # PER-LEG (the teeth): a control/law change that skipped its own leg is drift even
    # if it touched a DIFFERENT leg (e.g. a lever commit that touched only the DAG).
    if missing_legs(subjects, files):
        return "drift"
    # FALLBACK (the original any-leg net): substantive but touched no leg at all.
    if is_substantive(subjects) and not has_triality_touch(files):
        return "drift"
    return "clean"


_LEG_FIX = {
    "DSL": ("the DSL (src/tac/witness_dsl/) — a lever/launch/curriculum change must "
            "add or update its Lever/WitnessProgram there (the config-generator), not "
            "just a trainer flag"),
    "equations": ("the canonical equations (src/tac/canonical_equations/) — a measured "
                  "finding/verdict/byte-close/exact-row must register or refine an "
                  "EmpiricalAnchor there, so it is never re-derived"),
}


def build_reason(subjects: list[str], files: list[str] | None = None) -> str:
    """The one firm nudge injected on drift (concise + actionable + leg-specific)."""
    preview = "; ".join((s or "")[:70] for s in subjects[:3])
    miss = missing_legs(subjects, files or [])
    if miss:
        legs = " AND ".join(_LEG_FIX.get(m, m) for m in miss)
        return (
            "Triality drift-detector (per-leg): "
            f"{len(subjects)} commit(s) landed this turn that REQUIRE {legs}. "
            f"Commit(s): {preview}. Update the named leg(s) — plus a DAG FEED "
            "(.omx/research/sub015_DAG_*) and a MEMORY.md line for a durable "
            "finding — then finish. Pure chore/apparatus can opt out with "
            "[no-triality] in the commit message. This gate now enforces the "
            "RIGHT leg per change-type, not just any leg — so the DSL + equations "
            "can no longer silently drift while only the DAG is touched."
        )
    return (
        "Triality drift-detector: "
        f"{len(subjects)} commit(s) landed this turn without touching the "
        "DAG / DSL / equations / research-index. Substantive commit(s): "
        f"{preview}. Append a DAG FEED (.omx/research/sub015_DAG_*) recording "
        "this trajectory point (and a MEMORY.md line if it is a durable "
        "finding), then finish. If these commits are pure chore/apparatus that "
        "do not warrant a trajectory row, that is fine — note it and stop (add "
        "[no-triality] to such commit messages to skip this check next time). "
        "This is the structural backstop for the triality-consistency "
        "discipline; record proactively next turn so it need not fire."
    )


# ------------------------------- IO / glue -------------------------------------
def _git(root: str, *args: str, timeout: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, timeout=timeout
    )


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _log(root: str, msg: str) -> None:
    try:
        p = os.path.join(root, LOG)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as f:
            f.write(f"{_now()} {msg}\n")
    except Exception:
        pass


def _read_marker(root: str) -> dict:
    try:
        with open(os.path.join(root, MARKER)) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_marker(root: str, data: dict) -> None:
    try:
        p = os.path.join(root, MARKER)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except Exception:
        pass


def _advance_and_allow(root: str, head: str) -> None:
    """Advance the marker to HEAD, clear block state, allow the stop."""
    _write_marker(root, {"last_head": head, "last_block_head": None, "updated_utc": _now()})
    sys.exit(0)


def main() -> None:
    # --- read hook input (fail-open on anything) ---
    try:
        raw = sys.stdin.read()
        inp = json.loads(raw) if raw.strip() else {}
    except Exception:
        inp = {}
    stop_hook_active = bool(inp.get("stop_hook_active"))

    # resolve repo root: hook cwd → git toplevel → "."
    root = inp.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or ""
    if not root:
        try:
            r = _git(".", "rev-parse", "--show-toplevel")
            root = r.stdout.strip() if r.returncode == 0 else "."
        except Exception:
            root = "."

    try:
        r = _git(root, "rev-parse", "HEAD")
        if r.returncode != 0:
            sys.exit(0)  # not a git repo / no HEAD — fail open
        head = r.stdout.strip()
    except Exception:
        sys.exit(0)

    marker = _read_marker(root)
    last_head = marker.get("last_head")
    last_block_head = marker.get("last_block_head")

    # First run (no marker): initialize to HEAD; nothing to compare against.
    if not last_head:
        _advance_and_allow(root, head)

    # Already nudged for this exact state → advance + allow (loop-safe).
    if stop_hook_active or (last_block_head and last_block_head == head):
        _log(root, f"allow(already-nudged) head={head[:9]} sha_active={stop_hook_active}")
        _advance_and_allow(root, head)

    # No new commits since the last stop → nothing to check (leave marker as-is).
    if last_head == head:
        sys.exit(0)

    # --- gather this-window commits + touched files ---
    try:
        rl = _git(root, "log", "--format=%s", f"{last_head}..{head}")
        subjects = [s for s in rl.stdout.splitlines() if s.strip()] if rl.returncode == 0 else []
        rd = _git(root, "diff", "--name-only", f"{last_head}..{head}")
        files = [f for f in rd.stdout.splitlines() if f.strip()] if rd.returncode == 0 else []
    except Exception:
        _advance_and_allow(root, head)  # can't diff → fail open, advance

    verdict = classify(subjects, files)

    if verdict == "drift":
        # Persist last_block_head (do NOT advance last_head — so the fix commit
        # lands inside the next window and clears the drift on re-check).
        marker["last_block_head"] = head
        marker["updated_utc"] = _now()
        _write_marker(root, marker)
        _log(root, f"BLOCK head={head[:9]} n={len(subjects)} subj={subjects[:3]!r}")
        print(json.dumps({"decision": "block", "reason": build_reason(subjects, files)}))
        sys.exit(0)

    _log(root, f"allow(clean) head={head[:9]} n={len(subjects)}")
    _advance_and_allow(root, head)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # absolute last-resort fail-open — never wedge a session.
        sys.exit(0)
