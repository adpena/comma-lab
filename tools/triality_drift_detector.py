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
#   Calibrated 2026-07-06 (adversarial review r1+r2): DROPPED "launch" (launching an existing
#   config is not a lever change) AND the r1-added "seed/island/activation/birth" family — r2
#   proved those over-fire on ordinary chores AND on DAG-FEED commits that merely MENTION a
#   seed/island (the DAG feed is the RECORDING mechanism, it must not trip the gate). A real
#   lever commit says "lever"/"wire-in"/"wired"; the residual false-negative is accepted (the
#   claim is honestly "narrows, not closes").
#   ``lever(?!ag(?:e|ing))`` excludes "leverage"/"leveraged"/"leveraging" (r4+r6 cosmetic)
#   while keeping "lever"/"levers"/"lever-D"/"lever-wire".
DSL_REQUIRING = re.compile(
    r"\b(?:lever(?!ag(?:e|ing))\w*|wire.?in\w*|integrat\w*|curriculum\w*|carrier\w*|gauge\w*)",
    re.IGNORECASE,
)
#   LAW / measured finding → must register/refine a canonical equation.
#   DROPPED the over-broad single words "floor"/"law"/"erasure" (fired on "floor division
#   bug", "outlaw", "erasure coding") AND the r4-flagged "measur\w*" — the everyday word
#   "measured/measurement" over-fired on descriptive commits AND on DAG-FEED commits that
#   merely RECORD a measurement (touching only the DAG). The precise finding signals survive:
#   the enumerated stems below + the numeric-value detector `_MEASURED_ROW` (a real d_seg/
#   d_pose row). "exact[\s-]row" catches both the space and hyphenated spelling (r4).
#   DROPPED "pointer\w*" (dogfood over-fire 2026-07-06): the ubiquitous provenance FOOTER
#   "pointer 0.19110 UNMOVED (apparatus)" is boilerplate in ~every commit, not a measured
#   finding — it fired the equations requirement on apparatus commits. A genuine pointer MOVE
#   always states its mechanism (byte-close/exact-eval/exact-row) or a numeric d_seg/d_pose row
#   (_MEASURED_ROW), which the surviving stems catch; the bare word is redundant + boilerplate.
EQUATION_REQUIRING = re.compile(
    r"\b(?:byte.?clos\w*|exact.?eval\w*|exact[\s-]row|verdict\w*|"
    r"scored|attribution\w*|refut\w*|ratif\w*)",
    re.IGNORECASE,
)
#   A MEASURED numeric row (e.g. "d_seg 0.0031", "measured d_seg of 0.0031", "d_seg→0.0047")
#   — a finding with a DECIMAL value. r6: allow a short connector ("of"/"="/"to"/"→", ≤6 chars)
#   between the metric and the value, but REQUIRE a decimal point so non-findings like
#   "d_seg curriculum v2" / "d_pose head at ep50" (a version/epoch integer, no decimal) stay
#   clean. d_seg/d_pose values are always sub-1 decimals, so requiring "." is safe here.
#   The ``(?<![A-Za-z0-9])`` before the number excludes a version token adjacent to the metric
#   ("d_seg v2.0" — the r7 cosmetic note) while keeping real measurements ("0.0047", "=0.0047",
#   "→0.0047", "3.4e-5", ".0047"): a measurement's value starts at a non-alnum boundary, a
#   version's digit is preceded by "v" (letter) or a digit. Fixed-width lookbehind ⇒ no ReDoS.
_MEASURED_ROW = re.compile(r"\b(?:d_seg|d_pose)\b[^\n]{0,6}?(?<![A-Za-z0-9])\d*\.\d", re.IGNORECASE)

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

# --- CONSUMER LEG (2026-07-07; operator standing requirement "As the DSL evolves,
# update the costate controller and dashboard accordingly") ---
# When a window GROWS the DSL's PUBLIC surface (a new Uppercase factory / public
# class / __init__ export), the generic describe()/registry introspection surfaces
# usually absorb it — but a change that OUTGROWS them must not land silently. So:
# public-DSL-surface growth nudges unless the same window ALSO touched a consumer
# surface, or a commit asserts generic coverage via the [consumers-generic] token.
DSL_CONSUMER_SURFACES = (
    "src/tac/witness_dsl/schedule_readback.py",
    "tools/dashboard_server.py",
    "tools/costate_digest.py",
    "src/tac/witness_control/producer_bridge.py",
)
CONSUMERS_GENERIC_TOKEN = re.compile(r"\[consumers-generic\]", re.IGNORECASE)
# A PUBLIC surface addition on an added diff line (leading "+" already stripped):
# an Uppercase-named def (the DSL's Lever/WitnessProgram factory convention) or a
# public (non-underscore) class. Lowercase defs are ordinary helpers; _private
# defs/classes and docstring/comment edits stay silent by construction.
_DSL_PUBLIC_SURFACE = re.compile(r"^\s*(?:def\s+[A-Z]\w*\s*\(|class\s+[A-Za-z]\w*)")
# An export change on an added line inside witness_dsl/__init__.py: a (re-)export
# import or an __all__ mutation. Docstring-only __init__ edits stay silent.
_DSL_INIT_EXPORT = re.compile(r"^\s*(?:from\s+\S+\s+import\b|import\s+\w|__all__)")


# ----------------------------- pure logic (tested) -----------------------------
def is_substantive(subjects: list[str]) -> bool:
    """True if any commit subject names a score/trajectory/build event. ``str(s or "")``
    coerces defensively so a non-string subject can never raise (r5 robustness)."""
    return any(SUBSTANTIVE.search(str(s or "")) for s in subjects)


def is_opted_out(subjects: list[str]) -> bool:
    """True if any commit in the window opts out via [no-triality]/[skip-drift].

    NOTE (accepted by-design bound, r4): opt-out is WINDOW-WIDE — one [no-triality] commit
    suppresses the nudge for a genuinely-drifting sibling in the same turn. This is the
    fail-SAFE direction (never over-nag), and scoping it per-commit would require the
    per-commit judging that r2 proved breaks the mandated separate-DAG-FEED workflow. So it
    is documented, not "fixed" into a worse regression."""
    return any(SKIP_TOKEN.search(str(s or "")) for s in subjects)


def has_triality_touch(files: list[str]) -> bool:
    """True if any changed file is a triality leg (DAG/DSL/equations/index)."""
    return any((f or "").startswith(TRIALITY_PREFIXES) for f in files)


def touched_dsl(files: list[str]) -> bool:
    """True if the DSL leg (the config-generator) was updated this window."""
    return any((f or "").startswith("src/tac/witness_dsl/") for f in files)


def touched_equations(files: list[str]) -> bool:
    """True if the canonical-equations leg (the law) was updated this window."""
    return any((f or "").startswith("src/tac/canonical_equations") for f in files)


def touched_dsl_consumer(files: list[str]) -> bool:
    """True if any changed file is a known DSL-consumer surface (readback/dashboard/
    costate-digest/producer-bridge)."""
    return any((f or "") in DSL_CONSUMER_SURFACES for f in files)


def is_consumers_generic(subjects: list[str]) -> bool:
    """True if any commit in the window carries the [consumers-generic] token — the
    author's assertion that the DSL change is fully covered by the describe()/registry
    introspection surfaces (generic rendering), so no consumer edit is needed."""
    return any(CONSUMERS_GENERIC_TOKEN.search(str(s or "")) for s in subjects)


def dsl_public_surface_added(diff_text: str) -> bool:
    """True if the unified diff ADDS public DSL surface under src/tac/witness_dsl/:
    a new/renamed Uppercase factory ``def``, a public ``class``, or an ``__init__.py``
    export change. Diff-based (added lines only), so docstring/comment/private edits
    and pure deletions stay silent. Defensive ``str(... or "")`` — never raises on
    odd input (mirrors the r5 coercion discipline)."""
    current_path = ""
    for line in str(diff_text or "").splitlines():
        if line.startswith("+++ "):
            current_path = line[4:].strip()
            if current_path.startswith("b/"):
                current_path = current_path[2:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if not current_path.startswith("src/tac/witness_dsl/"):
            continue
        added = line[1:]
        if current_path.endswith("/__init__.py"):
            if _DSL_INIT_EXPORT.search(added):
                return True
        elif _DSL_PUBLIC_SURFACE.search(added):
            return True
    return False


def consumer_leg_missing(subjects: list[str], files: list[str], dsl_diff_text: str) -> bool:
    """CONSUMER LEG: True iff this window GREW the DSL's public surface but touched
    NO consumer surface and carries neither the [consumers-generic] assertion nor the
    window-wide [no-triality]/[skip-drift] opt-out. Window granularity (same grain as
    ``missing_legs``); advisory-nudge semantics, same voice as the other legs."""
    if not touched_dsl(files):
        return False
    if is_opted_out(subjects) or is_consumers_generic(subjects):
        return False
    if touched_dsl_consumer(files):
        return False
    return dsl_public_surface_added(dsl_diff_text)


def consumer_leg_missing_safe(subjects: list[str], files: list[str], dsl_diff_text: str) -> bool:
    """Fail-open wrapper for the consumer leg: any exception in the NEW leg ⇒ False
    (silent), so a bug here can never block a session or perturb the existing legs."""
    try:
        return consumer_leg_missing(subjects, files, dsl_diff_text)
    except Exception:
        return False


def missing_legs(subjects: list[str], files: list[str]) -> list[str]:
    """The SPECIFIC triality legs a change of this type REQUIRES but did not touch.

    A control change (lever/wire-in/curriculum/carrier/gauge) must touch the DSL; a measured-law
    change (measure/byte-close/verdict/exact-row/a numeric d_seg/d_pose row) must touch the
    canonical equations. Returns the leg names that are required-but-absent (empty == every
    required leg was touched).

    NOTE (honest bound, r2): judged at WINDOW granularity (the union of the turn's commits +
    files). This project's git discipline mandates a SEPARATE DAG-FEED commit per work commit,
    so per-commit judging false-fires that mandated workflow (r2 MEDIUM-1); the window union is
    the correct grain. The residual is a known LOW: an unrelated leg-touch in the same window
    can satisfy a different commit's requirement — accepted, since the alternative breaks the
    one-change-per-commit discipline. ``main`` calls this on the window union."""
    joined = " ".join(str(s or "") for s in subjects)   # str() coerces defensively (r5)
    miss: list[str] = []
    if DSL_REQUIRING.search(joined) and not touched_dsl(files):
        miss.append("DSL")
    if (EQUATION_REQUIRING.search(joined) or _MEASURED_ROW.search(joined)) and not touched_equations(files):
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


CONSUMER_NUDGE = (
    "Triality drift-detector (consumer leg): this turn ADDED/renamed PUBLIC DSL "
    "surface (src/tac/witness_dsl/ — a new factory/class or an __init__ export) "
    "without touching any consumer surface "
    "(src/tac/witness_dsl/schedule_readback.py, tools/dashboard_server.py, "
    "tools/costate_digest.py, src/tac/witness_control/producer_bridge.py). "
    "Per the standing operator requirement ('As the DSL evolves, update the "
    "costate controller and dashboard accordingly'), record the consumer leg "
    "proactively: update the consumer(s) that render/consume this surface, OR — "
    "if the change is fully covered by the describe()/registry introspection "
    "surfaces (generic rendering) — add [consumers-generic] to the commit "
    "message to assert that. Advisory: this NARROWS (does not eliminate) silent "
    "DSL-evolution drift past the generic surfaces."
)


def build_reason(subjects: list[str], files: list[str] | None = None) -> str:
    """The one firm nudge injected on drift (concise + actionable + leg-specific)."""
    preview = "; ".join(str(s or "")[:70] for s in subjects[:3])
    miss = missing_legs(subjects, files or [])
    if miss:
        legs = " AND ".join(_LEG_FIX.get(m, m) for m in miss)
        return (
            "Triality drift-detector (per-leg): "
            f"{len(subjects)} commit(s) landed this turn that REQUIRE {legs}. "
            f"Commit(s): {preview}. Update the named leg(s) — plus a DAG FEED "
            "(.omx/research/sub015_DAG_*) and a MEMORY.md line for a durable "
            "finding — then finish. Pure chore/apparatus can opt out with "
            "[no-triality] in the commit message. This gate enforces the RIGHT "
            "leg per change-type for the recognised control/measurement "
            "vocabulary — it NARROWS (does not fully eliminate) silent DSL/"
            "equation drift, so record the leg proactively regardless."
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

    # --- gather this-window subjects + the UNION of changed files (window granularity, r2) ---
    # The union of the range diff is the correct grain: this project mandates a SEPARATE
    # DAG-FEED commit per work commit (git discipline), so per-commit judging false-fired that
    # workflow AND broke on 0-file merge commits (r2 MEDIUM-1/2). ``git diff last..head`` also
    # correctly folds a merge's branch changes into the union.
    try:
        subjects = [
            s for s in _git(root, "log", "--format=%s", f"{last_head}..{head}").stdout.splitlines()
            if s.strip()
        ]
        files = [
            f for f in _git(root, "diff", "--name-only", f"{last_head}..{head}").stdout.splitlines()
            if f.strip()
        ]
    except Exception:
        _advance_and_allow(root, head)  # can't read log → fail open, advance

    # --- CONSUMER LEG (fail-open independently: a bug in the NEW leg can neither
    # block the session nor perturb the existing legs' verdict) ---
    consumer_missing = False
    try:
        if touched_dsl(files):
            dsl_diff = _git(
                root, "diff", f"{last_head}..{head}", "--", "src/tac/witness_dsl/"
            ).stdout
            consumer_missing = consumer_leg_missing_safe(subjects, files, dsl_diff)
    except Exception:
        consumer_missing = False

    if classify(subjects, files) == "drift" or consumer_missing:
        # Persist last_block_head (do NOT advance last_head — so the fix/DAG-FEED commit lands
        # inside the next window's UNION and clears the drift on re-check; the last_block_head
        # backstop clears the block once head is unchanged, so it cannot loop).
        marker["last_block_head"] = head
        marker["updated_utc"] = _now()
        _write_marker(root, marker)
        _log(
            root,
            f"BLOCK head={head[:9]} n={len(subjects)} miss={missing_legs(subjects, files)}"
            f" consumer_leg={consumer_missing}",
        )
        if classify(subjects, files) == "drift":
            reason = build_reason(subjects, files)
            if consumer_missing:
                reason = reason + " ALSO — " + CONSUMER_NUDGE
        else:
            reason = CONSUMER_NUDGE
        print(json.dumps({"decision": "block", "reason": reason}))
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
