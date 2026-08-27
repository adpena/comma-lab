#!/usr/bin/env python3
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Operator-P0 demand-update — a Claude Code ``Stop`` hook (landing 2 of two).

Operator binding 2026-07-15 (verbatim): "Do we need a hook or gate or something
to remind you if p0 to survive compaction? That also demands update when
complete or when new p0 designated."

Fires at the end of each main-agent turn and nags (ONCE per state, warn-grade,
fail-open — modeled on tools/triality_drift_detector.py) when:

  (A) TOUCHED-WITHOUT-UPDATE — this turn's commits touched a tracked open/
      in_progress operator-P0 (commit subject names its ``p0_id`` or a bound
      task ``#NNN``, or a changed file sits under one of its ``watch_paths``)
      but NO ledger row was appended this window (the ledger is gitignored-by-
      pattern-but-committed; we compare max ``written_at_utc`` against the
      marker, which also catches uncommitted appends).

  (B) NEW-DESIGNATION-WITHOUT-ROW — a NEW operator message in this session's
      transcript designates a P0 (word-bounded "p0" + a directive verb nearby)
      but no ledger row was appended since. Heuristic is deliberately
      conservative; scanning is incremental (marker stores per-transcript
      line offset) so old messages never re-fire.

Escape valve: a commit whose message contains ``[p0-ledger-ok]`` declares the
window deliberately ledger-neutral (e.g. mechanical chore touching a watch
path). Never binary — one nag, then allow.

Design invariants (NON-NEGOTIABLE for a Stop hook):
  * FAIL-OPEN — any error/timeout ⇒ exit 0. Never wedge a session.
  * LOOP-SAFE — ``stop_hook_active`` + persisted ``last_block_key`` backstop.
  * EVENT-TRIGGERED — silent when no new commits AND no new transcript lines.

Sister: tools/operator_p0_digest.py (ledger lib + SessionStart/compact digest).
Not in ``tac`` — Claude-workflow apparatus (CLAUDE.md tac-cleanliness rule).
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import re
import subprocess
import sys

MARKER_REL = ".omx/state/operator_p0_stop_marker.json"
LOG_REL = ".omx/state/operator_p0_stop_hook.log"

SKIP_TOKEN = re.compile(r"\[p0-ledger-ok\]", re.IGNORECASE)

# A NEW operator P0 designation: word-bounded "p0" plus a directive cue within
# the same message. Conservative on purpose (warn-grade): plain mentions of
# "P0" in status narration ("the P0 ledger is clean") do not carry a directive
# verb adjacent, and the ledger-update escape (any append) clears the nag.
_P0_WORD = re.compile(r"(?<![A-Za-z0-9_])[pP]0(?![A-Za-z0-9])")
_DIRECTIVE = re.compile(
    r"\b(?:pursue|must|now|fix|build|launch|elevate\w*|designat\w*|priorit\w*|"
    r"as\s+p0|treat\w*|aggressively|immediately|non[- ]?negotiab\w*)\b",
    re.IGNORECASE,
)
# Lines that are ABOUT the ledger/apparatus itself (self-reference) stay silent.
_SELF_REF = re.compile(r"operator_p0|p0[-_ ]ledger|p0[-_ ]digest|p0[-_ ]stop[-_ ]hook",
                       re.IGNORECASE)
# System-INJECTED regions inside user-role turns are NOT operator words. Empirical
# false-positive class (2026-07-17, fired 3x in one hour): a task-notification whose
# <summary> carried a subagent NAME containing "P0 ... fix" ("P0 launcher durability +
# bench validity fix") matched _P0_WORD+_DIRECTIVE on every heartbeat. Strip tagged
# blocks wholesale and drop hook-feedback turns before scanning.
_SYSTEM_BLOCK = re.compile(
    r"<task-notification>.*?</task-notification>"
    r"|<system-reminder>.*?</system-reminder>"
    r"|<command-[a-z-]+>.*?</command-[a-z-]+>"
    r"|<local-command-[a-z-]+>.*?</local-command-[a-z-]+>",
    re.DOTALL | re.IGNORECASE,
)
_HOOK_FEEDBACK_PREFIX = re.compile(r"^\s*Stop hook feedback:", re.IGNORECASE)


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(root: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", root, *args], capture_output=True, text=True, timeout=15
    )


def _load_digest_module(root: str):
    """Import the sibling ledger library (tools/ is not a package)."""
    path = os.path.join(root, "tools", "operator_p0_digest.py")
    spec = importlib.util.spec_from_file_location("operator_p0_digest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ----------------------------- pure logic (tested) -----------------------------
def is_opted_out(subjects: list[str]) -> bool:
    return any(SKIP_TOKEN.search(str(s or "")) for s in subjects)


def touched_p0s(rows: list[dict], subjects: list[str], files: list[str]) -> list[str]:
    """p0_ids of tracked open/in_progress P0s this window's commits touched.

    A P0 counts as touched when (a) a commit subject contains its p0_id, (b) a
    commit subject contains ``#<task_id>`` for one of its bound task numbers, or
    (c) a changed file starts with one of its ``watch_paths`` prefixes.
    """
    subj_blob = "\n".join(str(s or "") for s in subjects)
    hit: list[str] = []
    for row in rows:
        if row.get("status") not in ("open", "in_progress"):
            continue
        pid = str(row.get("p0_id") or "")
        if not pid:
            continue
        if pid and pid in subj_blob:
            hit.append(pid)
            continue
        task_ids = row.get("task_ids") or []
        if any(re.search(rf"#\s*{re.escape(str(t))}\b", subj_blob) for t in task_ids):
            hit.append(pid)
            continue
        watch = [str(w) for w in (row.get("watch_paths") or []) if str(w).strip()]
        if watch and any(
            (f or "").startswith(tuple(watch)) for f in files
        ):
            hit.append(pid)
    return hit


def extract_text(message: object) -> str:
    """Best-effort text of a transcript ``message`` (str or content-block list)."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
            return "\n".join(parts)
    return ""


def new_p0_designations(lines: list[str]) -> list[str]:
    """Snippets of USER (operator) transcript lines that look like NEW P0
    designations. Tool results / assistant turns / self-referential apparatus
    talk are excluded."""
    found: list[str] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "user":
            continue
        msg = obj.get("message")
        role = msg.get("role") if isinstance(msg, dict) else None
        if role not in (None, "user"):
            continue
        # skip synthetic tool-result user turns
        if (isinstance(msg, dict) and isinstance(msg.get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in msg["content"])):
            continue
        text = extract_text(msg)
        if not text:
            continue
        # System-injected user-role turns are not operator words: hook feedback
        # turns are skipped whole; tagged notification/reminder blocks stripped.
        if _HOOK_FEEDBACK_PREFIX.match(text):
            continue
        text = _SYSTEM_BLOCK.sub("", text)
        if not text.strip() or _SELF_REF.search(text):
            continue
        for line in text.splitlines():
            if _P0_WORD.search(line) and _DIRECTIVE.search(line):
                found.append(line.strip()[:160])
                break  # one snippet per message
    return found


# ---------------------- already-registered coverage filter (CLASS 3 fix) ----------------------
# Empirical false-positive (2026-07-17, NAG new_designations=1 while the P0s were ALREADY in the
# ledger): the designation demand fired without checking existing rows' ``verbatim_ask``. The
# operator experiences "register it NOW" for a P0 they already registered. Fix: drop a designation
# demand when an existing ledger row's verbatim_ask COVERS it. Coverage is a semantic-similarity
# judgment: deterministic token-containment is the AUTHORITY (always available, unit-tested); an
# fmtools advisory (subprocess, graceful-degrade) may CORROBORATE in the gray band but is NEVER the
# sole gate and NEVER a block. Fail-closed bias: covered only when confidently covered; uncertain
# or FM-absent ⇒ still demand.
_COVER_STOPWORDS = frozenset((
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "we", "should", "be", "is",
    "are", "our", "all", "this", "that", "it", "as", "with", "at", "by", "so", "if", "i", "you",
    "make", "sure", "also", "against", "very", "p0", "operator", "must", "now",
))
_COVER_STRONG = 0.60   # deterministic containment ⇒ covered outright
_COVER_GRAY = 0.40     # [gray, strong) ⇒ consult the FM advisory as a corroborator (never sole)


def _cover_tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return {t for t in toks if len(t) >= 3 and t not in _COVER_STOPWORDS}


def _containment(designation_tokens: set[str], ask_tokens: set[str]) -> float:
    """Fraction of the designation's meaningful tokens present in the ask (directional)."""
    if not designation_tokens:
        return 0.0
    return len(designation_tokens & ask_tokens) / len(designation_tokens)


def _best_containment(designation: str, rows: list[dict]) -> tuple[float, str]:
    """Best token-containment of ``designation`` against any ledger row's verbatim_ask."""
    d = _cover_tokens(designation)
    best = 0.0
    best_ask = ""
    for r in rows:
        ask = str(r.get("verbatim_ask") or "")
        if not ask:
            continue
        c = _containment(d, _cover_tokens(ask))
        if c > best:
            best, best_ask = c, ask
    return best, best_ask


_FM_COVER_SCRIPT = r'''
import asyncio, json, sys
try:
    import apple_fm_sdk as fm
    from fmtools import local_extract
except Exception:
    print("{}"); raise SystemExit(0)

@fm.generable()
class Cover:
    covered: bool = fm.guide(description="True iff the registered ask already covers the directive.")

@local_extract(Cover, retries=1, instructions=(
    "You judge whether a REGISTERED operator ask already covers a newly-detected operator "
    "directive (i.e. they are the same request). Answer covered=true ONLY when the registered "
    "ask clearly encompasses the directive; otherwise covered=false."))
async def _check(pair_text: str) -> Cover:
    """(instructions above)"""

async def _main():
    try:
        pair = json.load(sys.stdin)
    except Exception:
        print("{}"); return
    try:
        r = await _check(f"DIRECTIVE: {pair.get('directive','')[:600]}\nREGISTERED ASK: {pair.get('ask','')[:600]}")
        print(json.dumps({"covered": bool(getattr(r, "covered", False))}))
    except Exception:
        print("{}")

asyncio.run(_main())
'''


def _fm_python() -> str | None:
    for cand in (os.environ.get("VERDICT_SCOPE_FM_PYTHON"),
                 os.environ.get("DASH_FM_PYTHON"),
                 os.path.expanduser("~/Projects/fmtools/.venv/bin/python")):
        if cand and os.path.exists(cand):
            return cand
    return None


def _fm_covered(directive: str, ask: str, timeout: int = 15) -> bool:
    """fmtools advisory: does ``ask`` cover ``directive``? Fail-silent False on absence/timeout —
    advisory by construction, NEVER the sole gate (only consulted in the gray band + corroborated
    by a nonzero deterministic overlap)."""
    fm_py = _fm_python()
    if not fm_py:
        return False
    try:
        proc = subprocess.run(
            [fm_py, "-c", _FM_COVER_SCRIPT],
            input=json.dumps({"directive": directive, "ask": ask}),
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return False
        out = json.loads(proc.stdout.strip() or "{}")
        return bool(out.get("covered")) if isinstance(out, dict) else False
    except Exception:
        return False


def designation_already_covered(designation: str, rows: list[dict], *, use_fm: bool = True) -> bool:
    """True iff an existing ledger row's verbatim_ask already covers ``designation`` (so demanding
    re-registration would be the CLASS-3 false-positive). Deterministic token-containment is the
    authority; the FM advisory only corroborates in the gray band and never decides alone."""
    d = _cover_tokens(designation)
    if len(d) < 3:
        # Too few meaningful tokens to judge similarity — fail-closed (still demand).
        return False
    best, best_ask = _best_containment(designation, rows)
    if best >= _COVER_STRONG:
        return True
    if use_fm and best >= _COVER_GRAY and best_ask:
        # FM is consulted ONLY with a nonzero deterministic corroboration (never sole gate).
        return _fm_covered(designation, best_ask)
    return False


# ----------------------------- marker + main -----------------------------
def _read_marker(root: str) -> dict:
    try:
        with open(os.path.join(root, MARKER_REL)) as fh:
            m = json.load(fh)
            return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def _write_marker(root: str, marker: dict) -> None:
    try:
        path = os.path.join(root, MARKER_REL)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(marker, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def _log(root: str, msg: str) -> None:
    try:
        with open(os.path.join(root, LOG_REL), "a") as fh:
            fh.write(f"{_now()} {msg}\n")
    except Exception:
        pass


def main() -> None:
    try:
        # isatty guard (ddm_gh2, 2026-07-31): a bare sys.stdin.read() BLOCKS rather
        # than raises when stdin is a TTY or an inherited never-closed pipe, so the
        # try/except cannot save it. Sister hooks codex_landing_review_gate.py:516 and
        # auto_push_main.py:422 already carry this guard.
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        inp = json.loads(raw) if raw.strip() else {}
    except Exception:
        inp = {}
    stop_hook_active = bool(inp.get("stop_hook_active"))

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
            sys.exit(0)
        head = r.stdout.strip()
    except Exception:
        sys.exit(0)

    try:
        digest = _load_digest_module(root)
        ledger_max = digest.max_written_utc(root)
        rows = list(digest.read_ledger(root).values())
    except Exception:
        sys.exit(0)  # no ledger lib / unreadable ledger — fail open

    marker = _read_marker(root)
    last_head = marker.get("last_head")
    last_ledger_max = str(marker.get("last_ledger_max") or "")
    block_key = f"{head}:{ledger_max}"

    # First run: initialize, nothing to compare. Baseline the transcript offset
    # so PRE-EXISTING messages never retro-fire check B (only NEW designations
    # after the apparatus went live are in scope).
    if not last_head:
        offsets: dict = {}
        t = str(inp.get("transcript_path") or "")
        if t and os.path.exists(t):
            try:
                with open(t, errors="replace") as fh:
                    offsets[t] = sum(1 for _ in fh)
            except Exception:
                pass
        _write_marker(root, {"last_head": head, "last_ledger_max": ledger_max,
                             "offsets": offsets, "updated_utc": _now()})
        sys.exit(0)

    # Loop-safety: already nagged for this exact (head, ledger) state.
    if stop_hook_active or marker.get("last_block_key") == block_key:
        marker.update({"last_head": head, "last_ledger_max": ledger_max,
                       "last_block_key": None, "updated_utc": _now()})
        _write_marker(root, marker)
        _log(root, f"allow(already-nagged) head={head[:9]}")
        sys.exit(0)

    ledger_updated_this_window = ledger_max > last_ledger_max

    # --- window commits ---
    subjects: list[str] = []
    files: list[str] = []
    if last_head != head:
        try:
            subjects = [s for s in _git(root, "log", "--format=%s",
                                        f"{last_head}..{head}").stdout.splitlines()
                        if s.strip()]
            files = [f for f in _git(root, "diff", "--name-only",
                                     f"{last_head}..{head}").stdout.splitlines()
                     if f.strip()]
        except Exception:
            subjects, files = [], []

    # --- check A: touched a tracked P0 without a ledger update ---
    touched: list[str] = []
    if subjects and not ledger_updated_this_window and not is_opted_out(subjects):
        try:
            touched = touched_p0s(rows, subjects, files)
        except Exception:
            touched = []

    # --- check B: new operator-P0 designation without a ledger row ---
    designations: list[str] = []
    transcript = str(inp.get("transcript_path") or "")
    if transcript and os.path.exists(transcript) and not ledger_updated_this_window:
        try:
            offsets = marker.get("offsets") or {}
            start = int(offsets.get(transcript, 0))
            with open(transcript, errors="replace") as fh:
                lines = fh.readlines()
            new_lines = lines[start:]
            designations = new_p0_designations(new_lines)
            # CLASS-3 fix: drop designations already covered by an existing ledger row's
            # verbatim_ask (do not demand re-registration of an already-registered P0).
            if designations:
                designations = [
                    d for d in designations if not designation_already_covered(d, rows)
                ]
            offsets[transcript] = len(lines)
            marker["offsets"] = offsets
        except Exception:
            designations = []

    if touched or designations:
        marker.update({"last_block_key": block_key, "updated_utc": _now()})
        # do NOT advance last_head/last_ledger_max — the fixing append clears it
        _write_marker(root, marker)
        _log(root, f"NAG head={head[:9]} touched={touched[:4]} "
                   f"new_designations={len(designations)}")
        parts = []
        if touched:
            parts.append(
                "OPERATOR-P0 LEDGER STALE: this turn touched tracked operator-P0(s) "
                f"{sorted(set(touched))[:4]} but appended NO ledger update. Update each via "
                ".venv/bin/python tools/operator_p0_digest.py --update <p0_id> "
                "--status <open|in_progress|complete|superseded> --evidence '<artifact>' "
                "--next-action '<one line>'. Deliberate no-op: commit token [p0-ledger-ok]."
            )
        if designations:
            parts.append(
                "NEW OPERATOR-P0 DESIGNATED THIS SESSION with no ledger row: "
                + " | ".join(f"'{d}'" for d in designations[:2])
                + " — register it NOW via tools/operator_p0_digest.py --update <new_p0_id> "
                "--status open --verbatim-ask '<operator words>' --next-action '<first step>' "
                "so it survives compaction (operator: P0s must never be silently dropped)."
            )
        print(json.dumps({"decision": "block", "reason": " ALSO — ".join(parts)}))
        sys.exit(0)

    marker.update({"last_head": head, "last_ledger_max": ledger_max,
                   "last_block_key": None, "updated_utc": _now()})
    _write_marker(root, marker)
    _log(root, f"allow(clean) head={head[:9]} n={len(subjects)}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)  # absolute last-resort fail-open
