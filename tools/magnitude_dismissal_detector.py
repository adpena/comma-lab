#!/usr/bin/env python3
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Magnitude-dismissal detector — a Claude Code ``Stop`` hook.

Fires at the end of each main-agent turn. Detects the **MAGNITUDE-DISMISSAL**
bug class (operator recurring correction 2026-07-08, "we have had this discussion
before"): a DEFER / DOWNGRADE-to-WEAK / ORPHAN / KILL verdict on a lever or
finding that is justified by ABSOLUTE MAGNITUDE ("weak / negligible / noise /
small ΔS / little to gain / not worth it") WITHOUT either

  (a) a RELATIVE-significance computation — ΔS / remaining-gap-to-target stated
      at the current operating point (the memory's fix), OR
  (b) a cited MEASUREMENT of un-recoverability — the #141 label-noise case:
      "the residual is un-recoverable / at the noise floor" WITH a measurement /
      exit criterion, or a structurally-superseded verdict.

When it finds one, it injects ONE firm advisory nudge that TRIGGERS the
examine/audit/adversarial-review the operator asked for: name the dismissed
item, quote the passage, and instruct the agent to compute ΔS/(S_current −
S_target) at the current operating point + adversarially review before the defer
stands (or, if genuinely un-recoverable, cite the measurement).

This is landing #2 of the "bugs must be permanently fixed AND self-protected
against" pattern — landing #1 = the memory
``relative-not-absolute-significance-near-goal-dont-orphan-small-deltaS`` + the
live re-audit. The sister STATIC gate is
``tac.confound_gates.check_no_unjustified_magnitude_dismissal`` (#404), which
scans committed memos; this hook fires the moment the class is introduced.

Design invariants (mirrors ``tools/triality_drift_detector.py`` verbatim — the
canonical Stop-hook precedent):
  * FAIL-OPEN — any error, non-git dir, timeout, or missing fmtools ⇒ allow the
    stop (exit 0). A Stop hook must NEVER wedge a session, and never crash it.
  * ADVISORY ONLY — a Stop-hook "block" re-engages the agent at turn-end; it does
    NOT block a LAUNCH (heavy/paid/launch gating stays with the governor +
    launch_guard_hook, the correct authority boundary). This hook only nudges.
  * TWO-STAGE, ROBUST — a cheap deterministic PRE-FILTER finds candidate passages
    (dismissal-verb + magnitude-word linked in one claim, ABSENT a relative-
    significance ratio AND a measured-un-recoverability citation); then an fmtools
    SEMANTIC
    CLASSIFY confirms each candidate (killing the false positives — noise-floor,
    weak-supervision, measured-label-noise). fmtools runs ONLY on pre-filtered
    candidates (never per-line), in a SUBPROCESS under the fmtools venv (the pact
    venv gains zero deps), and if it is absent/errors the deterministic verdict
    stands with an honest "fmtools confirmation owed" label — the self-protection
    is live everywhere, never faked.
  * LOOP-SAFE — one firm nudge per drift state, guarded by BOTH Claude Code's
    ``stop_hook_active`` flag AND a persisted ``last_block_head`` backstop.
  * EVENT-TRIGGERED — silent when there are no new commits since the last stop.
  * ESCAPE VALVE ("never binary") — a commit message with ``[magnitude-ok]`` /
    ``[skip-magnitude]`` opts the window out; a per-line
    ``# MAGNITUDE_DISMISSAL_OK:<rationale>`` waiver opts a specific line out.

Not placed in ``tac`` — this is Claude-workflow apparatus, not contest/codec
logic (per the CLAUDE.md ``tac`` cleanliness rule). Pure logic lives in
module-level functions so it is unit-testable without a crafted git HEAD; the
confound-gate sister imports those predicates so there is ONE classifier SoT.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys

# --- state files (both gitignored under .omx/state/*) ---
MARKER = ".omx/state/magnitude_dismissal_marker.json"
LOG = ".omx/state/magnitude_dismissal_detector.log"
ADVISORIES = ".omx/state/magnitude_dismissal_advisories.jsonl"

# ============================ the classifier vocabulary ============================
# The DECISION: a defer / downgrade / orphan / kill / drop / shelve of a lever or
# finding. Broad-ish, but only ever fires in CO-OCCURRENCE with a magnitude word
# (below), which is what makes the pair specific. ``\b`` word-anchored.
_DISMISSAL = re.compile(
    r"\b(?:defer(?:red|ring|s)?|downgrad(?:e|ed|es|ing)|orphan(?:ed|ing|s)?|"
    r"kill(?:ed|ing|s)?|shelv(?:e|ed|es|ing)|de-?prioriti[sz]e[ds]?|"
    r"park(?:ed|ing)?|abandon(?:ed|ing|s)?|set\s+aside|table\s+(?:it|this)|"
    r"not\s+worth(?:\s+(?:it|the|pursuing|chasing))?|"
    r"leave\s+(?:it|this)\s+(?:for\s+)?later|deprecat(?:e|ed|es|ing)|"
    r"skip(?:ped|ping)?|(?:de-?)?scope\s+out|move\s+on\s+from)\b",
    re.IGNORECASE,
)
# The JUSTIFICATION: an ABSOLUTE-magnitude smallness claim.
_MAGNITUDE = re.compile(
    r"\b(?:weak(?:ly)?|negligibl[ey]|insignificant(?:ly)?|tiny|minimal|"
    r"trivial(?:ly)?|marginal(?:ly)?|too\s+small|so\s+small|"
    r"small\s+(?:Δs|delta.?s|gain|effect|impact|improvement|margin)|"
    r"little\s+to\s+gain|not\s+much(?:\s+to\s+gain)?|barely|diminishing|"
    r"nois[ey](?![-\s]?floor)|not\s+significant|hardly\s+moves)\b",
    re.IGNORECASE,
)
# EXEMPTION (a): a RELATIVE-significance computation is present.
_RELATIVE_SIG = re.compile(
    r"(?:relative\s+significance|remaining\s+(?:gap|descent|distance)|"
    r"gap.?to.?target|fraction\s+of\s+the\s+remaining|"
    r"%\s+of\s+the\s+(?:remaining|gap|target|descent)|"
    r"of\s+the\s+(?:remaining|target)\s+(?:gap|descent|d_seg|budget)|"
    r"Δs\s*/|delta.?s\s*/|s_current|s_target|/\s*\(?\s*s_?current|"
    r"per-?cent\s+of\s+the\s+remaining|[0-9]+\s*[-–—]?\s*[0-9]*\s*%\s+of\s+the)",
    re.IGNORECASE,
)
# EXEMPTION (b): a MEASURED un-recoverability citation, or structurally superseded.
# NOTE: bare "label-noise" does NOT exempt — the memory (#141 distinction) is
# explicit that label-noise defers only WITH a measurement / exit criterion.
_MEASURED_UNRECOVERABLE = re.compile(
    r"(?:un-?recoverab\w+|irreducibl\w+|noise\s+floor|information[-\s]?floor|"
    r"exit\s+criterion|measured\s+(?:no-?go|un-?recoverab\w+|ceiling|floor|"
    r"unachievab\w+)|MEASURED\s+NO-?GO|no\s+term\s+can\s+predict|"
    r"cannot\s+be\s+(?:predicted|recovered)|"
    r"structurally\s+supersed\w+|supersed(?:e|ed|es|ing)\s+by|"
    r"\[empirical:|\[contest-(?:cpu|cuda)|byte-?clos\w+|n600|"
    r"label[-\s]?noise\b[^\n]{0,80}?(?:measured|exit\s+criterion|n600|"
    r"un-?recoverab|irreducibl))",
    re.IGNORECASE,
)
# FALSE-POSITIVE magnitude usages that are not dismissals of a lever at all.
_LEGIT_NONDISMISSAL = re.compile(
    r"(?:weak\s+supervis|weakly[-\s]?supervis|weakly[-\s]?driven|"
    r"noise\s+inject|sigma\s+noise|gaussian\s+noise|signal[-\s]?to[-\s]?noise|"
    r"random[-\s]?walk|quantization\s+noise|weak\s+lensing|weak\s+form)",
    re.IGNORECASE,
)
# Per-line deliberate waiver (placeholder rationales rejected).
_WAIVER = re.compile(r"#\s*MAGNITUDE_DISMISSAL_OK\s*:\s*(\S.*)")
_WAIVER_PLACEHOLDER = re.compile(
    r"^<[^>]*>$|^(?:todo|tbd|reason|rationale|xxx)\.?$", re.IGNORECASE)
# Window-wide opt-out token (a deliberate chore/known-legit-dismissal commit).
_SKIP_TOKEN = re.compile(r"\[(magnitude-ok|skip-magnitude)\]", re.IGNORECASE)
# Rule-discussion / quoting cues: a line ABOUT the class, not committing it. NOTE the
# waiver marker MAGNITUDE_DISMISSAL_OK is deliberately NOT here — a line carrying it is
# decided by its OWN waiver validity (below), never self-exempted via its marker substring.
_DISCUSSION_CUE = re.compile(
    r"magnitude[-\s]?dismissal|relative[-\s]not[-\s]absolute|"
    r"\bnot\s+a\s+(?:defer|dismiss|kill)|never\s+dismiss|do\s+not\s+dismiss|"
    r"reopen|re-?open|re-?rank|re-?audit",
    re.IGNORECASE,
)
# A cross-line pair is one claim only when the text carries an explicit causal link,
# an anaphoric/magnitude-led sentence, or a colon that introduces the justification. Raw
# proximity is not a link: Markdown table rows and source/provenance inventories routinely
# place dismissal-shaped class nouns near unrelated magnitude vocabulary.
_CAUSAL_LINK = re.compile(
    r"\b(?:because|since|due\s+to|on\s+account\s+of|therefore|thus|hence|"
    r"consequently|as\s+a\s+result|which\s+(?:is|was|looks|seems)|"
    r"whose\s+(?:effect|gain|impact|improvement|margin|result))\b",
    re.IGNORECASE,
)
_ANAPHORIC_MAGNITUDE = re.compile(
    r"^(?:[-*+]\s+|\d+[.)]\s+|>\s*)*(?:its|their|this|that|the)\s+"
    r"(?:measured\s+)?(?:Δs|delta.?s|effect|gain|impact|improvement|margin|"
    r"result|signal|contribution)\b",
    re.IGNORECASE,
)
_MAGNITUDE_LEAD = re.compile(
    r"^(?:[-*+]\s+|\d+[.)]\s+|>\s*)*(?:weak(?:ly)?|negligibl[ey]|"
    r"insignificant(?:ly)?|tiny|minimal|trivial(?:ly)?|marginal(?:ly)?|"
    r"too\s+small|so\s+small|barely|diminishing|nois[ey]|"
    r"not\s+significant|hardly\s+moves)\b(?![-_])",
    re.IGNORECASE,
)
_MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def _has_valid_waiver(line: str) -> bool:
    """True iff the line carries # MAGNITUDE_DISMISSAL_OK:<real rationale> (≥4 chars,
    non-placeholder). The docstring's own <rationale> example cannot self-waive."""
    m = _WAIVER.search(str(line or ""))
    if not m:
        return False
    rationale = m.group(1).strip()
    return len(rationale) >= 4 and not _WAIVER_PLACEHOLDER.match(rationale)


def is_opted_out(subjects: list[str]) -> bool:
    """True iff any commit message in the window carries [magnitude-ok]/[skip-magnitude].
    Window-wide (fail-SAFE direction: never over-nag), mirrors the drift detector."""
    return any(_SKIP_TOKEN.search(str(s or "")) for s in subjects)


def _line_exempt(passage: str, wide: str) -> bool:
    """A candidate passage is EXEMPT (not a bug) when the wider window carries a
    relative-significance computation (a), a measured-un-recoverability / superseded
    citation (b), a non-dismissal magnitude usage, a valid waiver, or a
    rule-discussion/quoting cue. ``wide`` is the exemption-search window (⊇ passage)."""
    if _WAIVER.search(passage):
        # a waiver marker decides the passage by its OWN validity — it must NOT fall
        # through to the discussion-cue exemption (its marker substring would self-exempt).
        return _has_valid_waiver(passage)
    if str(passage or "").lstrip().startswith(">"):
        return True  # markdown quote — quoting a prior verdict, not issuing one
    return bool(
        _RELATIVE_SIG.search(wide)
        or _MEASURED_UNRECOVERABLE.search(wide)
        or _LEGIT_NONDISMISSAL.search(wide)
        or _DISCUSSION_CUE.search(wide)
    )


def _has_linked_cooccurrence(window: list[str]) -> bool:
    """True iff dismissal and magnitude matches belong to one claim.

    A same-line pair is linked.  A cross-line pair must carry an explicit causal,
    anaphoric, or magnitude-led bridge (or a colon introducing the next line), and
    distinct Markdown table rows never lend vocabulary to one another. This preserves
    wrapped-prose detection without treating arbitrary three-line proximity as semantic
    co-occurrence.
    """
    window = [str(ln or "") for ln in (window or [])]
    dismissal_lines = [i for i, line in enumerate(window) if _DISMISSAL.search(line)]
    magnitude_lines = [i for i, line in enumerate(window) if _MAGNITUDE.search(line)]
    for d_idx in dismissal_lines:
        for m_idx in magnitude_lines:
            if d_idx == m_idx:
                return True
            lo, hi = sorted((d_idx, m_idx))
            span = window[lo: hi + 1]
            if any(_MARKDOWN_TABLE_ROW.match(line) for line in span):
                continue
            joined = "\n".join(span)
            magnitude_line = window[m_idx].strip()
            if (
                _CAUSAL_LINK.search(joined)
                or _ANAPHORIC_MAGNITUDE.search(magnitude_line)
                or _MAGNITUDE_LEAD.search(magnitude_line)
                or (d_idx < m_idx and window[d_idx].rstrip().endswith(":"))
            ):
                return True
    return False


def magnitude_dismissal_candidates(lines: list[str]) -> list[dict]:
    """The deterministic PRE-FILTER. Slides a 3-line window over ``lines``; a window
    is a CANDIDATE iff it links a dismissal verb AND a magnitude word in one claim AND
    is not exempt (relative-sig / measured-unrecoverability / non-dismissal / waiver /
    discussion) within a wider 5-line window. Same-line pairs are linked directly;
    cross-line pairs require a causal/anaphoric bridge and cannot cross Markdown table
    rows. Adjacent candidates (within 2 lines) are collapsed so one dismissal is reported
    once. Returns ``[{line_no, passage}]``.

    Defensive ``str(... or "")`` throughout — never raises on odd input (r5 discipline)."""
    lines = [str(ln or "") for ln in (lines or [])]
    out: list[dict] = []
    last_center = -10
    for i in range(len(lines)):
        window = lines[max(0, i - 1): i + 2]
        passage = "\n".join(window)
        if not _has_linked_cooccurrence(window):
            continue
        wide = "\n".join(lines[max(0, i - 2): i + 3])
        if _line_exempt(passage, wide):
            continue
        if i - last_center <= 2:      # collapse adjacent windows into one report
            continue
        last_center = i
        snippet = " ".join(p.strip() for p in passage.splitlines() if p.strip())
        out.append({"line_no": i + 1, "passage": snippet[:200]})
    return out


def deterministic_flags(lines: list[str], source: str = "") -> list[str]:
    """Violation messages for one source's ``lines`` (the SoT the confound-gate sister
    reuses). Each message names ``source:line`` + the quoted passage + the exact fix."""
    src = str(source or "?")
    msgs: list[str] = []
    for c in magnitude_dismissal_candidates(lines):
        msgs.append(
            f"{src}:{c['line_no']}: magnitude-based dismissal without relative "
            f"significance or a measured-un-recoverability citation — \"{c['passage']}\""
        )
    return msgs


# ---------------------- fmtools SEMANTIC layer (advisory refine) -----------------------
# Mirrors triality_drift_detector.fm_scope_advisories: the FM runs under the fmtools
# venv in a SUBPROCESS (pact venv gains zero deps), classifying ONLY the pre-filtered
# candidates. Absent / errors / timeout ⇒ silent → the deterministic verdict stands.
_FM_SCRIPT = r'''
import asyncio, json, sys
try:
    import apple_fm_sdk as fm
    from fmtools import local_extract
except Exception:
    print("[]"); raise SystemExit(0)

@fm.generable()
class DismissalCheck:
    verdict: str = fm.guide(
        anyOf=["unjustified_magnitude_dismissal", "justified_or_not_a_dismissal"],
        description="Whether the passage dismisses a lever on absolute smallness alone.")

@local_extract(DismissalCheck, retries=1, instructions=(
    "You judge ONE passage from a machine-learning research memo. Return "
    "'unjustified_magnitude_dismissal' ONLY if the passage DISMISSES a lever / "
    "finding / technique (defers, downgrades, orphans, kills, drops, shelves it) "
    "and justifies that dismissal PURELY by ABSOLUTE smallness ('weak', "
    "'negligible', 'tiny', 'noise', 'not worth it', 'small delta') WITHOUT either "
    "(a) stating its RELATIVE significance — its score delta as a fraction of the "
    "remaining gap to the target operating point — OR (b) citing a MEASUREMENT that "
    "the effect is un-recoverable (an un-recoverability/noise-floor/label-noise "
    "result WITH a measurement or exit criterion, or a structurally-superseded "
    "verdict). Return 'justified_or_not_a_dismissal' if it states relative "
    "significance, cites a measured un-recoverability, is structurally superseded, "
    "or is NOT actually dismissing a lever (e.g. it describes noise injection, weak "
    "supervision, a noise floor, or merely mentions a small number)."))
async def _check(passage: str) -> DismissalCheck:
    """(instructions above)"""

async def _main():
    try:
        cands = json.load(sys.stdin)
    except Exception:
        print("[]"); return
    out = []
    for c in (cands if isinstance(cands, list) else []):
        try:
            r = await _check(str(c.get("passage", ""))[:800])
            if str(getattr(r, "verdict", "")) == "unjustified_magnitude_dismissal":
                out.append(c.get("id", ""))
        except Exception:
            # PER-CANDIDATE fault tolerance: a guardrail trip keeps the candidate
            # (fail toward flagging — the deterministic pre-filter already matched).
            out.append(c.get("id", ""))
    print(json.dumps(out))

asyncio.run(_main())
'''


def _fm_python() -> str | None:
    """The fmtools-venv interpreter (env override → DASH_FM_PYTHON → the known default).
    None when absent — the semantic layer is then silently ABSENT (never a stub)."""
    for cand in (os.environ.get("MAGNITUDE_DISMISSAL_FM_PYTHON"),
                 os.environ.get("DASH_FM_PYTHON"),
                 os.path.expanduser("~/Projects/fmtools/.venv/bin/python")):
        if cand and os.path.exists(cand):
            return cand
    return None


def fm_confirm(candidates: list[dict], timeout: int = 20) -> tuple[list[str], bool]:
    """Run the fmtools confirmer on candidate passages. Returns ``(confirmed_ids, ran)``
    where ``ran`` is True iff fmtools actually executed (so the caller can honestly label
    "fmtools confirmation owed" when it did not). Fail-silent on any error/absence."""
    if not candidates:
        return [], False
    fm_py = _fm_python()
    if not fm_py:
        return [], False
    try:
        payload = [{"id": c["id"], "passage": c["passage"]} for c in candidates]
        proc = subprocess.run(
            [fm_py, "-c", _FM_SCRIPT],
            input=json.dumps(payload), capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return [], False
        out = json.loads(proc.stdout.strip() or "[]")
        ids = [x for x in out if isinstance(x, str)] if isinstance(out, list) else []
        return ids, True
    except Exception:
        return [], False


def build_reason(hits: list[str], fm_ran: bool) -> str:
    """The one firm nudge injected on a confirmed magnitude-dismissal."""
    preview = " | ".join(h for h in hits[:3])
    tail = ("" if fm_ran else
            " (deterministic classifier; on-device fmtools semantic confirmation "
            "OWED — the pact venv has no FM, so this is the high-precision regex "
            "verdict, not a faked FM call.)")
    return (
        "Magnitude-dismissal detector: this turn recorded a DEFER / DOWNGRADE / "
        "ORPHAN / KILL justified by ABSOLUTE magnitude (weak/negligible/noise/"
        "small-ΔS/not-worth-it) WITHOUT (a) a RELATIVE-significance number "
        "(ΔS / (S_current − S_target) at the current operating point) OR (b) a "
        "cited MEASUREMENT of un-recoverability (the #141 label-noise/noise-floor "
        f"case, WITH a measurement). Passage(s): {preview}. Before this dismissal "
        "stands: compute ΔS / remaining-gap at the current operating point and "
        "state BOTH numbers, and adversarially review the verdict "
        "(verdict_scope: INSTANCE — one magnitude-based dismissal is not a kill). "
        "If the effect is genuinely un-recoverable, CITE the measurement (exit "
        "criterion), don't dismiss by eyeball. Deliberate exception: same-line "
        "'# MAGNITUDE_DISMISSAL_OK:<real rationale>', or '[magnitude-ok]' in a "
        "commit message. Memory: relative-not-absolute-significance-near-goal-"
        "dont-orphan-small-deltaS. Sister static gate: "
        "tac.confound_gates.check_no_unjustified_magnitude_dismissal (#404)."
        + tail
    )


# ------------------------------- IO / glue (mirrors drift detector) --------------------
def _git(root: str, *args: str, timeout: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, timeout=timeout)


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


def _persist_advisories(root: str, hits: list[str], head: str) -> None:
    """Durable advisory sink (surfaceable by the costate digest): a Stop hook only
    SHOWS text when it blocks, so record the flag durably regardless."""
    try:
        p = os.path.join(root, ADVISORIES)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as f:
            for h in hits:
                f.write(json.dumps({"ts": _now(), "head": head[:9], "flag": h}) + "\n")
    except Exception:
        pass


def added_lines_from_diff(diff_text: str) -> list[str]:
    """The ADDED lines of a unified diff (leading '+' stripped; '+++' headers skipped)."""
    out: list[str] = []
    for line in str(diff_text or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return out


def _advance_and_allow(root: str, head: str) -> None:
    _write_marker(root, {"last_head": head, "last_block_head": None, "updated_utc": _now()})
    sys.exit(0)


def main() -> None:
    # --- read hook input (fail-open on anything) ---
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
            sys.exit(0)  # not a git repo / no HEAD — fail open
        head = r.stdout.strip()
    except Exception:
        sys.exit(0)

    marker = _read_marker(root)
    last_head = marker.get("last_head")
    last_block_head = marker.get("last_block_head")

    if not last_head:
        _advance_and_allow(root, head)          # first run: initialize to HEAD
    if stop_hook_active or (last_block_head and last_block_head == head):
        _log(root, f"allow(already-nudged) head={head[:9]} sha_active={stop_hook_active}")
        _advance_and_allow(root, head)
    if last_head == head:
        sys.exit(0)                             # no new commits since last stop

    # --- gather this-window subjects/bodies + added lines of .omx/research/*.md ---
    try:
        subjects = [
            s for s in _git(root, "log", "--format=%s%n%b",
                            f"{last_head}..{head}").stdout.splitlines() if s.strip()]
        changed = [
            f for f in _git(root, "diff", "--name-only",
                            f"{last_head}..{head}").stdout.splitlines() if f.strip()]
    except Exception:
        _advance_and_allow(root, head)          # can't read log → fail open

    if is_opted_out(subjects):
        _log(root, f"allow(opted-out) head={head[:9]}")
        _advance_and_allow(root, head)

    # candidate corpus: memo/verdict added lines + the commit messages themselves.
    candidates: list[dict] = []
    try:
        # commit messages (dismissal language in commit bodies)
        for m in deterministic_flags(subjects, source="<commit-message>"):
            candidates.append(m)
        # .omx/research/*.md added lines (DAG feeds / verdict / defer memos)
        research_docs = [f for f in changed
                         if f.startswith(".omx/research/") and f.endswith(".md")]
        per_doc: dict[str, list[str]] = {}
        for f in research_docs:
            try:
                d = _git(root, "diff", f"{last_head}..{head}", "--", f).stdout
            except Exception:
                continue
            per_doc[f] = added_lines_from_diff(d)
        for f, lines in per_doc.items():
            candidates.extend(deterministic_flags(lines, source=f))
    except Exception:
        candidates = []

    if not candidates:
        _log(root, f"allow(clean) head={head[:9]} n={len(subjects)}")
        _advance_and_allow(root, head)

    # attach ids + passages for the fmtools confirmer (pre-filtered only).
    cand_objs = [{"id": f"c{i}", "message": msg,
                  "passage": msg.split(" — ", 1)[-1].strip('"')}
                 for i, msg in enumerate(candidates)]
    confirmed_ids, fm_ran = fm_confirm(cand_objs)
    if fm_ran:
        hits = [c["message"] for c in cand_objs if c["id"] in confirmed_ids]
    else:
        hits = [c["message"] for c in cand_objs]   # deterministic verdict stands

    if not hits:
        _log(root, f"allow(fm-cleared) head={head[:9]} cand={len(candidates)}")
        _advance_and_allow(root, head)

    # --- BLOCK (advisory nudge; do NOT advance last_head so the fix lands in-window) ---
    marker["last_block_head"] = head
    marker["updated_utc"] = _now()
    _write_marker(root, marker)
    _persist_advisories(root, hits, head)
    _log(root, f"BLOCK head={head[:9]} hits={len(hits)} fm_ran={fm_ran}")
    print(json.dumps({"decision": "block", "reason": build_reason(hits, fm_ran)}))
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # absolute last-resort fail-open — never wedge a session.
        sys.exit(0)
