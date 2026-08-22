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
# The subagent-commit-serializer append-only JSONL log. Its committed rows may now
# carry a STRUCTURED triality disposition (``--triality-legs`` / ``--triality-reason``,
# CANONICALIZATION UNIT 2 task #388) — a typed alternative to the ad-hoc
# ``[no-triality]`` commit-message token. When a committed row for this window
# declares a disposition, the core drift block SOFTENS to info (see
# ``triality_disposition_from_rows`` + ``main``).
SERIALIZER_LOG = ".omx/state/commit-serializer.log"

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
#   "pointer 0.19110 UNMOVED (apparatus)" is boilerplate in ~every commit, not a measured  # HISTORICAL_SCORE_LITERAL_OK:quotes_the_2026-07-06_commit_footer_verbatim_to_explain_why_the_pointer_stem_was_dropped_from_the_regex; not a current score claim (live pointer is .omx/state/canonical_frontier_pointer.json), no code reads this number
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


def triality_disposition_from_rows(
    rows: list[dict], window_shas: set[str]
) -> dict | None:
    """The latest COMMITTED serializer-log row that declared a structured triality
    disposition (``--triality-legs``) for a commit in THIS window.

    Returns ``{"legs": [...], "reason": str | None, "head": <short sha>}`` for the
    most recent (file-order-last) qualifying row, else ``None``. A row qualifies
    iff (a) its ``outcome`` is ``"committed"``, (b) it carries a non-empty
    ``triality_legs`` list, and (c) its ``head_after`` short sha matches a commit
    in ``window_shas`` (prefix match either direction, to tolerate short/long sha
    length differences) — OR ``window_shas`` is empty (the window could not be
    resolved), in which case a declared row still counts (fail toward softening,
    consistent with the hook's fail-open philosophy). Defensive throughout: a
    malformed row is skipped, never raised.
    """
    best: dict | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("outcome") != "committed":
            continue
        legs = row.get("triality_legs")
        if not legs:
            continue
        head = str(row.get("head_after") or "").strip()
        if window_shas and (
            not head
            or not any(
                s == head or s.startswith(head) or head.startswith(s)
                for s in window_shas
            )
        ):
            continue
        best = {
            "legs": list(legs) if isinstance(legs, list) else [str(legs)],
            "reason": row.get("triality_reason"),
            "head": head,
        }
    return best


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


# --- RECALL-DEPTH LEG (ddm_hw1, task #785; #713 recall-depth enforcement; operator standing
# complaint 2026-07-30 "I thought there was always a recall step"). The recall-EVIDENCE leg
# (below, in main) checks that DECISION-CLASS docs carry a "STORES CONSULTED" line (EXISTENCE).
# This DEPTH sibling targets LEDGER/DAG APPENDS: a new ledger row / DAG FEED whose commit
# subject AND appended body cite ZERO prior-artifact recall (no STORES-CONSULTED, grep, memory:,
# .omx/research path, per-[[...]] token) is an un-recalled append — the "did you check what we
# already know?" gap. ADVISORY-ONLY by construction (fmtools law: advisory on semantic surfaces,
# NEVER blocking): it appends to an existing block reason if one fires, else persists to the
# durable advisory sink for the costate digest, but NEVER triggers a standalone block. ---

# A ledger is any .omx/research/*ledger*.md (deferral queue, orchestration ledger, ...).
_LEDGER_PAT = re.compile(r"\.omx/research/.*ledger.*\.md$", re.IGNORECASE)
# Tokens that evidence prior-artifact recall was done before the append.
_RECALL_TOKENS = re.compile(
    r"stores?\s+consulted|recall\w*|re-?grep\w*|\bgrep(?:ped|s|ping)?\b|consult\w*|"
    r"per\s+\[\[|memory:|\.omx/research/|prior\s+(?:artifact|memo|finding|row|work)|"
    r"cross-?ref\w*|see\s+\.omx|corpus_query|already\s+(?:measured|built|solved|know)",
    re.IGNORECASE,
)


def is_ledger_or_dag_append(f: str) -> bool:
    """True iff a changed file is a ledger or the canonical DAG (a recall-depth-scoped append)."""
    f = f or ""
    return f.startswith(".omx/research/sub015_DAG_") or bool(_LEDGER_PAT.search(f))


def cites_prior_recall(text: str) -> bool:
    """True iff the text names any prior-artifact recall token (grep/store/memory/path/[[...]])."""
    return bool(_RECALL_TOKENS.search(str(text or "")))


def recall_depth_advisories(
    subjects: list[str], files: list[str], appended_text_by_file: dict[str, str]
) -> list[str]:
    """ADVISORY strings for ledger/DAG appends that cite ZERO prior-artifact recall.

    ``subjects`` = window commit subjects; ``files`` = window files; ``appended_text_by_file``
    = {ledger/DAG path: its added-lines text}. Returns [] when every ledger/DAG append cites
    recall (in the subject OR its added lines), or when there is no ledger/DAG append. Pure +
    testable; advisory-only by construction (the caller never blocks on it). Defensive
    ``str(... or "")`` throughout so odd input never raises."""
    subj_recall = cites_prior_recall(" ".join(str(s or "") for s in subjects))
    out: list[str] = []
    for f in files:
        if not is_ledger_or_dag_append(f):
            continue
        if subj_recall or cites_prior_recall(appended_text_by_file.get(f, "")):
            continue
        out.append(
            f"RECALL-DEPTH (#713; operator 'I thought there was always a recall step'): the "
            f"append to {f} cites ZERO prior-artifact recall — no STORES-CONSULTED / grep / "
            f"memory: / .omx/research path / per-[[...]] token in the commit subject or the "
            f"added lines. Before landing a ledger/DAG append, recall what we already "
            f"measured/built on this topic (tools/corpus_query.py '<topic>') and name it. "
            f"ADVISORY (never blocks)."
        )
    return out


def recall_depth_advisories_safe(
    subjects: list[str], files: list[str], appended_text_by_file: dict[str, str]
) -> list[str]:
    """Fail-open wrapper: any exception in the NEW leg ⇒ [] (silent), so a bug here can neither
    block a session nor perturb the existing legs' verdict."""
    try:
        return recall_depth_advisories(subjects, files, appended_text_by_file)
    except Exception:
        return []


# --- SHIFT-LEFT LEG CLASSIFIER (ddm_hw1, task #785; the "4 backstop fires in one day" tax).
# The Stop-hook legs (missing_legs + consumer_leg) fire AFTER the commit lands, so a missed leg
# is a re-finish. This pure classifier lets the commit serializer SUGGEST the same legs BEFORE
# the commit (a shift-left of the exact same heuristics), so the author tags/settles at commit
# time instead of being backstopped. Advisory only; the serializer never blocks on it. ---
def owed_legs_line(message: str, files: list[str], dsl_diff_text: str = "") -> str:
    """One-line 'triality legs likely owed' suggestion for a pre-commit advisory, or "".

    Reuses the EXACT Stop-hook leg heuristics (``missing_legs`` for DSL/equations +
    ``consumer_leg_missing`` for the consumer leg) so the shift-left suggestion matches the hook
    that would otherwise fire post-commit. ``message`` = the commit subject; ``files`` = staged
    files; ``dsl_diff_text`` = the staged/working DSL diff (optional; enables the consumer leg).
    Respects the [no-triality]/[skip-drift] opt-out. Fail-open: any error ⇒ "" (never breaks a
    commit)."""
    try:
        subjects = [str(message or "")]
        if is_opted_out(subjects):
            return ""
        owed: list[str] = []
        for leg in missing_legs(subjects, files):
            if leg == "DSL":
                owed.append(
                    "DSL (a lever/curriculum/carrier/gauge change — add its Lever factory in "
                    "src/tac/witness_dsl/, not just a trainer flag)"
                )
            elif leg == "equations":
                owed.append(
                    "equations (a measured finding/verdict/byte-close/exact-row — register an "
                    "EmpiricalAnchor in src/tac/canonical_equations/)"
                )
        if consumer_leg_missing_safe(subjects, files, dsl_diff_text):
            owed.append(
                "consumer (public DSL surface grew — update a consumer surface or tag "
                "[consumers-generic])"
            )
        if not owed:
            return ""
        return (
            "triality legs likely owed (shift-left; tag [no-triality]/[consumers-generic] or "
            "settle now to avoid the Stop-hook backstop): " + " · ".join(owed)
        )
    except Exception:
        return ""


# --- VERDICT-SCOPE LEG (requirement R, operator 2026-07-08: "ensure no falsifications or
# negative interpretations made based on naive or toy or binary; many techniques have
# multiple ways of formulation and one failure does not mean family dead"). Every negative
# verdict in a decision-class doc must NAME its scope on the 4-level ladder
# INSTANCE < FORMULATION < FAMILY < PARADIGM (defaulting narrowest); a FAMILY kill needs a
# theorem/citation or ≥2 structurally distinct formulations; a KILL/NO-GO at ≥ formulation
# names its reformulation queue. Deterministic layer BLOCKS on missing declarations only;
# the semantic over-scope judgment is the ADVISORY fmtools layer below (never a block).
# Judged on the window's ADDED diff lines, so appending one FEED to a large historical doc
# never demands retro-declarations for old verdicts. Fail-open throughout. ---

# Decision-class docs (extends the recall-evidence leg's pattern with the verdict-carrying
# surfaces: verdict/probe/review memos + DAG feeds).
VERDICT_DOC_PAT = re.compile(
    r"\.omx/research/.*(council|crucible|symposium|GROUNDING|position_S|DRAFT_|convening|"
    r"verdict|probe|review|sub015_DAG_)",
    re.IGNORECASE,
)
# Docs that are themselves rule/ledger/memory/index surfaces — they QUOTE verdicts and
# state the rule; scanning them would self-trip the discipline they encode.
VERDICT_DOC_EXEMPT = re.compile(
    r"(ORCHESTRATION_LEDGER|ledger|MEMORY|CANONICAL_RESEARCH_INDEX|verdict[-_]scope)",
    re.IGNORECASE,
)

# Load-bearing negative-verdict tokens. Single words are CASE-SENSITIVE UPPERCASE — this
# repo writes real verdicts uppercase ("Lever-D MEASURED NO-GO", "INERT", "FALSIFIED");
# lowercase "killed the process" / "dead code" / "no-go areas" prose stays silent.
_NEG_VERDICT_CS = re.compile(r"\b(?:KILL(?:ED|S)?|NO[-_]GO|FALSIFIED|REFUTED|DEAD|INERT)\b")
# Multiword phrases are inherently low-noise → case-insensitive.
_NEG_VERDICT_CI = re.compile(
    r"\b(?:family(?:\s+is)?\s+dead|at\s+chance|does\s+not\s+work)\b", re.IGNORECASE)
# The KILL-class subset (disposition kills) that requires a reformulation queue at
# scope ≥ formulation (req R binding rule (b)).
_KILL_CLASS = re.compile(r"\b(?:KILL(?:ED|S)?|NO[-_]GO)\b")

# The declaration: ``verdict_scope: formulation`` (one per verdict, or repeated as a
# scope table — each row its own ``verdict_scope:`` line).
_SCOPE_DECL = re.compile(
    r"verdict[_-]scope\s*[:=]\s*(instance|formulation|family|paradigm)", re.IGNORECASE)
# FAMILY-scope evidence: a citation (arXiv/DOI/theorem/impossibility bound) OR an
# explicit ≥2-distinct-formulations line.
_FAMILY_EVIDENCE = re.compile(
    r"(arxiv|\bdoi\b|theorem|impossibility\s+bound|"
    r"(?:≥|>=)\s*2\s+(?:\w+\s+)?formulations|two\s+(?:\w+\s+)?(?:distinct\s+)?formulations|"
    r"2\+\s+(?:\w+\s+)?formulations)",
    re.IGNORECASE,
)
# Reformulation queue: the kill memo enumerates the untested alternatives.
_REFORMULATION = re.compile(
    r"(reformulation|untested\s+formulations|alternatives\s*:)", re.IGNORECASE)

# Same-line waiver (placeholder rationales rejected, per Catalog #287 sister discipline).
_SCOPE_WAIVER = re.compile(r"#\s*VERDICT_SCOPE_OK\s*:\s*(\S.*)")
_WAIVER_PLACEHOLDER = re.compile(
    r"^<[^>]*>$|^(?:todo|tbd|reason|rationale|xxx)\.?$", re.IGNORECASE)
# Negation / prior-verdict-discussion cues: the line is ABOUT a verdict, not making one.
_NEG_DISCUSSION_CUE = re.compile(
    r"\b(?:not\s+a|never|no\s+longer|reopened?|over-?scop\w*|premature\w*|"
    r"prior\s+verdict|previous(?:ly)?|was\s+read\s+as|had\s+been|instead\s+of|"
    r"rather\s+than|would\s+(?:be|have)|avoid\w*)\b",
    re.IGNORECASE,
)
# Quoted spans (backtick code + straight double quotes) are stripped before token search —
# quoting a prior verdict is not issuing one.
_QUOTED_SPANS = re.compile(r"`[^`]*`|\"[^\"]*\"")


def added_lines_from_diff(diff_text: str) -> list[str]:
    """The ADDED lines of a unified diff (leading '+' stripped; '+++' headers skipped).
    Defensive str-coercion; never raises."""
    out: list[str] = []
    for line in str(diff_text or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return out


def _has_valid_scope_waiver(line: str) -> bool:
    """True iff the line carries # VERDICT_SCOPE_OK:<rationale> with a REAL rationale
    (placeholder literals like <rationale>/TBD rejected; ≥4 chars required)."""
    m = _SCOPE_WAIVER.search(str(line or ""))
    if not m:
        return False
    rationale = m.group(1).strip()
    return len(rationale) >= 4 and not _WAIVER_PLACEHOLDER.match(rationale)


# The subagent contract's own DEAD-ENDS section header (src/tac/subagent_contract.py:528
# instructs every arm to emit it). A heading NAMES a section; the verdicts are the rows
# beneath it, which carry their own scopes. MEASURED 2026-08-22: 198 of 372 arm finals
# carry this header — one template counted 198 times (#821's genus), and the case-sensitive
# heuristic above cannot save it because the header is legitimately uppercase. Deliberately
# ANCHORED to the bare header line: "## DEAD-END: the X family" still fires, and so does
# any prose use, because those ARE claims.
_TEMPLATE_SECTION_HEADER = re.compile(r"^#{1,6}\s+DEAD[-_]ENDS\s*$")


def _verdict_line_exempt(line: str) -> bool:
    """Line-level exemptions: markdown quotes, the contract's own DEAD-ENDS template
    header, rule-discussion lines (mention the declaration key itself), negation/
    prior-verdict cues, and a valid same-line waiver."""
    s = str(line or "").strip()
    if s.startswith(">"):
        return True  # markdown quote — quoting a prior verdict, not issuing one
    if _TEMPLATE_SECTION_HEADER.match(s):
        return True  # a section label, not a verdict — the rows beneath carry the scopes
    if _SCOPE_WAIVER.search(s):
        # a waiver marker decides the line by its OWN validity — it must not fall
        # through to the rule-discussion exemption below ("VERDICT_SCOPE_OK" would
        # otherwise self-exempt via its "verdict_scope" substring).
        return _has_valid_scope_waiver(s)
    low = s.lower()
    if "verdict_scope" in low or "verdict-scope" in low:
        return True  # the declaration line / discussing this rule
    return bool(_NEG_DISCUSSION_CUE.search(s))


def negative_verdict_tokens(added_lines: list[str]) -> tuple[list[str], bool]:
    """(tokens, kill_class_present) over non-exempt added lines, with quoted/backtick
    spans stripped so quoted prior verdicts stay silent."""
    tokens: list[str] = []
    kill = False
    for line in added_lines:
        if _verdict_line_exempt(line):
            continue
        searchable = _QUOTED_SPANS.sub("", str(line or ""))
        hits = [m.group(0) for m in _NEG_VERDICT_CS.finditer(searchable)]
        hits += [m.group(0) for m in _NEG_VERDICT_CI.finditer(searchable)]
        if hits:
            tokens.extend(hits)
            if _KILL_CLASS.search(searchable):
                kill = True
    return tokens, kill


def verdict_scope_violations(path: str, added_lines: list[str]) -> list[str]:
    """The deterministic (BLOCKING) requirement-R checks for ONE doc's added lines.
    Returns violation messages (empty == compliant). Each message shows the exact
    one-line fix so compliance is trivial."""
    tokens, kill_present = negative_verdict_tokens(added_lines)
    if not tokens:
        return []
    text = "\n".join(str(ln or "") for ln in added_lines)
    scopes = [m.group(1).lower() for m in _SCOPE_DECL.finditer(text)]
    if not scopes:
        uniq = sorted(set(tokens))[:4]
        return [
            f"{path}: negative-verdict token(s) {uniq} without a verdict_scope "
            "declaration — add: 'verdict_scope: formulation — <which formulation>' "
            "(or instance/family/paradigm; NARROWEST level the measurement supports)"
        ]
    viol: list[str] = []
    if any(s == "family" for s in scopes) and not _FAMILY_EVIDENCE.search(text):
        viol.append(
            f"{path}: 'verdict_scope: family' requires a citation (arXiv/DOI/theorem) "
            "OR an explicit '≥2 structurally distinct formulations' evidence line "
            "(req R: a family kill needs a theorem or kills across ≥2 formulations)"
        )
    if (kill_present
            and any(s in ("formulation", "family", "paradigm") for s in scopes)
            and not _REFORMULATION.search(text)):
        viol.append(
            f"{path}: KILL/NO-GO at scope ≥ formulation requires a reformulation "
            "queue — add: 'untested formulations / alternatives: <enumerate them>'"
        )
    return viol


def verdict_doc_in_scope(path: str) -> bool:
    """True iff this changed file is a decision-class .md the verdict-scope leg scans."""
    f = str(path or "")
    return (f.endswith(".md") and bool(VERDICT_DOC_PAT.search(f))
            and not VERDICT_DOC_EXEMPT.search(f))


def verdict_scope_evidence_text(added_lines: list[str], limit: int = 1800) -> str:
    """Evidence for the FM advisory: added lines MINUS declaration lines (measured
    2026-07-08: including the declaration makes the FM echo it back)."""
    keep = [str(ln or "") for ln in added_lines
            if "verdict_scope" not in str(ln or "").lower()]
    return "\n".join(keep)[:limit]


# --- SCHEDULE-PROVENANCE LEG (operator 2026-07-09, verbatim fury: "Fuck pr95... Never do it
# again. Add a gate and hook... move from hardcoded epochs to event based and deep math governed
# and costate controller"). Sister of the LAUNCHER-path STRICT gate (tools/schedule_provenance_
# gate.py): the launcher refuses EMISSION of a naked primary epoch; this leg flags AUTHORING it.
# A commit touching a schedule-owning surface (witness_autoconfig.py / witness_dsl / a launch
# config) that ADDS a naked hardcoded-epoch schedule param (a NEW *_start_epoch delta assignment
# or a --*-start-epoch argparse default, POSITIVE int) WITHOUT a co-added event/derived/cap
# provenance token — or that ADDS a NEW PR95-named stage sequence — drifts. Deterministic,
# window-granular (same grain as the other legs), fail-open, same-line waiver
# `# SCHEDULE_PROVENANCE_OK:<real rationale>`. Memory:
# elementwise_audits_launder_structural_cargocult_pr95_skeleton_20260709. ---
SCHEDULE_GOV_FILE_PAT = re.compile(
    r"(src/tac/witness_autoconfig\.py$|src/tac/witness_dsl/|"
    r"\.omx/operator_authorize_recipes/.+\.ya?ml$|launch[^/]*config)", re.IGNORECASE)
# a NEW schedule-epoch DEFINITION on an added line: a delta/config assignment
#   "<x>_start_epoch": 300  |  <x>_start_epoch = 300  |  argparse default for a --*-start-epoch.
_SCHED_EPOCH_ASSIGN = re.compile(r'["\']?(\w*_start_epoch)["\']?\s*[:=]\s*(\d+)')
_SCHED_EPOCH_ARGPARSE = re.compile(
    r'add_argument\(\s*"(--[a-z0-9-]*-start-epoch)"[^\n]*?default\s*=\s*(\d+)')
# PR95 discrete-stage NAMES — a NEW sequence literal carrying >= 2 distinct ones is the
# PR95-skeleton smell (CE->tau_softplus->l7->Muon at proportional boundaries).
_PR95_STAGE = re.compile(
    r"\b(ce|tau_softplus|l7|muon|smooth|qat|c1a|sigma|lambda_sweep)\b", re.IGNORECASE)
_PR95_SEQUENCE_CUE = re.compile(r"(stage|curriculum|schedule|sequence|phase)", re.IGNORECASE)
# governance / derivation tokens: presence in the window's added lines means the epoch is (or
# is being) governed by an event/law/cap — no drift (matches the launcher gate's three classes).
_SCHED_GOV_TOKEN = re.compile(
    r"(LawRef|equation_id|constants_manifest|schedule_governance|curriculum.?event.?triggered|"
    r"nucleus.?guard|plateau.?trigger|closed.?loop|event.?triggered|fail.?safe\s+cap|"
    r"derived.?at.?config|derived\s+from)", re.IGNORECASE)
_SCHED_WAIVER = re.compile(r"#\s*SCHEDULE_PROVENANCE_OK\s*:\s*(\S.*)")


def schedule_gov_file_in_scope(path: str) -> bool:
    """True iff this changed file is a schedule-owning surface the schedule-provenance leg
    scans (witness_autoconfig.py / witness_dsl / a launch-config recipe)."""
    return bool(SCHEDULE_GOV_FILE_PAT.search(str(path or "")))


def _has_valid_schedule_waiver(line: str) -> bool:
    """True iff the line carries # SCHEDULE_PROVENANCE_OK:<rationale> with a REAL rationale
    (placeholder literals rejected; >= 4 chars required — reuses the verdict-scope discipline)."""
    m = _SCHED_WAIVER.search(str(line or ""))
    if not m:
        return False
    rationale = m.group(1).strip()
    return len(rationale) >= 4 and not _WAIVER_PLACEHOLDER.match(rationale)


def naked_epoch_additions(added_lines: list[str]) -> list[str]:
    """The added lines that DEFINE a positive hardcoded schedule epoch (a *_start_epoch delta
    assignment or a --*-start-epoch argparse default), minus lines carrying a valid same-line
    waiver. Returns short ``name value`` descriptors (empty == none)."""
    out: list[str] = []
    for line in added_lines:
        s = str(line or "")
        if _has_valid_schedule_waiver(s):
            continue
        for pat in (_SCHED_EPOCH_ASSIGN, _SCHED_EPOCH_ARGPARSE):
            m = pat.search(s)
            if m:
                try:
                    if int(m.group(2)) > 0:
                        out.append(f"{m.group(1)} {m.group(2)}")
                except (TypeError, ValueError):
                    pass
                break
    return out


def pr95_stage_sequence_additions(added_lines: list[str]) -> list[str]:
    """Added lines that introduce a NEW PR95-named stage SEQUENCE (a stage/curriculum/schedule
    line naming >= 2 DISTINCT PR95 stages), minus valid-waiver lines. A stage NAME from the
    incumbent is itself a framing smell (memory: naming is framing)."""
    out: list[str] = []
    for line in added_lines:
        s = str(line or "")
        if _has_valid_schedule_waiver(s) or not _PR95_SEQUENCE_CUE.search(s):
            continue
        names = {m.group(1).lower() for m in _PR95_STAGE.finditer(s)}
        if len(names) >= 2:
            out.append(", ".join(sorted(names)))
    return out


def window_has_schedule_gov_cite(all_added_lines: list[str]) -> bool:
    """True iff the window's added lines carry a schedule-governance / derivation token
    (LawRef / schedule_governance / an event sensor / fail-safe cap / derived-from). Window
    granularity (same accepted bound as the other legs): a proper migration that adds the
    epoch AND its governance in the same turn is not nagged."""
    return any(_SCHED_GOV_TOKEN.search(str(ln or "")) for ln in all_added_lines)


def schedule_provenance_violations(subjects: list[str], per_file_added: dict) -> list[str]:
    """The schedule-provenance leg's messages over the window's governed-file diffs.
    ``per_file_added`` maps in-scope path -> its added lines. Window-wide opt-out via
    [no-triality]/[skip-drift]; window-wide governance-cite suppression; per-line waiver.
    Returns violation messages (empty == compliant). Fail-open is the caller's wrapper."""
    if is_opted_out(subjects):
        return []
    all_added = [ln for lines in per_file_added.values() for ln in lines]
    if window_has_schedule_gov_cite(all_added):
        return []
    viol: list[str] = []
    for path, added in per_file_added.items():
        naked = naked_epoch_additions(added)
        if naked:
            viol.append(
                f"{path}: NEW naked hardcoded schedule epoch(s) {naked[:4]} — a --*-start-epoch "
                "trigger must be EVENT-governed / LawRef-DERIVED / a TAGGED fail-safe cap")
        pr95 = pr95_stage_sequence_additions(added)
        if pr95:
            viol.append(
                f"{path}: NEW PR95-named stage sequence(s) {pr95[:3]} without a derivation cite "
                "(derive the SHAPE from the witness math, or cite the derivation)")
    return viol


def schedule_provenance_violations_safe(subjects: list[str], per_file_added: dict) -> list[str]:
    """Fail-open wrapper for the schedule-provenance leg: any exception => [] (silent), so a
    bug here can neither block a session nor perturb the existing legs' verdict."""
    try:
        return schedule_provenance_violations(subjects, per_file_added)
    except Exception:
        return []


# --- DSL-CONFIG-BYPASS LEG (operator 2026-07-08, verbatim: "The config must be defined in the DSL —
# no ad hoc or hand crafting ... pydantic ... integrate all with apparatus to prevent more dumbass
# bullshit"). Sister of the LAUNCHER-path DSL-authored-config gate (b0.6) + the confound preflight
# gate: a commit that ADDS a NEW config-emitting function (a `derive_*_config` / a `*_flags` argv
# assembler) OUTSIDE src/tac/witness_dsl WITHOUT routing it through the typed DSL layer
# (tac.witness_dsl.typed_config: TypedWitnessConfig / to_program / build_launch_manifest) is a NEW
# parallel hand-assembly path — the exact confound the operator forbids. Window-granular, fail-open,
# same-line waiver `# DSL_CONFIG_BYPASS_OK:<real rationale>`. Memory:
# config_must_be_dsl_defined_typed_validated_no_adhoc_20260708. ---
DSL_CONFIG_FILE_PAT = re.compile(
    r"(src/tac/witness_autoconfig\.py$|src/tac/witness_config[^/]*\.py$)", re.IGNORECASE)
# a NEW config-emitting function definition on an added line.
_NEW_CONFIG_EMITTER = re.compile(r'^\+?\s*def\s+(derive_\w*config|_?\w*_flags)\s*\(')
# tokens proving the added code routes through the typed DSL layer (a proper authoring path).
_TYPED_CITE = re.compile(
    r"(typed_config|TypedWitnessConfig|to_program|build_launch_manifest|dsl_program_manifest|"
    r"_attach_dsl_program_manifest)")
_DSL_CONFIG_WAIVER = re.compile(r"#\s*DSL_CONFIG_BYPASS_OK\s*:\s*(\S.*)")


def dsl_config_file_in_scope(path: str) -> bool:
    """True iff this changed file is a config-authoring surface the DSL-config leg scans
    (witness_autoconfig.py and sibling witness_config modules OUTSIDE witness_dsl)."""
    p = str(path or "")
    if "src/tac/witness_dsl/" in p:  # the typed DSL path itself is the SANCTIONED authoring surface
        return False
    return bool(DSL_CONFIG_FILE_PAT.search(p))


def _has_valid_dsl_config_waiver(line: str) -> bool:
    m = _DSL_CONFIG_WAIVER.search(str(line or ""))
    if not m:
        return False
    rationale = m.group(1).strip()
    return len(rationale) >= 4 and not _WAIVER_PLACEHOLDER.match(rationale)


def new_config_emitter_additions(added_lines: list[str]) -> list[str]:
    """Added lines that DEFINE a NEW config-emitting function (a derive_*_config / *_flags argv
    assembler), minus lines carrying a valid same-line waiver. Returns the function names."""
    out: list[str] = []
    for line in added_lines:
        s = str(line or "")
        if _has_valid_dsl_config_waiver(s):
            continue
        m = _NEW_CONFIG_EMITTER.search(s)
        if m:
            out.append(m.group(1))
    return out


def window_cites_typed_dsl(all_added_lines: list[str]) -> bool:
    """True iff the window's added lines route through the typed DSL layer (a proper migration
    that adds the config-emitter AND its typed authoring/validation in the same turn)."""
    return any(_TYPED_CITE.search(str(ln or "")) for ln in all_added_lines)


def dsl_config_bypass_violations(subjects: list[str], per_file_added: dict) -> list[str]:
    """The DSL-config-bypass leg's messages over the window's config-file diffs. A NEW config
    emitter added outside witness_dsl, in a window that does NOT cite the typed DSL layer, drifts.
    Window-wide opt-out via [no-triality]/[skip-drift]; per-line waiver. Fail-open in the wrapper."""
    if is_opted_out(subjects):
        return []
    all_added = [ln for lines in per_file_added.values() for ln in lines]
    if window_cites_typed_dsl(all_added):
        return []
    viol: list[str] = []
    for path, added in per_file_added.items():
        emitters = new_config_emitter_additions(added)
        if emitters:
            viol.append(
                f"{path}: NEW config-emitting function(s) {emitters[:4]} added outside "
                "src/tac/witness_dsl without routing through the typed DSL layer "
                "(tac.witness_dsl.typed_config)")
    return viol


def dsl_config_bypass_violations_safe(subjects: list[str], per_file_added: dict) -> list[str]:
    """Fail-open wrapper: any exception => [] (silent), so a bug here can neither block a session
    nor perturb the existing legs' verdict."""
    try:
        return dsl_config_bypass_violations(subjects, per_file_added)
    except Exception:
        return []


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


def _read_serializer_rows(root: str, tail: int = 800) -> list[dict]:
    """Parse the tail of the subagent-commit-serializer JSONL log. Fail-open:
    missing/unreadable log ⇒ [] (the softening simply does not apply)."""
    path = os.path.join(root, SERIALIZER_LOG)
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return []
    rows: list[dict] = []
    for ln in lines[-tail:]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _window_commit_shas(root: str, last_head: str, head: str) -> set[str]:
    """The full commit shas in the ``last_head..head`` window. Fail-open ⇒ set()."""
    try:
        out = _git(root, "rev-list", f"{last_head}..{head}").stdout
        return {s.strip() for s in out.splitlines() if s.strip()}
    except Exception:
        return set()


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


# --- VERDICT-SCOPE fmtools ADVISORY layer (semantic; NEVER a block) ------------------
# Runs ONLY when a decision-class doc in the window declared a family/paradigm-scope
# negative verdict and passed the deterministic layer. Isolation per the established
# dashboard_fm_events pattern: the FM runs under the fmtools venv in a SUBPROCESS —
# the pact venv gains zero deps; fmtools absent / model unavailable / timeout ⇒ silent.
# CALIBRATION (measured 2026-07-08, 3-case probe): the on-device FM discriminated
# instance-vs-family evidence correctly (cold 0.82s / warm 0.24s per call) but judged a
# genuine formulation-level case as instance — so advisories fire ONLY for declared
# family/paradigm (where over-scoping poisons the corpus), never for declared
# instance/formulation (unreliable there → would spam false advisories).
_FM_SCOPE_SCRIPT = r'''
import asyncio, json, sys
try:
    import apple_fm_sdk as fm
    from fmtools import local_extract
except Exception:
    print("[]"); raise SystemExit(0)

@fm.generable()
class ScopeCheck:
    supported_scope: str = fm.guide(
        anyOf=["instance", "formulation", "family"],
        description="The NARROWEST scope level this evidence supports.")

@local_extract(ScopeCheck, retries=1, instructions=(
    "You judge what SCOPE a negative experimental result's evidence supports, on a "
    "3-level ladder. 'instance' = the evidence tests ONE configuration/checkpoint/"
    "constant/scale only. 'formulation' = the evidence adequately tests one "
    "mathematical form of a technique (adequate scale, tuned), but only that one "
    "form. 'family' = the evidence includes a mathematical theorem/impossibility "
    "bound, OR failures across at least TWO structurally distinct formulations at "
    "adequate scale. Pick the NARROWEST level the evidence text itself supports."))
async def _check(evidence_text: str) -> ScopeCheck:
    """(instructions above)"""

_ORDER = {"instance": 0, "formulation": 1, "family": 2, "paradigm": 3}

async def _main():
    try:
        cands = json.load(sys.stdin)
    except Exception:
        print("[]"); return
    out = []
    for c in (cands if isinstance(cands, list) else []):
        try:
            declared = [d for d in c.get("declared", []) if d in ("family", "paradigm")]
            if not declared:
                continue
            top = max(declared, key=lambda d: _ORDER[d])
            r = await _check(str(c.get("evidence", ""))[:1800])
            judged = str(getattr(r, "supported_scope", "") or "")
            if judged in _ORDER and _ORDER[judged] < _ORDER[top]:
                out.append(
                    f"{c.get('path', '?')}: declared verdict_scope '{top}' but the "
                    f"evidence text reads {judged}-level — re-check the scope "
                    "(on-device FM, advisory only)")
        except Exception:
            continue
    print(json.dumps(out))

asyncio.run(_main())
'''

VERDICT_SCOPE_ADVISORIES = ".omx/state/verdict_scope_advisories.jsonl"


def _fm_python() -> str | None:
    """The fmtools-venv interpreter (env override → DASH_FM_PYTHON → the known default).
    None when absent — the advisory layer is then silently ABSENT (never a stub)."""
    for cand in (os.environ.get("VERDICT_SCOPE_FM_PYTHON"),
                 os.environ.get("DASH_FM_PYTHON"),
                 os.path.expanduser("~/Projects/fmtools/.venv/bin/python")):
        if cand and os.path.exists(cand):
            return cand
    return None


def fm_scope_advisories(candidates: list[dict], timeout: int = 20) -> list[str]:
    """Run the fmtools over-scope classifier on the candidate docs. Fail-silent []
    on any error/timeout/absence — semantic judgment is advisory by construction."""
    if not candidates:
        return []
    fm_py = _fm_python()
    if not fm_py:
        return []
    try:
        proc = subprocess.run(
            [fm_py, "-c", _FM_SCOPE_SCRIPT],
            input=json.dumps(candidates), capture_output=True, text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return []
        out = json.loads(proc.stdout.strip() or "[]")
        return [a for a in out if isinstance(a, str)] if isinstance(out, list) else []
    except Exception:
        return []


def _persist_scope_advisories(root: str, advisories: list[str], head: str) -> None:
    """Durable advisory sink (surfaced by tools/costate_digest.py): the Stop hook can
    only SHOW text when it blocks, and the advisory layer never blocks alone."""
    try:
        p = os.path.join(root, VERDICT_SCOPE_ADVISORIES)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as f:
            for a in advisories:
                f.write(json.dumps(
                    {"ts": _now(), "head": head[:9], "advisory": a}) + "\n")
    except Exception:
        pass


def _advance_and_allow(root: str, head: str) -> None:
    """Advance the marker to HEAD, clear block state, allow the stop."""
    _write_marker(root, {"last_head": head, "last_block_head": None, "updated_utc": _now()})
    sys.exit(0)


def main() -> None:
    # --- read hook input (fail-open on anything) ---
    try:
        # isatty guard (ddm_gh2, 2026-07-31): a bare sys.stdin.read() BLOCKS — it does
        # not raise — when stdin is a TTY or an inherited pipe nobody closes, so the
        # try/except below cannot save it. MEASURED: this hook ran >119s (never
        # returned) under a harness that left stdin open. Under a TTY it hangs until
        # Ctrl-D, which makes the tool un-runnable by hand. Sister hooks
        # codex_landing_review_gate.py:516 and auto_push_main.py:422 already carry
        # this guard; this makes the Stop-hook family consistent.
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
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

    # --- RECALL-EVIDENCE LEG (operator binding 2026-07-07: "you recorded that before and did
    # it again" — prose never fixed a behavior class, only enforcement did). DECISION-CLASS docs
    # (council/crucible charters, grounding packets, seat positions, design drafts, convening
    # records) must carry a "STORES CONSULTED" line naming which durable stores were recalled
    # before the conclusions were written. Fail-open independently; warn-grade nudge (folded
    # into the block reason only when a block already fires OR standalone as its own block). ---
    recall_missing: list[str] = []
    try:
        import re as _re
        decision_pat = _re.compile(
            r"\.omx/research/.*(council|crucible|symposium|GROUNDING|position_S|DRAFT_|convening)",
            _re.IGNORECASE)
        for f in files:
            if decision_pat.search(f) and f.endswith(".md"):
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    head_txt = open(fp, errors="replace").read(6000)
                    if ("STORES CONSULTED" not in head_txt.upper()
                            and "ORCHESTRATION_LEDGER" not in f):
                        recall_missing.append(f)
    except Exception:
        recall_missing = []

    # --- RECALL-DEPTH LEG (#713; ddm_hw1 task #785; fail-open independently). A ledger/DAG
    # append whose subject AND added lines cite ZERO prior-artifact recall is un-recalled.
    # ADVISORY-ONLY: appended to a block reason if one fires, else persisted to the durable
    # advisory sink for the costate digest — NEVER a standalone block trigger. ---
    recall_depth_advs: list[str] = []
    try:
        appended_by_file: dict[str, str] = {}
        for f in files:
            if not is_ledger_or_dag_append(f):
                continue
            try:
                fdiff = _git(root, "diff", f"{last_head}..{head}", "--", f).stdout
            except Exception:
                continue
            appended_by_file[f] = "\n".join(added_lines_from_diff(fdiff))
        if appended_by_file:
            recall_depth_advs = recall_depth_advisories_safe(subjects, files, appended_by_file)
    except Exception:
        recall_depth_advs = []
    if recall_depth_advs:
        _persist_scope_advisories(root, recall_depth_advs, head)

    # --- VERDICT-SCOPE LEG (requirement R; fail-open independently). Deterministic
    # layer: a decision-class doc whose ADDED lines carry load-bearing negative-verdict
    # tokens must declare verdict_scope (+ family-citation + reformulation-queue rules)
    # → BLOCK with the exact one-line fix. Semantic layer: fmtools over-scope check on
    # compliant family/paradigm declarations → ADVISORY only (appended to an existing
    # block, else persisted for the costate digest). ---
    scope_violations: list[str] = []
    fm_candidates: list[dict] = []
    try:
        for f in files:
            if not verdict_doc_in_scope(f):
                continue
            try:
                doc_diff = _git(root, "diff", f"{last_head}..{head}", "--", f).stdout
            except Exception:
                continue
            added = added_lines_from_diff(doc_diff)
            viols = verdict_scope_violations(f, added)
            scope_violations.extend(viols)
            if not viols:
                text = "\n".join(added)
                decls = [m.group(1).lower() for m in _SCOPE_DECL.finditer(text)]
                if decls and negative_verdict_tokens(added)[0]:
                    fm_candidates.append({
                        "path": f, "declared": decls,
                        "evidence": verdict_scope_evidence_text(added),
                    })
    except Exception:
        scope_violations, fm_candidates = [], []
    fm_advisories = fm_scope_advisories(fm_candidates) if fm_candidates else []
    if fm_advisories:
        _persist_scope_advisories(root, fm_advisories, head)

    # --- SCHEDULE-PROVENANCE LEG (requirement: operator 2026-07-09 "move from hardcoded
    # epochs to event based"; fail-open independently). A commit touching a schedule-owning
    # surface (witness_autoconfig.py / witness_dsl / a launch config) that ADDS a naked
    # hardcoded-epoch schedule param — or a NEW PR95-named stage sequence — WITHOUT a co-added
    # event/derived/cap provenance token drifts. Sister of the launcher-path STRICT gate. ---
    schedule_violations: list[str] = []
    try:
        per_file_added: dict[str, list[str]] = {}
        for f in files:
            if not schedule_gov_file_in_scope(f):
                continue
            try:
                fdiff = _git(root, "diff", f"{last_head}..{head}", "--", f).stdout
            except Exception:
                continue
            per_file_added[f] = added_lines_from_diff(fdiff)
        if per_file_added:
            schedule_violations = schedule_provenance_violations_safe(subjects, per_file_added)
    except Exception:
        schedule_violations = []

    # --- DSL-CONFIG-BYPASS LEG (operator 2026-07-08 "config must be defined in the DSL, no ad hoc";
    # fail-open independently). A commit adding a NEW config-emitter (derive_*_config / *_flags) outside
    # witness_dsl WITHOUT routing through the typed DSL layer drifts. Sister of the launcher b0.6 gate. ---
    dsl_config_violations: list[str] = []
    try:
        dsl_cfg_added: dict[str, list[str]] = {}
        for f in files:
            if not dsl_config_file_in_scope(f):
                continue
            try:
                fdiff = _git(root, "diff", f"{last_head}..{head}", "--", f).stdout
            except Exception:
                continue
            dsl_cfg_added[f] = added_lines_from_diff(fdiff)
        if dsl_cfg_added:
            dsl_config_violations = dsl_config_bypass_violations_safe(subjects, dsl_cfg_added)
    except Exception:
        dsl_config_violations = []

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

    # --- CANONICALIZATION UNIT 2 (task #388): structured triality-legs softening.
    # A serializer ``--triality-legs`` declaration (recorded in the commit log) is a
    # STRUCTURED alternative to the ad-hoc [no-triality] token: when a committed row
    # for THIS window declared legs (or none+reason), the author consciously
    # dispositioned the triality, so SOFTEN the core classify() drift to info (log +
    # do not block). This softens ONLY the core DAG/DSL/equations drift; the
    # independent legs (consumer/recall/scope/schedule/dsl-config) keep their own
    # disciplines + waivers. Fail-open: any error ⇒ no disposition (block as before).
    core_drift = classify(subjects, files) == "drift"
    triality_disposition: dict | None = None
    try:
        triality_disposition = triality_disposition_from_rows(
            _read_serializer_rows(root), _window_commit_shas(root, last_head, head)
        )
    except Exception:
        triality_disposition = None
    if core_drift and triality_disposition is not None:
        _log(
            root,
            f"soften(triality-legs disposition legs={triality_disposition.get('legs')} "
            f"reason={triality_disposition.get('reason')!r}) head={head[:9]}",
        )
        core_drift = False

    if (core_drift or consumer_missing or recall_missing
            or scope_violations or schedule_violations or dsl_config_violations):
        # Persist last_block_head (do NOT advance last_head — so the fix/DAG-FEED commit lands
        # inside the next window's UNION and clears the drift on re-check; the last_block_head
        # backstop clears the block once head is unchanged, so it cannot loop).
        marker["last_block_head"] = head
        marker["updated_utc"] = _now()
        _write_marker(root, marker)
        _log(
            root,
            f"BLOCK head={head[:9]} n={len(subjects)} miss={missing_legs(subjects, files)}"
            f" consumer_leg={consumer_missing} scope_violations={len(scope_violations)}"
            f" schedule_violations={len(schedule_violations)}"
            f" dsl_config_violations={len(dsl_config_violations)}",
        )
        if core_drift:
            reason = build_reason(subjects, files)
            if consumer_missing:
                reason = reason + " ALSO — " + CONSUMER_NUDGE
        elif consumer_missing:
            reason = CONSUMER_NUDGE
        else:
            reason = ""
        if recall_missing:
            reason = (reason + (" ALSO — " if reason else "") +
                      "RECALL-EVIDENCE missing: decision-class doc(s) "
                      f"{recall_missing[:3]} lack a 'STORES CONSULTED:' line naming which "
                      "durable stores (memories/DAG/equations/DSL/tasks/run-artifacts) were "
                      "recalled before concluding. Add the line (list the stores actually "
                      "consulted — honestly, including 'none' if so) and re-finish. "
                      "This is the enforcement hook for the recall-before-decide class "
                      "(L27/L28/L83): prose rules recurred; this fires without volition.")
        if scope_violations:
            reason = (reason + (" ALSO — " if reason else "") +
                      "VERDICT-SCOPE (req R: one failed formulation is NOT a dead family): "
                      + " | ".join(scope_violations[:3]) +
                      " Ladder: INSTANCE < FORMULATION < FAMILY < PARADIGM — declare the "
                      "NARROWEST level the measurement supports (naive/toy/binary can only "
                      "produce INSTANCE-level negatives). Deliberate exception: same-line "
                      "'# VERDICT_SCOPE_OK:<real rationale>'.")
        if schedule_violations:
            reason = (reason + (" ALSO — " if reason else "") +
                      "SCHEDULE-PROVENANCE (operator 2026-07-09 'Fuck pr95... move from "
                      "hardcoded epochs to event based'): " + " | ".join(schedule_violations[:3]) +
                      " Every --*-start-epoch schedule TRIGGER must be EVENT-governed (a named "
                      "co-emitted sensor: --curriculum-event-triggered / --curriculum-nucleus-"
                      "guard / --plateau-trigger / --closed-loop-control), LawRef-DERIVED (a "
                      "constants_manifest value), or a TAGGED fail-safe cap (schedule_governance "
                      "class=cap). Derive the SHAPE from the witness math, not the PR95 skeleton. "
                      "Deliberate exception: same-line '# SCHEDULE_PROVENANCE_OK:<real rationale>'. "
                      "Sister of the launcher-path STRICT gate (tools/schedule_provenance_gate.py).")
        if dsl_config_violations:
            reason = (reason + (" ALSO — " if reason else "") +
                      "DSL-CONFIG-BYPASS (operator 2026-07-08 'config must be defined in the DSL, no "
                      "ad hoc or hand crafting'): " + " | ".join(dsl_config_violations[:3]) +
                      " A launch config must be AUTHORED + typed-validated through "
                      "tac.witness_dsl.typed_config (TypedWitnessConfig.to_program + "
                      "build_launch_manifest), not a parallel hand-assembled derive_*/*_flags path. "
                      "Route the new emitter through the typed layer (and attach the DSL-provenance "
                      "manifest the launcher gate checks). Deliberate exception: same-line "
                      "'# DSL_CONFIG_BYPASS_OK:<real rationale>'. Sister of the launcher b0.6 gate + "
                      "the confound preflight gate check_launch_config_authored_in_dsl.")
        if fm_advisories:
            reason = (reason + " ADVISORY (verdict-scope, on-device FM — never blocking): "
                      + " | ".join(fm_advisories[:2]))
        if recall_depth_advs:
            reason = (reason + " ADVISORY (recall-depth #713 — never blocking): "
                      + " | ".join(recall_depth_advs[:2]))
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
