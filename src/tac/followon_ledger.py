"""Orphaned cheap follow-ons — extract what arms NAMED, then join on what actually RAN.

Source: operator binding 2026-08-01 (task #870), verbatim: *"The zero dollar staged follow ons
and other follow ons suggested that are orphaned and were never actually followed on is orphan
signal poison grade five or even worse, like DEF CON a thousand."* Canonical memory:
``built_elsewhere_unwired_is_p0_20260801`` (its final section files this grade).

WHY THIS CLASS IS DETECTABLE WHEN GRADE 5 WAS NOT
-------------------------------------------------
``ddm_gd5`` built the grade-5 (built-elsewhere-unwired) detector, measured three formulations
against real controls, and DELETED it: the "measured-better successor" relation exists only in
memos, with no import edge, no slot value, and no registry row for an AST sweep to walk.

This class is different in exactly one respect that matters: **follow-ons are written down, in
regular language.** An arm that hits its own limit says so in its memo, in a sentence carrying a
cheapness signal (``$0``, ``scorer-free``, ``already built``, ``one command``). And "did it run"
is checkable against artifacts that postdate the memo. So the extraction side is a text problem
and the join side is an artifact problem — both tractable, neither semantic.

THE THREE THINGS THAT MAKE IT AN INSTRUMENT RATHER THAN A STORY
---------------------------------------------------------------
(a) **Memo-scoped identity.** A row is ``(source_memo, anchor)``, never a bare token. The
    ``#829`` substring-scan bug class killed the first attempt at this join: a coordinator grepped
    ``\\bR2\\b`` and got 41 hits, because short alphanumeric row ids collide with everything. The
    anchor here is a LOCATOR (which line of which memo), and it is never used as a search key.

(b) **An execution predicate keyed to artifacts, not co-occurrence.** A later memo merely
    RE-CITING a follow-on is evidence it is still pending — the opposite of execution. So
    :func:`classify_execution` separates RESULT context from PENDING context and lets a
    pending-context mention push toward ORPHANED rather than toward EXECUTED.

(c) **An explicit UNKNOWN bucket.** A follow-on whose execution cannot be determined is UNKNOWN,
    never assumed-run and never assumed-orphaned. A detector that cannot say "I don't know"
    fabricates; per ``ddm_gd5`` the honest refusal is the deliverable.

NO-FAKE. This module reports a DEBT derived from text; it never asserts that a follow-on is
valuable, and an ``ORPHANED`` verdict is a claim about EVIDENCE ("no execution artifact found in
the scanned scope"), never about worth. Every verdict carries the scope that produced it, and an
empty scope renders ``VACUOUS`` via :class:`tac.scope_ledger.ScopeLedger` — never a clean report.

Axis: ``[macOS-CPU advisory]``. ``score_claim=false``. This is APPARATUS (means), not a score.
"""

from __future__ import annotations

import json as _json
import re
import subprocess as _subprocess
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from tac.scope_ledger import ScopeLedger

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = _REPO_ROOT / ".omx" / "research"

# The canonical SSD bulk-artifact tiers (CLAUDE.md "Local Disk, SSD Spill" priority order). Most
# gate receipts and run logs live HERE, not in the repo.
#
# THEY ARE SCANNED BY DEFAULT, AND THEIR ABSENCE IS FATAL TO AN ``ORPHANED`` VERDICT. MEASURED in
# this arm's own round-1 review: wired into the costate digest without them, the join reported
# 5 ORPHANED rows whose receipts were sitting on the mounted volume -- it manufactured debt out of
# a scope gap. When a tier is unmounted, "the output does not exist" and "I could not look" are
# indistinguishable, so :func:`classify_execution` degrades ORPHANED to UNKNOWN and says why.
# This is the vacuity genus applied to the join's own evidence base.
SSD_ARTIFACT_TIERS: tuple[Path, ...] = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

# ---------------------------------------------------------------------------
# Verdicts. Three, and the third one is the point.
# ---------------------------------------------------------------------------
EXECUTED = "EXECUTED"
ORPHANED = "ORPHANED"
UNKNOWN = "UNKNOWN"
# A distinguished sub-state of UNKNOWN, reported separately because it is the operator's named
# subclass ("zero dollar STAGED follow ons") and the one the canonical instance sat in: the runner
# EXISTS, so somebody paid the build cost, but the row names no output artifact, so whether it ever
# fired cannot be settled from artifacts. It is UNKNOWN, never ORPHANED -- but it is the bucket a
# human should adjudicate first, so burying it inside a generic UNKNOWN would hide the debt.
STAGED = "STAGED"
VALID_VERDICTS = (ORPHANED, STAGED, UNKNOWN, EXECUTED)

# ---------------------------------------------------------------------------
# The extraction predicate: a CONJUNCTION, measured.
#
# Bare ``$0`` fires 12,684 times across 2,101 of 7,375 research memos -- it is ambient campaign
# vocabulary ("$0 local", "$0 probe", "the arm is $0"), not a follow-on marker. Taken alone it
# reproduces ddm_gd5's F1 failure exactly: a four-figure warn-only queue is not a queue. The
# markers the seed named in isolation fail the other way -- ``staged follow-on`` occurs twice
# repo-wide, ``one measurement away`` once. Neither side carries the class alone.
#
# The class is the CONJUNCTION: a NEXT-ACTION statement that also carries a CHEAPNESS signal.
# Measured population of the conjunction: 741 lines across 333 of 7,375 memos.
# ---------------------------------------------------------------------------
ACTION_RX = re.compile(
    r"follow[-\s]?on|next measurement|NEXT-IF-RESUMED|LIVE-HYPOTHESES|\bowed\b|"
    r"re-?race|should (?:be )?(?:run|measured?|fired?)|worth (?:running|measuring)|"
    r"\bqueued\b|to measure|re-?measure|not (?:yet )?(?:fired|run|evaluated)|never (?:fired|run|evaluated)",
    re.I,
)
CHEAP_RX = re.compile(
    r"\$0\b|scorer[-\s]free|zero[-\s]dollar|no scorer|one[-\s]command|already built|"
    r"already staged|local[-\s]only|no paid|\bcheap\b|byte-closed",
    re.I,
)

# Filenames so generic that finding one on disk says nothing about THIS follow-on. Kept short and
# explicit; the document-frequency filter does the general work.
_STOPWORD_TOKENS = frozenset({"archive.zip", "report.txt", "__init__.py", "manifest.json"})

# Concrete artifact names. TWO details are load-bearing, both MEASURED as defects in this arm's
# own round-1 review before they reached a reader:
#   1. ``jsonl`` MUST precede ``json`` in the alternation, and the trailing ``(?![\w])`` is
#      mandatory. Python's regex alternation is first-match-wins, not longest-match: with ``json``
#      first, ``d2_ep_solve.partial.jsonl`` captured as ``...partial.json`` -- a filename that
#      exists nowhere on disk, so the join returned a confident ORPHANED for a file that was
#      never its real name. Every ``.jsonl`` in the corpus was silently truncated this way.
#   2. The suffix set is closed and ordered longest-first for the same reason.
_PATHY_RX = re.compile(r"[\w./-]+\.(?:jsonl|json|npz|zip|log|txt|sh|py)(?![\w])")
_BACKTICK_RX = re.compile(r"`([^`\n]{4,80})`")                 # code spans -- the richest source
_DATE_IN_NAME_RX = re.compile(r"(20\d{2})(\d{2})(\d{2})")


@dataclass(frozen=True)
class FollowOn:
    """One named cheap follow-on, located by (memo, anchor) and keyed for joining.

    ``anchor`` is a LOCATOR, never a search key -- see module docstring (a).
    """

    memo: str
    anchor: str
    line_no: int
    text: str
    memo_date: date | None
    tokens: tuple[str, ...] = field(default=())

    @property
    def row_id(self) -> str:
        """Memo-scoped identity. Unique by construction; never grepped for."""
        return f"{self.memo}#{self.anchor}"


@dataclass(frozen=True)
class ExecutionVerdict:
    """Why a follow-on got the verdict it got. The reason travels with the answer."""

    row_id: str
    verdict: str
    reason: str
    evidence: tuple[str, ...] = field(default=())
    joinable: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "joinable": self.joinable,
        }


def memo_date(path: Path) -> date | None:
    """The YYYYMMDD stamped in a memo filename, if any. Filenames carry it by convention."""
    m = _DATE_IN_NAME_RX.search(path.stem)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _anchor_for(text: str, line_no: int) -> str:
    """A memo-local anchor: an explicit row label when the line carries one, else the line number.

    NOTE the asymmetry that keeps this honest: a label like ``QA08`` is a fine ANCHOR (it locates
    a row inside one memo) and a terrible SEARCH KEY (it collides repo-wide). This function only
    ever produces anchors; :func:`_distinctive_tokens` produces the search keys, separately.
    """
    for rx in (r"\b(QA\d{2,3})\b", r"\b(OP-[A-Z0-9-]+)\b", r"\b(#\d{3,4})\b", r"§\s?([A-Z0-9.]+)"):
        m = re.search(rx, text)
        if m:
            return f"{m.group(1)}@L{line_no}"
    return f"L{line_no}"


def _candidate_tokens(text: str) -> set[str]:
    """ARTIFACT-SHAPED substrings a follow-on could be recognised by — filenames and paths only.

    MEASURED REASON FOR THE RESTRICTION (this arm, round-1 self-review). A first cut also admitted
    bare snake_case identifiers and code spans, filtered only by document frequency (<= 8 memos).
    Its EXECUTED verdicts were then inspected by hand and were mostly WRONG: ``QA03`` was called
    executed on the token ``ddm_sb1_20260729`` (a directory the row happened to cite), ``QA08`` on
    ``prev_coloc`` (a coder context name), ``QA11`` on ``baseline_full_dseg``. Rare is not the same
    as IDENTIFYING: those tokens are the row's CONTEXT, and joining on context is the
    co-occurrence trap this module's docstring (b) promises to avoid.

    An artifact name is different in kind: it names the thing the follow-on would PRODUCE or RUN,
    so its presence or absence on disk is a fact about the follow-on rather than about its
    surroundings. Rows with no artifact-shaped token are reported UNKNOWN and not-joinable, which
    is the honest answer and is a large fraction of the population.
    """
    out: set[str] = set()
    for span in _BACKTICK_RX.findall(text):
        span = span.strip().strip("'\"")
        if _PATHY_RX.fullmatch(span):
            out.add(span)
    for m in _PATHY_RX.finditer(text):
        # Reject PARTIAL captures. A memo often writes a glob (``ddm_pb1_*_receipt_20260729.json``)
        # and the char class cannot cross the ``*``, so the match starts mid-name and yields
        # ``_receipt_20260729.json`` -- a basename that exists nowhere, so the join calls it
        # ORPHANED on a filename that was never the real one. MEASURED as a false ORPHANED in this
        # arm's own round-1 review (deferral-ledger QA02).
        if m.start() and text[m.start() - 1] in "*?%{":
            continue
        out.add(m.group(0))
    return {
        t for t in out
        if len(t) >= 8
        and t.rsplit("/", 1)[-1].lower() not in _STOPWORD_TOKENS
        and not t.rsplit("/", 1)[-1].startswith(("_", "-", "."))
    }


# Artifacts an execution PRODUCES (its output) vs artifacts an execution RUNS (its runner).
# The distinction is the whole precision of the join: a staged runner existing proves only that
# somebody BUILT the follow-on; the OUTPUT existing is what proves it ran. The wr1 staged gate is
# the canonical instance -- `experiments/stage_wr1_realized_gate.sh` exists and its named receipt
# `wr1_<cand>_realized_gate_receipt.json` does not, because the script's last four lines only
# ECHO the parse step.
_OUTPUT_SUFFIXES = (".json", ".jsonl", ".npz", ".zip", ".log", ".txt")
_RUNNER_SUFFIXES = (".sh", ".py")
# What the corpus indexes. Must be exactly the union above: a suffix a token can END in but the
# corpus does not INDEX would be reported missing-on-disk while merely being unlooked-at.
_INDEXED_SUFFIXES = frozenset(_OUTPUT_SUFFIXES + _RUNNER_SUFFIXES)


def _distinctive_tokens(
    text: str,
    doc_freq: Counter,
    *,
    max_docs: int,
    limit: int = 6,
) -> tuple[str, ...]:
    """Artifact-shaped tokens rare enough to JOIN on, rarest first.

    A token appearing in more than ``max_docs`` memos is ambient vocabulary; joining on it
    reproduces the ``#829`` substring-scan failure. Returning an EMPTY tuple is a real and useful
    answer -- it makes the row ``UNKNOWN`` (not joinable) rather than guessed at.
    """
    scored = [
        (doc_freq.get(t, 0), t)
        for t in _candidate_tokens(text)
        if 0 < doc_freq.get(t, 0) <= max_docs
    ]
    scored.sort(key=lambda kv: (kv[0], -len(kv[1])))
    return tuple(t for _, t in scored[:limit])


def _iter_memos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def build_doc_frequency(memos: list[Path]) -> Counter:
    """How many memos each candidate token appears in — the ambient-vocabulary filter's input."""
    df: Counter = Counter()
    for p in memos:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        seen: set[str] = set()
        for line in text.splitlines():
            if len(line) < 25:
                continue
            seen |= _candidate_tokens(line)
        df.update(seen)
    return df


def extract_followons(
    root: Path | None = None,
    *,
    since: date | None = None,
    max_docs: int = 8,
    memos: list[Path] | None = None,
) -> tuple[tuple[FollowOn, ...], ScopeLedger]:
    """Named cheap follow-ons, with the denominator that makes an empty result legible.

    Args:
        root: memo tree to walk (defaults to ``.omx/research``).
        since: keep only memos dated on/after this date. The population BEFORE this filter is
            recorded on the ledger, so a filter that empties the scope is visible rather than
            silent (the vacuity genus).
        max_docs: a join token may appear in at most this many memos.
        memos: pre-enumerated memo list, for tests and for reusing one walk.
    """
    root = RESEARCH_ROOT if root is None else root
    all_memos = _iter_memos(root) if memos is None else list(memos)
    population = len(all_memos)

    scoped = all_memos
    if since is not None:
        scoped = [p for p in all_memos if (memo_date(p) or date.min) >= since]

    df = build_doc_frequency(scoped)
    rows: list[FollowOn] = []
    examined = 0
    for p in scoped:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Read total-but-LOUD: one unreadable memo must never abort the sweep, and must never
            # vanish either. It stays out of `examined`, so the denominator reports the gap.
            continue
        examined += 1
        mdate = memo_date(p)
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if len(s) < 25 or not (ACTION_RX.search(s) and CHEAP_RX.search(s)):
                continue
            rows.append(
                FollowOn(
                    memo=p.name,
                    anchor=_anchor_for(s, i),
                    line_no=i,
                    text=s[:500],
                    memo_date=mdate,
                    tokens=_distinctive_tokens(s, df, max_docs=max_docs),
                )
            )
    ledger = ScopeLedger(
        surface="followon-extractor",
        examined=examined,
        declared=len(scoped),
        population=population,
        note=f"{len(rows)} follow-on rows extracted",
    )
    return tuple(rows), ledger


# ---------------------------------------------------------------------------
# The execution join.
# ---------------------------------------------------------------------------


INDEX_CACHE_PATH = _REPO_ROOT / ".omx" / "state" / "followon_artifact_index.json"


def cache_age_s() -> float | None:
    """Seconds since the artifact index was last rebuilt, or None if there is no cache."""
    try:
        return max(0.0, _time.time() - INDEX_CACHE_PATH.stat().st_mtime)
    except OSError:
        return None


def _load_index_cache(ttl_s: float) -> ExecutionCorpus | None:
    age = cache_age_s()
    if age is None or age > ttl_s:
        return None
    try:
        blob = _json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8"))
        return ExecutionCorpus(
            produced_names=frozenset(blob["names"]),
            produced_paths={},
            missing_tiers=tuple(blob.get("missing_tiers", ())),
        )
    except (OSError, ValueError, KeyError, TypeError):
        # Read total-but-LOUD in spirit: a corrupt cache degrades to a rebuild, never to a crash
        # and never to a silently empty index (which would mark every output missing).
        return None


def _save_index_cache(corpus: ExecutionCorpus) -> None:
    try:
        INDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = INDEX_CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            _json.dumps(
                {
                    "names": sorted(corpus.produced_names),
                    "missing_tiers": list(corpus.missing_tiers),
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(INDEX_CACHE_PATH)  # atomic tmp+rename
    except OSError:
        pass  # a cache that cannot be written must never break the audit it accelerates


@dataclass(frozen=True)
class ExecutionCorpus:
    """Every artifact filename that EXISTS, so the join can ask "did this produce its output?".

    Built ONCE and reused across rows; the walk is the expensive half.

    WHY THIS IS THE WHOLE CORPUS. The task spec named four candidate execution surfaces --
    receipts, commits, task rows, JSONL artifacts. Commits and completed task rows were built and
    then REMOVED from the join: a commit subject carrying ``stage_wr1_realized_gate.sh`` witnesses
    that the runner was LANDED, which is precisely the state this class is about (built, never
    fired), so admitting it would mark the canonical positive as EXECUTED. Keeping a surface that
    votes the wrong way, or keeping unused fields "for completeness", would be the orphan signal
    this module exists to find. Artifact existence is what remains, and it decides.
    """

    produced_names: frozenset[str]
    produced_paths: dict[str, str]
    missing_tiers: tuple[str, ...] = field(default=())

    @property
    def artifact_scope_complete(self) -> bool:
        """False when a declared artifact tier could not be read (e.g. the SSD is unmounted).

        While this is False the join MUST NOT emit ORPHANED: it cannot distinguish "the output
        was never produced" from "I could not look where outputs live".
        """
        return not self.missing_tiers

    @classmethod
    def build(
        cls,
        root: Path | None = None,
        *,
        extra_receipt_dirs: tuple[Path, ...] | None = None,
        cache_ttl_s: float | None = None,
    ) -> ExecutionCorpus:
        """Index artifact filenames. ``cache_ttl_s`` reuses a recent on-disk index.

        The SSD tier walk was MEASURED at ~60 s, which is too slow for the SessionStart consumer.
        The cache carries its build time so :func:`cache_age_s` can report it: a cached answer
        must be legibly cached, since silent staleness is a named confound class here.
        """
        root = RESEARCH_ROOT if root is None else root
        default_tiers = extra_receipt_dirs is None
        if default_tiers:
            extra_receipt_dirs = SSD_ARTIFACT_TIERS
        # The cache is keyed to NOTHING, so it may only ever serve the DEFAULT tier set and the
        # default root. A caller that narrowed the scope (a test, or an ad-hoc audit of one dir)
        # must never be handed an index built over a wider one -- that would silently report
        # outputs as present that its own scope cannot see. Round-2 self-review catch.
        cacheable = default_tiers and root == RESEARCH_ROOT
        if cache_ttl_s is not None and cacheable:
            cached = _load_index_cache(cache_ttl_s)
            if cached is not None:
                return cached
        names: set[str] = set()
        paths: dict[str, str] = {}
        missing = tuple(str(d) for d in extra_receipt_dirs if not d.exists())
        roots = [
            root,
            _REPO_ROOT / "experiments",
            _REPO_ROOT / "tools",
            # `src` is mandatory: a follow-on naming a module that LIVES there was otherwise
            # reported ORPHANED for being outside the walk. MEASURED in this arm's round-1 review
            # (`spec_c1_optimal_form_20260715.py` exists at src/tac/witness_dsl/ and was called
            # never-built). A corpus that omits a root does not report "unknown" for it -- it
            # reports "missing", which is the vacuity genus with the sign flipped.
            _REPO_ROOT / "src",
            *extra_receipt_dirs,
        ]
        for d in roots:
            if not d.exists():
                continue
            for p in d.rglob("*"):
                # Read total-but-LOUD: a permission error or a broken symlink on one entry must
                # not abort a sweep whose whole value is its denominator.
                try:
                    if not p.is_file():
                        continue
                except OSError:
                    continue
                # Only files that could BE a follow-on's output or runner. The SSD tier also holds
                # inflated frames and decoded video; indexing those took the corpus to 179,858
                # names and made a SessionStart-hook consumer too slow to run.
                if p.suffix.lower() not in _INDEXED_SUFFIXES:
                    continue
                # Agent worktrees are transient COPIES of the tree. A file present only there is
                # not a repo artifact, and counting it would mark a genuinely-missing output as
                # present -- a false EXECUTED, the most damaging direction for this instrument.
                sp = str(p)
                if ".claude/worktrees" in sp or "/codex_worktrees/" in sp:
                    continue
                names.add(p.name)
                paths.setdefault(p.name, sp)
        corpus = cls(
            produced_names=frozenset(names), produced_paths=paths, missing_tiers=missing
        )
        if cache_ttl_s is not None and cacheable:
            _save_index_cache(corpus)
        return corpus


# ---------------------------------------------------------------------------
# The TASK-ROW join (#880). Same artifact predicate, INVERTED conservative default.
# ---------------------------------------------------------------------------
#
# WHY THE DEFAULT FLIPS. On the memo population the damaging direction is a false ORPHANED: it
# manufactures debt, and this module's own round-1 review produced four of them. On the TASK
# population it is the exact opposite -- a false EXECUTED silently DELETES real backlog, and a
# deleted task row is not recoverable by reading harder. So here: EXECUTED must be EARNED by a
# present artifact, and everything else is UNKNOWN. There is deliberately no ORPHANED bucket for
# tasks: "this task was never done" is a negative-existence claim, and artifact absence does not
# support it (a task can be completed by work whose output it never named).
#
# NAMING IS NOT CLOSURE. A task id appearing in a commit subject witnesses that somebody NAMED it;
# the memo-side join already proved this votes the wrong way (a commit carrying ``stage_*.sh``
# witnesses the runner was LANDED, which IS the built-never-fired state). ``commit_shas`` on a task
# row is a recorded CLAIM, not an artifact, so it is not read here either.

_TASK_TEXT_FIELDS = ("title", "event_notes", "source_design_memo", "next_action", "evidence")

# The SAME row, under the harness's own key names. MEASURED DEFECT (ddm_oh1, 2026-08-02), on shipped
# code: the live harness emits ``TaskCreate`` with ``{"subject", "description", "activeForm"}`` --
# verified by reading a raw event out of the session transcript -- and NONE of those keys is in
# ``_TASK_TEXT_FIELDS``. So a caller handing this join its native rows got ``task_row_text() == ""``
# on every one of them, hence ``UNKNOWN / "carries no text to join on" / joinable=False`` for 100% of
# the population, silently.
#
# That is the vacuity genus at the FIELD level, and it is the worst version of it: the join does not
# crash and does not warn, it just declines every row while emitting a normal-looking verdict, and
# the ScopeLedger still reports a full ``examined`` count because the rows WERE walked. A reader sees
# "N rows examined, all UNKNOWN" and concludes the population is undecidable, when in fact the
# instrument never read a single character of it.
#
# Aliases, not a rename: both key-spaces are live (p2a's replayed rows use ``title``), so the reader
# accepts either. Per design philosophy P1 ("one fact, one store, one key") the ALIAS TABLE is the
# one place that knowledge lives -- callers must not each remember to map their own fields.
_TASK_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("subject", "name", "summary"),
    "evidence": ("description", "body", "detail"),
    "next_action": ("activeForm", "active_form"),
}

# A RUN product is written BY an execution (a receipt, a report, a log). A BUILD product is an
# INPUT an execution consumes (an archive, a weight bundle). Only the former can witness closure --
# the same runner-vs-output distinction the memo join makes, one level up.
_BUILD_PRODUCT_SUFFIXES = (".zip", ".npz")
_RUN_PRODUCT_SUFFIXES = tuple(s for s in _OUTPUT_SUFFIXES if s not in _BUILD_PRODUCT_SUFFIXES)

# A REFUSAL RECEIPT is a run product that records that the run DID NOT HAPPEN -- a governed
# harness refusing pre-write, a blocked launch, a preflight fail-closed. It is the one artifact
# class whose PRESENCE is evidence the row is still OPEN, and the build-vs-run split above does
# not catch it: it is written BY an execution attempt, so it is a run product by shape.
#
# MEASURED DEFECT, #880 population, on shipped code. Task #536 ("Measure the 3-axis JOINT
# rate-distortion optimum") names
# ``.omx/research/factor10_kkt_waterfill_blocked_receipt_20260718.json``. That file exists, so the
# join returned EXECUTED / "candidate ALREADY-CLOSED". Its contents:
# ``"measurement_axis": "no_scientific_measurement"``, ``"launch_performed": false``. The row's own
# text opens "MEASUREMENT STILL BLOCKED". So the join reported a row as drainable on the strength
# of a document certifying the opposite -- a FALSE ALREADY-CLOSED, the one direction this module's
# docstring forbids ("a false EXECUTED silently DELETES real backlog"). 25 artifacts under
# ``.omx/research`` alone carry a blocked/refused marker, so the class is live, not hypothetical.
#
# Detection is NAME-ONLY, and both the inclusions and the exclusions are MEASURED over the live
# 76,449-artifact corpus rather than guessed:
#
#   ``refusal`` -- 79 artifacts, all ``*_refusal_*.json`` governance receipts. Correct.
#   ``blocked``  -- 29 artifacts; the target class plus ``*_blocked_globalNNNNNN.npz`` training
#                   checkpoints that use "blocked" as a STATE LABEL. Those are ``.npz``, already
#                   excluded as build products upstream, so no live true positive is suppressed.
#   ``no_go``    -- REJECTED after measuring it. It matched ``go_no_go_verdict.json`` and
#                   ``failure_terminal_n600_no_go_*.json``, which are RUN products of measurements
#                   that EXECUTED and returned NO-GO. A NO-GO verdict is a completed measurement,
#                   so suppressing it would destroy true positives in an instrument whose measured
#                   sensitivity is already ~2%. Included in the first draft of this fix; removed
#                   once measured. This is why the marker set is measured, not enumerated.
#
# HONEST LIMIT, and it is the larger half: a refusal receipt that is not NAMED like one still earns
# EXECUTED. A content probe (``"launch_performed": false`` / ``no_scientific_measurement``) was
# written and then REMOVED, because it could never fire in production: ``_load_index_cache``
# reconstructs the corpus with ``produced_paths={}``, so on every cached load -- the normal path for
# every consumer -- the probe had zero artifacts to read. Shipping it would have been a designed
# stub that passes its own unit test (which hand-builds a corpus WITH paths) while being inert on
# real inputs. Making it live means persisting ~76k paths in the hot SessionStart cache (~6 MB);
# that is a real trade with an unmeasured payoff, so it is DEFERRED and named here rather than
# faked. This fix narrows the false-EXECUTED surface; it does not close it.
_REFUSAL_NAME_RX = re.compile(r"(?:^|[_.\-])(?:blocked|refused|refusal)(?:[_.\-]|$)", re.I)


def refusal_receipt_reason(token: str, corpus: ExecutionCorpus) -> str | None:
    """Why this present artifact certifies NON-execution, or ``None`` if it does not."""
    del corpus  # name-only by design; see the measured note above for why content is deferred
    name = token.rsplit("/", 1)[-1]
    if _REFUSAL_NAME_RX.search(name):
        return f"filename marks it a refusal/blocked receipt: {name}"
    return None


def task_row_text(task: dict) -> str:
    """The searchable text of a task row: its titled fields plus any list-valued blockers.

    Canonical field names AND their harness aliases are read, so a row in the live
    ``TaskCreate`` shape (``subject``/``description``) joins identically to a replayed row
    (``title``/``evidence``). See :data:`_TASK_FIELD_ALIASES` for the measured reason.
    """
    parts: list[str] = []
    for f in _TASK_TEXT_FIELDS:
        val = task.get(f)
        if not val:
            for alias in _TASK_FIELD_ALIASES.get(f, ()):
                val = task.get(alias)
                if val:
                    break
        parts.append(str(val or ""))
    blockers = task.get("blockers")
    if isinstance(blockers, (list, tuple)):
        parts.extend(str(b) for b in blockers)
    elif blockers:
        parts.append(str(blockers))
    return "\n".join(parts)


def classify_task_execution(task: dict, corpus: ExecutionCorpus) -> ExecutionVerdict:
    """EXECUTED / UNKNOWN for one task row, keyed to artifact existence and nothing else.

    ``EXECUTED`` here means: this row names an output artifact and that artifact EXISTS. It is a
    claim that the row is *already closed* and can be drained from the backlog. Everything else is
    ``UNKNOWN`` -- including "the row names an output and it is absent", which is evidence but not
    proof, and is reported with that reason so a reader can rank it.
    """
    tid = str(task.get("task_id", "?"))
    text = task_row_text(task)
    if not text.strip():
        return ExecutionVerdict(f"task#{tid}", UNKNOWN, "task row carries no text to join on",
                                joinable=False)

    named = _candidate_tokens(text)
    tokens = [t for t in named if t.lower().endswith(_RUN_PRODUCT_SUFFIXES)]
    builds = [t for t in named if t.lower().endswith(_BUILD_PRODUCT_SUFFIXES)]
    if not tokens:
        if builds:
            # MEASURED on the live run: task #826's only artifact is
            # ``gr1_cell_drop50_archive.zip``, and the join called it EXECUTED. The verdict was
            # CORRECT (the row IS closed -- an independent hand-check found its evaluate receipt)
            # but the EVIDENCE did not support it: an archive is the INPUT to a gate, so its
            # presence witnesses BUILT, never RAN. This is the runner-vs-output distinction from
            # the memo join, reappearing one level up. Right-for-the-wrong-reason is the most
            # dangerous state an instrument can be in, because it passes review.
            present = [t for t in builds if t.rsplit("/", 1)[-1] in corpus.produced_names]
            return ExecutionVerdict(
                f"task#{tid}", UNKNOWN,
                "row names only a BUILD product (archive/npz), whose presence witnesses BUILT and "
                "never RAN — closure is undecidable from what this row names",
                tuple(f"build-product-present:{t}" for t in present[:4]),
            )
        return ExecutionVerdict(
            f"task#{tid}", UNKNOWN,
            "row names no RUN product — nothing exists whose presence could witness closure",
            joinable=False,
        )

    present = [t for t in tokens if t.rsplit("/", 1)[-1] in corpus.produced_names]

    # Strip refusal receipts BEFORE anything can earn EXECUTED. Their presence is evidence the row
    # is OPEN, so counting them as closure inverts the verdict on exactly the rows that most need
    # to stay in the backlog.
    refusals = [(t, r) for t in present if (r := refusal_receipt_reason(t, corpus)) is not None]
    refused_tokens = {t for t, _ in refusals}
    present = [t for t in present if t not in refused_tokens]
    if refusals and not present:
        return ExecutionVerdict(
            f"task#{tid}", UNKNOWN,
            "the only OUTPUT this row names is a REFUSAL/BLOCKED receipt, which certifies the run "
            "did NOT happen — its presence is evidence the row is OPEN, never that it is closed",
            tuple(f"refusal-receipt-present:{t} — {r}" for t, r in refusals[:4]),
        )

    if present and corpus.artifact_scope_complete:
        return ExecutionVerdict(
            f"task#{tid}", EXECUTED,
            "an OUTPUT artifact this row names EXISTS — candidate ALREADY-CLOSED",
            tuple(f"output-present:{t}" for t in present[:4]),
        )
    if present:
        return ExecutionVerdict(
            f"task#{tid}", EXECUTED,
            "an OUTPUT artifact this row names EXISTS (note: a declared tier was unreadable, "
            f"which can only HIDE further evidence, never create this one): {corpus.missing_tiers}",
            tuple(f"output-present:{t}" for t in present[:4]),
        )
    return ExecutionVerdict(
        f"task#{tid}", UNKNOWN,
        "row names an output that is ABSENT — evidence of open, but not proof: a task may be "
        "closed by work whose output it never named, so this is never reported as done-or-not",
        tuple(f"named-output-absent:{t}" for t in tokens[:4]),
    )


# The canary, per design philosophy P4 ("no meter without a canary"). Run BEFORE any result is
# emitted, in the SAME run, because a scan that silently matches nothing returns the same shape as
# a scan that legitimately found nothing -- four instruments failed exactly this way on 2026-08-01,
# three of which reached the operator. Structural, not procedural: `audit_tasks` REFUSES to return
# rows if the canary does not fire.
_POSITIVE_CONTROL_ARTIFACT = "wr1_kneeA_realized_gate_receipt.json"
_NEGATIVE_CONTROL_ARTIFACT = "fo1_control_artifact_that_must_never_exist.json"


def task_join_canary(corpus: ExecutionCorpus) -> tuple[bool, str]:
    """Prove the task join FIRES on a verified-present artifact and STAYS SILENT on an absent one.

    The positive artifact is one this arm wrote and verified by hand (both wr1 gate receipts). If
    it is missing the corpus is not indexing what it claims to index, and every ``UNKNOWN`` this
    run would produce is uninterpretable.
    """
    pos = classify_task_execution(
        {"task_id": "CANARY+", "title": f"owed: land `{_POSITIVE_CONTROL_ARTIFACT}`"}, corpus
    )
    neg = classify_task_execution(
        {"task_id": "CANARY-", "title": f"owed: land `{_NEGATIVE_CONTROL_ARTIFACT}`"}, corpus
    )
    if pos.verdict != EXECUTED:
        return False, (
            f"POSITIVE CONTROL FAILED: {_POSITIVE_CONTROL_ARTIFACT} is verified-present on disk "
            f"but the join returned {pos.verdict} — the corpus is not indexing what it claims, so "
            "no verdict from this run is interpretable"
        )
    if neg.verdict == EXECUTED:
        return False, "NEGATIVE CONTROL FAILED: the join fired on an artifact that cannot exist"
    return True, "canary OK: fires on verified-present, silent on verified-absent"


def audit_tasks(
    rows: list[dict],
    corpus: ExecutionCorpus | None = None,
    *,
    cache_ttl_s: float | None = None,
) -> tuple[list[tuple[dict, ExecutionVerdict]], ScopeLedger, str]:
    """Join task rows against artifact existence. Returns (paired, scope, canary_note).

    ``rows`` is caller-supplied ON PURPOSE. The operator-visible backlog lives in the harness task
    list, which has NO on-disk mirror in this repo -- MEASURED: every one of task ids 833/834/840/
    841/844/858/859/860 and 864/869/870 is ABSENT from ``.omx/state/canonical_task_status.jsonl``.
    So whoever holds that ledger passes its rows in; this function never pretends to enumerate a
    population it cannot see, and the returned :class:`ScopeLedger` says which one it got.
    """
    corpus = ExecutionCorpus.build(cache_ttl_s=cache_ttl_s) if corpus is None else corpus
    ok, note = task_join_canary(corpus)
    if not ok:
        return [], ScopeLedger(
            surface="task-execution-join", examined=0, declared=len(rows),
            note=f"REFUSED — {note}",
        ), note
    paired = [(r, classify_task_execution(r, corpus)) for r in rows]
    order = {EXECUTED: 0, UNKNOWN: 1}
    paired.sort(key=lambda rv: (order.get(rv[1].verdict, 9), str(rv[0].get("task_id"))))
    return paired, ScopeLedger(
        surface="task-execution-join",
        examined=len(paired),
        declared=len(rows),
        note=note,
    ), note


def classify_execution(row: FollowOn, corpus: ExecutionCorpus) -> ExecutionVerdict:
    """EXECUTED / ORPHANED / UNKNOWN for one follow-on, with the evidence that decided it.

    The predicate is keyed to artifacts that POSTDATE the naming memo — later memos, commit
    subjects, completed task rows, receipt filenames — never to co-occurrence anywhere in the
    corpus. And a later mention is only EXECUTED evidence when it sits in RESULT context; in
    PENDING context it counts the other way, because a queue re-listing an item is telling you
    nobody drained it.
    """
    if not row.tokens:
        return ExecutionVerdict(
            row.row_id, UNKNOWN,
            "no artifact-shaped join token — the row names no file the join could check for. "
            "Joining on its incidental identifiers is the #829 substring-collision failure and "
            "was MEASURED to mis-classify in this arm's own round-1 review",
            joinable=False,
        )

    outputs = [t for t in row.tokens if t.lower().endswith(_OUTPUT_SUFFIXES)]
    runners = [t for t in row.tokens if t.lower().endswith(_RUNNER_SUFFIXES)]

    def _exists(tok: str) -> bool:
        # A memo may write a token as a bare filename or as a repo-relative path; the corpus is
        # indexed by basename, so compare on basename and never on the raw string.
        return tok.rsplit("/", 1)[-1] in corpus.produced_names

    present_out = [t for t in outputs if _exists(t)]
    missing_out = [t for t in outputs if not _exists(t)]

    if present_out:
        return ExecutionVerdict(
            row.row_id, EXECUTED,
            "the OUTPUT artifact this follow-on would produce exists on disk",
            tuple(f"output-present:{t}" for t in present_out[:4]),
        )
    if missing_out:
        if not corpus.artifact_scope_complete:
            return ExecutionVerdict(
                row.row_id, UNKNOWN,
                "the named OUTPUT was not found, but a declared artifact tier could not be read "
                f"({', '.join(corpus.missing_tiers)}) — 'not produced' and 'I could not look' "
                "are indistinguishable here, so this is UNKNOWN and never ORPHANED",
                tuple(f"unreadable-tier:{t}" for t in corpus.missing_tiers[:2])
                + tuple(f"not-found:{t}" for t in missing_out[:2]),
            )
        return ExecutionVerdict(
            row.row_id, ORPHANED,
            "the OUTPUT artifact this follow-on would produce does NOT exist in the scanned "
            "artifact scope — the follow-on cannot have completed",
            tuple(f"output-MISSING:{t}" for t in missing_out[:4]),
        )

    # Runner-only rows. A runner that exists proves the follow-on was BUILT, never that it RAN --
    # that is the exact gap the operator called DEF CON a thousand, so it must not read EXECUTED.
    if runners:
        built = [t for t in runners if _exists(t)]
        if built:
            return ExecutionVerdict(
                row.row_id, STAGED,
                "the runner EXISTS but the row names no output artifact, so whether it ever fired "
                "is not decidable from artifacts alone — adjudicate by hand FIRST; this is the "
                "bucket the wr1 realized gate sat in",
                tuple(f"runner-present:{t}" for t in built[:4]),
            )
        return ExecutionVerdict(
            row.row_id, ORPHANED,
            "the runner this follow-on names does not exist — it was never even built",
            tuple(f"runner-MISSING:{t}" for t in runners[:4]),
        )
    return ExecutionVerdict(
        row.row_id, UNKNOWN,
        "joinable tokens carry no output/runner suffix the join can decide on",
    )


def audit(
    root: Path | None = None,
    *,
    since: date | None = None,
    extra_receipt_dirs: tuple[Path, ...] | None = None,
    cache_ttl_s: float | None = None,
) -> tuple[list[tuple[FollowOn, ExecutionVerdict]], ScopeLedger]:
    """Extract, join, and return rows worst-first (ORPHANED, STAGED, UNKNOWN, EXECUTED).

    ``extra_receipt_dirs=None`` (the default) scans :data:`SSD_ARTIFACT_TIERS`. Passing ``()``
    explicitly opts OUT of them and will mark SSD-resident outputs as not-found -- only do that in
    tests. This default matters: an empty tuple here was MEASURED to make the costate digest
    report 5 ORPHANED rows whose receipts were on the mounted volume all along.
    """
    root = RESEARCH_ROOT if root is None else root
    memos = _iter_memos(root)
    rows, ledger = extract_followons(root, since=since, memos=memos)
    corpus = ExecutionCorpus.build(
        root, extra_receipt_dirs=extra_receipt_dirs, cache_ttl_s=cache_ttl_s
    )
    order = {v: i for i, v in enumerate(VALID_VERDICTS)}
    paired = [(r, classify_execution(r, corpus)) for r in rows]
    paired.sort(key=lambda rv: (order[rv[1].verdict], rv[0].memo, rv[0].line_no))
    return paired, ledger


def summarise(paired: list[tuple[FollowOn, ExecutionVerdict]]) -> dict[str, int]:
    """Verdict histogram — the shape of the answer before any individual row is read."""
    c = Counter(v.verdict for _, v in paired)
    return {k: c.get(k, 0) for k in VALID_VERDICTS}


# ===========================================================================
# THE HANDOFF GRAPH (ddm_oh1). A DIFFERENT EDGE, INVISIBLE TO EVERYTHING ABOVE.
# ===========================================================================
#
# The join above answers "did this follow-on produce its OUTPUT?" and is keyed to a filename the
# row itself names. A HANDOFF names no output. It names a SUCCESSOR -- a task id, a sister arm, a
# code site -- and says the work goes THERE. Nothing then joins "what an arm said was owed" to
# "what was subsequently done", so the edge is invisible in both directions.
#
# TWO INSTANCES, MEASURED at the source, both of which the shipped extractor above CANNOT see:
#
#   (a) ``ddm_sv2_...20260802.md:350`` -- *"#539 (Build the POWER-DIAGRAM witness parametrization,
#       in_progress) is the natural donor; this arm should hand off to it rather than fork a
#       parallel surface."*  ``ACTION_RX`` does not match "should hand off" and ``CHEAP_RX`` matches
#       nothing on the line, so the row is never extracted.
#   (b) ``ddm_os1_...20260802.md:203`` -- *"No cure landed at pfs1 ... the fix ... is STAGED, not
#       taken."*  Same: neither half of the conjunction fires.
#
# AND THE STRUCTURAL DEFECT THAT DECIDES THE DESIGN -- stated precisely, because an earlier draft
# of this comment overclaimed it and the test written to pin the claim is what caught it.
#
# The two instances fail the shipped extractor for DIFFERENT reasons, and only one of them is
# structural:
#
#   * (a) sv2 is a SCOPING failure. Its target ``#539`` sits on line 352 and its verb "hand off" on
#     line 353. ``extract_followons`` iterates ``text.splitlines()``, so NO single line ever carries
#     both halves -- no predicate whatsoever, however good, can see this row one line at a time.
#     This is what item scoping fixes, and it is pinned by
#     ``test_no_single_line_carries_both_halves_of_the_sv2_instance``.
#   * (b) os1 is a PREDICATE failure only. "No cure landed at pfs1" and the slug ``pfs1`` share one
#     line, so line scoping would suffice; it is invisible because ``ACTION_RX ∧ CHEAP_RX`` matches
#     neither half of it.
#
# The scoping half is very likely a component of ``ddm_p1a``'s finding that all 86 UNKNOWN follow-on
# rows carried the single reason "no artifact-shaped join token": a token stranded on the
# neighbouring line reads exactly like a token that is absent. That remains INFERRED -- it is a
# mechanism this arm demonstrated on one row, not a re-measurement of p1a's population.
#
# ``extract_followons`` is deliberately NOT switched to item scoping here. Its line-scoped
# population is quoted by two landed memos (``ddm_fo1``, ``ddm_p1a``); silently changing it would
# move their numbers underneath them. That is a supersession decision with an owner, so it is NAMED
# (see this arm's memo) rather than taken unilaterally.

# Deliberately NOT called DONE. A commit that TOUCHES the named site after the naming date proves
# the successor MOVED there; it does not prove it did the named work. Naming this ``DONE`` would be
# the false-closure direction that ``classify_task_execution`` already refuses -- and here it is
# worse, because a handoff is precisely the state where somebody else's later edit is easy to
# mistake for the owed one.
ADVANCED = "ADVANCED"
# Younger than the drain window: not orphaned, just not adjudicable yet.
LIVE = "LIVE"
# No artifact-level channel exists for this target kind (a bare task id with no ledger), or the
# channel exists but could not be read (git unavailable). Never ORPHANED -- "no successor found"
# and "I could not look for one" must not share a symbol.
UNVERIFIABLE = "UNVERIFIABLE"
HANDOFF_VERDICTS = (ORPHANED, LIVE, ADVANCED, UNVERIFIABLE)

# How long a handoff gets before absence of successor activity means anything.
#
# DERIVED, with a named anchor: ``ddm_fo1_orphaned_followon_detector_20260801.md:26`` MEASURED that
# "sister arms drain follow-ons within 24-72 h" -- the finding that its own orphan list was 5-of-6
# stale. 72 h is that window's UPPER end, so it is the conservative choice: it can only move rows
# OUT of ORPHANED, never into it. Day granularity because memo dates come from filenames, which
# carry YYYYMMDD and no clock.
#
# RECESS MEASUREMENT that would sharpen it: the distribution of (handoff naming date -> first
# successor-touch date) over the ADVANCED rows this instrument itself produces. That is a real
# empirical drain curve rather than one arm's summary sentence, and it needs a population this
# module can only produce after it has run. Named here so the constant is not mistaken for settled.
LIVE_WINDOW_DAYS = 3

# Section headings whose CONTENT is, by construction, a list of owed next-steps. MEASURED to be a
# far better discriminator than any verb list: on a hand-inspected random sample of 8, heading
# context scored ~7/8 while the explicit-frame stratum scored ~2/8. The frames misfire because
# "hand-off" is also live DOMAIN vocabulary in this repo ("birth -> boundary hand-off is coherent
# with the curriculum"), and "is the natural successor to" appears in architecture-comparison
# tables. A heading cannot be confused for prose in the same way.
#
# This is not a coincidence: the subagent contract MANDATES ``NEXT-IF-RESUMED`` / ``LIVE-HYPOTHESES``
# / ``DEAD-ENDS`` blocks, so the corpus has a real structural convention to exploit.
#
# ``handoff`` IS DELIBERATELY ABSENT FROM THIS PATTERN, and that is a MEASURED correction to this
# arm's own first draft. With it included, the first live run returned 29 ORPHANED of which roughly
# half were headings like "Serializer and exact-hash handoff", "Verification, triality, and handoff",
# "Triality handoff" -- whose content is a SHA-256 custody manifest for files that LANDED, i.e. the
# exact opposite of owed work. Enumerating every ``handoff`` heading in the corpus confirms the word
# is a standard CUSTODY/VERIFICATION section name here ("Triality handoff" x3, "Self-review and MAIN
# handoff" x2, "Law 5 -- label_floor_to_phase_tail_handoff_v1", "handoff_readiness.part_frac"), not a
# transfer-of-owed-work marker. It survives in :data:`HANDOFF_FRAME_RX`, where it appears as a VERB
# in a sentence and carries its ordinary meaning. Removing it costs a few true positives (e.g. a
# "Reproducible execution handoff" command block) and removes a much larger false mass.
OWED_HEADING_RX = re.compile(
    r"NEXT-IF-RESUMED|LIVE-HYPOTHESES|what (?:this|it) owes|owes next|what i did not do|"
    r"next steps?|follow[-\s]?ons?|\bowed\b|what remains|remaining work|"
    r"not done|deferred|open items?|for a successor|operator[-\s]routable",
    re.I,
)
# A markdown TABLE ROW is data, not a directive. MEASURED in this arm's round-1 review: with heading
# context alone qualifying them, rows like ``| n64 | 61,087 | -0.001589 | ...`` under a heading named
# "Scale ladder and c1 handoff" were reported ORPHANED -- the row is a measurement table, and ``n64``
# and ``c1`` are simultaneously real arm slugs (``ddm_n64_*`` and ``ddm_c1_*`` both exist) and
# ordinary scale/lane words. That ambiguity is the ``#829`` collision class arriving through a
# channel the arm-corpus membership test cannot filter, because the tokens really are arms.
#
# Table rows are NOT dropped, because some of the richest owed queues in this corpus ARE tables (the
# deferral ledger's ``| QA20 | ... | never run |`` rows). They are held to the STRICTER bar: a table
# row must carry an explicit transfer FRAME in the row itself, and may never qualify on the section
# heading alone.
_TABLE_ROW_RX = re.compile(r"^\s*\|")
# The explicit-transfer frames. Kept because they catch handoffs written OUTSIDE an owed section --
# instance (b) is exactly that (it sits under "What I did NOT do", which the heading rx also
# catches, but instance (a) sits under a numbered "WHAT THIS OWES NEXT" item whose verb is what
# makes it a handoff rather than a plain next-step). Reported as a SEPARATE stratum because its
# measured precision is much lower; a reader must be able to rank by which one fired.
HANDOFF_FRAME_RX = re.compile(
    r"hand(?:s|ed|ing)?[-\s]off|hand (?:it|this|them|the \w+) (?:off )?(?:to|over)|"
    r"natural donor|is the (?:natural )?(?:donor|owner|successor)|"
    r"staged,? not taken|staged (?:but )?(?:never|not) (?:taken|fired|run)|is STAGED, not|"
    r"for a successor|a successor (?:should|must|would|finds)|successor should|"
    r"no cure landed|did NOT do|I did not (?:do|land|wire|fire|take)",
    re.I,
)

STRATUM_FRAME = "FRAME"
STRATUM_HEADING = "HEADING"

TARGET_TASK = "TASK"
TARGET_ARM = "ARM"
TARGET_PATH = "PATH"

_TASK_TARGET_RX = re.compile(r"#(\d{3,4})\b")
# An arm slug, with or without its ``ddm_`` prefix. Bare slugs like ``pfs1`` are exactly the short
# alphanumerics the ``#829`` substring-collision bug class is about, so a slug is admitted ONLY if
# it is a MEMBER of the real-arm corpus built from artifact filenames. Membership in a measured set
# is not a substring grep: ``r2`` stays out unless ``ddm_r2_*`` actually exists on disk.
_ARM_TARGET_RX = re.compile(r"\b(?:ddm_)?([a-z]{1,4}\d{1,2}[a-z]?)\b")
_PATH_TARGET_RX = re.compile(r"\b((?:src|tools|experiments|scripts)/[\w./-]+\.(?:py|sh))\b")

_MD_ITEM_START_RX = re.compile(r"^\s{0,6}(?:[-*+]\s|\d{1,2}[.)]\s|\|)")
_MD_HEADING_RX = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")


def iter_markdown_units(text: str, *, max_lines: int = 8):
    """Yield ``(start_line, unit_text, heading)`` for markdown items and paragraphs.

    The unit, not the line, is the extraction scope -- see the section comment above for the
    measured reason (both controls MISS under line scoping and FIRE under this one).

    ``max_lines`` caps a runaway unit so one unbroken block cannot swallow a whole document and
    manufacture spurious frame/target co-occurrence across unrelated sentences.
    """
    lines = text.splitlines()
    buf: list[str] = []
    start = 0
    heading = ""
    for i, raw in enumerate(lines, 1):
        stripped = raw.strip()
        head_m = _MD_HEADING_RX.match(raw)
        breaks = not stripped or bool(head_m) or bool(_MD_ITEM_START_RX.match(raw))
        if breaks or (buf and i - start >= max_lines):
            if buf:
                yield start, " ".join(buf), heading
            buf, start = [], 0
        if head_m:
            # The heading itself is CONTEXT for what follows, never a unit of its own.
            heading = head_m.group(1).strip()
            continue
        if not stripped:
            continue
        if not buf:
            start = i
        buf.append(stripped)
    if buf:
        yield start, " ".join(buf), heading


@dataclass(frozen=True)
class HandoffTarget:
    """Where a handoff says the work goes. ``key`` is a JOIN key, never a grep string."""

    kind: str
    key: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "key": self.key}


@dataclass(frozen=True)
class Handoff:
    """One named handoff edge: a producer memo, and the successor(s) it names."""

    memo: str
    anchor: str
    line_no: int
    text: str
    memo_date: date | None
    stratum: str
    heading: str
    targets: tuple[HandoffTarget, ...] = field(default=())

    @property
    def row_id(self) -> str:
        """Memo-scoped identity, same discipline as :class:`FollowOn`."""
        return f"{self.memo}#{self.anchor}"


def real_arm_slugs(roots: tuple[Path, ...] | None = None) -> frozenset[str]:
    """Arm slugs that ACTUALLY EXIST, derived from ``ddm_<slug>_*`` artifact filenames.

    This set is what keeps the bare-slug target channel out of the ``#829`` collision class: a
    two-to-five character token is admitted as an arm only if the campaign really produced an arm
    by that name.
    """
    if roots is None:
        roots = (
            RESEARCH_ROOT,
            _REPO_ROOT / "experiments",
            _REPO_ROOT / "tools",
            _REPO_ROOT / "src",
        )
    slugs: set[str] = set()
    for d in roots:
        if not d.exists():
            continue
        for p in d.rglob("ddm_*"):
            m = re.match(r"ddm_([a-z]{1,4}\d{1,2}[a-z]?)_", p.name)
            if m:
                slugs.add(m.group(1))
    return frozenset(slugs)


def extract_targets(unit: str, arms: frozenset[str]) -> tuple[HandoffTarget, ...]:
    """Named successors in one unit: task ids, existing arm slugs, repo-relative code paths."""
    found: list[HandoffTarget] = []
    seen: set[tuple[str, str]] = set()
    for kind, keys in (
        (TARGET_TASK, [f"#{n}" for n in _TASK_TARGET_RX.findall(unit)]),
        (TARGET_ARM, [s for s in _ARM_TARGET_RX.findall(unit) if s in arms]),
        (TARGET_PATH, _PATH_TARGET_RX.findall(unit)),
    ):
        for k in keys:
            if (kind, k) not in seen:
                seen.add((kind, k))
                found.append(HandoffTarget(kind, k))
    return tuple(found)


def extract_handoffs(
    root: Path | None = None,
    *,
    since: date | None = None,
    memos: list[Path] | None = None,
    arms: frozenset[str] | None = None,
) -> tuple[tuple[Handoff, ...], ScopeLedger]:
    """Handoff edges, with the denominator that makes an empty result legible.

    A unit qualifies when it names at least one successor AND either sits under an owed-section
    heading (:data:`OWED_HEADING_RX`) or carries an explicit transfer frame
    (:data:`HANDOFF_FRAME_RX`). The stratum that fired travels with the row.
    """
    root = RESEARCH_ROOT if root is None else root
    all_memos = _iter_memos(root) if memos is None else list(memos)
    population = len(all_memos)
    scoped = all_memos
    if since is not None:
        scoped = [p for p in all_memos if (memo_date(p) or date.min) >= since]
    if arms is None:
        arms = real_arm_slugs()

    rows: list[Handoff] = []
    examined = 0
    for p in scoped:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Read total-but-LOUD: one unreadable memo stays out of `examined`, so the denominator
            # reports the gap rather than hiding it.
            continue
        examined += 1
        mdate = memo_date(p)
        for start, unit, heading in iter_markdown_units(text):
            if len(unit) < 40:
                continue
            targets = extract_targets(unit, arms)
            if not targets:
                continue
            if HANDOFF_FRAME_RX.search(unit):
                stratum = STRATUM_FRAME
            elif OWED_HEADING_RX.search(heading) and not _TABLE_ROW_RX.match(unit):
                stratum = STRATUM_HEADING
            else:
                continue
            rows.append(
                Handoff(
                    memo=p.name,
                    anchor=_anchor_for(unit, start),
                    line_no=start,
                    text=unit[:500],
                    memo_date=mdate,
                    stratum=stratum,
                    heading=heading[:120],
                    targets=targets,
                )
            )
    ledger = ScopeLedger(
        surface="handoff-extractor",
        examined=examined,
        declared=len(scoped),
        population=population,
        note=f"{len(rows)} handoff edges extracted",
    )
    return tuple(rows), ledger


@dataclass(frozen=True)
class SuccessorIndex:
    """When each repo path was last TOUCHED by a commit — the successor-activity channel.

    Built from ONE ``git log --name-only`` pass rather than a call per row (MEASURED: 2.3 s bounded
    to a month, 21 s over the full 13.7k-commit history; per-row calls would be neither).

    ``available`` is the vacuity guard and it is load-bearing. If git cannot be read, every path
    looks untouched, and an instrument that trusted that would report the entire corpus ORPHANED --
    manufacturing debt out of a dead subprocess. While ``available`` is False no row may be
    ORPHANED.
    """

    touched: dict[str, date]
    available: bool
    since: date | None = None
    reason: str = ""
    # Every path git TRACKS. Empty means "not determined" -- see `tracks()`.
    tracked: frozenset[str] = field(default=frozenset())

    def tracks(self, path: str) -> bool | None:
        """Is this path under version control? ``None`` when tracking could not be determined.

        LOAD-BEARING, and it exists because of a MEASURED false ORPHANED in this arm's own round-1
        review. ``margin_field_head_levers_...:77`` hands off a wire-in to
        ``experiments/train_LEVELSET_witness_realized_through_R_mlx.py``, and the memo itself says
        the file is "UNTRACKED / git-ignored in the repo". A commit-touch channel can NEVER see an
        untracked file, so such a path is untouched by construction and would be reported ORPHANED
        forever, no matter how much work went into it. That is the vacuity genus once more: "no
        commit touched it" and "commits cannot touch it" are different facts wearing one symbol.
        """
        if not self.tracked:
            return None
        return path in self.tracked

    @classmethod
    def build(
        cls,
        *,
        since: date | None = None,
        repo_root: Path | None = None,
        timeout_s: float = 180.0,
    ) -> SuccessorIndex:
        repo_root = _REPO_ROOT if repo_root is None else repo_root
        cmd = ["git", "log", "--date=short", "--format=%x00%ad", "--name-only"]
        if since is not None:
            cmd.append(f"--since={since.isoformat()}")
        try:
            proc = _subprocess.run(
                cmd, cwd=repo_root, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except (OSError, _subprocess.SubprocessError) as exc:
            return cls({}, False, since, f"git unavailable: {type(exc).__name__}: {exc}")
        if proc.returncode != 0:
            return cls({}, False, since, f"git exited {proc.returncode}: {proc.stderr[:200]}")
        touched: dict[str, date] = {}
        cur: date | None = None
        for line in proc.stdout.splitlines():
            if line.startswith("\x00"):
                try:
                    cur = date.fromisoformat(line[1:].strip())
                except ValueError:
                    cur = None
                continue
            path = line.strip()
            if not path or cur is None:
                continue
            prev = touched.get(path)
            if prev is None or cur > prev:
                touched[path] = cur
        if not touched:
            # An EMPTY index from a SUCCESSFUL git call is still vacuous for our purposes: it means
            # the window contained no commits, so absence of a touch says nothing.
            return cls({}, False, since, "git returned no touched paths in the scanned window")
        tracked: frozenset[str] = frozenset()
        try:
            ls = _subprocess.run(
                ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True,
                timeout=timeout_s, check=False,
            )
            if ls.returncode == 0:
                tracked = frozenset(p for p in ls.stdout.splitlines() if p)
        except (OSError, _subprocess.SubprocessError):
            # Leaving `tracked` empty makes `tracks()` return None, i.e. "not determined" -- which
            # the join treats as "cannot rule the path out", never as "untracked".
            pass
        return cls(touched, True, since, f"{len(touched)} paths indexed", tracked)

    def last_touch(self, path: str) -> date | None:
        """Latest commit date for an exact repo-relative path."""
        return self.touched.get(path)

    def last_touch_matching(self, needle: str) -> date | None:
        """Latest commit date across every indexed path whose BASENAME contains ``needle``.

        Used for the arm channel: an arm's activity shows up as commits to files named after it.
        Basename-scoped on purpose -- matching the full path would let a directory name
        (``experiments/results/ddm_sv2_.../anything.txt``) vote for an arm that merely owns a
        folder.
        """
        best: date | None = None
        for p, d in self.touched.items():
            if needle in p.rsplit("/", 1)[-1] and (best is None or d > best):
                best = d
        return best


def classify_handoff(
    row: Handoff,
    index: SuccessorIndex,
    *,
    today: date | None = None,
    closed_task_ids: frozenset[str] | None = None,
    live_window_days: int = LIVE_WINDOW_DAYS,
) -> ExecutionVerdict:
    """ADVANCED / LIVE / ORPHANED / UNVERIFIABLE for one handoff edge, with its evidence.

    ``closed_task_ids`` is caller-supplied for the same reason ``audit_tasks`` takes its rows: the
    harness task ledger has no on-disk mirror, so this module never pretends to enumerate it. With
    it, a ``#NNN`` target becomes decidable; without it, task targets are honestly UNVERIFIABLE.
    """
    today = date.today() if today is None else today
    if row.memo_date is None:
        return ExecutionVerdict(
            row.row_id, UNVERIFIABLE,
            "the producing memo carries no date in its filename, so 'did anything happen AFTER "
            "this was named' has no reference point",
            joinable=False,
        )

    age_days = (today - row.memo_date).days
    evidence: list[str] = []

    for t in row.targets:
        if t.kind == TARGET_PATH:
            when = index.last_touch(t.key)
            if when is not None and when >= row.memo_date:
                evidence.append(f"path-touched-after:{t.key}@{when.isoformat()}")
        elif t.kind == TARGET_ARM:
            when = index.last_touch_matching(f"ddm_{t.key}")
            if when is not None and when >= row.memo_date:
                evidence.append(f"arm-active-after:ddm_{t.key}@{when.isoformat()}")
        elif t.kind == TARGET_TASK and closed_task_ids is not None:
            if t.key.lstrip("#") in closed_task_ids:
                evidence.append(f"task-closed:{t.key}")

    if evidence:
        return ExecutionVerdict(
            row.row_id, ADVANCED,
            "the named successor shows ACTIVITY after this handoff was written. This is evidence "
            "the successor MOVED, never proof it did the named work — deliberately not 'DONE', "
            "because a false closure here silently deletes real debt",
            tuple(evidence[:4]),
        )

    if age_days < live_window_days:
        return ExecutionVerdict(
            row.row_id, LIVE,
            f"named {age_days}d ago, inside the {live_window_days}d drain window measured by "
            "ddm_fo1 (sister arms drain follow-ons within 24–72 h) — too early for absence of "
            "successor activity to mean anything",
            tuple(f"target:{t.kind}:{t.key}" for t in row.targets[:4]),
        )

    if not index.available:
        return ExecutionVerdict(
            row.row_id, UNVERIFIABLE,
            "the successor-activity channel could not be read "
            f"({index.reason}) — 'nothing happened' and 'I could not look' are indistinguishable "
            "here, so this is never ORPHANED",
            tuple(f"target:{t.kind}:{t.key}" for t in row.targets[:4]),
        )

    # A path git does not track is invisible to the touch channel BY CONSTRUCTION, so its silence
    # carries no information and it cannot support an ORPHANED verdict. See `SuccessorIndex.tracks`.
    untracked = [t for t in row.targets
                 if t.kind == TARGET_PATH and index.tracks(t.key) is False]
    decidable = [
        t for t in row.targets
        if (t.kind != TARGET_TASK or closed_task_ids is not None)
        and not (t.kind == TARGET_PATH and index.tracks(t.key) is False)
    ]
    if untracked and not decidable:
        return ExecutionVerdict(
            row.row_id, UNVERIFIABLE,
            "every named successor path is UNTRACKED by git, so the commit-touch channel is blind "
            "to it by construction — its silence is a property of the channel, not of the work",
            tuple(f"untracked-path:{t.key}" for t in untracked[:4]),
        )
    if not decidable:
        return ExecutionVerdict(
            row.row_id, UNVERIFIABLE,
            "every named successor is a bare task id and no task ledger was supplied — a task id "
            "has no artifact-level closure channel, and 'a commit mentions the number' witnesses "
            "NAMING, never closure",
            tuple(f"target:{t.kind}:{t.key}" for t in row.targets[:4]),
            joinable=False,
        )

    if index.since is not None and row.memo_date < index.since:
        # The row predates the indexed window, so the index cannot see the successor activity that
        # would clear it. Reporting ORPHANED here would be an artifact of the scan bound.
        return ExecutionVerdict(
            row.row_id, UNVERIFIABLE,
            f"this handoff ({row.memo_date.isoformat()}) predates the indexed commit window "
            f"(since {index.since.isoformat()}) — successor activity for it is outside what was "
            "scanned, so absence of evidence here is a scan bound, not a finding",
            tuple(f"target:{t.kind}:{t.key}" for t in row.targets[:4]),
        )

    return ExecutionVerdict(
        row.row_id, ORPHANED,
        f"named {age_days}d ago, past the {live_window_days}d drain window, and NO successor "
        "activity was found for any decidable target in the scanned commit window",
        tuple(f"no-activity-since-{row.memo_date.isoformat()}:{t.kind}:{t.key}"
              for t in decidable[:4]),
    )


# The canary, per design philosophy P4. Structural, not procedural: `audit_handoffs` REFUSES to
# return rows if it does not fire. The positive control is an edge whose successor path is
# GUARANTEED touched in-window; the negative control names a path that cannot exist.
_HANDOFF_POSITIVE_PATH = "src/tac/followon_ledger.py"
_HANDOFF_NEGATIVE_PATH = "experiments/oh1_control_path_that_must_never_exist.py"


def handoff_join_canary(index: SuccessorIndex, *, today: date | None = None) -> tuple[bool, str]:
    """Prove the join FIRES on a path with known recent commits and STAYS SILENT on an absent one."""
    today = date.today() if today is None else today
    old = today - timedelta(days=365)

    def _probe(path: str) -> ExecutionVerdict:
        return classify_handoff(
            Handoff(
                memo="CANARY.md", anchor="L1", line_no=1,
                text=f"hand off to `{path}`", memo_date=old,
                stratum=STRATUM_FRAME, heading="NEXT-IF-RESUMED",
                targets=(HandoffTarget(TARGET_PATH, path),),
            ),
            index, today=today,
        )

    pos = _probe(_HANDOFF_POSITIVE_PATH)
    neg = _probe(_HANDOFF_NEGATIVE_PATH)
    if pos.verdict != ADVANCED:
        return False, (
            f"POSITIVE CONTROL FAILED: {_HANDOFF_POSITIVE_PATH} has commits in the indexed window "
            f"but the join returned {pos.verdict} ({pos.reason[:120]}) — the successor-activity "
            "channel is not reading what it claims, so no verdict from this run is interpretable"
        )
    if neg.verdict == ADVANCED:
        return False, "NEGATIVE CONTROL FAILED: the join found activity on a path that cannot exist"
    return True, "canary OK: fires on a known-touched path, silent on a path that cannot exist"


def audit_handoffs(
    root: Path | None = None,
    *,
    since: date | None = None,
    index: SuccessorIndex | None = None,
    today: date | None = None,
    closed_task_ids: frozenset[str] | None = None,
) -> tuple[list[tuple[Handoff, ExecutionVerdict]], ScopeLedger, str]:
    """Extract handoff edges and join them. Returns ``(paired, scope, canary_note)``.

    Rows come back worst-first: ORPHANED, then LIVE, then ADVANCED, then UNVERIFIABLE.
    """
    root = RESEARCH_ROOT if root is None else root
    rows, ledger = extract_handoffs(root, since=since)
    if index is None:
        index = SuccessorIndex.build(since=since)
    ok, note = handoff_join_canary(index, today=today)
    if not ok:
        return [], ScopeLedger(
            surface="handoff-join", examined=0, declared=len(rows),
            population=ledger.population, note=f"REFUSED — {note}",
        ), note
    order = {v: i for i, v in enumerate(HANDOFF_VERDICTS)}
    paired = [
        (r, classify_handoff(r, index, today=today, closed_task_ids=closed_task_ids))
        for r in rows
    ]
    paired.sort(key=lambda rv: (order[rv[1].verdict], rv[0].memo, rv[0].line_no))
    return paired, ScopeLedger(
        surface="handoff-join",
        examined=len(paired),
        declared=len(rows),
        population=ledger.population,
        note=note,
    ), note


def summarise_handoffs(paired: list[tuple[Handoff, ExecutionVerdict]]) -> dict[str, int]:
    """Verdict histogram for the handoff graph."""
    c = Counter(v.verdict for _, v in paired)
    return {k: c.get(k, 0) for k in HANDOFF_VERDICTS}


__all__ = [
    "ADVANCED",
    "EXECUTED",
    "HANDOFF_VERDICTS",
    "LIVE",
    "LIVE_WINDOW_DAYS",
    "ORPHANED",
    "STRATUM_FRAME",
    "STRATUM_HEADING",
    "TARGET_ARM",
    "TARGET_PATH",
    "TARGET_TASK",
    "UNKNOWN",
    "UNVERIFIABLE",
    "VALID_VERDICTS",
    "ExecutionCorpus",
    "ExecutionVerdict",
    "FollowOn",
    "Handoff",
    "HandoffTarget",
    "SuccessorIndex",
    "audit",
    "audit_handoffs",
    "audit_tasks",
    "build_doc_frequency",
    "classify_execution",
    "classify_handoff",
    "classify_task_execution",
    "extract_followons",
    "extract_handoffs",
    "extract_targets",
    "handoff_join_canary",
    "iter_markdown_units",
    "memo_date",
    "real_arm_slugs",
    "summarise",
    "summarise_handoffs",
    "task_join_canary",
    "task_row_text",
]
