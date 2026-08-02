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
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
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


__all__ = [
    "EXECUTED",
    "ORPHANED",
    "UNKNOWN",
    "VALID_VERDICTS",
    "ExecutionCorpus",
    "ExecutionVerdict",
    "FollowOn",
    "audit",
    "build_doc_frequency",
    "classify_execution",
    "extract_followons",
    "memo_date",
    "summarise",
]
