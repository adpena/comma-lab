#!/usr/bin/env python3
"""Recall-NEIGHBORHOOD check: does a new memo cite the corpus that already covers its topic?

WHY THIS EXISTS (operator correction 2026-08-17, third recurrence of
"never recall from working memory alone"). The recall apparatus already had two legs and
BOTH passed a memo that missed the one decisive prior artifact:

  * `triality_drift_detector.is_ledger_or_dag_append` -> False for `.omx/research/*.md`
    verdict/finding memos. The highest-volume, highest-consequence doc class was OUT OF SCOPE.
  * `triality_drift_detector.cites_prior_recall` -> True on the same memo. It is an EXISTENCE
    check: "did you cite SOMETHING?" A memo that confidently cites three stale artifacts and
    misses the one decisive one scores 100%.

That is the `jr1`/#1084 genus: **the gauge had no denominator**. This check supplies one.
It does NOT introduce a new instrument -- `tools/corpus_query.py` already ranks the corpus by
topic and would have surfaced the missed artifact at rank 3 in seconds. The defect was that
nothing RAN it at the moment of writing (the #936 write-only-API genus). So this is a WIRING.

MEASURED ANCHOR (the incident, both controls executable from repo history):
    memo   `.omx/research/ddm_frd077_lever_verdict_and_zero_row_nan_20260817.md`
    missed `.omx/research/ddm_sf1_semantic_film_pose_map_20260817.md` -- landed the SAME DAY,
           same directory, "film" in the filename, and it REFUTED the memo's blast-radius claim.
    Control, run against both revisions with the hand-chosen terms below:
      pre-correction  13f793a1f1 (0 sf1 mentions) -> sf1 PRESENT in the uncited list.
      post-correction HEAD       (5 sf1 mentions) -> sf1 ABSENT; every other row identical.
    The check still emits ADVISORY post-correction (other uncited rows remain) -- the control
    criterion is **"sf1 leaves the list"**, NOT "the check passes". Stated exactly, because a
    fresh-eyes review measured that the earlier "must PASS" wording did not reproduce.

ADVISORY BY CONSTRUCTION (fmtools law: advisory on semantic surfaces, never blocking). Topic
overlap is a semantic judgement; a false positive must never refuse a landing. `--strict`
exists for tests and for an author who wants a hard stop, and is never the default.

FAIL-LOUD ON AN EMPTY DENOMINATOR (fixed 2026-08-17 after fresh-eyes finding S1). `checked == 0`
is an ERROR, never an OK. Four measured paths used to print
`RECALL-NEIGHBORHOOD OK: ... top 0 same-topic artifacts all cited` -- unknown `stores` value,
`--terms` that tokenize away entirely, an absent `.omx/research`, and `--top-k <= 0` -- and all
four returned rc=0 even under `--strict`. That is `VACUITY==PASS` inside the instrument built
to cure the missing-denominator genus. A check whose failure mode is a confident pass is worse
than no check.

AMBIGUOUS NAMING IS NOT CITATION (fixed 2026-08-17, findings S2/S3/S4). A reference is only
"cited" when the memo names THAT artifact, not a group containing it. MEASURED in this corpus:
68 basenames are duplicated across 399 files (85x `NEXT_IF_RESUMED.md`, 72x `RECEIPT.md`,
27x `CHARTER.md`), and 278 arm codes are shared by more than one memo covering 746 files
(8.5%). Bare-substring matching therefore let one prose mention of the word "CHARTER" pass
18.3% of memos on all 27 `CHARTER.md` artifacts, and one mention of `wc1` silence all 8
`ddm_wc1_*` memos -- including whichever one refutes you. Ambiguous names now count as NOT
cited, so the advisory fires. Conservative in the direction that preserves signal.

⚠ AUTO-EXTRACTED TERMS ARE REFUTED (MEASURED 2026-08-17, both controls executed). Three
rankings were tried against the incident above; NONE surfaced the decisive artifact:

  | ranking over auto-extracted title terms | sf1 rank | corrected memo |
  |---|---|---|
  | `corpus_query` raw score                | 35 (not in top 6) | fires 6/6 (no discrimination) |
  | score / sqrt(doc chars)                 | ~290-400, procedure not checked in | over-corrects to short docs |
  | filename-token overlap >= 2             | absent (shares only `film`) | 1 candidate, wrong one |

MECHANISM, corrected 2026-08-17 by fresh-eyes score decomposition -- the earlier "un-normalized
term frequency, so long documents win" explanation is REFUTED BY THE CODE. `corpus_query`
scores `min(total_hits,100)*0.3`, which **saturates at exactly 30.0 for every top candidate**,
so frequency carries ZERO ordering information; and the score already contains a
length-normalized density term that FAVORS short documents (7.7 and 4.1 for the short files vs
1.9 and 0.8 for the long ones). The real discriminator is `distinct_terms * 10` -- TERM
COVERAGE. Length is a confound for coverage, not the mechanism. Consequence for the next
attempt: the lever is IDF / rare-term weighting, NOT length normalization. Recording the
corrected mechanism matters more than the dead rankings, because the wrong one misdirects.

WHAT IS MEASURED TO WORK: a HAND-CHOSEN discriminative query. `corpus_query.py "FiLM row lever
seg zeroing quantiser"` put sf1 at **rank 3** in seconds (reproduced). Term SELECTION did the
work, not the ranker -- which is why `--terms` is the supported mode.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_query import run_query

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESEARCH_DIR = _REPO_ROOT / ".omx" / "research"

# Only stores whose `ref` is a real file path can be judged by "does the memo name this file?".
# `dag`/`council`/`tasks`/`equations` emit row ids or `path :: ## FEED-xx` fragments; a fresh-eyes
# review MEASURED `stores=["dag"]` returning 4/4 uncited unconditionally (a guaranteed flood, the
# #1085 failure this file's own docstring warns about). Refuse them by name instead of flooding.
_PATH_STORES = frozenset({"research"})

# Words that carry no topic signal. Deliberately small: over-stripping loses the discriminator.
_STOP = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been", "it", "its",
    "of", "to", "in", "on", "at", "for", "from", "by", "with", "as", "that", "this", "these",
    "not", "no", "but", "so", "if", "then", "than", "what", "which", "when", "where", "how",
    "does", "do", "did", "has", "have", "had", "can", "could", "will", "would", "makes", "make",
    "one", "two", "his", "her", "their", "our", "we", "i", "you", "they",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")
# Arm/memo codes look like ddm_<code>_... ; the code is how memos cite each other in prose.
_MEMO_CODE = re.compile(r"^ddm_([a-z0-9]{2,8})_")

_INDEX: _CorpusIndex | None = None


class _CorpusIndex:
    """How many distinct artifacts answer to each name. Ambiguity is the whole point.

    A basename, stem, or arm code that maps to more than one file cannot IDENTIFY an artifact --
    naming it names a GROUP. Counting that as a citation is a FALSE NEGATIVE that suppresses the
    advisory this tool exists to raise, which is the expensive direction to be wrong in.
    """

    def __init__(self, root: Path) -> None:
        self.basenames: Counter[str] = Counter()
        self.stems: Counter[str] = Counter()
        self.codes: Counter[str] = Counter()
        if not root.is_dir():
            return
        for path in root.glob("**/*.md"):
            if not path.is_file():
                continue
            name = path.name
            self.basenames[name] += 1
            stem = name[:-3] if name.endswith(".md") else name
            self.stems[stem] += 1
            m = _MEMO_CODE.match(stem)
            if m and any(ch.isdigit() for ch in m.group(1)):
                self.codes[m.group(1)] += 1

    def unique_basename(self, name: str) -> bool:
        return self.basenames.get(name, 0) <= 1

    def unique_stem(self, stem: str) -> bool:
        return self.stems.get(stem, 0) <= 1

    def unique_code(self, code: str) -> bool:
        return self.codes.get(code, 0) <= 1


def corpus_index(root: Path | None = None) -> _CorpusIndex:
    """Module-cached ambiguity index over `.omx/research`."""
    global _INDEX
    if root is not None:
        return _CorpusIndex(root)
    if _INDEX is None:
        _INDEX = _CorpusIndex(_RESEARCH_DIR)
    return _INDEX


def ref_path(ref: str) -> str | None:
    """The file path a `ref` denotes, or None when it does not denote one.

    `dag` refs arrive as `path :: ## FEED-xx`; row-id refs (council/tasks) denote no file at all.
    Returning None makes the caller treat the artifact as UNJUDGEABLE rather than silently
    uncited-and-reported.
    """
    head = str(ref or "").split("::", 1)[0].strip()
    if not head or not head.endswith(".md"):
        return None
    return head


def title_terms(text: str, max_terms: int = 12) -> list[str]:
    """Topic terms from the memo's own H1 (its self-declared subject), stopwords removed.

    The H1 is used rather than the whole body because the body is long and its frequent tokens
    are dominated by boilerplate; the title is the author's own statement of what the memo is
    ABOUT, which is exactly the axis a neighborhood should be computed on.

    Fenced code and table rows are skipped: a fresh-eyes scan of all 8,784 research memos found
    5 files whose "H1" was a bash comment inside a fence (`# Output:`, `# Returns: 42 ...`) and a
    377-file no-H1 population whose first-body-line fallback grabbed markdown TABLE HEADERS
    (`| path | branch | uncommitted | ... |`) as the memo's subject. Both are pure boilerplate.
    """
    head = ""
    in_fence = False
    lines = str(text or "").splitlines()
    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if s.startswith("# "):
            head = s[2:]
            break
    if not head:  # no H1 -> first non-empty, non-frontmatter, non-table, non-fenced line
        in_fence = False
        for line in lines:
            s = line.strip()
            if s.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or not s:
                continue
            if s.startswith(("---", ">", "#", "|", "*", "-", "=")):
                continue
            head = s
            break
    out: list[str] = []
    for m in _WORD.finditer(head):
        w = m.group(0).lower()
        # Split underscore-glued filename tokens: `ddm_wc1_advisory_decode_wallclock` survived as
        # ONE term (measured on 230 files, 261 of which emit <= 2 terms total), which made the
        # neighborhood rank on whatever generic word sat beside it.
        for part in w.split("_"):
            if part in _STOP or len(part) < 3 or part in out:
                continue
            out.append(part)
            if len(out) >= max_terms:
                return out
    return out


#: Citation strength. A boolean cannot carry what the corpus actually contains.
CITE_NONE = "none"
CITE_WEAK = "weak"     # names a GROUP that contains this artifact (ambiguous basename/arm code)
CITE_EXACT = "exact"   # names THIS artifact and no other


def cite_strength(text: str, ref: str, index: _CorpusIndex | None = None) -> str:
    """How specifically does the memo name THIS artifact?

    Three forms, in descending specificity:
      1. `parent_dir/basename`, or a corpus-unique basename/stem  -> CITE_EXACT.
      2. an AMBIGUOUS basename or arm code                        -> CITE_WEAK.
      3. nothing                                                  -> CITE_NONE.

    WEAK is a real third state, not a rounding of the other two, and collapsing it either way
    is a measured defect:
      * Round WEAK->exact and one prose mention of `wc1` silences all 8 `ddm_wc1_*` memos --
        including whichever sibling refutes you. MEASURED: 278 arm codes are shared by >1 memo
        across 746 files (8.5% of the corpus); one mention of the word "CHARTER" passed 18.3%
        of memos on all 27 `CHARTER.md` artifacts.
      * Round WEAK->none and this tool fails its OWN anchor: `sf1` names 2 memos, so the
        frd077 memo -- which genuinely does cite sf1's arm -- reads as not citing it.
    So WEAK is reported as its own line: the reader sees "you named the arm, not the artifact"
    and decides. That is more information than either collapse, and it is the honest state.
    """
    body = str(text or "")
    path = ref_path(ref)
    if path is None:
        return CITE_NONE
    p = Path(path)
    name = p.name
    if not name:
        return CITE_NONE
    idx = index if index is not None else corpus_index()
    parent_qualified = f"{p.parent.name}/{name}" if p.parent.name else name
    if parent_qualified in body:
        return CITE_EXACT
    stem = name[:-3] if name.endswith(".md") else name
    if name in body:
        return CITE_EXACT if idx.unique_basename(name) else CITE_WEAK
    if stem and stem in body:
        return CITE_EXACT if idx.unique_stem(stem) else CITE_WEAK
    m = _MEMO_CODE.match(stem)
    if m:
        code = m.group(1)
        # The second token is an ARM CODE only when it carries a digit (sf1, qs2, frd077, wd3,
        # pk4, rfo1 -- every real one does). Without this guard the regex reads
        # `ddm_deferral_queue_ledger_*` as code `deferral`, and any memo merely SAYING the word
        # "deferral" counts as CITING that ledger -- a measured FALSE NEGATIVE.
        # Word-bounded so `sf1` does not match inside `sf10`/`xsf1`.
        if any(ch.isdigit() for ch in code) and re.search(rf"\b{re.escape(code)}\b", body):
            return CITE_EXACT if idx.unique_code(code) else CITE_WEAK
    return CITE_NONE


def cites(text: str, ref: str, index: _CorpusIndex | None = None) -> bool:
    """Back-compat boolean: WEAK counts as cited. Prefer `cite_strength`."""
    return cite_strength(text, ref, index) != CITE_NONE


def uncited_neighbors(
    memo_path: str | Path,
    top_k: int = 6,
    stores: list[str] | None = None,
    terms_override: list[str] | None = None,
) -> dict:
    """Rank the corpus by topic; return the top hits the memo does NOT cite.

    `terms_override` is the SUPPORTED mode (measured to work). Auto-extracted title terms are
    refuted -- see the module docstring's ranking table. Pass 4-8 DISCRIMINATIVE terms: the ones
    that name this memo's specific object, not its generic verbs.

    Returns {"memo", "terms", "checked", "uncited": [{ref, score, date}], "error"}.
    `error` is set (and `checked` is 0) whenever the neighborhood could not be examined --
    an empty denominator is never reported as a pass.
    """
    p = Path(memo_path)

    def fail(msg: str) -> dict:
        return {"memo": str(p), "terms": [], "checked": 0, "uncited": [], "error": msg}

    if top_k <= 0:
        return fail(f"--top-k must be >= 1 (got {top_k}); nothing would be examined")
    use_stores = list(stores) if stores else ["research"]
    bad = [s for s in use_stores if s not in _PATH_STORES]
    if bad:
        return fail(
            f"unsupported store(s) {bad}: citation is judged by filename, and only "
            f"{sorted(_PATH_STORES)} emit file-path refs (dag/council/tasks emit row ids)"
        )
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return fail(str(exc))

    terms = list(terms_override) if terms_override else title_terms(text)
    if not terms:
        return fail("no topic terms (empty --terms, or no usable H1/first line)")

    res = run_query(" ".join(terms), stores=use_stores, top=top_k + 4)
    hits = res.get("hits", [])
    if not hits:
        return fail(
            f"query returned 0 hits for terms {terms} -- terms may have tokenized away, "
            f"or the corpus store is empty/absent"
        )

    self_resolved = p.resolve()
    uncited: list[dict] = []
    checked = 0
    idx = corpus_index()
    for hit in hits:
        ref = str(hit.get("ref") or "")
        rp = ref_path(ref)
        if rp is None:
            continue
        # Self-exclusion by RESOLVED PATH, not basename: 399 files sit in duplicate-basename
        # groups, and basename matching silently discarded 3 other arms' RECEIPT.md as "self".
        try:
            if (_REPO_ROOT / rp).resolve() == self_resolved:
                continue
        except OSError:
            pass
        # BOUND BEFORE COUNTING. The reverse order (count, then break on >) leaves `checked`
        # at top_k+1 and the report then overstates its own denominator by one -- measured on
        # this tool's first run (`--top-k 5` printed "top 6"). Sister of #1084.
        if checked >= top_k:
            break
        checked += 1
        strength = cite_strength(text, ref, idx)
        if strength != CITE_EXACT:
            uncited.append({"ref": rp, "score": hit.get("score"), "date": hit.get("date"),
                            "cite": strength})
    if checked == 0:
        return fail(
            "0 same-topic artifacts examined (every hit was self or non-path) -- "
            "an empty denominator is not a pass"
        )
    return {"memo": str(p), "terms": terms, "checked": checked, "uncited": uncited, "error": None}


def format_report(result: dict) -> str:
    if result.get("error"):
        return f"RECALL-NEIGHBORHOOD INCONCLUSIVE: {result['memo']}: {result['error']}"
    memo, terms = result["memo"], ", ".join(result["terms"])
    if not result["uncited"]:
        return (f"RECALL-NEIGHBORHOOD OK: {memo}\n"
                f"  topic [{terms}] -- top {result['checked']} same-topic artifacts all cited.")
    lines = [
        f"RECALL-NEIGHBORHOOD ADVISORY: {memo}",
        f"  topic [{terms}]",
        f"  {len(result['uncited'])} of the top {result['checked']} same-topic artifacts are "
        f"NOT cited by this memo:",
    ]
    for u in result["uncited"]:
        lines.append(f"    - {u['ref']}  (score {u['score']}, {u['date']})")
    lines.append("  These may already answer, refute, or supersede what this memo concludes.")
    lines.append("  Read them before landing, then cite or explicitly rule out each one.")
    lines.append("  ADVISORY -- never blocks. Topic overlap is a judgement, not a defect.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("memos", nargs="+", help="memo path(s) to check")
    ap.add_argument("--top-k", type=int, default=6, help="how many same-topic hits to demand")
    ap.add_argument("--terms", default=None,
                    help="SUPPORTED MODE: space-separated DISCRIMINATIVE topic terms. "
                         "Auto-extraction from the title is REFUTED (see module docstring).")
    ap.add_argument("--strict", action="store_true",
                    help="rc=1 when any top hit is uncited OR the check is inconclusive "
                         "(default: advisory rc=0)")
    args = ap.parse_args(argv)

    override = args.terms.split() if args.terms else None
    if override == []:
        # `--terms "   "` is not the same as omitting --terms: it used to suppress the NOTE and
        # silently fall back to the refuted auto-extractor.
        print("ERROR: --terms was empty/whitespace. Omit it to use auto-extraction, or "
              "pass real terms.")
        return 2
    if override is None:
        print("NOTE: auto-extracted terms are REFUTED as a ranker (module docstring). "
              "Prefer --terms with 4-8 discriminative words.")
    any_uncited = False
    any_error = False
    for m in args.memos:
        result = uncited_neighbors(m, top_k=args.top_k, terms_override=override)
        print(format_report(result))
        if result.get("error"):
            any_error = True
        elif result["uncited"]:
            any_uncited = True
    return 1 if (args.strict and (any_uncited or any_error)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
