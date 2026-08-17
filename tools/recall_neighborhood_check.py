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
    pre-correction commit 13f793a1f1 cites sf1 0 times  -> this check must FIRE.
    post-correction working tree cites it               -> this check must PASS.

ADVISORY BY CONSTRUCTION (fmtools law: advisory on semantic surfaces, never blocking). Topic
overlap is a semantic judgement; a false positive must never refuse a landing. `--strict`
exists for tests and for an author who wants a hard stop, and is never the default.

⚠ AUTO-EXTRACTED TERMS ARE REFUTED (MEASURED 2026-08-17, both controls executed). Three
rankings were tried against the incident above; NONE surfaced the decisive artifact:

  | ranking over auto-extracted title terms | sf1 rank | corrected memo |
  |---|---|---|
  | `corpus_query` raw score                | not in top 6 | fires 5/6 (no discrimination) |
  | score / sqrt(doc chars)                 | 19       | over-corrects to short docs |
  | filename-token overlap >= 2             | absent (shares only `film`) | 1 candidate, wrong one |

MECHANISM, measured: `corpus_query` scores UN-NORMALIZED term frequency, so omnibus documents
win on length -- the false positives were 330,488 / 133,021 / 64,997 chars against sf1's 33,479.
Length-normalizing inverts the bias instead of removing it. So the check FAILED its own bar
(`structural_beats_procedural_and_the_detector_that_zeroes_on_the_cure`): the corrected memo
scored identically to the uncorrected one.

WHAT IS MEASURED TO WORK: a HAND-CHOSEN discriminative query. `corpus_query.py "FiLM row lever
seg zeroing quantiser"` put sf1 at **rank 3** in seconds. Term SELECTION did the work, not the
ranker -- which is why `--terms` is the supported mode and auto-extraction is labelled below.
Filed rather than deleted so the next attempt does not re-run these three dead scorings
(sister of #1085: a check that floods is worse than no check, because it trains the reader to
ignore it).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_query import run_query  # noqa: E402

# Words that carry no topic signal. Deliberately small: over-stripping loses the discriminator.
_STOP = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been", "it", "its",
    "of", "to", "in", "on", "at", "for", "from", "by", "with", "as", "that", "this", "these",
    "not", "no", "but", "so", "if", "then", "than", "what", "which", "when", "where", "how",
    "does", "do", "did", "has", "have", "had", "can", "could", "will", "would", "makes", "make",
    "one", "two", "its", "his", "her", "their", "our", "we", "i", "you", "they",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")
# Arm/memo codes look like ddm_<code>_... ; the code is how memos cite each other in prose.
_MEMO_CODE = re.compile(r"^ddm_([a-z0-9]{2,8})_")


def title_terms(text: str, max_terms: int = 12) -> list[str]:
    """Topic terms from the memo's own H1 (its self-declared subject), stopwords removed.

    The H1 is used rather than the whole body because the body is long and its frequent tokens
    are dominated by boilerplate; the title is the author's own statement of what the memo is
    ABOUT, which is exactly the axis a neighborhood should be computed on.
    """
    head = ""
    for line in str(text or "").splitlines():
        s = line.strip()
        if s.startswith("# "):
            head = s[2:]
            break
    if not head:  # no H1 -> fall back to the first non-empty, non-frontmatter line
        for line in str(text or "").splitlines():
            s = line.strip()
            if s and not s.startswith(("---", ">", "#")):
                head = s
                break
    out: list[str] = []
    for m in _WORD.finditer(head):
        w = m.group(0).lower()
        if w in _STOP or len(w) < 3 or w in out:
            continue
        out.append(w)
        if len(out) >= max_terms:
            break
    return out


def cites(text: str, ref: str) -> bool:
    """True iff the memo names this artifact -- by full basename, by stem, or by arm code.

    Memos in this corpus cite each other three ways in practice, all of which count:
    the full filename, the stem, or the bare arm code (`sf1`) in prose. Anything narrower
    would fire on correctly-recalled memos and train the reader to ignore the check.
    """
    body = str(text or "")
    name = Path(str(ref or "")).name
    if not name:
        return False
    if name in body:
        return True
    stem = name[:-3] if name.endswith(".md") else name
    if stem and stem in body:
        return True
    m = _MEMO_CODE.match(stem)
    if m:
        code = m.group(1)
        # The second token is an ARM CODE only when it carries a digit (sf1, qs2, frd077, wd3,
        # pk4, rfo1 -- every real one does). Without this guard the regex reads
        # `ddm_deferral_queue_ledger_*` as code `deferral`, and any memo merely SAYING the word
        # "deferral" counts as CITING that ledger -- a measured FALSE NEGATIVE that silently
        # suppresses advisories. English words in that slot are never arm codes.
        if any(ch.isdigit() for ch in code):
            # Word-bounded so `sf1` does not match inside `sf10`/`xsf1`.
            if re.search(rf"\b{re.escape(code)}\b", body):
                return True
    return False


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
    """
    p = Path(memo_path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"memo": str(p), "terms": [], "checked": 0, "uncited": [], "error": str(exc)}

    terms = list(terms_override) if terms_override else title_terms(text)
    if not terms:
        return {"memo": str(p), "terms": [], "checked": 0, "uncited": [],
                "error": "no topic terms in title"}

    res = run_query(" ".join(terms), stores=stores or ["research"], top=top_k + 4)
    self_name = p.name
    uncited: list[dict] = []
    checked = 0
    for hit in res.get("hits", []):
        ref = str(hit.get("ref") or "")
        if Path(ref).name == self_name:  # the memo always ranks #1 on its own terms
            continue
        # BOUND BEFORE COUNTING. The reverse order (count, then break on >) leaves `checked`
        # at top_k+1 and the report then overstates its own denominator by one -- measured on
        # this tool's first run (`--top-k 5` printed "top 6"). Sister of #1084: the denominator
        # is the thing to get right, including in the instrument that checks denominators.
        if checked >= top_k:
            break
        checked += 1
        if not cites(text, ref):
            uncited.append({"ref": ref, "score": hit.get("score"), "date": hit.get("date")})
    return {"memo": str(p), "terms": terms, "checked": checked, "uncited": uncited, "error": None}


def format_report(result: dict) -> str:
    if result.get("error"):
        return f"RECALL-NEIGHBORHOOD: {result['memo']}: {result['error']} (advisory)"
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
                    help="rc=1 when any top hit is uncited (default: advisory rc=0)")
    args = ap.parse_args(argv)

    override = args.terms.split() if args.terms else None
    if override is None:
        print("NOTE: auto-extracted terms are REFUTED as a ranker (module docstring). "
              "Prefer --terms with 4-8 discriminative words.")
    any_uncited = False
    for m in args.memos:
        result = uncited_neighbors(m, top_k=args.top_k, terms_override=override)
        print(format_report(result))
        if result["uncited"]:
            any_uncited = True
    return 1 if (args.strict and any_uncited) else 0


if __name__ == "__main__":
    raise SystemExit(main())
