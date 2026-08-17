# Fresh-eyes adversarial review: 3 arms, 20 findings, 6 fixed, 1 HIGH defect handed off

**Status:** MEASURED (2026-08-17). Three independent Opus arms, fresh context, read-only, on
orthogonal surfaces. Charters: `ddm_rar1_review_charters_20260817.md`. **No score claim.**
Frontier hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]` UNMOVED.

## ANSWER FIRST

Fresh eyes found what self-review could not, in both directions:

1. **My tool shipped the exact genus it was built to cure.** Four paths printed
   `RECALL-NEIGHBORHOOD OK: ... top 0 same-topic artifacts all cited` from an EMPTY denominator,
   and `--strict` returned **rc=0** on all four. VACUITY==PASS inside the denominator-checking
   instrument. FIXED — `checked == 0` is now an error, never a pass.
2. **The corrections to `frd077` were TRUE but UNDER-claimed.** Arm 2 re-derived all 24 numeric
   claims from primary artifacts: every one CONFIRMED, several stronger than stated. It then found
   the memo *under*-stated its own defect. FIXED at source.
3. **The whole line of work rests on a refuted assumption** (arm 3). Not fixed — routed, with the
   cheap discriminating test named. This is the most valuable finding of the three.

## The assumption challenge (arm 3) — ACCEPTED, with MAIN's own evidence

Four apparatus layers ask escalating versions of ONE question: *did the author search?*
(memory rule → `STORES CONSULTED` → `cites_prior_recall` → this tool). It is the **fourth
iteration of one idea**, and the idea is refuted by measurement I had already made:

* **The author DID search.** `cites_prior_recall(frd077) = True` — frd077 cites #1058, mz2, ns1,
  source files with line ranges. Not a lazy memo. **The corpus did not return sf1.**
* **My own docstring contains the refutation:** *"Term SELECTION did the work, not the ranker."*
  That sentence says the RANKER is the defect; my response was to make `--terms` the supported
  mode — handing the hard part back to the author. Choosing `"FiLM row lever seg zeroing
  quantiser"` requires already suspecting what you seek. The retrieval bootstrap problem was
  relocated, not solved.
* **n=1.** Three rankings, ONE incident, in a repo whose standing law calls n=1 "not evidence —
  EVER." I applied that allergy to score rows and exempted the instrument I built.
* **Advisories measurably decay here.** Over 25,313 Bash invocations: `corpus_query` executed
  **10** times (0.04%) while named in 87 memos; `convene.py` and `suggest_sister_links.py`
  **0**. The advisory sink `verdict_scope_advisories.jsonl` shows **9 fires on ONE file across 9
  commits with no behavior change**.

**THE CHEAP DISCRIMINATING TEST (named, not run):** the corpus contains its own ground truth —
each memo's cited-artifact set is a human relevance judgment. Build `(query = memo H1,
relevant = artifacts it cites)` over the ~469 memos of the last 7 days (the extraction code
already exists: `title_terms` + `cite_strength`), then measure **recall@10** for
(1) the incumbent scorer, (2) BM25, (3) embeddings. $0, ~1h, **n≈469 instead of n=1**. Both
outcomes decisive: (1)≪(2) ⇒ the ranker is the defect and the whole attestation ladder is symptom
treatment; (1)≈(2) ⇒ prose is not indexable and the honest cure is the verdict index #936 already
reached and parked.

## Fixed this turn (all verified by measurement)

| # | defect | fix |
|---|---|---|
| S1 | 4 paths → confident OK from `checked==0`; `--strict` rc=0 | `checked==0` is an ERROR; strict rc=1 on inconclusive |
| S2/S3 | bare-substring cites: word "CHARTER" passed **18.3% of 300 memos** on all 27 `CHARTER.md`; one `wc1` silenced all 8 `ddm_wc1_*` (278 codes shared across **746 files**) | **tri-state** `CITE_EXACT / CITE_WEAK / CITE_NONE` — WEAK reported on its own line |
| S4 | self-exclusion by basename dropped 3 other arms' `RECEIPT.md` as "self" | exclude by RESOLVED PATH |
| S5 | `stores=["dag"]` → 4/4 uncited unconditionally (refs are row ids, not paths) | refuse non-path stores by name |
| S6 | 377 memos have no H1 → fallback grabbed TABLE HEADERS; 5 took an "H1" from inside a code fence; 230 emitted glued filenames as one term | skip fences/tables; split underscore tokens |
| S7 | `--terms "   "` suppressed the REFUTED warning and silently used auto-extraction | rc=2 with an explicit message |
| A2-1 | frd077 named ONE implementation and `bits=4` | corrected: **three** implementations, **all** bit depths (2–16) |
| A2-4 | code still said "**deployed** quantiser" — the exact word the memo retracted | corrected in comment AND runtime message |

**Why tri-state, not a boolean:** collapse WEAK→exact and one `wc1` mention silences 8 siblings
including the refuter; collapse WEAK→none and the tool fails its OWN anchor (`sf1` names 2 memos,
so frd077 — which genuinely cites sf1's arm — reads as not citing it). Verified: the control now
reports sf1 as **weak**, discriminating from **none** pre-correction.

## The genuinely new finding neither memo had (arm 2)

**SegNet silently ABSORBS NaN on MPS.** An all-NaN frame yields `logits nan_frac = 0.0` and
predicts Undrivable-everywhere — bit-identical to the all-zero-frame prediction. On **CPU** the
NaN propagates and argmax returns Road-everywhere: the degenerate constant would be
**90,557,431 (76.77%)**, not 59,551,382 (50.48%). So the 50.48% constant is **MPS-specific**, and
the NaN guard's placement (probe the RENDER, before SegNet) is **load-bearing, not incidental** —
a guard after the scorer would be invisible on this axis. Recorded in the code.

## HANDED OFF — HIGH, live, independent of this work

**`tools/costate_digest.py:2251` is a bare `return` inside `if ddm_live:` (:2200).** Everything
from :2253–:2330 is UNREACHABLE in the normal live state — **18 sections dropped**, including
`verdict_scope` and `corpus_recall`. Worse, `:2219` SHADOWS `data["verdict_scope"]` with an
unrelated provenance dict and `:2227` hard-binds `corpus_recall` to `[]`, so a JSON consumer reads
a plausible dict and a plausible empty list and never learns that 7 recall advisories fired in
14 days. Measured absent in default, `--full`, AND `--session-start`. The block's own comment
names THREE sections to suppress; the return takes eighteen.

**This falsifies the standing belief that the recall advisory reaches a reader.** It reaches a
sink whose only reader is dead code. Not fixed here (unread file, three arms running); owner
needed. Sister genus: `measured_object_vs_named_object_20260816` — the one I filed against myself
this morning, recurring inside the recall apparatus.

## Also handed off, honestly

* **The tool is UNWIRED** — 0 callers, 0 tests. Under the 6-pillar canonical-helper discipline it
  must carry `research_only=true` + a named blocker, and it carries neither. Arm 3's disposition
  (keep as a BACKTEST HARNESS for the recall@10 test, drop the advisory framing, do NOT wire a
  fifth nag) is the one I find persuasive.
* **`#1085` cited in the docstring is a bare-id collision** (that id is the ALWAYS-KEEP-THE-PAYLOAD
  row) — re-introduced 2h32m after commit `173c0a10fd` cured the same collision elsewhere. The
  m89 law is CITE CONTENT, never a bare id.
* **The recall apparatus covers ≤5% of the corpus** — EVIDENCE leg 331 files (4.5%), DEPTH leg
  ~0.5%, and sf1 is in **neither**. MAIN's earlier claim ("one leg misses the class") is correct
  and UNDERSTATED.
* **300 tests GREEN** across the touched/adjacent modules; `--select F` clean; both modules import.
