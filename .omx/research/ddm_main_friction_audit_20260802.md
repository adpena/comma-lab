# MAIN friction audit — the walls I actually hit this session, and the design changes that dissolve them

**UTC** 2026-08-03 · **author** MAIN (Opus) · **prompted by** operator 2026-08-02: *"Examine all of
your reasoning… did you face challenges or hit walls that would be greatly simplified by design-level
changes?"*

**Scope:** ONE session (2026-08-02 → 08-03). Every row below is something I hit, with the receipt.
This is not a wishlist — items I did not personally trip over are excluded.

**The meta-pattern, stated first because it reframes the list:**

> **Almost every wall was an INSTRUMENT failure, not a knowledge failure.** The corpus knew the
> answer; my ability to *ask* it, or to *keep* an answer once found, is what broke. Four separate
> times an instrument returned a confident wrong answer rather than an error. That is the
> **vacuity==pass genus** (`m50`) showing up in the orchestration layer instead of the gate layer —
> and it means the highest-leverage design work right now is on *how MAIN queries and retains*, not
> on new measurement capability.

---

## 1. FINDINGS DIE WITH THE ARM — the single most expensive one today

**What happened.** Four arms were terminated mid-flight by a provider weekly limit. Their last words:

| arm | final message | recoverable? |
|---|---|---|
| `ddm_gd2` | *"Phase A has surfaced a **structural blocker that changes what the slot should measure** — let me confirm the bytes and report"* | **NO.** The blocker is gone. |
| `ddm_pb3` | *"Rigorous measured win in hand. Writing the final memo."* | **YES — only by luck**: it had already written files to disk |
| `ddm_cb2` | *"Consumption proved (76.3% of camera pixels change)"* | partial |
| `ddm_bo1` | *"the order agent returned prior art that **corrects my §0**"* | partial |

**gd2's finding is permanently lost.** It was about the highest-value row on the board.

**Why the existing apparatus didn't save it.** `subagent_checkpoint.py` carries `step`, `status`,
`files_touched`, `next_action`, `notes` — **it has no FINDINGS field.** The crash-resume protocol was
designed to answer *"where do I resume?"*, not *"what did we learn?"* Those are different questions
and only the first is served.

**This is task #878 with a corrected scope.** #878 says "arm final-message NEXT-IF-RESUMED blocks are
structurally unpersisted (9 occurrences in 5 files)". The real population is not 9 occurrences — it is
**every arm death, always**, and it is silent.

**DESIGN CHANGE (small, high leverage):** add a **`findings`** field to the checkpoint schema and make
the standard subagent contract require a one-line finding at every checkpoint, not only at the end.
Cost: one schema field plus one contract sentence. Today it would have saved gd2's blocker outright.
Stronger variant: arms write the memo **incrementally** (stub all sections at step 1, fill as they go),
so a kill leaves a partial memo instead of nothing — which is exactly why pb3 survived and gd2 did not.

## 2. NO RELEASE PATH FOR A DEAD ARM'S FILE CLAIM (new, found today)

Catalog #340 refused my pb3 recovery commit because **pb3's own 16-minute-old claim was still live** —
on a process that no longer existed. I wrote a terminal `blocked` checkpoint with empty
`files_touched` first; **the guard ignored it**, because it unions claims over a 60-minute window and
has no concept of release.

**Consequence:** any killed arm locks its own files for an hour, and the only exit is the paired-env
override — which trains the reflex of overriding a guard that is *usually* right.

**DESIGN CHANGE:** a terminal checkpoint (`status ∈ {complete, blocked-terminal}`) should **clear**
that arm's claims, and the guard should prefer a liveness check over pure recency. Two-landing.

## 3. TWO TASK LEDGERS, NO JOIN

The harness TaskList (~911 rows) and `.omx/state/canonical_task_status.jsonl` (409 rows, 264 ids) are
**different stores**. Arms see only the repo. `na1` was told to sweep 7 ids; **all 7 were absent**, and
10 of 14 cited ids overall. Verified independently in Python.

Cure in force today: **cite CONTENT, never bare ids.** Structural bridge OWED. Note the id spaces
aren't even the same shape (canonical max id = 99292), so re-numbering would be wrong.
Memory `m89`; this is the concrete instance of the long-open MEMO-corpus ↔ TASK-row JOIN (#880).

## 4. MEASURED CONSTANTS ARE RE-TYPED, NOT SOURCED — I published a wrong denominator twice

I published **"1% of gap = 11,892 B"**. The equation module caught my arithmetic on its first run
(→ 10,908). Then `ddm_na1` caught the **input**: PR130 is **191,052 B, not 190,952** (190,952 gives
floor 0.1720751, which does not reproduce PR130's published 0.172141; 191,052 gives 0.1721417, which
does). Final: gap **0.7262358**, 1% = **10,907 B**.

**The equation was right both times; its inputs were not.** We have a value-provenance ladder for
*derived* constants and CONSTANTS-ARE-POISON for *borrowed* ones — but a **measured external constant
retyped by hand into a memo** has no rung, and it propagated into an executable law.

**DESIGN CHANGE:** a small **measured-constants registry** (value + source artifact + measurement
date + who), and canonical equations **import** from it instead of accepting literals. The gap
equation already refuses unsourced rows at the dataclass boundary — this extends the same fail-closed
idea one level up, to where the number is typed.

## 5. MY AD-HOC SHELL PROBES ARE THE RECURRING BROKEN INSTRUMENT (3× in one session)

1. **zsh parsed `[^0-9]` as a math expression** — my id-existence loop errored for all 7 ids and printed
   uniform `NOT-FOUND`. **A broken command produced a plausible, uniform, wrong answer.** Caught only
   because the uniformity was suspicious.
2. **`grep -rilE` over `.omx/research/` (9,704 docs) timed out at 2 min** — and a scoped re-run found
   matches a timed-out "count: 0" had implied were absent.
3. **`--files "$FILES"`** passed six paths as one string; the serializer rejected it (correctly).

**DESIGN CHANGE:** a tiny typed query surface for the three things I keep asking in shell —
*does this id exist in store X · which docs mention this content · what changed in window W* — each
**fail-closed and reporting its denominator**. `tools/corpus_query.py` is the model; it just doesn't
cover ids or diffs.

## 6. THE CORPUS INDEX SILENTLY COVERS 76%

`ls .omx/research` = **9,706** docs vs the corpus index's **7,398** — **~2,300 unindexed, cause
unknown** (surfaced by `ddm_rd2`). Every corpus-scoped negative in this campaign inherits that hole,
and the index reports no denominator, so a partial index and a complete one look identical.

**DESIGN CHANGE:** the query surface reports `matched / indexed / total` on every call. Textbook
vacuity==pass cure; we already wrote the law, we just haven't applied it here.

## 7. CONTEXT FLOODING FROM THE TASK LIST

Nearly every tool result carries the full ~900-row task list. I need on the order of ten rows. It
crowds out live receipts and makes the *newest* signal the hardest to see — which is precisely
backwards for an orchestrator whose job is to route the newest finding.

**DESIGN CHANGE:** show recently-touched + explicitly-referenced rows by default; full list on request.

---

## Ranked by (leverage × cheapness)

1. **§1 findings-in-checkpoint / incremental memo** — would have saved gd2's blocker today. One schema
   field + one contract sentence.
2. **§2 claim release** — two-landing; today it forced a guard override that was legitimate but
   habit-forming.
3. **§4 measured-constants registry** — the only item here that has already corrupted a published
   number, twice.
4. **§5+§6 typed query surface with denominators** — cures a class that produced three wrong answers
   in one session.
5. **§3 ledger JOIN** — highest structural value, largest build; content-citation is a working cure
   meanwhile.

**What I am NOT claiming:** none of this is why the pointer sits where it does. The score gap is a
representation problem, not an apparatus problem. These changes buy *retained* signal per session —
which matters because today the single most expensive loss was a finding that existed, was measured,
and evaporated.
