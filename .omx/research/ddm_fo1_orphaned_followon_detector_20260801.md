# ddm_fo1 (#870) — the orphaned-cheap-follow-on class: measured much smaller than claimed, and the 22 minutes of already-paid scorer time nobody harvested

**UTC 2026-08-01** · axis **[macOS-CPU advisory]** · `score_claim=false` · `promotion_eligible=false` ·
`pointer_moved=false` · `ready_for_exact_eval_dispatch=false` · **zero scorer forwards** · `upstream/` untouched.
**review_status:** pre-registered controls · own-round-1-and-2-reviewed (six defects found in my own code,
four of which produced confident FALSE ORPHANED) · control set independently hand-adjudicated.

**Pointer honesty, first.** Nothing here lowered any score. `effective_frontier` **0.172** (official
leaderboard) UNMOVED; our own-vehicle line **v4d 0.9476091** UNMOVED. This is APPARATUS — MEANS, not END.

**STORES CONSULTED:** `tools/corpus_query.py` (research 7364 · equations 865 · memory 2044 · dag 909 ·
council 292 · tasks 396 · docs 96) → loaded memory `built_elsewhere_unwired_is_p0_20260801` (the §"DEF CON
a thousand" section that files this class), `ddm_gd5_grade5_detector_is_not_autoderivable_20260801`,
`ddm_wr1_reverse_waterfill_20260729`, `ddm_tw1_token_waterfill_state_dependence_20260801`,
`ddm_wr2_wire_or_retire_adjudication_20260801`, `ddm_ck1_composed_kneeA_20260729`,
`src/tac/scope_ledger.py`, `src/tac/witness_dsl/activation_ledger.py`, `tools/costate_digest.py`.
**Deliberately NOT loaded:** the pose-menu receipts (owned by `ddm_pw1`), the waterfill price curve
(owned by `ddm_tw1` — I consumed its §0 corrections as CONTROLS and re-derived the one that decides my
landing), the 7-row WIRE-or-RETIRE verdicts (owned by `ddm_wr2`).

---

## §0 HEADLINE (answer first)

**The class is real, the instrument is buildable — and its live population is near zero, because sister
arms drain follow-ons within 24–72 h. The failure that actually remains is RETRIEVAL, not orphaning: our
memos and ledgers keep asserting "never run" about measurements that ran hours earlier.**

Three measured statements, in the order that decides them:

1. **My own dispatch's orphan list was 5-of-6 STALE.** An independent hand-adjudication of the six named
   live instances found **five already EXECUTED**, most of them the same day. The one true non-execution
   (`stage_ck1_composed_gate.sh`) was *deliberately* not fired — its own memo says *"a NEGATIVE control,
   NOT the winning candidate. Do not spend the winning slot on it."* That is a correctly-retired row, not
   a debt.

2. **The genuinely orphaned thing was a HARVEST, not a measurement.** `stage_wr1_realized_gate.sh` ended
   in four `echo` lines that *described* parsing `report.txt` into a receipt and wrote nothing. Both
   candidates were then really fired — **~22 minutes of real n600 scorer time** — and **neither produced
   a receipt.** I harvested both (§2). This is the class in its most expensive form: not a follow-on
   nobody ran, but one that RAN and whose answer never reached a machine-readable surface.

3. **The detector is honest and mostly says UNKNOWN, which is the correct answer.** Over the live 14-day
   scope: **0 ORPHANED · 4 STAGED · 84 UNKNOWN · 10 EXECUTED**, out of 93 extracted rows across 1,129 of
   7,375 memos. Only ~15% of named follow-ons are decidable from artifacts at all.

---

## §1 WHAT I RE-DERIVED vs WHAT I COULD ONLY ASSUME

| seed claim | status |
|---|---|
| "`wr1` §5 staged gate (BUILT, one-command, **never fired**)" | **FALSE.** `kneeA_gate_run.log` on SSD holds a real n600 `evaluate.py` run (9m41s); `gateB_stdout.log` holds kneeB (11m47s). `ddm_tw1` §0 row 2 had already corrected this. |
| "`wr1` §6 QA08 re-race — orphaned" | **FALSE.** `ddm_tw1` §4 raced it: QA08 = r7 codec `kt_o8_prev5_backoff`, **9.7% → 21.4% WORSE**, so wr1's "can only improve the byte column" is FALSIFIED. |
| "#826 `cell_drop50` (byte-closed, **never evaluated**)" | **FALSE.** `ddm_ep2_20260731/gr1_eval/d1_eval_receipt.json`, 600 samples, S = 2.2027435921. The receipt **predates** the memo calling it un-run. |
| "`ba31` §B.3 split (nobody ran it)" | **FALSE.** `ddm_dc1` ran the decidable version and REFUTED ba31's hypothesis (LABEL 0.082–0.255 B/flip). |
| "`pw1`'s `stage_v4d_realized_gate.sh` — the one that did not rot" | **TRUE, and understated:** it fired **twice**; the second (`v4d_pw1`) moved our own-vehicle line to **S 0.9476091**, ΔS −0.0163787. |
| "gc11–gc17 tabled 47, built 18, never built 23 (49%)" | **NOT re-derived.** Inherited from `ddm_gc17`'s backcast; I did not reproduce it and do not lean on it. |
| pointer `0.7754681` gap / v4d `0.9476091` | **RE-DERIVED** against `canonical_frontier_pointer` + the `v4d_pw1` report. |

**ASSUMED, could not derive:** whether the 84 UNKNOWN rows contain live debt. They are UNKNOWN because
they name no artifact the join can check, not because they were adjudicated. I report the bucket, never a
count of debt inside it.

---

## §2 THE FIRED FOLLOW-ON — both wr1 realized gates, harvested (MEASURED)

`experiments/harvest_wr1_realized_gate_receipt.py` parses an existing report/stdout log, **recomputes S
from its components**, and refuses if the recomputation disagrees with the report's own stated score.
Zero scorer forwards; the scorer time was paid on 07-29 and 07-30.

| candidate | d_seg | d_pose | archive B | **S** | ΔS vs REF 2.256641 | verdict |
|---|---:|---:|---:|---:|---:|---|
| pfs1 D1 (REF) | 0.00389011 | 0.22144216 | 569,996 | 2.256641 | — | — |
| **Knee A** | 0.00553676 | 0.28002128 | 274,333 | **2.409727** | **+0.153086** | **REJECT** |
| **Knee B** | 0.01001419 | 0.48164272 | 174,578 | **3.312299** | **+1.055658** | **REJECT** |

**Term decomposition for Knee A — the finding wr1's own text does not carry:**

| term | REF | realized | Δ |
|---|---:|---:|---:|
| seg (100·d_seg) | 0.389011 | 0.553676 | **+0.164665** |
| pose (√(10·d_pose)) | 1.488093 | 1.673384 | **+0.185290** |
| rate (25·B/37,545,489) | 0.379537 | 0.182667 | **−0.196870** |

**The rate lever delivered exactly what wr1 predicted (−0.1968 S) and the archive still got WORSE**, because
both of wr1 §7's premises are refuted at this instance: "PREDICTED-zero seg cost" (actual +0.164665 S) and
§4's "pose-favorable by construction" (actual +0.185290 S). The 486 zero-flip cells are not free.
Preregistered break-even (wr1 §5 / gc6 row 6) is ΔS < 25·ΔB/U = −0.196870; realized +0.153086 fails it.

**`verdict_scope: INSTANCE.`** This refutes THESE two candidates against THIS reference row. It does not
close the reverse-waterfill family — `ddm_tw1` independently shows the per-unit byte price is
state-dependent, so a re-priced knee is a different object.

Receipts: `/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_{kneeA,kneeB}_realized_gate_receipt.json`
(schema `ddm_wr1_realized_gate.v1`, source-log sha256 bound). **Class fix:** the echo-only harvest in
`stage_wr1_realized_gate.sh` is replaced by a real call, so a future firing cannot silently drop its
answer again.

---

## §3 THE INSTRUMENT (`tac.followon_ledger`) — measured, with denominators

**Extraction is a CONJUNCTION, and both halves are necessary — measured over all 7,375 memos:**

| marker | occurrences | files |
|---|---:|---:|
| bare `$0` | **12,684** | 2,101 | 
| `staged follow-on` | 2 | 2 |
| `NEXT-IF-RESUMED` | 9 | 5 |
| `one measurement away` | 1 | 1 |
| **ACTION ∧ CHEAP (the predicate)** | **741 lines** | **333** |

Bare `$0` alone is `ddm_gd5`'s F1 failure exactly — a four-figure warn-only queue is not a queue. The
seed's other markers fail the opposite way. Only the conjunction carries the class. **The
contract-mandated `LIVE-HYPOTHESES` / `DEAD-ENDS` / `NEXT-IF-RESUMED` lines are nearly absent from disk
(9 occurrences in 5 files) because they are written to the coordinator, not to a memo — the richest
follow-on surface we have is structurally unpersisted.**

**The join keys on ARTIFACT EXISTENCE, never on co-occurrence**, and separates the runner from its output:

| verdict | predicate |
|---|---|
| `EXECUTED` | the named OUTPUT artifact exists |
| `ORPHANED` | the named OUTPUT artifact does not exist, **and every declared artifact tier was readable** |
| `STAGED` | the RUNNER exists but no output is named — built, firing undecidable; adjudicate first |
| `UNKNOWN` | no artifact-shaped token, or a tier could not be read |

**Live 14-day scope: 0 ORPHANED · 4 STAGED · 84 UNKNOWN · 10 EXECUTED** over 93 rows / 1,129 memos
(`ScopeLedger` renders the denominator; an empty scope is `VACUOUS`, never a pass).

**Ranked adjudicate-first table (the whole live output):**

| # | row | evidence | note |
|---|---|---|---|
| 1 | `ddm_deferral_queue_ledger_20260729.md#QA06@L39` | `stage_wr1_realized_gate.sh` present | **RESOLVED by this arm** — both candidates fired and now harvested (§2) |
| 2 | `ddm_deferral_queue_ledger_20260729.md#QA52@L85` | `experiments/ddm_kl1_pose_field_receiver.py` present | ξ trajectory coding — unadjudicated |
| 3 | `generator_description_crux_synthesis_20260719.md#L56` | runner present | unadjudicated |
| 4 | `schmidt_icml2026_optstep_crosswalk_20260721T203954Z.md#L157` | `spec_c1_optimal_form_20260715.py` present | a spec module, not a runnable follow-on — likely a false lead |

**Validated precision.** The 6-item control set is hand-adjudicated from primary artifacts (SSD logs,
receipts, git), never from memo self-description — *three of the six memos state their own status
wrongly*. Against it the join is **correct on 6/6** and, critically, **never labels live debt EXECUTED**
(asserted by a dedicated test). Its ORPHANED precision on the live campaign is **UNVALIDATED at n=0** —
there are currently no ORPHANED rows to check, so I report the bucket as empty rather than claiming a
precision I could not measure.

---

## §4 MY OWN ROUND-1 AND ROUND-2 REVIEW — six defects, four of them false ORPHANED

Every one of these was in code I had already written and believed:

1. **Regex alternation truncated every `.jsonl` to `.json`.** Python alternation is first-match-wins:
   `d2_ep_solve.partial.jsonl` captured as `...partial.json`, a filename existing nowhere → confident
   FALSE ORPHANED. Fixed by ordering longest-first plus `(?![\w])`.
2. **`src/` was absent from the corpus.** `spec_c1_optimal_form_20260715.py` exists at
   `src/tac/witness_dsl/` and was reported never-built. A corpus that omits a root does not report
   "unknown" for it — it reports "missing", the vacuity genus with the sign flipped.
3. **The SSD artifact tiers were unscanned in my digest wiring** → **5 fabricated ORPHANED rows** whose
   receipts were on the mounted volume all along. Now scanned by default, and an unreadable tier
   **degrades ORPHANED to UNKNOWN**: "not produced" and "I could not look" are the same observation.
4. **Partial glob captures.** `ddm_pb1_*_receipt.json` yielded the orphan basename `_receipt.json`.
5. **Joining on incidental context tokens.** The first cut admitted rare-but-non-identifying identifiers
   and called `QA03` executed on `ddm_sb1_20260729` (a cited directory), `QA08` on `prev_coloc`. Rare is
   not IDENTIFYING. Join tokens are now artifact-shaped only.
6. **(round 2) The on-disk index cache is keyed to nothing**, so a caller that narrowed the scope could be
   handed a wider index and see outputs its own scope cannot reach. Now only the default scope is cached.

Defects 1–5 all pushed the same direction: **manufacturing debt**. That is `ddm_gd5`'s measured failure
mode reappearing inside a different instrument, which is the strongest evidence I have that the
conservative default (UNKNOWN over ORPHANED) is the right one.

---

## §5 THE REFRAME — this class is dominated by RETRIEVAL, not by orphaning

The seed's premise is that named cheap follow-ons rot. Measured, they do not — they get drained fast. What
persists is the **stale claim**: `ddm_gc17` made "#826 has never been exact-eval'd" its rank-1
recommendation while the receipt had been in custody 5 h 27 m; `ddm_pw1` wrote *"I did not fire one"*
twelve minutes before its gate fired; the memory that filed #870 lists as live two instances a sister arm
had already closed. **The highest-value use of this instrument is therefore its EXECUTED bucket — telling
us which live "never run" claims are FALSE — not its ORPHANED bucket.**

That is also why the consumer is a *session-time* surface rather than a static list: the answer has a
half-life of about a day.

---

## §6 CROSS-FINDINGS

**→ the `p0_864` / #870 owner.** The memory `built_elsewhere_unwired_is_p0_20260801` §"DEF CON a thousand"
lists five live instances; **two were already closed when it was written** (`wr1` §5 gate fired 07-29/30;
`wr1` §6 QA08 raced by `tw1`) and two more closed the same day. Please restate the class as *stale
never-run claims*, and note that its own "no false positives" bullet is measured wrong: my first three
formulations produced false positives at a high rate, all in the debt-manufacturing direction.

**→ whoever owns `ddm_wr1`.** §7's "−295,663 B at PREDICTED-zero seg cost" and §4's "pose-favorable by
construction" are both **REFUTED at INSTANCE scope** by the gate's own realized row (§2). The rate lever is
real and exact; the archive still lost 0.153 S. Knee B loses 1.056 S.

**→ whoever owns the deferral ledger.** QA06 can be closed: both candidates fired, both REJECT, receipts
now exist and are machine-readable.

**→ any future arm tempted to build a follow-on registry.** Do not. The extraction is auto-derived from
memo text and the join from artifact existence; a hand-maintained list of orphans would go stale in
roughly a day, which is precisely the failure this arm measured.

---

## §7 WHAT THIS DID NOT DO

It did not lower any score, did not fire a scorer job, and did not take the n600 slot (MAIN owns it). It
did not adjudicate the 84 UNKNOWN rows. It did not validate ORPHANED precision on live data, because the
live ORPHANED set is empty. What it did is convert an assumed-rampant class into a measured-near-empty
one, name the real residual (stale never-run claims), pay the one genuine debt it found, and fix the
echo-only harvest step that created it.

---

# APPENDIX (#880) — the task-row join: 0 of 62 open rows detectably closed, at a MEASURED recall of 16%

Coordinator-directed extension, 2026-08-01. Same artifact predicate as §3, **conservative default
INVERTED**: on memos a false ORPHANED manufactures debt, but on tasks a false EXECUTED silently
DELETES real backlog — and a deleted row is not recoverable by reading harder. So `EXECUTED` must
be EARNED by a present RUN product; everything else is `UNKNOWN`. **There is deliberately no
ORPHANED bucket for tasks**: "never done" is a negative-existence claim that artifact-absence
cannot support.

## A.1 SCOPE BOUNDARY — the operator-visible backlog is not in this repo

**MEASURED:** task ids **833 · 834 · 840 · 841 · 844 · 858 · 859 · 860** (the coordinator's
never-named list) and **864 · 869 · 870** (the arms running today, including this one) are **ALL
ABSENT** from `.omx/state/canonical_task_status.jsonl`. That store holds 143 distinct ids — 37
numeric, range **383–828** — and no on-disk mirror of the harness task list exists anywhere under
`.omx/state` (checked every `.jsonl`; `graph_memory/nodes.jsonl` carries no task nodes).

So `audit_tasks` takes rows as **caller-supplied input**. Whoever holds the harness ledger feeds it;
the function never pretends to enumerate a population it cannot see, and its `ScopeLedger` states
which population it actually got. The coordinator's ~105-row population and mine are different sets.

## A.2 THE MEASUREMENT (canary lit in the same run)

| population | examined | EXECUTED | UNKNOWN |
|---|---:|---:|---:|
| **open** (pending + in_progress + blocked) | **62 of 62** | **0** | 62 |
| **completed** (ground truth known) | **81 of 81** | **13** | 68 |

**The completed-row control is what makes the open-row result readable.** It is a recall
measurement on rows whose closure is already known: **13/81 = 16.0% sensitivity.** Therefore
"0 of 62 open rows are already closed" is a **lower bound from a detector with 16% measured
sensitivity** — *not* a claim that no open row is already closed. Stating the first without the
second would be the negative-existence error the coordinator flagged.

**Only 2 of 62 open rows carry ANY artifact evidence at all**, so the task population is **~97%
undecidable by artifact join** — worse than the memo side's 84.3%. The two:

| row | status | verdict | evidence |
|---|---|---|---|
| `#826` | pending | UNKNOWN | `build-product-present: gr1_cell_drop50_archive.zip` |
| `#578` | blocked | UNKNOWN | `named-output-absent:` an absolute path under `.omx/tmp/codex_runs/…prompt.txt` — a **transient scratch location**, which CLAUDE.md forbids as durable evidence |

## A.3 THE ROUND-1 CATCH — right for the wrong reason

The first run returned exactly one `EXECUTED`: **#826**, on evidence *"`gr1_cell_drop50_archive.zip`
is present"*. **The verdict was correct** — §1 independently established via hand-check that #826's
evaluate receipt exists (S = 2.2027435921, 600 samples). **The evidence did not support it.** An
archive is the **INPUT** to a gate, so its presence witnesses **BUILT**, never **RAN**. This is the
runner-vs-output distinction from §3 reappearing one level up.

**Right-for-the-wrong-reason is the most dangerous state an instrument can hold, because it passes
review.** Build products (`.zip`, `.npz`) can no longer earn `EXECUTED`; they are still reported,
just not counted as closure. That change is what took the open-row EXECUTED count from 1 to 0 — and
the 1 was never real evidence.

## A.4 THE CANARY IS NOW STRUCTURAL

Per the coordinator's method warning (a misplaced `-E` returned 0-of-54 and was caught only by a
positive control; three sibling instruments failed the same way today and reached the operator
first), `audit_tasks` runs a positive **and** negative control **before emitting any row** and
**REFUSES** — empty result, `VACUOUS` scope, reason string — if the positive does not fire. A run
whose canary is dark cannot publish. This is P4 ("no meter without a canary") made structural
rather than procedural, and it is asserted by a dedicated test.

## A.5 THE ANSWER TO "HOW MUCH OF THIS BACKLOG IS REAL"

**Not answerable from artifacts, and the reason is a recordkeeping gap rather than a detector gap:
task rows do not name the artifact that would witness their own closure.** 68 of 81 rows *known*
to be completed name no run product that exists. The limit is on the producing side, exactly as
`ddm_gd5` §5 found for the rival relation — and the cure has the same shape: a `closing_artifact`
field written at the moment a task is closed, while the information is in hand and free, makes the
join mechanical. Absent that, both populations stay dominated by an honest, large `UNKNOWN`.

**Pointer UNMOVED**: `effective_frontier` 0.172 official; own-vehicle v4d 0.9476091.
`[macOS-CPU advisory]`, `score_claim=false`, zero scorer forwards.
