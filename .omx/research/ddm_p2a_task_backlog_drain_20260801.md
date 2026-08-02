# ddm_p2a (#880, population 2) — the 114-row harness backlog: three closure detectors measured, two dead, one false-ALREADY-CLOSED defect found and fixed on shipped code

**UTC 2026-08-01/02** · axis **[macOS-CPU advisory]** · `score_claim=false` · `promotion_eligible=false` ·
`pointer_moved=false` · `ready_for_exact_eval_dispatch=false` · **zero scorer forwards** · `upstream/` untouched.
**review_status:** pre-registered controls on every sweep · own-round-1-reviewed (three defects in my own
work, two of them in code I had already written and tested) · every ALREADY-CLOSED hand-verified against
primary artifacts.

**Pointer honesty, first.** Nothing here lowered any score. `effective_frontier` **0.172** (official
leaderboard) UNMOVED; own-vehicle line **v4d 0.9639878** UNMOVED. This is APPARATUS — MEANS, not END.

**STORES CONSULTED:** `ddm_fo1_orphaned_followon_detector_20260801` (its #880 APPENDIX is the direct
predecessor — I re-derived its two load-bearing numbers rather than inheriting them),
`src/tac/followon_ledger.py`, `src/tac/tests/test_followon_ledger.py`, `.omx/state/operator_p0_ledger.jsonl`,
`.omx/state/canonical_task_status.jsonl`, memories `vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801`,
`negative_existence_claims_are_the_days_dominant_error_class_20260731`,
`designed_stub_is_orphan_signal_and_a_no_fake_violation_20260731`,
`built_new_machinery_instead_of_paying_identified_debt_20260731`.
**Deliberately NOT loaded:** population 1 (the 86 UNKNOWN memo follow-ons — owned by `ddm_p1a`).

---

## §0 HEADLINE (answer first)

**The backlog cannot be drained by any automatic instrument, and I can now say that with three
independent measurements instead of one. What I did drain is small and exact: 2 rows are
hand-verified ALREADY-CLOSED, 2 are positively-evidenced REAL-OPEN, 18 have never been named by a
single commit in 13,736. And the shipped detector was returning "candidate ALREADY-CLOSED" on a
document certifying the run never happened — fixed.**

| # | statement | measured |
|---|---|---|
| 1 | **The shipped join had a false ALREADY-CLOSED.** Task #536 earned `EXECUTED` on `factor10_kkt_waterfill_blocked_receipt_20260718.json` — a receipt reading `"launch_performed": false`, `"measurement_axis": "no_scientific_measurement"`. The row's own first words are "MEASUREMENT STILL BLOCKED". | live run, now fixed + guarded |
| 2 | **Detector sensitivity on THIS population is ~2.4%, not 16%.** 18 of 762 known-completed rows are detectable. fo1's 16% was measured on a different, smaller population and does not transfer. | 18/762 |
| 3 | **Two further closure detectors are DEAD, measured.** Text closure-markers: every marker's lift collapses to ≤1 once era-controlled. Self-supersession markers: 17 of 18 hits are false. | see §3 |
| 4 | **The one instrument that works is never-named-in-commits**, over a fully enumerated 13,736-commit corpus with passing controls: **18 of 114** open rows have never been named. | §4 |

---

## §1 SCOPE — how I got a population at all, and its exact boundary

The harness task ledger has no on-disk mirror (fo1 §A.1, re-verified: ids 833/834/840/841/844/858/859/860
are all absent from `.omx/state/canonical_task_status.jsonl`), and I do not hold `TaskList`. So I
**reconstructed the ledger by replaying the session transcript**
(`~/.claude/projects/-Users-adpena-Projects-pact/89ff112f-….jsonl`, 3.0 GB): **882 `TaskCreate` + 1,607
`TaskUpdate` + 7 `TaskGet` + 5 `TaskList`** events, replayed create-then-update in timestamp order.

| reconstruction check | result |
|---|---|
| tasks recovered | **880** |
| id range | 1 … 880, **zero gaps** |
| `TaskCreate` results that did not parse | 2 |
| status split | 762 completed · 69 pending · 45 in_progress · 4 deleted |
| **open population (this arm's scope)** | **114** |

**BOUNDARY, stated so it is not over-read.** This is a *replay*, not the live ledger. It is complete only
up to the last event in the transcript (`2026-08-02T02:48:49Z`) and only for events in *this* session. Two
gaps-of-principle: 2 unparsed creates, and any mutation made outside this transcript is invisible. The
coordinator's "~105 open" and my 114 differ by exactly the kind of drift this implies — I report 114 with
its timestamp rather than reconciling to a number I cannot see. Ids are chronological, so the reconstruction
is falsifiable in one step: any missing id would show as a gap, and there are none.

---

## §2 THE DEFECT — a refusal receipt earning "candidate ALREADY-CLOSED" (fixed)

`classify_task_execution` already split BUILD products (`.zip`, `.npz` — an input a gate consumes) from RUN
products (a receipt/report/log an execution writes). **A refusal receipt defeats that split**: a governed
harness refusing pre-write *does* write a receipt, so it is a RUN product by shape — while certifying the
opposite of execution. It is the one artifact class whose **presence is evidence the row is OPEN**.

Measured instance, on shipped code, live population:

```
#536  "Measure the 3-axis JOINT rate-distortion optimum …"      status: in_progress
      evidence: .omx/research/factor10_kkt_waterfill_blocked_receipt_20260718.json   → EXECUTED
      receipt contents: "measurement_axis": "no_scientific_measurement"
                        "launch_performed": false
      row's own text:   "MEASUREMENT STILL BLOCKED (honest, 2026-07-18)"
```

This is the direction the module's own docstring forbids — *"a false EXECUTED silently DELETES real
backlog"* — and it survived a dedicated test asserting the join never labels live debt EXECUTED. **25
artifacts under `.omx/research` alone carry a blocked/refused marker**, so the class is live.

**Fix (landed):** refusal receipts are stripped before anything can earn EXECUTED, and the row returns
UNKNOWN with a reason stating its presence is evidence of OPEN. Verified live: #536 flipped
`EXECUTED → UNKNOWN`, open-population EXECUTED went 6 → 5, canary lit both runs, denominator 114/114.
Guarded by 4 tests bound to the real instance (40 pass; `-k refusal` positive-control confirms mine ran).

**The inversion worth keeping:** artifact *absence* can never prove a row is open (negative existence).
A refusal receipt is the only artifact I found whose *presence* proves it. That is a positive-evidence
channel for OPEN, and it is the cheap half of the cure in §6.

---

## §3 TWO CLOSURE DETECTORS I BUILT AND MEASURED DEAD

Before concluding "artifacts cannot settle this", I tested the two cheapest alternatives. Both fail.

**(a) Closure markers in the row text.** Raw lift looked promising — `VERDICT:` appeared 3.74× more often
in completed rows than open. **It is entirely an era artifact.** Open rows are recent (`id` median 606) and
completed rows old (median 417); description style changed. Controlling by id band:

| marker | raw lift | lift, ids ≥ 700 | lift, ids ≥ 800 |
|---|---:|---:|---:|
| `VERDICT:` | 3.74 | 0.76 | 0.00 |
| `MEASURED` | 0.43 | 0.30 | 0.53 |
| `owed` | 0.46 | 0.60 | 0.19 |
| pending-language | 0.27 | 0.23 | 0.53 |

Era-controlled, **no marker exceeds 1.0**. The ledger text does not predict closure. (The inverted raw
lift on `MEASURED` was the tell that made me control for era at all.)

**(b) Self-supersession markers.** A regex for `SUPERSEDE|RE-DISPOSED|REFUTED|DEAD|CLOSED` fires on 18 of
114 rows. Hand-reading the match context: **17 are ambient** — the row is superseding something *else*
(#539 "supersedes coord-INR"), or citing another arm's refutation (#870), or describing superseded *code*
(#861). Exactly one (#862) genuinely records its own supersession, and even that one was *reframed*, not
closed. Same failure genus as bare `$0` in fo1 §3: campaign vocabulary, not a class marker.

**Why this matters more than a null result:** it forecloses the two cheapest things a successor would try,
and it independently confirms fo1's diagnosis from a different direction. The limit is on the **producing**
side. Nothing in a task row reports its own closure, in artifacts *or* in prose.

---

## §4 THE ONE WORKING SIGNAL — never-named, over the full history

**Instrument check first, because it failed.** `git log --oneline | wc -l` returned **50**; the identical
pipeline inside a `&&` chain returned **13,736** (= `git rev-list --count HEAD`). Reproduced 3×. The `rtk`
token-proxy hook rewrites the bare form and caps it. **Any git-derived count in this repo taken from a bare
`git log` is truncated at 50 and looks like a real number.** I ran git from Python (no shell hook) and
asserted `len(commits) == 13736` before using anything.

Controls: `#826` → 6 mentions, `#870` → 3, `#9999` (cannot exist) → 0. Ids extracted as exact integers via
`#(\d{1,4})\b`, never substring — per fo1's `#829` collision lesson.

**Result: 99 of 114 open rows named at least once; 15 never named.** Plus 3 rows (#198, #236, #375) whose
*every* mention sits in `Catalog #N` / `PR #N` context, i.e. not this row at all → **18 never-named**.

All 13 of the coordinator's list are **confirmed over the full 13,736 commits** (their 400-commit window
was sound for these). It missed 2 (**#775, #877**) and could not disambiguate 3 (**#198, #236, #375**).

| # | status | upd | created | subject |
|---|---|---:|---|---|
| 198 | pending | 0 | 06-30 | Canonical fleet-config loader + preflight self-protect |
| 236 | pending | 0 | 07-02 | Consolidate to ONE dashboard web app → named tunnel |
| **375** | pending | 0 | 07-09 | Auto-push Stop hook — **hand-verified ALREADY-CLOSED, see §5** |
| 450 | pending | 0 | 07-12 | Lens Engine — multi-lens analyzer over the double one-object |
| 556 | pending | 0 | 07-19 | FilmPolarSPDNormalMomentum (Muon follow-on from #552) |
| 670 | pending | 0 | 07-24 | Warn-only-purgatory + registry-debt cluster |
| 706 | pending | 0 | 07-25 | Post-J8F queued wave — 4 arms |
| 716 | pending | 0 | 07-25 | S-primacy + derived-exchange-rate |
| 775 | in_progress | 4 | 07-29 | QA43 tail-targeted heteroscedastic (behind sb1) |
| 833 | pending | 0 | 07-31 | Degenerate-baseline control for capacity/floor probes |
| 834 | pending | 1 | 07-31 | reclaim↔spend asymmetry — PREMISE CONTESTED |
| 840 | pending | 0 | 08-01 | ddm_cf1 extension — the unswept 91% |
| **841** | pending | 0 | 08-01 | wr1 ceiling column — **hand-verified ALREADY-CLOSED, see §5** |
| 844 | pending | 0 | 08-01 | Triality DSL-drift detector syntactic predicate |
| 858 | pending | 0 | 08-01 | Receiver admits an ABSENT token_codec |
| 859 | pending | 0 | 08-01 | SMEVR base-rule race −2,781 B, blocked on a receiver |
| 860 | pending | 0 | 08-01 | 6 of 25 dev gates RED, ~316 violations |
| 877 | pending | 0 | 08-02 | REPORT censoring — evaluate.py 2-decimal consumers |

**Read this correctly.** Never-named is a *hard negative-existence fact over an enumerated corpus* — it says
no commit message mentions the row. It is **not** proof the work is undone (#375 and #841 are both
never-named and both closed). It is the cheapest available prior on "nothing has visibly touched this."

---

## §5 THE ADJUDICATION — every candidate ALREADY-CLOSED hand-verified

The join's 6 EXECUTED were the only automatable candidates, and a false one here **deletes real backlog**,
so each was checked against primary artifacts, never against the row's self-description.

| # | verdict | evidence (hand-verified) | cost-to-falsify |
|---|---|---|---|
| **375** | **ALREADY-CLOSED** | `tools/auto_push_main.py` exists (24 KB); `auto_push` **is** present in `.claude/settings.json` Stop array. *The join's stated evidence was wrong* — settings.json merely EXISTING witnesses nothing; its CONTENT does. Right verdict, wrong reason, again. | **~0** (1 file read) |
| **841** | **ALREADY-CLOSED** | The ceiling-reversal finding IS landed: `ddm_wi1_wrong_instrument_sweep_20260731.md:109` states it with the exact numbers (2.0058 → 2.0108 → 2.1186 → 2.2755). The named receipt **predates the row by 3 days** — the row is a finding *derived from* it, never a task producing it. | **~0** (1 grep, controlled) |
| **536** | **REAL-OPEN** (positively evidenced) | Its artifact is a refusal receipt certifying non-execution (§2). Rare and valuable: positive evidence of open. | **~0** |
| **873** | **REAL-OPEN** | `final_pw1.jsonl` is **prior** work the row builds ON, cited as context. Created 08-02, 0 updates. | **~0** |
| **862** | **SUPERSEDED** (original framing) | Row subject reads "RE-DISPOSED by ddm_rg5"; its original claim (rate gradient anti-correlated) is refuted and the row rewritten to a successor hypothesis. The *original* is drainable; the row is not. | low (read row) |
| **824** | **UNDECIDABLE** | Token `/tr1_window_receipt.json` is a partial-path capture matching on basename. Row's own text says the diagonal IS harvested, but its scope explicitly extends past it (off-diagonal owed). | medium |

**Corrected count: 2 ALREADY-CLOSED of 114 (1.8%).** The remaining 108 are UNDECIDABLE — **104 of 114
(91%) carry no artifact token at all**, so there is nothing for any join to check.

**Ranked by cost-to-falsify, not by predicted ΔS** (per `gc17`: all six #1-ranked convocation levers were
refuted by the measurement they ordered, bookings ~100× optimistic):

1. **T0, ~free (6 rows):** #375, #841 → drain now. #536, #873 → confirm open. #862 → close original framing. #824 → read one receipt.
2. **T1, one controlled grep each (16 rows):** the remaining never-named. Cheapest per bit of the whole backlog.
3. **T2, a human read (~92 rows):** no automatic signal exists; §3 forecloses the cheap shortcuts.
4. **T3, needs a measurement:** unknown subset of T2, not separable without doing T2 first.

**Confirmed for the coordinator:** #578's named output *is* an absolute path under
`.omx/tmp/codex_runs/…prompt.txt` — transient scratch, which CLAUDE.md forbids as durable evidence. The
join reports it `named-output-absent`, i.e. the path does not resolve. Two rows (#820, #821) and #878 name
`.omx/state/` outputs that are likewise absent.

---

## §6 THE CURE — and why I did not build a registry

fo1 named it and I re-derived it independently from a different population: **68 of 81 there, 744 of 762
here — rows *known* completed name no run product that exists.** The producing side is the limit.

**Recommended, and it needs no schema change and no new surface:** a one-line convention in the
`TaskUpdate` that closes a row — `CLOSING-ARTIFACT: <path>`. The description field is already free text and
already the thing being written at closure, when the information is in hand and free. The existing
`classify_task_execution` join would then run at ~100% instead of 2.4%, with zero new code. Cost: one line,
once, by whoever is already typing the closing update.

**Second half, already landed:** refusal receipts as a positive-evidence-of-OPEN channel (§2). Artifact
absence can never prove a row open; this is the only artifact class that can.

**Explicitly NOT built:** a follow-on/orphan registry. fo1's closing warning applies verbatim — a
hand-maintained list goes stale in about a day. §3 adds the second reason: there is nothing to auto-populate
it from.

---

## §7 MY OWN ROUND-1 REVIEW — three defects, two in code I had already written and tested

1. **`no_go` in the refusal marker set destroyed true positives.** Swept over the live 76,449-artifact
   corpus, it matched `go_no_go_verdict.json` and `failure_terminal_n600_no_go_*.json` — RUN products of
   measurements that EXECUTED and returned NO-GO. At 2.4% sensitivity I cannot afford to lose true
   positives. **Removed after measuring**; a test now pins it. The marker set is measured, not enumerated.
2. **My content probe was INERT in production — a designed stub that passed its own test.** I wrote a
   `"launch_performed": false` content check; it fired 0 times over the whole corpus. Cause:
   `_load_index_cache` reconstructs with `produced_paths={}`, so on every *cached* load — the normal path
   for every consumer — it has zero files to read. My unit test passed only because it hand-built a corpus
   *with* paths. **Removed, not left in.** Making it live costs ~6 MB of paths in the hot SessionStart
   cache for an unmeasured payoff, so it is DEFERRED and named in the source, not faked.
3. **My own `grep --include=*.md` positive control silently failed** — zsh expanded the glob, the command
   errored, and the empty output read as "no matches found." I nearly recorded a false negative-existence
   claim about #841. Caught only because the control printed nothing *at all* rather than nothing-in-scope.

Defects 1–2 are the same shape as the bug I was fixing: an instrument that looks right and does nothing on
real inputs. Defect 3 is the vacuity genus in my own hands, one hour after writing about it.

**Also found, environmental and load-bearing beyond this arm:** `git log --oneline | wc -l` returns **50**
instead of 13,736 under the `rtk` hook (§4), and it is silent about it.

---

## §8 CROSS-FINDINGS

**→ the #880 / `p0_864` owner.** Population 2 is drained as far as evidence permits: 2 rows closeable now
(#375, #841), 2 confirmed open (#536, #873), 1 superseded (#862), 18 never-named, 108 undecidable. The
appendix figure "0 of 62 EXECUTED at 16% recall" does **not** transfer to the harness population; the
matching numbers here are **2 of 114 at 2.4% recall**. Please carry the recall with the count.

**→ whoever runs any git-derived sweep.** Bare `git log --oneline | wc -l` is capped at 50 by the `rtk`
hook and looks like a real number. Use `git rev-list --count HEAD` as the assertion, or run git without the
shell hook. This plausibly affected the 400-commit sampling window that produced the seed list.

**→ `ddm_p1a`.** §3's era-controlled null is a *transferable* negative: if you are considering a
text-marker closure detector on the memo population, control for era first — the raw lift is a mirage.

**→ whoever owns #862.** Its original framing ("rate gradient points the WRONG way") is refuted by `ddm_rg5`
and the row already says so. It can be closed as SUPERSEDED and the successor hypothesis re-opened as a new
row, so the backlog stops carrying a dead claim under a live id.

---

## §9 WHAT THIS DID NOT DO

It did not lower any score, fire a scorer job, or take an n600 slot. It did not adjudicate the 108
UNDECIDABLE rows — no instrument can, and §3 shows the two cheap ones do not work. It did not touch
population 1. It did not implement `CLOSING-ARTIFACT`, which needs a producer-side decision, not another
detector. Its own reconstruction is a transcript replay with a stated timestamp boundary, not the live
ledger.

**Pointer UNMOVED**: `effective_frontier` 0.172 official; own-vehicle v4d 0.9639878.
`[macOS-CPU advisory]`, `score_claim=false`, zero scorer forwards.
