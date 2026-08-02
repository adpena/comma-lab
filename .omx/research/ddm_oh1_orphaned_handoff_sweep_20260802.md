---
schema: ddm_oh1_handoff_graph.v1
date_utc: 2026-08-02
arm: ddm_oh1 (the orphaned-handoff sweep — code-derivable)
lane_id: "lane_ddm_oh1_orphaned_handoff_sweep_20260802"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
ready_for_exact_eval_dispatch: false
axis: "[macOS-CPU advisory; $0 read-only join over memo text + git history; zero scorer forwards; upstream/ untouched]"
inherits: "ddm_fo1 (#870) tac.followon_ledger · ddm_p1a (#879) the 86 UNKNOWN · ddm_p2a (#880) the task-row join"
verdict_scope: "INSTANCE (this extraction window since=2026-07-01, this commit window, this artifact scope)"
empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
review_status: "pre-registered controls · own-round-1-reviewed (four defects in my own code, three of which produced confident FALSE ORPHANED, one an overclaim in a design comment caught by a test I wrote to pin it)"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_oh1 — the handoff edge: an arm names a successor, and nothing joins that to what was done

**Pointer honesty, first.** Nothing here lowered any score. `effective_frontier` **0.172** (official
leaderboard) UNMOVED; our own-vehicle line **v4d 0.9639878** UNMOVED. This is APPARATUS — MEANS, not
END.

**STORES CONSULTED:** `tools/corpus_query.py` (research/equations/memory/dag/council/tasks/docs) →
loaded `ddm_fo1_orphaned_followon_detector_20260801`, `ddm_p1a_followon_unknown_adjudication_20260801`,
`ddm_p2a_task_backlog_drain_20260801`, `ddm_gd5_grade5_detector_is_not_autoderivable_20260801`,
`src/tac/followon_ledger.py`, `src/tac/tests/test_followon_ledger.py`, `src/tac/scope_ledger.py`,
`.omx/state/canonical_task_status.jsonl`, and the two live instance memos
(`ddm_sv2_survival_engineering_and_the_rebase_20260802`,
`ddm_os1_optimization_sweep_termination_census_20260802`). Memories: `built_elsewhere_unwired_is_p0_20260801`,
`vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801`,
`negative_existence_claims_are_the_days_dominant_error_class_20260731`,
`built_new_machinery_instead_of_paying_identified_debt_20260731`.
**Deliberately NOT loaded:** the harness task-ledger transcript replay (owned by `ddm_p2a`; I consumed
its *conclusion* that no on-disk mirror exists and RE-DERIVED that one fact against
`canonical_task_status.jsonl` rather than inheriting it).

---

## §0 HEADLINE (answer first)

**The class is real and code-derivable, the instrument now exists and ships with controls — and the
two instances the dispatch named as orphaned are NOT orphaned. They are 0 days old.** Both were
committed on 2026-08-02, inside the 24–72 h drain window `ddm_fo1` measured. The instrument returns
`LIVE` for both, which is the honest answer, and it is the same failure mode `ddm_fo1` found in its
own first run: *the debt is asserted faster than it is checked.*

| # | statement | evidence |
|---|---|---|
| 1 | **Both named instances are invisible to the shipped extractor** — neither line matches `ACTION_RX ∧ CHEAP_RX`. The class was genuinely undetectable, not merely unmeasured. | §1 |
| 2 | **One of them is invisible for a STRUCTURAL reason no predicate could fix**: `ddm_sv2:350` puts the target `#539` on line 352 and the verb "hand off" on line 353. `extract_followons` iterates `splitlines()`, so no single line ever carries both halves. | §1 |
| 3 | **Both are `LIVE`, not `ORPHANED`** — named 0 days ago. The dispatch's premise is over-stated in the same direction `ddm_fo1` corrected. | §2 |
| 4 | **9 ORPHANED rows** in the 2026-07-01→ window, of 348 handoff edges over 2,155 in-scope memos (population 7,388). **Two are hand-verified true orphans against primary artifacts.** | §3, §4 |
| 5 | **I produced 29 ORPHANED first, and 20 of them were mine.** Three false-ORPHANED classes, all measured on real corpus text, all fixed. | §5 |
| 6 | **A silent 100%-vacuity on shipped code, found and fixed**: the live harness emits `TaskCreate` with `subject`/`description`, neither of which `_TASK_TEXT_FIELDS` reads. Every native row returned "no text to join on" — with a full `examined` count. | §6 |

---

## §1 WHY THIS EDGE WAS INVISIBLE

The shipped join in `tac.followon_ledger` asks *"did this follow-on produce its OUTPUT?"* and is keyed
to a filename the row itself names. **A handoff names no output.** It names a SUCCESSOR — a task id, a
sister arm, a code site — and says the work goes there. So the artifact-existence predicate has
nothing to look for, and the row is never even extracted.

Both live instances, MEASURED at source:

| instance | text | why the shipped extractor misses it |
|---|---|---|
| **(a)** `ddm_sv2_…20260802.md:350` | *"`#539` (Build the POWER-DIAGRAM witness parametrization, in_progress) is the natural donor; this arm should hand off to it rather than fork a parallel surface."* | **SCOPING.** Target on line 352, verb on line 353. No single line carries both halves — **no predicate whatsoever can see this row one line at a time.** |
| **(b)** `ddm_os1_…20260802.md:203` | *"No cure landed at pfs1 … the fix … is STAGED, not taken."* | **PREDICATE only.** It fits on one line; `ACTION_RX ∧ CHEAP_RX` simply matches neither half. |

That distinction is not cosmetic and I state it precisely because **an earlier draft of my own design
comment claimed both were scoping failures, and the test I wrote to pin the claim failed.** The
scoping half is plausibly a component of `ddm_p1a`'s finding that all 86 UNKNOWN follow-on rows
carried the single reason "no artifact-shaped join token" — a token stranded on the neighbouring line
reads exactly like a token that is absent. That remains **INFERRED**: I demonstrated the mechanism on
one row; I did not re-measure p1a's population.

**`extract_followons` is deliberately NOT switched to item scoping.** Its line-scoped population is
quoted by two landed memos. Changing it would move their numbers underneath them — a supersession
decision with an owner, so it is named here rather than taken unilaterally.

---

## §2 THE JOIN, AND THE ONE DIRECTION IT IS ALLOWED TO BE WRONG IN

Four verdicts. The naming is the argument:

* **`ADVANCED`** — the named successor shows activity after the handoff was written. **Deliberately
  not `DONE`.** A commit touching the named site proves the successor MOVED; it does not prove it did
  the named work. Per the seed's own bar — *"a commit mentions the number is NOT closure"* — the
  channel is a commit that **TOUCHES the named path**, which is an artifact-level fact, not a naming
  one. It is still an upper bound on closure and is labelled as one.
* **`LIVE`** — younger than `LIVE_WINDOW_DAYS = 3`. **DERIVED**, anchored to
  `ddm_fo1_…20260801.md:26` (*"sister arms drain follow-ons within 24–72 h"*), taking the window's
  UPPER end so the constant can only move rows OUT of ORPHANED, never into it. The recess measurement
  that would sharpen it — the real distribution of (naming date → first successor touch) over this
  instrument's own ADVANCED rows — is named in code so the constant is not mistaken for settled.
* **`ORPHANED`** — past the window, and no successor activity for any **decidable** target.
* **`UNVERIFIABLE`** — every channel gap lands here: dead git, untracked path, bare task id with no
  ledger, row predating the scan window, memo with no date.

**The instrument is allowed to be wrong by saying UNVERIFIABLE and is not allowed to be wrong by
saying ORPHANED**, because a false ORPHANED manufactures debt out of a gap in the instrument. That is
exactly what `ddm_fo1` measured in its own first run (5-of-6 stale) and what I reproduced in §5.

**The TASK channel is honestly dead.** RE-DERIVED, not inherited: `.omx/state/canonical_task_status.jsonl`
holds **398 rows / 143 distinct ids**, and **`#539` is not among them**. There is no artifact-level
closure signal for a bare task id, so task targets are `UNVERIFIABLE` unless a caller supplies
`closed_task_ids` — the same caller-supplies-the-ledger discipline `audit_tasks` already uses.

---

## §3 THE MEASUREMENT, WITH ITS DENOMINATOR

Window `since=2026-07-01`; commit index bounded to the same window; `today=2026-08-02`. Canary passed
(fires on a verified-touched path, silent on a path that cannot exist); `audit_handoffs` refuses to
return rows otherwise.

| scope | count |
|---|---|
| memo population | **7,388** |
| in scope after `since` filter (= examined) | **2,155** |
| handoff edges extracted | **348** |
| — by stratum | HEADING 295 · FRAME 53 |
| — by target kind | TASK 331 · ARM 169 · PATH 64 |
| commit index | 18,949 paths touched · 39,648 tracked |

| verdict | n |
|---|---|
| **ORPHANED** | **9** |
| LIVE | 26 |
| ADVANCED | 128 |
| UNVERIFIABLE | 185 |

**All 185 UNVERIFIABLE carry one reason: bare task ids with no ledger.** That is the single highest-value
unlock for this instrument and it is not a code problem — it is the missing task ledger `ddm_p2a`
reconstructed and did not persist.

---

## §4 THE RANKED ORPHANED LIST, WITH COST-TO-FALSIFY

Ranked by strength of evidence. **Rows 1–2 are hand-verified true orphans against primary artifacts.**
Rows 3–6 point at real owed items but extracted a wrong or weak target. Rows 7–9 are false positives I
report rather than hide, because the precision figure is the reader's calibration.

| # | producer | target | verdict basis | cost to falsify |
|---|---|---|---|---|
| **1** | `custom_sparse_adjoint_metal_wall_MEASURED_20260714.md:121` — *"**eq (OWED):** append the MEASURED achieved/ceiling anchor (`achieved=0.7078x`, `η=0.3205`, `2.2086x`) … NOT touched here"* | `src/tac/canonical_equations/custom_sparse_adjoint_achieved_ceiling_20260713.py` | **HAND-VERIFIED TRUE ORPHAN.** File exists; last commit **2026-07-13**, one day BEFORE the memo that owes the append; the three anchor values appear **0 times** in it. 19 days open. | 1 grep (`grep -cE "0\.7078\|0\.3205\|2\.2086" <file>`) — falsified if > 0 |
| **2** | `fcntl_lock_canonicalization_plan_20260710.md:402` — *"Batch 5 — DEFERRED, own design pass … New sibling helper `write_json_atomic_locked`"* | `#128`, `#131`, `tools/extract_master_gradient_mlx.py` | **HAND-VERIFIED TRUE ORPHAN.** `write_json_atomic_locked` appears in **0** `.py` files under `src/`+`tools/` (positive+negative control run on the grep). 23 days open. | 1 controlled grep — falsified if the helper exists under any name |
| 3 | `codex_findings_g111_macro_release_path_20260727:96` — "Exact owed execution sequence" | ARM `v6` | Real owed sequence; but G121 HAS commits (`ead282f6f0`), so the *sequence* moved — the arm-slug target `v6` was the wrong key. **Likely ADVANCED in truth.** | read `ead282f6f0`; falsified if it closes the G121/G119 step |
| 4 | `gpu_verdict_hybrid_20260708.md:71` — "Measurement 2 … (DEFERRED, governor …)" | `tools/safe_run.py` | Real DEFERRED measurement, but the extracted path is the tool that *refused*, not the successor. Right row, wrong target. | re-run the n600 GPU-vs-CPU verdict; falsified by any receipt for it |
| 5 | `owed16_bounded_ab_and_drystart_20260710.md:141` — "item 1 — owed-16 A/B" | `tools/safe_run.py` | Same shape as #4: a real owed A/B, target extracted from a command block. | check for an owed-16 A/B receipt |
| 6 | `pantheon_all_time_roster…:155` — "adjudication owed to the council" | ARM `g4` | Real owed adjudication; `g4` is a weak target. | council row for ORDER 2c |
| 7 | `DRAFT_derived_optimal_next_run_for_council_20260707:851` | `experiments/launch_split_by_head_basin.py` | **FALSE POSITIVE** — the path is cited as a LOCATION in a verification statement, not owed. | — |
| 8 | `codex_findings_g102_final_y1…:170` | `src/tac/witness_dsl/v10_production_receiver.py` | **FALSE POSITIVE** — a SHA-256 custody line under a heading whose own words are "already present versus owed". | — |
| 9 | `position_S3_control_20260709.md:25` | `tools/witness_control_monitor.py` | **FALSE POSITIVE** — a design position under "ANSWER FIRST". | — |

**Measured precision on the ORPHANED bucket: 2/9 hand-verified true orphans (22%), 6/9 point at a real
owed item (67%), 3/9 false (33%).** Reported rather than tuned away: tuning the predicate until the
9 rows look clean would be agreeing with the test.

---

## §5 THE THREE FALSE-ORPHANED CLASSES I SHIPPED AND THEN FOUND (own round-1 review)

The first live run returned **29 ORPHANED**. Hand-inspecting all 29 found **20 were artifacts of my own
instrument.** All three causes are now fixed and pinned by tests; **29 → 9.**

1. **`handoff` is ambient CUSTODY vocabulary in this corpus, not an owed-work marker.** Enumerating
   every `handoff` heading shows "Serializer and exact-hash handoff", "Verification, triality, and
   handoff", "Triality handoff" ×3, "Self-review and MAIN handoff" ×2, "Law 5 —
   `label_floor_to_phase_tail_handoff_v1`", "handoff_readiness.part_frac". Their content is SHA-256
   custody for files that **landed** — the exact opposite of owed. Removed from `OWED_HEADING_RX`;
   retained in `HANDOFF_FRAME_RX` where it appears as a verb.
2. **Markdown DATA TABLES read as debt.** `| n64 | 61,087 | -0.001589 | … |` under "Scale ladder and c1
   handoff" was reported ORPHANED. `n64` and `c1` are *simultaneously* real arm slugs (`ddm_n64_*` and
   `ddm_c1_*` both exist, 12 and 4 artifacts) and ordinary scale/lane words — the `#829` collision
   class arriving through a channel the arm-corpus membership test **cannot** filter, because the
   tokens really are arms. Table rows are not dropped (the deferral ledger's owed queue IS a table);
   they are held to the stricter bar of an explicit frame in the row itself.
3. **Untracked paths are invisible to a commit-touch channel BY CONSTRUCTION.**
   `margin_field_head_levers…:77` hands off a wire-in to
   `experiments/train_LEVELSET_witness_realized_through_R_mlx.py`, and the memo *itself* says the file
   is "UNTRACKED / git-ignored". `git ls-files --error-unmatch` confirms it. Such a path is untouched
   forever no matter how much work goes into it. Now `UNVERIFIABLE` — the vacuity genus again: "no
   commit touched it" and "commits cannot touch it" are different facts wearing one symbol.

Plus a fourth, in prose rather than code: **the design comment overclaimed that both instances were
scoping failures.** The test written to pin that claim failed, and §1 now states the two mechanisms
separately.

---

## §6 A SILENT 100% VACUITY ON SHIPPED CODE, FIXED

**MEASURED, by reading a raw event out of the live session transcript:** the harness emits
`TaskCreate` with `{"subject", "description", "activeForm"}`. **None** of those keys is in
`_TASK_TEXT_FIELDS = ("title", "event_notes", "source_design_memo", "next_action", "evidence")`.

So a caller handing `classify_task_execution` its native rows got `task_row_text() == ""` on every
one, hence `UNKNOWN / "carries no text to join on" / joinable=False` for **100% of the population,
silently.** It does not crash and does not warn — and the `ScopeLedger` still reports a full
`examined` count, because the rows *were* walked. A reader sees "N examined, all UNKNOWN" and
concludes the population is undecidable, when the instrument never read a character of it.

Fixed as an ALIAS TABLE, not a rename: both key-spaces are live (`ddm_p2a`'s replayed rows use
`title`), and per design philosophy P1 the alias table is the ONE place that knowledge lives, so
callers do not each have to remember to map their own fields.

---

## §7 WHAT I DID NOT DO, AND WHY

* **No task-ledger reconstruction.** `ddm_p2a` replayed the 3.0 GB transcript and owns that surface;
  re-doing it here would be building new machinery instead of paying the identified debt. It is the
  single highest-value unlock (185 of 348 rows), and it needs `ddm_p2a`'s reconstruction PERSISTED,
  which is a producer-side decision.
* **No full-history scan.** The window is `since=2026-07-01`, stated as a bound, and rows predating it
  return `UNVERIFIABLE` rather than a scan artifact. A full scan also needs a perf fix first:
  `SuccessorIndex.last_touch_matching` is O(indexed paths) per arm target — fine at 18,949 paths,
  not at the ~200,000 of full history. Named, not built.
* **No change to `extract_followons`.** See §1.
* **No fix to the preflight hook**, which timed out at 30 s and then produced no output in 120 s
  standalone. That is another arm's surface and is reported, not patched.
* **No scorer forwards.** The n600 slot is occupied (pid 18732); `experiments/ddm_v4c_resolve.py`
  untouched.

## §8 FALSIFIERS

* If `custom_sparse_adjoint_achieved_ceiling_20260713.py` gains the `0.7078`/`0.3205`/`2.2086` anchor,
  row 1 is closed and my strongest true positive is gone.
* If `write_json_atomic_locked` exists anywhere I did not scan (`src/`, `tools/`, `*.py`), row 2 is a
  false ORPHANED and my negative-existence claim is over-scoped.
* If a successor drains a `LIVE` row inside 3 days, the `LIVE_WINDOW_DAYS = 3` anchor is confirmed; if
  the real drain distribution has a long tail, the constant is too tight and rows are being called
  ORPHANED early.
* If the ADVANCED bucket contains rows whose successor activity is unrelated to the named work — which
  it certainly does, since `ADVANCED` is an upper bound — then 128 overstates real progress, and the
  ORPHANED count of 9 is a LOWER bound on the debt.

## §9 SCOPE OF THE NEGATIVE

`verdict_scope: **INSTANCE**`. "9 ORPHANED" is scoped to this extraction window, this commit window,
and this artifact scope. It is **not** a claim that the campaign has only 9 orphaned handoffs: 185
rows are undecidable for want of a task ledger, the window starts 2026-07-01, and the FRAME stratum's
measured precision is low. Nothing here supports a FAMILY-level statement about handoff orphaning.

---

**Landed:** `src/tac/followon_ledger.py` (handoff graph + field-alias fix) ·
`src/tac/tests/test_followon_handoff_graph.py` (40 tests, all passing).

**CLOSING-ARTIFACT: .omx/research/ddm_oh1_orphaned_handoff_sweep_20260802.md**
