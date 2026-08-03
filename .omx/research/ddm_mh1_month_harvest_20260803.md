# ddm_mh1 — MONTH-SCALE no-signal-loss / no-orphan harvest (2026-07-04 → 2026-08-03)

**Arm:** `ddm_mh1` · **Date:** 2026-08-03 · **Cost:** $0 (no scorer runs) · **Axis:** none (audit; no score claim)

Operator directive 2026-08-03, verbatim: *"You might need to look and harvest an[d ensure] no signal loss
or orphaned over the course of the past month, honestly."*

**Headline (the unflattering answer the "honestly" asked for):** the month's signal loss is **not**
mostly abandoned research. It is **status-rot in the tracking layer plus a structurally broken join**.
Of 24 silently-parked window-1 ledger rows, **14 are rot** (the work continued under other names; the
row was never closed) and **10 are true orphans**. And the repo-visible task ledger covers **4.8%** of
the primary window's memos, of which only **9.8%** are reachable by the `task #N` convention arms use.
A naive count would have reported "24 orphans" — that would have been a false claim, and §6 records it
as the primary thing this arm refutes.

---

## §1 — Denominators, and what was OUT of scope

Every scope below reports its denominator. An empty cell reported empty is a result; reported as
neutral is a failure (per the vacuity law).

### Instrument verification (done BEFORE any counting)

Two counting instruments were tested against known answers and **both failed**:

| Instrument | Reported | True | Verdict |
|---|---:|---:|---|
| `git log --oneline \| wc -l` (window) | 50 | 4,309 | **86× silent undercount** — pipe truncation. NEVER use. |
| `git log --since=A --until=B --name-only` | subset 2,011 > superset 1,087 | — | **Internally contradictory** — `--since` truncates graph traversal at merge points. NEVER use for counting. |

The second failure is the more dangerous one: the dating instrument **passed a single known-answer
test** (it correctly dated two known files) and only failed the *internal-consistency* test
(a subset window reported larger than its superset). A one-point verification is not verification.

**Method actually used:** one full-history pass
(`git log --diff-filter=A --format=... --name-only -- .omx/research/`), earliest-add per path,
bucketed in Python. Result reconciles: July total 2,165 ≈ window1 1,052 + remainder.

### Denominators

| Scope | Count | Method |
|---|---:|---|
| Commits, 07-04 → 08-04 | **4,309** | `git rev-list --count` |
| Commits, 07-04 → 07-19 (primary) | **2,414** | `git rev-list --count` |
| Research `.md` ever added under `.omx/research/` | **7,446** | full-pass, distinct paths |
| Research `.md` added **07-04 → 07-19 (PRIMARY WINDOW)** | **1,052** | full-pass, earliest-add |
| Research `.md` added 07-20 → 08-03 | **1,158** | full-pass, earliest-add |
| Untracked `.omx/research/*.md` (signal at risk) | **3** | `git ls-files --others` |
| Repo task ledger rows / distinct ids | **423 / 149** | `canonical_task_status.jsonl` |

### OUT of scope, and why

- **07-20 → 08-03 research-finding sweep** — owned by live sister arm `wk1` (codex). Not duplicated.
  Used here only as the *target corpus* for testing whether window-1 items were later consumed.
- **07-24 → 07-31 consumption/coherence** — landed by `ddm_cn3`.
- **Unmerged-branch / worktree harvest** — landed by `ddm_cs1`.
- **Wire-or-retire adjudication of built-unwired successors** — landed by `ddm_wr2`; its RECIPIENT
  scope-correction is adopted here, not re-derived.
- **"ΔS quoted against a dead baseline"** (find class 5) — already measured and landed by `ddm_qd1`
  (`a_delta_without_its_baseline_is_unanchored...`). **Deliberately not re-derived**; rediscovery is
  the cardinal signal-loss sin. Cross-referenced only.
- **`experiments/results/`** — 78% of repo `.py` is vendored there. Excluded from all consumption
  greps; stated on every scan.
- **Scorer runs** — $0 constraint; the eval slot is held by `ob1`.

---

## §2 — The 07-04 → 07-19 finds

### 2a. The ledger is nearly absent for this window — which is *why* it went unswept

| Measure | Value |
|---|---:|
| Window-1 memos | 1,052 |
| Window-1 ledger rows (any id form) | **51** |
| → memos with ANY ledger row | **4.8%** |
| → of those 51, resolvable by bare `task #N` | **5 (9.8%)** |

There was no repo-visible tracking substrate for this window. Anything that stalled here could not
have been caught by a ledger-driven sweep, because the ledger barely covers it.

### 2b. Status of the 51 window-1 rows

`pending` 21 · `completed` 19 · `blocked` 8 · `in_progress` 3 → **32 non-completed**.

Of the 32: **8 carry a named blocker** (a legitimate exit under the operating rules) and
**24 are silently parked** (no blocker, no fire-condition) — 3+ weeks stale.

### 2c. The 24 silently-parked rows, split honestly

Tested against all **1,158** window-2 memos, excluding audit/ledger memos (a mention inside a
backlog-drain memo is *re-listing*, not consumption).

**STATUS-ROT — work continued under another name, row never closed (14):**

| Row | Later substantive memos |
|---|---:|
| `D52b` cured HOSC activation | 16 |
| `jrd` v9 CGauge entropy route | 16 |
| `costate_organ_duty_queue` | 12 |
| `PDW1` palette section | 9 |
| `D44` converged local margin-saliency/taper | 5 |
| `D49` current-V9 optimal micro-batch | 5 |
| `D46` SPS after real temporal engagement | 3 |
| `D52c` FreSh governed execution | 3 |
| `D42` whole-teacher decision-quotient student | 2 |
| `D43` custom sparse-adjoint execution | 2 |
| `jrd` witness rate-instrument route | 2 |
| `phase_residual_carrier_store_half_359` | 2 |
| `494` compute-substrate authority ladder | 1 |
| `jrd` pose-decoupling R1 composition | 1 |

**TRUE ORPHAN — no later substantive mention in 1,158 window-2 memos (10):**

`C1-WITNESS-CLEAN-STAGE-EMA` · `D45` AdamW optimality / MLX semantics gap · `D47` transported /
event-triggered costate reuse · `D48a` YOPO validation cadence · `D48b` feature-ball suffix
certification · `D48c` INSTANT native projected execution · `D50` ANE/CoreML advisory ·
`D51` exact-integer megakernel · `jrd` n600 tensor prior · `jrd` training-time entropy edge route

(8 of these 10 are D-series rows — see §3b.)

Scope statement: "did not find in the 1,158 window-2 `.omx/research/*.md` memos, excluding
audit/ledger memos." This is **not** a claim that no artifact anywhere references them.

**A false positive I caught in my own instrument.** My first probe for `D50` used the substring
`'ane '`, which matches "l**ane** ", "pl**ane** ", "membr**ane** " — it reported **607** later memos and
would have classified D50 as healthy status-rot. Word-boundary `\bANE\b|\bCoreML\b` returns **0**.
D50 is a true orphan. Control retained in the record: 607 (bad term) vs 0 (correct term).

### 2d. Phantom arms — chartered, built, never fired

Sub-sweep denominators: 199 charter-shaped window-1 filenames extracted, 198 testable; **99** had zero
later-research-filename match, **7** zero commit-subject match, **6** zero on **both**. All 13,961
commit subjects scanned via `git log --format=%s` (count verified against `git rev-list --count`);
`experiments/results/` excluded.

**Instrument note that changed the sweep:** the `ddm_xx1` arm-code convention **does not exist in this
window** — the earliest `ddm_*` artifact is 2026-07-22, and **0 of 924** fall in 07-04→07-19. Window-1
arms are named by charter-memo stem + task number. A generic `[a-z]{2}\d` regex returned 100 codes of
which the top 60 were noise (`fp32`, `ep0`, `bf16`, `se3`); single-letter labels (939 distinct) were
rejected as untestable — grepping `C1` matches anything mentioning it, the self-match trap.

| # | Phantom | Chartered to measure | Evidence of no output | Conf |
|---|---|---|---|---|
| 1 | **#563 duty-queue fire chain** (4 tickets) | paired OFF/ON n600 ep725-fork A/B on d_seg for top-2 EV levers | all 4 `dry_start_receipt.json`: `trainer_or_governed_launch_executed:false`, rc=6; verdict cards say "no trainer was invoked". **Operator approved TRAINING GOs on 07-19; 2,040 commits since, zero further mention.** Not covered by `ddm_qd1` | HIGH |
| 2 | **v7.5.3 increment** | 5 deltas incl. a 10-rung registered-OFF ladder | built 07-10 with a 22-test suite; last commit naming `v753` = **07-11**; vehicle jumped v9c→v10→TR1; ladder never launched | HIGH |
| 3 | **ARM-C `LaneSkipBand` #524** | d_seg A/B of stride-2 skip-band supervision | **0 rows in `lever_activation_ledger.jsonl`** (37 levers tracked; `lane_skipband` absent); `arm_c_skiplever`/`ema_comparator` = 0 hits outside the charter | HIGH |
| 4 | **P0 FORCES phase-2** (FORCE 1/2/3) | "one per crucible increment with a measured n600 A/B" | `temporal_screw_consistency` (13 rows), `margin_band_satisficing` (12), `tie_locus_displacement` (12) — **every row `verdict_ref: null`**; all events `fired`, never `measured` | HIGH |
| 5 | **ARM-BASISPROD** | close 3 P0 build obligations without weakening V9 provenance | 8 files, **0 commit subjects**; all downstream hits are re-listings | HIGH |
| 6 | **`modal_moneysafety`** | pre-paid-launch Modal money-path audit | 2 files repo-wide, neither a result | MED-HIGH |

**`fired` does not mean `measured`.** The lever ledger holds 251 rows — 243 `fired`, 7 `measured`, 1
`built` — where `fired` means *present in a launched config*. **29 of 37 levers carry `verdict_ref:
null` on every row**: they ran and were never adjudicated. This is a distinct orphan class from
"never ran," and it is invisible to any sweep that counts activations rather than verdicts.

### 2e. Named-$0-follow-ons never run

2,256 OWED/`$0` lines across 693 of 1,052 memos; **414** across 204 files name both a score axis and an
action verb. The `default_off_decision_table_20260710.jsonl` holds 185 rows; **31** are dispositioned
fire-now/measure-cheap with `ever_fired=false`; **26 of those 31 still show no lever-ledger firing**
after 07-10 — a 24-day-undrained queue.

**Two that transfer to the live TR1 line** (both rest on measured *scorer* facts, not vehicle facts):

- **`blind_coordinate_generic_fill_401`** — generic-fill the ~230,904 camera px/frame that **no scorer
  resize reads**. Rate-side, vehicle-independent: `m86` measured `D` as disjoint 2×2 sampling with
  **22.70% of camera px blind to BOTH scorers**, and the lattice is unchanged by the vehicle pivot.
- **ARM-C `lane_skipband`** — derives from measured SegNet structure (single 16-ch stride-2 skip at
  (192,256); ablating it → 8,072 flips, **77% Road↔Lane**), and `m91` prices Road↔Lane at **22.1% of
  the entire remaining gap**. The charter's finding that the witness carried only ~10% of GT's Lane
  skip-band energy (1.68e-4 vs 1.70e-3) is **witness-scoped** and must be re-measured on TR1.

**What explicitly does NOT transfer.** The #563 tickets carry HWM at *43.8% of remaining descent* and
StepNative at *31.6%*, both labelled MEASURED. They were priced against the **0.19108 borrowed
pointer on the witness vehicle**. Per `m07`/`L18` those are **hypotheses on TR1, not results**. The
mechanisms remain aligned with the level-set physics; the EV numbers must be re-derived before any
launch is routed on them. Recording this explicitly, because "MEASURED" labels on ancestor-vehicle
numbers are exactly how a dead baseline gets divided by.

---

## §3 — Cross-window finds the sub-window sweeps could not structurally see

A window-scoped sweep cannot see any of the following, because each requires holding **both** windows
at once. These are the finds that justify a month-scale arm existing at all.

### 3a. Status-rot is invisible to either window alone

Detecting "row says `pending`, work actually continued" requires the row (window-1) **and** the
continuation (window-2). A window-1 sweep sees a pending row and calls it an orphan. A window-2 sweep
sees active work and calls it healthy. **Both are wrong.** Only the join shows 14 rot / 10 orphan.
This is the single largest correction in this memo (§6).

### 3b. The D-series lineage break — orphaning **by rename**

Window-1 created deferral rows `D41`–`D53` in `canonical_task_status.jsonl`. Window-2 created a new
surface, `ddm_deferral_queue_ledger_20260729.md`, which runs its **own fresh series** — `D1, D2, D5,
D8, D16` — and carries only **2 of the 13** prior rows (`D42`, `D52c`).

The other **11 D-rows were dropped at the rename** while remaining `pending` in the canonical ledger.
That is precisely where §2c's true orphans concentrate: 8 of the 11 true orphans are D-series rows.

**Orphaning by rename is a distinct mechanism** from orphaning by neglect. Nobody decided to drop
`D48a`; a new ledger simply started counting from D1. Two live surfaces now share one namespace, and
the new series counting upward will collide with the old at `D41`.

### 3c. The tracking apparatus is the month's fastest-growing orphan generator

Deferral/backlog/orphan-named memos created this month:

| Window | Count | Files |
|---|---:|---|
| W1 07-04→07-19 | **10** | `sweep_C_task_research_orphan_lever_ledger` · `costate_controller_deorphan_inventory` · `owed1_repaired_pose_gate_build` · `v752_owed_gates_build` · `v752_owed15_isolation_runbook` · `owed16_bounded_ab_and_drystart` · `ladder_owed_measurables` (+DAG_FEED) · `consolidation_quarantine_orphan_tests` · `m5_burndown_orphan_trainers_incident` |
| W2 07-20→08-03 | **6** | `ddm_deferral_queue_ledger` · `ddm_p2a_task_backlog_drain` · `ddm_fo1_orphaned_followon_detector` · `ddm_rs2_orphan_resumption` · `ddm_oh1_orphaned_handoff_sweep` · `ddm_qd1_backlog_drain` |

**16 tracking surfaces in 31 days — one roughly every two days**, sustained across the whole month,
plus `canonical_task_status.jsonl` plus the harness TaskList: ~18 competing answers to "what is owed."

**This memo would be the 17th.** That is the finding, and it implicates this arm directly. Each sweep
has *created a surface* rather than *draining a canonical one* — which is why the same items keep
being re-found: `ddm_qd1`, `ddm_fo1`, `ddm_cn3` and now `ddm_mh1` have each independently re-listed
portions of the D-series.

**Binding consequence adopted for this arm's own §5:** routing lands as **rows in
`canonical_task_status.jsonl`**, the store subagents can actually read. A memo-only routing section
would reproduce the exact failure this section documents.

### 3d. Three substantive memos are UNCOMMITTED — signal at literal risk

`git ls-files --others --exclude-standard -- '.omx/research/*.md'` returns **3** files, ~54 KB, and
`git check-ignore` confirms **none is gitignored** — they are simply never committed:

| File | Size | Age | Note |
|---|---:|---|---|
| `ddm_cr1_composition_row_827_20260801.md` | 17.5 KB | **2 days** | Headline claim: *"the seg+rate prize is 2.4× larger than recorded, gated by a MEASURED pose wall"* |
| `ddm_op3_canonical_operating_point_20260803.md` | 29.5 KB | same-day | one canonical live operating point |
| `ddm_dn1x_IMPLEMENTATION_SPEC_20260803.md` | 7.2 KB | same-day | `dn1x` is one of the four arms recovered from transcripts on 08-03 |

**Uncommitted is worse than orphaned.** An orphaned committed memo is at least *findable* by the next
sweep; an uncommitted one is invisible to every subagent (they read the repo, not the working tree)
and is lost outright on a clean checkout.

Honest scoping: `op3` and `dn1x` are **same-day** and plausibly belong to still-live arms — not yet a
failure. **`cr1` at 2 days is the real find**, and it carries a large substantive claim. Note the
recursion: `dn1x`'s spec being uncommitted means the 08-03 phantom-arm *recovery* is itself
incompletely landed.

This arm did not commit them: `cr1` is explicitly protected from this arm's edits, and capturing a
live arm's partial write would be worse than flagging it. Routed in §5 instead.

### 3f. The anti-forgetfulness apparatus is itself losing memories

Checked because the arm needed to land a durable law and found it could not.

- **`MEMORY.md` is at 17,402 B against its own stated 17,408 B budget — 6 bytes of headroom.** That
  budget exists so the index **fully loads** at session start. At 99.97% capacity, any new law from
  any arm either does not fit or pushes the index into partial loading — **silent memory loss for
  every future session.** This is a P0 apparatus risk, not a formatting nit.
- **Two of 90 keys are orphaned.** 88 keys are referenced from `MEMORY.md`; **`m64`** and **`m92`**
  are not, though both files exist. An unreferenced key never loads, so the memory is effectively
  lost. `m92` is the pointed one: its content is *"findings die with the arm — the crash-resume
  checkpoint has no FINDINGS field."* **The law about lost findings is itself a lost finding.**
- **Concurrency observed:** the on-disk `MEMORY.md` no longer contains the `(m64)` reference that
  this session's context snapshot shows — `MEMORY.md` is being edited by a **concurrent live arm**.

**What this arm did, and deliberately did not do.** It wrote the law file
(`orphan_sweeps_that_do_not_write_the_store_are_the_disease_20260803.md`) and registered key **`m93`**
in the low-contention keys file. It **did not edit `MEMORY.md`**: editing a 6-byte-headroom file
under an active concurrent writer is the documented absorption / commit-swap bug class, and shaving
bytes off other arms' laws under time pressure is not this arm's call. The index pass is OWED and
routed (`mh1_memory_index_saturated_and_two_orphaned_keys`).

Honest consequence: **this arm's own law file is, at the moment of writing, reachable only via the
keys file — not from the loaded index.** That is the disease, and naming it is better than pretending
the routing is complete.

### 3e. Re-listing is not consumption

A mention inside a backlog-drain memo means an item was *re-noticed*, not *worked*. Counting those as
consumption inflates health: for `jrd`, 2 of 11 later mentions are audit memos; for `PDW1`, 5 of 16.
All §2c classifications exclude audit/ledger memos for this reason. Any future orphan sweep that
greps for "is it mentioned later?" without this exclusion will report the backlog as healthier than
it is — and will be counting *its own predecessors' re-lists* as progress.

## §4 — The join coverage number (task #880)

**There is no single join number.** Four different denominators measure four different things, and
collapsing them would itself be a signal-loss move. All four are reported.

| # | Question | Coverage | Denominator |
|---|---|---:|---|
| J1 | Of task ids **cited in prose**, how many resolve in the repo ledger? | **25.0%** | 16 ids (pre-registered) |
| J2 | Of the ~921 harness TaskList rows, how many have a bare-int repo row? | **4.6%** | 42 of ~921 |
| J3 | Of window-1 ledger rows, how many are reachable by a bare `task #N` lookup? | **9.8%** | 5 of 51 |
| J4 | Of window-1 memos, how many have ANY ledger row? | **4.8%** | 51 of 1,052 |

### J1 is pre-registered, not cherry-picked

The 16 ids are exactly those cited in this arm's own briefing — fixed **before** any result was seen:
`#870 #864 #868 #880 #885 #875 #827 #826 #766 #383 #411 #346 #247 #522 #863 #871`.

- **Resolve (4):** `383`, `826`, `827`, `871`
- **Miss (12):** `247 346 411 522 766 863 864 868 870 875 880 885`
- Prefix-parse (`^N[_-]`) recovers **zero** additional — the generous and exact readings coincide.

This is independently consistent with `na1`'s separately-measured 10-of-14 miss (28.6% coverage).
Two samples, two methods, ~25–29%.

### The mechanism, measured

The 149 distinct repo ids are **42 bare-int + 107 slug/composite**. Only **10 of the 107** slugs carry
a leading integer recoverable by prefix-parse. So **97 of 149 ids (65%) contain no integer in any
position a `task #N` lookup would find.** Examples: `deferral_ledger::D42`,
`AUTH-C4-MOD19-LINUX-X86_64-20260715`, `costate_organ_duty_queue_20260711`.

### The ledger is not a sample — it is one backfill plus scatter

Bare-int ids present: `383 438 455 456 494 578 603 793 799` → **contiguous 800–828** → `850 871 873 882 909`.

Dating the rows shows the block **793–828 was written entirely on 2026-07-31**, within hours. That is a
**one-off backfill event**, not steady mirroring. Since then only 5 rows landed across the 81 id-slots
829–909 (**6%**). Before it, the primary window has **5 bare-int rows in 16 days**.

**Consequence:** the ledger's apparent health is an artifact of a single backfill day. Sampling it on
07-31 or later would show a dense, healthy-looking recent block and conceal that both the preceding
month and the following days are near-empty.

### Confirmation that the missing rows exist only in the harness

Searching the whole repo (`.omx/state/`, `.omx/research/`, `src/tac/`, `tools/`) for the *content* of
the concepts behind the missing ids: the phrase `named-$0-follow-on` returns **0 repo hits**. That
class exists only in the harness TaskList — structurally invisible to every subagent, which can see
only the repo store.

**This is the mechanism of the whole failure:** an arm told "sweep task #N" is not being lazy when it
finds nothing. It is querying a store that, for 65% of ids and 95% of the primary window, cannot
answer. An id-not-found is a **missing join**, never an absent row.

## §5 — Ranked routing (LANDED as ledger rows, not memo-only)

**Why this section is short and the ledger is long.** §3c measured that 16 tracking surfaces were
created this month and that sweeps *create* surfaces instead of *draining* canonical stores. The
proof, caught in the act: **`ddm_qd1` (08-03, today) dispositioned window-1 rows as "SUPERSEDED" in
its memo and wrote ZERO ledger rows** — those rows still read `pending` with last events dated
07-13/07-14/07-15. That is why I am the 4th arm to re-find the same D-series.

So this arm wrote its routing **into `canonical_task_status.jsonl`** via the canonical writer
(`register_task` / `append_note`, fcntl-locked, append-only):

- **12 new task rows** registered, each with a **named owner** (no row exits "unowned" — `m45`).
- **24 in-ledger annotations** on the parked window-1 rows: 14 tagged STATUS-ROT with their successor
  evidence, 10 tagged TRUE-ORPHAN with exact scope. The next sweep inherits the evidence instead of
  re-deriving it.
- **2 arithmetic-correction notes** on this arm's own rows (13→14, 11→10), since ids are immutable.

### Ranked head

Ranking is by **gap-share of the axis touched × vehicle-transferability × cost**, not by novelty.
Seg is the majority axis and rate is the cheapest; `$0` items outrank funded ones.

| # | Row | Axis | Why ranked here |
|---|---|---|---|
| 1 | `mh1_materialize_harness_rows_into_repo_ledger` | apparatus | **Root cure.** Caps every subagent sweep at ~25% visibility until fixed. Everything below is re-findable only if this lands. |
| 2 | `mh1_recover_lane_skipband_arm_c_524` | **seg** | Rests on measured SegNet structure; `m91` prices Road↔Lane at **22.1% of the entire gap**. Never fired — absent from all 37 tracked levers. |
| 3 | `mh1_recover_blind_coordinate_generic_fill_401` | **rate** | `$0`, vehicle-independent (`m86`: 22.70% of camera px blind to BOTH scorers). Undrained 24 days. |
| 4 | `mh1_redrain_default_off_table_and_lever_verdicts` | apparatus | 26/31 fire-now rows unfired; **29/37 levers never adjudicated**. A wired gate did not drain the queue. |
| 5 | `mh1_close_13_status_rot_window1_rows` *(=14)* | apparatus | Removes the 2.4× inflation from every future sweep. Evidence already in-ledger; only adjudication remains. |
| 6 | `mh1_adjudicate_563_duty_queue_fire_chain` | seg | Operator-approved TRAINING GO on 07-19, trainer never invoked, 2,040 commits silent. **EV numbers need re-derivation first** (ancestor rule). |
| 7 | `mh1_commit_or_disposition_uncommitted_research_memos` | custody | `ddm_cr1` 2 days uncommitted carrying a "2.4× larger seg+rate prize" claim. Uncommitted is worse than orphaned. |
| 8 | `mh1_gap_decomposition_mixed_denominators_recompute_at_pu2` | measurement | Cited shares sum to 109.6%. Owner `ddm_op3`. Cheap; prevents routing on a stale denominator. |
| 9 | `mh1_reconcile_dseries_lineage_break_and_namespace` | apparatus | Two live D-series; will collide at D41. |
| 10 | `mh1_adjudicate_11_true_orphans_window1` *(=10)* | mixed | Wire-or-retire; 8 of 10 are rename casualties. |
| 11 | `mh1_consolidate_16_orphan_tracking_surfaces` | apparatus | The meta-fix; without it, sweep #18 re-finds all of this. |
| 12 | `mh1_adjudicate_v753_ten_rung_ladder_never_launched` | — | Expected **RETIRE-with-lessons** (witness lineage, `m34` ban), not revival. Listed last deliberately. |

**No ΔS is claimed for any row.** This arm ran no scorer (`$0`; the eval slot is held by `ob1`).
Every row states its axis and its evidence; none states a predicted score delta, because a predicted
delta without a re-derived baseline is exactly the class §6/R5 and row #8 exist to stop.

## §6 — What this arm refutes (including its own working hypotheses)

### R1 — "24 window-1 orphans" — REFUTED by this arm's own data (2.2× inflation)

The natural output of a ledger-driven sweep is: *32 non-completed rows, 8 legitimately blocked,
therefore **24 orphans**.* Tested against 1,158 window-2 memos, **14 of the 24 are status-rot** — the
work continued under other names and only the row went stale. The true orphan count is **10**.

A naive count inflates by **2.4×**. This arm was on course to publish that number before running the
consumption test. **Any orphan count not paired with a forward-consumption test is inflated by
construction**, and every one of the 16 tracking surfaces in §3c is a candidate for having done this.

### R2 — This arm's own D50 probe was a false positive (607 → 0)

Probing `D50 ANE/CoreML` with the substring `'ane '` matched "l**ane** ", "pl**ane** ", "membr**ane** "
and returned **607** later memos, classifying a true orphan as healthy. Word-boundary
`\bANE\b|\bCoreML\b` returns **0**. The implausibility of 607 is what triggered the recheck — a
plausible-looking wrong number (say, 12) would have passed silently.

**Sub-lesson:** short/common-substring probes are the orphan-sweep analogue of the vacuity law. A probe
that cannot distinguish its target from a common English fragment reports *health it did not measure*.

### R3 — "Research memo output collapsed after 07-19" — REFUTED as an instrument artifact

Date-filtered `git log` reported 1,087 memos in window-1 vs **25** in window-2, suggesting a 43×
collapse in output. The true counts from a full-history pass are **1,052 vs 1,158** — output slightly
*increased*. The "collapse" was `--since/--until` truncating graph traversal.

### R4 — "The primary window went unswept because no one got to it" — REFUTED

It went unswept because **there was nothing to sweep from**: the repo ledger holds 51 rows against
1,052 memos (**4.8%**), and only 5 are reachable by the `task #N` convention (**9.8%**). A
ledger-driven sweep of window-1 would have returned near-empty and — per the vacuity law — that empty
result would have read as "clean."

### R5 — "Cite CONTENT, never bare ids" is NECESSARY BUT NOT SUFFICIENT — sharpened

The standing cure (m89) is correct as far as it goes: 65% of repo ids contain no integer a `task #N`
lookup can match, so id-citation is structurally broken. **But content-citation fails on the same
items.** The phrase `named-$0-follow-on` returns **0 hits** across `.omx/state/`, `.omx/research/`,
`src/tac/`, `tools/`. An arm told to sweep that class *by content* also finds nothing, because the
class exists only in the harness TaskList.

**The binding conclusion is therefore stronger than a citation-style rule:** citation style cannot fix
a store that does not contain the row. The cure is to **materialize harness rows into
`canonical_task_status.jsonl`** — the only store subagents can read. Until that exists, both
id-citation and content-citation fail on harness-only items, and every subagent sweep inherits a
~25% ceiling on what it can even see.

### R6 — What this arm does NOT claim

- Not claimed: that the 11 true orphans are dead. Scope is exactly *"did not find in the 1,158
  window-2 `.omx/research/*.md` memos, excluding audit/ledger memos."* They may live in code, state,
  commits, or the harness.
- Not claimed: any ΔS. This arm ran no scorer; it is `[audit]`, `score_claim=false`, pointer UNMOVED.
- Not claimed: that ~921 is the exact harness row count — it is the briefing's figure, used only for
  the order-of-magnitude J2 ratio, and J1/J3/J4 do not depend on it.
