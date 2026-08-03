# ddm_mh1 — MONTH-SCALE no-signal-loss / no-orphan harvest (2026-07-04 → 2026-08-03)

**Arm:** `ddm_mh1` · **Date:** 2026-08-03 · **Cost:** $0 (no scorer runs) · **Axis:** none (audit; no score claim)

Operator directive 2026-08-03, verbatim: *"You might need to look and harvest an[d ensure] no signal loss
or orphaned over the course of the past month, honestly."*

**Headline (the unflattering answer the "honestly" asked for):** the month's signal loss is **not**
mostly abandoned research. It is **status-rot in the tracking layer plus a structurally broken join**.
Of 24 silently-parked window-1 ledger rows, **13 are rot** (the work continued under other names; the
row was never closed) and **11 are true orphans**. And the repo-visible task ledger covers **4.8%** of
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

**STATUS-ROT — work continued under another name, row never closed (13):**

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

**TRUE ORPHAN — no later substantive mention in 1,158 window-2 memos (11):**

`C1-WITNESS-CLEAN-STAGE-EMA` · `D45` AdamW optimality / MLX semantics gap · `D47` transported /
event-triggered costate reuse · `D48a` YOPO validation cadence · `D48b` feature-ball suffix
certification · `D48c` INSTANT native projected execution · `D50` ANE/CoreML advisory ·
`D51` exact-integer megakernel · `jrd` n600 tensor prior · `jrd` training-time entropy edge route

Scope statement: "did not find in the 1,158 window-2 `.omx/research/*.md` memos, excluding
audit/ledger memos." This is **not** a claim that no artifact anywhere references them.

**A false positive I caught in my own instrument.** My first probe for `D50` used the substring
`'ane '`, which matches "l**ane** ", "pl**ane** ", "membr**ane** " — it reported **607** later memos and
would have classified D50 as healthy status-rot. Word-boundary `\bANE\b|\bCoreML\b` returns **0**.
D50 is a true orphan. Control retained in the record: 607 (bad term) vs 0 (correct term).

---

## §3 — Cross-window finds the sub-window sweeps could not structurally see

A window-scoped sweep cannot see any of the following, because each requires holding **both** windows
at once. These are the finds that justify a month-scale arm existing at all.

### 3a. Status-rot is invisible to either window alone

Detecting "row says `pending`, work actually continued" requires the row (window-1) **and** the
continuation (window-2). A window-1 sweep sees a pending row and calls it an orphan. A window-2 sweep
sees active work and calls it healthy. **Both are wrong.** Only the join shows 13 rot / 11 orphan.
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

### 3d. Re-listing is not consumption

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

## §5 — Ranked routing

*(populated below)*

## §6 — What this arm refutes

*(populated below)*
