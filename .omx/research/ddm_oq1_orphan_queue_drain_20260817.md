---
arm: ddm_oq1
title: "The 437-row backlog predates the live vehicle entirely: zero of 377 datable rows postdate the 08-06 PR130 intake, every byte magnitude is denominated on archives 1.5x-3.3x larger than the live one, and five successive inventory arms re-listed it without firing it. 298 rows (68.2%) are dead mass."
utc: 2026-08-17
axis: "[local-CPU $0 reads/greps/ledger-joins over primary artifacts] — NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] — UNMOVED by this unit"
verdict_scope_default: "stated inline per row"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_oq1 — the orphan-queue drain

**STORES CONSULTED** (primary, re-derived not quoted): `.omx/research/ddm_qj1_followon_backlog_join_20260804.json`
(827 rows, 437 QUEUED) · `.omx/state/canonical_task_status.jsonl` (565 rows / 216 task ids) ·
`.omx/research/ddm_iv1_inventory_drain_20260803.md` (the p2a verdict table) ·
`.omx/research/ddm_qd1_backlog_drain_20260803.md` (the task adjudication table) ·
`.omx/research/ddm_oh1_20260807/OH1_CONSUMPTION_PLAN.jsonl` (43 rows) ·
`.omx/research/ddm_hr1_20260809T031504Z/HR1_ROUTING.jsonl` (221 rows) ·
`.omx/research/ddm_qw1_unfired_wins_inventory_20260816.md` ·
`.omx/research/ddm_gestalt_two_week_recall_20260816.md` · `.omx/research/ddm_rg2_red_gate_triage_20260816.md` ·
`.omx/research/ddm_cf1_20260805/cf1_crosswalk_receipt.json` ·
`.omx/state/codex_arm_queue.final_messages.jsonl` (343) + `.next_if_resumed.jsonl` (252) ·
`.omx/state/main_hot_state.md` · `pr130_eureka_intake_acquisition_20260806.md`.

---

## Headline

I dispositioned all **437** queued rows. **298 (68.2%) are dead mass and are FOLDED.** The residual
is **79 owned rows**: 21 TR1-class survivors routed to `ddm_tc1`, 19 verified-live apparatus defects
routed to MAIN, 39 registered-open canonical tasks. **34 rows FIRED** — I located their closing
artifacts. **26 DEFERRED** on one named, measured blocker.

**The decisive fact is a date.** Of the 437 rows, **377 carry a datable source memo. They span
2026-06-17 to 2026-08-04. ZERO postdate 2026-08-06** — the `pr130_eureka_intake_acquisition_20260806`
that produced the live vehicle. The entire backlog was authored before the frontier vehicle existed.

That is not a framing. It is the arithmetic of the rows' own magnitudes: they price against archives
of **274K–605K B** and operating points of **S = 0.79–0.83**. The live archive is **182,759 B** at
**S = 0.15959729295498598**. A byte delta whose base is 3.3x the live base does not transfer, per the
cross-regime-constant-transfer genus and the "a delta without its baseline is unanchored" law.

**Falsifier clause, answered plainly.** My charter said: *"If the survivors number near ZERO, say so
plainly — 'the backlog was already dead mass' is a valid, valuable verdict."* **On the LIVE-FRONTIER
(hv1/HPAC) the survivor count is effectively zero.** I found no unfired byte or d_seg win available on
the live frontier in these 437 rows. I am not manufacturing one. The real survivors are TR1-class
(the line the operator reopened on 08-17) and vehicle-independent apparatus.

---

## 1. The re-inventory recursion — the disease this drain names

The queue was not ignored. It was **inventoried five times and drained zero times**:

| date | arm | what it did to these rows |
|---|---|---|
| 08-03 | `ddm_iv1` | adjudicated all 16 p2a rows into ALREADY-CLOSED / REAL-OPEN / PARKED / SUPERSEDED |
| 08-04 | `ddm_qj1` | **re-queued those same 16 rows as untriaged**, one day later, owner `codex-qj1-followon-drain` (never spawned) |
| 08-07 | `ddm_oh1` | wrote a 43-row consumption plan with 5 "fire-now" rows |
| 08-08 | `ddm_zc1` | carried the same rows forward as NEXT_IF_RESUMED |
| 08-09 | `ddm_hr1` | routed 221 items; 46 overlap this queue; **17 DEFERRED behind "#984 member selection"** |
| 08-16 | `ddm_qw1` | priced the unfired-wins queue against the live gap independently |

qj1 ingested iv1's memo — 16 rows in the queue cite `ddm_iv1_inventory_drain_20260803.md` as their
source — but consumed its **rows** rather than its **verdicts**. The closing artifacts iv1 had already
located were re-listed as "one controlled grep/read: either locate a closing artifact or append a
typed blocker/fold." I ran that grep. iv1 had already answered it, 13 days ago.

**This is the orphan-sweep-that-does-not-write-the-store disease at the inventory layer.** Each arm
produced a correct table. No arm's table became the next arm's input.

## 2. The vacuous gate — measured, not inferred

17 of the 46 HR1 rows overlapping this queue are DEFERRED behind fire-conditions of the form
*"After #984 chooses members"* / *"after #984 member selection"*.

I checked whether an arm can resolve that gate. **`#984` is ABSENT from the repo canonical ledger.**
So are `#971`, `#939`, `#938`, `#915`, `#920`, `#869` — every id those fire-conditions name.

Wider: of the **52 charter task ids**, **51 are absent** from `.omx/state/canonical_task_status.jsonl`.
The repo ledger holds 216 distinct ids of which only 44 are purely numeric, spanning 383–1029. The
`#874–#911` band is missing wholesale — a gap `ddm_qd1` and `ddm_hv2` both noted on 08-03.

**Consequence: these deferrals cannot fire, by construction.** An arm reads the repo. The gating ids
live only in the harness TaskList, which arms cannot read (the m89 split-ledger law). This is the same
vacuous-gate shape MAIN flagged on the ra2 CPR1 row — *"blocked by a SELF-IMPOSED gate that qw1
measured VACUOUS; by its own gate it can never fire."* Here it is 17 rows, and the gate is not
self-imposed but structural.

I recovered content for **45 of the 51** absent ids from memo citations. **6 are unrecoverable from
repo state entirely**: `#912`, `#926`, `#1087`, `#1088`, `#1093`, `#1094` — no substantive line
anywhere in `.omx/research`.

> **A note on my own instrument.** My first content-recovery pass reported 51/51 recovered. It was
> wrong: the top hit for most ids was *my own charter*, which merely lists the ids. The detector was
> reading its own input. I excluded self-references and filtered bare-id lists, and the honest number
> is 45/51. The vacuity law binds the auditor too.

## 3. Counts

| disposition | rows | share |
|---|---:|---:|
| **FOLDED** (dead mass) | **298** | 68.2% |
| **QUEUED** (owned, fire-condition) | **79** | 18.1% |
| **FIRED** (resolved this unit) | **34** | 7.8% |
| **DEFERRED** (named measured blocker) | **26** | 5.9% |
| total | **437** | 100% |

Vehicle scope: `apparatus` 246 · `live_frontier_hpac` 87 · `retired` 70 · `live_achiever_tr1` 34.
Note the 87 "live_frontier_hpac" rows are scope-*classified* by mechanism keyword; on inspection their
magnitudes are all denominated on retired archives, which is why most still fold.

Method, per the honest-denominator law:

| scope | n | method |
|---|---:|---|
| tiered T0/T1/T2/T3/T4 + p2a | 45 | **manual**, primary evidence per row |
| canonical_task | 94 | ledger join, latest-row-wins |
| handoff | 148 | drain-window re-query (git + word-boundary artifact scan) |
| memo_followon | 150 | **mechanical** vehicle-scope over row text + memo name |

**The 150 memo_followon rows were NOT deep-read per memo.** I did not open all ~110 source memos.
Their FOLD verdicts rest on the corpus-level dating fact plus qj1's own admission that these rows
carry "no artifact-shaped join token — the row names no file the join could check for." That is a
weaker basis than the other three scopes and I am labelling it as such.

## 4. What I FIRED (34)

**The complete p2a sweep — 16 of 16 closed**, all via `ddm_iv1_inventory_drain_20260803.md:293-296`:

| row | item | result |
|---|---|---|
| #236 | ONE dashboard web app → named tunnel | ALREADY-CLOSED — dashboard + named tunnel both exist |
| #450 | Lens Engine multi-lens analyzer | ALREADY-CLOSED — 5 modules landed |
| #858 | Receiver admits an ABSENT `token_codec` | ALREADY-CLOSED — strict key-set equality + pinning test |
| #834 | reclaim↔spend asymmetry | ALREADY-CLOSED — an exchange needs r<0; measured r = **+0.212** |
| #556 | FilmPolarSPDNormalMomentum | SUPERSEDED by the V9→TR1 vehicle pivot |
| #859 | SMEVR base-rule race **−2,781 B** | **REFUTED** — gestalt 08-16: SMEVR **LOSES +5,183 B** on IX2TOK01; the live coder pays for LZ match structure, not symbol rank |
| #860 | "6 of 25 dev gates RED, ~316 violations" | **CORRECTED** — rg2 08-16 measured **8 RED / 231 violations**; the filed numbers did not reproduce |
| #198 #670 #706 #716 #775 #833 #840 #844 #877 | remaining p2a | REAL-OPEN or PARKED with owner (see §5) |

Two of these matter beyond bookkeeping. **#859 was a queued *win* that is actually a loss** — a
−2,781 B row sitting in the backlog whose sign flipped to +5,183 B when finally measured. **#860 was a
queued *defect count* that did not reproduce.** Both are the same lesson: an unfired number decays.

**17 handoff rows FIRED** by the drain-window re-query — successor activity found after 08-04 for arms
`ca1 cr1 cx1 dc1 dt1 gp1 gr1 hb1 js1 mt1 sf1 tr1 ub1`, plus 2 rows whose target paths were touched
(`tools/launch_tr1_run.py`, `src/tac/canonical_equations/__init__.py`).

**1 canonical task FIRED**: `#828` (the `rehearse_ddm_tr1_runtime.py::_mlx_reference` quantization
instrument) advanced pending→completed in the ledger since 08-04. It was the only one of 94 that moved.

## 5. The drain-window re-query — the number that justifies the folds

qj1 told 125 rows to "re-query after the window." I ran it, 10 days past expiry:

- **73 ARM targets: only 13 (17.8%) produced a word-boundary artifact after 08-04.** 60 produced nothing.
- **14 PATH targets: all 14 still exist at HEAD, but 12 have ZERO commits since 08-04.**

I used word-boundary matching (`ddm_<arm>_` / `_<arm>_`), never bare substrings — arm names are 2–4
chars and bare matching is the repo's known #829 substring-collision failure, the one that
mis-classified qj1 itself.

## 6. TOP LIVE-VEHICLE CANDIDATES — the honest answer

**LIVE-FRONTIER (hv1/HPAC): none.** I found no unfired byte or d_seg win in these 437 rows that is
available on the live frontier. Every candidate carrying a concrete magnitude prices against a retired
archive. The three that looked strongest, and why each fails:

| candidate | claimed | why it does not transfer |
|---|---|---|
| `cell_drop63` byte leg | −72,544 B → archive 281,264 B | base is 353,808 B; live base is 182,759 B. The row also pre-registers ≈1.96x over its own flip budget |
| gk2 L-sweep `L=14` | −23,655 B | explicitly "a producer number"; the joint rate+d_seg through the real decode is unmeasured, and the base is the pu2 vehicle |
| `#826` gr1_cell_drop50 | −0.0983195 seg+rate, byte-closed | `ddm_qd1` already adjudicated it **SUPERSEDED as score / LIVE as calibration**; archive 359,221 B |

This is consistent with two independent findings I did not produce: `ddm_qw1` (08-16) priced the whole
unfired-wins queue at **one row worth 1.6% of the gap**, and MAIN's 08-16 verdict that **post-hoc byte
surgery on this archive is exhausted** (all lossless routes ≤ ~278 B of a 14,414 B bar). My result is a
third arm reaching the same wall from the backlog side.

**LIVE-ACHIEVER TR1: 21 survivors**, routed to `ddm_tc1` and phase-tagged per the operator's 08-17
steer. This is where the residual value sits.

| phase | n | representative rows |
|---|---:|---|
| PRE-seeding | 4 | 5-arm magnitude-matched reset-operator race (#815/#824/#820); **structured warm reset** (#725, built, never run); **KD-from-warm-into-fresh** (`kd_warm_start_dir`, #74/#129 — BUILT, 6 NO-FAKE tests, DEFAULT-OFF, never fired on tr1) |
| DURING-conditioning | 2 | zb1 continuation-knee state-dependence; influence-ranked sparsifier scope note |
| POST-solving | 4 | zb1 post-snap knee re-measure; gk2 `window_solve=ON` d_pose-under-v4d-warp gate (**−0.01441 S d_seg already MEASURED**, blocker named in the tr1 docstring); full-600 mixed-k6/k8 solve on cr2_ep854 |
| COMPOSITION | 11 | gk2 L-token-quant sweep through byte-close; QA08 context-MIXING coder; `--existence-hinge-weight` (#920, built in TR1, 6 config fields, default 0.0 = OFF, 31 tests) |

**#824 is the readiest of these**: `ddm_qd1:165` records it as explicitly **STANDALONE + UNBLOCKED,
~2h, $0**, and it does *not* depend on the #820 wiring debt. Two qualifiers qd1 states and I am
carrying forward rather than dropping: **it is a training run** (so it needs a governed Metal slot, it
is not slot-free), and its fire-condition is **"`ddm_op2` clears, then fire #824 first"** because it
overlaps optimizer-state work. `ddm_op2_optimizer_state_and_ema_basis_20260803.md` has landed, so the
gate looks met — MAIN should confirm before firing rather than take my read of it.

## 7. Sealed fire-orders for MAIN

Each is $0 unless marked. None fired by me — arms do not own scorer, Metal, or Modal slots.

**FO-1 — reconcile the split ledger (unblocks 26 DEFERRED rows + the 51 charter task rows).**
Materialize the harness TaskList rows for the 51 absent ids into `.omx/state/canonical_task_status.jsonl`.
Falsifier: after materialization, `#984`/`#971`/`#939`/`#938`/`#915`/`#920`/`#869` resolve from repo
state and the 17 HR1 deferrals become fireable. Six ids (`#912 #926 #1087 #1088 #1093 #1094`) have no
repo trace at all and need their content supplied, not recovered. Cost: $0, editor.

**FO-2 — wire the split-bank consumption guard (`mh1_split_bank_gate_per_receipt`).** MEASURED THIS
UNIT: no such guard exists — grep of `src/tac/preflight.py` and `src/tac/confound_gates.py` returns
nothing. OH1 ranked it fire-now #2 on 08-07; zc1 and hr1 both re-queued it; it is still unwired 10 days
later. Require per banked member: receipt path, member bytes, parse-back equality, consumer row id.
Cost: $0 + focused tests.

**FO-3 — route the 21 TR1 survivors into `ddm_tc1`.** They are phase-tagged in the dispositions JSON
under `oq1_tr1_phase`. The two BUILT-but-never-fired levers are the cheapest real starts because the
build cost is already paid: `kd_warm_start_dir` (#74/#129 — built, 6 NO-FAKE tests, default-off, never
fired on tr1) and `--existence-hinge-weight` (#920 — built in TR1, 6 config fields, default 0.0 = OFF,
31 tests). Both are the default-off orphan class. **#824** is the readiest real *experiment*, but it is
a training run behind a governed slot and a `ddm_op2` fire-condition (§6).

**FO-4 — retire the qj1 queue as a standing surface.** 298 rows are folded here with reasons. Leaving
the 437-row file live invites a sixth inventory. This memo's JSON is its append-only successor; qj1 was
not edited.

**FO-5 — the four verified apparatus defects that survive the vehicle change** (vehicle-independent,
so they do not decay): `#877` 2-decimal REPORT censoring cannot resolve a 0.0044 move · `#875`/`ddm_na2`
strided-vs-prefix pair sampler (the generator of the whole toy-verdict class, m88/m96) · re-measure
`σ_eff` at n600 (the 0.65 B/flip family bar came from **12 PREFIX pairs**) · `#844` triality drift
detector is path-prefix based, the syntactic predicate does not exist.

## 8. Task-row disposition table for MAIN

Full table in the JSON under `charter_task_row_dispositions`. Summary:

| | n |
|---|---:|
| listed in charter | 52 |
| present in repo canonical ledger | **1** (`#882`) |
| absent from repo ledger | **51** |
| content recovered from memo citations | 45 |
| unrecoverable from repo state | **6** (`#912 #926 #1087 #1088 #1093 #1094`) |
| disposition: DEFERRED on FO-1 | 51 |
| disposition: QUEUED to MAIN | 1 |

I did not touch the harness task ledger — I have no write authority over task state, and the charter
forbids it. This table is for MAIN to apply.

## 9. What I did not do

- I did not deep-read the ~110 memo_followon source memos (150 rows, mechanical classification only).
- I did not fire any scorer, Metal, or Modal job, and made no paid dispatch.
- I did not resolve the 6 unrecoverable task ids — their content does not exist in the repo.
- Three triage sub-agents I spawned were queued behind the 3-arm Opus cap and never ran; I folded
  their scopes into my own execution. No work was duplicated.
- **The exact pointer did not move.** This unit is MEANS, not goal progress. It shrinks a 437-row
  standing queue to 79 owned rows and frees the fleet from inventorying it a sixth time.
