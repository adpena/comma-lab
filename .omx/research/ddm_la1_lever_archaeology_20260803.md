# ddm_la1 — lever archaeology across all v-lineages: re-adjudication against `cx1`

**Date:** 2026-08-03 · **Arm:** `ddm_la1` · **Axis:** apparatus + lever re-adjudication.
**Scorer-free** (`ddm_pu1` holds the slot). **Pointer UNMOVED** (`0.1910828242` [contest-CPU custody]).
Nothing here is a score claim. Every number below is either MEASURED by this arm (labelled) or
RE-DERIVED from constants (labelled) — none is re-typed from a prior memo.

Operator directive: *"We should also look at all levers in all worktrees and branches from all
current and past v versions for signal we can now leverage for truly optimal."*

---

## ANSWER FIRST

**The signal is not in the branches, and it is not in the levers' verdicts. It is in the
instruments that report on levers — all three of which are pointed at a vehicle we no longer
ship.** The operator's premise inverts, and the inversion is the finding.

1. **The branch surface is empty.** 64 local branches, 54 worktrees, **only 5 not merged into
   `main`** — one of which is a live sister arm. Two prior arms already swept it.
2. **The lever-activation ledger is a dead instrument.** Last write **2026-07-27**, 7 days stale;
   it covers **37 of 180** levers; its writer `record_activation` has **zero automatic callers**.
   `never_fired()` returns **178 of 180** — and that number is **VACUOUS**, not evidence.
3. **The coverage query everyone runs answers about the dead vehicle.**
   `completeness()` defaults to the *levelset* trainer and reports a healthy **81.9%**. The
   **shipped** vehicle (`train_tr1_partition_renderer_mlx.py`) is a different trainer with a
   different flag set.
4. **The one correct instrument — `package_lever_factories()` — localises the whole problem to a
   single missing declaration.** Of 180 factories, **149 bind to the retired trainer and 31 to
   the live one; all 31 live factories are built and valid (0 missing flags)**. The 10 that
   cannot fire are all on the retired side — and **8 of those were written FOR the live vehicle**,
   mis-filed because `spec_tr1_renderer_20260728.py` is the **only** module in the package that
   declares `TRAINER_RELPATH`. Everything else silently defaults to the dead trainer.

**Consequence:** every "this lever was never used" and every "this lever is covered" statement in
the corpus that cites the ledger or the default `completeness()` call is unanchored — and two
08-02/07-29 sister memos already disagree with each other about which levers ever fired (§6.1).
`ddm_gd1` found that *nothing forces a never-fired row to be drained*; **§3.1 is the mechanism —
the levers designed for the live vehicle are filed under the dead one, so no live-vehicle query
can surface them.** The fix is one line per module. Ranked actions in §7.

---

## §1 — SCOPE + DENOMINATORS (state the basis; an empty scope is VACUOUS, never PASS)

| surface | denominator | measured how |
|---|---:|---|
| local branches | **64** | `git branch \| wc -l` |
| worktrees | **54** | `git worktree list \| wc -l` |
| **branches NOT merged into `main`** | **5** | `git branch --no-merged main` |
| …of which a live sister arm | 1 | `codexwt/ddm_de1_…` |
| corpus units queried | **12,096** | `corpus_query.py` STORES CONSULTED: research 7426 · equations 869 · memory 2061 · dag 915 · council 292 · tasks 417 · docs 96 |
| run dirs | **2,955** | `ls -d experiments/results/*/` |
| launch artifacts (`launch.sh`/`cmd.txt`/`argv.json`, depth≤3) | **169** | `find` |
| …belonging to the **live** vehicle | **0** | scanned all 169 for `train_tr1_partition_renderer_mlx` |
| package Lever factories | **180** | `package_lever_factories()` |
| modules scanned by that surface | **16** | `build_completeness().modules_scanned` |

**Counting-basis note (reconciling with the charter's 66/55/6):** the charter counted **remote
refs included**; the table above is **local refs only** (`git branch`, no `-a`). Both are correct
at their basis. Publish the basis, not just the count — the discrepancy is basis, not drift.

**Prior art I did NOT re-run** (checked first, per the anti-rediscovery law): the unmerged-branch
/ worktree harvest (07-28) and `ddm_cu1_consolidation_disposition_20260803.md` (155 files, all
155 dispositioned). `ddm_cu1` is **file custody**, not lever archaeology — the surfaces are
disjoint, which I verified by reading it before starting.

---

## §2 — THE THREE DEAD INSTRUMENTS (this is the actual finding)

### 2.1 The activation ledger is not run truth — and has now also stopped being written

MEASURED, `.omx/state/lever_activation_ledger.jsonl`:

| property | value |
|---|---|
| rows | **251** |
| distinct levers with ANY row | **37** (of 180 known = **20.6%**) |
| events | `fired` 243 · `measured` 7 · `built` 1 |
| first → **last** write | 2026-07-07T17:48:22Z → **2026-07-27T21:17:34Z** |
| staleness at this writing | **7 days** |
| automatic (non-test, non-module) callers of `record_activation` | **0** |

`never_fired()` → **178 of 180**. **This is an instrument artifact, not evidence.** Two
independent reasons, both verified by source inspection:

* **By construction:** the campaign already registered a canonical equation for exactly this —
  `activation_ledger_not_run_truth_v1` (`t5_crucible_measured_laws_20260707.py:819`), whose
  anchor states the ledger *"records ONLY `--dsl-lever-path` launches"* and that
  *"~10 of the '36 never-fired' DID raw-flag fire per launch.sh"*. Its named fix —
  **R1 = argv→lever reverse-map + engagement-predicate backfill** — is **still not wired** four
  weeks later. I did not re-derive this law; I confirmed it still holds and that its fix is owed.
* **By disuse:** even the narrow path stopped. Nothing has been recorded since 07-27, which is
  *before* the entire own-vehicle frontier era (`v4d → pw1 → ms8 → dc1_fold → pj2 → cx1`).
  **Every lever that moved the frontier six times is invisible to this ledger.**

### 2.2 The coverage query answers about the dead vehicle

MEASURED:

| scope | trainer flags | mapped | unmapped | stale | coverage |
|---|---:|---:|---:|---:|---:|
| `completeness()` **default** (levelset + base) | 443 | 363 | 80 | 3 | **81.9%** |
| `completeness('…train_tr1_partition_renderer_mlx.py')` (**shipped**) | **73** | 13 | 60 | 312 | 17.8% |

**I must correct my own first reading of this, and the correction matters.** The 17.8% row is a
*real computation* but the **wrong instrument**, so it must not be quoted as a live-vehicle gap:
`completeness()` compares against `dsl_referenced_flags()`, which ASTs **one file**
(`curriculum_dsl.py`, via `_module_source()` default). The live vehicle's levers live in
`spec_tr1_renderer_20260728.py` — **32 factories** — which that call cannot see. This is
**by design and documented** (`lever_registry.py:111-125`): widening `lever_factories()` in place
would report every tr1 flag as stale, *"swapping a vacuous PASS for a false FAIL."*

So the honest statement is: **the default `completeness()` call is scoped to a retired vehicle,
and its reassuring 81.9% describes that vehicle — not a defect in the DSL's TR1 coverage.**
`unmapped` (80) and `stale` (3) are dead-vehicle populations and must be typed **separately from
never-fired**; they are not the same set and have been conflated in casual use.

### 2.3 The launch-provenance surface has zero live-vehicle rows

**0 of 169** launch artifacts belong to the shipped vehicle. The ledger's ground truth of record
(`launch.sh`, per the canonical equation above) therefore cannot see the live vehicle either.
The live line runs through `tools/launch_tr1_run.py`, whose provenance is a DSL **ticket**, not a
`launch.sh` — so the two provenance conventions do not join. *(Scope note: `find` depth ≤3 over
`experiments/results`; I did not reach deeper nestings or SSD-tier run dirs.)*

---

## §3 — THE AUTHORITATIVE SURFACE, AND THE REAL NEVER-FIRED POPULATION

`package_lever_factories()` resolves **each module's own trainer** and is the correct instrument.
MEASURED: **180 factories · 16 modules · stubs 10 · silent_stubs 2 · label_drift 2.**

**10 factories emit a flag their own trainer does not accept — they are structurally unable to
fire.** This is the true "designed but never fired, with a reason" population:

| factory | module | missing flag | class |
|---|---|---|---|
| `TieLocusEdgeWeighted` | `fh1_adapted_force_levers_20260731` | `--tie-locus-edge-weight` | DESIGNED-STUB |
| `ErfBirthContextCoadapt` | `fh1_adapted_force_levers_20260731` | `--erf-birth-context-weight` | DESIGNED-STUB |
| `MarginSatisficeCap` | `fh1_adapted_force_levers_20260731` | `--margin-weight-fn-satisfice-cap` | DESIGNED-STUB |
| `BirthPlateauKneeConjunct` | `fh1_adapted_force_levers_20260731` | `--knee-requires-birth-plateau` | DESIGNED-STUB |
| `XiAdvectedTokenBase` | `fh1_adapted_force_levers_20260731` | `--token-temporal-mode-xi-advected` | DESIGNED-STUB |
| `Qa80MarginBoundedPhotometric` | `ph3_s10_frontloaded_levers_20260731` | `--photometric-margin-budget-weight` | DESIGNED-STUB |
| `Qa81LaneCarrierComposite` | `ph3_s10_frontloaded_levers_20260731` | `--lane-carrier-composite` | DESIGNED-STUB |
| `Ax1Frame0CarriedWarp` | `ax1_derived_levers_20260730` | `--frame0-carried-warp` | DESIGNED-STUB |
| `WeightNormTelemetryRow` | `constants_telemetry_build_wave_20260715` | `--weight-norm-telemetry` | **silent stub** |
| `IntegerPlaneEmitter` | `curriculum_dsl` | `--integer-plane-emitter-{basis,mode,policy-sha256}` | **silent stub** |

### 3.1 THE MECHANISM: 8 of the 10 are TR1-targeted levers graded against the RETIRED trainer

Round-1 self-attack forced me to reconcile three different factory counts from three instruments
(116 / 180 / 202). Doing so produced the sharpest finding in this memo. **AUTHORITATIVE
reconciliation** (`package_lever_factories()`; the 202 was my own looser ad-hoc AST heuristic and
is discarded; 116 is the narrow one-module surface):

| binding | factories | with missing flags |
|---|---:|---:|
| bound to **retired** `train_witness_realized_through_R_mlx.py` | **149** | **10** |
| bound to **live** `train_tr1_partition_renderer_mlx.py` | **31** | **0** |

**Two things follow, and the second is the finding.**

1. **The live vehicle's DSL is healthy: all 31 TR1 factories are built and fireable TODAY**, none
   missing a flag. This *corrects* the impression left by §2.2 — TR1's problem is not coverage.
2. **`spec_tr1_renderer_20260728.py` is the ONLY module in the entire package that declares
   `TRAINER_RELPATH`** (MEASURED: `grep -l TRAINER_RELPATH src/tac/witness_dsl/*.py` → that file
   plus the registry that defines the regex). Every other lever module therefore **silently binds
   to the retired levelset trainer**, including:
   * `fh1_adapted_force_levers_20260731.py` — whose entire charter is *"adapted forms for the
     CURRENT TR1 vehicle"* (5 factories),
   * `ph3_s10_frontloaded_levers_20260731.py` (2), `ax1_derived_levers_20260730.py` (1).

**So 8 of the 10 "cannot fire" factories are TR1-targeted levers whose flags are being checked
against a trainer they were never written for.** Their grades are right only by accident (no
trainer has those flags), but the binding is wrong — and the consequence is exactly `gd1`'s
unexplained meta-finding: **the levers designed for the live vehicle are filed against the dead
one, so no TR1-scoped query can ever surface them for drainage.** A queue cannot drain what is
filed under the wrong vehicle.

> **Evidence grading of that "8", forced by Round-2 self-attack (do not let it be read as
> uniform).** **7 VERIFIED**: the 5 `fh1` factories (its charter states TR1 explicitly) and the 2
> `ph3_s10` factories (`Qa80`/`Qa81` are TR1 ledger rows QA80/QA81, the former with a field
> MEASURED at n600). **1 INFERRED, not verified**: `Ax1Frame0CarriedWarp` — I inferred TR1 intent
> because its *sibling* module `ax1_pool_a_levers_20260730` is imported by the TR1 trainer
> (`train_tr1_partition_renderer_mlx.py:771`), but the mis-bound module is the *different* file
> `ax1_derived_levers_20260730.py`, which I did not confirm. **Read the claim as 7 verified + 1
> inferred.** The `TRAINER_RELPATH` fix is correct regardless — a module that does not declare
> its trainer should refuse, whichever trainer it turns out to want.

**Fix (named, owned, deliberately NOT forced through here):** one line per module —
`TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"` — plus a structural
self-protect so a lever module that does not declare its trainer **refuses** rather than silently
defaulting (a silent default is the orphan generator, per the standing "'Off' is a tracked queue"
law). I did not land this: `src/tac/witness_dsl/` is hot with live arms and MLX-gated tests
contend with the live chain (the `ddm_cu1` lesson — an accurate boundary beats a forced commit).
**Fire-condition:** next quiet boundary in `src/tac/witness_dsl/`.

Two further observations that change how these should be read:

* **The 5 `fh1` forces are the exact levers the registry-repair comment named** as the
  grade-(1) DESIGNED-STUBs that *"could never surface: the registry could not see the file they
  live in."* The repair made them **visible**; nobody then **built** them. Visibility was
  mistaken for closure.
* **`WeightNormTelemetryRow` is observability, and observability defaults ON by law.** Under
  the "'Off' is a tracked queue" non-negotiable, a read-only telemetry row that cannot change
  weights/bytes/`d_seg`/`d_pose` has **no safety reason to be gated**. It is the cheapest row in
  this memo: byte-identity is preserved by construction, so it needs no A/B, only a build.

---

## §4 — RE-ADJUDICATION ARITHMETIC (re-derived here, not re-typed)

All values below recomputed this session from the two anchors. **Self-consistency verified:
`cx1` seg+pose+rate = 0.8264972 = S exactly**, and both rate terms reproduce from their byte
counts (`25·353808/DEN = 0.2355862`; `25·191052/DEN = 0.1272137`).

| quantity | value | status |
|---|---:|---|
| `cx1` (live best) `S` | **0.8264972** | MEASURED n600 (prior arm), archive `1d3ab694`, 353,808 B |
| PR130 floor `S` | 0.172141 | the BAR, 191,052 B |
| **gap** | **0.6543562** | RE-DERIVED |
| seg gap | 0.4015190 = **61.36%** | RE-DERIVED |
| pose gap | 0.1444640 = **22.08%** | RE-DERIVED |
| rate gap | 0.1083722 = **16.56%** | RE-DERIVED |
| `dS/dB` | 6.6586e-07 /byte | `25/DEN` |
| **1% of gap** | **9,827.2 B** | RE-DERIVED — confirms the charter; the older **10,907 B is `dc1_fold`-era and 11.0% too large** |
| **1% of gap in flips** | **1,930 flips** | RE-DERIVED (new unit) |
| `W` (seg↔rate) | **1.2731082153320312** B/flip | **DERIVED**, not measured (`4·DEN/PX`; only `DEN` is measured) |
| one flip | 1.273108 B = 8.4771e-07 S | RE-DERIVED |
| pose marginal `dS/d(d_pose)` | **31.3024** at `cx1` | RE-DERIVED (`5/pose_contribution`) |

**The pose re-pricing rule (the single highest-yield line in this memo).** The pose leg is
concave, so its marginal *rises* as pose improves: it is up **1.73×** since `pw1`. **Any pose
lever ever rejected for missing its bar by less than 1.73× is now potentially live**, with no new
measurement required to re-rank it — only arithmetic. Nobody has swept that direction.

**The seg re-pricing rule.** `W` is **exactly invariant** (pure constants, no archive-size term),
so any lever rejected as "the bytes cost too much" re-adjudicates in one step and **its answer has
not changed with the operating point**. A seg lever is worth its bytes iff it removes
> `bytes/1.273108` flips.

**Seg is where the gap is, and no post-base surface has moved it.** seg gap is **constant at
0.4015190 across `v4d → pw1 → ms8 → dc1_fold → pj2 → cx1`** — its *share* rose 50.7% → 61.36%
only because all six wins were pose or rate.

---

## §5 — VERDICT CLASSES USED IN THIS TABLE

* **NEVER-FIRED** — designed and/or built, no measurement exists. Ranked by expected value.
* **VERDICTED-AT-A-STALE-POINT** — has a number, filed against a superseded baseline.
* **HONESTLY CLOSED** — verdict survives re-pricing. A confirmed closure is a first-class result.
* **JUSTIFIED-BY-AN-UNLICENSED-IDENTITY** *(new, from `ddm_de1`)* — the lever's *reason* cites
  "the head IS a Morse–Smale complex ⇒ persistence/separatrix-flow applies". Power/Laguerre and
  tropical **are** exact at the terminal-feature head; **Morse–Smale is not an identity** (needs
  extra potential/flow hypotheses; none found in a 7,387/9,706 corpus slice). This does **not**
  refute such a lever — it means the lever needs a real reason or a measurement.
* **NAME-ASSERTS-MECHANISM-BODY-LACKS** — NO-FAKE class. Confirmed instance:
  `src/tac/boundary_math/contour_codec.py` claims an explicit boundary-**edge** codec but
  serializes every uint8 label in raster order and LZMA-compresses the dense array — no 1-D wire
  representation, none of the stated edge+region mechanism. **A dedicated class sweep for further
  instances is IN FLIGHT and had not returned when this memo was written — see NEXT-IF-RESUMED.
  This memo therefore carries ONE confirmed instance and makes NO claim about how many others
  exist.**

**A dominated row is a SPECIFICATION, not a kill** — every dominated row below carries the byte
or flip target that would revive it.

---

## §6 — WHAT SISTER ARMS ALREADY SETTLED (credited, NOT re-derived)

I read four adjacent memos in full before ranking anything. Three of them already re-adjudicated
old verdicts against a moved operating point. **Do not redo these:**

* **`ddm_ja1` (ledger QA73), FIRED 07-31 — the sensitivity atlas is ALREADY re-anchored** at the
  live v4c/v4d base; prior bundles were stale at dead bases. Its result **constrains my ranking
  and I have honored it**: *"seg is the LARGEST axis (0.431) but its BYTE pool is
  MEASURED-SATURATED at the cell_drop50 knee… every cheap LIVE byte lever is POSE.
  **Biggest-axis-first is measured-wrong.**"*
* **ledger QA41** — re-measured accept/reject rows off the pose-EXPLODE base: accept count
  28→28 but **the SET CHANGED** (3 rej→acc, 3 acc→rej). The re-measured object is the
  ACCEPTED-SET, not the law.
* **`ddm_mt1` (08-02)** — re-checked all 8 already-measured rows under the split rule and reports
  a **negative**: no already-measured verdict is overturned by the split. Its live-best anchor is
  `dc1_fold`; `cx1` (this memo's anchor) supersedes it.
* **`ddm_gd1` (07-31)** — states the meta-finding my §2 measures the mechanism for:
  *"a lever can be correctly registered, correctly typed, and still never fire… **The gap is that
  nothing forces a never-fired high-blast-radius row to be drained.**"*

**My increment over `gd1` is precise and is the point of this arm:** `gd1` says nothing *forces*
the drain. I measured *why the forcing surface cannot even see what to drain* — the ledger is
7 days stale with 20.6% coverage and zero automatic writers; the default coverage query is scoped
to a retired vehicle; the live vehicle has zero launch-provenance rows. **A queue cannot drain a
lever it cannot see.**

### 6.1 The cross-store contradiction: "never-fired" is not reliable as a label

Composing the sister survey against my instrument measurement produces a hard, checkable defect.
**Four levers are labelled never-fired / no-occupancy in `ddm_mt1` (08-02) while the deferral
ledger records them as FIRED — in two cases as fired-and-LOST:**

| lever | `ddm_mt1` §3.4 | deferral ledger |
|---|---|---|
| `token_delta_group_sparsity` / `delta_sparsity_engage` | "never-fired… none ever swept on this vehicle" | **QA89 FIRED-MEASURED, `worse_s`** (INSTANCE-closed) |
| `token_quant_margin_coupling` | never-fired | QA89 margin_quant **endpoint measured** (+0.0015 d_seg) |
| `margin_weighted_loss` | never-fired | QA24/bc1 §3.2 **landed + FIRED** |
| `distill_form{kd_logits,margin_field,argmax_ce}` | no-occupancy | **QA75 dw1 FIRED-MEASURED → CLOSED at FORMULATION** |

Neither memo is careless — **they consulted different stores, and no store joins them.** This is
the same genus as §2 and it has a real cost in both directions: a fired-and-lost lever re-listed
as never-fired invites a wasted run; a never-fired lever mislabelled fired stays orphaned forever.
**Verdict: any "never-fired" claim must cite the store it was read from.** This memo's own
never-fired rows (§3) cite `package_lever_factories()`, which is structural (a missing trainer
flag) and therefore not subject to this defect.

### 6.2 The drain failure, instantiated

`ddm_fh1` (07-31) ranked **`--class-weight-lane` (R6) as rank 1, *"the cheapest entrant in the
whole table"*, $0**, explicitly flagging it *"PRESENT-NEVER-FIRED (default 1.0)… the
default-off-is-orphaned-signal case, now with a derived value to race."* **Three days later it is
still never-fired.** That is `gd1`'s meta-finding happening in real time, on the cheapest row any
arm has nominated. It is in §7 for that reason, not because I discovered it.

### 6.3 NAME-ASSERTS-MECHANISM-BODY-LACKS — class sweep result

Dedicated sweep, **5 defensible hits**. The sweep explicitly **refused to pad to 10**, reporting
that this codebase has unusually strong anti-fake hygiene: most mechanism-named modules opened
(curvelet/shearlet frames, power-diagram, persistence, KKT active-set, column-generation LP,
wavelet codec, Mamba-2, NCDE, Lanczos curvature, context-model range codec) **genuinely implement
what they name**, often with an explicit swap-test certificate. That is itself a finding, and a
good one.

**Denominator (scoped, never existential):** 715 `.py` files machine-scanned across six
directories (`boundary_math` 86 · `optimization` 328 · `witness_dsl` 171 · `v2_compose` 9 ·
`witness_control` 99 · `torch_vehicle` 22) plus a name-regex/compressor-import sweep over 459
top-level `src/tac/*.py`; **~60 bodies actually read.** **Not reached:** all of `v2_compose`
bodies, 96/99 `witness_control` (notably the `g111_*`/`g120_*` verdict-controller family), 20/22
`torch_vehicle`, 451/459 top-level, and the `substrates`/`packet_compiler`/`codec`/
`canonical_equations`/`information_geometry` trees. **No claim is made about the unread bodies.**

| # | module | claim | what the body does | severity |
|---|---|---|---|---|
| **1** | `src/tac/optimization/inverse_steganalysis_operation_set_compiler.py:222,226` | emits `"chosen_operation_sequence_is_permutation": True` and `"byte_closed_operation_count": len(operations)` | **asserts both properties without checking either** — no byte accounting anywhere in the 241-line module | **LOAD-BEARING** |
| 2 | `src/tac/stc_boundary_codec.py` | name asserts **STC** = syndrome-trellis coding (fixed meaning in this repo) | raster gap-delta coding of boundary positions + generic arithmetic container; `grep -ciE "trellis\|viterbi\|syndrome\|parity"` = **0** | LOAD-BEARING (name only; docstring is honest) |
| 3 | `src/tac/optimization/math_optimal_joint_solver.py:768-899` | *"the KKT/water-fill solve"* | 5-deep nested `for` + `min(grid, key=…)`; the cited KKT solver `run_desk_calc` is **never called** — NO-FAKE #6 | MODERATE (partly self-disclosed) |
| 4 | `src/tac/witness_dsl/bregman_dual_metric_guard.py:39-128` | enforces *"a typed `H^-1` linear solve"*, `solve_elided must be false` | string-compares caller labels; no Bregman divergence, no Hessian, no solve — its canonical binding **passes its own validator by construction** | MODERATE |
| 5 | `src/tac/optimization/postdecode_selector_waterfill.py:227-238` | `"pairwise_local_score_waterfill"` | unconstrained per-pair `argmin` — no budget, no water level, no shared multiplier | COSMETIC |

**Why HIT 1 is the one to escalate, and it is not mine to adjudicate.** Its sibling producer
`byte_shaving_campaign.py:1504-1511` performs the **real** byte-closed check, and
`:1346` computes the permutation property for real. **Both producers' rows land in the same
`packet_ir_operation_sets` list and are SUMMED at `byte_shaving_campaign.py:2382-2384` into
`packet_ir_byte_closed_operation_count`** — a readiness figure — and the permutation flag
propagates into candidate rows (`src/tac/optimizer/candidate_queue.py:1581,1632`). So an
unchecked `True` is being added to a total that a reader will take as checked. Five non-test
production importers. **This is NO-FAKE forbidden class 1 (returns-canonical-markers-without-
doing-work) and class 4 (a declared value in a canonical data field).** Flagged, not fixed:
adjudication and the two-landing fix belong to MAIN.

**Honest scope limit:** hits 2–5 are *naming* defects of varying seriousness; only hit 1
corrupts a number. I did not verify hits 2–5 independently at source — they are
**VERIFIED_VIA_SOURCE_INSPECTION by the sweep, INFERRED by me.**

---

## §7 — TOP 3 FIREABLE, RANKED, WITH THE EXACT NEXT COMMAND

Ranked by `(cost to fire) × (blast radius) × (whether the fire-condition is already met)`.
**None requires the scorer slot** (`ddm_pu1` holds it) except where noted as a queued gate.

### #0 — the `TRAINER_RELPATH` binding fix — $0, one line per module, un-orphans 8 designed levers

Promoted to the top by Round-1 self-attack (§3.1). It is the **cheapest row in this memo and the
only one that fixes a cause rather than a symptom**: three lever modules written for the live
vehicle are graded against the retired one because they do not declare their trainer, and
`spec_tr1_renderer_20260728.py` is the only module in the package that does. Until this is fixed,
every TR1-scoped lever query under-reports by 8, and `gd1`'s "nothing forces the drain" has a
concrete mechanism nobody had named.

* **Cost:** one line per module. **No scorer. No training run. No byte effect.**
* **Blast radius:** re-grades 8 factories and makes them visible to every TR1-scoped query,
  including the costate duty-to-measure queue.
* **Self-protect (the second landing, per the two-landing law):** refuse a lever module that does
  not declare `TRAINER_RELPATH` instead of silently defaulting.
* **Next command:**
  ```bash
  # after adding TRAINER_RELPATH to fh1_/ph3_s10_/ax1_ modules:
  .venv/bin/python -c "from tac.witness_dsl.lever_registry import build_completeness as b; print(b())"
  # expect: the 8 TR1-targeted factories re-bind; their missing_flags re-computed vs TR1
  ```
* **Fire-condition:** next quiet boundary in `src/tac/witness_dsl/` (hot with live arms now).

### #1 — `--class-weight-lane` — a $0-to-nominate lever that a sister arm already ranked #1 and nobody fired

* **Class:** NEVER-FIRED (drain failure, not a discovery).
* **Status:** flag EXISTS on the live vehicle; DSL lever EXISTS (`spec_tr1_renderer:80`);
  **default 1.0 = never fired.** Derived race value already produced by `fh1` R6.
* **Why it is #1:** it is the only row in this memo that is simultaneously (a) live-vehicle,
  (b) already built end-to-end, (c) already carries a derived value, and (d) already nominated
  #1 by an independent arm. Its cost is one training window, not a build.
* **Honest caveat:** it targets seg, and `ja1` measured seg's *byte* pool saturated. This lever
  spends **no bytes** — it re-weights a loss — so the saturation finding does not close it. State
  that explicitly when firing; do not re-import "seg is biggest, attack seg."
* **Next command (nominate into the live DSL ticket, then fire under the governed launcher):**
  ```bash
  .venv/bin/python tools/launch_tr1_run.py --dry-run --ticket <new> --out-dir <SSD>
  # with the compiled ticket carrying spec_tr1_renderer's class-weight-lane lever at fh1's
  # derived value (never hand-add the flag: compile it through the DSL, per never-invent-flags)
  ```

### #2 — `ot_head_offsets_288`: re-adjudicate a BUILT-UNWIRED real solver against the right rival

* **Class:** VERDICTED-AT-A-STALE-POINT **and** BUILT-BUT-UNWIRED.
* **What is built:** `damped_newton_ot_offsets` (`src/tac/boundary_math/laguerre_logit_offset.py:245`)
  — a genuine Kitagawa–Mérigot–Thibert damped-Newton semi-discrete OT solve for the zero-sum
  per-class offset `b*`, whose hard `τ→0` limit **is** the Aurenhammer power-diagram weight.
  It is a **real Newton solve, not a candidate search** — it passes the NO-FAKE #6 test, which
  most things named "solver" in this repo do not.
* **The stale verdict:** `spec_c2_surgical_20260716.py:126-133` records
  `SLOT_BUILT_UNWIRED` with cite *"ot_newton mode MEASURED-worse at mod32cap ep650…
  flip_median advisory arbiter is what this config consumes instead."*
* **Why that verdict does not transfer — MEASURED:** it was filed on the **dead witness vehicle**
  against **two rivals the live vehicle does not have.** `grep` of
  `experiments/train_tr1_partition_renderer_mlx.py` for
  `head_offset|solve_head_offsets|class_offset|logit_offset` returns **empty**. On TR1 the
  comparison is **OT-Newton vs NOTHING**, which is a different question from OT-Newton vs Menon.
* **Identity licence:** power/Laguerre **is** exact at the terminal-feature head (per `ddm_de1`),
  so this lever is NOT in the JUSTIFIED-BY-AN-UNLICENSED-IDENTITY class. It does not depend on
  the Morse–Smale claim.
* **Rule-118 status — OWED, and I flag it rather than assert it:** `spec_c2`'s own unlock says
  *"byte-free decode-time"*. Per-class target masses fitted to THIS clip are video-derived and
  would be COUNTED; 5 classes zero-sum = 4 scalars (tens of bytes), so even counted it is cheap —
  but **"byte-free" is the spec's claim, not a measurement, and must be closed before any rate
  claim.**
* **Next command ($0, scorer-free, uses the existing probe):**
  ```bash
  .venv/bin/python experiments/probe_laguerre_logit_offset_sweep.py --help   # confirm arg surface
  # then solve b* on the LIVE cx1 decoded field via
  # tac.boundary_math.laguerre_logit_offset.solve_head_offsets(mode="ot_newton", ...)
  ```
  **Queued scorer gate (do NOT take the slot):** the realized-through-R n600 A/B of
  `b*`-folded vs unfolded argmax on the `cx1` archive (`1d3ab694`, 353,808 B).

### #3 — `XiAdvectedTokenBase`: a designed-stub whose $0 gate is already OPEN

* **Class:** NEVER-FIRED, with a fire-condition that is **already met**.
* **Status:** the live `--token-temporal-mode` has choices `("shared_base","independent")` where
  the trainer's own help says `shared_base = identity-xi advection` — i.e. **ξ = identity, no real
  advection**. The stub asks for a third choice doing **real** per-pair screw advection.
  It is one of the 8 mis-bound levers in §3.1, which is *why* no TR1 query has surfaced it.
* **Why now:** its $0 gate exists and is open — **ledger QA90**, *"OPEN — $0, fires at any quiet
  boundary"* (the temporal-coherence read of the delta stream). ξ is the dual-use ego-screw, so
  the deep-math licence is Chasles/se(3), **not** the unlicensed Morse–Smale identity.
* **Falsifier, pre-registered by the stub itself:** QA90 shows no coherent advected delta
  structure, or the raced arm is worse on `(d_seg, smevr bytes)` jointly.
* **Blocker named by the stub, not by me:** *"decode side must mirror the warp (byte-close plan
  required before fire)."* Do not fire the training arm before that plan exists.

**Demoted from this list by Round-1 self-attack — `WeightNormTelemetryRow`.** I had it at #3 on
the correct standing-law argument that read-only observability defaults ON and is byte-identical
by construction. **But it binds to the RETIRED trainer** (`constants_telemetry_build_wave_20260715`
declares no `TRAINER_RELPATH`), so building it as-is would instrument a vehicle we do not ship.
It becomes cheap and correct **only after #0**. Recording the demotion rather than deleting it,
because the reasoning was right and only the binding was wrong.

**Runner-up, named so it is not lost:** `XiAdvectedTokenBase` (`fh1` R2, rank 6) — the live
`--token-temporal-mode` has choices `("shared_base","independent")` where `shared_base` is
*identity*-ξ advection; the stub asks for a third choice doing **real** per-pair screw advection.
Its $0 gate already exists and is open (**ledger QA90**, *"OPEN — $0, fires at any quiet
boundary"*), it carries its own falsifier, and ξ is the dual-use ego-screw. Blocker named by the
stub itself: *"decode side must mirror the warp (byte-close plan required before fire)."*

---

## §8 — RE-ADJUDICATION TABLE

`re-priced` uses §4: `W = 1.2731082153320312` B/flip · 1% of gap = **9,827.2 B = 1,930 flips** ·
pose marginal **31.3024** at `cx1` (**up 1.73× since `pw1` — every pose lever is under-priced by
up to that factor**).

| lever | v-lineage | build grade | verdict as filed | filed against | re-priced vs `cx1` | revival spec |
|---|---|---|---|---|---|---|
| `--class-weight-lane` | TR1 (live) | BUILT, flag+DSL | PRESENT-**NEVER-FIRED** (default 1.0) | fh1 07-31, burn ep399 | **LIVE** — 0 bytes, so `ja1` byte-saturation does not close it | fire at fh1's derived value; 1 window |
| `ot_head_offsets_288` | v-witness → TR1 | **BUILT-UNWIRED** (real Newton solve) | "ot_newton MEASURED-worse" | **mod32cap ep650 (dead vehicle)**, vs Menon/flip_median | **RE-OPENED** — live rival is NOTHING (grep-verified) | $0 solve on cx1 field; then n600 A/B |
| `WeightNormTelemetryRow` | v9 constants wave | **silent stub**, **mis-bound to retired trainer** | — (never surfaced) | — | observability logic correct, **binding wrong** | free **after #0**; then build flag, no A/B |
| `XiAdvectedTokenBase` | v9 Force-1 → TR1 | **DESIGNED-STUB**, **mis-bound** | never-fired, QA90-gated | fh1 07-31 | gate **OPEN** ($0, quiet boundary) | QA90 read; then byte-close plan |
| `TieLocusEdgeWeighted` | fh1 R1+R4 | DESIGNED-STUB | rank 3, "the one genuinely new LOSS term" | fh1 07-31 | unbuilt; needs trainer flag | build `--tie-locus-edge-weight` |
| `MarginSatisficeCap` | fh1 R3 | DESIGNED-STUB | rank 4, SMALL | fh1 07-31 | unbuilt | build flag |
| `BirthPlateauKneeConjunct` | fh1 R8 | DESIGNED-STUB | rank 5, rides P2/tp1 | fh1 07-31 | **blocked** — tp1 telemetry port owed | port tp1 first |
| `ErfBirthContextCoadapt` | fh1 R14 | DESIGNED-STUB | "speculative rung, post-window-1" | fh1 07-31 | honestly deferred | birth telemetry P2 |
| `Qa80MarginBoundedPhotometric` | ph3 s10 | DESIGNED-STUB | producer BUILT, field MEASURED, **consumers OPEN** | ledger QA80 | consumer still open | build consumer |
| `Qa81LaneCarrierComposite` | ph3 s10 | DESIGNED-STUB | **TYPED BLOCKER** (held on parallel WIP) | ledger QA81 | blocker stands | unblock WIP |
| `Ax1Frame0CarriedWarp` | ax1 07-30 | DESIGNED-STUB | never surfaced | — | unbuilt | build `--frame0-carried-warp` |
| `IntegerPlaneEmitter` | curriculum_dsl | **silent stub** (3 missing flags) | — | — | dead-vehicle module | close or build |
| the **139** valid retired-trainer factories | v2–v10 witness | BUILT + valid | various | witness vehicle (retired) | **HONESTLY CLOSED by vehicle retirement** | n/a unless vehicle returns |
| the **31** live TR1 factories | TR1 | **BUILT, 0 missing flags — fireable today** | mostly never-swept (`mt1` §3.4) | TR1 | **the live opportunity surface** | 1 training window each |

**Two honest closures, both first-class results.**
* **139 factories are closed by vehicle retirement.** The great majority of the designed-lever
  corpus is bound to a vehicle we no longer ship. That is not ore to re-mine — it is a surface
  that can **stop being swept**, which is worth more than another inventory of it.
* **All 31 live-vehicle factories are already built and valid.** The live DSL is not the problem.
  The problem is that most have never been swept on this vehicle (`mt1` §3.4: *"every one shipped
  at its control value; none ever swept on this vehicle"*), and 8 more that were *written* for
  this vehicle are filed under the retired one (§3.1).

---

## §9 — ADVERSARIAL REVIEW (3 clean passes required; counter resets on any finding)

**Counter: 1 clean pass achieved (R3). R1 and R2 each produced findings and reset it.** This memo
is therefore **PROVISIONAL on the review axis** and says so rather than claiming a seal.

### Round 1 — FINDING (counter → 0)
**Three mutually inconsistent factory counts from three instruments (116 / 180 / 202), all mine.**
Root cause: I mixed the narrow one-module surface (`lever_factories()` = 116), the authoritative
package surface (`package_lever_factories()` = 180), and **my own ad-hoc AST heuristic (202)**,
which was looser than the authoritative one. **Fix:** discarded the 202 entirely; rebuilt §3 on the
authoritative surface only. **This is what produced §3.1** — reconciling the counts is what exposed
the `TRAINER_RELPATH` binding defect. *The most valuable finding in this memo came out of a
self-attack on my own arithmetic, not out of the sweep.*

### Round 2 — FINDING (counter → 0)
**My "8 of 10 are TR1-targeted" was uniformly asserted but non-uniformly evidenced.** 7 are
verified at source; 1 (`Ax1Frame0CarriedWarp`) was inferred from a *sibling* module's import.
**Fix:** the blockquote in §3.1 now grades them 7 VERIFIED + 1 INFERRED. Also corrected in the
same round: my original §7 #3 (`WeightNormTelemetryRow`) was ranked on a correct standing-law
argument but a wrong binding — demoted, with the demotion recorded rather than deleted.

### Round 3 — CLEAN (1 of 3)
Re-derived every load-bearing number from primaries; no new finding. Checks passed: `cx1`
components sum exactly to `S`; both rate terms reproduce from byte counts; `W` recomputed from
constants; 1%-of-gap = 9,827.2 B (and the inherited 10,907 B confirmed `dc1_fold`-era and 11.0%
too large); ledger row/lever/date counts re-read from the JSONL; `--class-weight-lane default=1.0`
verified directly at `train_tr1_partition_renderer_mlx.py:1933` rather than taken from `fh1`.

### Assumption-challenge (required each round)

| # | assumption this work operates within | verification status | if violated |
|---|---|---|---|
| 1 | `cx1` is the live best and the right denominator | **VERIFIED_VIA_EMPIRICAL_ANCHOR** (n600, archive `1d3ab694`) — but it moved 6× in a day | every ΔS in §4/§8 re-prices; the *method* survives, the *numbers* do not |
| 2 | `package_lever_factories()` is itself honest | **VERIFIED_VIA_SOURCE_INSPECTION** (per-module trainer resolution read in full) | if it too has a scope bug, §3 inherits it — this is the single largest structural risk in the memo |
| 3 | a "designed-stub" is worth building | **ASSUMED_AWAITING_VERIFICATION** | a stub may have been abandoned for an unrecorded good reason; §7 rows are nominations, **not** endorsements |
| 4 | levers transfer across vehicles when they act on the frozen scorer | **INFERRED_FROM_DOMAIN_LITERATURE** (power/Laguerre exact at the terminal head; Morse–Smale **not** licensed) | §7 #2's re-opening weakens; its "vs NOTHING" argument survives independently |
| 5 | the corpus stores I queried are representative | **PARTIAL** — `corpus_query` indexes ~76% (7,398/9,706); `find` reached depth ≤3 | rows may exist outside; every negative here is scoped, none is existential |

**The strongest challenge I could not close:** *is a never-fired lever actually signal, or is
"never fired" often a correct silent verdict?* `ja1` already measured one version of this —
biggest-axis-first is measured-wrong — and `mt1` found that re-checking 8 measured rows under a
new rule overturned **none**. So the base rate of "old verdict flips" in this campaign is **low**,
and §7 is deliberately ranked by **cost-to-fire**, not by hoped-for effect. **#0 is the only row
whose value does not depend on that base rate at all.**

---

## NEXT-IF-RESUMED

**State:** §1–§9 complete. Review counter **1 of 3 clean** — memo is PROVISIONAL on that axis.

1. **FIRE #0** (`TRAINER_RELPATH` in `fh1_`/`ph3_s10_`/`ax1_derived_` + the refuse-if-undeclared
   self-protect) at the next quiet boundary in `src/tac/witness_dsl/`. $0, no scorer, no bytes.
   This is the highest-value row in the memo and the only one whose value does not depend on the
   low base rate of verdict-flips (§9).
2. **ESCALATE §6.3 HIT 1 to MAIN** — `inverse_steganalysis_operation_set_compiler.py:222,226`
   asserts `byte_closed`/`permutation` without checking, and the value is SUMMED with a sibling
   producer's genuinely-checked rows into a readiness figure. NO-FAKE classes 1 + 4, five
   production importers. Needs adjudication + the canonical two-landing fix (fix + STRICT gate).
   **Not mine to adjudicate; not fixed here.**
3. **Complete review rounds 2 and 3** to seal (currently 1 clean).
4. **Un-taken, and I name it rather than implying coverage:** I did **not** sweep the 149
   retired-trainer factories individually for transfer to TR1 — I closed them as a population by
   vehicle retirement (§8). If that closure is ever doubted, the per-factory sweep is the work.
   Likewise the `witness_control/g111_*` verdict family is where §6.3 says to look next; it read
   none of them.
5. **Apparatus debt surfaced here, owned, NOT fixed:** (a) activation-ledger fix R1 (argv→lever
   reverse-map) still unwired four weeks after its canonical equation named it; (b)
   `record_activation` has zero automatic callers; (c) the live vehicle has no `launch.sh`
   provenance join, so ledger and launcher conventions do not meet.
6. **Do NOT re-run:** the unmerged-branch/worktree harvest (07-28), `ddm_cu1` 155-file
   disposition (08-03), `ddm_ja1` sensitivity-atlas re-anchoring (QA73, 07-31), `ddm_mt1`'s
   8-row re-check (08-02). All four are credited in §6 and none needs redoing.
