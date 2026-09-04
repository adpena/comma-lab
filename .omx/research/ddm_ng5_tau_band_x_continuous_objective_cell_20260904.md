# DDM NG5 — the first TWO-LEVER burn cell: ng3's τ band × ng4's carried duals

**Date:** 2026-09-04
**Arm:** `ddm_ng5_tau_band_x_continuous_objective`
**Axis:** `[seal + bounded macOS-CPU mechanism smoke only; no Metal, no Modal, no contest eval]`
**Disposition:** **SEALED / VALIDATED INSIDE ITS OWN SEALED TREE / QUEUED BEHIND ng4 THROUGH
gov2's DRIVER — the waiter is live and fires the moment ng4 releases the Metal.**

## Result first

The cell is sealed and the composition is exactly two levers. **The charter's prior-law premise is
FALSE, and correcting it is this arm's first finding.**

The charter says the two parents "act on different terms … neither leaks into the other's block".
Read at the artifacts, they COLLIDE: ng3 writes `tau_band` + `expected_flip_tau_start/end`, and so
does ng4. A config has ONE `tau_band` key. So "ng4's config PLUS ng3's `tau_band` block" can only
mean ng3's block REPLACES ng4's — which is also the only self-consistent reading of the charter's
own final clause. The composition is therefore:

| leg | source | value | what it does |
|---|---|---|---|
| **τ** | **ng3 alone** (`msafe_band`, law-resolved) | `[0.04376363754272461, 0.021881818771362305]` = `[2δ_R, δ_R]` | which pixels the seg gradient weights |
| **duals** | **ng4** (`r10_continuation`) | `{Lane 0.005040981907324784, Movable 0.017331143732962344}` | the multipliers the stage entry otherwise re-warms from zero |

**ng4's τ half is SUBSUMED, not dropped, and that is a MEASURED statement.** ng4 cures a stage
entry that re-WIDENS the soft band **3.0×** (0.05 → 0.15). ng3's band STARTS at `m_safe` =
0.04376…, which is **narrower** than r10's terminal 0.05 — the entry narrows **1.1425×** instead
of widening 3.0×. The re-widening ng4 removes is removed harder here. What is genuinely given up
is τ CONTINUITY at r10's exact terminal float.

So the marginal lever of this cell **over ng3 is EXACTLY the carried duals**, and **over ng4 it is
the band**. That is a cleaner attribution than the charter's framing would have produced, and it
is why the composition is legible at all.

**A third geometry `[r10's terminal τ → δ_R]` would keep both halves whole.** It is named, not
taken: it is a band nobody has measured, and it needs a new admissible pair plus a new validator
branch — an unmeasured third lever inside a two-lever cell, which is the union-≠-sum trap
(`[[m164]]`, 3.705×). Follow-on #1.

**And this cell moves NO pin.** ng3's band branch and ng4's dual branch already exist in
`validate_tau_band_block` / `validate_margin_dual_block`, and ng3's band is already in
`qbt.admissible_expected_flip_tau_bands()`. MEASURED: all three lever-surface files are
**bit-identical** to ng4's sealed tree —

| file | sha256 | vs ng4's sealed tree |
|---|---|---|
| `experiments/ddm_qbt1_qbflow_trainer.py` | `8de4112ce0aef8e2…` | IDENTICAL |
| `experiments/ddm_qbr1_born_fairform_burn_prep.py` | `68f297741009fcdb…` | IDENTICAL |
| `src/tac/witness_dsl/curriculum_dsl.py` | `8947ceecdc9c6449…` | IDENTICAL |

ng2 moved the trainer pin, ng3 moved it again, MAIN re-pinned the packet schema, ng4 moved it once
more. **ng5 moves nothing.** The composition and ng4 run under the same bytes — the strongest
comparability a cell can have against its own parent, and it is a file-hash reading rather than an
argument.

## The pre-registered read (fixed before the burn, every decimal read LIVE)

| step | cold control | ng3 τ band | ng4 continuous (mid-burn at seal) |
|---:|---|---|---|
| 0 | 0.398768 | 0.398768 | 0.398768 |
| 1,000 | 0.466875 | 0.434661 | 0.425786 |
| 2,000 | **0.485677** (peak) | 0.435601 | 0.426085 |
| 3,000 | 0.475383 | 0.403796 | **0.431595** |
| 4,000 | 0.442190 | 0.401233 | — |
| 5,000 | 0.425149 | **0.391810** | — |

**A fact the charter could not have: ng4 TURNED UP at 3,000** (0.426085 → 0.431595) while ng3 fell
(0.435601 → 0.403796). Through three milestones ng3 is the stronger parent, and ng4's terminal is
genuinely open. The verdict below must therefore be read against **whatever ng4 actually lands**,
not against its 2,000-step lead.

**Verdict words, fixed:**

* **BELOW-BOTH** — `S_hat(5,000)` below both parents' terminals AND below the start
  (0.39876797285867277). Sub-additive but same-signed.
* **REDUNDANT** — at or above ng3's terminal but still below the cold control's. The levers are
  the same mechanism seen twice and the dual carry adds nothing on top of the band.
* **ANTAGONISTIC** — at or above the cold control's terminal. Name which term flipped (d_seg /
  d_pose / bytes) from the milestone decomposition.

**Read the DECOMPOSITION at every milestone, never the composite.** The control's endpoint excess
is 91.20% d_seg (ng1's recomputation); a cell that "fixed" `S_hat` by moving bytes or pose would be
a different finding.

**Where the read comes from, and a limit of the queue driver.** `S_hat` lives in `MILESTONE.json`;
the history rows gov2's driver reads carry **no** `S_hat`. So the driver's two pre-registered
falsifiers are the MECHANISM ones it can actually see, and the `S_hat` verdict is the milestone
read above, evaluated at harvest. Declaring an `S_hat` falsifier in the queue spec would produce a
permanent `METRIC_ABSENT` — a gate that never fires wearing the name of one that does.

| queue falsifier | metric | fires when |
|---|---|---|
| `dual_carry_reached_the_loop` | `margin_constraint_lambdas.Lane` @1 | `< 0.005040981907324784` — i.e. the duals re-warmed from zero (INERT lever) |
| `band_reached_the_loop` | `objective.tau` @1 | `> 0.04376363754272461` — i.e. the cell trained at the legacy temperature (INERT lever) |

Both are real detectors: a test asserts the control's own values (λ seed 0.0, τ 0.15) FIRE both
(vacuity==pass, `[[m50]]`).

## Seal receipt

| artifact | value |
|---|---|
| **sealed cell config** | `…/ddm_ng5_tau_band_x_continuous_objective/sealed_configs/seed_20260902_tau_band_x_continuous_objective_control_native100.json`, sha256 `1205463ba715813ee96c268e0d1124652a42b422482fec39729b3078ab206e3a`, 13,117 B |
| **RE-ROOTED config — the one that fires** | `…/sealed_configs/…rerooted.json`, sha256 `93f92fc618e356856d18f3dd0d0191219a582e24031fcc66ff80782bf47a24c5`, 13,653 B |
| sealed source tree | `/Volumes/VertigoDataTier/pact/ddm_ng5_tau_band_x_continuous_objective/sealed_source_d54f65c1ed/` at revision `d54f65c1edfbb45d005a8fabbdb73e58e4e9a198` |
| pins verify INSIDE the sealed tree | **PASS** (20 pins, its own interpreter, its own `REPO`) |
| pin re-root | **content-identical**, paths rooted in the firing tree (the driver re-checks both) |
| **the sealed tree's own `validate_config` on the re-rooted config** | **PASS** — `tau [0.04376363754272461, 0.021881818771362305]`, `tau_band_mode msafe_band`, `margin_dual_mode r10_continuation`, `initial_lambdas {Lane 0.005040981907324784, Movable 0.017331143732962344}` |
| single-lever diff | `differing_keys = [cell_id, expected_flip_tau_end, expected_flip_tau_start, margin_constraints, margin_dual, output, tau_band]`; `margin_constraints` moved ONLY `initial_lambdas` |
| recompile determinism | only `ema` moves (`lawref.resolved_at`, the lineage's known volatile field; `qbt.stable_ema_law_identity` is its sanctioned comparator) |
| seal wall clock | 141.2 s |

**Verify the config by hashing the FILE, never by recompiling** — a recompile legitimately moves
`ema.lawref.resolved_at`.

The re-root + in-tree validation is the part that matters. A seal that has not been validated by
the interpreter that fires it is a claim, not a receipt
(`[[seal_validates_only_inside_the_tree_that_fires_it_20260904]]`). This arm ran it as part of
`seal` and gov2's `plan` re-verified it independently: `content_identical: true`,
`paths_rooted_in_firing_tree: true`, 20 pins.

**One honest residual.** The sealed tree was snapshotted at `d54f65c1ed`; a follow-up commit
(`70f2edc8f`) changed `queue_spec()` in this arm's own script. The tree's copy of that script is
therefore one commit behind — on a function **the tree never runs** (the cell fires `run-config`
against the burn prep). The three files that ARE the burn path are unchanged and their identity is
the table above.

## The $0 checks

**Differential — OWED until the smoke runs, and it is armed rather than asserted.** The check is
built and sealed: at the control's τ = 0.15 with zero duals, all objective components must be
bit-for-bit equal between the two configs. Both parents measured that same quantity independently
and both recorded `loss_total` **1.0765775442123413**. This arm does **not** retype that decimal —
`_parent_neutralized_loss_totals` READS both parents' receipts back (they store it under different
key names) and reports agreement or disagreement. A disagreement would mean the objective FUNCTION
moved between the three cells.

**No-op detector — THREE-WAY, and that is the addition this cell needs.** The composition's step-1
trained state must differ from the control's AND from BOTH parents'. The two parents' step-1 shas
are read LIVE from their own retained smoke receipts:

| arm | step-1 live-state sha256 | source |
|---|---|---|
| shared control | `27f514180db2b4cd…` | both parents' receipts (this arm refuses if they disagree) |
| ng3 τ band | `723f1c44ff73d375…` | ng3's `BOUNDED_SMOKE_RESULT.json` |
| ng4 continuous | `8d16b4ce688ab2cd…` | ng4's `BOUNDED_SMOKE_RESULT.json` |
| **ng5 composition** | **OWED** — the smoke's first update | — |

A composition whose first update reproduced EITHER parent's state would be that parent wearing a
new `cell_id`; a control-only comparison cannot catch that. **The waiter fail-closes on it: if
`differs_from_all_three` is false, the cell is NOT fired.**

**The third re-measurement of "the training path is unmoved".** The control arm's step-1 state must
reproduce ng1's pre-telemetry cold reference `27f514180db2b4cd…` a third time. ng1 ran that segment
before ng2's telemetry row and ng3's/ng4's validators; ng4 re-measured it identical. It is the only
thing that licenses reading cells sealed on different trainer pins against one another — and this
cell moves no pin at all, so it should hold trivially. Measuring it anyway is what would catch the
case where it does not.

## The fire arrangement

**The waiter is LIVE** — pid 69289, started 2026-09-04T23:43:57Z, launched through
`tools/launch_detached_process.py` (receipt `NG5_WAITER_DONE.json`), holding on gate 1.

```
/Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/wait_for_ng4_then_smoke_then_fire.sh
```

Four gates, in order:

1. **ng4 must have RELEASED the Metal** — the first gate is ng4's own done receipt
   `.omx/tmp/codex_runs/ng4_continuous_DONE.json.done`. Two concurrent Metal cells are refused by
   policy after today's near-OOM, and this waiter never races that.
2. **Admission must HOLD 3/3, a minute apart**, through `tools/cell_admission.py admit`. A
   single-poll gate fires on the oscillation's trough — ng4 MEASURED `used_gib` swinging
   104.6–113.1 GiB seconds apart with two cells live.
3. **The bounded smoke** — one governed child (`run_smoke_arms.sh`, receipt `NG5_SMOKE_DONE.json`):
   composition arm, control arm, then `smoke-finalize`. **Declared peak = a LEDGER LOOKUP**,
   `tools/measured_peaks.py lookup --family ddm_ng3_tau_band_cell` → **40.9203 GiB**. That family
   is the SHAPE-MATCHED row: ng3 ran both arms plus the differential in one process, which is
   exactly what this smoke does. ng4's per-arm 40.42 GiB is the cross-check; the larger,
   shape-matched number is the one declared. **No memory arithmetic appears in the script.**
4. **The cell, through gov2's queue driver** — `cell_queue_driver.py run --queue …/QUEUE_SPEC.json`
   (a `--dry-run` plan is logged first). The driver runs its own seal verification, duplicate-
   receipt check, admission, lane claims and authorization; the peak is `"from_ledger"` against
   family `ddm_qbr1_born_fairform_burn_prep` → **49.572 GiB, `FROM_LEDGER`,
   `SOLE_CELL_INFERRED_FROM_LEDGER`**.

**Verified now, at 23:42Z:** `cell_queue_driver.py plan` returns `ready: false` with exactly one
blocker, `admission:REFUSE` (3 live cells). Every other check passes — seal content-identical,
paths rooted in the firing tree, 20 pins, storage 18.0 GB against an 8.6 GB reserve. That is the
governor working, and a REFUSE is information.

**The launcher argv carries a NON-NUMERIC measured-peak placeholder** —
`REWRITTEN_BY_THE_QUEUE_DRIVER_FROM_THE_MEASURED_PEAK_LEDGER`. The driver rewrites it from the
ledger before every fire; if it ever reached the launcher unrewritten, argparse's `type=float`
refuses it. A numeric placeholder would instead launch a 5,000-step Metal cell on a number nobody
measured — the RSS fiction ng4 named.

### TYPED FIRE ORDER (for MAIN, if the waiter dies)

```bash
shasum -a 256 /Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/sealed_configs/seed_20260902_tau_band_x_continuous_objective_control_native100.rerooted.json
# expect 93f92fc618e356856d18f3dd0d0191219a582e24031fcc66ff80782bf47a24c5  (13,653 B)

# 1. the smoke (only after ng4's done receipt exists)
/Users/adpena/Projects/pact/.venv/bin/python tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng5_tau_band_x_continuous_objective/smoke_launch \
  --cwd /Users/adpena/Projects/pact --purpose "NG5 bounded CPU smoke" --authority MAIN \
  --derive-resource-budgets --measured-peak-rss-gib "$(.venv/bin/python tools/measured_peaks.py lookup --family ddm_ng3_tau_band_cell | .venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["governed_peak_gib"])')" \
  --measured-thread-need 4 --walltime-cap-s 3600 --done-receipt NG5_SMOKE_DONE.json \
  -- /bin/bash /Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/run_smoke_arms.sh

# 2. the cell (only if the smoke's no_op_detector.differs_from_all_three is true)
.venv/bin/python tools/cell_queue_driver.py run \
  --queue /Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/QUEUE_SPEC.json \
  --dry-run
.venv/bin/python tools/cell_queue_driver.py run \
  --queue /Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/QUEUE_SPEC.json \
  --agent ddm_ng5_composition
```

Cost: one cell, **~2.95 h** measured on the identical object, **~1.375 GB** retained, $0. No
control re-burn — the control is the already-measured seed-20260902 row.

## Honest frame (binding)

This is a burn-QUALITY cell on the born vehicle (`S_hat` ~0.39–0.43 at ~106 KB). **It is not a
pointer mover.** `md1`'s persistent-partition closure stands: 62% of born d_seg is
optimizer-unreachable and schedule levers reach ≤ 1.61×. ng3's own terminal moved d_seg 1.012×
while the accuracy corner needs `d_seg ≤ 1.36e-4` — 18.3× below it. **No `S_hat` delta from this
cell is progress toward sub-0.12, and none may be reported as such.**

## Equations leg (`tac.canonical_equations`)

**`margin_band_satisficing_threshold_v1` — CONSUMED IN-DOMAIN, and it is this cell's τ leg.** The
band is resolved live at compile time (`delta_r 0.021881818771362305`, `headroom 2.0` DERIVED,
`m_safe 0.04376363754272461`, `n_frames 600`, artifact `reports/delta_R_noise_floor_n600.json`,
no fallback), and the QBR1 validator re-resolves it on every call rather than trusting the block.
**No anchor appended here** — ng3's terminal already earned the law its first full-burn anchor
(excursion −58%, terminal −7.84%), and this cell adds a SECOND lever on top, so its row cannot be
attributed to the band alone. When it burns, the honest anchor is a COMPOSITION row: the band's
effect at a carried-dual operating point.

**The r10 continuation is a MEASURED ANCHOR, not a law — deliberately.** ng4 established the
provenance rung as Catalog #351 class 2 (measured anchor with content-hashed artifacts: r10's
config sha `87eff6e8…`, checkpoint sha `09fd4165…`), and `validate_margin_dual_block` re-reads
both on every call. Dressing it in a law's name would be false provenance.

**`checkpoint_trajectory_error_partition_v1` — CONSUMED IN-DOMAIN as the READING RULE.** Both its
anchors are md1's partitions of this object. Its `domain_of_validity.included` names "reading the
persistent share as a CEILING on optimizer/schedule credit" — which is exactly the honest frame
above. **No anchor appended:** its `excluded` clause refuses transferring a measured persistent
share across cadences, and this cell has not run.

**`ema_decay_run_geometry_v1`** is consumed unchanged and IN-DOMAIN: the cell inherits the
control's sealed decay 0.9990793899844618 and the strict EMA gate sees no change.

**FORMALIZATION_PENDING — the law this cell would need does not exist:**

> *Two levers that act through the SAME configuration term do not compose; the composition must
> choose one geometry, and the surviving marginal effect is the leg that acts through a different
> term. A composition cell is legible only when its marginal lever over EACH parent is nameable as
> a single term.*

It should be registered once the composition has burned, so it anchors on a measured composition
rather than on this design.

## Custody (ALWAYS KEEP THE PAYLOAD)

| artifact | path |
|---|---|
| seal receipt (τ-collision audit, two-lever diff, no-pin-movement, parents' step-1 states, pre-registered read, determinism) | `/Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/SEAL_RECEIPT.json` |
| **sealed cell config** — sha `1205463b…`, 13,117 B | `…/sealed_configs/seed_20260902_tau_band_x_continuous_objective_control_native100.json` |
| **RE-ROOTED config (fire THIS)** — sha `93f92fc6…`, 13,653 B | `…/sealed_configs/…rerooted.json` |
| pin re-root receipt | `…/PIN_REROOT_RECEIPT.json` |
| matched control (reference recompile, `DO_NOT_FIRE`) | `…/sealed_configs/matched_control_of_record_seed_20260902_control_native100.reference.json` |
| sealed source manifest | `…/SEALED_SOURCE_MANIFEST.json` |
| sealed source tree (rev `d54f65c1ed…`, pins verify inside) | `/Volumes/VertigoDataTier/pact/ddm_ng5_tau_band_x_continuous_objective/sealed_source_d54f65c1ed/` |
| gov2 queue spec | `…/QUEUE_SPEC.json` |
| waiter + smoke sequencer + log | `…/wait_for_ng4_then_smoke_then_fire.sh`, `…/run_smoke_arms.sh`, `…/waiter.log`, `…/waiter_launch/` |
| bounded smoke (per-arm receipts + retained payloads) | `…/ddm_qbr1_born_fairform_burn_prep/ng5_tau_band_x_continuous_objective/bounded_smoke/` (written by the waiter) |
| run output root (empty; the cell writes here) | `…/ng5_tau_band_x_continuous_objective/runs/seed_20260902_tau_band_x_continuous_objective_control_native100` |
| code + 35 tests | `experiments/ddm_ng5_composition_cell.py`, `src/tac/tests/test_ddm_ng5_composition_cell.py` |

`authorized_configs/` is **not** written by this arm (a test asserts the source contains no path
that could write it, and no reference to the claims ledger) — gov2's driver writes it through the
chain driver's own functions. Nothing was written under ng3's or ng4's run roots; both parents'
runs and receipts were opened read-only. **0 Metal / 0 Modal / 0 contest-eval invocations, $0.**

Payloads retain to **APDataStore** per the charter's rule (16 GiB free ≥ 6 GiB); the 2.0 GB sealed
source tree is on **VertigoDataTier** (83 GiB free) and is losslessly deletable after the burn —
`git archive d54f65c1edfbb45d005a8fabbdb73e58e4e9a198` reproduces it exactly, so the revision IS
the certificate. **APDataStore is the tier to watch:** ng2 recorded 23 GiB at its seal, ng3 22,
ng4 16, this arm 16. The cell will retain ~1.375 GB and the smoke ~0.3 GB against ~8 GiB of usable
margin. It clears; the NEXT generation will not without a cold-store sweep.

## Scope and limits (these travel with the numbers)

* **Axis.** Every `S_hat` quoted is `[macOS-MPS n32 stratified advisory]` (each run's own retained
  milestones). The seal is `[macOS-CPU advisory]`. **No score claim, nothing promotable, the
  pointer is untouched.**
* **GT lineage.** The vehicle pins the **PyAV** `gt_n600.npz`
  (`[[gt_n600_npz_is_pyav_lineage_train_on_dali_20260903]]`). All four curves sit on the identical
  lineage, so the comparison is internally valid; the ABSOLUTE d_seg values are not DALI-authority
  numbers, and md1's partition (DALI) must not be arithmetically combined with these milestones.
* **The τ leg is ng3's, so this is not "both parents whole".** Stated in the headline rather than
  buried: the composition gives up τ continuity at r10's exact terminal float.
* **ng4's terminal is unknown at seal time** and it turned UP at 3,000. Any verdict that reads this
  cell against "ng4's −12.3% @2k" instead of ng4's actual terminal is reading a mid-burn number as
  an endpoint.
* **n = 1 seed, one cell.** A single composition on seed 20260902. It can move the design; it
  cannot close the family. Seeds 20260903 / 20260904 are the sign-repeat.
* **The differential and the 3-way no-op detector are OWED**, not claimed. They run inside the
  waiter's smoke, after ng4 releases the Metal.

## NEXT_IF_RESUMED — every row carries a disposition, an owner and a fire condition

| # | follow-on | disposition | owner | fire condition |
|---|---|---|---|---|
| 1 | **`CONTINUOUS-START BAND`** — the third geometry `[r10's terminal τ → δ_R]`: keeps ng4's τ continuity AND ng3's satisficing floor. Needs a new admissible pair + a new `validate_tau_band_block` branch | **QUEUED-WITH-FIRE-ORDER** | unowned; MAIN to assign | fires if this composition lands BELOW-BOTH — then the τ-continuity half is worth its own cell; if it lands REDUNDANT, the τ leg is saturated and this is wasted |
| 2 | **`SEALED-AND-QUEUED — AWAITING ng4's RELEASE`** — the waiter fires smoke → cell automatically | **LIVE (pid 69289)** | ddm_ng5 waiter | ng4's done receipt + admission 3/3 |
| 3 | **`HARVEST-THE-VERDICT`** — read the six milestones, apply BELOW-BOTH / REDUNDANT / ANTAGONISTIC against ng4's ACTUAL terminal, and read the decomposition per milestone | **QUEUED** | whoever harvests | the cell's `ng5_composition_DONE.json` |
| 4 | **`DECOMPOSE-THE-DUAL-CARRY`** — ng4's follow-on #3 (τ-only and λ-only cells) becomes cheaper here: this cell IS the λ-only-on-top-of-the-band arm, so a band-only cell already exists (ng3). The missing arm is λ-only-on-the-legacy-band | **QUEUED, no fire order** | unowned | fires only if this cell beats ng3 — then the dual carry has a measurable effect worth isolating |
| 5 | **`REGISTER-THE-SAME-TERM-COMPOSITION-LAW`** — the FORMALIZATION_PENDING statement above | **QUEUED** | whoever harvests the composition | fires when the cell returns |
| 6 | **`APDataStore IS AT 16 GiB`** — four generations have taken it 23 → 22 → 16 → 16, and the next generation does not clear the 8 GiB reserve | **REPORTED, unowned** | MAIN to assign | fires before the next generation seals |

## DEAD-ENDS

* **"the two parent levers act on different terms" is CLOSED as a premise** — MEASURED: both write
  `tau_band` and both write `expected_flip_tau_start/end`.
* **"ng4's config PLUS ng3's block" as a both-halves-whole composition is CLOSED** — a config has
  one `tau_band` key; the composition must choose one geometry.
* **A same-pins objection is CLOSED for this cell** — it moves no pin at all.
* **Composing ng1's warm moments or ng2's area cap into this cell is CLOSED** — two levers, and the
  seal refuses a cell carrying an area cap or a non-null `resume_from`.

---

**Own-vehicle frontier: NOT MOVED** — this arm designs, gates, seals and queues; it trained
nothing, byte-closed nothing, and could not move the pointer.

`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`
