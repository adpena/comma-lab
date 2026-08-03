# ddm_lr2 — repairing the lever instruments: they were all bound to a retired vehicle

**Date:** 2026-08-03 · **Arm:** `ddm_lr2` · **Axis:** apparatus repair + one NO-FAKE adjudication.
**Scorer-free, $0** (`ddm_pu2` holds the scorer slot). **Pointer UNMOVED** (`0.1910828242`
[contest-CPU custody]). Nothing here is a score claim. Every number is MEASURED by this arm or
RE-DERIVED from primaries — including the constants my own charter handed me, which I re-derived
rather than inherited (§5).

Predecessor: `.omx/research/ddm_la1_lever_archaeology_20260803.md` (commit `869d975014`). Its
§-numbered rows were my work order; §0 records what reproduced and what did not.

---

## ANSWER FIRST

**All three dead instruments turned out to be ONE bug class, and it is the same class as the
levers themselves: the campaign re-homed to the TR1 vehicle on 2026-07-28 and its instruments
did not.** That is not a metaphor — it is a date.

* The activation ledger's **last write is 2026-07-27T21:17:34Z**. `tools/launch_tr1_run.py`, the
  governed launcher of the vehicle we ship, was **added 2026-07-28**. The ledger did not decay
  from disuse; **it stopped on the day the vehicle changed launchers**, because its only writer
  is `tools/launch_witness_run.py` — the *retired* vehicle's launcher.
* The lever modules written *for* TR1 kept binding to the retired trainer because only one module
  in the package declared `TRAINER_RELPATH`.
* The default coverage query kept describing the retired trainer without saying so.

Four things landed, all $0, all with executed positive controls:

1. **§1 — the re-home.** 3 modules now declare their trainer. Census moved **149/31 → 141/39**
   (MEASURED): the 8 TR1-targeted factories are re-filed under the vehicle we ship. **STRICT gate
   at live count 0** refuses a stub verdict that rests on an undeclared binding.
2. **§2 — the instruments now state their basis or refuse.** `never_fired()`'s 178 is now
   formally declared **VACUOUS with a reason**; `completeness()` carries the vehicle it measured
   (RETIRED 0.819 vs LIVE 0.178) and **fails closed** on unknown; both are wired into
   `tools/costate_digest.py`, the SessionStart consumer, and VERIFIED printing.
3. **§3 — the NO-FAKE escalation, adjudicated and fixed.** Confirmed at source; both fields now
   COMPUTED; **STRICT gate at live count 0** refuses re-introduction.
4. **§4 — the store join.** Measured, and the reason a naive fix does not work (§2.4).

**The finding I did not expect, and it changes what "wire the ledger up" costs (§2.4).** The
live vehicle's levers are constructed with **f-string names parameterised by their own
arguments** — `Lever(name=f"tr1_token_grid_D{downsample}_c{code_width}")`. So the ledger's key
space (factory names) and the launcher's (instance names) are **structurally disjoint: 0 of 43
joinable, MEASURED**. Wiring `record_activation` into the live launcher today would pile rows
under keys nothing joins to and leave `never_fired()` reporting every factory as never-fired
forever — manufacturing a *second* cross-store contradiction while appearing to fix the first.

---

## §0 — VERIFYING la1 BEFORE ACTING ON IT (it asked to be checked; it reproduced)

la1 reached `180 = 149 + 31` by reconciling three of its own inconsistent counts during a
round-1 self-attack, and said so. Re-derived here directly from `package_lever_factories()`:

| claim | la1 | ddm_lr2 | verdict |
|---|---:|---:|---|
| total factories | 180 | **180** | REPRODUCED |
| bound to retired trainer | 149 | **149** | REPRODUCED |
| bound to live TR1 trainer | 31 | **31** | REPRODUCED |
| factories with missing flags | 10 | **10** | REPRODUCED |
| …all on the retired side | yes | **yes** | REPRODUCED |
| of those, TR1-targeted | 8 | **8** | REPRODUCED |
| `spec_tr1_renderer` is the ONLY declarer | yes | **yes** (`grep` over 170 modules) | REPRODUCED |

**Three corrections.**

* **C1 — the re-home does NOT make the 8 fireable.** My charter asked me to "prove the 8 levers
  become fireable". They do not, and claiming it would be the fake. **MEASURED: none of the 8
  flags exists on the TR1 trainer either** (73 flags, checked individually). Re-homing makes
  their debt **correctly attributed and visible to TR1-scoped queries** — which is the actual
  value, because an invisible debt cannot be drained. Pinned by a test that fails if they ever
  silently grade BUILT.
* **C2 — la1's "8 of 10, 7 VERIFIED + 1 INFERRED" is now 8 VERIFIED.** la1 could not confirm
  `ax1_derived_levers_20260730` and honestly graded it INFERRED from a *sibling* module's import.
  Read at source: that module's own **FOLD-AND-DELETE LOG names `spec_tr1_renderer_20260728`
  three times** as the supersession target for its own factories. Its disposition path is TR1.
* **C3 — "`record_activation` has ZERO automatic callers" is wrong, and the true answer is
  sharper.** It has **exactly one** non-test caller: `tools/launch_witness_run.py:3421` — the
  RETIRED launcher. "Zero callers" reads as neglect; **one caller on the dead vehicle** is
  structural, dated, and diagnostic (§2.1). *I nearly published the same false negative myself:
  my first grep returned truncated output and I wrote "ZERO non-test callers" into a checkpoint
  before re-running it explicitly. The instrument lied about the instrument.*

---

## §1 — THE RE-HOME (landed, MEASURED, gated)

### 1.1 What was wrong

`module_trainer_paths()` **silently defaults** to the retired trainer for any module that does
not declare `TRAINER_RELPATH`, and only `spec_tr1_renderer_20260728.py` did. So an *undeclared*
module is not "bound to the levelset trainer" — it is bound to whatever the default happens to
be, and **no reader can distinguish an intentional binding from an author who never considered
the question**. A silent default is an orphan generator, exactly per the standing "'Off' is a
tracked queue, never a forgotten default" law.

Verified at source, each module's own docstring naming its target:

| module | its own words | factories |
|---|---|---:|
| `fh1_adapted_force_levers_20260731` | *"the v8/v9/v10 forces ADAPTED to the TR1 vehicle"* | 5 |
| `ph3_s10_frontloaded_levers_20260731` | *"supersedes each stub … in `spec_tr1_renderer_20260728`"* | 2 |
| `ax1_derived_levers_20260730` | FOLD-AND-DELETE LOG → `spec_tr1_renderer_20260728` ×3 | 1 |

**This is the mechanism behind `ddm_gd1`'s long-unexplained "nothing forces the drain."** A queue
cannot drain a lever filed under the wrong vehicle.

### 1.2 The fix and its MEASURED effect

One line per module. Census, before → after:

| binding | before | after |
|---|---:|---:|
| retired (`train_levelset_witness…` + base) | 149 | **141** |
| live (`train_tr1_partition_renderer_mlx.py`) | 31 | **39** |
| stubs | 10 | **10** (unchanged — re-binding regrades, it does not build) |

### 1.3 The self-protect (second landing), and why its scope is narrow

`check_lever_module_declares_its_trainer` — **STRICT from byte one, live count 0** — refuses a
factory graded a DESIGNED-STUB whose module never declared its trainer, i.e. exactly when a
default nobody chose decides a verdict. A factory whose flags all exist needs no binding argument
to be graded, so it is out of scope by construction.

**The obvious objection, and I measured it rather than argued it.** The scope has a conceivable
blind spot: a TR1-targeted module whose flags happen to exist on the retired trainer would grade
clean and stay mis-bound forever. Per-module flag census:

| undeclared module | flags | on retired | on TR1 |
|---|---:|---:|---:|
| `curriculum_dsl` | 300 | 297 | 4 |
| `spec_c2_surgical_20260716` | 72 | 72 | 4 |
| `spec_v9c3_duty_ab_20260719` | 65 | 65 | 4 |
| `spec_v9_cgauge` | 18 | 18 | 0 |
| `constants_telemetry_build_wave_20260715` | 8 | 7 | 1 |
| (5 more) | ≤5 | all | 0 |

Every undeclared module's flags sit on the retired trainer, consistent with the default it
inherits. **The blind spot has live count 0 today** — a scoped negative, not an existential one.

To reach live count 0 I also made the two genuinely-retired stub-graded modules declare the
retired pair **explicitly**, via a new `TRAINER_RELPATHS = (...)` plural form. `curriculum_dsl`
legitimately spans two trainers (**MEASURED: 35 of its flags exist only on the base**), so
forcing it into the singular form would have dropped those 35 and manufactured false
missing-flag grades — *a false FAIL replacing a silent PASS*, the trade this registry's own
repair note refuses. A test pins that both resolve to the identical two paths as before.

### 1.4 Positive control — EXECUTED

* **Negative:** live tree → **0** violations.
* **Positive:** remove one declaration → **5** violations, all `fh1`, and `strict=True` raises.
* **Restored:** → 0.

---

## §2 — THE THREE INSTRUMENTS

### 2.1 The activation ledger: not stale, *structurally orphaned* — with a date

| property | MEASURED |
|---|---|
| rows / distinct levers | 250 / **37 of 180 (20.6%)** |
| last write | **2026-07-27T21:17:34Z** |
| non-test writers | **1** — `tools/launch_witness_run.py` (**RETIRED** launcher) |
| `tools/launch_tr1_run.py` calls `record_activation` | **NO** |
| `launch_tr1_run.py` added | **2026-07-28** |
| governed TR1 launch receipts on the SSD tier | **31** |
| …that produced a ledger row | **0** |

**Also a correction to la1 §2.3**, which reported *"0 of 169 launch artifacts belong to the live
vehicle"* — scoped, as it said, to `experiments/results` at depth ≤3. The live vehicle **does**
have launch provenance: **31 `ddm_lv1_tr1_governed_launch_receipt.v1` receipts**, on the SSD tier
where the storage discipline puts run bulk. They were not missing. **Nothing reads them.**

### 2.2 The repair: report the denominator, and declare VACUOUS

`ledger_coverage()` returns the basis, and `never_fired()`'s contract is untouched (its consumers
depend on it) — the number simply can no longer be quoted without its basis. Live output:

```
known_levers 180 · levers_with_any_row 37 · coverage 20.6% · rows 250
last_write 2026-07-27T21:17:34Z · live_launch_receipts 31 · joined_to_ledger 0
roots_scanned [VertigoDataTier, experiments/results] · roots_unavailable [APDataStore]
is_vacuous TRUE — "31 governed live-vehicle launch receipt(s) exist and NONE has a ledger row:
the ledger's only writer is the RETIRED vehicle's launcher; never_fired() is measuring the
writer, not the levers."
```

Note `roots_unavailable`. **An unmounted SSD must never be indistinguishable from a clean tree** —
that is the vacuity genus in one line, and it is pinned by a test.

### 2.3 The coverage query now carries its vehicle

`Completeness` gained `trainer_path` + `vehicle_label` + `describes_live_vehicle`, which **fails
closed** (unknown vehicle is not live):

* default → `[RETIRED vehicle: train_levelset_witness_realized_through_R_mlx.py]` cov **0.819**
* TR1 → `[LIVE vehicle: train_tr1_partition_renderer_mlx.py]` cov **0.178**

Both are correct computations; only one is about the vehicle we ship. **`tools/costate_digest.py`
— the SessionStart hook — now prints the label on its `dsl-orphan` row and the VACUOUS warning
under its duty-to-measure row. VERIFIED by running both sections.** *(A repaired instrument
nobody calls is the failure mode; this one is called at every session start.)*

### 2.4 The finding that changes the cost of the "obvious" fix

The named four-week-old fix (`activation_ledger_not_run_truth_v1`: *"R1 = argv→lever reverse-map
+ engagement-predicate backfill"*, ground truth `launch.sh`) **would itself have repaired the
dead vehicle** — the live line emits `launch_receipt.json`, never `launch.sh`. Retargeted, the
reverse-map is not even needed: the receipt names a sealed **ticket**, and the ticket lists its
levers outright (`live_launch_lever_names()`, landed, reads exactly this).

But the join does not close, and this is the deep one:

| measurement | value |
|---|---:|
| receipts read / tickets read | 31 / 31 |
| distinct lever **instance** names launched | **43** |
| joinable to `known_levers()` (**factory** names) | **0** |

The live spec builds `Lever(name=f"tr1_variant_{variant}")`, `f"tr1_token_grid_D{downsample}_c{code_width}"`
— **names computed at call time from the factory's own arguments**. The instance-name space is
therefore *not statically enumerable at all*. Recording ticket names into the ledger as-is would
leave `never_fired()` reporting every factory as never-fired forever while rows piled up under
unjoinable keys. **The join must be authored (a factory↔instance mapping emitted at ticket-compile
time) before the wire-in — that is the owed work, and it is now measured rather than guessed.**

---

## §3 — THE NO-FAKE ESCALATION: ADJUDICATED, FIXED, GATED

**Confirmed at source.** `inverse_steganalysis_operation_set_compiler.py` emitted
`"byte_closed_operation_count": len(operations)` and
`"chosen_operation_sequence_is_permutation": True` — declaring both properties while the module
performed no byte accounting and no comparison. Its sibling `byte_shaving_campaign.py` computes
**both** for real (`:1346` permutation; `:1504-1511` the blocker-prefix count). **Both producers'
rows land in the same `packet_ir_operation_sets` list and are SUMMED at `:2382-2384` into
`packet_ir_byte_closed_operation_count`, a readiness figure.** An unchecked value was being added
to a total a reader takes as checked. **NO-FAKE forbidden class 1 AND class 4**, simultaneously —
and unambiguous precisely because the correct behaviour exists in the same repo, in the sibling.

**Landing (a) — the fix.** Both fields are now DECIDED: the byte-closed count applies the
sibling's exact blocker-prefix test (shared constant, so the two producers cannot drift while
their outputs are summed); the permutation predicate is an exact multiset match on
`operation_id`, so neither a dropped nor a duplicated operation can pass. Scope held: no rewrite,
because the sibling already showed the intended mechanism.

**Landing (b) — the gate.** `check_no_asserted_packet_ir_readiness_fields`, **STRICT at live
count 0**, refuses an *assertion form* (bare `True`/`False`, bare `len(...)`) as the value of
either readiness key across the packet-IR producer surface. It refuses the shape that **cannot**
have done the work and never judges whether a real computation is correct.

**Positive controls — EXECUTED.** Behaviourally: an operation carrying a not-byte-closed blocker
now yields **1, not 2** (the old code would have said 2), while an *unrelated* blocker still
counts. Gate-wise: re-introducing either form fires it (2 violations), `strict=True` raises,
restoring returns to 0.

**Catalog discipline:** no new number. Rides **Catalog #351** (the fake-claim-guard row, which
CLAUDE.md already records refusing "a marker without an authenticated byte effect" and which
carries a documented prior scope extension), per the post-#400 Catalog #299 consolidation rule.
The §1 gate likewise rides `check_no_stub_lever_factories`'s row as its third extension.

---

## §4 — THE CROSS-STORE CONTRADICTION

la1 §6.1: 4 levers labelled never-fired by `ddm_mt1` are recorded FIRED by the deferral ledger,
2 of them fired-and-**LOST**. Neither survey was careless — they read different stores and
nothing joins them. The cost runs both ways: a fired-and-lost lever re-listed as never-fired
invites a wasted run; a never-fired lever mislabelled fired stays orphaned forever.

**I could not establish the join** (§2.4 is why: the namespaces are structurally disjoint, and
authoring the mapping is a real piece of work, not a lookup). **So I landed the rule instead, in
code:** every row from `activation_report()` now carries `state_store` — the path its state was
read from. Same genus as the measured harness-TaskList-vs-repo-task-ledger split, same cure:
**cite CONTENT and its SOURCE, never a bare label; any "never-fired" claim must cite its store.**

---

## §5 — CORRECTIONS CARRIED (re-derived here from primaries, not re-typed)

Verified this session; `cx1` components sum to `S` exactly and both rate terms reproduce:

| quantity | value | route |
|---|---:|---|
| `cx1` S | 0.8264972 | seg 0.4311790 + pose 0.1597320 + rate 0.2355862 = **exact** |
| `cx1` rate from bytes | 0.2355862 | `25·353808/DEN` |
| PR130 rate from bytes | 0.1272137 | `25·191052/DEN` |
| **gap** | **0.6543562** | `0.8264972 − 0.172141` |
| `W` | 1.2731082153320312 | `4·DEN/PX` |
| **1% of gap** | **9,827.2 B** | `(gap/100)/(25/DEN)` |
| **1% of gap in flips** | **7,719.1** | route A `bytes/W`; route B `(ΔS/100)·PX` — **agree** |
| `cx1` total flips | **508,639** | `d_seg·PX` |

* **la1's "1,930 flips" is 4× too small** — confirmed: `1930×4 = 7720`. It divided bytes by `4W`,
  double-applying the 4 already inside `W = 4·DEN/PX`. Its *bytes* (9,827.2) are right.
* **MAIN's own sister error confirmed:** `cx1` total flips are **508,639**, not 50,863,944 — the
  latter multiplies by the seg *term* (0.4311790) instead of `d_seg`.

---

## §6 — ADVERSARIAL REVIEW (3 clean passes required; counter resets on any finding)

**Counter: 1 clean pass (R3). R1 and R2 each produced findings and reset it. This memo is
PROVISIONAL on the review axis** and says so rather than claiming a seal.

### Round 1 — FINDING (counter → 0)
**I asserted "`record_activation` has ZERO non-test callers" from truncated grep output**, and
checkpointed it. Re-run explicitly: **one** caller, `tools/launch_witness_run.py:3421`. Reaching
for a tool is exactly when to check the instrument — the arm's own subject matter, committed by
the arm. **Fix:** C3 in §0; the true answer (*one writer, on the dead vehicle*) is strictly
sharper than the false one and produced the dated mechanism in §2.1.

### Round 2 — FINDING (counter → 0)
**My own positive-control fixture did not parse**, so the gate skipped the file on `SyntaxError`
and returned 0 — and my test read that as "the waiver worked". A control that cannot fire is
worse than no control, which is the failure this whole arm exists to repair. **Fix:** fixture
rebuilt to stay parseable, plus a dedicated test that `ast.parse`s the fixture before asserting,
so a future malformed fixture fails loudly instead of passing vacuously. Also corrected in the
same round: I briefly changed `modules_globbed`'s meaning while parameterising the scan — caught
and reverted before it reached a test.

### Round 3 — CLEAN (1 of 3)
Re-derived every load-bearing number from primaries; no new finding. Checks passed: la1's
`180 = 149 + 31` reproduced from `package_lever_factories()`; census 141/39 after the re-home;
`verdict_relevant_undeclared == 0`; both plural declarations resolve to the identical two paths;
all four constants in §5 recomputed with two independent routes for the flips; both gates'
positive controls executed and restored to 0; `ruff --select F` clean on all 12 touched files.

### Test state, stated with its denominator
**19/19** of this arm's tests pass. **181/182** in `test_confound_gates.py`. Two failures elsewhere
are **PRE-EXISTING and proven not mine**:

* `test_lever_registry.py::test_332_coverage_rose_from_deorphaning` — asserts `stale == []`;
  `stale` is the 3 `--integer-plane-emitter-*` flags. **Proven at HEAD**: HEAD's `curriculum_dsl`
  already emits the flag, HEAD's trainer does not declare it, and the trainer file is unmodified
  in the working tree. My diff adds **no `--flag` string at all**. Independently corroborated by
  la1's pre-arm measurement (`stale=3`) and its own table listing `IntegerPlaneEmitter` as a
  pre-existing silent stub. *It is also the same class this arm repairs — a stub on the retired
  vehicle — and fixing it means either building 3 flags on a retired trainer or weakening the
  assert. Neither is mine; recorded, not touched.*
* `test_confound_gates.py::…[check_levelset_hosc_requires_beta_end]` — 10 vs bound 9, over
  `experiments/results/*/launch.sh`. **Proven environmental:** those files are **untracked local
  run dirs** (`git ls-files` → not known to git); nothing in the repo can fix it.

### Assumption-challenge (required each round)

| # | assumption | status | if violated |
|---|---|---|---|
| 1 | `package_lever_factories()` is itself honest | **VERIFIED_VIA_SOURCE_INSPECTION** (read in full; parameterised it) | §1 inherits its bug — the largest structural risk here, as la1 also flagged |
| 2 | the 3 modules target TR1 | **VERIFIED_VIA_SOURCE_INSPECTION** (each module's own docstring/log) | the re-home is wrong; but "declare your trainer" is right either way |
| 3 | the 31 receipts are the live vehicle's complete launch record | **PARTIAL** — 2 of 3 roots readable; `APDataStore` unmounted, reported not assumed | counts in §2.1 are a lower bound; the vacuity verdict is unaffected (it needs ≥1 unjoined receipt) |
| 4 | the namespace join is genuinely unauthored, not merely unfound | **VERIFIED_VIA_SOURCE_INSPECTION** (f-string names are not statically enumerable) | if a mapping exists somewhere, §2.4's owed work shrinks to wiring |
| 5 | a re-homed stub is worth surfacing at all | **ASSUMED_AWAITING_VERIFICATION** | §1 is attribution, not value; it makes debt visible, and visibility was mistaken for closure once already (la1 §3.1) |
| 6 | the gate's narrow scope has no blind spot | **VERIFIED_VIA_EMPIRICAL_ANCHOR** (per-module flag census, §1.3) | scoped to today's tree; a future TR1 module with retired-trainer flags would evade it |

**The strongest challenge I could not close:** is any of this on the critical path to a lower
exact score? Honestly — **not directly**. This is apparatus. Its whole claim is that the campaign
has been ranking and dismissing levers using three instruments that could not see the vehicle it
ships, and that a duty queue cannot drain what it cannot see. That is a precondition for
choosing well, not a score mover, and **the pointer is UNMOVED at `0.1910828242`.**

---

## NEXT-IF-RESUMED

**State:** §1–§4 landed with executed positive controls; two STRICT gates at live count 0 with
registered controls (the uncovered ceiling held at 17, not raised). Review counter **1 of 3**.

1. **AUTHOR THE FACTORY↔INSTANCE JOIN, then wire the live launcher** (§2.4). This is the highest-
   value owed row and the one measurement nobody had: emit the factory name alongside the
   constructed `Lever.name` at ticket-compile time in `spec_tr1_renderer_20260728`, then have
   `tools/launch_tr1_run.py` call `record_activation`. **Do NOT wire the launcher first** — 43
   unjoinable keys would manufacture a second cross-store contradiction.
2. **Backfill the 31 existing receipts** through that join once it exists. The live vehicle's
   whole activation history is sitting on the SSD, readable, unread.
3. **Complete review rounds 2 and 3** to seal (currently 1 clean).
4. **Un-taken, named rather than implied:** I did not sweep the 141 retired-trainer factories
   individually (la1 closed them as a population by vehicle retirement, and I did not re-open
   that). I did not read the `witness_control/g111_*` verdict family that la1 §6.3 named as the
   next place to look for NAME-ASSERTS-MECHANISM instances. I adjudicated only HIT 1 of la1's
   five; **hits 2–5 remain open and unadjudicated by me** (`stc_boundary_codec`,
   `math_optimal_joint_solver`, `bregman_dual_metric_guard`, `postdecode_selector_waterfill`).
5. **Two pre-existing reds recorded, not touched** (§6, both proven not mine): the
   `IntegerPlaneEmitter` stale-flag assert, and the environmental hosc bound over untracked local
   run dirs.
6. **Do NOT re-run:** la1's branch/worktree harvest, `ddm_cu1`'s disposition, `ddm_ja1`'s atlas
   re-anchoring, `ddm_mt1`'s 8-row re-check — all credited in la1 §6, none needs redoing.
