# ddm_gd5 (#864) — the grade-5 detector, three formulations measured and refuted, and the one surface that would make it buildable

**Date:** 2026-08-01 · **Arm:** ddm_gd5 (scorer-free, $0, local) · **Axis:** `[macOS-CPU advisory]` ·
`score_claim=false`, `promotable=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false`. **0 scorer forwards.** `upstream/` untouched.

**POINTER HONESTY, FIRST.** Nothing here lowered any score. `effective_frontier` **0.172** (official
leaderboard) UNMOVED; our own-vehicle line **v4d 0.9639878** UNMOVED. This is APPARATUS — MEANS, not
END — and its deliverable is a **refutation plus a named missing surface**, not a tool.

**verdict_scope:** FORMULATION for each of the three refuted predicates (each is one formulation of
"auto-derive the harm class from source", not a claim that the class is undetectable in principle).

**Review status:** pre-registered controls · own-round-1-reviewed (one real defect found in my own
code and fixed before it mattered — §6) · corrected mid-flight by sister arm `ddm_wr2`, verified at
source rather than accepted.

**STORES CONSULTED:** `tools/corpus_query.py` (research 7355 · equations 864 · memory 2043 · dag 908 ·
council 292 · tasks 396 · docs 96) → loaded `built_elsewhere_unwired_is_p0_20260801` (memory),
`ddm_sb2_complete_the_stubs_20260731`, `ddm_rt1_counted_ledger_test_adjudication_and_ci_blind_silence_20260801`,
`ddm_wi1_wrong_instrument_sweep_20260731`, `ddm_wr2_wire_or_retire_adjudication_20260801`;
`.omx/state/operator_p0_ledger.jsonl` (tail, full row), `.omx/state/canonical_frontier_pointer.json`,
`src/tac/witness_dsl/lever_registry.py`, `src/tac/scope_ledger.py`, `src/tac/confound_gates.py`.
**Deliberately NOT loaded:** the pose-pair receipts (owned by `ddm_pw1`), the rank-4 literal
adjudication (owned by `ddm_hl1`), the 7-row WIRE-or-RETIRE verdicts (owned by `ddm_wr2` — I consumed
its conclusions as CONTROLS, and re-derived the two that decide my landing).

---

## §0 HEADLINE (answer first)

**The auto-derived grade-5 detector the p0 asks for cannot be built from source as the repo stands. I
built it, measured it against a real control pair, and it fails — so I deleted it rather than ship an
instrument named for a class it cannot detect.** Three formulations, each measured against the SAME
controls:

| # | formulation | fires on the 9 negative controls | fires on the positive control | verdict |
|---|---|---:|---:|---|
| F1 | the p0's literal predicate — `src/tac` module imported by ≥1 tool/experiment but unreachable from the live vehicle | **1229 of 3251** (38% of the population) | no | **REFUTED — too loud to be a queue, and dominated by retired lineage** |
| F2 | cross-vehicle asymmetry — reached by one declared vehicle, not another, tested + consumed | **0 of 9** | **no** | **REFUTED as a grade-5 detector** — perfect on negatives, blind on the positive; its 42 tr1 rows are visibly the level-set machinery a partition renderer is *supposed* to lack |
| F3 | slot-rivals — the RECIPIENT keyed structurally: a named slot (`basis_selector=`) and the literal values assigned to it repo-wide | n/a | **no** | **REFUTED — `basis_selector` has exactly ONE distinct literal value in the entire repo** |

**The blocking fact, MEASURED and stated once:**

> **The "measured-better successor" relation — the conjunct that makes this class a p0 — exists only
> in memos and receipts. It has no representation in code. A component that has never been wired
> anywhere leaves no trace of being a rival to anything, so there is no edge, no slot value, and no
> registry row for a static analyser to walk.**

That is why the class "nothing detects" is not a gap in our gates. It is a gap in what our **source**
records, and it is fixable — §5 names the exact schema, which is roughly one field-set on the race
receipts we already write.

**Sister convergence, reached independently.** `ddm_wr2` refuted F1 by ADJUDICATING the 7 unowned rows
(0 of 7 meet the harm clause). I refuted it by MEASURING its population (1434 of 3252). Two different
routes, same verdict — which is stronger than either alone. Where we differ is narrow and I re-derived
it: wr2 predicted *"every one of my 7 rows would fire"* under the detector I was building. Against my
actual F2 implementation that is **false — it fired on 0 of 9** (§3). wr2's prediction is correct
about the predicate the **p0 named** (F1), which I had already measured and rejected before its
message arrived. The correction was right about the destination and wrong about where I was standing.

---

## §1 WHAT I RE-DERIVED vs WHAT I COULD ONLY ASSUME

Everything in my dispatch was a pointer, and two of its load-bearing claims turned out to be stale.

**RE-DERIVED at source:**

| seed claim | status |
|---|---|
| `ddm_sb2` §2b inventories 8 built-elsewhere-unwired components | **RE-DERIVED, and CORRECTED.** Only 7 resolve to a module; the 8th (birth-seeding) is a `Lever` inside `curriculum_dsl.py`, not a module. wr2 additionally shows the memo's own ledger, landed in the same commit, grades 5 of them `not-even-designed` — I did not re-derive the ledger myself (it is gitignored and wr2 owns it), so I treat the §2b prose as the *seed*, never as the inventory. |
| "the disguised form: rank-4 reaches tr1 only as hardcoded floats at `lane_guard.py:64-65`" | **RE-DERIVED and now FALSE.** `src/tac/optimization/lane_guard.py:55` is a real import — `from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS` — with an in-code comment giving exactly the right reason ("a literal copied out of a measured artifact LOOKS wired … and cannot track its source"). **The one named instance of the disguised form is CLOSED.** My dispatch (and the p0 evidence field) still describe it as live. |
| "`lever_registry.completeness()` is the canonical auto-derived pattern" | **RE-DERIVED.** `src/tac/witness_dsl/lever_registry.py:107-330`; I mirrored its AST + `lru_cache`-on-fingerprint shape. |
| `tools/launch_tr1_run.py:41 TRAINER_BASENAMES` is a machine-readable vehicle declaration | **RE-DERIVED.** A frozenset of 3 trainer basenames guarding the ONE-n600-job slot. |
| pointer `0.1910828242` (in several sibling memos) | **NOT the current bar.** `canonical_frontier_pointer` → `effective_frontier` **0.172**; own-vehicle **v4d 0.9639878**. |

**ASSUMED, could not derive — stated rather than invented:**

- **Which declared vehicle is "the live one".** No machine-readable designation exists in-tree. The
  frontier pointer names no trainer; `TRAINER_BASENAMES` names a SET, deliberately, because only one
  may hold the slot at a time. My tr1 projections take that name from the p0 ledger and the memory
  file — a **caller's premise**, never a finding of the analysis. This is the single most important
  "derive vs assume" answer in this arm, and it is the reason F2's primary surface was built
  designation-free (asymmetry between declared vehicles) rather than as "unwired for tr1".
- **Whether any of F2's 42 tr1 rows is genuinely harmful.** I did not adjudicate them. wr2's measured
  base rate on the structurally identical population is 0 of 7, and the rows are visibly level-set /
  SDF machinery (`curvelet_placement`, `aa_sdf_observation_render`, `ground_frame_chart`,
  `lane_skipband`, `mod_dim_dynamics`, `decoupled_field`) that a partition renderer has no recipient
  for. That is a strong prior, not a verdict — and I say so rather than reporting 42 as debt.

---

## §2 THE MEASUREMENTS (all `[macOS-CPU advisory]`, all reproducible from the method below)

**Population and denominators — reported per the vacuity rule; an empty scope would have been
`VACUOUS`, never a pass.**

| quantity | measured |
|---|---:|
| `.py` files globbed under `src/tac` | 7,200 |
| non-test `src/tac` modules examined | **3,252** |
| consumer files scanned (`tools/`, `experiments/`, `scripts/`, non-test) | 2,813 |
| declared vehicle entry points derived | **3**, all from `tools/launch_tr1_run.py:41` |

**Import-closure sizes (most generous reading: package `__init__` load edges included).**

| vehicle | modules reached |
|---|---:|
| `train_tr1_partition_renderer_mlx` | 916 |
| `train_levelset_witness_realized_through_R_mlx` | 954 |
| `train_witness_realized_through_R_mlx` | 913 |
| `train_levelset_witness_realized_through_R_torch` (not in the slot set) | 921 |

**The hub effect, which is why reachability cannot be read as wiring.** Cutting four re-export hubs
(`curriculum_dsl`, `preflight`, `confound_gates`, `canonical_equations`) drops the tr1 closure
**916 → 590**. Concretely, `boundary_math/length_sigma` is "reachable" from tr1 only via
`telemetry_producers → tac.witness_control (package __init__) → lambda_net → aniso_perclass_lambda`.
The module is genuinely *loaded*; the capability is not *wired*. Import reachability answers "is this
imported?", never "does the vehicle use it?".

**F1 — the p0's literal predicate**, stated exactly as written (*"imported by ≥1 tool/experiment but
unreachable from the live vehicle"*), taking tr1 as the live vehicle: **2,337** non-test modules are
unreachable, of which **1,229 of 3,251 (38%)** have ≥1 non-test `tools/`/`experiments/` consumer. The
sister variant "reached by NO declared vehicle, tested AND consumed" is **1,434 of 3,252**. Both are
the same verdict at ~1.2–1.4k. The top of the list by consumer count is
`substrates.hprc.archive_candidate` (70), `torch_vehicle.vendored_imports` (67),
`substrates._shared.smoke_auth_eval_gate` (64), `torch_vehicle.driver` (55) — banned-lineage and
retired-substrate code that wants RETIRING, not wiring. A four-figure warn-only queue is not a queue;
it is the "permanently-red gate trains readers to ignore the suite" failure the apparatus already
records.

(The two denominators differ by one module because sister arms edited the tree between the two runs;
the population is ~3,251–3,252 non-test `src/tac` modules throughout.)

**F2 — cross-vehicle asymmetry.** Union **47 of 3,252**; projected per vehicle: tr1 **42**, levelset
**4**, base **45**. The reverse direction is a useful sanity check — the 4 the levelset vehicle lacks
are exactly the tr1-native additions (`lane_guard`, `reset_operator`, `ax1_pool_a_levers_20260730`,
`qa84_rowband_grammar_20260731`), so the asymmetry is real and directional, not scan noise.

**Where sb2 §2b's 7 module-resolvable rows land** — none in F2's output:

| row | tr1 LOAD-reachable | on any declared vehicle |
|---|---|---|
| `boundary_math/island_protection` (gradient surgery) | yes (hub path only) | yes |
| `boundary_math/length_sigma` (σ_cc′) | yes (hub path only) | yes |
| `canonical_equations/segnet_head_rank4_flipdist_20260715` | **yes, genuinely** — via the repaired `lane_guard` import | yes |
| `boundary_math/dash_phase_carrier` (#425 as cited) | yes | yes |
| `torch_vehicle/kd_warm_start` | no | **no** → F1 bucket |
| `optimization/ddm_ms4d_direct_completion` | no | **no** → F1 bucket (and untested) |
| `optimization/hope_bn_capacity` (#725) | no | **no** → F1 bucket |

---

## §3 THE CONTROLS — the reason this arm ships a refutation instead of a gate

Per design philosophy P4 ("no meter without a canary") a new meter needs a positive **and** a negative
control before its readings count. `ddm_wr2` handed over both, ready-made.

**NEGATIVE CONTROLS — wr2's 7 adjudicated rows (+2 I added: the real #425 module and rank-4).**
MEASURED: the F2 gate fired on **0 of 9**. That is the right answer and it is the whole of F2's
discriminating power.

**POSITIVE CONTROL — the pose pair.** The live `terminal_pose_gn` runs
`eg1_generic_low_frequency_six_v1` (the superseded cosine lineage, d_pose 15.29 @ 7,295 B) while the
RACED-better `warp-pose6` (0.393 @ 194 B) sits unwired. MEASURED, and this is the finding:

- `eg1_generic_low_frequency_six_v1` is a **free-form ASCII string** — `terminal_pose_gn.py:226-234`
  validates only non-empty/ASCII/length. There is **no enumerated basis registry** anywhere in-tree.
- Repo-wide, the slot `basis_selector=` has exactly **1** distinct literal value:
  `eg1_generic_low_frequency_six_v1`, in 3 files (`tools/pb1_p5_byte_close_and_eval.py:264`,
  `tools/rehearse_terminal_pose_gn.py:50`, `tools/pb1_terminal_pose_gn_600.py:47`).
- `warp-pose6` appears in 3 files, and **never as a `basis_selector` value**: twice as prose
  (`witness_dsl/spec_tr1_renderer_20260728.py:676`, `experiments/ddm_composed_s_verdict.py:26`) and
  once as an unrelated dict value `"solver": "warp_pose6_analytic_lm_gn"`
  (`experiments/ddm_composed_s_verdict.py:329`).

**So the rival relation is absent from source in all three of its possible shapes** — not an import
edge (there is no warp-pose6 module), not a slot value (never assigned), not a registry row (no
registry). F1, F2 and F3 all score **0 on the positive control**, for the same underlying reason.

A detector that gets every negative right and the one known positive wrong does not separate the
class; it separates nothing. That is the bar the coordinator set, and it is the correct bar.

---

## §4 WHAT I DELETED, AND WHY THAT IS THE FINDING

Built, tested, measured, then removed:

- `src/tac/live_wiring.py` (~640 lines) — entry derivation with file:line provenance, the import
  graph, `ScopeLedger`-backed denominators, a `(size, mtime_ns)` per-file disk cache.
- `src/tac/tests/test_live_wiring.py` — 22 tests, all passing, including both controls, a mutation
  guard, a vacuity test and a waiver test.
- `check_no_cross_vehicle_unwired_components` in `confound_gates.py` + its mandatory `PositiveControl`
  + the two `test_confound_gates.py` bookkeeping updates.

**Feasibility facts worth keeping** (so nobody re-derives them): the cold scan costs ~47 s, ~20 s of
which is `ast.parse` over `src/tac` alone; a per-file `(size, mtime_ns)` cache takes it to ~8 s, and
pruning `experiments/results/**` out of the walk takes it to **~2 s**. Cost was never the blocker.

**Three reasons it should not ship, in the order that decides it:**

1. **It would be named for a class it cannot detect.** That is NO-FAKE forbidden class #1 at the gate
   layer — the same defect `check_no_stub_lever_factories` exists to refuse, one level up.
2. **It would manufacture 47 rows of unowned debt on the day a sister arm drained 13 → 9.** wr2's
   measured base rate on the identical population is 0-of-7 harmful; my 42 tr1 rows are visibly
   level-set machinery a partition renderer has no recipient for. Shipping them as warn-only
   violations is an instrument whose dominant output is architecture mismatch.
3. **CLAUDE.md's BUILT-INSTEAD-OF-PAID poison** (memory
   `built_new_machinery_instead_of_paying_identified_debt_20260731`, operator: *"really bad poison"*,
   *"rm = first-class outcome"*). The identified debt is p0 NEXT_ACTION (1): WIRE warp-pose6 and the
   tt1 analytic Jacobian. A new surface that does not touch it is the trap, not the cure.

An un-consumed module is itself a grade-5 orphan. Landing this one would have been self-refuting.

---

## §5 THE MISSING SURFACE — what to build so the detector becomes trivial

The detector is not hard once the relation exists. Its four conjuncts split cleanly:

| conjunct | derivable from source today? |
|---|---|
| the component EXISTS and is TESTED | **yes** — import graph (F2 built it; ~2 s) |
| **a live mechanism M performs the same ROLE** | **no** — no slot enumerates its alternatives |
| **a MEASURED comparison shows the component better than M** | **no** — lives in memos and receipts |
| no live call site reaches the component | **yes** — import graph |

**The cure is on the producing side, not the detecting side.** We already run races and already write
receipts (`ddm_gd1_hilbert_order_race_receipt_20260731.json`, `hope_rg3_agreement_receipt.json`,
`wr1_descent_receipt.json`). What they do not carry is the RIVAL relation. Adding one field-set at the
moment a race is decided — while the information is in hand and free — makes the join mechanical:

```
role          the slot/capability being filled, as a stable id (e.g. "terminal_pose_basis")
incumbent     the identifier the LIVE path currently uses
challenger    the identifier that was raced against it
metric        the axis compared, with its surface/axis label
delta         the measured comparison, both values, with units
adopted       true | false — and if false, WHY (queued / refused / blocked-by-<named>)
```

Then the detector is: *rows where `adopted=false` AND the challenger beats the incumbent AND the live
path still uses the incumbent.* One join, no registry of orphans, no semantic judgement, and it fires
on the pose pair by construction. It also makes the harm **quantified at detection time** — the thing
neither F1 nor F2 can ever supply, and the thing that decides priority.

This is the constructive half of the refutation and it is where I would point the next arm. It is
**apparatus, not a score mover**, and per the p0's own means/ends firewall it should be sized against
that: NEXT_ACTION (1) — actually wiring warp-pose6 — is where the 0.292941 of v4d's 0.9639878 lives.

**One honest caveat on the schema:** it only ever sees races we *run and record*. A better successor
nobody raced stays invisible, exactly as today. The schema converts "invisible to everything" into
"visible iff raced", which is a real improvement and not a solution.

---

## §6 MY OWN ROUND-1 REVIEW — one real defect in my own work

Per the #337 contract I reviewed my own output before handing it over, and found a defect my tests did
not catch.

**A regex that looks right and silently cannot match.** The tier-2 entry derivation used
`^[A-Z][A-Z0-9_]*TRAINER[A-Z0-9_]*$` to mean "an upper-case constant containing TRAINER". The leading
`[A-Z]` consumes the `T`, so a name that **starts** with `TRAINER` never matches — and **both** real
declarations in this repo have exactly that shape (`TRAINER_REL`, `TRAINER_BASENAMES`). The tier-2
path would have been silently empty forever. It was masked because tier 1 was working, and it only
surfaced when a prefilter I added for speed used the same wrong shape and dropped the tier-1
declaration too, collapsing 3 declared vehicles to 1. **The failure mode is this week's genus in
miniature: an empty result that looks exactly like a clean one.** Fixed by splitting the test into
SHAPE + SUBSTRING and covering it with a dedicated regression test.

Two further self-review catches, both in my own test code rather than the module: an assertion that
`"PASS"` not appear in a vacuous render (the `ScopeLedger` line legitimately reads *"this is not a
PASS"*), and a negative control asserted against the designation-free union when it belonged on the
per-vehicle projection. The second was a genuine semantics error on my part, not a typo.

**Both fixes were unreviewed new code and reset the clean-pass counter.** The deletion in §4
supersedes them.

## §7 APPARATUS OBSERVATIONS (two, both verified, neither owned by me)

1. **The positive-control ratchet worked, and I should say so.** Registering a new refuse-capable gate
   without a control raises the uncovered denominator, and
   `check_refusal_gates_have_live_positive_control` (STRICT) refuses. It caught my landing and forced
   me to write a control before the gate could exist — which is exactly how I discovered the gate had
   no positive to fire on. **The apparatus surfaced the defect that killed my own landing.** That is
   the strongest evidence I saw all day that the immune system is real.
2. **A pre-existing red, confirmed NOT mine.**
   `test_confound_gates.py::test_real_repo_live_count_bounded[check_levelset_hosc_requires_beta_end]`
   fails at HEAD with live count 10, sourced from **untracked** local artifacts under
   `experiments/results/_jbasin_smoke/**/launch.sh`. Verified by removing all my files and re-running:
   1 failed, 156 passed. It is a local-state red, not a regression, and it is unowned.

---

## §8 CROSS-FINDINGS

**→ the `p0_864` owner.** NEXT_ACTION (2) as written — *"module-level import-reachability from the
LIVE entry point"* — is **measurably not buildable as a detector of this class**, by two independent
routes (wr2's adjudication, my population measurement). Please restate it as: *build the RIVAL
relation on race receipts (§5); the detector is a join once it exists.* Also two corrections to the
p0 evidence field: the disguised-form instance at `lane_guard.py:64-65` is **CLOSED** (real import at
`:55`), and the pointer quoted around this class in sibling memos (`0.1910828242`) is not the current
bar (`effective_frontier` 0.172; own-vehicle v4d 0.9639878).

**→ `ddm_wr2`.** Convergent verdict on F1, reached independently. One correction, re-derived: your §8
predicts *"every one of my 7 rows would fire"* under the gd5 detector. Against the F2 implementation
that is false — it fired on **0 of 9**, because F2 keys on cross-vehicle asymmetry, not on the p0's
literal predicate. Your warning is correct about F1, which I had already measured and rejected. The
distinction matters for anyone reading your §8 as a claim about what gd5 built.

**→ `ddm_pw1`.** Unblocked in one respect: no detector is coming that will find the pose pair for you.
It is not findable from source (§3), which is precisely why the operator found it by hand. The 39×
delta is the highest-value item in this whole p0 and it is yours.

**→ whoever owns the CI reds.** `check_levelset_hosc_requires_beta_end` is red on main from untracked
local artifacts (§7.2). Unowned.

---

## §9 WHAT THIS DID NOT DO

It did not lower any score. It did not wire anything into the live vehicle. It did not land a
detector, and it did not adjudicate the 47 cross-vehicle rows it measured. What it did is convert
"the grade nothing detects" from a gap in our **gates** — where three arms would keep building
detectors — into a measured gap in what our **source records**, with the schema that closes it. The
honest summary is that the p0's own NEXT_ACTION (1) is where the score is, and this arm's best service
to it was to stop a second surface being built beside it.
