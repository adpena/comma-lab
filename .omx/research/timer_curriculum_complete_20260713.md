# Timer + curriculum completion — CE-only component probe and epoch-budget class guard

Date: 2026-07-13/14 UTC
Lane: `timer_curriculum_complete`
Authority: BUILD + local static/config validation only. No Metal boot, training, evaluator, archive,
or paid dispatch was fired. The component timer is MEANS: it resolves the forward/backward split;
only a byte-closed exact evaluator row can move a contest pointer.

## Outcome

- **Timer configuration: PASS.** Both n24/four-epoch arms are one-stage CE-only programs:
  `--no-curriculum`, tau frozen at one, unified `L_tau(1)=CE`, pose and every identified non-CE
  weight zero, no typed parent stage/regularizer, no inherited parent lever, and no emitted
  `--*-start-epoch` or `--*-start-event` token.
- **Trainer boot-guard static clearance: PASS for every schedule/curriculum guard audited below.**
  This is source-derived/config-parsed proof, not a live Metal boot. Main still owes the real
  `--dry-start 2` confirmation on a Metal-capable process.
- **Class fix: BUILT + WIRED.** `schedule_epoch_budget_violations` resolves the trainer's real
  argparse defaults, aggregates every out-of-budget stage cap, runs in `WitnessProgram.validate`,
  and wraps every named launcher derivation before any run-dir write.
- **Named-config audit: 16 clean, 1 latently broken and now compile-refused.** Every requested
  high-risk family compiles at its sealed budget. The broader all-choice audit also found
  `fresh_seeded`'s enabled E=1000 curriculum with l7@1001; it now refuses before writes. At E=4,
  12 non-timer choices refuse, three families scale all starts into-budget and pass, and the two
  purpose-built CE-only timers pass.
- **Pointer delta: none.** `pointer_moved=false`; no score claim.

## 1. Defect and fix

MEASURED reproduction before the fix: the partial muon/l7 cap hit the next live parent invariant,
`LADDER lane/movable window 340/260 >= muon_start_epoch 4`. The deeper cause was structural: the
trainer's `--curriculum` switch gates the discrete CE/tau/l7 dispatch, but independent Muon,
pose-finish, LADDER, Polyak, birth, lane/chroma/screw, and tail controllers are composed outside it.
Turning off only the master would not make the timer clean.

The timer now compiles the v7.5.2 parent at its feasible sealed budget, then derives a four-epoch
child that retains the representation/performance substrate while deleting the score-moving
schedule. The only typed lever is the timer treatment. The async and solo emitted argvs are equal
after removing `--async-verdict` versus `--no-async-verdict`.

### The requested 163/163 count was a stale defect signature

The governed dry-runs MEASURED **99/99**, not 163/163. This is intentional and fail-honest:

- old failed artifact: 163 distinct flags;
- corrected artifact: 99 distinct flags;
- 70 inherited controller/schedule flags removed;
- 6 explicit BooleanOptionalAction OFF tokens added;
- identity: `163 - 70 + 6 = 99`.

The removed set includes the exact contaminating families: curriculum/events, Muon/pose caps,
LADDER, Polyak, birth/ramp, tail, chroma/screw start controls, lane-band start/event, and the
parent tau-event controller. Padding the config back to 163 would reintroduce inert/contaminating
apparatus just to preserve a historical count. Verdict scope: **flag-count expectation only**;
the real-argparse existence gate is fully PASS at 99/99.

## 2. Boot-guard clearance table

Both variants parse identically on these fields through `build_real_trainer_parser`; only async
differs. “Default” means the value comes from the trainer's real argparse, not an invented mirror.

| Trainer boot guard | Effective corrected value | Clearance |
|---|---|---|
| generic epoch-budget compile law | `epochs=4`, `curriculum=false`, zero explicit start-epoch/event tokens | **PASS — vacuous disabled-curriculum branch** |
| eikonal ramp actuation | `eikonal_weight=0`, `eikonal_weight_end=None` | **PASS — constant/off, no orphan ramp** |
| temperature anneal denominator | `anneal_epochs=3000` | **PASS — >=1** |
| LR anneal denominator | `lr_anneal_epochs=1000` | **PASS — >=1** |
| curriculum tau/l7 ordering | `curriculum=false`; parser defaults tau=300/l7=800 are not emitted/active | **PASS — guard skipped by construction** |
| lane-edge never-engages guard | `lane_edge_weight=0`, start default 0 | **PASS — inactive** |
| unified-tau vs explicit tau-start contradiction | `seg_form_unify_tau=true`; no explicit `--tau-softplus-start-epoch` token | **PASS** |
| tau-advance event geometry guard | `tau_advance_mode=clock`; tau start=end=1 | **PASS — event branch inactive, positive endpoints** |
| lane-thin never-engages guard | `lane_thin_weight=0`, start default 0 | **PASS — inactive** |
| margin-saliency never-engages guard | `margin_saliency_weight=0`, start default 0 | **PASS — inactive** |
| polar-chart requires Muon | `film_polar_chart_spel=false`, `muon_start_epoch=None` | **PASS — inactive** |
| Muon budget/frozen-decoder guard | `muon_start_epoch=None` | **PASS — no finisher** |
| LADDER↔Muon stagger guard | `ladder_island_homotopy=false`, `muon_start_epoch=None` | **PASS — inactive** |
| LADDER needs amplify at run setup | LADDER false, `amplify_weight=0` | **PASS — inactive** |
| stage-transition rewarmup domain | rewarmup=0 (parser default), no transition controller emitted | **PASS — inactive** |
| wider-EMA finisher start guard | decay=None, start=None | **PASS — inactive** |
| Polyak finisher window | arm false, start default 0 | **PASS — inactive** |
| pose-finish window | `w_pose=0`, no emitted start, parser default start=0 | **PASS — inactive objective** |
| seed/witness-alone coupling | seed=false, witness-alone=false | **PASS — neither side armed** |
| area/birth detector+ramp coupling | area=false, completion-event=false, ramp=false | **PASS — inactive** |
| remaining curriculum-weighted terms | persistence/amplify/chroma/screw/entropy/logit/eikonal/length/weight-decay/pose all 0 | **PASS — CE-only objective** |

Static scope: this table proves the current parsed config cannot trip the source-inspected
schedule/curriculum boot guards. It does not prove MLX model construction, first-step execution,
wall-clock ratios, training quality, score, archive closure, or promotion.

## 3. Config-build-time class guard

Derived law for an enabled curriculum:

`m_sched = E - max(S_active)` and configuration feasibility requires `m_sched >= 0`.

The implementation obtains parser defaults from the trainer source, normalizes emitted `--no-*`
tokens, excludes resume metadata `--warm-start-epoch`, reports all offending stage caps in one
message, and directs the author to disable curriculum for a true single-stage program or compile a
feasible schedule. Disabled curriculum returns an explicit vacuous PASS; it is not mislabeled as a
measurement. The launcher wrapper covers typed and legacy named configs, including dry-run,
calibration, dry-start, and real launch callers.

Triality:

- DSL: `tac.witness_dsl.curriculum_dsl.schedule_epoch_budget_violations`,
  `WitnessProgram.validate`, timer typed program, and launcher derive wrapper.
- Equation: `curriculum_epoch_budget_feasibility_v1`.
- Registry: locked append through `populate_curriculum_epoch_budget_feasibility_v1`.
- DAG: `FEED-timer-curriculum-complete-20260713`.

req-R: repeat the compile audit whenever trainer stage-start argparse/defaults or named-config
schedule ownership changes.

## 4. All launcher named-config choices audit

| Config | Sealed budget result | Four-epoch override |
|---|---:|---|
| `proven_base` | PASS E=1000, 55 flags | **REFUSE** epoch-budget class guard |
| `all_levers` | PASS E=1000, 81 | PASS; derived tau@1/l7@4/lane-band@1 |
| `sealed_205` | PASS E=1000, 84 | PASS; derived tau@1/l7@4/lane-band@1 |
| `store_nothing_205` | PASS E=1000, 85 | PASS; derived tau@1/l7@4/lane-band@1 |
| `fresh_seeded` | **LATENTLY BROKEN/REFUSE** E=1000 with l7@1001 | **REFUSE** |
| `crucible_v6` | PASS E=3000, 106 | **REFUSE** |
| `crucible_v7` | PASS E=3000, 158 | **REFUSE** |
| `crucible_v752` | PASS E=3000, 159 | **REFUSE** |
| `crucible_v753` | PASS E=3000, 156 | **REFUSE** |
| `v9_cgauge_432` | PASS E=3000, 199 | **REFUSE** |
| `v9_cgauge_truly_optimal_core` | PASS E=3000, 219 | **REFUSE** |
| `v9_cgauge_ideal_mod19` | PASS E=3000, 219 | **REFUSE** |
| `v9_cgauge_ideal_mod32` | PASS E=3000, 219 | **REFUSE** |
| `next_launch_all_levers_20260713` | PASS E=3000, 231 | **REFUSE** |
| `next_launch_all_levers_trimmed_20260713` | PASS E=3000, 219 | **REFUSE** |
| timer async | PASS E=4, 99 | PASS E=4 |
| timer solo | PASS E=4, 99 | PASS E=4 |

“Latently broken” classification: before this landing, 12 non-timer short overrides above could be
constructed with schedule caps outside their run budget (typed families might fail later on a
narrower stagger guard, legacy families could reach boot). `fresh_seeded` was also broken at its
own sealed default: l7@1001 cannot engage in E=1000. These are now
**SELF-PROTECTED/COMPILE-REFUSED**, not silently launchable. `all_levers`, `sealed_205`, and
`store_nothing_205` remain clean at E=4 because their derivation scales every active cap into the
budget; they are curriculum programs, not substitutes for the CE-only timer. The other 16 sealed
defaults are clean.

## 5. Curriculum owed closure

Recall-before-decide was performed through graph memory, the council posterior, corpus grep,
current source, and the live candidate-pool/digest read path.

1. **#302 inherited-PR95 schedule audit — SETTLED, consumed, not re-derived.** The surviving stage
   order was independently justified, while the fixed 300/726 clock was classified cargo. This
   landing closes the new consistency hole: a shortened run can no longer inherit those caps and
   pretend to measure a stage that cannot fire.
2. **#315/#334/#339 witness-native annealing and DSL ownership — BUILT and consistency-checked.**
   Event handoff, first-class `Curriculum`/schedule objects, support gaps, and schedule provenance
   already exist. The new epoch-budget law is additive and uses the same real-parser source; it does
   not invent a second schedule registry.
3. **#430 coherent synergistic whole — DSL compile owed is already CLOSED by #432.**
   `V9_CGAUGE_432_CASCADE_REALIZATION` maps the state-gated cascade onto real trainer events and
   dispositions. The live A/B remains operator-GO/run evidence, not a $0 wiring omission. N7
   label-floor engage and other named trainer gaps remain BUILD-OWED under their existing owners;
   this anti-collision landing did not edit the live trainer.
4. **#403 candidate-pool duty — current and consumed.** Current read: 57 candidates = armed 7,
   built-never-fired 20, needs-build 21, reformulation 6, retired 3; **47 owed**, 0 measured.
   `costate_digest --json` surfaces the pool and top fireable rows. Forty focused pool/store tests
   pass; one separate process-pool atomicity test is sandbox-blocked by `SC_SEM_NSEMS_MAX`
   permission, not a pool/schema failure. No fake row was added: this timer repair creates no new
   score candidate or empirical anchor.

Open run evidence remains explicitly open: the timer Metal boot/run, the #430/V9 live A/B, and all
candidate rows still marked duty-to-measure. Negative verdict scope: none of those families is
killed by absence of a run in this contained build.

## 6. Governed dry-run evidence

Both corrected arms returned rc=0 on fresh durable paths:

- real-argparse flag gate: **99/99**;
- expected-active-lever manifest: one timer lever, PASS;
- schedule provenance: **no positive-epoch schedule triggers emitted**;
- DSL config gate: typed-validated, PASS;
- memory preflight: projected 21.2 GiB, PASS;
- system admission: PASS;
- spawn: **NOT performed** (`--dry-run`).

The process emitted a headless Metal atexit warning after the rc=0 dry-run, confirming why this is
not a live boot. No Metal device was available in this Codex process.

## 7. One-GO-ready Metal packet for main

Use the fresh paths below; do not overwrite the pre-existing failed
`experiments/results/throughput_component_timer_async_20260713` evidence.

First, live boot-test each config on Metal (operator GO):

```bash
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
  --num-pairs 24 \
  --config throughput_component_timer_async_20260713 \
  --out-dir experiments/results/throughput_component_timer_async/ce_only_20260713 \
  --no-dashboard --dry-start 2

.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
  --num-pairs 24 \
  --config throughput_component_timer_solo_20260713 \
  --out-dir experiments/results/throughput_component_timer_solo/ce_only_20260713 \
  --no-dashboard --dry-start 2
```

After both boots pass, launch the matched four-epoch arms sequentially (operator GO):

```bash
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
  --num-pairs 24 \
  --config throughput_component_timer_async_20260713 \
  --out-dir experiments/results/throughput_component_timer_async/ce_only_20260713 \
  --no-dashboard

.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
  --num-pairs 24 \
  --config throughput_component_timer_solo_20260713 \
  --out-dir experiments/results/throughput_component_timer_solo/ce_only_20260713 \
  --no-dashboard
```

Harvest the raw n24 component ratios. Any multiplication by 25 must be labeled
`n24-linear-extrapolation`, never n600 measured. No score/pointer claim follows from timing alone.

## 8. Verification and stores consulted

- 121 combined timer/guard/equation/launcher regression tests: PASS.
- Ruff on all owned Python surfaces: PASS (`RUF005` excluded as the repository's pre-existing
  unrelated launcher style class).
- canonical equation entities: two clean review-tracker passes.
- governed launcher dry-run async+solo: PASS rc=0.
- real parser parse of both full argvs: PASS.
- candidate pool/store tests: 40 PASS; one sandbox-only cross-process semaphore failure.

STORES CONSULTED: `CLAUDE.md`; `AGENTS.md`; craft handoff; v7.5/v8 specs; #302 symposium and
canonical curriculum laws; #315/#334/#339 DSL sources and DAG FEED; #403 pool memo/current JSONL/
`costate_digest`; #430 schedule backtest; #432 coherent DSL materialization; graph-memory keyword
reconstruction; `query_anchors_by_topic` curriculum/schedule posterior; trainer real argparse and
boot guards; current lane/subagent registries; latest sister memos/directives. Already-settled work
was consumed as authority, not re-measured.

Overall verdict: **BUILD-COMPLETE / STATIC-BOOT-CLEAR / METAL-BOOT-AND-RUN-OWED-ON-OPERATOR-GO**.
