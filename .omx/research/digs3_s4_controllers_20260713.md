# DIG-S3 finite-bank information gain + DIG-S4 transactional option exit

**Date:** 2026-07-13  
**Lane / checkpoint id:** `lane_digs3_s4_20260713` / `digs3_s4`  
**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE`  
**Actuation:** `NONE` — `$0` local, read-only trace replay and design only  
**Pointer:** `0.1880443979880752` read through the current canonical P8 helper; **UNMOVED**

## Answer first

> **S3:** no learned/acquisition ranker beats current P8 on admissible evidence. The strict
> 72-bank join has **0** real custody-complete descriptor/outcome/cost rows and therefore **0**
> chronological test folds. Winner vs P8 = `NONE_IDENTIFIED; P8_NOT_BEATEN`; backtested regret =
> `NOT_IDENTIFIED`, not zero. The S3 primary falsifier fires on uncalibrated uncertainty, so the
> disposition is **stratified cheapest-first**, with P8 retained as a floor/axis guard and no
> RND/ICM theater. Double-Q is mandatory on any future nonempty backtest, but cannot create support.
>
> **S4:** stages and tau rungs admit a clean transactional `OPTION` type, but **no proposed sensor
> beats the existing gates on the read-only traces**. Five real logs contain 82 verdict rows and 130
> checkpoint rows but **0** explicit common-checkpoint, common-horizon `{stay, advance}` pairs.
> The S4 primary falsifier therefore fires: preserve the fixed schedule. Sensor outputs remain
> advisory and conjunctive with the existing topology/full-facet gates.

`verdict_scope:` S3 rejects promotion of contextual information-gain, posterior-sampling,
pseudo-count, Double-Q, or any other learned selector on the current 72-row evidence surface. It does
not reject those families after real outcome/cost support exists. S4 rejects learned or sensor-only
termination as a replacement for the current fixed schedule on the selected traces. It does not
reject transactional option exit after a custody-complete same-checkpoint same-horizon fork exists.

## Containment and source epoch

The charter's bank is frozen at the crosswalk source commit
`31bb1e324fe7b4a649442b98c1f0ce4da06c8827`. Static AST reconstruction finds 73 registered DSL
factories at that epoch; canonical `duty_to_measure` semantics leave exactly 72 owed because a row is
discharged only after it has both fired and been measured. A mechanism measurement without a matching
fired arm remains owed.

Four factories landed after the charter epoch and are explicitly excluded from the 72-row bank:

- `AdamWReferenceSemantics`
- `FilmPolarChartSPELManifoldMuon`
- `MarginCompandedGroundChart`
- `MuonAtCheckpointBoundary`

This is a fixed experimental-design bank, not a moving count and not the separate curriculum pool.
No trainer, run directory, checkpoint, controller state, provider, GPU, or organ actuator was mutated.

## S3 — finite contextual experimental-design bank

### Typed row contract

Each frozen bank row has this descriptor, derived without executing a trainer:

```text
DutyDescriptor := {
  lever: exact source-epoch DSL factory identity,
  flags: transitive emitted trainer flags,
  flag_count: integer,
  stratum: one of {
    boundary_margin, curriculum_tau, optimizer_compute, other,
    pose_temporal, rate_payload, render_receiver, topology_birth
  },
  activation_state, ever_fired, ever_measured,
  measurement_cost_epochs: statically resolved default epochs_delta or UNKNOWN,
  current P8 estimate/label/axis where present
}

OutcomeCost := {
  activation_ts,
  exact source-epoch factory identity,
  structured n_pairs = 600,
  exact receiver-realized Delta S (positive means improvement),
  exact comparable cost in seconds,
  verdict artifact path + SHA-256
}
```

`measurement_cost_epochs=0` means only “no incremental epoch extension in the DSL factory.” It does
**not** mean a free experiment or zero wall time. Seventy rows have a statically resolved
`epochs_delta`; two have unknown cost. Forty-eight resolve to zero incremental epochs. The bank strata
are: boundary/margin 20, topology/birth 11, curriculum/tau 10, other 10, optimizer/compute 7,
pose/temporal 6, rate/payload 6, and render/receiver 2.

### Strict custody join

Seven historical `measured` ledger events were audited. **Zero** satisfies the complete join.
Rejections overlap because one event may violate more than one field:

| rejection | count |
|---|---:|
| no exact source-epoch factory descriptor | 4 |
| no exact comparable cost | 2 |
| no exact comparable Delta S | 2 |
| verdict reference not a structured JSON artifact | 2 |
| verdict reference not a file | 2 |
| structured cohort not n600 | 1 |

The closest apparent exception is `LaneBandResCoder`: its JSON has a real n600 pure-rate outcome
(`-10,634` bytes, derived `+0.0070807441 S`) and `20.345265 s`. It is **not** a source-epoch factory
identity, so importing it would require an invented descriptor mapping. It is rejected rather than
silently used. `DashComb` has a real n600 mechanism result but lacks a comparable exact cost and
complete score outcome. `HeadOffsetSolver` is n24. The remaining records are non-structured,
field-characterization, alias, or incomplete-cost rows.

### Chronological comparator result

With zero complete rows, every requested comparator has zero train/test folds:

| policy | calibration | simple regret | top-k discovery | measurements to first confirmed improvement |
|---|---|---|---|---|
| VIME-style information gain | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` |
| bootstrapped posterior sampling | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` |
| descriptor pseudo-count anti-starvation | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` |
| Double-Q debiased selection | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` |
| current P8 | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` |
| cheapest-first | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` |
| family round-robin | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` |
| random, seed 0 | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` | `NOT_IDENTIFIED` |

The live P8 reconstruction is still useful as a **current display prior**, not a backtested winner:

| P8 row | estimated Delta S | relative significance | label |
|---|---:|---:|---|
| `DsegAwareTaper` | 0.030 | 0.788552 | `ESTIMATED` |
| `HorizonWeightedMargin` | 0.018 | 0.473131 | `MEASURED` oracle/screen anchor, not a treatment response |
| `StepNativeActivation` | 0.013 | 0.341706 | `MEASURED` screen anchor, not an adoption effect |

Those three values are not eligible response labels for this backtest because they lack the complete
descriptor/outcome/cost treatment row. Thus “P8 not beaten” is the honest comparison; it is not “P8
wins with zero regret.”

### Double-Q debiasing guard

The negative audit's Double-Q addition is binding. On every future chronological fold `t`, split
only the past custody-complete rows `D_<t` into disjoint `A_t` and `B_t`:

```text
a_t = argmax_a Q_select(a; A_t)
reported_value_t = Q_eval(a_t; B_t)
```

The same noisy posterior realization may not select and evaluate an arm. This guards winner's-curse
optimism. It does not solve offline support, calibrate uncertainty, or manufacture outcomes. With
zero folds the guard is specified but inactive.

### Primary falsifier and fallback

The charter's exact falsifier is met through its second disjunct: **uncertainty is uncalibrated**.
Descriptor response generalization cannot even be tested. Therefore:

1. partition the fixed 72 rows by the typed stratum above;
2. order each stratum by known `epochs_delta`, then exact lever identity; unknown costs sink;
3. round-robin strata, taking the cheapest remaining row in each, preventing family starvation;
4. retain P8's measured term-floor/axis guard as a veto, not as fabricated response evidence;
5. atomically append posterior state only after a new exact n600 receiver-realized outcome and exact
   cost row lands;
6. do not use RND, ICM, pseudo-reward, synthetic response labels, or memo-derived Delta S as reward.

The first 12 deterministic fallback rows are recorded in the S3 receipt. This fallback is a design
disposition only; it did not fire any lever.

## S4 — live stages and tau rungs as transactional options

### Option type

```text
OPTION O_j := (
  state_j, initiation_set_j, policy_j, eligibility_j,
  C_switch_j, dwell_min_j, hysteresis_j, termination_j,
  rollback_checkpoint_j
)
```

Every state includes the input checkpoint hash, stage/tau rung, optimizer and resume-controller
state, fixed in-stage weights, seed/data ordering, and the full facet vector. `C_switch` is a measured
common-horizon score/runtime quantity, not a guessed penalty. It is currently `UNIDENTIFIED` except
for typed components such as the configured 8-epoch rewarmup/reset operation.

| option | live initiation / eligibility | dwell + hysteresis | switching-cost components | rollback |
|---|---|---|---|---|
| `O_unify_tau(k)` geometric tau rung | current `--seg-form-unify-tau`; existing nucleus/topology/pose/rate guards; `--tau-advance-mode event` | `--curriculum-min-stage-epochs 250`; require confidence separation at a common horizon | tau change, any re-treatment, equal-horizon Delta S and runtime; total `UNIDENTIFIED` | complete pre-rung EMA + resume state; persisted tau-controller state |
| `O_lane_birth_boundary` | `lane_nucleus` born+formed guard before lane-band engagement | event latch plus stage dwell; no repeated toggle | treatment engage/re-treatment and receiver consequence; `UNIDENTIFIED` | immediately preceding complete checkpoint |
| `O_annulus_sharpen` | `annulus_plateau` before chroma/temporal-screw engagement | detector defaults: 4 trailing verdicts and 150-epoch span; event latch | loss-form engagement and re-treatment; `UNIDENTIFIED` | immediately preceding complete checkpoint |
| `O_muon_finish` | `powerlaw_meat` plus nucleation-complete guard; epoch 726 is a backstop | no chattering after optimizer identity switch; configured 8-epoch rewarmup/reset | optimizer replacement, moment reset/rewarmup, receiver Delta S and runtime | pre-Muon complete checkpoint including optimizer and controller state |
| `O_pose_finish` | conditioning `sigma_min_plateau`, d_seg basin, separatrix/full-facet non-regression | 3-consecutive hysteresis in the live gate; epoch 726 backstop | pose-training window and possible d_seg/rate harm; `UNIDENTIFIED` | pre-pose checkpoint; otherwise ship banked R1 |
| `O_tail(k)`, `k<=2` | terminal turnpike cycle after eligible finish | `--tail-dwell-min 237`, cycle floor `387.09`, marginal-S stop `0.0001` | tau halving `0.5`, LR re-proportion, runtime and receiver effect | preserve each pre-cycle stage checkpoint |

Under unified tau the old CE-to-tau boundary is **not** a live option boundary. Older CE/tau traces
remain read-only evidence, not authority to reintroduce the dissolved switch.

### Read-only sensor replay

The deterministic receipt hashes five logs and replays only existing, typed sensors. It never reads
the outputs into training.

| trace | verdicts | checkpoints | loss plateau fire, min150 / min250 | annulus final fire | power-law final exhaustion | common-horizon pairs |
|---|---:|---:|---|---|---|---:|
| crucible v6 run1 | 14 | 24 | none / none | no | no | 0 |
| mod32cap | 41 | 68 | ep200 / ep819 | no | yes | 0 |
| 2026-07-09 witness | 5 | 7 | none / none | no | insufficient | 0 |
| v7.5.2 baseline | 10 | 15 | none / none | no | yes | 0 |
| v9 CGauge arm | 12 | 16 | ep187 / none | no | yes | 0 |

Fresh falsification details:

- The default 150-row loss plateau fired at mod32cap ep200, after which d_seg continued to descend
  `0.004963 -> 0.004869` by ep225. This independently reproduces the charter's inherited result that
  a slope trigger can fire while d_seg still descends (the crosswalk's inherited trace places its
  fire near ep151; this replay does not overwrite that source-specific epoch).
- The live 250-row dwell still fired at mod32cap ep819, followed by a further
  `0.004163 -> 0.004152` descent. Dwell delays but does not turn a slope into a counterfactual exit
  certificate.
- V9's 150-row loss plateau fired at ep187, after which d_seg rose
  `0.034976@175 -> 0.037564@200`. A scalar slope does not distinguish continued useful descent from
  receiver erosion.
- `powerlaw_meat` declared final exhaustion on mod32cap, v7.5.2, and v9. Those are sensor readings,
  not proof that advance would beat stay; v7.5.2 and v9 also lack the conjunctive annulus/full-facet
  exit receipt.
- `annulus_plateau` fired zero times. On these 25-epoch verdict-cadence traces, its trailing four
  points span 75 epochs, while `min_epochs=150`. Therefore the current cadence/guard combination
  cannot fire by event before its numeric backstop. This is a trace-specific apparatus blocker to
  relay to the trainer owner, not permission to weaken the guard here.
- Twenty-four `lane_nucleus would_fire` rows exist, but they are explicitly observability-only and
  provide no stay/advance counterfactual.

No selected log contains both policies with the same explicit checkpoint hash and same preregistered
horizon. Similar epochs in independent runs are not a common-horizon pair.

### Termination law and why it does not close yet

For an eligible option boundary with common horizon `H`, the proposed directional rule is:

```text
ADVANCE
iff existing topology/nucleus/pose/rate guards pass
and dwell + hysteresis pass
and UCB[L_advance(H) + C_switch] < LCB[L_stay(H)]
and the pre-boundary checkpoint is rollback-loadable.
```

Otherwise return `STAY`, `ROLLBACK`, or `INSUFFICIENT_INFORMATION`; never coerce the missing branch.
Here both confidence bounds and `C_switch` are unidentifiable because the pair count is zero. The
primary falsifier therefore fires exactly: **no affordable common-horizon counterfactual is present;
preserve the fixed schedule and say so.**

## Triality, DAG, and integration

- **DSL:** no trainer flags invented and no live controller changed. The S3 fallback and S4 option
  selector are policy objects, not single DSL levers; pool rows state that explicitly. Existing
  launch flags and resume/checkpoint surfaces are only read back.
- **DAG:** `digs3_s4_controllers_DAG_FEED_20260713.md` records the S3 refusal/fallback and S4 fixed-
  schedule verdict with receipt hashes.
- **Equations:** no canonical equation row is registered. The Double-Q split and option-confidence
  inequality are conditional design equations, but the empirical law does not close with zero S3
  folds and zero S4 branch pairs. Registering them as measured laws would be false authority.
- **Sensitivity / Pareto / bit allocator / autopilot:** no Delta S was minted, so no sensitivity or
  bit-allocation value changes. The only safe autopilot output is the S3 stratified cheapest-first
  queue and S4 fixed-schedule/insufficient-information refusal. Pose/rate remain hard eligibility
  constraints, not pooled scalar guesses.
- **Continual learning:** empirical signal is canonicalized in the receipts and candidate pool:
  contextual acquisition is reformulation-gated; the stratified fallback is build-owed; option exit
  is reformulation-gated on a common-horizon fork.
- **Probe disambiguator:** the eight S3 policies and five S4 sensor families are represented in the
  receipt. The probe result is refusal, not a synthetic winner.

The canonical pool helper `record_candidate` appended exactly these latest-row-wins records:

| candidate | status | reactivation / build gate |
|---|---|---|
| `digs3_contextual_information_gain_ranker` | `reformulation-queue` | at least two custody-complete chronological rows, one real test fold, calibrated held-out uncertainty, then a chronological win against P8 and the non-learned comparators |
| `digs3_stratified_cheapest_first_fallback` | `needs-build` | advisory stratum round-robin with known-cost ordering, unknown-cost sink, P8 floor/axis veto, and exact-outcome append discipline |
| `digs4_transactional_option_exit_shadow` | `reformulation-queue` | identical rollback-loadable checkpoint and horizon for both branches, then all existing guards plus the confidence-separated loss inequality |

## Receipts

- `tools/digs3_s4_backtest.py`
- `experiments/results/digs3_s4_20260713/s3_finite_bank_backtest.json`
- `experiments/results/digs3_s4_20260713/s4_option_trace_backtest.json`
- `experiments/results/digs3_s4_20260713/receipt_manifest.json`

All receipt files are atomically written. The manifest records bytes and SHA-256. Re-running the
tool is deterministic at seed 0 over the frozen source epoch and current read-only ledgers/traces.

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`
- `.omx/research/spinningup_keypapers_crosswalk_20260713.md` (S3/S4 charters and primary falsifiers)
- `.omx/research/negative_audit_wave_20260713.md` (N11 Double-Q debiasing guard)
- `.omx/research/hcm_causal_attribution_dig_20260713.md`
- `.omx/research/tofupov_ranker_allocation_20260713.md`
- `.omx/research/adaptivebayes_costate_intrinsictime_20260713.md`
- `.omx/research/solver_pack_junction_sigma_powerlaw_20260707.md`
- `src/tac/witness_dsl/{curriculum_dsl,lever_registry,activation_ledger,curriculum_candidate_pool}.py`
- `src/tac/witness_control/{event_wirings,powerlaw_exit,tau_advance,producer_bridge}.py`
- `experiments/train_levelset_witness_realized_through_R_mlx.py`
- `.omx/state/lever_activation_ledger.jsonl`; `.omx/state/lever_relative_significance.jsonl`;
  `.omx/state/curriculum_candidate_pool.jsonl`; `.omx/state/subagent_progress.jsonl`;
  `.omx/state/lane_registry.json`
- the five hashed read-only `run.log` files enumerated in the S4 receipt

## Pointer-delta honesty

This arm produced a bank descriptor, strict refusal/backtest receipts, an option contract, a DAG
feed, and controller-pool rows. It produced **no candidate archive, no score, no live action, no
provider spend, and no pointer movement**. The organ remains advisory.
