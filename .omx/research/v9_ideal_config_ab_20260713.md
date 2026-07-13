# V9 CGauge ideal event-native config + decisive mod19/mod32 A/B — BUILD RECEIPT

**Outcome:** actuation fix **PASS**; three DSL programs validate with **MEASURED-at-build 0
violations**; core/mod19/mod32 governed dry preflights **PASS**; both matched A/B commands are
ready but **HELD — operator-GO; fire AFTER the 95%-kill P0 to avoid GPU-timing contention**.
No training or evaluator was launched. Pointer unchanged.

`review_status: REVIEW-GATED, uncommitted; main review/serializer landing owed`

`verdict_scope: build/compile/dry-preflight only; treatment score is UNMEASURED`

## Actuation fix

The exact orphan was re-derived from source. Under `--seg-form-unify-tau`, the main loop chooses
`seg_form = unify_tau` before the discrete curriculum resolver. `_evt_on` remains true when
`--curriculum-event-triggered` is present, but `_evt_state["tau"]` remains unset. The old eikonal
schedule therefore received the unfired sentinel `1 << 30` forever; adding
`--eikonal-weight-end` alone was inert.

The repaired flow is:

`tau relaxation/cap proposes rung k -> prime lambda_eik(k) -> reset/rewarm AdamW treatment ->
assign lower tau_k -> co-assign hosc beta_k -> save complete stageOctave<k> checkpoint -> optimize`.

The new **DERIVED** control law is

`lambda_eik,k = lambda_0 + (lambda_N - lambda_0) * k/N`,

with the settled V9 endpoint transition **MEASURED/ledger 0.01 -> 0.05**. Its sole state owner is
the persisted `TauAdvanceController.rung`; resume therefore reconstructs the same lambda. The
trainer emits `eikonal_retention_prime` before `model.softmax_temp` receives the lower rung. A
startup invariant now refuses an armed unified-tau ramp unless `--tau-advance-mode event` supplies
the reachable actuator. The old discrete sentinel remains valid only for non-unified stage flows.

The same landing closes the beta2 config orphan: each tau-rung proposal is now an actual AdamW
treatment boundary. It clears the old spike scale, resets moments when armed, and anchors the
**DERIVED** `ceil((1/(1-0.999))/75) = 14` epoch cosine rewarm. The octave checkpoint moved after
both tau and beta assignment so controller/model continuation state is complete together.

The equation leg is registered as `eikonal_retention_couples_to_tau_rung_v1`. This is a
**DERIVED/source-inspected actuation law**, not a score law. The exact V9 treatment delta-S remains
**UNMEASURED**.

## Composed core

The core inherits the coherent #432 V9 trunk and adds only the favorable, composition-safe set the
operator requested. Numbers below retain their authority labels.

| Lever / force | Disposition and provenance |
|---|---|
| Unified-tau eikonal hold | **INCLUDED.** Base **MEASURED 0.01**, settled end **MEASURED/ledger 0.05**; interpolation/order **DERIVED** by the new rung law. Exact V9 delta-S **UNMEASURED**. |
| Lane render-band gate | **INCLUDED/inherited and actuated.** `lane_nucleus` is the state event; epoch 500 is a fail-safe cap. Weight **inherited typed 1.0**; treatment delta-S in this arm **UNMEASURED**. |
| Chroma boundary force | **INCLUDED/inherited and actuated.** `annulus_plateau` is the state event; epoch 450 is a cap. Weight **inherited typed 0.1**. Chroma DOF direction is **MEASURED favorable**; add-back delta-S **UNMEASURED**. |
| Temporal screw force | **INCLUDED/inherited and actuated.** Same annulus event/cap, weight **inherited typed 0.1**, ground-GT screw source. Direction **DERIVED favorable**; V9 treatment delta-S **UNMEASURED**. |
| `LengthSigma("fitted-20260707")` | **INCLUDED.** Road-Lane sigma **MEASURED 0.377**, CI **MEASURED [0.317,0.441]**; uniform over-penalty **DERIVED about 2.7x**. Treatment delta-S **UNMEASURED**. |
| `TieLocusDisplacement` | **INCLUDED** at factory values: weight **ASSUMED/DSL 0.3**, band **ASSUMED/DSL 1.0**, start 0; edge source `pa_flipmass` has durable artifact custody at `reports/pa_edge_weights.json`. Boundary-placement direction **DERIVED favorable**; treatment delta-S **UNMEASURED**. |
| `ClosedLoopEikonalControl` | **INCLUDED as containment, not the proactive mechanism.** Bump 0.05, ceiling 0.20, two bumps and three-window thresholds are **ASSUMED/DSL factory values**. Score effect **UNMEASURED**. |
| `Beta2WindowRewarmup` | **INCLUDED and now rung-actuated.** beta2 **MEASURED/config 0.999**, 75 steps/epoch **typed provenance**, window **DERIVED 14 epochs**. |
| Muon / pose conditioning | **INCLUDED/inherited.** Muon fires on power-law meat with nucleus positive control; pose finish uses sigma-min plateau. The prior Muon effect cited by the design is **MEASURED -32% dseg** on its source arm, not claimed for V9. |
| Phase advection | **INCLUDED/inherited** at weight **DERIVED 0.4**, terminal cap 726. Label-smooth flicker floor **MEASURED 0.005318**; this treatment's n600 rate/score remains **UNMEASURED**. |

The expected-active-lever manifest contains 15 named levers and fails closed if any is silently
dropped. The launcher also derives overlapping constants-manifest values from compiled argv. This
repairs the documented hosc duplicate owner: inherited manifest **10.0** versus emitted **3.177**
now records emitted **3.177** as the single owner and checks exact identity.

## Explicit exclusions

- `MarginBandSatisficing`: **OFF**. The sibling `marginband_satisfice_fix` owns its factory/equation
  provenance repair (`0.06` versus **DERIVED 0.0392`). Compose-in candidate only after review.
- StepNative endpoint and FreSh/FINER fresh-start: **ISOLATE** as distinct representation/basin
  treatments; stacking would confound the core.
- OT head offset: **EXCLUDED** from core; matched source result was **MEASURED +5.1% worse**.
- `EikonalViscosity`: **ISOLATE** because smoothing may repeat the thin-lane failure.
- Horizon margin: **ISOLATE** until its V9 treatment-weight custody is derived.
- AA-coverage supersample: **ISOLATE** as a separate compute/receiver treatment. The inherited
  `render-aa ipe` baseline is not that supersample arm.
- Hardness oversample: **ISOLATE** until equal-step accounting exists.
- Code spectral entropy: **EXCLUDED**; weight is **ASSUMED** and no matched V9 score/rate receipt exists.
- Dense stored phase carrier: **ISOLATE at byte close**; n600 exact rate/benefit is owed.
- Margin-saliency source alone: **EXCLUDED as inert** because no active saliency weight consumes it.

## Event-native curriculum realization

1. Deterministic structured initialization, seeds, pose carrier, exact R, and initial custody.
2. High-tau region/nucleus formation with area, persistence, ladder, and seed containment.
3. Annulus/boundary formation with fitted class-pair surface tension.
4. Repeat one rung at a time: complete checkpoint custody; prime eikonal retention; reset/rewarm;
   lower tau; actuate lane/chroma/screw plus tie-locus/length repair; evaluate through-R state.
5. Preserve every accepted rung and stop cleanly on sustained erasure. A rejected proposal resumes
   from its preserved pre-rung checkpoint through the governed campaign path.
6. Freeze tau and enter Muon only after its existing power-law/nucleus gate.
7. Hand label-floor residual to phase appearance, then joint pose/rate close and exact byte-close.

**Honest implementation boundary:** current accept-or-rollback is an operational governed boundary:
complete pre/post rung checkpoints plus the closed-loop stop/fuse make rollback executable and
resumable. The trainer does **not** yet autonomously restore a pre-rung optimizer/model snapshot from
a full-facet comparator inside the same process. That stronger transactional controller remains a
separate REVIEW-GATED build; it is not falsely claimed here. The tau advance sensor itself consumes
through-R dseg history, while lane/chroma/screw/Muon/pose each consume their named state sensors.

## Decisive family A/B

The two arms use the same seed, n600 cache, 3000-epoch ceiling, 25-epoch checkpoint cadence, all-pair
verdict (`--verdict-pairs 0`), stage checkpoints, levers, events, weights, and receiver flow. The
only scientific argv delta is **DERIVED mod dimension 19 versus measured-safe control 32**; output
directories differ only for custody.

Pre-registered decision rule:

`residual = (d_seg_mod19 - d_seg_mod32) / d_seg_mod32`.

If the matched n600 through-R residual is **DERIVED > +2%**, revert to mod32. Report holistic
per-class dseg, nuclei/anchors, dpose against need, exact archive bytes, receiver parse-back, and
axis; a composite headline alone is inadmissible. A remaining Road/Lane-bound corrected trunk is
evidence to escalate to SPEC-v8 edge-centric carriers, not proof from this build.

## HELD governed launch commands

**HELD — operator-GO; fire AFTER the 95%-kill P0 to avoid GPU-timing contention. Do not execute
before both conditions are satisfied.**

```bash
.venv/bin/python tools/launch_witness_run.py \
  --config v9_cgauge_ideal_mod19 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --out-dir experiments/results/v9_cgauge_ideal_mod19_20260713 \
  --no-dashboard
```

```bash
.venv/bin/python tools/launch_witness_run.py \
  --config v9_cgauge_ideal_mod32 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --out-dir experiments/results/v9_cgauge_ideal_mod32_20260713 \
  --no-dashboard
```

No extra trainer flags are present; the config is exclusively DSL-compiled.

## Dry-start evidence (compile + preflight only; no run)

- Core, mod19, mod32: **MEASURED 0 validation violations**, **MEASURED 213/213 real argparse flags**,
  expected-lever manifest **PASS**, DSL provenance gate **PASS**, schedule-provenance gate **PASS**.
- All three: memory projection **DERIVED by preflight 24.48 GiB** versus safe ceiling
  **DERIVED 89.6 GiB**; safe-compile fingerprint **PASS**.
- System admission: core **ADMIT 68.7 <= 99.5 GiB**; mod19 **ADMIT 64.3 <= 101.7 GiB**; mod32
  **ADMIT 62.6 <= 102.6 GiB**. These are volatile preflight snapshots, not future launch authority.
- Dry commands used `--dry-run --skip-throughput-gate --no-dashboard`; they wrote only `launch.sh`
  and `constants_manifest.json`. No `run.log`, checkpoint, PID, daemon, training step, or evaluator row
  exists. Exit code was 0. A post-exit headless Metal atexit warning reported no device; it occurred
  after the successful non-spawn dry gate and is not GPU evidence.
- Focused regression evidence after final rerun: eikonal/tau, closed-loop, event curriculum, tau
  resume/control, DSL config, launcher resolution/composition, manifest parity, and equation tests:
  **MEASURED 121 passed in 2.85 s**. The same headless post-exit Metal warning was non-fatal.

## Triality and pointer honesty

- DSL: `src/tac/witness_dsl/spec_v9_cgauge.py` and named launcher branches.
- Equation: `eikonal_retention_couples_to_tau_rung_v1`, registered through the locked canonical
  equation helper; pure callable is consumed by the trainer.
- DAG: `.omx/research/sub015_DAG_v9_ideal_config_ab_20260713.md`.
- Pointer delta: **NONE**. This launch-ready config is **MEANS**. Only a receiver-closed n600 exact
  archive row on its declared CPU/CUDA axis can move the pointer. Treatment delta-S for most included
  levers is **UNMEASURED until run**.

## STORES CONSULTED

Full CLAUDE.md and AGENTS.md contracts; `docs/operating_manual_craft_handoff.md`; committed V9 ideal
design; #432 config design; dead run launch/log/costate; canonical frontier, lane, subagent, equation,
posterior, and blocker surfaces; latest sister memos; trainer argparse, unified-tau/event/eikonal,
tau-controller, checkpoint/resume, lane/chroma/screw/Muon/pose gate paths; DSL curriculum/gauge/campaign,
typed compiler and launcher; `reports/pa_edge_weights.json`; governed memory/safe-compile/system-admission
preflights. No live run, paid provider, remote GPU, archive evaluator, or contest pointer was actuated.
