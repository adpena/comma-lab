# Wall-clock burn-down BUILD+MEASURE — task #509 continuation (2026-07-15)

**Operator (verbatim):** "12.5 is unacceptable must continue optimizing." Charter: STANDING P0
min-wall-clock d_seg convergence — the JOINT objective epochs-to-target × sec/epoch
(`feedback_wallclock_to_target_joint_objective_drift_ok_if_gradient_good_20260715`). Operator
resequence mid-unit: telemetry-first (#2 alongside #1), verdict fix HELD until telemetry lands.
Parent ledger: `.omx/research/v9_missing_signal_constants_audit_20260715.md` (§A/§B/§D).

**Pointer 0.19108 UNMOVED. Everything here is MEANS.** All numbers `[macOS-MLX advisory]`
NON-PROMOTABLE; verdict_scope on every negative as tagged.

---

## 1. LANDED (commit `c219841d8c`, serializer, reviewed 2×, tests green, ruff clean)

### 1a. #480 real-path epoch decomposition (the 63%-attribution instrument) — BUILT + WIRED
`tac.witness_control.telemetry_producers.COMPONENT_FIELDS` extended with FIVE **disjoint
main-thread wall intervals** (unlike the inclusive probe fields, these ARE summable;
`real_path_sum_s` emitted per row; `epoch_total_s − real_path_sum_s − checkpoint_io_s` =
honest remainder):

| field | region (trainer, `run_train`) |
|---|---|
| `real_grad_accum_s` | per-chunk fused value_and_grad + mx.eval (serial + micro-batch paths) |
| `real_optimizer_s` | clip + per-group clip + opt.update + EMA + mx.eval (two paired intervals) |
| `real_loss_terms_telemetry_s` | the `_loss_terms_for_chunk` no-grad recompute |
| `real_epoch_probes_s` | the D-A decomposition probe + SPS emit themselves |
| `real_verdict_submit_s` | MAIN-thread verdict join/decide/snapshot/schedule (the +903s suspect) |

Score-neutral by construction (perf_counter_ns around EXISTING statements; no graph/RNG effect);
schema string stays `witness_component_wallclock.v1` (additive-legacy per the launch-ticket AST
pin); `tools/analyze_witness_throughput_corpus.py` verified unaffected (keys its own 8-field
copy). The `real_verdict_submit_s` field **decides** the audit §A-3 "+903s submit-block"
hypothesis vs the coincident cadence-25 checkpoint + `_mdd_ablation_checkpoint` work (both fire
at ep25) — the #306-falsification question is now instrumented, not argued.

**First real-config consumer:** the sister governed n600 dry-start
`levelset_n600_drystart3_telemetry480_20260715` (launched 11:27, after this commit) runs these
fields at the REAL config — its `witness_component_wallclock.jsonl` rows are the first n600
attribution (§3 below records what landed by session end).

### 1b. #B-4 grad-clip cure — AutoClip percentile law, FULL TRIALITY
- **equations leg:** `tac.canonical_equations.autoclip_percentile_grad_clip_20260715`
  (`autoclip_percentile_threshold_v1`, registered APPEND-ONLY in the JSONL registry; evaluator
  in `LAWREF_BUILTIN_EVALUATORS`). Law: `clip_t = percentile_p(‖g‖ history, window w)`
  (AutoClip arXiv:2007.14469; ZClip 2504.02507 sibling; #500 Fisher trust region = successor).
- **mechanism:** `tac.witness_control.adaptive_grad_clip` — ring-buffer observe-then-threshold,
  fixed-clip warmup fallback, nonfinite norms never poison the history, per-group states (the C4
  anti-starvation sibling preserved), resume-safe under `__acl_` (canonical registry; fixed mode
  persists NOTHING ⇒ byte-identical sidecars). 14 behavior tests.
- **trainer wiring:** `--grad-clip-mode {fixed,autoclip}` (default fixed = BYTE-IDENTICAL
  incumbent) + percentile/window/warmup flags; clip-site dispatch (global + per-group);
  per-epoch `grad_clip_autoclip` telemetry row; loud arm banner.
- **DSL leg:** `curriculum_dsl.AdaptiveGradClip` — composable by bare name; the three numeric
  constants carry LawRef custody (`_v9_scientific_constant_custody`); registry-mapped
  (completeness test). Sister `LaneBandStaticCache` lever holds the cache flag.

### 1c. #509-3 lane-band pair-static constant cache — BUILT, default ON, bit-identical
`make_lane_band_compose_fn(cache_static=True)`: weighted stop-grad coverage cached per UNIQUE
prior + gt-source u_mask per code; θ-dependent (callable) margin NEVER cached. Bit-identity
unit-proven (`np.array_equal` cached-vs-uncached on real-geometry MLX composite). Memory ~0.5
GiB @ n600 (inside the preflight envelope).

---

## 2. MEASURED this session (receipts)

### 2a. Lane-band cache: **−0.04 s/ep — the −40–60 s/ep hypothesis is REFUTED at the conversion locus**
$0 microbench, real `gt_n24.npz` lstars, real priors, MLX steady-state (cache-fill excluded),
6 reps × 48 codes: per-call 0.356 ms (off) → 0.323 ms (on), Δ 0.033 ms/call ⇒ **0.04 s/ep at
n600** (1200 calls). *verdict_scope: INSTANCE (microbench; compose+eval only).* The +75 s/ep
lane-band cost (C0 ep33+) is therefore dominated by the **θ-dependent witness margin +
lane-appearance forwards added to the fused graph** (~2 extra partial forwards/pair + backward
through the appearance leg) — intrinsic to the lever's mechanism, NOT static-geometry recompute.
The audit's D.3-3 "precompute/cache −40–60 s/ep" ranking is corrected: the cache is a KEEP
(free, correct) but NOT a burn-down item. The real lane-band sec/ep lever is SCORE-AFFECTING
(fold `call_margin` into the main forward's own soft output, or margin-refresh cadence) —
queued as a ticket, needs its own A/B (values change).

### 2b. **CONFOUND FOUND (FORMULATION-level, source-inspected): C0's clip saturation is INERT — masked by `--grad-normalize per-param`**
C0's launch.sh line 144 runs `--grad-normalize per-param`
(`tac.witness_stability.per_param_normalize_grads`: `g_p ← g_p/(‖g_p‖+eps)` per tensor, applied
AFTER the clip). A uniform per-tensor scale is divided out exactly ⇒ **any norm-based clip
(fixed 0.5, per-group, or autoclip) has ZERO effect on the applied update under per-param
normalize.** Corrections to the parent audit:
- §A-1's mechanism claim "effective step = lr·0.5/‖g‖ ≈ lr/12" is **REFUTED for C0** — the
  telemetry (frac_clipped=1.0) is real but the clip is a downstream no-op. C0's actual
  magnitude law = per-tensor unit-norm × LR (SignSGD-like; magnitude information discarded).
- The corollary "loss weights only set direction under saturation" transfers, with the sharper
  cause: per-param normalize discards per-TENSOR magnitude by design ("ALTERS the seg-vs-pose
  gradient SCALE ratio … NOT proven for our objective" — its own docstring; an owed A/B since
  #146).
- The honest epochs-to-target A/B is therefore **magnitude-LAW vs magnitude-LAW**:
  A = incumbent (per-param normalize + inert clip 0.5) · B = `--grad-normalize none` +
  `AdaptiveGradClip` · optional C = `--grad-normalize none` + fixed 0.5 (isolates
  normalize-vs-clip). Registered as the equation's OWED anchor
  (`autoclip_descent_speed_effect_n24_ab_owed_20260715`, arm definitions updated by this memo).

### 2c. NAMED BLOCKER: the whole `crucible_v752`/timer-ticket family is currently UNLAUNCHABLE under today's #406 enforcement
Every governed launch of `throughput_component_timer_async_20260713` — clean, `--dsl-lever`, or
`--extra-trainer-flags` — refuses rc=8:
`LawRef compiled value for 'hosc_beta_end' differs from WitnessProgram flag --hosc-beta-end:
10.0 != 3.177` (and, on the AdaptiveGradClip compose, an earlier
`inputs must be a non-empty mapping of name->InputRef` in the same TypedWitnessConfig
self-recompile). Root: `lawref_builtins.HOSC_BETA_FIREBAND_PIN` (10.0, v6.3 pin) is still the
custodied LawRef in the v752 parent lineage while the emitted flag was rephased to 3.177; the
#406 DSL-compile-hash gate (landed TODAY, `fa5a671330`) now self-recompiles and catches the
mismatch. The v9_cgauge family custodies 3.177 correctly (its dry-starts launch). **Unlock
(owed, one edit + reseal):** rebind the v752/timer lineage's beta-end custody to the 3.177
rephase declaration (10.0 preserved as `historical_non_authorizing` per CLAUDE.md #351), or
compile the planned burn-down A/B ticket directly on the v9 lineage. The governor REFUSE is
information — not bypassed.

### 2d. n600 real-config attribution (sister dry-start) — see §3 status at session end.

---

## 3. What is BUILT vs STILL OWED (honest ledger)

| Item | Status | Receipt / unlock |
|---|---|---|
| #480 real-path decomposition | **BUILT+WIRED** (defaults follow `--component-wallclock-telemetry`) | first n600 rows: sister dry-start `…drystart3_telemetry480…` |
| Grad-clip AutoClip lever | **BUILT** (law+mechanism+DSL+resume+tests) | descent A/B **OWED** — bounded governed ticket, arms per §2b, on the v9 lineage (v752 family blocked §2c); admission bar = gradient-quality + no-flicker, d_seg per WALL-CLOCK |
| Per-param-normalize confound | **MEASURED (source)** | fold into the A/B (arm C); parent-audit §A-1 correction recorded here + DAG |
| Lane-band static cache | **BUILT+MEASURED** (−0.04 s/ep; keep, not a burn-down item) | n600 confirmation rides any next governed launch (default ON) |
| Verdict-submit unblock (#4, authorized) | **HELD pending telemetry** (by design): `real_verdict_submit_s` decides submit-block vs cadence-25 checkpoint/mdd-ablation coincidence | build after the first verdict-epoch row (needs a run with eval cadence hit; dry-starts never fire a verdict) |
| bf16/fp16 + mx.compile arm (max-Metal #3) | **BUILD-OWED** — no dtype seam exists in either trainer (0 grep hits); model-level surgery (fp32 master + low-precision compute) | n24 gradient-quality gate per the m5max campaign memory; `--safe-compile-regions hosc_activation` + `--mx-compile` (R) already live = the compile leg partially fired |
| MAX-METAL ceiling row | ESTIMATE-ONLY (flagged): fp16 rate ~2× fp32 on M-series GPU ⇒ grad-accum share (if ~60-70% of wall) → ~1.5-1.8× sec/ep ceiling from precision alone; megakernel GPU receipts 1.12–1.21× (#356) now training-reopenable | becomes a measured row only after the bf16 arm builds |

## 4. Projection arithmetic (honest)

Baseline (parent audit): post-ep33 steady **325 s/ep** + 36 s/ep verdict amortization ≈ 361
s/ep ⇒ 3000 ep ≈ **12.5 days** vs 8.314 budget. Receipts landed this session change sec/ep by
**−0.04 s/ep** (cache) ⇒ **the honest projection is still ~12.5 days.** No lowered number is
claimed without receipts. The named path down:
1. the n600 attribution rows (instrument live NOW) → rank the real sec/ep burns inside the 63%;
2. the verdict decision row → either the async-submit fix (−36 s/ep, −10%) or the checkpoint/
   ablation coincidence fix (same order);
3. the magnitude-law A/B → epochs-to-target (a multiplicative win the sec/ep table cannot show;
   at the measured 12× step-suppression *bound* the upside is large, but per §2b the true C0
   suppressor is normalize, so the win is UNKNOWN until measured — no number invented);
4. bf16 compute arm (ceiling ~1.5–1.8×, estimate-flagged).

**STORES CONSULTED:** parent audit + AdaptivizationTicketQueue · C0 launch.sh/telemetry ·
throughput_component_timer spec + its 07-13 GREEN run · launch_witness_run gates (#399/#406
refusals recorded verbatim) · lawref_builtins · witness_stability · m5max campaign memory ·
telemetry_producers/corpus analyzer · gt_n24 cache. verdict_scope: all negatives INSTANCE- or
FORMULATION-scoped as tagged; nothing here is a score claim; **pointer 0.19108 UNMOVED**.

---

## 5. Session-end addendum (post-§2c developments)

- Sister landed `fef6d3cc2b` (dsl_compile_hash round-trip CLOSED) mid-session. After it +
  flipping `AdaptiveGradClip(scientific_declaration=False)` (the DsegAwareTaper precedent;
  lawref-carrying INTERNAL levers still trip the #406 self-recompile — scoped, documented in
  the factory docstring), **the magnitude-law A/B arm-B now composes END-TO-END on the v9
  lineage**: dry-run emitted launch.sh with `--grad-normalize none --grad-clip-mode autoclip`
  through every gate:
  `tools/launch_witness_run.py --config v9_cgauge_ideal_mod19 --dsl-lever AdaptiveGradClip
  --dsl-lever GradNormalizeNone …` (new `GradNormalizeNone` arm lever expresses the §2b
  confound in the DSL). Remaining unlock for a BOUNDED run: a short-epoch governed ticket
  (the v9 config compiles at epochs 3000; the sealed timer ticket refuses resizing + still
  carries the §2c hosc custody mismatch).
- The n600 attribution rows: the sister dry-start (`…drystart3_telemetry480…`, pid 14828) was
  still in boot at session end (~19 min; n600 boot is long). Harvest is one command once
  `dry_start/witness_component_wallclock.jsonl` exists — the reader script is pre-staged
  (per-epoch real_* fields + `unattributed_s/frac`); every row after ep1 IS the 63%-answer.
- Pre-existing failures noted (NOT mine, verified by stash): 2 Metal-parity costate tests
  (`test_costate_warmstart_cluster` / `test_costate_requential_curriculum`) fail while the GPU
  is held by the dry-start.

---

## 6. BATCH 3 (same subagent-id continuation, 2026-07-15 ~17:00Z onward)

### 6a. LANDED (serializer, reviewed, tests green, ruff clean)
- **`9d3bfc837b`** — verdict-parallel-workers TRAINER WIRING (`--verdict-parallel-workers`,
  default 0 = sequential byte-identical; ThreadPoolExecutor chunk fan-out on the ADVISORY
  CPU verdict, bit-identical values by construction: same chunk spans, unchanged intra-op
  torch threads, Executor.map ordered aggregation) + 34 custody/equivalence tests. The DSL
  lever landed in batch 2; this closes the mechanism leg. Verdict-wall bench still OWED.
- **bf16/fp16 COMPUTE SEAM (the §3 BUILD-OWED — now BUILT).** Mechanism
  `tac.witness_control.compute_dtype_seam` (fp32 masters; params cast INSIDE the traced
  loss — fp32 grads via the astype VJP; entry shims `__call__/call_batch/sdf/call_margin/
  render_lane_appearance` cast inputs down + outputs back to fp32 ⇒ render/R incl. the
  fused-R Metal kernel, FROZEN-SCORER forwards, verdict, EMA, checkpoints, decode ALL fp32;
  masters restored after every call ⇒ resume-safe by construction, nothing persisted).
  Trainer `--compute-dtype {fp32,bf16,fp16}` (default fp32 = seam never constructed =
  byte-identical) + `--compute-dtype-quality-check N` (first N opt steps compute BOTH arms
  from the same masters, compare POST-normalize update direction — cosine + rel-norm, the
  C0 lesson — receipts to `compute_dtype_quality.jsonl`, and STEP WITH THE FP32 REFERENCE).
  Refusals: micro-batch>1 (un-seamed twin), QC×autoclip, QC×seed-islands, QC×(per-group ∧
  normalize-none). DSL leg `curriculum_dsl.ComputeDtype`; law leg
  `bf16_compute_seam_gradient_quality_v1` (registered APPEND-ONLY; n24 QC anchor OWED;
  admission thresholds cos_min 0.99 / rel-band [0.9,1.1] PROPOSED until the QC
  distribution is measured). 26 new tests. **Cross-check (operator-shared intake
  `pluralis_stoa_rl_on_macs_intake_20260715.md`): Pluralis runs fp32-master/bf16-trainer at
  production scale on Macs — existence proof for the MLX low-precision path; their
  tolerance regime is RL off-policy (laxer than our dense descent) so the QC gate is still
  the admission authority, never the analogy.**
- **`19e380a849`** — micro-batch parity-receipt loss-abs bound DERIVED from the loss scale
  (isclose form `floor + rel_tol·|serial|`; strict relative check stays binding; canonical
  stored constants unchanged). Fixes 8/11 routed pre-existing failures
  (`blocked_codex_landing_recovery_20260715` finding). The remaining 3 are DISTINCT:
  (a)×2 `test_v9_gpu_surface_emits_real_metal_backend_receipts` +
  `test_v9_faithful_384x512_metal_maps_and_area_value_vjp` fail on a Metal shader JIT
  breakage — `mlx/backend/metal/kernels/utils.h:443: invalid parameter name: 'signed' is a
  keyword` — the custom v9-lever kernel build is broken against the current MLX/Metal
  compiler (#478 kernel surface; deterministic, 0.6s fail, NOT contention);
  (b) `test_temporal_uses_raw_f0_provider_while_pose_keeps_general_carrier_render`
  negative-control fails with BOTH oracle and wrong-provider values = 0.0 — the temporal
  term is zero in the fixture (fixture/routing regression, loss-internals surface, owed).

### 6b. Custody-fix CONFIRMED + the ready-to-fire magnitude-law A/B ticket
The compose arm's hosc_beta_end reconciliation (34e375a2e7..30c216617e) is verified live:
arm-B (`--dsl-lever GradNormalizeNone --dsl-lever AdaptiveGradClip`) and arm-C
(`GradNormalizeNone` alone) both compile END-TO-END through every launcher gate at n24 on
`v9_cgauge_ideal_mod19` (dry-run: dsl-config gate OK 230 flags; admission ADMIT).
**Bounded-window shape:** `--epochs 120` is REFUSED by the typed epoch-budget feasibility
gate (fixed caps 450–800 in the sealed schedule) — the correct bounded form is the SEALED
config + `--dry-start 39` (the C0 saturation window ep1-39; bounded execution, schedule
integrity preserved, pass1/pass2 resume round-trip exercises the new `__acl_` state).
Three arms, SEQUENTIAL (never concurrent — wall-clock hygiene):
```
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz --num-pairs 24 \
  --config v9_cgauge_ideal_mod19 --dry-start 39 --no-dashboard \
  --out-dir experiments/results/levelset_n24_maglaw_armA_20260715 \
  --purpose "m5_burndown_509 batch3: magnitude-law A/B arm A (incumbent per-param normalize + inert clip)"
# arm B: + --dsl-lever GradNormalizeNone --dsl-lever AdaptiveGradClip   (out-dir ..._armB_...)
# arm C: + --dsl-lever GradNormalizeNone                                (out-dir ..._armC_...)
```
Metric: d_seg v0→ep25 verdict delta + per-epoch ep_loss descent slope over ep1-39 =
the epochs-to-target proxy; arm A also produces the FIRST verdict-epoch
`real_verdict_submit_s` row (cadence-25 verdict + cadence-25 checkpoint + mdd-ablation all
fire at ep25) = the §A-3 +903s decision datum (batch-3 step 4).

### 6c. M1 attribution status (data-gated)
The M1 dry-start DOUBLE-SPAWNED (launcher retry while attempt-1's trainer was live — the
`launcher_buffered_log_not_hung_orphan_spawn_respawn_id_collision_20260715` class; the
coordinator killed the duplicate, sole writer preserved). Pass1 (dry_start/) ran ep1 in
2582s gross and was SIGTERM'd by the harness's resume-round-trip design BEFORE the ep1
component row flushed ⇒ 0 rows from pass1. Pass2 (dry_start_resume/, pid 65778) resumed at
ep2 and spends its first ~43 min in the blocking 600-pair v0 CPU verdict (the 2555.7s §D.2
wall, now visible as boot-cost too). First #480 rows land when ep2 completes. NOTE for the
harvest: ep1's 2582s gross ≈ v0-verdict-dominated, NOT a steady sec/ep; the real_* fields
exist precisely to decompose this.

---

## 7. THE n600 ATTRIBUTION ROW (harvested) + the operator mem-for-compute fire-now (2026-07-15 ~18:30Z)

### 7a. ep2 attribution (drystart3 pass2, the FIRST #480 real-path row at the real config)
`experiments/results/levelset_n600_drystart3_telemetry480_20260715/dry_start_resume/witness_component_wallclock.jsonl`
(pass1's ep1 row was LOST — the dry-start harness SIGTERMs pass1 after ep1 for the resume
round-trip BEFORE the row flushes; pass2 hit the 3300s pass timeout at ep3's terminating-epoch
row hold, so ep3's row was withheld too — exactly ONE clean row):

| component | s/ep | share |
|---|---:|---:|
| epoch_total (ep2, n600) | **1095.6** | 100% |
| real_grad_accum (fused vag + eval) | 232.1 | 21.2% |
| real_optimizer (clip+normalize+opt+EMA) | 14.2 | 1.3% |
| real_loss_terms + probes + checkpoint | <1.0 | <0.1% |
| real_verdict_submit | 0 (not invoked ep2) | — |
| **UNATTRIBUTED remainder** | **848.5** | **77.4%** |

The 63% became a MEASURED 77.4% at ep2. Against C0's post-ep33 steady ~325 s/ep, the 848s is
consistent with an early-epoch/one-time class (MLX kernel JIT + first-build lazy graphs +
cache fills) — but ONE row cannot separate warmup vs periodic (jacobian-basin every-4 /
annulus / event sensors) vs in-loop-gap. **CLOSED STRUCTURALLY: #480 v2 landed 3 disjoint
epoch SPANS** (`span_pre_loop_s` / `span_accum_loop_s` / `span_epoch_tail_s`; NOT
real_-prefixed so real_path_sum semantics unchanged) — the next instrumented epochs (the live
n24 A/B arms emit them at every epoch) classify the remainder mechanically:
in-loop-gap = span_accum_loop − (grad+opt+terms), tail = sensors/telemetry/verdict region.

### 7b. Operator directive (mem-for-compute fire-now) — actions taken
Measured headroom: run RSS ~27 GiB of 128 (M1 pass2 peak 33.4); ~78-100 GiB idle.
1. **Verdict workers**: bench `tools/bench_verdict_parallel_workers.py` IN FLIGHT (w=0/8/6,
   n600, thread-law pinned, value-equality asserted across arms; receipt →
   `experiments/results/verdict_parallel_bench_*/receipt.json`). Sizing derivation from
   headroom: ~5-6 GB/worker ⇒ 8 workers ≈ 48 GB inside the 0.70-safe-frac single-run
   envelope ⇒ compose `VerdictParallelWorkers(8)` into the NEXT n600 launch once the ÷N
   receipt lands (lever + wiring already landed; trainer default stays 0 = byte-identical).
2. **Warmup amortization**: data-gated on the span rows (7a). Candidate artifact classes
   enumerated: MLX/Metal kernel JIT (process-scoped; macOS system shader cache partially
   persists — measure before building a custom cache), structured-init (already skipped on
   resume), gt-cache load (MEASURED 1.9-4.2s — not the burn), cf-feats fill (mem-preflight
   shows 0.07-3.4 GiB class), lane-band statics (cached, batch 2). Warm-restart ep-time
   before/after = the owed receipt once spans classify the 848s.
3. **Complete pinning**: pair-static enumeration = cf_mx feats (cached per pair today —
   resident pin = the existing cache, already ON), lane-band statics (ON, batch 2), GT
   skeleton (--cache-gt-skeleton ON), R index maps/weights (fused-R kernel constants —
   OWED: verify they are built once not per-call), verdict GT argmax/pose targets (loaded
   once). No new resident pin is justified until a span row shows a pair-static recompute
   actually burning wall (the lane-band −0.04 s/ep lesson: conversion loci can be tiny).
4. **Micro-batch**: sequenced AFTER the bf16 QC gate per the directive (preconditions
   changed; the 1.0-1.07x receipt was unfused fp32).

### 7c. The magnitude-law A/B — LIVE
Bounded-window mechanics: `--epochs 120` REFUSED (typed feasibility, caps 450-800);
`--dry-start` capped at 3 ⇒ the arms run as REAL governed launches with a PLANNED CLEAN STOP
after the ep39 checkpoint (`--ckpt-every 1` ⇒ any stop is resumable; stop command =
`tools/spawn_durable_daemon.py --stop levelset_witness_levelset_n24_maglaw_arm<X>_20260715`).
Launch-readiness rungs (HorizonWeightedMargin/StepNativeActivation) DEFERRED with the A/B
isolation rationale (arms must differ ONLY in the magnitude law). First launch attempt
crashed all 3 arms at boot: `int(_idet.movable_cls)` TypeError — the island self-detector
legitimately returns None on a 24-pair subset; FIXED `d2cc57dc6e` (None-guard, n600
byte-identical). State at memo time: **arm A (incumbent) + arm B (normalize-none+AutoClip)
LIVE past boot; arm C (normalize-none+fixed-clip) governor-REFUSED on the memory ceiling
(bench + 2 arms active-growth projection) — bounded governed retry loop armed (6×10 min)**.
Metric: ep1-39 CE ep_loss descent slope + v0→ep25 verdict d_seg delta (epoch-indexed ⇒
GPU-contention-immune); arm A also emits the FIRST verdict-epoch `real_verdict_submit_s` +
`span_*` rows at ep25 (the §A-3 +903s decision datum, batch-3 step 4).

### 7d. Projection (honest)
**Still ~12.5 days — no receipts have changed sec/ep yet.** Named, instrumented paths down:
(1) the span rows classify the 848.5s/77.4% remainder (arms emitting now); (2) the verdict
÷N receipt (bench in flight) → compose workers=8 → verdict-epoch amortization shrinks by
~1/N; (3) the ep25 submit row decides the async-submit −36 s/ep-class fix; (4) the
magnitude-law A/B → epochs-to-target (multiplicative; unknown until measured); (5) bf16 QC
gate → the ~1.5-1.8× estimate-flagged compute ceiling. pointer 0.19108 UNMOVED.
