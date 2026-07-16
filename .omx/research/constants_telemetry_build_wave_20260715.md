# CONSTANTS + TELEMETRY BUILD WAVE (#515) — DERIVED levers + B0 instruments

Date: 2026-07-15 (executed into 07-16Z) · Arm: CONSTANTS+TELEMETRY BUILD-WAVE (P0, campaign #515)
· $0 build+derive, nothing launched, live dry-start (pid 31576) untouched, NO trainer edits.
Research-only: `true`. **Pointer: submittable 0.19108 UNMOVED — everything here is MEANS.**

**Operator directive (verbatim):** "All the constants stuff and unbuilt telemetry and more must be
built and measured as well." This is the BUILD half: every dimension of `c2_optimal_form` gets a
built + DERIVED carrier; the A/B battery (operator-GO on cost) MEASURES.

**Landed code (commit `63b1137a0b`, 32 tests green):**
- `src/tac/witness_dsl/constants_telemetry_build_wave_20260715.py` — 7 Lever factories + 5 typed
  `HardcodedWaiverCustody` rows + 4 registered LawRef evaluators + the updated battery + c2 recipe.
- `src/tac/witness_control/weight_norm_telemetry.py` — the per-tensor ‖W‖ producer (the optdyn
  unlock; pure functions over the trainer's `live_np`/`ema_np`; resume-safe baseline).
- tests: `test_weight_norm_telemetry.py` (10) + `test_constants_telemetry_build_wave_20260715.py` (22).

**STORES CONSULTED:** `adaptivization_tickets_20260715.py` (the 14-ticket queue this drains) ·
`c1_config_differential_audit_20260715.md` §2/§3/§5/§6/§8 · `v9_missing_signal_constants_audit_
20260715.md` (FEED-510) · `telemetry_enhancement_audit_v7x_v8_20260710.md` (#408 Q1–Q7) ·
`cgauge_parametrization_optima_20260711.py` (Law 4) · `costate_lambda_marginal_ds_20260705.py` ·
`verdict_parallel_workers_speedup_20260715.py` · `optimizer_dynamics_followup_20260715.md` ·
trainer argparse (every emitted flag verified; never-invent-flags).

---

## 1. BUILD TABLE

| item | derived value + law/equation_id OR waiver | DSL lever | A/B arm + threshold | trainer wire-in |
|---|---|---|---|---|
| `--adam-beta2` (0.999 inherited) | **0.997691 DERIVED** — log-midpoint of the #223 Law-4 stationarity sandwich `cgauge_beta2_window_v1` window [0.98667, 0.9996] @ S=75/T_c=100 (N\*=√(75·2500)≈433 steps; point criterion = max log-distance from both failure modes); custody `stationarity_window_log_midpoint_v1` | `DerivedAdamBeta2()` | B3b {0.999 vs 0.99769}; d_seg@ep150 n24 SCREEN → n600 through-R confirm; falsify: Δ < noise band ⇒ 0.999 stays (in-window DONT-CARE) | done (flag exists; lever compiles) |
| `--ema-decay` (0.997 Quantizr/L18 ancestor) | **0.997691 DERIVED** — same sandwich transferred as a WINDOW argument to the weight-EMA (data-cycle floor / curvature-drift ceiling); incumbent 0.997 (N=333) is IN-WINDOW near the midpoint — the derivation broadly CONFIRMS its order and retires the ancestor literal AS A DERIVATION pending B3a | `DerivedEmaDecay()` | B3a EXTENDED {0.997, **0.99769-derived**, 0.99, 0.999}; EMA-verdict d_seg@ep150 n24 SCREEN → n600 confirm; falsify: best-vs-worst < noise ⇒ ticket closes as window-confirmation | done |
| `--w-pose` (1.0 inherited) | **0.3941 DERIVED** — λ_pose/λ_seg = (5/√(10·d_pose))/100 at the banked R1 operating point d_pose 0.001610 (`costate_lambda_marginal_ds_v1`, MEASURED n600 anchor L68); surrogate-calibration caveat RECORDED in the lever | `DerivedWPoseAtEngage()` | B3e {1.0 vs 0.394}; fork-from-c2-ep726/sigma_min checkpoint; metric d_pose-at-engage + d_seg non-regression; falsify on d_seg regression > noise | lever done; **LIVE engage-time consumer queued** (`--w-pose-costate-engage`, insertion @ pose-finish engage ~L11698) |
| `--eval-every` (25 inherited) | **25 CONFIRMED-DERIVED (conditional)** — amortization law eval\* = max(25-floor, ⌈effective_inflation/(0.10·sec_per_ep)⌉); MEASURED C0 economics (900 s inflation/verdict, 325 s/ep base) + the VPW ladder (`verdict_parallel_workers_speedup_v1`, saving 305.5 s): **with VPW(8) → 19 < 25 floor ⇒ 25 holds at ~7% overhead; WITHOUT VPW → 28 (incumbent runs ~11%, over-budget)**; custody `verdict_cadence_amortization_v1` | `DerivedEvalEvery()` (REQUIRES `VerdictParallelWorkers(≥2)` in the same config) | B3h {25-derived vs 50}; wall/ep + F10 dwell re-derivation + read-latency; falsify: cadence-50 breaks would-fire calibration ⇒ 25 stays (now DERIVED, no longer ancestor-suspect) | done |
| `--muon-lr/momentum/ns-steps/final-frac` | **class-4 WAIVER** (typed custody): η_rel-pin cure ticket-only-unbuilt; optdyn MEASURED the unchosen ×1.40 per-layer relative-LR drift | custody row in `CLASS4_WAIVERS` | B3d measure-first: B0 ‖W‖ rows quantify the drift on the c2 run; pin-vs-flat fork-from-ep726 later | ‖W‖ stream = this wave's producer; pin consumer = ticket |
| `--hosc-beta-end` (3.177) | **class-4 WAIVER**: control-preserving rephase (`hosc_beta_fireband_pin_v1` custody); endpoint contested by built step_iso 8.0 (34.2% duty) | custody row | B3f {3.177 vs 8.0}; d_seg@ep150 n24 SCREEN + saturation telemetry → n600 confirm | n/a (arm exists) |
| `--accum-pairs` (8) | **class-4 WAIVER**: joint objective has no closed form; memory permits {4,16} (41.86 GiB measured peak @128) | custody row | B3c {8,4,16} joint with B1 winner; d_seg@ep150 + sec/ep n24 SCREEN → n600 confirm | n/a |
| `--grad-clip` (0.5) | **class-4 WAIVER**: INERT under `--grad-normalize per-param` (C0 confound memo); measured alternative (naive AutoClip) REVERSED post-ep25 | custody row (AdaptiveGradClip/GradNormalizeNone levers already BUILT elsewhere) | B1 ≥150-ep magnitude-law A/B + S4 causal rebase (gates everything — the descent clock) | n/a |
| ladder birth constants | **class-4 WAIVER**: saddle-node critical-λ derivation unbuilt (continuation reframe) | custody row | derivation-gated (not sweep-gated) — #318/#344/#180 reduced-order model | n/a |
| `--lr-anneal / --hosc-beta-anneal / --seg-phase-advect-start` | **ALREADY CURED LIVE** (no build needed): event-LR trainer:12216, β-rung co-anneal trainer:12139, phase-advect @ label_floor lever in c1 — the #510 "unwired" rows are stale at HEAD (c1 audit §3 corrections) | existing | n/a | done (verified by the c1 audit) |
| per-tensor ‖W‖ telemetry row (optdyn unlock) | producer BUILT: `weight_norm_row()` — live/EMA L2 norms + `rel_from_t0` drift + `eta_rel` (= update/weight, the pin's read stream), resume-safe `baseline_from_row`; consumer laws already registered (`inr_weight_norm_radial_ode_v1`) | `WeightNormTelemetryRow()` — **FAIL-CLOSED** (`TrainerWireInQueued`) until the flag lands; auto-unlocks | B0 acceptance: rows appear at verdict cadence; identity by construction (read-only) | **QUEUED behind pid-31576**: flag `--weight-norm-telemetry` (BooleanOptionalAction default TRUE), insertion @ the EMA-verdict emit where `live_np`/`ema_np` are materialized (~L9332 region) |
| `--verdict-batch 64` | **MEASURED anchor** (never-slower vs 32; FEED-510 §C.2/D.3-7 Tier-0; c1 audit rec-row 2); bit-identical verdict values per the #240 chunking law; custody `build_wave_custody_identity_v1` @ rung measured_anchor | `VerdictBatch64()` | B0 first-verdict value-identity check | done (flag exists) |
| `--mod-dim-dynamics` | score-neutral D18 k90 sensor, default-True but argv-silent in c1 ⇒ registry-unmapped orphan; explicit emission = custody + owner | `ModDimDynamicsOn()` | B0 (rows appear; ~7 KB k90-truncate rate lead feeds the D18 byte-close) | done (flag exists) |
| `--verdict-live-gap-every` (#408 Q3) | **ALREADY BUILT** — `VerdictLiveGap()` lever exists (curriculum_dsl:3240, single emitter; trainer default −1 auto-warmup); no duplicate home built (one-owner-per-flag) | existing | B0 (confound-H2 discriminator for the first decisive read) | done |
| #408 Q1/Q2/Q4–Q7 producers | **ALREADY BUILT + TRAINER-WIRED** (verified this audit, do NOT rebuild): `ClipActivationAggregator` @ trainer:11322 (Q1 per-group clip rows) · `term_inert_rows` (Q2 chroma/levers term-domination + inert alarm) · `tail_cycle_endpoint_row` (Q4) · `would_fire_row` @ 10534/10552 (Q5) · `ladder_birth_complete_row` @ 11393 (Q6) · `lever_engage_row` @ 6091/6095/11303/11396/11620 (Q7 uniform schema) — all in `tac.witness_control.telemetry_producers` | n/a (`[no-triality-lever]` read-only apparatus class per the #408 audit) | already emitting on the live config | done (pre-existing) |
| `VerdictParallelWorkers(8)` | existing lever (curriculum_dsl:2763) + trainer-wired `9d3bfc837b` + bench 5.686× receipt — composes into c2 per the recovery table | existing | B0 identity self-check built-in | done |

**Derived vs waiver count:** of the constants dimension, **4 constants now emit DERIVED values**
(beta2, ema-decay, w-pose, eval-every — eval-every as a conditional confirmation of the incumbent)
+ **3 were already cured live** (event-LR, β co-anneal, phase-advect event) + **5 carry typed
waiver custody with a named duty-to-measure arm** (muon internals, hosc-beta-end, accum-pairs,
grad-clip, ladder birth). Zero silently-inherited literals remain in the #515 scope.

**EmpiricalAnchors:** none NEW registered — this wave MEASURED nothing (all $0 derivations over
existing measured anchors: C0 verdict economics, the VPW bench receipt, banked R1 d_pose, the
optdyn norm-shrink measurement). Registering an anchor for a derivation would be a fake
measurement; the battery produces the anchors.

## 2. UPDATED A/B BATTERY (folds this wave into c1-audit §8)

Machine-readable: `BUILD_WAVE_BATTERY` in the module (`BuildWaveManifest().compile_contract()`).
Deltas vs §8: **B3a gains the derived-EMA arm (+3.2 h)**; B3b/B3e's "law value" arms are now BUILT
levers (no cost change); B0 gains ‖W‖ + explicit mod-dim-dynamics + vb64 custody (build done, ~0
marginal); B3d's named unlock (the ‖W‖ stream) is now BUILT (producer) — B3d remains
measure-first. B2/B3g/B4a-c/B5 unchanged from §8 (inherited by reference).

**Total-cost delta:** n24 battery 47–53 h → **50–56 h (≈2.1–2.3 GPU-days; +3.2 h)**; B2 n600
bounded basis pair +4.0 GPU-days unchanged ⇒ **full ≈6.3 GPU-days (was 6.2) · compressed ≈2.3
(was 2.2)**; cloud spend $0 throughout.

## 3. c2_optimal_form COMPOSITION RECIPE (winner-sets-flag; machine-readable `c2_composition_recipe()`)

- **B0 unconditional:** `VerdictParallelWorkers(8)` + `VerdictBatch64()` + `VerdictLiveGap()` +
  `ModDimDynamicsOn()` + `WeightNormTelemetryRow()` (the last after its queued trainer wire-in).
- **B1 winner** → magnitude-law flags (AdaptiveGradClip/GradNormalizeNone or incumbent-stay);
  the GRAD_CLIP waiver custody retires either way.
- **B2** → `--basis` per the no-regression rule (curvelet opt-in, never a silent flip).
- **B3 winners** → `DerivedEmaDecay`/`DerivedAdamBeta2`/accum/`DerivedEvalEvery`/beta-end enter
  IFF their arms win; losers close their adaptivization tickets as measured confirmations.
- **B3d/B3e** → fork-from-checkpoint tails on the c2 run itself (`DerivedWPoseAtEngage` is the
  B3e carrier; LIVE pin/λ_pose consumers enter c3).
- **B4b** → `ComputeDtype("bf16")` on QC ADMIT, else fp32.
- Everything else byte-identical to c1; budget re-anchored to the measured composite; dry-start
  GREEN required.

## 4. TRAINER WIRE-IN QUEUE (behind pid-31576 dry-start exit; exact insertion points in `TRAINER_WIREIN_QUEUE`)

1. `--weight-norm-telemetry` (default TRUE) → emit `weight_norm_row(ep, live_np, ema_np,
   baseline)` at the EMA-verdict site (~L9332 region); baseline restored via `baseline_from_row`
   on resume. Lever auto-unlocks (fail-closed today).
2. `--w-pose-costate-engage` → engage-time λ_pose from the run's OWN verdict d_pose at the
   pose-finish gate (~L11698 region). Until then `DerivedWPoseAtEngage` carries the banked-anchor
   static value.
3. `--verdict-event` (NCDE cadence consumer) — ticket-only, NOT this wave.

## 5. Triality legs + honesty

- **DSL:** 7 lever factories + typed waiver custody in `constants_telemetry_build_wave_20260715.py`
  (this landing's commit).
- **equations:** 4 evaluators registered (`stationarity_window_log_midpoint_v1`,
  `costate_w_pose_engage_ratio_v1`, `verdict_cadence_amortization_v1`,
  `build_wave_custody_identity_v1` — the last explicitly NON-DERIVATIONAL identity custody);
  all thin executable adapters over ALREADY-REGISTERED laws (no new physics claimed).
- **DAG:** FEED-515build appended (sub015 DAG).
- **verdict_scope: instance** — every number above is a derivation or a citation of an existing
  measured anchor on THIS config family (`v9_cgauge_ideal_mod19` tags on every LawRef input);
  nothing here is a score claim; sec/ep economics are [macOS-MLX advisory] NON-PROMOTABLE.
- **NOT goal progress:** these are MEANS. Pointer 0.19108 UNMOVED; it moves only through a
  byte-closed n600 `upstream/evaluate.py` exact row.
