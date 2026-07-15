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
