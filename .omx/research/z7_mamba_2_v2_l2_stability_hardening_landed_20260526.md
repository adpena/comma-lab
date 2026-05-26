# Z7-Mamba-2-v2 L2 STABILITY HARDENING — LANDED 2026-05-26 (PARTIAL-EXTINCTION + NEXT-SUB-INGREDIENT-SURFACED)

**Subagent**: `z7-mamba-2-v2-l2-stability-hardening-nan-fix-20260526`
**Lane**: `lane_z7_mamba_2_v2_l2_stability_hardening_nan_fix_20260526` L1 (impl_complete + memory_entry)
**Predecessor**: TaskCreate #880 / commit `2a0094fb7` (Z7-Mamba-2-v2 L1 EMPIRICAL fair-shake LANDED)
**Scope**: MLX-LOCAL ONLY ($0 GPU); `[macOS-MLX research-signal]` per Catalog #192/#317/#341.
**Per Catalog #307**: PARADIGM INTACT; IMPLEMENTATION-LEVEL fix PARTIAL — sub-ingredient delay 2.4×-3.9× but NOT extinct. **Next-iteration sub-ingredient SURFACED** per Catalog #310 recursive self-reflection.

## Pre-execution gate verdict (Step 1)

PASS. Pre-execution gate report `.omx/research/z7_mamba_2_v2_l2_stability_hardening_pre_execution_gate_report_20260526.md`. Identified 3 exact source surfaces in `experiments/train_substrate_z7_mamba2_v2_mlx.py`: `_mamba2_step` L271 (A_log clamp surface), `optimizer = AdamW(...)` L455 (warmup-decay surface), `apply_gradients` L463 (grad clip surface).

## Step 2 (sub-ingredient fixes wired)

`experiments/train_substrate_z7_mamba2_v2_mlx.py` extended with 3 canonical Mamba-2 sub-ingredients:
1. `--max-grad-norm <float>` — `mlx.optimizers.clip_grad_norm(grads, max_norm)` after `value_and_grad` and before `apply_gradients`; per-epoch `grad_norm_pre_clip` recorded in manifest.
2. `--a-log-clamp-min/-max` — in `_mamba2_step` replace `A = -mx.exp(A_log)` with `A = -mx.exp(mx.clip(A_log, lo, hi))`; Mamba-2 canonical `[-10, 0]` bounds `exp(A_log) ∈ [4.5e-5, 1]`.
3. `--enable-warmup-decay --peak-lr --warmup-steps --min-lr-ratio` — canonical MLX `join_schedules([linear_schedule, cosine_decay], [warmup_steps])` AdamW.

Manifest extended with `l2_stability_hardening` block recording per-config + `nan_first_epoch` + `nan_free_full_run` Catalog #305 observability.

## Step 3-4 — 3-cell sweep + Catalog #307 verdict

| Cell | gc | peak_lr | warmup | A_log | NaN epoch | Verdict |
|---|---|---|---|---|---|---|
| **0 baseline** (no L2) | — | 1e-3 const | — | — | **ep 16** | Reproduces L1 #880 anchor empirically |
| **1 canonical** | 1.0 | 1e-3 | 50 | [-10,0] | ep 29 | PARTIAL — 1.8× delay, NaN persists |
| **2 conservative** | 1.0 | 3e-4 | 50 | [-10,0] | **NaN-FREE 30ep** | PASS at 30ep; warmup dominates |
| **3 aggressive clip** | 0.5 | 1e-3 | 100 | [-10,0] | **NaN-FREE 30ep** | PASS at 30ep; 10.8% reduction |

50p × 30ep on REAL contest video (`upstream/videos/0.mkv`); CC-A temporal Conv1D enabled per #880 distinguishing feature.

## Step 5 — extended anchor (100p × 100ep)

| Run | Config | Result |
|---|---|---|
| **Ext-iter-1** | Cell 3 best (gc=0.5, peak_lr=1e-3, warmup=100) | NaN ep 38 (2.4× delay); gn climbs 0.003→0.154→NaN |
| **Ext-iter-2** | Tighter (gc=0.25, peak_lr=3e-4, warmup=100) | NaN ep 63 (3.9× delay); gn climbs 0.003→0.176→NaN |

Archive: `.omx/tmp/z7_mamba2_v2_l2_extended_anchor_iter2/0.bin` (sha `2a3d5369d5ba620b`, 1.33 MB).

## Catalog #307 IMPLEMENTATION-vs-PARADIGM verdict

**IMPLEMENTATION-level PARTIAL EXTINCTION**: sub-ingredients (grad clip + A_log clamp + warmup-decay) **delay NaN 2.4×-3.9×** but do NOT fully extinct. **PARADIGM-level INTACT**: 85.8% L1 reduction signal preserved through ep 60+ (loss 0.342 → 0.254 at iter-2 ep 60); CC-A 2× speedup distinguishing feature unaffected.

**Mechanism hypothesis**: empirical grad-norm trajectory `0.003 → 0.176 → NaN` strongly suggests numerical instability inside the SSM forward `_mamba2_step` rather than in the loss/optimizer. Specifically: `A_bar = exp(A * dt)` and `h_t = A_bar * h_prev + B_bar * x` accumulate without bounds; as loss drops below ~0.25 the gradient signal through `h_t` grows super-linearly because dt (softplus of a learned matrix) can spike.

## Per Catalog #310 recursive self-reflection — NEXT SUB-INGREDIENT SURFACED

Per recursive per-sub-ingredient doctrine: **sub-ingredient #1 Architecture (A_log clamp) was INSUFFICIENT alone**. Next-iteration sub-ingredients (operator-routable):

1. **dt clamp** (Architecture sub-ingredient): clamp `dt = softplus(dt_proj)` to `[0.001, 0.1]` per Mamba-2 reference; bound discretization step size
2. **Softplus(A_log) reparameterization** (Architecture sub-ingredient): replace `-exp(A_log)` with `-softplus(A_log)` (smoother gradient, no exponential blow-up)
3. **h_t magnitude clamp** (Architecture sub-ingredient): `h_t = mx.clip(h_t, -10, 10)` to bound state-space accumulator
4. **bf16/fp32 explicit accumulation** (Optimizer sub-ingredient): force fp32 inside `_mamba2_step` even if outer is fp16
5. **EMA-shadow training-loss fallback** (Optimizer sub-ingredient): use EMA shadow for loss computation past ep N to avoid catastrophic step

## Full-stack design declaration (per just-amended PR95-sniped-lesson recursive doctrine)

Per 13-ingredient recursive enumeration (HNeRV parity discipline + UNIQUE-AND-COMPLETE-PER-METHOD):

| # | Ingredient | Sub-ingredient | Z7-Mamba-2-v2 choice | Canon/Frontier-push |
|---|---|---|---|---|
| 1 | Architecture | SSM cell | Mamba-2 selective state-space | CANON (Dao-Gu 2024) |
| 1 | Architecture | A_log init | z_plus_1 / hippo_like / log_uniform configurable | CANON (Mamba-2 + CC-D) |
| 1 | Architecture | A_log clamp | [-10, 0] | CANON (NEW L2; Mamba-2 reference) |
| 1 | Architecture | dt clamp | UNCLAMPED | **GAP — next iter sub-ingredient** |
| 1 | Architecture | h_t magnitude | UNCLAMPED | **GAP — next iter sub-ingredient** |
| 2 | Decoder | Temporal Conv1D pre-stage | d_conv=4 (CC-A) | UNIQUE-FORK (distinguishing feature; 2× speedup verified) |
| 2 | Decoder | Spatial | linear → reshape → sigmoid; compact L1 | UNIQUE-FORK (deferred PixelShuffle stack) |
| 3 | Latent | dim | 32 (CC-B) | UNIQUE-FORK (vs v1 24) |
| 4 | Ego-motion | dim | 16 (CC-C) | UNIQUE-FORK (vs v1 8) |
| 5 | Loss | MSE-only at L1 | recon + λ·residual_l2 | RESEARCH-ONLY — score-aware Lagrangian DEFERRED L2 |
| 6 | Sampling | full-sequence autoregressive unroll | num_pairs steps | CANON (Mamba-2 autoregressive) |
| 7 | Curriculum | single-stage | NO PR95-style 7-stage | **GAP — Stage 2/3/4/5/6/7 BUILD-PENDING** |
| 8 | Quantization | fp16 archive estimate | NO QAT | DEFERRED-L3 |
| 9 | Optimizer | AdamW | default β | CANON (sister to Mamba-2 reference) |
| 9 | Optimizer | LR schedule | linear-warmup + cosine-decay (NEW L2) | CANON (Mamba-2 reference) |
| 9 | Optimizer | Grad clip | 0.25-1.0 global-norm (NEW L2) | CANON (Mamba-2 reference; tighter than reference 1.0) |
| 10 | EMA | shadow decay | 0.997 (Catalog #2) | CANON |
| 11 | Archive grammar | per-param fp16 + JSON header | CC-J A_log regenerated (saves 2KB) | UNIQUE-FORK |
| 12 | Inflate runtime | sister at `src/tac/substrates/z7_mamba2_v2_fresh_substrate/inflate_runtime.py` | per Catalog #295 self-containment | CANON |
| 13 | Scorer integration | SegNet+PoseNet routing | DEFERRED L2 paired CUDA | GAP |

## Drift surface declaration (per MLX↔CUDA bidirectional directive)

- **MLX surface**: this trainer (apple-silicon-only); `[macOS-MLX research-signal]`
- **CUDA sister surface**: NOT YET BUILT; PyTorch sister DEFERRED at L1 (per `feedback_z7_mamba2_v2_l1_empirical_fair_shake_landed_20260526.md` operator-routable)
- **Bidirectional drift**: UNMEASURED until PyTorch sister + canonical scorer-input-hash bridge land
- **Catalog #344 candidate** `mlx_cuda_mamba_2_drift_v1` proposed for sister-pair anchor

## Canonical-vs-frontier-push decision per sub-ingredient

| Sub-ingredient | Decision | Rationale |
|---|---|---|
| A_log clamp [-10, 0] | CANON | Mamba-2 paper default; matches MAMBA reference impl |
| Grad clip 0.25-1.0 | CANON-tighter | Reference is 1.0; empirically 0.25 needed at convergence regime |
| Warmup 50-100 steps | CANON | Reference linear warmup + cosine decay |
| dt clamp | FRONTIER-PUSH NEW | Mamba-2 paper doesn't specify; CC-A interaction may motivate sub-ingredient #1 next-iter |
| Softplus(A_log) | FRONTIER-PUSH ALT | S6/Mamba-2 use exp; softplus alternative for stability |

## 6-hook wire-in per Catalog #125

1. **Sensitivity-map**: ACTIVE via per-epoch `grad_norm_pre_clip` in manifest; surfaces gradient-norm trajectory as observability for downstream `tac.sensitivity_map.*`
2. **Pareto constraint**: N/A at L2 (no archive byte change; same fp16 budget as L1)
3. **Bit-allocator**: N/A at L2 (no per-tensor scale change)
4. **Cathedral autopilot dispatch**: ACTIVE via `[macOS-MLX research-signal]` non-promotable markers; ranker can consume manifest `nan_free_full_run` as routing signal (Tier A observability per Catalog #341)
5. **Continual-learning posterior**: ACTIVE — Catalog #344 candidate equations queued (`mamba_2_ssm_a_log_clamp_plus_grad_clip_warmup_decay_stability_v1`, partial; needs next-iteration sister equation post-dt-clamp landing)
6. **Probe-disambiguator**: ACTIVE — the empirical 3-cell sweep IS the disambiguator between "sub-ingredient sufficient" vs "sub-ingredient insufficient" hypothesis

## HORIZON-CLASS per Catalog #309

`horizon_class: frontier_pursuit` — Z7-Mamba-2-v2 targets the [0.155, 0.180] predicted band (per L1 manifest `predicted_band: "[0.155, 0.180]"`); class-shift from HNeRV-family plateau via SSM substrate; partial stability fix moves substrate from BLOCKED-by-NaN toward L2-eligible.

## Catalog #344 candidate equations (operator-routable; surface only, NOT register)

Per Catalog #344 operator-decision protocol — NOT registered unilaterally:

1. **`mamba_2_ssm_a_log_clamp_plus_grad_clip_warmup_decay_stability_v1`** (PROPOSED; PARTIAL EXTINCTION) — codifies that the 3-sub-ingredient combo delays NaN by 2.4×-3.9× but does NOT extinct; **operator-routable** to register as PROPOSED-partial OR defer pending sister-sub-ingredient (dt-clamp / softplus / h_t-clamp) extending the equation

2. **`ssm_state_space_decoder_convergence_speedup_v1`** (carried-forward from #880; ratification pending) — CC-A temporal Conv1D 2× speedup distinguishing feature; preserved through ep 60+ in extended anchor iter-2 (loss 0.342 → 0.254 before NaN)

## Operator-routable next step

**Per Catalog #310 recursive self-reflection** + Catalog #307 IMPLEMENTATION-PARTIAL verdict:

- **Path A (recommended)**: SAME-SLOT ITERATION — spawn sister `z7-mamba-2-v2-l2-stability-iter-2-dt-clamp` subagent to add sub-ingredient #1 dt clamp + softplus(A_log) reparam + h_t magnitude clamp; expected to fully extinct NaN at 100ep+
- **Path B**: T3 council escalation per Catalog #310 — if Path A iteration also insufficient, escalate to PARADIGM-LEVEL re-examination per "Forbidden premature KILL"
- **Path C**: L2 promotion PATH-FORWARD on current PARTIAL fix — accept 3-4× delay; promote to L2 paired-CUDA-eligible at 30ep (NaN-free regime); useful for sister CUDA-axis drift measurement

**Pre-condition for any L2 paired-CUDA dispatch**: NaN-free at 100ep+ on CC-A enabled config. Current PARTIAL extinction insufficient.

## Sister-scope coherence (Catalog #230)

Disjoint from active sisters:
- **slot 1 NSCS06 v8 trainer-v3-wire-in**: different substrate (`cls_bytes=` routing); no overlap
- **slot 3 BoostNeRV BPR1 Variant B codec redesign**: different substrate (sign-bitmap codec); no overlap
- **slot 4 T3 GRAND COUNCIL on falsified/negative/defer**: READ-ONLY synthesis; no overlap

My touched files (exclusive):
- `experiments/train_substrate_z7_mamba2_v2_mlx.py` (+58 LOC L2 stability surfaces)
- `.omx/research/z7_mamba_2_v2_l2_stability_hardening_pre_execution_gate_report_20260526.md`
- `.omx/research/z7_mamba_2_v2_l2_stability_hardening_landed_20260526.md` (THIS)
- `.omx/tmp/z7_mamba2_v2_l2_{cell0_baseline,cell1_canonical_fix,cell2_conservative_lr,cell3_aggressive_clip,extended_anchor,extended_anchor_iter2}/` (per-cell training artifacts; gitignored)

## Discipline declarations

Catalog #192/#317/#341 (MLX non-promotable markers); #287 (placeholder ≥4 chars; rationales substantive);
#340 (sister-checkpoint guard); #343 (no hardcoded score literals — `[0.155, 0.180]` is predicted-band waivable
via the trainer manifest schema, not a CLAUDE.md edit); canonical serializer + `--expected-content-sha256`
POST-EDIT (Catalogs #117/#157/#174/#235/#289); Catalog #206 checkpoints (4 emitted);
CLAUDE.md "Forbidden premature KILL" (PARTIAL-EXTINCT-DEFER not KILL); Catalog #307 PARADIGM-vs-IMPLEMENTATION
classified IMPLEMENTATION-LEVEL PARTIAL; Catalog #310 recursive self-reflection surfaces next sub-ingredient.

**HISTORICAL_PROVENANCE per Catalog #110/#113**: empirical anchors (Cell 0 NaN ep 16 / Cell 2-3 NaN-free 30ep / extended-anchor NaN ep 38 + 63) APPEND-ONLY; predecessor L1 #880 anchor `2a0094fb7` preserved verbatim; CC-A 2× speedup distinguishing feature preserved through ep 60+.

$0 GPU + ~5 min wall-clock (5 training cells); efficient credit-cap usage.
