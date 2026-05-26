# Z7-Mamba-2-v2 L2 STABILITY HARDENING — Pre-Execution Gate Report 2026-05-26

**Subagent**: `z7-mamba-2-v2-l2-stability-hardening-nan-fix-20260526`
**Lane**: `lane_z7_mamba_2_v2_l2_stability_hardening_nan_fix_20260526`
**Predecessor**: TaskCreate #880 / commit `2a0094fb7` (Z7-Mamba-2-v2 L1 EMPIRICAL fair-shake LANDED)
**Scope discipline**: MLX-LOCAL ONLY ($0 GPU); per "Remember all on MLX" + Catalog #192/#317/#341 canonical non-promotable markers.

## Operator-pre-approved follow-up scope

Per just-amended **PR95-sniped-lesson recursive per-sub-ingredient doctrine** + sister directives (MLX↔CUDA drift / pushing-frontier-research-on-optim). FIRST sub-ingredient optimization pass on canonical Mamba-2 SSM stability:

- **Ingredient #9 Optimizer** — sub-ingredients: gradient clipping + warmup-decay
- **Ingredient #1 Architecture** — sub-ingredient: A_log clamp (SSM-specific NaN risk)

## Empirical anchor — bug class signature

L1 EMPIRICAL (commit `2a0094fb7`, ed79819194c8517e archive sha) **CONFIRMED**:
- 85.8% MONOTONIC loss reduction (0.340194 → 0.048467) at 50p×15ep, 0.2s wall
- CC-A temporal Conv1D distinguishing feature = 2× convergence speedup (47.6% vs 23.5% at 12ep)
- **NaN AT EP 16-18 across 3 INDEPENDENT RUNS** (LR 1e-3 + 3e-4) — empirically validates 2026-05-18 multi-week stability path memo hypothesis
- Per Catalog #307: PARADIGM INTACT; IMPLEMENTATION-level fix REQUIRED for L2 + paired-CUDA eligibility

## PV scope — exact source locations

Read `experiments/train_substrate_z7_mamba2_v2_mlx.py` (606 LOC):

| Sub-ingredient | Surface | Line(s) | Current state |
|---|---|---|---|
| **#9 Gradient clipping** | `optimizer.apply_gradients(grads, params)` | L463 | NONE — bare unscaled grads |
| **#1 A_log clamp** | `A = -mx.exp(self.mamba_A_log)` | L271 | NONE — unclamped `exp(A_log)` is canonical Mamba-2 NaN risk source per Dao-Gu 2024 + S4 lineage |
| **#9 Warmup-decay** | `optimizer = mlx_optim.AdamW(learning_rate=args.learning_rate)` | L455 | NONE — constant LR |

**Empirically verified MLX APIs** (canonical):
- `mlx.optimizers.clip_grad_norm(grads, max_norm) -> (clipped_grads, total_norm)` (signature verified)
- `mlx.optimizers.linear_schedule(init, end, steps)`
- `mlx.optimizers.cosine_decay(init, decay_steps, end)`
- `mlx.optimizers.join_schedules([s1, s2], [boundary])` (canonical warmup-then-decay composition)

## Implementation plan per recursive per-sub-ingredient doctrine

### Step 2 — Apply 3 canonical Mamba-2 stability sub-ingredients

1. **Gradient clipping** (`--max-grad-norm`, default=1.0; sweep {0.5, 1.0, 2.0})
   - Inject `mlx.optimizers.clip_grad_norm(grads, args.max_grad_norm)` AFTER `loss_and_grad` returns, BEFORE `apply_gradients`
   - Emit `grad_norm` per epoch into `per_epoch_metrics` (observability per Catalog #305)

2. **A_log clamp** (`--a-log-clamp-min`, default=-10; `--a-log-clamp-max`, default=0; sweep canonical [-10,0] vs unclamped)
   - In `_mamba2_step` at L271: replace `A = -mx.exp(self.mamba_A_log)` with `A = -mx.exp(mx.clip(self.mamba_A_log, args.a_log_clamp_min, args.a_log_clamp_max))`
   - Canonical Mamba-2 reference: clamp `A_log ∈ [-10, 0]` so `exp(A_log) ∈ [4.5e-5, 1]` (state-space spectrum stays bounded)

3. **Warmup-decay** (`--warmup-steps`, default=50; `--peak-lr`, default=1e-3; `--min-lr-ratio`, default=1e-2)
   - Replace bare AdamW with `mlx.optimizers.AdamW(learning_rate=join_schedules([linear_schedule(0, peak_lr, warmup_steps), cosine_decay(peak_lr, total_steps - warmup_steps, peak_lr * min_lr_ratio)], [warmup_steps]))`

### Step 3 — 3-cell stability sweep + baseline

| Cell | gc | peak_lr | warmup | A_log clamp | Expected | Verification |
|---|---|---|---|---|---|---|
| 0 (baseline) | none | 1e-3 (const) | none | none | NaN ep 16-18 | Reproduces L1 anchor |
| 1 | 1.0 | 1e-3 | 50 | [-10, 0] | NaN-free 30ep | Canonical fix |
| 2 | 1.0 | 3e-4 | 50 | [-10, 0] | NaN-free 30ep | Conservative LR |
| 3 | 0.5 | 1e-3 | 100 | [-10, 0] | NaN-free 30ep | Aggressive clip + longer warmup |

50p×30ep per cell (CC-A enabled per #880 distinguishing-feature validation).

### Step 4 — NaN-FREE convergence verdict per Catalog #307

- **IF all 3 fix-cells reach 30ep NaN-free** → STABILITY BUG CLASS EXTINCTED at IMPLEMENTATION level
- **IF some cells still NaN** → identify INSUFFICIENT sub-ingredient; recursive next iteration
- **IF all FAIL** → paradigm-level escalation per Catalog #307 + T3 council escalation (DEFERRED-pending-research per Catalog #310; alternative reducers = bf16-only / fp32 explicit accumulation / sister stable SSM parameterization e.g. softplus(A_log))

### Step 5 — Extended anchor (post-stability-fix)

- 100p×100ep (stage-2-style); CC-A enabled; best stability cell config
- Verify 85.8% L1 loss reduction PRESERVED post-fix (no convergence-speed regression)

### Step 6 — Catalog #344 candidate equations (surface as operator-routable, NOT register unilaterally)

- `mamba_2_ssm_a_log_clamp_plus_grad_clip_warmup_decay_stability_v1` (NEW; sub-ingredient stability)
- `ssm_state_space_decoder_convergence_speedup_v1` (sister from #880; CC-A temporal Conv1D 2× speedup; awaits operator approval)

## Sister-scope discipline (Catalog #230 ownership map)

- **NSCS06 v8 trainer-v3-wire-in** (slot 1): different substrate; `cls_bytes=` pack_archive routing — DISJOINT
- **BoostNeRV BPR1 Variant B codec redesign** (slot 3): different substrate; sign-bitmap codec — DISJOINT
- **T3 GRAND COUNCIL on falsified/negative/defer verdicts** (slot 4 over-cap): READ-ONLY synthesis — DISJOINT

**My touched files (exclusive)**:
- `experiments/train_substrate_z7_mamba2_v2_mlx.py` (add CLI flags + clip_grad_norm + A_log clamp + warmup-decay)
- `.omx/research/z7_mamba_2_v2_l2_stability_hardening_pre_execution_gate_report_20260526.md` (THIS)
- `.omx/research/z7_mamba_2_v2_l2_stability_hardening_landed_20260526.md` (landing memo Step 7)
- `.omx/tmp/z7_mamba2_v2_l2_stability_*/` (per-cell training artifacts; gitignored)

## Discipline declarations (all gates honored)

- Catalog #192/#317/#341 (MLX non-promotable markers); Catalog #287 (placeholder ≥4 chars);
  Catalog #340 (sister-checkpoint guard); Catalog #343 (no hardcoded score literals);
  Canonical serializer + `--expected-content-sha256` POST-EDIT (Catalogs #117/#157/#174);
  Catalog #206 (checkpoints ~10 tool uses); CLAUDE.md "Forbidden premature KILL" (DEFER-not-KILL);
  Per Catalog #307: classify PARADIGM-vs-IMPLEMENTATION-vs-SUB-INGREDIENT.

## Operator-routable next step (post-execution)

- IF stability extincted → L2 promotion path = PR95-style multi-stage curriculum BUILD per Full-stack design declaration
- IF still NaN → T3 council escalation per Catalog #310 recursive self-reflection protocol

---
*Pre-execution gate report; awaits Step 2-7 execution.*
