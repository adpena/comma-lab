# CAPSTONE batch-size fixed-point + B=64 launch-ready rescaled schedule (2026-06-21)

**Operator ask (2026-06-21):** work out the concrete first-order rescaled schedule a larger-batch capstone
vehicle would start from, and fold this session's throughput learnings into the capstone materials. **DESIGN
MEMO ONLY** — no training files edited, no run launched, no archive emitted.

**Authority discipline (CLAUDE.md, binding).** Every number below the bs=8 SOURCE row is `[predicted]` /
first-order-derived, **NON-PROMOTABLE** (`promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`). The B=64 schedule is a **search INITIALIZATION**, NOT a validated
curriculum — the explicit point of §3 is that it MUST be empirically re-solved. NO MPS authority. NO paid
dispatch. Pointer UNMOVED 0.19110 [contest-CPU]. Sister/canonical: `optimal_capstone_vehicle_spec_20260611.md`
(the canonical vehicle spec; assumes bs≤8 — THIS memo is the batch-size axis it did not cover) +
`throughput_floor_latency_bound_bs8_scorer_20260621.md` (the measured floor that motivates this).

---

## 1. Why every other scalar is a coordinate of a batch-size fixed point (the coupling)

SGD/Adam is a discretized stochastic differential equation. The minibatch gradient covariance carries B:
`Cov[ĝ_B] ≈ C₁/B`, so the update `θ←θ−η·ĝ_B` realizes a Langevin diffusion whose **temperature `T ∝ η/B`**.
The dynamics — which minimum, how flat, how much implicit regularization — depend on θ through `η/B`, not η and
B separately. So **every LR in the curriculum is really a choice of temperature η/8.** Change the 8 and every
temperature moves → a different anneal → different converged d_seg/d_pose.

Four scalars are explicitly B-coupled, each via a DIFFERENT power law (the non-obvious part):

| Scalar | B-invariant it encodes | Derivation | Scaling law |
|---|---|---|---|
| LR η (AdamW) | temperature, Adam-preconditioned | Malladi 2022 SDE for adaptive optimizers | **η ∝ √B** |
| LR η (Muon, stage 8) | noise-floor random-walk step size | orthogonalization normalizes away the noise magnitude (§2.1) | **≈ B⁰ (B-INVARIANT); ×B^{1/4} upper probe** |
| EMA decay ρ | shadow-averaging window **in epochs** | τ_epochs = B / [N(1−ρ)] held constant | **(1−ρ) ∝ B** |
| σ noise | regularizer dose/epoch | injected variance/epoch = σ²·N/B held | **σ ∝ √B** |
| C1a λ | regularizer effect/epoch (η-coupled) | (N/B)·η·λ held, with η∝√B | **λ ∝ √B** |
| epochs | **data exposure = memorization budget** | epochs·N is B-invariant | **HOLD (∝ B⁰)** |

```
KEY non-obvious results:
• EMA scales LINEARLY (1−ρ ∝ B, ×8) while LR scales as √B (×2.83). A naive
  "scale everything by √B" MIS-SETS the EMA → re-introduces the shadow-lag bug
  (the one we just fixed). EMA window lives in STEPS (∝ N/B), LR lives in
  temperature (∝ η/B) — different denominators of B.
• epochs do NOT scale with B. epochs·N (data exposure) is the single-video
  MEMORIZATION budget and is B-invariant. Holding epochs at B=64 means 8× FEWER
  optimizer UPDATES (296K vs 2.22M) — that is the regime to validate, NOT a bug.
  (Scaling epochs ∝ B to preserve update count would give ZERO wall-clock win —
  the wrong invariant.)
```

## 2. The concrete B=64 first-order rescaled schedule (k = B/8 = 8, √k = 2.828)

SOURCE (exact, from vendored PR95 `src/stages/*.py`, bs=8, 29,650 epochs) → B=64 STARTING point:

| Stage | epochs (HOLD) | adamw_lr (×√8) | muon_lr | ema ρ ((1−ρ)×8) | λ (×√8) | σ (×√8) |
|---|---|---|---|---|---|---|
| 1 CE | 3000 | 1e-3 → **2.83e-3** | — | 0.999 → **0.992** | 0 | 0.2→0.57* |
| 2 softplus | 5650 | 1e-3 → **2.83e-3** | — | 0.992 | 0 | 0.2→0.57 |
| 3 smooth | 1500 | 1e-4 → **2.83e-4** | — | 0.992 | 0 | 0.2→0.57 |
| 4 QAT | 500 | 1e-4 → **2.83e-4** | — | 0.992 | 0 | 0.2→0.57 |
| 5 C1a-L7 | 9000 | 3e-5 → **8.49e-5** | — | 0.992 | 0.01→**0.028** | 0.2→0.57 |
| 6 λ-sweep | 2000 | 3e-5 → **8.49e-5** | — | 0.992 | 0.02→**0.057** | 0.2→0.57 |
| 7 σ-sweep | 3000 | 3e-5 → **8.49e-5** | — | 0.992 | 0.02→**0.057** | 0.1→0.28 |
| 8 Muon | 5000 (extend if under-polished) | 1e-5 → **2.83e-5** (AdamW part, √8) | 2e-4 → **2e-4 (×1, B-invariant; ≤3.4e-4 probe)** | 0.992 | 0.02→**0.057** | 0.1→0.28 |

\* σ flagged: σ has a DUAL role (pure regularizer dose ∝√B AND a uint8-quant-noise simulation that is
B-INVARIANT). The loop's explicit eval_roundtrip `round()` already does the physical quant sim, so σ here is
mostly a pure regularizer → √B is defensible; but if it over-regularizes, sweep DOWN toward the unscaled value.
λ likewise has a fixed brotli-target role on top of the dose → √B start, sweep.

**Batching note:** B=64 gives 10 steps/epoch with a ragged last batch of 24 (600 = 9·64 + 24). For clean even
division prefer **B=60** (10 even batches, k=7.5) or **B=75** (8 even batches, k=9.375); B=64 keeps the k=8
scaling arithmetic exact at the cost of the ragged step. Recommend **B=60** for production cleanliness, B=64 if
power-of-2 GPU kernels matter more than the ragged batch.

### 2.1 Why Muon's LR is ≈ batch-INVARIANT (the derivation; replaces the earlier √8→×8 sweep)

Muon's step is `θ ← θ − η·polar(M)`, `M = β·M_prev + ĝ_B`, where `polar(M)=UVᵀ` (Newton-Schulz sets every
singular value of `M=UΣVᵀ` to 1). The decisive property: **`polar(c·M) = polar(M)`** for any scalar `c>0` — the
update magnitude is `η × unit-spectral-norm`, **decoupled from the gradient (and thus the minibatch-noise)
magnitude**. SGD/Adam carry `‖ĝ_B‖` — and its `1/√B` noise — into the step linearly; Muon does not. So the
SGD `∝B` and Adam `∝√B` rules (both derived from the noise magnitude propagating into the step) **do not
apply**.

Where B enters: write `M = G + N` (G = EMA of the true gradient, N = residual minibatch noise,
`‖N‖ ∝ 1/√(B·τ_eff)`, `τ_eff=(1+β)/(1−β)`). The direction noise is `δ(polar) ∝ ‖N‖/σ(G)`. Two regimes:
- **Signal-dominated** (early/descent, σ(G) large): `δ(polar) ∝ 1/√B` — larger B → cleaner directions, but the
  step MAGNITUDE is η regardless.
- **Noise-floor** (late polish, near the min, σ(G)→0): `polar(G+N) ≈ polar(N)` = a RANDOM unit matrix. Here
  `‖N‖∝1/√B` is **normalized away** — `polar(N)` is unit no matter how small N is. The noise-floor random-walk
  step = **η, INDEPENDENT of B**.

Stage 8 (Muon-finetune) IS the d_seg-finishing polish — it lives in the noise-floor regime, where the step that
sets the final d_seg resolution is η, B-invariant. **Therefore `muon_lr = 2e-4` is HELD at B=64** (×1).
Inflating it (√8→5.66e-4, or ×8→1.6e-3) would enlarge the noise-floor random walk √8–8× and *worsen* the very
d_seg resolution the stage exists to sharpen. A mild `×B^{1/4}≈1.68` (→3.4e-4) is the only defensible upward
probe (for the early signal-dominated fraction).

The competing concern — at held epochs, B=64 does **8× fewer Muon steps/epoch** → 8× less total
polish-displacement — is NOT fixed by raising η (that trades away resolution). Fix it, *if a stage-5/8
checkpoint shows under-polished d_seg*, by **extending stage-8 epochs** (the cleaner large-batch momentum makes
each step more productive — the same fewer-updates bet as the AdamW stages, resolved on the LR-preserving
side). This is the principled replacement for "sweep η toward ×8": **hold the fine step, give it more steps.**

## 3. Why this is a STARTING point requiring empirical RE-SOLVE (not a validated schedule)

The first-order laws are only the SDE small-step limit. Three effects break them → empirical re-solve required:
1. **Adam(√B) vs Muon(unknown) divergence** — stages 1-7 and stage 8 obey different scaling exponents; the
   schedule can't be rescaled with one rule. Muon's batch-scaling has no published result (2024 optimizer).
2. **Discretization error** — holding T by raising η as B grows enlarges the per-step jump; the SDE
   approximation degrades at large k, breakdown point for this tiny-model/single-video task UNMEASURED.
3. **Non-convex d_seg landscape** — the minimum SELECTED depends on the full trajectory through the
   high-curvature argmax-boundary Hessian; first-order rescaling preserves local dynamics, not which basin.
4. **The 8× fewer-updates bet** — 296K vs 2.22M updates at held epochs. Whether cleaner large-batch gradients
   reach the same d_seg in fewer updates is THE empirical question. Mitigation: validate at a stage-5
   checkpoint; if under-trained, the FIRST sweep axis is epochs (×1.5–3, partially eating the speedup).

## 4. Session throughput learnings folded into the capstone (results→intelligence)

1. **The bs=8 latency wall IS the reason this spec exists.** Measured: A10G ≈ T4 ≈ M5 Max MPS, all ~11–13
   s/ep → the per-epoch is GPU-INVARIANT → bottlenecked by 75 serial bs=8 optimizer-per-batch steps, not
   hardware. B is the only escape, and it's score-locked for the FAITHFUL run → a capstone (re-solve) lever.
2. **`defer_batch_sync` KEEP-ON (proven bit-identical, +2%).** The B=64 vehicle should use it too (fewer
   per-batch syncs remain, still worth removing). Committed `eb4bcf4cd`, test `test_batch_sync_deferral_bit_identical`.
3. **Pose is numerically fragile across kernel backends — keep fp32-exact.** torch.compile of the FROZEN
   scorers drifted PoseNet 22% (REJECTED); MPS drifts pose 23×. d_seg (argmax) is robust; d_pose (continuous
   MSE on FastViT) is not. The capstone's cross-hardware margin-hinge covers d_seg; pose MUST stay fp32 (no
   compile, no MPS authority). Re-validate the POSE axis on ANY kernel/precision change at B=64.
4. **Validate LOCAL first; Modal only if B=64 becomes GPU-bound.** At bs=8 Modal was NO-GO (~0% faster, $20
   < the verdict cost). But B=64 saturates the GPU → at B=64 a faster GPU COULD finally help (unlike bs=8).
   The M5 Max 128GB handles B=64; if it turns GPU-bound there, THEN a paid A100 is worth re-pricing.
5. **Expected B=64 wall-clock: ~2–4× faster.** 10 steps/epoch (vs 75), each GPU-saturated at ~2–4× the bs=8
   step latency (not 8×) → ~2–4× per-epoch → the ~4-day run → ~1–2 days, IF the fewer-updates bet holds.

## 5. Launch-ready spec (gated, NOT fired)

The B=64 capstone is the canonical-spec vehicle (`optimal_capstone_vehicle_spec_20260611.md` §0:
fresh-init base_ch=24 HNeRV + 28-d latent + FiLM + grid-PE + L1 weight-tie) trained under the §2 B=60/64
rescaled schedule, `--defer-batch-sync`, fp32-exact pose, CPU authority. It is a **new-vehicle build** (a fresh
fixed-point solve), launched only AFTER the current faithful bs=8 decisive run returns its stage-5 verdict
(don't fork the bet before the control reads out). Validation gates: (a) stage-1 d_seg descent matches/beats
the bs=8 trajectory at matched DATA-EXPOSURE (epoch), not matched updates; (b) stage-5 checkpoint d_seg within
the bs=8 band → else sweep epochs; (c) pose axis re-validated fp32 at B=64; (d) byte-close + CPU-authority
exact eval before any score claim.

## NO-FAKE ledger
- MEASURED (this session): bs=8 schedule (exact, vendored source); A10G≈T4≈MPS ~13 s/ep; defer +2% bit-identical;
  compile pose drift 22%; eval-freq ~0%.
- DERIVED (first-order, [predicted]): the B=64 η/ρ/σ/λ rescaling + the ~2–4× speedup estimate + the 296K-update count.
- NOT claimed: the B=64 schedule is NOT validated; no score moved; pointer UNMOVED 0.19110; this is a search
  initialization for a future re-solve, gated behind the bs=8 verdict.

## Cross-references
- `optimal_capstone_vehicle_spec_20260611.md` — canonical vehicle spec (architecture; this memo adds the batch axis).
- `throughput_floor_latency_bound_bs8_scorer_20260621.md` — the measured latency floor that motivates B>8.
- `yousfi_r3_taper_marginhinge_e5_stage1_verdict_20260621.md` — the live faithful bs=8 run this gates behind.
- Malladi et al 2022 "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms" (the Adam √B rule);
  Goyal et al 2017 (SGD linear rule); Smith & Le 2018 (noise scale); Mandt et al 2017 (SGD-as-SDE).
