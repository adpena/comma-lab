---
name: Full arbitrariness audit catalog 2026-04-27 — every lane's heuristic knobs + fix strategy
description: Per user mandate "no arbitrariness, extreme optimization, optimal engineering everywhere possible". Catalogs ALL hard-coded heuristics across 19 lanes + maps each to canonical optimal-fix strategy (Lagrangian rate-distortion, Bayesian sweep, DARTS arch search). 8 lanes have V2 fixes in flight; 11 still have arbitrary constants.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**The mandate** (user 2026-04-27): "no arbitrariness and extreme optimization and learning and optimal engineering everywhere possible".

**The audit** (full):

| Lane | Arbitrary Knob | Optimal Fix Strategy | V2 Status |
|------|---------------|---------------------|-----------|
| Ω-V1 | water-fill α=0.5, hard-coded budget | Lagrangian dual ascent (V2 in flight) | DONE — V2 dispatched |
| W | K=30, weight=5.0 | Continuous per-pair Lagrangian | DONE — V2 dispatched |
| SI | threshold_quantile=0.5, gradient-norm | Lagrangian on rate-target + Hessian saliency | DONE — V2 dispatched |
| PS | per-class weights "1,5,5,1,1" | Lagrangian on per-class distortion equalization | DONE — V2 dispatched |
| LR | rank=1 | Learnable rank via masked-norm pruning | DONE — V2 dispatched |
| V | mask_half_sim_prob=1.0 cold-start | Annealed schedule | DONE — V2 dispatched |
| LM | centroid math (0.017 correlation) | Endpoint tracking | DONE — V2 dispatched |
| OS | YUV approx, no features_buffer, linear cal | YUV420 planar, RNN state, learnable cal | DONE — V2 dispatched |
| GH | ratio s=2 | DARTS-style learnable ratio | **GAP** |
| A | TTO steps=500, batch-pairs=8 | Convergence-detection auto-stop + dynamic batching | **GAP** |
| S | target_bits=2.5, init=8.0, ramp_frac=0.3 | Bayesian hyperparam sweep | **GAP** |
| I | hidden_dim, latent_grid_res | DARTS or Bayesian | **GAP** |
| F-V3 | int8=50, fp4=500, lr=2.5e-6 | Bayesian QAT schedule | **GAP** |
| V (kl_distill_weight=0.002, 5-phase split) | Lagrangian KL/scorer ratio + auto-phase | **GAP** |
| K | base_ch=24, mid_ch=32 (88K target) | DARTS-style channel-width search | **GAP** |
| D-V2 | phase LRs hand-tuned | Cosine-restart with learnable amplitude | **GAP** |
| G v3 | kl_distill_weight=0.002 | Lagrangian on KL/scorer signal-to-noise ratio | **GAP** |
| Ω | target_bits=600,000 | Lagrangian rate-distortion frontier sweep | **GAP** |
| SZ | block_size=16, ternary clip | Per-layer learnable block size + clip | **GAP** |
| Hessian profiler | top_k=30 (overlaps W) | Continuous (in W-V2) | DONE via W-V2 |

**Three Optimal-Fix Strategies (the meta-pattern)**:

### Strategy A: Lagrangian Rate-Distortion
For knobs that trade off compression rate vs distortion (target_bits, thresholds, weights).
- Add a Lagrangian λ multiplier
- Define loss = distortion + λ × (rate - rate_target)²
- Use dual ascent: λ_{t+1} = λ_t + η × (rate_t - rate_target)
- Annealing schedule (ramp_frac) is itself arbitrary; use SAGA/RAdam-style auto-warmup
- Already applied: Lane S, W, Ω-V2, SI-V2, PS-V2

### Strategy B: Bayesian Hyperparameter Optimization
For knobs that don't have clean rate-distortion duals (LRs, epoch counts, schedules).
- Use Optuna or Ax-style Bayesian sweep
- Define search space + ACQ function (EI or UCB)
- Run on a few trial budgets ($0.30-1.00 per trial)
- Best-trial-wins
- Could apply: Lane A (TTO), Lane S (SC), Lane F-V3 (QAT), Lane V (Quantizr), Lane D-V2 (LR), Lane K (DSConv)

### Strategy C: Differentiable Architecture Search (DARTS)
For arch knobs (channels, ratios, kernel sizes).
- Build supernet with all candidate architectures coexisting via softmax weights
- Train end-to-end; the softmax weights become learnable arch choices
- At convergence, prune to single arch
- Could apply: Lane GH (ratio), Lane K (channels), Lane I (Cool-Chic dims), Lane SZ (block size)

**Council recommendations on prioritization**:

- **Lagrangian fixes are CHEAPEST**: extends existing Lane S/W infrastructure. Highest-EV next batch.
- **Bayesian sweeps are MEDIUM**: each "trial" is a full Vast.ai run ($0.30-2). With $300 budget, can afford ~150 trials. Suitable for the highest-impact lanes.
- **DARTS is EXPENSIVE**: requires supernet training (3-5× cost of single arch). Only worthwhile for Lane K (smaller arch from scratch) where the DARTS arch could BEAT the 88K hand-tuned guess.

**Status (2026-04-27 EOD)**:
- 8 V2 lanes covering arbitrariness in flight (S, W, SI, PS, LR, V, LM, OS, Ω)
- 11 GAPS remain, grouped by strategy A/B/C
- Next dispatch wave: Group A first (Lagrangian fixes for G, Ω target_bits) → Group B (Optuna sweep for one canonical lane) → Group C deferred until Vast.ai budget commits

**Related memories**:
- `project_lane_w_hard_pair_self_compress_premise_20260427`
- `project_lane_omega_bit_budget_hessian_aware_quantization`
- `project_lane_taxonomy_stacking_strategy_20260427`
- `project_self_compression_breakthrough` (foundational SC discovery)
- `feedback_curriculum_must_use_full_score`
