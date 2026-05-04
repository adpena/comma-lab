---
name: ALL-IN — push 6-month plan aggressively in time remaining, no shortcuts no conservatism, continue to Shannon floor regardless
description: 2026-04-30 ~10:10 CDT user mandate: "we are pushing for the six month plan in the time remaining to us as aggressively as possible no shortcuts or conservatism and we will just be happy and grateful with where we end up but still continue our work to the shannon floor regardless for the science; iland znd design and impleent all". This is the DEFINITIVE strategy directive. Spawn ALL paradigm shifts + ALL Phase 2/3/4 lanes in parallel. Wall-clock priority is supreme. Final contest submission still requires user approval but the dev work is unconstrained.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The mandate

User: "we are pushing for the six month plan in the time remaining to us as aggressively as possible no shortcuts or conservatism and we will just be happy and grateful with where we end up but still continue our work to the shannon floor regardless for the science; iland znd design and impleent all"

## Decoded directive

1. **Compress 6-month plan into time remaining** before contest deadline → maximum parallelism
2. **NO shortcuts, NO conservatism** → full extreme rigor, all paradigm shifts attempted
3. **Happy and grateful** with whatever lands at deadline → result-tolerant, process-rigorous
4. **Continue to Shannon floor regardless** → POST-deadline work continues for the science
5. **Land + design + implement ALL** → no Tier sequencing, no budget gates, all in flight

## Implementation

### Phase 1 (deadline-critical, currently)
- Lane G v3 = 1.05 [contest-CUDA] (anchor, no further work needed)
- Lane PFP16 dispatch — **dispatch NOW**
- Lane Pint12-PCA impl + dispatch — **impl + dispatch NOW**
- Lane 12/17/19/20/8 contest-CUDA results (in flight)
- HM-S result harvest (imminent)
- SC++ V5 recovery (block_fp_codec tolerance)
- All Phase 1.5 lanes (Joint-ADMM coordinator, J-NWC shared corpus)

### Phase 2 ACCELERATE (Weeks 2-4 → compress to days)
- Lane 10 ADMM real-codec → contest-CUDA validation
- Lane 12 NeRV mask codec → Level 3 (contest-CUDA)
- Lane 17 IMP 10-cycle → mid-cycle (don't wait, parallel)
- Lane 19 SegNet logit-margin → Level 3
- Lane 20 Ballé hyperprior → Level 3
- Wavelet mask domain (Mallat) — **DESIGN NOW, IMPL NOW**
- Full DARTS-S sweep — **PARALLEL DISPATCH**
- Full STC clean-source rebuild — **DESIGN + IMPL**
- Joint-ADMM 4-stream deployment — **IMPL + INTEGRATION**
- NeRV mask codec full deployment — **PARADIGM α SHIFT**

### Phase 3 (Weeks 5-12 → compress to 1-2 weeks)
- Multi-pass compress (Lane 8 GPU inner-step generalization)
- Bit-level archive optimization — **scaffold landed (#295), push to L2**
- MDL/Bayesian (MacKay) — **scaffold landed (#295), push to L2**
- Full IMP 10-cycle (Lane 17 in flight)
- RAFT/radial pose — **#295 design landed, IMPL NOW**
- SegNet logit-margin boundary (Lane 19 in flight)
- Ballé hyperprior (Lane 20 in flight)
- Decoder systems rewrite — **DESIGN + scope**
- Sensitivity-map module + GPU dispatch — **#275 in flight, push to completion**
- Ω-W-V3 (PoseNet-sensitivity-weighted layer protection) — **DESIGN + IMPL + DISPATCH** — paradigm shift β implementation
- C3 coordinate-MLP residual codec — **DESIGN + IMPL**
- Self-Compressing NN — **DESIGN + IMPL**
- Custom binary container (Carmack pattern) — **IMPL**

### Phase 4 (Weeks 13-24 → compress to ~1 week)
- Integration: optimal stack composition per Grand Council #294 verdict
  - representation → prediction → quantization → hyperprior → arithmetic → pack
  - Sensitivity-map → Self-Compress NN → IMP → Lane 19 logit-margin → NeRV mask + RAFT pose preimage (parallel) → Ω-W-V3 + wavelet residual + STC clean-source + PFP16 + LCT → Joint-ADMM + Ballé hyperprior + Lane 16 MDL ranking → arithmetic + bit-level optimizer + deterministic ZIP
- Paper harness (arXiv submission preparation)
- arXiv 2604.24658 Agent-Native Research Artifact integration
- Final stack: predicted **0.20 central, [0.18, 0.30] band** [prediction]
- Sub-Quantizr 0.33 in 1 month: 30% probability
- Shannon 0.28 floor in 6 months: 15% probability

## Wall-clock parallelization (operational)

### Currently active GPU fleet (5 Vast.ai 4090s)
- 35885106 HM-S (eval phase, completing soon)
- 35899275 Lane 17 IMP (cycle 1, 80h ETA)
- 35899672, 35899702, 35899850 (contest-CUDA wave retries — Lane 12/19/8)

### Capacity to scale to
- Vast.ai 4090s: 10+ simultaneous before rate-limit
- Modal A10G/T4: ~$70 credits, ~4-6 simultaneous
- bat00 RTX 2070S → 3090: 1 instance, free, WSL2 GPU passthrough
- Kaggle T4/P100: 2 sessions × 30hr/wk free

### Spawn order (next dispatches, all parallel)
1. Lane PFP16 (~30 min on 4090) → #303 to dispatch
2. Lane Pint12-PCA (~30 min on 4090, 30 min impl first) → #303 or new agent
3. SC++ V5 recovery (~15 min on 4090 after block_fp_codec tol fix) → #303 to dispatch
4. Sensitivity-map dispatch (~1-2h on 4090) → respawn #275 if dead
5. Ω-W-V3 design + impl + dispatch (~1 day dev + $1-2 GPU) → new agent
6. Wavelet mask domain impl + dispatch (~1 day dev + $1-2 GPU) → new agent
7. C3 coordinate-MLP residual codec impl + dispatch (~3 days dev) → new agent
8. Self-Compressing NN impl + dispatch (~3 days dev) → new agent
9. Custom binary container impl (Carmack) (~1 day dev) → new agent
10. Decoder systems rewrite design (~2 days) → new design agent
11. All-scores forensic hidden-gem recoveries (60% of killed = bugs) → #303 to iterate

## Final-approval gate (still strict)

The user MUST approve before:
- Submitting to comma.ai contest
- Public PR / leaderboard entry
- arXiv paper publication
- Cloudflare site URL broadcast (per CLAUDE.md Strategic Secrecy Rule)

NO approval needed for:
- Individual lane dispatches (any cost)
- Council reviews / design docs
- Implementation iterations
- Adversarial review rounds
- Memory updates / commits
- Vast.ai / Modal scaling

## Cross-refs

- feedback_no_monetary_commit_20260430.md
- feedback_priority_time_to_floor_with_final_approval_20260430.md
- feedback_budget_30_day_team_parallel_20260429.md (the original "team to parallelize" mandate)
- project_grand_council_paradigm_shift_to_shannon_floor_20260430.md (#294 paradigm shifts α/β/γ/δ/ε/ζ/η/θ/ι)
- project_6month_strategic_plan_20260429.md (the original 6-month plan)
- project_phases_2_3_4_design_implementation_math_provenance_20260429.md (per-lane Phase 2/3/4 designs)
