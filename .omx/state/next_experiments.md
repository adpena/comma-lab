# Next Experiments -- 2026-04-12

## Promoted floor: 1.33 (dilated_h64, exact_current)
## Auth score: 1.97 (robust_current, Lightning ep851)
## Proxy score: 0.9238

## Tripartite Pact Consensus (Yousfi + Fridrich + Contrarian + Karpathy + Tao)

### Priority 1: GPU Lane (high ceiling)
1. `exp1_fridrich_proper.py` -- Fridrich constrained gen, 100 frames, 2000 steps
   - Pre-registered: seg < 0.03, pose < 0.1, TV < 1.0
   - Smoke showed seg=0.025, pose=0.078 -- near-feasible
2. `exp2_tiny_dp_sims_proper.py` -- Tiny DP-SIMS, 100 frames, 5000 steps
   - Pre-registered: score < 1.0 including rate at 78KB FP4
3. `exp3_lbfgs_refinement.py` -- L-BFGS polish, 10 steps, chains after exp1/exp2

### Priority 2: CPU Lane (safe gains)
4. `exp4_cpu_trick_stack.py` -- all tricks independently then stacked
   - CRF sweep, quantization rounding, TTO, multi-pass, deblock

### Priority 3: Infrastructure
5. `exp5_auth_scorer_setup.py` -- Lightning auth scorer verification

## Decision Gates
- 2026-04-17: GPU path decision (Fridrich vs DP-SIMS)
- 2026-04-21: Lock final approach
- 2026-05-03: DEADLINE

## Killed Techniques (NEVER retry)
- KL distill loss mode
- Adaptive rebalance weights
- PoseNet gradient caps
- SegNet loss weight > 100
- Brightness shift trick
- PSD architecture
