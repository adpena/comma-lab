# Current Focus -- 2026-04-12T22:00:00-05:00

## Floor
- **Official score**: 1.33 (exact_current, dilated_h64, Modal A10G)
- **Auth score**: 1.97 (robust_current, Lightning ep851)
- **Proxy score**: 0.9238 (robust_current, Lightning ep851)

## Active Experiments (Tripartite Pact + Karpathy + Tao)

### GPU Lane
1. **Fridrich Constrained Gen (PROPER)** -- exp1_fridrich_proper.py
   - 100 frames, 2000 steps, relaxed boundaries (seg < 0.03, pose < 0.1)
   - S-UNIWARD cost weighting + ego-motion flow constraint
   - Success: PROMOTE to full 1200. Kill: pose diverges or seg > 0.10

2. **Tiny DP-SIMS (PROPER)** -- exp2_tiny_dp_sims_proper.py
   - 100 frames, 5000 steps, channels=(32,16,8,4), 78KB FP4
   - Success: score < 1.0 including rate. Kill: score > 3.0

3. **L-BFGS Refinement** -- exp3_lbfgs_refinement.py
   - 10 Newton steps on GPU output, ~1 minute
   - Chains after exp1 or exp2

### CPU Lane
4. **Trick Stacking** -- exp4_cpu_trick_stack.py
   - CRF sweep (32-38) + TTO + multi-pass + deblock + quantization rounding
   - Score each independently AND stacked

### Infrastructure
5. **Auth Scorer Setup** -- exp5_auth_scorer_setup.py
   - Verify Lightning T4 scoring pipeline end-to-end
   - DALI vs PyAV parity check

## Decision Date
- 2026-04-17: commit to best GPU path for full 1200 frames

## Single Source of Truth
- `src/tac/research/competition_state.py` -- all configs and decisions
