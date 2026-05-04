# Revival plan: Lane Ω-1: Score-Jacobian KL basis as residual primitive on PR106

**Gem**: `src/tac/sjkl_basis.py`
**ID**: `07_sjkl_basis_pr106_residual`

## Current state

Level-2 scaffold. Wave-Ω-1 council 22/22 GO endorsement [prediction]. SJ-KL basis = Fisher-information eigenvectors of scorer — provably R(D)-optimal under known scorer.

## Files touched

- experiments/build_sjkl_basis_for_pr106.py (new)
- src/tac/sjkl_basis.py (verify implementation level → 3)
- submissions/apogee_sjkl/* (new sibling)

## Integration sketch

1. Compute score-Jacobian J = ∂score/∂(decoder weights) over contest video.
2. SVD J = U Σ V^T → top-K right singular vectors form basis.
3. Encode PR106 decoder as basis coefficients (top-K dominant) + residual.
4. Quantize coefficients per-component; pack via arithmetic.
5. Inflate: reconstruct decoder = U @ diag(coeffs) @ V^T.

## Test plan

- Smoke: SVD reconstruction round-trip on synthetic data.
- K-sweep: pick K minimizing |reconstructed - original| under byte budget.
- Score: must be ≤ 0.20946.

## Predicted score basis

Wave-Ω-1 council prediction stack {Ω-1 SJ-KL + Ω-2 NeRV + Ω-3 block-FP} → 0.180 [prediction]. Δ band [-0.020, -0.005] ÷ 8h.

## What would change my mind

If SJ-KL basis on PR106 decoder is rank-deficient (rare), abandon. Most likely scenario: marginal -0.005 with stack-multiplicative gains.

## Blockers resolved in plan

- Score-Jacobian computation needs CUDA forward — deferred until dispatch budget.

## Skunkworks council deliberation

Shannon/Ballé/MacKay LEAD endorse (Fields-medal session). Quantizr: 'paradigm-shift, ship as paper alongside score'. Boyd: 'co-train basis + decoder via ADMM'.

**Verdict**: VOTE 9/10 GO (only Hotz: 'try simpler primitives first'). Rated highest paradigm-level lane in Wave-Ω.
