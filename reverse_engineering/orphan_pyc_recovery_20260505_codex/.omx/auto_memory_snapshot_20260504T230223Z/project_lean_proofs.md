---
name: Lean 4 Formal Proofs for Writeup
description: Formally verify hyperparameter derivations in Lean 4 — unprecedented for a competition writeup
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Lean 4 Proofs to Write (for best-writeup prize)

1. **segnet_weight_optimal**: w_s*(p,T) = 20·sqrt(p/0.1)/T² minimizes
   score at the current operating point under first-order Taylor expansion.

2. **boundary_ceiling**: For boundary fraction β, the effective amplification
   bw/(β·bw + (1-β)) has limit 1/β as bw→∞. At β=0.05, ceiling is 20x.

3. **per_channel_dominates**: Per-channel quantization error variance ≤
   per-tensor for all weight tensors. Strict inequality when channels have
   heterogeneous dynamic ranges.

4. **ws_T2_invariant**: The compound variable w_s·T² determines the effective
   SegNet gradient magnitude independent of T. Changing T without adjusting
   w_s breaks the gradient balance.

5. **crossover_theorem**: SegNet training effort dominates PoseNet when
   η_s/η_p ≥ sqrt(10)/(200·sqrt(p)). At p=0.002, threshold is 0.17.

## Implementation
- Install: `elan install lean4`
- Write proofs in `proofs/` directory
- Export to HTML for embedding in writeup
- Reference from docs/writeup_draft.md

## Why This Matters
- No other competition submission has formally verified hyperparameter choices
- Demonstrates the derivations are provably correct, not empirical guesses
- The PoseNet regression (250x over-weighting) would have been caught by
  type-checking the w_s·T² invariant
