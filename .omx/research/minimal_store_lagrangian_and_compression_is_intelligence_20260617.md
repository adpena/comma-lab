# "Storing more than we need" — the math-optimal minimal store (Lagrangian) + compression-is-intelligence

**Operator 2026-06-17: "Yousfi storing more than we need" + "All must be math/algebra/geometry/calculus
optimal" + the 3Blue1Brown "Reinventing Entropy | Compression is Intelligence Part 1" video.** This note
pins the optimal framing so the witness/partition subagents + the next iteration consume it. All
`[contest-CPU advisory]`; the pointer (0.19110) is unmoved — this is means until an exact row crosses.

## The measurement that refutes the NAÏVE "more than we need" (resolution) and points to the real one
"Store at the scorer's decision resolution (stride-2 stem ⇒ d_seg decided ~192×256)" is the obvious
guess. **MEASURED FALSE** (gt_targets_n100, area-pool→majority→nearest-upsample, real GT argmax):

| store res | induced d_seg | seg term 100·d_seg | bytes/frame (zlib) | byte cut |
|---|---|---|---|---|
| 384×512 (full) | 0 | 0 | 988 | 1.0× |
| 192×256 (half) | **0.00554** | 0.554 | 386 | 2.6× |
| 96×128 (¼) | 0.01062 | 1.06 | 125 | 7.9× |

Half-res induces d_seg 0.0055 — **7× the sub-0.15 budget (~0.0008)**. Spatial downsampling is NOT free
(the "d_seg is HF-blind" intuition is FALSIFIED again, like the FP4 weight-domain bridge). The partition
carries real sub-half-res detail that d_seg counts. **So the lever is NOT resolution.**

## The real "more than we need" — three lossless/tolerance levers, unified by a Lagrangian
The minimal store is a constrained optimization, not a heuristic. Candidate corrections = the argmax
FLIPS (pixels where decoder-prediction ≠ GT). Each flip i has:
- **c_i** = its marginal byte cost under the contour/temporal/arithmetic coder (NOT per-pixel — the DoF is
  the boundary curve; flips are contour-clustered, so coding boundaries captures full-res d_seg at far
  fewer bytes than 988 B/frame; this is the *lossless* "more than we need" — pixels vs boundaries),
- **s_i** = its round-trip survival probability (the uint8↑874/↓384 step undoes ~53% of corrections),
- **b_i = 1/N** = its d_seg benefit — **UNIFORM** (every fixed surviving flip removes exactly one counted
  disagreement).

Minimize bytes `B(S) = Σ_{i∈S} c_i` s.t. expected d_seg `d0 − Σ_{i∈S} s_i/N ≤ budget`. Because the benefit
is uniform, the **closed-form optimum** (Lagrangian / fractional-knapsack) is: **include flips in
increasing order of `c_i / s_i` (byte-per-expected-fixed-flip) until d_seg hits the budget — and STOP.**
Three consequences = the three real "more than we need" savings:
1. **Residual, not full partition** — code flips vs the decoder/temporal/basin PRIOR, never the whole map.
2. **Tolerance** — fix only `(d0 − budget)·N / mean(s)` of the cheapest-survivable flips, not all of them.
3. **Survival-filter** — flips with low s_i are dominated (bad c_i/s_i) → never coded (they wouldn't take).
This is the optimal Yousfi-tolerance + survival + contour unification. The prototype probes did the wrong
corner (full partition / all flips / per-pixel / no survival sort) — hence 543 KB & 0.27–0.35 rate.

## Compression-is-intelligence (3B1B) → why Track-A and the witness are ONE objective
Shannon Noiseless Coding: optimal code length = `−log2 p(x)`; the best compressor IS the best predictor.
Applied here: the minimal store = the partition's residual against the **best predictor**, coded to its
**conditional entropy** `H(partition | predictor)`. The predictor stack = the decoder (latents→partition)
+ temporal (t−1→t) + spatial contour context. Therefore:
- **Improving the decoder (the corrected Track-A run) DIRECTLY shrinks the witness residual** — a better
  predictor lowers `H(partition | predictor)`, i.e. fewer/cheaper flips to code. Track-A (better predictor)
  and the witness (code the residual to conditional entropy) are the **same** objective, not rivals.
- The `c_i` coder MUST hit the conditional entropy: arithmetic/range coding against a context model
  (neighbor classes + temporal + margin), NOT LZMA-over-raw-labels (the prototype) and NOT per-pixel.

## Apply-on-return (subagents are mid-flight; SendMessage unavailable)
When the witness / partition top-AIML subagents return, verify they implement: (a) the `c_i/s_i`
Lagrangian selection to the d_seg budget (not a drop_frac sweep); (b) arithmetic coding to the conditional
entropy (not LZMA); (c) residual-vs-predictor (not full store); (d) NO spatial downsampling (measured
non-free). If a returned coder used a sweep or LZMA or full-store, re-dispatch the refinement.
Sister: [[small-basis-micro-macro-audit-sub015-path]] (the d_seg-sensitivity map = the same boundary band)
+ the corrected Track-A run #130 (improves the predictor → shrinks every residual).
