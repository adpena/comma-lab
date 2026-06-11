# Full-stack carrier V2 — the baggage-free, scorer-and-information-optimal design (2026-06-11)

**Operator directive (2026-06-11):** "start moving retraining and new full-stack carrier and substrate
design based on all of our new knowledge — what would a full-stack optimized around the contest scorer and
information look like now, not stuck in local minima or inheriting baggage." This memo is the clean-slate
synthesis of everything MEASURED this session into one coherent full-stack design + the retraining plan.

**Authority:** design; every economic fact is [MEASURED] (this session's exact-`modules.py` artifacts) or
[DERIVED]. Frontier UNMOVED 0.19110, `N=37,545,489`. S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/N. The
oracle is upstream/evaluate.py + modules.py + 0.mkv; pointer moves only on Linux-x86_64 + CUDA-T4.

## 0. THE BAGGAGE WE SHED (explicit — the local minima we refuse to inherit)

1. **Weights == bytes** (conv-HNeRV): FALSE. Capacity and coded-size are separable. → put capacity in
   cheap-to-code latents (Cool-Chic 17× [MEASURED]), not stored weights.
2. **Pixel-perfect reconstruction** (CE/MSE): WASTEFUL. The scorer only needs the correct *side* of each
   `modules.py` decision boundary. → margin-polytope hinge (free −29% vs CE at equal bytes [MEASURED]).
3. **Uniform capacity**: WASTEFUL. ~80% of appearance is in the scorer's invisible subspace [DERIVED]. →
   scorer-shaped allocation by the joint P18/P19 marginal-value field.
4. **Store the scored quantity** (partition/pose from pixels): the partition costs 3–5× the neural
   regenerator [MEASURED, B-WITNESS/Yousfi]; pose can't be reconstructed from a coarse render [MEASURED,
   pose-fold]. → AMORTIZED appearance-regeneration for seg + STORED pose scalars + FiLM.
5. **Lossless coding**: WRONG GAME. It's inverse steganalysis — code to the detector's *tolerance*
   [Yousfi]. → drop everything below the d_seg flip-threshold + outside the pose tube + in the invisible
   subspace.
6. **One generic objective**: → the objective IS the exact scorer geometry (hinge on `modules.py`
   boundaries + per-dim-Mahalanobis pose), not a reconstruction proxy.

## 1. WHAT WE KNOW (the measured design constraints)

| Fact | Value | Source |
|---|---|---|
| Contest scores only 3 terms | d_seg (argmax-flip frame1) + d_pose (6-dim, both frames) + bytes | modules.py/evaluate.py |
| Rate is cheaply solvable | Cool-Chic latent+ARM = 17× under conv-HNeRV; flip-delta 1.28 B/flip; firmware pack | [MEASURED] B1-CLOSE/#98 |
| d_seg basin is CAPACITY-bound | compact synth plateaus 0.018–0.028; hinge −29% but doesn't break; −10%/2.7×-bytes | [MEASURED] hinge A/B/C |
| Corrected bar (sub-frontier) | d_seg ~0.0011–0.0017 (2–3× basin), pose collapsed, rate 13–90KB | [DERIVED] §5b prior memo |
| Pose needs dense texture | stored-6+FiLM holds tube IF render has texture (coarse synth can't) | [MEASURED] pose-fold |
| Flips are diffuse argmax-instability | post-hoc repair a wash; the wall is render argmax-stability | [MEASURED] deep seg-profile |
| Margin-polytope lowers capacity need | correct-side-of-boundary ≪ pixel-perfect | [MEASURED] hinge |

## 2. THE ARCHITECTURE (the full stack, scorer-shaped)

**A hierarchical, scorer-shaped, latent-capacity-heavy neural renderer** that produces the frame pair
(frame1 for SegNet, both for PoseNet) ARGMAX-CORRECT + pose-faithful, not pixel-perfect:

- **Shared synthesis engine (modest, amortized over 600 frames → cheap):** a small backbone that turns
  per-frame latents → RGB pair. Sized to *argmax-stability + pose-subspace texture*, NOT fidelity — the
  hinge minimizes how big this must be. Stored once (block-FP/sub-byte packed).
- **Per-frame latent fields (RICH, where the basin capacity lives — coded cheaply by ARM):** multiresolution
  grids, resolution/channels ALLOCATED by the P18/P19 marginal-value field — fine at SegNet decision
  boundaries + the pose-sensitive (low-freq-horizontal) subspace, coarse/absent in the invisible interiors.
  This is the capacity-in-cheap-to-code-latents bet (the latent-heavy test decides if it reaches the bar).
- **ARM entropy model** (autoregressive, context-conditioned) coding the latents — the rate-efficient core,
  with the validated QA-entropy so coded bits == real archive bytes.
- **Stored pose: 6 scalars/pair (range-coded, ~1KB) + FiLM-conditioning** of the synthesis → pose tube.
- **Final-mile d_seg: the 1.28 B/flip margin-normal flip-delta** (THE-LAW-screened) cleans up the last
  residual flips once the base is near the bar (only viable at ~few flips/frame).

## 3. THE OBJECTIVE (the loss IS the scorer geometry)

`L = 100·hinge_polytope(SegNet(render_f1), L*) + Σ_k mahalanobis_k(PoseNet(render_pair), pose*) + λ·rate(ARM)`
- **hinge_polytope** = `relu(margin_target − margin)` on `modules.py` logits — correct-side-of-boundary,
  hard-pixel curriculum below the 5e-3 surrogate floor (the now-default seg loss, banked).
- **pose** = per-dim Σ⁻¹-weighted MSE over the 6 dims, both frames (the differentiable-YUV6 path).
- **rate** = the ARM −log2 p (QA-entropy), λ from the score-domain Lagrangian (THE LAW).
- Eval-roundtrip (uint8/resize) in the inner loop; EMA warmup. All measured against exact `modules.py`.

## 4. CAPACITY ALLOCATION (the optimal-capacity solve — the original core)

The RD optimum: minimize bytes s.t. d_seg ≤ bar(0.0011) ∧ d_pose ≤ tube. The decision variables are the
THREE capacity pools (synth size, latent resolution/channels per scorer-region, ARM context order). The
P18/P19 field is the allocator: spend on seg-flip-binding boundaries + pose-sensitive subspace, shed the
invisible. **The empirical unknown the latent-heavy test resolves:** does the basin-reaching capacity fit
in the cheap-to-code latents (→ sub-frontier on a modest synth) or does it need a frontier-class synth (→
the synth bytes ≈ frontier → paid PR95-scale compute, capstone #90).

## 5. BYTE-ENCODING (firmware-grade) + INFLATE

Monolithic 0.bin: ARM-coded latents + sub-byte/block-FP synth weights + PR95 L20–L32 per-section
multi-representation + flip-delta (colex-rank+sign+quantum) + pose scalars. Native bit-exact lowering
(runtime-rs/tac-packet-compiler + golden vectors). Inflate: scorer-free, numpy-portable, ≤100 LOC, ≤2 deps,
CPU+CUDA — latent upsample → modest synth → pair → RGB; SegNet/PoseNet re-derive seg+pose at eval.

## 6. TRAINING (compute-aware)

PR95-scale curriculum with the hinge as the seg loss + the pose tube + the rate Lagrangian. Compute scales
with the capacity the bar requires: if latent-heavy suffices → feasible locally/cheap-paid; if frontier-class
synth → PR95-scale (~10³–10⁴ epochs) paid GPU (the capstone #90 spend, an operator budget decision). MLX-first
with the FP32-exact arch-override; torch-CPU authority for exact d_seg; dual CPU+CUDA contest eval to promote.

## 7. INNOVATION ACCOUNTING (NO-FAKE)

- **Ours-original:** the scorer-shaped capacity-allocation (capacity in P18/P19-allocated cheap-to-code
  latents + margin-polytope-minimized synth), the code-to-tolerance discipline (drop sub-flip-threshold +
  pose-tube + invisible), the margin-polytope hinge as a trainable seg loss, the flip-delta final-mile. The
  decoupling of capacity from bytes shaped by the exact scorer is not an off-the-shelf codec.
- **Borrowed (substrate, reused per no-duplicative-code):** Cool-Chic latent-grid + ARM, the PR95 L20–L32
  entropy stack, the curriculum, the seg-core/P18-P19/null-space infra, the byte-close pipeline — all
  in-repo, all CONSUMED.

## 8. THE MOVE — retraining/build plan (what "start moving" means)

1. **Resolve the capacity fork** (latent-heavy test, running) → modest-synth-sub-frontier OR frontier-class.
2. **Land the V2 trainer**: the scorer-shaped renderer (modest synth + P18/P19-allocated multires latents)
   + the hinge+pose+rate objective + the byte-close pipeline — top-AIML, export+numpy-parity from byte zero.
3. **n48 advisory S** against the corrected bar → if sub-0.189 advisory, **scale to n600 + paired
   Linux-x86_64 + CUDA-T4 exact eval** (the pointer-mover). Compute scale per the fork (cheap or paid PR95).
4. **Compose the banked levers**: hinge (default) + flip-delta (1.28 B/flip) + firmware pack + stored-pose.
5. **The pointer moves only** when a byte-closed V2 archive's exact `evaluate.py` S beats 0.19110.

## 9. HONEST BOTTOM LINE (non-sycophantic)

This V2 is the coherent, baggage-free design everything we measured points to — and it is genuinely original
in its organizing principle (scorer-shaped capacity≠bytes + code-to-tolerance + margin-polytope objective).
But it does NOT escape the one measured hard constraint: the d_seg basin needs *enough render capacity to
stabilize the argmax*, and whether that capacity fits in cheap latents (sub-frontier, original summit) or
demands frontier-class compute (paid) is the empirical fork the latent-heavy test decides. The V2 is the
vehicle either way; the fork sets its compute price. We are not stuck in a local minimum — we have a clean
design and the banked ingredients (rate, codec, loss, pose) to build it; the open question is the capacity
price of the argmax-stable render, measured against the exact oracle.
