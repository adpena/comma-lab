# Evaluator response surface across ALL dimensions → solve the lowest score mathematically

UTC 2026-06-09 · claude · answer to operator: "get segnet and posenet curves across all dimensions in
upstream evaluate.py bearing contest value across the contest video and compare all curves and solve
them mathematically." Verdict: **YES, possible and tractable** — it is the rigorous limit of the
evaluator-action waterfiller + meta-Lagrangian/Pareto solver the repo already scaffolds. Here is the
math, the curves, the solver, the hard parts, and the build order.

## The object we are solving
The evaluator E is FROZEN and DIFFERENTIABLE on the contest video 0.mkv. A witness produces frames
f = {f_t}_{t=0..1199}. The exact score:

    S(f, B) = 100·d_seg(f) + √(10·d_pose(f)) + 25·B/N,    N = 37_545_489

    d_seg(f)  = (1/600) Σ_pairs  (1/HW) Σ_p  1[ argmax_c SegNet(f_last)[c,p] ≠ argmax_c SegNet(src_last)[c,p] ]
    d_pose(f) = (1/600) Σ_pairs  (1/6) Σ_{k<6} ( PoseNet(f_pair)[k] − PoseNet(src_pair)[k] )²

We MINIMIZE S over f (and the encoding that sets B). RGB fidelity is irrelevant — only the argmax
field, the 6 pose dims, and the byte count carry contest value.

## "Curves across all dimensions" = the evaluator response surface (3 orders)

### Order 0 — the GLOBAL tolerance curve (DONE: evaluator_cell_tolerance.v1)
d_seg/d_pose vs global cheapening (downsample/blur/quantize). Already measured: the cell is LARGE
(survives ~1/64 spatial DOF + 3-4 bits + mild blur). This is the coarse envelope.

### Order 1 — the PER-PIXEL / PER-DIM sensitivity fields (the real "curves across all dimensions")
- **SegNet margin field** m_p = logit_src_class(p) − max_{c≠src_class} logit(p), per pixel, per
  last-frame. d_seg = fraction with m_p < 0. The margin is the 0th-order per-pixel curve (how far each
  pixel is from flipping class). Its histogram over the 600 frames = the SegNet "curve across all pixel
  dimensions in contest value": robust pixels (large +m_p) are FREE to cheapen; fragile pixels
  (|m_p|≈0, the boundaries) must be preserved. THIS IS THE DEFORESTATION MAP.
- **SegNet boundary Jacobian** J_seg(p) = ∂m_p/∂f = how a frame perturbation moves the margin (autograd
  through SegNet). Tells us the *direction* + magnitude of the cheapest safe perturbation per pixel.
- **PoseNet Jacobian** J_pose = ∂PoseNet[:6]/∂f, per pair (6×HW). d_pose's per-pixel sensitivity. Tells
  us which pixels/regions the 6 scored pose dims depend on (most of the frame is pose-null — free).
- **Per-class / per-pose-dim / per-region / per-frequency slices** are aggregations of these two fields.

### Order 2 — curvature (optional, for the trust region)
Hessian-vector products of m_p and pose give the local quadratic model → how far a perturbation stays
valid before the linearization breaks (the safe step size per pixel).

## "Solve them mathematically" = the score-domain Lagrangian / KKT / waterfilling
Introduce a fidelity variable φ_p ∈ [0,1] per pixel/region (1 = keep source-exact, 0 = drop to the
cheapest code) and a rate model B(φ) (bytes as a function of where we spend fidelity). Minimize:

    L(φ, λ) = 100·d_seg(φ) + √(10·d_pose(φ)) + 25·B(φ)/N

Stationarity (∂L/∂φ_p = 0) gives the WATERFILLING condition: at the optimum, the marginal score
reduction per byte is EQUALIZED across all pixels/regions/pose-dims:

    [ 100·∂d_seg/∂φ_p + (√10 / (2√d_pose))·∂d_pose/∂φ_p ]  /  (∂B/∂φ_p)  =  λ   ∀p

Interpretation (the lowest-score witness structure, in closed form given the fields):
- Spend fidelity (bytes) where the SegNet margin is FRAGILE (|m_p|≈0, boundaries) AND/OR the PoseNet
  Jacobian is LARGE.
- Drop fidelity to the cheapest code where the margin is ROBUST (large +m_p) AND pose-null.
- The √(10·d_pose) term is CONCAVE → its marginal grows as d_pose→0, so pose gets protected late but
  hard (matches the tolerance cliffs). The 100·d_seg term is LINEAR → seg is a steady per-pixel trade.
This is exactly the per-pixel argmax-margin waterfilling + pose-null projection the deforestation plan
named — now as the KKT solution of the exact score on the contest video.

## The HARD parts (and how the math handles them)
1. **d_seg argmax is non-smooth.** The margin m_p is the smooth surrogate; d_seg = Σ 1[m_p<0] is a step
   function. Relax with the margin for the gradient; the combinatorial "which pixels to let flip" is
   solved by ORDERING on m_p / byte-cost (let the most-robust, cheapest-to-keep pixels flip last). This
   is a fractional-knapsack / waterfilling, not a brute-force search.
2. **Null space (the whole point).** Many f → same (argmax, pose). We optimize over the EQUIVALENCE
   CLASS, not RGB. "Fidelity" = distance to the class boundary (the margin), NOT distance to source RGB.
   This is why the renderer's 21 dB RGB failed: it minimized the wrong distance.
3. **Rate model B(φ) is grammar-dependent.** For V3 direct grammar, B = entropy of the encoded
   skeleton (mask-argmax RLE + pose trajectory + sparse carrier). For a NeRV, B = decoder + latents.
   The waterfilling needs B(φ) per grammar — measured, not assumed (the rate attack lessons L20-L32).

## Connection to existing scaffolding (this is not greenfield)
- `tac.optimization.evaluator_action_waterfill` — the DISCRETE per-atom ΔS admission (the integer
  version of this KKT; admit atom σ iff ΔS<0). The continuous field version generalizes it.
- `tac.sensitivity_map` — per-axis sensitivity rows (the aggregation target for J_seg/J_pose).
- meta-Lagrangian / Pareto solver (CLAUDE.md non-negotiable) — the score-domain Lagrangian host.
- `tools/hi_nerv_renderer_sanity_ladder.py evaluator-cell-tolerance` — the Order-0 envelope (DONE).
- `_score_pair_with_margin` (just landed) — already computes the per-pixel margin field per pair.

## Build order (each a typed artifact feeding the solver; all on the contest video, in score units)
1. **`segnet_margin_field.v1`** — per-pixel m_p over all 600 last-frames + histogram + the robust/fragile
   partition (what fraction of pixels are free to cheapen). EXTENDS `_score_pair_with_margin` to all
   pairs; CHEAP (no autograd, just SegNet forwards). Directly = "the SegNet curve across all pixel
   dimensions in contest value." THE deforestation map.
2. **`posenet_pixel_jacobian.v1`** — ∂PoseNet[:6]/∂f per pair (autograd; heavier). The pose-null vs
   pose-critical pixel partition. = "the PoseNet curve across all pixel dimensions."
3. **`segnet_boundary_jacobian.v1`** — ∂m_p/∂f (autograd) — the safe-perturbation direction per pixel.
4. **`evaluator_waterfill_solution.v1`** — combine 1-3 + a measured rate model → the KKT fidelity
   allocation → the optimal evaluator-inverse witness skeleton + its predicted (d_seg, d_pose, B, S).
5. **Validate**: build the archive from the skeleton → full upstream evaluate.py → exact S. Reseed the
   solver with the residual (the meta-Lagrangian continual-learning loop).

## What this buys (the north star)
A mathematically-grounded, contest-video-specific map of EXACTLY where bytes must go to stay in the
evaluator cells, and where they can be stripped to zero — the lowest-score witness as the KKT solution
of the exact score, not a trial-and-error sweep. It subsumes V1/V2/V3: any witness (NeRV, SNeRV, direct
grammar) is scored by the same response surface; the waterfilling says which representation spends the
fewest bytes for the required margin + pose fidelity.

NEXT CONCRETE STEP: build `segnet_margin_field.v1` (cheap, all 600 frames) — it is the Order-1 SegNet
curve across all dimensions and the deforestation map, and it requires only SegNet forwards we already run.
