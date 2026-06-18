# Campaign math review — the calculus/algebra/geometry of what happened + how to optimize (2026-06-18)

**Operator: "review the math/algebra/calculus/geometry of what happened over time and why and how to
optimize."** A first-principles synthesis of the score dynamics, grounded in the measured rows (G3 exact,
the 50k run, the probes). All `[advisory]`; pointer UNMOVED 0.19110. This memo is the design basis for the
approved pivot (higher-capacity decoder + score-aware FP-shrink QAT).

## 1. The master gradient (calculus) — one scalar governs everything
S(θ) = 100·d_seg + √(10·d_pose) + 25·B/B₀,  B₀ = 37,545,489.
- **∂S/∂d_seg = 100** — constant, linear. d_seg is the *binding* term; every unit is worth 100.
- **∂S/∂d_pose = 5/√(10·d_pose)** — NONLINEAR, convex-decreasing. At d_pose=0.00034 → **85.8**; at 0.0001 → 158;
  → ∞ as d_pose→0. Pose is *fragile*: near the operating point a small d_pose wobble costs ~86× its size.
- **∂S/∂B = 25/B₀ = 6.66e-7 /byte** — tiny per byte, but B is 89K–177K, so the term is 0.06–0.12.
This gradient EXPLAINS the campaign: (a) d_seg dominates → all the d_seg work; (b) pose's 85.8 derivative is
why the FiLM-carrier *variance* (not its mean) blew up the 50k run (spikes to S=0.64); (c) rate is a bulk
lever (many bytes × tiny gradient), best attacked by cutting B structurally (QAT), not per-byte recode.

## 2. What happened over time, and why (the dynamics)
### 2a. d_seg descent = stretched-exponential, capacity+label-noise FLOORED (not power law)
d_seg(ep) ≈ a·exp(−(ep/τ)^β), a=0.00566, τ=4263, β=0.860 (16× better fit than a power law; glassy/annealing
dynamics). The 50k run confirmed the *shape*: 0.0079(ep50)→**0.00227(ep4600)** monotone — the margin-hinge
lever WORKS (descended below the basin 0.0026). **Geometry of the floor:** d_seg = the argmax-flip RATE =
area of the symmetric difference between argmax(SegNet(recon)) and argmax(SegNet(GT)) regions ≈ a *perimeter
integral* over the SegNet decision boundary. It is 882× concentrated at margin<0.5 (1.3% of pixels) and ~half
the residual flips sit at GT-margin<0.137 — pixels where the detector is unsure of the GT *itself*. So d_seg
floors at a (capacity ∧ label-noise) limit: **bc20 ~0.0022, frontier ~0.0003** (the existence proof that the
label-noise proxy is soft, not a hard wall — capacity buys you below it).
### 2b. The 50k run's FAILURE = the √-pose-fragility + a curriculum discontinuity
- **Pose:** the FiLM-v2 carrier diverted pose to a noisy rgb_0 head; its variance, amplified by the 85.8
  derivative, produced S spikes (0.18, 0.64). The basin's *trunk*-pose was stable (0.00034). best-on-S
  protected the artifact but no clean low-S checkpoint emerged → best (0.401) WORSE than the basin (0.378).
- **d_seg curriculum discontinuity:** at the stage-0→1 boundary (ep5059, softplus/LR change) d_seg jumped
  0.00227→0.0035 — a curriculum artifact (a discontinuity in the training operator), not the lever failing.
### 2c. rate = at the entropy floor; FP-shrink real but PTQ-collapses
Lossless recode recovers 0 (every section ≈8.0 bits/byte). Post-hoc bit-shrink: int4 cuts 47.7% bytes
(Δrate −0.0283) but S RISES monotone (int4 d_pose ×322) — because ∂S/∂d_seg=100 and the √-pose make the
distortion spill outweigh the byte save. **PTQ collapses; the rate win needs QAT.**

## 3. The capacity↔rate Pareto geometry (the crux, and why the pivot is correct)
Minimize over decoder capacity p:  S(p) = 100·d_seg(p) + √(10·d_pose) + 25·B(p)/B₀,
with d_seg(p) decreasing (more params fix more boundary flips) and B(p) increasing.
**Stationarity (the optimum):** d/dp[100·d_seg(p)] = −d/dp[25·B(p)/B₀], i.e.
**100·d_seg′(p) = −(25/B₀)·B′(p)** — the marginal d_seg-gain-per-param equals the marginal byte-cost-per-param.
- bc20 (p=83K): d_seg 0.0022 @ B 89K (rate 0.059). frontier: d_seg ~0.0003 @ B 177K (rate 0.118). Two points
  on the d_seg(p)↔B(p) trade. Neither is sub-0.15: bc20's d_seg too high (0.0022·100=0.22), frontier's B too
  high (rate 0.118 + d_seg 0.03 + pose 0.058 ≈ 0.21... it's at 0.191).
- **The sub-0.15 contour** {100·d_seg + √(10·d_pose) + 25·B/B₀ = 0.15} requires a point with BOTH low d_seg
  (needs capacity) AND low B (needs few bytes) — OFF the native HNeRV (d_seg,B) Pareto curve.
- **FP-shrink QAT shifts B(p) DOWN** (fewer bytes per param at fixed d_seg) → it translates the entire B(p)
  curve toward the origin → the stationary optimum moves to **HIGHER capacity (lower d_seg) at the same byte
  budget.** THIS is why "higher capacity + QAT-shrink" is the math-optimal pivot: QAT decouples capacity from
  rate, letting us buy frontier-grade d_seg without paying frontier-grade bytes.

## 4. How to optimize (the math-optimal pivot design)
1. **Capacity-RD sweep (the deferred G1 #124):** measure d_seg(p) at p ∈ {bc20=83K, ~2×, ~4×, frontier-class}
   to locate the stationary optimum 100·d_seg′(p) = −(25/B₀)·B′(p). Pick the capacity whose *QAT-shrunk* byte
   budget + d_seg floor minimizes S. (bc20's d_seg floor 0.0022 is too high; go up until d_seg(p)→~0.0004.)
2. **Score-aware FP-shrink QAT (water-filling on the master gradient, the #141 map):** allocate weight-precision
   bits by ∂S/∂(precision of weight w). High-margin-saliency weights (d_seg-critical, the |∂margin/∂input| map)
   stay high-precision; d_seg-blind weights (the stem-Nyquist band, where the SegNet can't see) coarsen to int4.
   QAT trains the decoder to be robust to that grid (PTQ collapses because the net never learned to tolerate it).
   Break-even: hold d_seg within +0.0003 of the high-capacity float floor while cutting B ~40–47%.
3. **Pose: stabilize, don't chase zero.** Use the stable *trunk*-pose (basin held 0.00034), NOT the noisy FiLM
   carrier (the √-derivative punishes variance). The 1-DOF radial-zoom codec (#140-adjacent) stores it cheaply.
   d_pose ≈ 0.0003 is near the sweet spot (driving it lower raises ∂S/∂d_pose, diminishing returns + fragility).
4. **Curriculum: warm LR across stage boundaries** to kill the stage-transition d_seg discontinuity (the
   ep5059 regression was a discontinuous training operator, fixable with a warm/anneal across the boundary).

## 5. The one-line optimum
**Sub-0.15 = argmin_p [100·d_seg(p) + √(10·d_pose) + 25·B_QAT(p)/B₀], where B_QAT shifts B(p) down via
score-aware (margin-saliency-water-filled) QAT, capacity p is raised until d_seg(p)→~0.0004, and pose is held
stable on the trunk at ~0.0003.** The pivot build = the capacity-RD sweep (step 1) → the score-aware QAT
(step 2). Cross-refs: G3 exact row; `label_noise_floor_RESOLUTION_*`; `fp_shrink_ptq_smoke_*`;
`yousfi_council_checkin_unified_margin_saliency_*`; `tac.margin_saliency_map`; `tac.post_hoc_weight_shrink`.
