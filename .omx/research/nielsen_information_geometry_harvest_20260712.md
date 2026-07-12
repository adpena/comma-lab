# Frank Nielsen / Bregman-manifold information-geometry harvest — signal report (2026-07-12)

**Status:** ADVISORY RESEARCH / HARVEST. `score_claim=false`, `promotion_eligible=false`,
pointer UNMOVED. This is a MEANS report: it exists to make witness training faster/better,
not to move the exact score. Every claim below is labeled MEASURED (our own artifact),
DERIVED (math), or ASSESSED (honest judgement). Operator-directed harvest of Nielsen's
GitHub + papers connected to "cache/pretrain custom geometric structure up front to avoid
expensive per-step work."

## The problem this feeds
~95% of witness training wall-clock is backprop through the FROZEN SegNet (EfficientNet-B2)
every pair every step, to get ∂d_seg/∂θ_witness. Goal: replace/reduce that with precomputed
cached geometric structure.

---

## Sources harvested (with URLs)

- **pyBregMan** — Python library for Bregman / dually-flat manifolds. Repo
  `github.com/alexandersoen/pyBregMan`; PyPI `pyBregMan`; docs
  `franknielsen.github.io/pyBregMan`; paper **arXiv:2408.04175** (Soen, Nielsen 2024).
  Modules: `bregman.base` (`Point`, `LAMBDA_COORDS`, `DualCoords`), `bregman.manifold`,
  `bregman.application.distribution.exponential_family` (`GaussianManifold`, categorical/
  multinomial manifold), `bregman.barycenter` (`BregmanBarycenter`, `SkewBurbeaRaoBarycenter`),
  `bregman.dissimilarity` (Bregman/KL divergence, Chernoff), `Geodesic` classes,
  `bregman.visualizer`.
- **Nielsen, "Beyond scalar quasi-arithmetic means"** — **arXiv:2301.10980** (2023). The
  quasi-arithmetic center M_{∇F}(θ;w) = ∇F*(Σ_i w_i ∇F(θ_i)); gradient map of a Legendre-type
  potential is strictly comonotone with global inverse; dual geodesics = straight lines in the
  dual coordinate system; sided barycenters closed under the mixture operation.
- **Nielsen, "Quasi-arithmetic Centers … Jensen–Shannon ∇-Divergences"** — GSI 2023,
  Springer `link.springer.com/chapter/10.1007/978-3-031-38271-0_15`. Fenchel–Young canonical
  divergences; α-mixtures as density centroids; Mahalanobis = Bregman (quadratic F).
- Publications hub: `franknielsen.github.io/npublications.html`. Sister repos:
  jMEF (mixtures of exponential families / k-MLE), Bregman k-means/soft-clustering,
  Chernoff information & Fisher–Rao geodesics.

## Our own already-cached geometry (the redundancy baseline — DO NOT rebuild)
- `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` — GT argmax `lstars` + per-pixel margins,
  all 600 pairs, 2,551,382 separatrix pixels. **The categorical manifold's η-coordinates are
  already sitting on disk** (softmax probs / logits are one forward pass away).
- `.omx/research/evaluator_gradient_atlas_20260609.json` (#36) — JtJ spectrum + "SegNet margin
  VJP directions" (a cached linearization, n48 advisory, pose-weighted).
- `.omx/research/segnet_margin_field_20260609.json` (#141) — the margin field.
- **MEASURED**: margin field ≈ Fisher-metric surrogate, Pearson **0.978** (unified level-set
  memo). This is the one IG connection we already exploit and it pays.

---

## Deliverable 1 — "Pretrain/cache the Bregman geometry to make witness→GT motion closed-form"

**The honest decomposition holds exactly, and it is also exactly why the win is small.**

The d_seg gradient factors by the chain rule through two maps:

  ∂d_seg/∂θ  =  (∂pixels/∂θ)  ·  **[ J = ∂logits/∂pixels ]**  ·  (∂d_seg/∂logits)
                witness fwd        deep CNN backward (the 95%)     output-space geometry

- **(a) OUTPUT-space geometry (logits ↔ class-probs) is EXACTLY a Bregman / dually-flat
  categorical manifold.** F = log-partition (convex); ∇F = softmax : θ(logits)→η(probs);
  ∇F* = its inverse (centered log); Fisher = ∇²F = diag(p)−ppᵀ; canonical divergence = KL.
  So "where should the logits move to flip this pixel's argmax into the GT cell" HAS a
  closed-form answer: the Bregman/dual geodesic in the categorical manifold from the current
  logit to the nearest point of the GT argmax cell. **DERIVED, correct, and cheap.**
- **(b) The pixel→logit map is the deep EfficientNet-B2.** Its backward (VJP) is the entire
  expense. There is no closed form; the Jacobian J is (5·H·W)×(3·H·W) — only its ACTION (a
  VJP) is affordable, and computing that action IS the backward we are trying to avoid.
- **(a)+(b) together:** a closed-form logit-target (Bregman) still has to be *pulled back
  through J* to become a pixel gradient. Caching a closed-form target for the cheap factor
  (a) does not remove the expensive factor (b).

**ASSESSED verdict — the win is on the wrong side of the bottleneck.** The Bregman closed
form buys the ∂d_seg/∂logits factor, which is already ~O(pixels) cheap and which we already
approximate with the margin field (0.978 Fisher-faithful). It does **not** buy the CNN
backward. So "replace per-step backprop with closed-form Bregman geodesics" is, as literally
framed, **NOT** a training-cost win.

**Where a real cost win could hide (the frozen/stale-Jacobian idea, ASSESSED):** the only way
(b) gets cheaper is to CACHE the *action* of J and REUSE it across steps/pairs instead of
re-linearizing every step (Shamanskii / frozen-Jacobian quasi-Newton; the atlas #36 already
caches VJP directions at n48). The killer is staleness precisely where the signal lives: all
d_seg action is on the separatrix annulus (~4.7% area, #333), which is exactly where the CNN
is most nonlinear and where a cached linearization decays fastest as the argmax boundary
moves. A frozen-J reuse would be valid in the flat argmax interior (dark, zero-signal) and
invalid at the boundary (all the signal). This matches our MEASURED solve-don't-train wall
(see D-flag below). **Low-confidence; would need a measured "how many steps until the cached
VJP direction misaligns > X°" probe before any build.**

---

## Deliverable 2 — pyBregMan primitives: adoptable? MLX-portable?

The categorical-manifold primitives are ~10 lines each and MLX-trivial (we already have
softmax). **Porting the library is NOT worth it; adopting the formulas is.**

| primitive | pyBregMan surface | closed-form? | MLX port | our use |
|---|---|---|---|---|
| ∇F  θ→η (natural→expectation) | categorical manifold `.theta_to_eta` | yes (softmax) | already have | trivial |
| ∇F* η→θ (expectation→natural) | `.eta_to_theta` | yes (centered log) | 1 line | needed for centroids |
| Bregman / KL divergence | `bregman.dissimilarity` | yes | already have | margin already better (D3) |
| dual (η-flat) geodesic | `Geodesic` in `DualCoords` | yes (straight line in η) | 1 line | logit-target field (D1a) |
| entropic / Bregman centroid | `BregmanBarycenter` | yes (η-mean → ∇F*) | ~5 lines | **v8 per-class carrier (D3)** |
| Chernoff information | `bregman.dissimilarity` | 1-D convex opt | ~15 lines | MEASURED-CLOSED (D3) |

**Concrete adoptable formula (the one to actually port):** entropic centroid of a set of
per-pixel logit vectors {θ_i} with weights {w_i} = ∇F*( Σ_i w_i ∇F(θ_i) ) = centered-log of
the weighted mean of softmaxes. That is the closed-form "prototype logit" for a class region.

---

## Deliverable 3 — Bregman centroids for v8 per-class carriers + Chernoff for boundaries

**Centroids (ASSESSED, cheap, worth a $0 probe):** the v8 per-class carrier line (#380/#386,
"a class region's center is a Bregman centroid") is a genuine, correct fit. For each class
region, the closed-form entropic centroid of its pixels' softmaxes gives a prototype
logit/probability vector in ONE pass — a cheap seed/target for the per-class carrier codebook,
computed from `gt_n600.npz` with no training. **Win: modest (initialization/target of a
carrier, not the whole descent). Feasibility: high (data cached, formula 5 lines).** Proposal:
a $0 advisory probe that computes per-class entropic centroids on n600 and measures whether
seeding the v8 carrier from them lowers epochs-to-target vs random/mean init. Label advisory;
do NOT claim score.

**Chernoff for class-boundary placement (MEASURED — ALREADY CLOSED, do NOT re-open):**
`.omx/research/chernoff_vs_margin_probe_20260706.md` (n600, authority-faithful, argmax
reproduced 597/600, boundary |Δ|=2.1e-6). Result: Spearman(margin, Chernoff)=0.82;
disagreement pixels are 8.3× ENRICHED on triple-junctions (only 1.6% of the separatrix) and
2.7× DEPLETED on class-1 LANE pixels — but our d_seg residual lives on lane/along-tangent
pixels, not junctions. Flip-survival AUC: margin 0.777 vs Chernoff **0.730** (Chernoff WORSE
by 0.047). **Verdict already recorded: degenerate-equivalent-or-worse; raw margin (= the
Fisher surrogate) is simpler AND empirically better for our residual.** Multi-class Chernoff
is closed for our current residual. Reactivation only if a distinct junction-pixel residual
component ever appears.

---

## Deliverable 4 — Natural-gradient / dual-coordinate optimization (conditioning)

**ASSESSED — mostly redundant with Muon; the tractable version we already run.** Working in
dual (η) coordinates gives natural-gradient conditioning *for the categorical output layer's
own parameters*. But our optimized parameters are the WITNESS (the pixel generator), not the
categorical params. The Fisher we would need to precondition is the categorical Fisher
**pulled back through J** — i.e. Jᵀ (diag(p)−ppᵀ) J — which again requires the intractable CNN
Jacobian (same wall as D1b). The affordable approximations are exactly what we already have:
(i) **Muon** orthogonalized/whitened updates (MEASURED −32% d_seg vs AdamW; the conditioning
win is banked); (ii) the atlas JtJ spectrum (#36) as a coarse pose-side preconditioner.
**No fewer-epochs win beyond Muon is available without the cached full Jacobian.** Connects to
the Muon/conditioning line but does not extend it.

---

## Deliverable 5 — Ranked findings (win × feasibility × fit; honest on redundancy)

1. **KEEP margin = Fisher surrogate (MEASURED, banked).** The one IG connection that pays
   (Pearson 0.978). Nielsen's framework CONFIRMS why (categorical Fisher = ∇²F, its curvature
   is the flip-resistance), but adds no new actuator here. Redundant-but-validated.

2. **Bregman entropic-centroid seed for v8 per-class carriers — BUILD a $0 advisory probe.**
   Closed-form, data already cached (`gt_n600.npz`), ~5 lines MLX, direct fit to #380/#386.
   Modest win (carrier init/target, not the backprop). Only genuinely new, cheap, non-redundant
   proposal. OSS: `bregman.barycenter.BregmanBarycenter`, formula ∇F*(Σ w_i ∇F(θ_i)).

3. **Frozen/stale-Jacobian reuse to amortize the CNN backward — PROBE-BEFORE-BUILD, low
   confidence.** The only idea that targets the real 95% bottleneck. Blocked by
   separatrix-locality staleness (linearization worst exactly where signal lives). Gate:
   measure VJP-direction misalignment vs step count on the boundary annulus BEFORE any build.
   Partially pre-built in atlas #36 (VJP directions cached).

4. **Closed-form Bregman logit-geodesic replacing backprop — HONEST NEGATIVE (as framed).**
   Correct math for the cheap output-space factor; does not bypass the expensive pixel→logit
   CNN backward. The literal operator framing ("closed-form instead of per-step backprop")
   does not yield a training-cost win. Records the honest boundary.

5. **Multi-class Chernoff for boundary placement — CLOSED (MEASURED 2026-07-06).** Worse than
   margin for our lane residual. Do not re-open absent a junction-pixel residual component.

6. **Natural gradient in dual coords — REDUNDANT with Muon.** Tractable form = Muon (banked);
   true-Fisher form needs the intractable pulled-back Jacobian.

7. **Solve-don't-train via Bregman reframing — WALL ALREADY MEASURED.** The categorical
   Bregman geodesic reframes the *target* of the #341/#342 solve, but the transfer wall is
   unchanged: `compress_time_seed_and_solve_dseg_verdict_20260617.md` (latent solve +0.75%
   no-op, 3.7× slower) and `basin_finisher_head_solve_probe_measured_20260707.md` (head solve
   subset-overfits: −3.4% in-subset, +5.2% held-out). Proxy/logit-space optimum ≠ realized
   argmax through R. Bregman does not cross that wall.

## Bottom line
The output geometry of our scorer IS the Bregman categorical manifold Nielsen's tools
implement — but that geometry sits on the CHEAP side of our bottleneck. The single
non-redundant, cheap, correctly-fitting harvest is the **Bregman entropic centroid as a
closed-form seed/target for the v8 per-class carriers** (rank 2, $0 probe). Everything that
would touch the expensive CNN backward (D1b, frozen-J, natural gradient) is either an honest
negative, redundant with Muon, or gated behind an unmeasured staleness assumption. Two adjacent
leads (Chernoff, solve-don't-train) are already MEASURED-and-closed in our own artifacts —
recorded here so they are not re-researched.
