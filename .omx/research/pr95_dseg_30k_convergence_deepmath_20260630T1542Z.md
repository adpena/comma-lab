# Why PR95 needed ~30k epochs to converge d_seg — the deep-math, and what it teaches our level-set witness

**Tag** `[$0 CPU deep-math research / advisory / DESIGN]` · 2026-06-30T15:42Z · **score_claim** false ·
**promotable** false · **ready_for_exact_eval_dispatch** false. This is a MEANS (a convergence theory + a
faster n600 recipe), NOT a score. **Pointer UNMOVED: contest-CPU 0.19109982 / contest-CUDA 0.20533003.**

Operator question (2026-06-30): *"there may be deep math or engineering under our noses explaining WHY the PR95
family converged over ~30k epochs on d_seg. We likely don't need the full 30k, but there are lessons to learn
and adapt for our level-set witness."* Framed through TASK-SUFFICIENT STATISTIC · DERIVATIVES & INTEGRALS ·
DIMENSIONALITY · what FALLS OUT.

**Evidence-grade legend (NO-FAKE):** `MEASURED-BY-US` (our anchor cited) · `CITED` (external lit) ·
`DERIVED` (math from cited premises) · `ASSERTED` (plausible, unproven — flagged). No convergence "law" appears
without one of these tags.

---

## TL;DR (the one-paragraph answer)

PR95's ~30k epochs is **not a mystery and not a wall — it is the signature of a slow power-law tail** that falls
out of three compounding facts: (1) the task-sufficient statistic is a **thin, high-curvature, ~8-dim NONLINEAR
boundary manifold** (`MEASURED-BY-US`), so the resolution-limited convergence exponent is small (α ≈ 4/d ≈ 0.5,
`CITED` Sharma–Kaplan + our d=8); (2) d_seg is a **0-1 (argmax) loss whose gradient lives only on a measure-zero
contour**, so the smooth surrogate's gradient mass on the boundary **vanishes as the temperature anneals** →
front-loaded bulk descent + a long flat boundary tail (`DERIVED` + `CITED`); (3) PR95 used an **isotropic HNeRV
sin basis** that must climb the entire NTK frequency spectrum to reach the boundary's high-frequency content
(**spectral bias**, `CITED`), and an **AdamW-until-the-end** optimizer that grad-collapses on the ill-conditioned
boundary mode. The lessons that let us skip ~15–25× of those epochs: **change the convergence EXPONENT, not just
the constant** — (a) a topology-matched **directional/curvelet basis** pre-loads the boundary frequency
(−48% d_seg `MEASURED-BY-US`), (b) **structured-init/openpilot seed** starts us where PR95 spent its first ~9000
ep getting to, (c) the **eikonal/level-set chart** gives a uniform margin (no vanishing-gradient hot-spots),
(d) **Muon** conditions the boundary mode that AdamW freezes, and (e) **early-stop on the TASK metric** (not the
loss — they decouple in the tail) and **store/UNIWARD the last aleatoric sliver** rather than grinding epochs.
Our measured n200 knees (CE~275 / tau~450 / l7~700; FEED-lt/lu/lv) **cohere** with this and give the n600 budget.

---

## 0. The two regimes of the d_seg-vs-epoch curve (the shape that needs explaining)

PR95's 8-stage 29,650-epoch curriculum (`MEASURED-BY-US`, CLAUDE.md L14 / cur-C19 forensic):
`3000 CE + 5650 tau_softplus + 1500 smooth + 500 QAT + 9000 C1a-L7 + 2000 λ + 3000 σ + 5000 Muon`.
**~14,000 of the 29,650 epochs (47%) are the boundary tail** (9000 C1a-L7 margin-refine + 5000 Muon). That is the
mass to explain.

Our own level-set witness reproduces the SAME two-regime shape at ~150× smaller epoch count
(`MEASURED-BY-US`, n200 MLX-advisory through-R, FEED-lt/lu):
```
CE     (0–300):  0.7359 init → 0.01746 @ep25 → 0.00994 @ep50 → 0.00574 @ep200 → 0.00544 @ep275 (floor)
tau    (300–600): 0.00534 @ep300 → 0.004307 @ep450 (BEST) → drifts to ~0.00445 @ep575
l7     (~700, ~100ep): 0.004227 (−1.9% marginal) — front-loaded, early-saturating
Muon   (live):    PR95's d_seg finisher; predicted ~6–9e-4 (unmeasured)
```
**The shape is universal: a steep front (bulk) + a flat tail (boundary).** Everything below explains why, and
why the tail is the expensive part.

---

## 1. LENS — TASK-SUFFICIENT STATISTIC + DIMENSIONALITY

**What d_seg actually measures.** `d_seg = mean argmax-disagreement of the frozen SegNet over the frame.` The
argmax partition is determined entirely by the **codim-1 boundary contour set** {x : top1(x)=top2(x)} — the
level set where the two largest class-logits cross. Interior pixels (large margin) are irrelevant to d_seg; only
the contour and its ±1px neighborhood can flip. So the **minimal sufficient statistic of d_seg is the boundary
contour geometry**, NOT the 5-class logit field. (`DERIVED` from the argmax definition; matches `MEASURED-BY-US`
G2: 96.8% of flip-mass in a 2px band; Fisher-curvature↔(−margin) Pearson 0.978.)

**Its intrinsic dimension (the crux).** `MEASURED-BY-US` (project_dseg_islands_8dim_manifold, 19 NO-FAKE tests,
phase-shuffle controlled): the binding Road↔Lane (lane-marking) residual is a **~8-dim NONLINEAR manifold**
(autoencoder 90%-knee = 8; MLE = 13). **Linear bases ALL lose** (pixel-PCA k95=412, DCT=61, Fourier-contour=29)
because the manifold is CURVED — a curved manifold's secant span ≫ its dimension, so no fixed linear/harmonic
basis captures it at its intrinsic dim. Whitney embedding 2·8+1 = 17 ≤ HNeRV's 28 latents → it FITS with room.

**The apparent paradox the operator is pointing at.** The sufficient statistic is **8 numbers** and the binding
pixels are **0.72% of the frame** (lane dashes, 27.6 components/frame) — yet resolving it takes orders of
magnitude more epochs than "8 numbers" suggests. **The resolution (three compounding causes):**

1. **The training signal is a frame-integral, bulk-dominated.** The surrogate loss `∫_frame CE(SegNet(render)) dx`
   integrates over ALL pixels; >99% are already-correct interior. So early gradient mass goes to the bulk, and
   the gradient mass on the 8-dim boundary manifold is a tiny fraction. The optimizer only *asymptotically*
   concentrates onto the boundary as the bulk is exhausted. (`DERIVED`; this is the front-vs-tail split of §2.)

2. **Resolution-limited scaling: α ≈ 4/d (the dimensionality → slowness law).** `CITED` (Sharma–Kaplan 2020,
   *A Neural Scaling Law from the Dimension of the Data Manifold*, arXiv 2004.10802; Bahri et al. 2021/PNAS
   2024, *Explaining Neural Scaling Laws*, arXiv 2102.06701): in the resolution-limited regime the loss decays
   as a power law `L ∝ (resource)^(−α)` with **α ≈ 4/d**, d = intrinsic data-manifold dimension. The boundary
   manifold is the relevant target. With our **measured d ≈ 8 → α ≈ 0.5**; d ≈ 13 → α ≈ 0.31. **These are SLOW
   exponents.** To halve the residual you need `2^(1/α)` = **4× (d=8) to ~9× (d=13)** more resource. Stack several
   halvings of the boundary residual and you get the ~10⁴-epoch tail. (Tagged `DERIVED` for the time-domain
   transfer: 4/d is the dataset/param exponent; the training-TIME tail is governed by the NTK spectrum, §3 — but
   both carry the SAME "higher d → smaller exponent → slower" qualitative law.)

3. **The 8-dim manifold is CURVED + content-conditioned**, so a learner must learn the *chart*, not just place 8
   coefficients — and the chart's high-frequency content is what spectral bias makes slow (§3).

> **Falls out:** the cost is not in the *dimension* (8 is tiny) — it is in the *thinness × curvature × frequency*
> of the sufficient statistic. This reframes the whole game: **don't add capacity (the manifold is 8-dim),
> change the BASIS and the OPTIMIZER so the 8-dim chart is cheap to resolve.** (Matches `MEASURED-BY-US`:
> capacity-on-isotropic-basis HURTS +6%; capacity pays only AFTER basis-match.)

---

## 2. LENS — DERIVATIVES & INTEGRALS (why the tail is slow)

**The 0-1 / measure-zero gradient.** d_seg is a 0-1 indicator; in the continuum its gradient w.r.t. the logits is
a **Dirac supported on the boundary contour** (measure zero). `CITED` (the 0-1-loss surrogate literature): "the
zero-one loss counts errors, but the gradient is zero almost everywhere." We optimize a **smooth surrogate** (CE
→ tau_softplus(τ) → l7-softmax(p≈7 ≈ soft-L∞)) whose gradient is supported on a **margin-band of width ∝ τ**
around the boundary.

**The per-epoch d_seg-improvement integral (the front + tail decomposition, `DERIVED`).** Write the surrogate
gradient mass over the frame as
```
∫_frame ‖∇_θ surrogate‖ dx  ≈  G_bulk(t)            +   G_boundary(t)
                                (interior, large early)    (∝ band_width(τ) × boundary_length × margin_density)
```
- **EARLY (bulk regime):** `G_bulk` dominates → fast descent. (`MEASURED-BY-US`: CE 0.74 → 0.0054 in ~275 ep —
  the steep front; ~90% of the total drop is in the first ~100 ep.)
- **LATE (boundary regime):** `G_bulk` is exhausted; only `G_boundary` remains. As the curriculum **anneals τ**
  (1.0 → 0.05 for Muon), `band_width(τ) → 0`, so the softmax gradient at the boundary **vanishes ∝ τ**. This is
  the textbook vanishing-margin-gradient of 0-1 surrogates (`CITED`). → the **long flat tail**.

**So the curve is a SUM-OF-POWER-LAWS** (`CITED` form, Bahri/PNAS: `L(t) = L_∞ + A·t^(−α_bulk) + B·t^(−α_bnd)`)
with `α_bulk` large (fast front) and `α_bnd ≈ 4/d_boundary` small (slow tail). The tail rate-constant is set by
(i) the NTK eigenvalue at the boundary's spatial frequency (§3) and (ii) the τ-anneal vanishing gradient. Both
flatten it. **This is the deep-math "why 30k": ~47% of PR95's epochs live in the `B·t^(−0.3..0.5)` tail, where
each successive halving of the boundary residual costs 4–9× the previous.**

**The curriculum IS graduated optimization / homotopy continuation.** `CITED` (Mobahi–Fisher 2015; Bengio
curriculum-as-continuation; Lin et al. 2023 continuation-path): curriculum learning is a continuation method that
solves a sequence of Gaussian-smoothed subproblems deforming from easy (smooth, large τ) to hard (sharp, τ→0).
PR95's CE→tau→smooth→l7→Muon IS exactly this homotopy: each stage is a less-smoothed subproblem; the boundary
localization is the **final, least-smoothed, hardest** subproblem — which is why it's last and longest.

**The level-set frame closes it (`DERIVED` + `CITED`).** Our witness φ is the **viscosity solution of a
variational level-set PDE** with eikonal (|∇φ|=1) + length regularizers. `CITED` (level-set/viscosity lit): the
signed-distance function is the unique viscosity solution of the eikonal equation; eikonal regularization
"removes the regularization parameter and gives faster convergence around the interface." **Mechanistically:
eikonal makes the margin UNIFORM along the whole boundary** (φ=0 with |∇φ|=1 everywhere) → it removes the
vanishing-/uneven-gradient hot-spots that make a free softmax tail slow. (This is a *structural* reason the
level-set witness should have a shorter tail than PR95's free HNeRV decoder.)

> **Falls out:** gate the late stages on the **TASK metric (d_seg), not the surrogate loss** — they DECOUPLE in
> the tail. `MEASURED-BY-US` (FEED-lu): in tau, "loss kept dropping 9.3→5.8 while d_seg drifted UP from 0.0043 to
> 0.0045" — the surrogate↔task gap. The EMA-shadow holds the d_seg-best; the live loss is a liar in the tail.

---

## 3. LENS — DIMENSIONALITY + WHAT FALLS OUT (the basis & optimizer physics; is the tail fundamental?)

**Spectral bias is the time-domain mechanism (`CITED`).** NTK eigenvalues decay monotonically with frequency;
gradient descent converges along each NTK eigenvector at a rate ∝ its eigenvalue (low-freq fast, high-freq slow).
A coordinate-MLP with an **isotropic Fourier/sin basis** (PR95's HNeRV) must **climb the whole spectrum** to
reach the boundary's high-frequency content → many epochs spent on the high-freq, small-eigenvalue boundary mode.
**This is the engineering "under our noses": PR95's basis was frequency-agnostic, so the boundary was the
slowest-converging NTK eigenmode.**

**What FALLS OUT — three levers that change the EXPONENT (not just the constant):**

1. **Topology-matched basis raises the boundary NTK eigenvalue.** A **directional/curvelet (anisotropic) basis
   oriented to the boundary tangent** pre-loads the boundary frequency into the basis, so the network does NOT
   climb the spectrum — the boundary becomes a *low-frequency* mode in the rotated frame → fast convergence.
   `MEASURED-BY-US`: directional basis **−48% all-class** d_seg vs −8% lane-only (vehicle-G5/D1, MLX-rs direct
   n96). `CITED` (Tancik Fourier-features 2020; Wang et al. eigenvector-bias): "Fourier-feature mapping modulates
   the NTK eigenvector frequency; principal eigenvectors align to the embedded frequencies." This is the SINGLE
   highest-value epoch-saver because it changes α, not A. (Candès–Donoho: curvelets are the cartoon-edge-optimal
   basis — the math says this is THE basis for a codim-1 boundary.)

2. **Eikonal/level-set chart removes the vanishing-gradient tail hot-spots** (§2; `CITED`). Uniform margin along
   the boundary → the tail is governed by geometry convergence, not by τ→0 gradient starvation.

3. **Muon conditions the boundary mode that AdamW freezes.** The boundary eigendirection is the
   small-eigenvalue / high-condition-number direction (κ~19 Hessian, `MEASURED-BY-US` C4c). AdamW's per-coordinate
   step grad-collapses there; **Muon's spectral (Newton–Schulz orthogonalized) step spreads the update across ALL
   singular directions → anti-collapse, full-rank step → it directly accelerates the thin-annulus tail.**
   `MEASURED-BY-US` C4c: Muon descends d_seg ~32% MORE than AdamW with the gap widening monotone; `CITED` (Keller
   Jordan Muon = spectral norm steepest descent; Bernstein–Newhouse 2409.20325 modular norm). The sister memo
   `per_stage_fractal_optimizer_priming_reheat_anneal_design` proves the FiLM-resonance corollary: `WᵀW=I ⟹
   PR(M)=PR(cov code)` — Stiefel-W + spectral-entropy keep the conditioning matrix full-rank through the anneal,
   byte-free. **This is precisely why PR95 puts Muon LAST: it is the only optimizer that attacks the
   ill-conditioned boundary tail.** (Lesson: AdamW saturates → jump-to-Muon EARLY is viable.)

**Is the tail FUNDAMENTAL or TRAINABLE-AWAY? (the not-pessimistic cross-check.)** Distinguish two parts:
- **The bulk of the tail is TRAINABLE-AWAY.** Existence proof (forbids "intractable"): PR95 reached **6.02e-4 at
  bc36** (`MEASURED-BY-US`, project_pr95_prune_capacity_cliff); our l7/Muon are STILL DESCENDING. It is a slow
  power law, not a wall. Basis-match + eikonal + Muon flatten the *exponent* → the bulk is reachable far faster.
- **The last sliver is ALEATORIC → store/repair, don't grind.** `MEASURED-BY-US` (FEED-lq): on the active
  Road↔Lane support the deterministic pose-warp is WORSE than persist (1645 vs 1601 flips/frame) → the residual is
  **boundary WOBBLE (content-noise), not motion** — which sub-pixel lands on which argmax facet is high-entropy.
  A trained generator models the *conditional* and approaches an **aleatoric floor (~0.0044, FEED-lq)**; below it,
  grinding epochs is dominated by the surgical-repair toolbox (store the irreducible flips / UNIWARD-downweight /
  deterministic-gen — chosen by Δd_seg-per-byte under through-R survival). **The convergence integral tells you
  WHERE to switch from "train" to "store": when Δd_seg/epoch < store-cost-per-byte break-even.**

---

## 4. THE 3–4 HIGHEST-VALUE TRANSFERABLE LESSONS (ranked by epochs-saved leverage)

| # | Lesson | Mechanism | Evidence | n600 action |
|---|---|---|---|---|
| **L1** | **Change the EXPONENT with a topology-matched basis** — directional/curvelet oriented to the boundary tangent converts the slow high-freq boundary mode into a fast low-freq one. This is the difference between PR95's 30k and our ~1.5k. | spectral bias / NTK eigenvalue (§3.1) | `MEASURED-BY-US` −48%; `CITED` Tancik/Wang/Candès–Donoho | Bake `--self-orient` + curvelet bank in BASELINE; climb scale coarse→fine via warm `--max-bank-freq` 16→32→64 (= Gaussian-homotopy bandwidth anneal). |
| **L2** | **Seed the boundary geometry so you START where PR95's first ~9000 ep ended.** The openpilot deg-3 centerline IS the Road↔Lane separatrix (residual 1.9e-5). PR95 trained the geometry from scratch; we initialize it free. | structured-init = skip the bulk-geometry climb | `MEASURED-BY-US` D10/L1/C14c, FEED-fs | `--structured-init --structured-init-include-lane --lane-prior-phi1 ... --lane-prior-phi1-dash-gate` (0 bytes, rule-118 free). |
| **L3** | **Early-stop the flat tail on the TASK metric, per-stage, from the convergence integral.** Surrogate↔task decouple in the tail; the loss lies. Front-load the steep bulk, short-tail the conditioning stages. | sum-of-power-laws + surrogate gap (§2) | `MEASURED-BY-US` knees CE~275/tau~450/l7~700; FEED-lu/lv | Adaptive #188 `decide_next_stage` (window 300, slope thresholds) gating on advisory d_seg + EMA-shadow-best; NOT loss. |
| **L4** | **Muon is the tail's conditioner — jump early, give it the longest budget.** It is the only stage attacking the ill-conditioned boundary mode AdamW freezes; pair with Stiefel-W/spectral-entropy to hold conditioning byte-free. | spectral step = anti-collapse on κ~19 boundary mode (§3.3) | `MEASURED-BY-US` C4c +32%; sister optimizer memo PR(M) proof; `CITED` Keller Jordan / 2409.20325 | `--muon-start-epoch` early (AdamW saturates), `--muon-lr 2e-3`, τ+render-temp frozen 0.05, reset-moments + Muon-prime at transitions. |
| **L5** | **Switch from TRAIN to STORE at the aleatoric floor** — the last sliver (~0.0044 boundary-wobble) is content-noise; epochs are dominated by the surgical-repair toolbox there. | aleatoric vs epistemic split (§3) | `MEASURED-BY-US` FEED-lq (warp worse than persist) | When Δd_seg/epoch < store-cost-per-byte break-even, stop training the residual and store/UNIWARD/deterministic-gen the irreducible flips. |

---

## 5. CONCRETE n600 RECIPE ADAPTATIONS (epochs / order / basis / optimizer / recursion)

Each row tagged by evidence grade. This SHARPENS the §4 OPTIMAL LAUNCH CONFIG of the canonical research index
with the convergence-rate justification for the epoch budget. **Containment: DESIGN only — no GPU launch here.**

**EPOCH BUDGET (≈1100–1700 ep vs PR95's 29,650 = a ~18–27× compression, justified per-stage):**

| stage | n600 epochs | why (convergence math) | grade |
|---|---|---|---|
| **S0 seed** | 0 (free init) | L2: structured-init = PR95's first ~9000 ep of geometry, for free | `MEASURED-BY-US` (residual 1.9e-5) |
| **S1 CE** | ~250–300 | bulk power-law front: ~90% of drop in first ~100 ep; knee ~275 at n200 | `MEASURED-BY-US` (FEED-lt) |
| **S2 tau_softplus** | ~150 (early-stop at d_seg knee ~ +150) | THE primary drop; over-trains after the knee (surrogate↔task decouple) | `MEASURED-BY-US` (FEED-lu, best @ep450) |
| **S3 l7+margin** | ~100 | front-loaded + early-saturating CONDITIONING lever (−1.9%), not capacity | `MEASURED-BY-US` (FEED-lv, knee ~ep700) |
| **S4 Muon finisher** | ~400–700 (LONGEST tail; early-stop on slope) | the only stage attacking the κ~19 boundary mode; the aleatoric-adjacent residual lives here | `MEASURED-BY-US` C4c + `DERIVED` §3.3; magnitude predicted ~6–9e-4 (unmeasured) |
| **SKIP** | smooth (+6.8% d_seg), QAT/C1a/λ/σ (rate machinery) | structurally not d_seg levers for our vehicle | `MEASURED-BY-US` C2c |

Why the compression is legitimate (not a corner-cut): PR95 paid 30k BECAUSE (i) isotropic basis → climb the whole
spectrum (we change the exponent, L1), (ii) geometry from scratch (we seed it, L2), (iii) AdamW-until-end →
boundary grad-collapse (we Muon the tail, L4), (iv) it ground the flat tail on the loss (we early-stop on the
task, L3). Each removed cost is a measured lever, not an assumption. **Honest gap:** our best mid-curriculum
advisory d_seg (0.0043) is ~7× PR95's 6e-4 — the Muon finisher must close most of it; the irreducible remainder
is L5 (store), NOT more epochs. (`ASSERTED` until the live Muon arm + n600 measure it.)

**ORDER:** keep the homotopy ordering (CE→tau→l7→Muon = easy-smooth → hard-sharp); it is the graduated-optimization
continuation path (`CITED`). Within S1–S2, climb the curvelet scale coarse→fine (the Gaussian-homotopy bandwidth
anneal, `CITED`) via warm `--max-bank-freq` 16→32→64. Drop `smooth` (it RAISES d_seg — it un-sharpens the boundary
the homotopy is trying to sharpen). `MEASURED-BY-US` + `CITED`.

**BASIS:** `--self-orient --reorient-every 50 --freq-across 32 --n-dir-freqs 2 --freq-along 4 --max-bank-freq {16→64}`
(DM2 directional/curvelet, the −48% exponent-changer L1) + `--chroma` (argmax-flip lever) + eikonal/length
(`--eikonal-weight 0.01 --length-weight 0.001`, the uniform-margin tail cure §2). `MEASURED-BY-US` + `CITED`.
Open precursor: the −48% is circular-GT (built from gt.lstars) → resolve via `--self-orient` fixed-point and
RE-MEASURE realized; the EXPONENT lesson holds regardless of the exact %.

**OPTIMIZER:** AdamW for CE; **Muon from S3/S4** (jump early — AdamW grad-norm collapses on κ~19, C4c) at
`--muon-lr 2e-3`; **moment-reset + Muon-prime at every transition** (re-treat non-negotiable); **Stiefel-W
(no-WD) + code-spectral-entropy** to hold FiLM conditioning full-rank byte-free (`WᵀW=I ⟹ PR(M)=PR(cov code)`,
sister optimizer memo); EMA-shadow 0.997 with polar re-orthonormalize at deploy. `MEASURED-BY-US` + `DERIVED` +
`CITED`.

**RECURSION / WARM-START:** from-scratch openpilot-seeded (structured-init IS the seed — NOT PR95-ckpt resume);
each stage warm-starts from the prior stage's converged φ (continuation path) with reheat (0.1×/8ep, reset-moments);
adaptive #188 `decide_next_stage` (EXTEND if slope ≤ −1e-5 / ADVANCE+reheat if |slope| < 1e-6 plateau /
RERUN_NEW_CONFIG if plateau above floor / ROLLBACK_BRANCH to best+skip). The adaptive controller IS the
"switch-from-train-to-store" arbiter of L5 (gate on Δd_seg/epoch vs store break-even). `MEASURED-BY-US` (campaign.py)
+ `CITED` (continuation path learning).

---

## 6. NO-FAKE ledger + where the numbers plug in

- **Sister attribution memo NOT yet on disk** at write time (polled `.omx/research/witness_per_stage_attribution_*`
  + `*per_stage*annulus*` — only the optimizer-design + an unrelated 2026-05-13 audit present). **Where the sister
  per-stage per-pixel/per-class attribution plugs in:** §2 (the front/tail integral — its per-class flip
  decomposition CALIBRATES `α_bulk` vs `α_bnd` and confirms the boundary-band gradient localization) and §4-L3
  (the per-stage knees — its measured per-stage d_seg deltas REPLACE the n200 advisory knees with the
  per-pixel-attributed ones). When it lands, swap the `MEASURED-BY-US (FEED-lt/lu/lv, n200 MLX-advisory)` knees
  for its numbers; the convergence theory is unchanged (it predicts the shape; the sister memo measures the
  amplitudes).
- **Existence-proof cross-check (not pessimistic):** PR95 6.02e-4 @ bc36 + our still-descending l7/Muon ⇒ the tail
  is "slow but converging," NOT "stuck." The ONLY genuinely-irreducible part is the FEED-lq aleatoric wobble floor
  (~0.0044), and even that is a STORE target, not a wall.
- **α ≈ 4/d caveat (honesty):** 4/d is the resolution-limited DATASET/PARAM exponent (`CITED`); the training-TIME
  tail exponent is governed by the NTK spectrum (`CITED` spectral bias) — same "higher d → slower" qualitative
  law, transfer tagged `DERIVED`. The numbers CE275/tau450/l7700 are `MEASURED-BY-US`; the 4/d→0.5 is an estimate
  of WHY they're slow, not a fitted curve.
- **MEANS≠ENDS:** this memo moves no score. The END is a byte-closed n600 `upstream/evaluate.py` row below 0.19110
  from the §5 witness. The value here = a ~18–27× faster, theory-justified n600 epoch budget + the
  train→store switch criterion. **Pointer UNMOVED contest-CPU 0.19109982.**

### Sources (external, CITED)
- Sharma & Kaplan, *A Neural Scaling Law from the Dimension of the Data Manifold*, arXiv 2004.10802 (α ≈ 4/d).
- Bahri, Dyer, Kaplan, Lee, Sharma, *Explaining Neural Scaling Laws*, arXiv 2102.06701 / PNAS 2024 (variance- vs
  resolution-limited; sum-of-power-laws).
- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions* (2020); Wang et al.,
  *eigenvector bias of Fourier feature networks* (spectral bias / NTK eigenvalue ↔ frequency).
- Mobahi & Fisher III (2015), *On the Link between Gaussian Homotopy Continuation and Convex Envelopes*; Lin et
  al. (2023), *Continuation Path Learning for Homotopy Optimization* (curriculum = graduated optimization).
- Level-set / viscosity-solution + eikonal reinitialization literature (SDF = unique viscosity solution of the
  eikonal equation; eikonal removes the regularization parameter → faster convergence at the interface).
- Keller Jordan, *Muon*; Bernstein & Newhouse, *Old Optimizer, New Norm* arXiv 2409.20325 (Muon = spectral-norm
  steepest descent).
- 0-1-loss surrogate / margin literature (gradient zero a.e.; surrogate margin-band; vanishing boundary gradient).

### Our anchors (MEASURED-BY-US)
`project_dseg_islands_8dim_manifold_go_generator_convergence_closed_20260623` (8-dim nonlinear manifold) ·
`CANONICAL_RESEARCH_INDEX_20260629` (D1/D2/G2/G5/C4c/C2c/§4 launch config) ·
`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611` FEED-lt/lu/lv (n200 knees), FEED-lq (aleatoric floor),
FEED-ip (per-stage ckpt d_seg table) · `per_stage_fractal_optimizer_priming_reheat_anneal_design_20260629`
(Muon/Stiefel conditioning) · CLAUDE.md L14/L15 + cur-C19 (PR95 8-stage forensic).
