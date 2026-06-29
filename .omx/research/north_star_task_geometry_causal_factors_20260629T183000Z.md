# North-Star R1 — Is the minimal description of a fixed task ALWAYS the causal/generative structure of the world it attends to, read off the task's Fisher geometry?

**UTC** 2026-06-29T18:30Z · **tag** `[macOS advisory / research-signal]` · **pointer 0.19110 UNMOVED** (no exact score; this is theory + a $0 CPU/numpy measurement, NOT a byte-closed row).
**Lineage** DAG FEED-iy (North-Star), recursion R1 / play-fleet F2 (task-geometry → causal-factors map). Sister: FEED-id (Fisher↔margin co-location, MEASURED), grok-test FEED-ja (pose→d_seg generative half, MEASURED).
**Deliverables landed here:** (a) the general map + a conditions theorem; (b) literature grounding + novel-vs-known verdict; (c) a NEW $0 measurement (`tools/north_star_fisher_manifold_dim.py`, `experiments/results/north_star_fisher_manifold_20260629T182433Z/results.json`); (d) the codec / 10-yr-program implication + how to validate.

---

## 0. One-paragraph answer (the North-Star "ALWAYS" = **NO**)

**The strong claim — "minimal description = the causal generative factors, for ANY fixed task" — is FALSE, and provably so.** Three published barriers: (B1) Locatello 2019 impossibility (no identifiability without inductive bias); (B2) a **rank ceiling** — `F=diag(p)−ppᵀ` for K classes has rank K−1, so a single K-class task's input-Fisher metric `G=JᵀFJ` has rank ≤ K−1 and **cannot resolve more than K−1 latent directions** (for SegNet's per-pixel K=5: ≤4); (B3) observational ≠ causal — the causal partition is a *coarsening* of the observational sufficient partition and equality needs **interventions** (Chalupka–Eberhardt–Perona 2017). What a single (multi-class) task DOES recover is the **task-relevant subspace, up to a linear/component-wise indeterminacy** (Roeder–Metz–Kingma 2021; Khemakhem 2020) — a *scramble of the projection of the factors the task attends to*, **not** "the" causal factors.

**What is TRUE (and the contest realizes it):** the chain *input-Fisher nullspace = task invariances · high-Fisher manifold = the task's minimal sufficient statistic · = the indirect-RD-optimal code* holds and is **entirely assembled from known results** (Tron et al. 2022 input-Fisher foliations; Tishby 1999 IB=min-suff-stat; Wolf–Ziv 1970 + task-oriented indirect-RD). The contest **lifts the rank-(K−1) ceiling** (a *dense* per-pixel field × moving sequence × 6-DOF pose = a high-rank task + built-in auxiliary variability, the Roeder/iVAE rescue) so its manifold's relevant projection ≈ the scene's pose-rideable geometry. **Our genuinely-original, defensible piece is NOT a theorem** — it is the **measured bridge** on the real frozen contest scorer: the SegNet Fisher manifold's motion is **51% (linear) / 70% (2nd-order) explained by the 6-DOF ego-pose, Road (0.53) ≫ Lane (0.36)** (the pure-Fisher-side reproduction of the grok depth-stratification) + the explicit accounting of which factors the manifold captures vs sends to its nullspace + the codec instantiation. Pointer 0.19110 UNMOVED; advisory.

---

## 1. The general map (deep-math, deliverable a)

**Object.** A frozen task is a smooth map `T = softmax∘f : X → Δ^{k-1}` (inputs → label simplex), inducing the statistical model `{p_x = T(x) : x∈X}` indexed by the input.

**Output Fisher.** `F(x) = diag(p_x) − p_x p_xᵀ` — the Fisher information of the Categorical family in its logit (natural) parameters; rank ≤ k−1; **vanishes at the simplex vertices** (confident output) and is **maximal where p is spread** (a decision boundary).

**Pullback (induced Fisher–Rao) metric on input space.** `G(x) = J(x)ᵀ F(x) J(x)`, `J = ∂f/∂x`. This is the pullback of the Fisher–Rao metric under `T` (Amari information geometry). It is a PSD `d×d` field on `X`.

### FACT 1 — Fisher nullspace = task invariances (KNOWN; published for the INPUT metric)
`v ∈ ker G(x) ⟺ vᵀG v = 0 ⟺ ‖F^{1/2}J v‖ = 0 ⟺ D_KL(p_x ‖ p_{x+εv}) = o(ε²)` — perturbing `x` along `v` does not change the output distribution. So **`ker G` is the tangent space of the task's local invariances** (the directions to discard). Two sources of nullity: F-null (saturated output = confident interiors) and J-null (logits don't depend on `v`). **This exact input-space statement is published**: Tron, Couëllan et al. 2022/2024 (arXiv:2203.00922) equip input space with `G=JᵀFJ`, prove it is generically semi-definite, and show its **kernel foliates the data manifold** while the transverse (high-curvature) directions carry task sensitivity. So the "read invariances off the input-pullback Fisher" idea is NOT ours-novel.

### FACT 2 — high-Fisher manifold = minimal sufficient statistic coordinates (KNOWN)
Define `x ∼ x' ⟺ p_x = p_{x'}`. The **minimal sufficient statistic** `S*` for the target `Y` is the quotient by `∼` (Fisher–Neyman: a statistic is sufficient iff `p(Y|x)` factors through it; *minimal* = coarsest = the likelihood-equivalence partition). The tangent spaces of those equivalence classes are exactly `ker G` (FACT 1); the **transverse, high-curvature directions (col-space of `G`) parameterize `S*`.** (IB = minimal sufficient statistic, Tishby–Pereira–Bialek 1999; deterministic-net version Cvitkovic–Koliander 2019; weight-Fisher↔invariance bridge Achille–Soatto 2018.) **Rank ceiling (the precise barrier):** rank `F ≤ K−1` ⇒ rank `G ≤ K−1`, so a *single* K-class output resolves ≤ K−1 directions of `S*`. A coarse task has a tiny `S*`; the contest escapes this only because its task is a *dense field* (per-pixel block-Fisher of total rank up to `H·W·(K−1)`) — density LIFTS the ceiling.

### FACT 3 — the conditional bridge: S* = causal generative factors (TRUE under C1–C4; the NOVEL part)
Source SCM: latent factors `Z = (Z_R, Z_I)` (relevant/irrelevant); observation `x = h(Z)` (the renderer, including the contest `R` operator); target `Y = c(Z_R)`. Then `S*(x) = Z_R` up to the identifiability group **iff**:

- **C1 Recoverability** — `h(·, Z_I)` is injective in `Z_R` (`x` retains `Z_R`). *Violation = the R-survival wall (GAP2): the partition can be geometrically correct yet `R` (↑874 bicubic → uint8 → ↓384 → argmax) destroys the thin-lane info → `S*` loses it.*
- **C2 Identifiability variability (THE CRUX)** — the labelling `c`, with the available variation, must separate all `Z_R` configurations. A single coarse label **cannot** (Locatello et al. 2019 impossibility for the unsupervised limit); identifiability requires **auxiliary variability** (Hyvärinen–Morioka TCL/PCL; Khemakhem et al. 2020 iVAE: recover latents given an auxiliary `u` rendering sources conditionally-independent / exponential-family). **The contest satisfies C2 by construction**: the "task" is not one label — it is a **dense k=5 per-pixel field × a moving temporal sequence × a 6-DOF pose regression**. The pixel multiplicity + temporal index + pose ARE the auxiliary variable `u` → `c` is effectively injective on the scene factors. (This is *why* the demo below finds the manifold's motion pose-explained: pose is the identifying auxiliary signal.)
- **C3 Disentanglement** — `Z_I` enters `x` only through `ker G` (the task is genuinely invariant to the irrelevant factors), else `G` mixes relevant/irrelevant and `S*` is contaminated. *Empirically: the invariance set is 86% of area but only 18% of Fisher mass (demo §B) — the task discards most of `x`.*
- **C4 Causal (not merely statistical) sufficiency / ICM** — the recovered `S*` are the causal parents of `Y`, not spurious correlates (Independent-Causal-Mechanisms, Schölkopf et al. 2021). **Holds by construction when the frozen task is a well-trained perception net whose target IS the physical scene**: SegNet/PoseNet were trained to recover road geometry + ego-motion → their minimal sufficient statistic ≈ the world's causal factors. *The task attends to the world because it was trained on the world's mechanism.*

**Verdict on the North-Star claim:** the composite is **TRUE-UNDER-CONDITIONS**, the conditions are explicit, **C2 is the generic barrier** (single-task identifiability), and the contest is a microcosm that satisfies C1 (mostly — GAP2 is the C1-residual), C2 (label richness), C3, C4 (perception-net-by-construction). It is **NOT a universal law** ("always") — a single coarse fixed task generally fails C2.

---

## 2. Literature grounding + novel-vs-known (deliverable b)

Web-verified, load-bearing citations (real; arXiv/venue confirmed this session — by me and/or the lit subagent):

| Link in the chain | Status | Anchor (verified) |
|---|---|---|
| Parameter-space Fisher / natural gradient | **KNOWN** | Amari 1998 *Natural Gradient Works Efficiently in Learning*, Neural Comp 10(2); spectrum anisotropy Karakida–Akaho–Amari 2019 (arXiv:1806.01316). (Parameter, not input.) |
| **input-space** `G=JᵀFJ`: `ker G` = data-manifold foliation / invariances; transverse = task sensitivity | **KNOWN (published)** | **Tron, Couëllan et al. 2022/2024**, *Adversarial attacks through canonical Riemannian foliations* (arXiv:2203.00922). ⟵ this is *our G*, already in the literature. |
| `ker G` = invariances; min-suff-stat ↔ Fisher | **KNOWN** | Fisher–Neyman (classical); Achille–Soatto 2018 *Emergence of Invariance & Disentanglement*, JMLR 19 (weight-Fisher ↔ invariance/minimality). |
| IB-optimal representation = minimal sufficient statistic | **KNOWN** | Tishby–Pereira–Bialek 1999 (arXiv:physics/0004057); deterministic-net version Cvitkovic–Koliander 2019 (arXiv:1905.07822). |
| remote/CEO under log-loss = IB; **indirect-RD optimum factors through a sufficient statistic of the target** | **KNOWN** | **Courtade & Weissman 2014** (arXiv:1110.3069); **Wolf–Ziv 1970** estimate-then-compress separation; task-oriented IB Shao–Mao–Zhang 2022 (IEEE JSAC); indirect-RD semantic sources arXiv:2201.12477, arXiv:2602.12866. |
| single fixed task **cannot** identify generative factors | **KNOWN (impossibility)** | Locatello et al. 2019, ICML (arXiv:1811.12359); root Hyvärinen–Pajunen 1999. |
| **rank-(K−1) ceiling** on a single K-class task's Fisher | **KNOWN (first-principles)** | `rank(diag(p)−ppᵀ)=K−1` ⇒ ≤ K−1 resolvable directions. |
| auxiliary variability / class-multiplicity **rescues** identifiability (to a SUBSPACE) | **KNOWN** | Khemakhem 2020 iVAE (arXiv:1907.04809, up to perm+componentwise); **Roeder–Metz–Kingma 2021** (arXiv:2007.00810, multi-class softmax identifiable **up to a linear transform**); Hyvärinen–Sasaki–Turner 2019. |
| observational ≠ causal sufficiency (need interventions) | **KNOWN** | Chalupka–Eberhardt–Perona 2017 *Causal Feature Learning* (causal macrovariable = sufficient partition, **interventional**; causal partition = coarsening of observational); Schölkopf et al. 2021 (arXiv:2102.11107). |
| task manifold is low-dim; input-Jacobian spectrum probes it | **KNOWN** | Ansuini et al. 2019 (arXiv:1905.12784); Pope et al. 2021 (arXiv:2104.08894). |

**What is GENUINELY NOVEL vs assembled-from-known-pieces (corrected, sharpest statement):** *No new theorem, and the "left half" is MORE known than first drafted.* The chain *input-Fisher → minimal sufficient statistic → low-dim manifold → indirect-RD-optimal code* is **fully assembled from published results** (Tron 2022 input-Fisher foliations; Tishby 1999; Wolf–Ziv 1970 / task-oriented indirect-RD; Ansuini 2019) — integration, not invention. The *causal* terminus is not merely unproven but **provably blocked in general** (Locatello; the rank-(K−1) ceiling; observational≠causal), recoverable only up to a **linear/component-wise scramble of the task-relevant projection**, and only to *causal* status if interventional structure is added. **The honest defensible contribution is therefore (i)** the **measured** Fisher-manifold→ego-pose decomposition on the *real frozen contest scorer* (§3: eff-dim ~6, pose R² 0.51/0.70, Road≫Lane) — that specific empirical bridge is new; **(ii)** the explicit **accounting of which generative factors the contest manifold captures (the pose orbit, ~70%) vs sends to its invariance nullspace**, with the rank-(K−1)→density-lift mechanism for the C2 escape; **(iii)** the **codec instantiation** (store `S*` = pose orbit free + the ~21-dim residual). The North-Star "minimal description = causal factors" as a *universal law* is **rejected**.

**Adversarial overturn-audit (both directions, per FEED-ix):**
- *Overturning my own too-easy "novel" (the subagent caught this):* the input-pullback-Fisher → manifold/invariance reading IS published (Tron 2022); the codec spine IS published (Wolf–Ziv; task-oriented indirect-RD). My first-draft "operational recipe novelty" was demoted accordingly. **Honesty win, not loss.**
- *Overturning a too-easy "totally novel":* punctured above — the buildable part is known; the causal claim is Locatello/rank-ceiling/observational-blocked.
- *What survives both:* the **measurement** (no one measured the SegNet Fisher-manifold→pose decomposition) and the **contest-specific factor accounting + codec instantiation** stand as ours-original, advisory-grade.

---

## 3. The NEW $0 measurement (deliverable c) — Fisher-manifold dimension + generative factors

`tools/north_star_fisher_manifold_dim.py` on cached `gt_n96` (96 frames, CPU/numpy, 3.2 s, NO GPU). κ = `2·σ(margin)·(1−σ(margin))` = the top-2 block of `F=diag(p)−ppᵀ`, exact given the margin (FEED-id validated margin = byte-faithful Fisher surrogate, Pearson 0.978 vs full `‖F‖`). We do the half FEED-id/grok did **not**: the manifold's **dimensionality** and its **generative-factor regression**.

**(A) Fisher-mass concentration (codim-1 thinness).** 50% of the Fisher mass lives in **3.3%** of pixels; Gini(Fisher mass) = **0.80**. (90% needs 32% — κ has a slow sigmoid tail, so the *very* high-curvature core is thin while the shoulder is broad; honest.)

**(B) Nullspace = invariances.** `{κ < 0.02}` (margin ≳ 4.6) = **86.1% of area but only 17.6% of Fisher mass**; the manifold = **13.9% of area / 82.4% of Fisher mass**. Clean "low-curvature = discard, high-curvature = the statistic."

**(C) Per-class Fisher mass (flip distribution from PURE Fisher, no flip labels).** Road **0.470**, Undrivable 0.252, MyCar 0.133, Lane 0.084, Movable 0.060 (absolute). **Mass-per-area:** Lane **14.3** ≫ Movable 3.9 > Road 2.0 ≫ Undrivable 0.51 ≈ MyCar 0.52. → independently re-derives, from geometry alone: **Road carries the most absolute Fisher mass** (longest boundary perimeter; matches the known ~50% class-0 flip mass) and **Lane is the highest-density unstable orbit** + Undrivable/MyCar are the stable cores (#139). No hand-labelling used.

**(D) Intrinsic dimension + generative-factor (pose) regression — the headline.**
- pose effective dim (of 6) = **4.08**; moving-Fisher-manifold effective dim (participation ratio of the downsampled continuous κ field) = **5.96** (90%-var needs 28 dims; 95% needs 44 — the high-freq lane/movables tail).
- **Manifold motion explained by the 6-DOF ego-pose: linear R² = 0.514; 2nd-order (Taylor of the homography action) R² = 0.700** (in-sample, df 28/96 — directional, not held-out).
- **Per-class:** Road κ-field pose-R² = **0.527** ≫ Lane κ-field pose-R² = **0.363** — the **pure-Fisher-side confirmation of the grok depth-stratification** (Road = ground-plane homography, pose-explained; Lane = survival residual, less pose-explained).
- **Off-pose residual:** 48.6% of variance, effective dim ≈ 21 — the lane-survival + movables irreducible part (the only learned bytes).

**Rank-ceiling consistency (delightful, flagged as consistency NOT proof):** the per-pixel K=5 output Fisher has rank K−1 = **4**; the measured **pose effective dim = 4.08** sits right at that ceiling, while the *field* effective dim lifts to **5.96** (density across pixels/frames lifts the per-output ceiling, exactly as FACT 2 predicts). Consistent with the theory; not a proof (could be coincidence at this n).

**The chain, demonstrated end-to-end on real cached scorer geometry:** Fisher metric (read off margin) → manifold (14% area / 82% Fisher mass, eff-dim ~6) → **its motion is a smooth function of the 6-DOF ego-pose (R² 0.51→0.70, Road≫Lane)** → the residual is low-dim. That is "high-curvature manifold = sufficient statistic, its generative factors = the ego-pose world-model," **measured** — though per §2 this recovers the task-relevant *projection* (the pose-rideable scene geometry), not a certified causal-factor set.

---

## 4. Implication for task-space codecs + the 10-yr program (deliverable d)

**The optimal task-space codec IS the minimal sufficient statistic of the task** (Courtade–Weissman: remote-RD under log-loss = IB = minimal sufficient statistic; our seg surrogate is log-loss). Reading it off the Fisher geometry gives a **constructive design recipe** — *derive the representation from the task geometry, don't brute-learn it*:

1. **Surrogate the task Fisher metric** (here: the margin field, free — FEED-id).
2. **High-curvature manifold = what to store** (here: the codim-1 annulus = union of inter-class edges).
3. **Its generative factors = the parameterization → store THOSE** (here: the ego-pose orbit — free / dual-use d_pose+d_seg; §3D shows ~70% of the manifold rides it).
4. **Off-factor residual = the only learned bytes** (here: lane-survival + movables, ~half the variance, ~21-dim — bounded, small).
5. **Nullspace = discard = free rate** (here: 86% of area, the confident interiors).

This is exactly the **v2 store-canonical + per-class pose-warp** codec (grok FEED-ja / FEED-iv arithmetic): pose free + small residual + integer decode. §3 quantifies the residual budget (the binding learned part) and confirms the bulk is pose-rideable.

**10-yr program ("perception for a purpose"):** the C1–C4 theorem is the spec for *any* frozen-task codec — and a research instrument: C1 measures rendering-information loss (the R-survival probe = GAP2), C2 measures label-richness / identifiability (when does a task pin its world?), C3 measures disentanglement (invariance mass), C4 is the perception-net-by-construction guarantee. The self-mirror (FEED-iy #3): the recipe applied to the *research* task is the DAG↔DSL↔equations triality.

**How to validate (turn advisory → real):**
- **Held-out frames** for the pose-R² (the 0.70 is in-sample, df 28/96) → use `gt_heldout_n400` / `gt_strided_n200` for an out-of-sample R².
- **Through-R** version (GAP2): the §3 manifold is PRE-R; the C1-violation residual is the binding wall — measure the Fisher manifold THROUGH the R operator (the queued $0 follow-up).
- **The only end (means≠ends, NO-FAKE):** a byte-closed exact `evaluate.py` row < 0.19110. This memo sharpens *what to store* (pose orbit free + the ~21-dim residual) and *why*; it does **not** move the pointer. Pointer **0.19110 UNMOVED**.

---

## 5. Borrowed-substrate accounting

- **Borrowed (known, cited §2) — larger than first drafted:** Fisher–Rao/pullback metric (Amari); **the input-space `G=JᵀFJ` foliation = manifold/invariance reading (Tron 2022)**; minimal sufficient statistic / Fisher–Neyman; weight-Fisher↔invariance (Achille–Soatto); IB = min-suff-stat (Tishby); remote/CEO under log-loss = IB + **estimate-then-compress / indirect-RD = sufficient statistic (Courtade–Weissman, Wolf–Ziv, task-oriented IB)**; single-task non-identifiability + rank-(K−1) ceiling (Locatello; first-principles); class-multiplicity rescue to a *linear/componentwise* indeterminacy (Roeder; Khemakhem/Hyvärinen); observational≠causal (Chalupka; Schölkopf); low-dim task manifold (Ansuini, Pope).
- **Ours-original (narrowed, honest):** (a) the **measured** Fisher-manifold→ego-pose decomposition on the real frozen contest scorer (`north_star_fisher_manifold_dim.py`: eff-dim ~6, pose R² 0.51/0.70, Road≫Lane, rank-ceiling consistency) — not previously measured; (b) the explicit **per-factor accounting** of what the contest manifold captures (pose orbit ~70%) vs nullspaces, with the rank-(K−1)→density-lift C2-escape mechanism; (c) the **codec instantiation** (store `S*` = pose orbit free + ~21-dim residual = the v2 arithmetic). NOT ours: the general input-Fisher→sufficient-statistic→indirect-RD chain (published) — and NOT a theorem.
- **Status:** advisory/research-signal; a measurement + contest-specific synthesis, not a new theorem; the universal "minimal description = causal factors" law is **rejected**; a means toward the codec, not an exact row. Pointer 0.19110 UNMOVED.

## 6. Artifacts
- tool `tools/north_star_fisher_manifold_dim.py`
- JSON `experiments/results/north_star_fisher_manifold_20260629T182433Z/results.json`
- this memo; DAG FEED appended; pointer 0.19110.
