---
title: "Deep-Math Lens — Chapter 3 keystone: Singularity/Catastrophe theory of the argmax + the HARD↔SOFT DUALITY theorem"
paper: "Amortizing the Argmax (task #284), Chapter 3 (the keystone)"
date: 2026-07-04
axis_tag: "[advisory only] — deep-math synthesis; MEANS not ends; moves NO exact pointer"
pointer_state: "0.19110 UNMOVED (this memo is $0 research; #205 sacred/read-only)"
evidence_grade: "derivation + cross-check against our own MEASURED anchors (n600 CPU-torch SegNet, bit-exact to cached margins)"
no_fake_note: "The proven-vs-conjectured boundary of the DUALITY is the single load-bearing honest call. It is stated explicitly in §3 and every claim in §1–§2 carries a PROVEN / PARTIAL / CONJECTURE tag."
triality: "DAG(FEED-03t simplex probe, FEED-gi CE→tau, post-muon annulus) ↔ equations(scalar_top1_top2_margin_is_exact_distance_to_flip_v1 + tau homotopy) ↔ this memo (the WHY)"
---

# Chapter 3 — The argmax is a max-singularity; softmax→argmax is Maslov dequantization; and the three ladders share one foot

## §0. The verdict up front (the honest one-paragraph answer)

**The three-way "duality" the paper stands on is REAL but it is NOT one theorem.** It is *three rigorous, textbook limit theorems whose limit OBJECT coincides*, plus *one genuinely new, rigorous bridge that we can prove for our specific setting*, plus *one heuristic that we should stop calling a theorem*:

- **PROVEN (textbook), and the spine of the chapter:** softmax/log-sum-exp at temperature `τ` → hard `argmax`/`max` as `τ→0` **IS Maslov dequantization** — the deformation of the ordinary semiring `(+,×)` into the tropical (max-plus) semiring `(max,+)` as the "Planck constant" `ħ = τ → 0`. This is exact, it is our `τ` curriculum, and the hard partition it converges to is a **tropical hypersurface complement / power diagram**, with the decision boundary a codim-1 tropical variety and triple junctions its codim-2 vertices.
- **PROVEN (textbook), a DIFFERENT limit:** the phase-field (Modica–Mortola / vector Allen–Cahn) energy `Γ`-converges to a **weighted perimeter** as the *spatial* interface width `ε → 0`, with triple junctions obeying a Young/Herring angle law. This is our `--length-weight` (`∫ds`) and `--eikonal-weight` (`|∇φ|=1`) regularizers. Its small parameter is spatial `ε`, **not** the logit temperature `τ`.
- **PROVEN (textbook):** the softmax family carries a **Fisher–Rao metric** under which the training flow is **mirror descent = natural-gradient descent** (Amari; Raskutti–Mukherjee 2015). This is the Ch.2 optimization-geometry view.
- **NEW & PROVABLE for us (the bridge that makes it ONE picture):** the **Fisher-information field of the softmax_τ is a caustic** — it is bright (rank-1 ridge) on the margin-zero set and dark in cell interiors — and this caustic **is exactly the `τ`-smoothing of the tropical hypersurface**, collapsing onto it as `τ→0`. Our own measured `Fisher-curvature ↔ (−margin)` Pearson **0.978** is this bridge, and it is *near-tautological by the 2-logit softmax formula* (which is why it is trustworthy, and why it does **not** license "margin = UNIWARD steganographic cost", a separate claim we measured near-zero).
- **CONJECTURE / heuristic (do NOT overclaim):** that our specific `CE→tau→l7→Muon` training trajectory *is* a discretized natural-gradient descent along the Maslov deformation whose endpoint is the `Γ`-limit sharp partition. It is well-motivated but it is not a single theorem: `τ→0`, `ε→0`, and the `l7` (`p→∞`, `Lᵖ→L^∞`) leg are **three different small parameters**, Muon is a preconditioner outside the continuum limit, and our scorer is a deep net so the tropical object is only **locally** polyhedral.

Everything below expands, proves, or bounds each of these, always tied to a MEASURED number from our own runs.

---

## §1. The argmax as a max-singularity: stratified space, discriminant, and where the residual lives

### 1.1 The object. `d_seg` is a functional of a `max`, and a `max` of smooth branches is a canonical non-smooth (singular) object

The scorer's segmentation verdict is `L⋆(x) = argmax_{k∈{0..4}} ℓ_k(x)`, `K=5` classes `[Road, Lane, Undrivable, Movable, MyCar]` (canonical comma10k order; NON-NEGOTIABLE, do not luma-sort). `d_seg = mean_x 𝟙[L⋆_witness(x) ≠ L⋆_gt(x)]`. Write the **upper envelope**
```
Φ(x) = max_k ℓ_k(x).
```
`Φ` is smooth exactly where a single branch strictly wins; it is **non-smooth precisely on the set where two or more branches tie for the maximum**. This tie-set is the classical object:

- **Maxwell set / conflict set** (singularity theory of families of functions): the locus where the value of the max is achieved by ≥2 branches. This is the honest name for what the paper calls the "decision boundary."
- **PROVEN structure (Thom/Arnold "boundary & corner" / min-type-function theory):** for a *generic* finite family of smooth branches, the Maxwell set is stratified by **how many branches tie**:
  - `codim-0`: 1 branch wins — cell interior. `Φ` smooth. (Argmax stable.)
  - `codim-1`: exactly 2 branches tie — the **wall/facet** (a smooth hypersurface away from junctions).
  - `codim-2`: exactly 3 branches tie — the **triple junction** (an edge; in 2D, isolated points).
  - `codim-≥3`: 4+ tie — higher catastrophes (in 2D image space, non-generic; measure zero and essentially absent).

This is not a metaphor: it is the generic stratification of `x ↦ max_k ℓ_k(x)`, a piecewise-smooth (`PL` if the `ℓ_k` are affine) function whose singular set is a stratified space (CW/Whitney).

### 1.2 The stratification is MEASURED, and it tells us where to spend capacity

Our own probes pin the mass on each stratum (all n600, frozen CPU-torch SegNet, bit-exact to the cached margins):

| Stratum | Geometric object | MEASURED occupancy / error mass | Source anchor |
|---|---|---|---|
| codim-0 (cell interior) | 2-cells (class regions) | argmax stable → "dark" in the Fisher metric; ~0 error | margin-field / Fisher 0.978 |
| **codim-1 (wall/facet)** | the **annulus / separatrix** | **97.7–98.5% of all `d_seg` error is codim-1**; separatrix AUC **0.999**; boundary anisotropy 9.56:1 (gradient) / 37.8:1 (structure-tensor) | `post_muon_application_plan_optimal_form_20260630` |
| codim-2 (triple junction) | 0-cells / saddles | **0.027% of pixels; ~1–2% of flip mass; 53.9% are Road\|Undriv\|Movable car-corners** — NOT the lane tail | 5-logit simplex probe `a4c66f2f` |

**Engineering corollary already banked:** the residual is a **codim-1 phenomenon**. `~400–834` triple junctions/frame exist (dashed-lane markings appear as birth–death saddle pairs, ~2700 events over 600 frames), but they carry ~1–2% of the flip mass and are car-corner-dominated. The junction is a *separate, small* degree of freedom (the "junction-aware eikonal relax" lever, WEAK-BANKED), **not** the lane residual. Capacity is allocated **by codimension**: none to interiors, ~all to the codim-1 annulus, a thin junction-specific correction to codim-2.

### 1.3 The scalar-margin theorem: the codim-1 stratum is *exactly* a binary facet (why the discriminant is simpler than it looks)

A subtle, MEASURED, and load-bearing fact (our probe `a4c66f2f`, canonical eq `scalar_top1_top2_margin_is_exact_distance_to_flip_v1`):
```
gap13(x) := ℓ_top1(x) − ℓ_top3(x)  ≥  gap12(x) := ℓ_top1(x) − ℓ_top2(x)   at ALL 118M pixels (min difference = 0.0).
```
This is trivially true as an inequality (`ℓ_top3 ≤ ℓ_top2`), but its *consequence* is the theorem: **the scalar `m = gap12 = ℓ_top1 − ℓ_top2` is the EXACT distance-to-flip of the argmax.** Only the 2nd-place class can be the first to overtake the winner; the 3rd+ logit can never surface a flip-onset fragility that the scalar margin missed. Therefore:

- **PROVEN (given our measurement):** the codim-1 stratum, restricted to what `d_seg` can feel, is a **binary facet** whose defining function is the single field `m(x)`. The full 5-way discriminant is *observationally equivalent* (for flip onset) to the zero-set of one scalar. The paper's "discriminant/caustic" is, at the level that scores, a **1-parameter margin field**, not a 4-parameter arrangement.
- The triple junction (codim-2, gap13 also `→0`) is the only place the scalar can hide flip *structure* — and we measured it carries ~1–2% of flips. So the "higher catastrophe" content is real but small.

This is why the whole apparatus (curvelet basis, margin-saliency waterfill, along-tangent frequency) targets **the facet** (`m→0` codim-1 set), and the junction lever is dominated. The singularity theory *earns its keep* here: it tells us the residual is a curve (codim-1), the junctions are a measure-zero decoration, and the object to render is the zero-level-set of one field.

### 1.4 The discriminant/caustic vs the Maxwell set — the honest distinction

Two singular sets are being conflated in casual talk and we should separate them:

- **Maxwell set** = where the *value* of the max is tied (2+ branches equal). This is the **hard decision boundary**, the `τ=0` object. It is what `d_seg` scores.
- **Discriminant / caustic** = where the *gradient map degenerates*. At finite `τ`, the natural caustic is where the **Fisher information of the softmax peaks** (§2.4). It is a *fattened, τ-dependent* neighborhood of the Maxwell set — the **annulus/ring**.

**PROVEN bridge (§2.4):** the caustic (Fisher-bright ridge) is the `τ`-smoothing of the Maxwell set, and collapses onto it as `τ→0`. So "discriminant" and "Maxwell set" are the same object at two temperatures — the caustic at `τ>0`, the tropical variety at `τ=0`. Our measured `~1px annulus` is the caustic at the operating `τ`.

---

## §2. THE DUALITY — three ladders, one foot

The claim to prove-or-bound: the `argmax` (tropical/combinatorial, Ch.1) and the `softmax` (Fisher/phase-field, Ch.2/4) are two ends of ONE `Γ`-convergence that is *simultaneously* (a) a mirror-descent/natural-gradient flow, (b) a tropical degeneration, (c) a phase-field sharp-interface limit. Below, each ladder is stated with its real theorem and its exact small parameter; then §2.4 states precisely what is shared and what is not; then §3 gives the verdict.

### 2.1 Ladder A — Maslov / tropical dequantization (small parameter: logit temperature `τ`). **THE SPINE. PROVEN.**

**Theorem (Maslov dequantization; Litvinov 2007; Kolokoltsov–Maslov 1997).** Define the Maslov deformation of `(+,×)` by conjugating with `x ↦ e^{x/ħ}`:
```
u ⊕_ħ v  :=  ħ · ln( e^{u/ħ} + e^{v/ħ} )   →   max(u, v)   as ħ → 0⁺.
```
More generally `ħ · logsumexp_k(ℓ_k/ħ) → max_k ℓ_k`. The limit semiring is the **tropical (max-plus) semiring** `(ℝ∪{−∞}, max, +)`, in which `⊕ = max` is idempotent and `⊗ = +`.

**Identification with our softmax (exact).** Our soft partition uses `p_k(x) = softmax(ℓ(x)/τ)_k` and the smooth surrogate value `Φ_τ(x) = τ·logsumexp_k(ℓ_k(x)/τ)`. Setting `ħ = τ`:
```
Φ_τ(x)  →  Φ_0(x) = max_k ℓ_k(x)   pointwise as τ → 0,   with   0 ≤ Φ_τ − Φ_0 ≤ τ·ln K   (K=5 ⇒ ≤ τ·1.609).
```
The bound `Φ_τ − Φ_0 ∈ [0, τ ln K]` is the exact, uniform dequantization error. **This is our `τ` curriculum** (`τ: 1.0 → 0.05`, cosine), i.e. a *controlled dequantization from the classical (smooth) to the tropical (hard) regime*.

**The hard object is polyhedral-tropical.** If the logit fields were affine, `ℓ_k(x) = a_k·x + b_k`, then:
- `Φ_0 = max_k (a_k·x + b_k)` is a **tropical polynomial** (max-plus linear form); it is convex piecewise-linear.
- Its non-differentiability locus (the Maxwell set) is a **tropical hypersurface** — a polyhedral complex of codim-1 (facets) and codim-2 (vertices = triple junctions), matching §1.1 exactly.
- The argmax cells form a **power (Laguerre) diagram**: site `a_k`, weight `b_k`. This is our banked identity `argmax-of-SDF ≡ power-diagram (per-class offset = Laguerre weight)`.

**Honest caveat (PARTIAL for a deep scorer):** our `ℓ_k(x)` come from EfficientNet-B2, so they are **not affine**. Consequences, all fine because `d_seg` is a boundary phenomenon:
- `Φ_0` is a "tropical *rational*"/non-Archimedean-curved object, only **locally** polyhedral. The power-diagram identity holds in the **tangent (linearized) sense at each boundary point** — the boundary has a well-defined tropical tangent structure everywhere on the smooth codim-1 stratum, which is precisely where flips live.
- Cross-reference (supporting, not load-bearing): arXiv **2601.09775** ("Disclosing the Transformer as a Tropical Polynomial Circuit", 2026) makes the general "deep ReLU/attention net = tropical polynomial" case; for us only the *final K-logit max* needs the tropical view, which is exact.

**Verdict A: PROVEN.** `softmax_τ → argmax` is literally Maslov dequantization; the hard partition is (locally) a tropical hypersurface complement = a power diagram; junctions are its codim-2 vertices. This is the rigorous spine.

### 2.2 Ladder B — Modica–Mortola / phase-field sharp-interface limit (small parameter: spatial width `ε`). **PROVEN, but a DIFFERENT limit.**

**Theorem (Modica–Mortola 1977, conj. De Giorgi; multiphase: Baldo, Fonseca–Tartar, Sternberg).** The scalar Allen–Cahn / Ginzburg–Landau functional
```
E_ε[u] = ∫ ( ε|∇u|² + (1/ε) W(u) ) dx     Γ-converges (as ε→0)  to   c_W · Per(∂{u=1}),
```
i.e. to a constant times the **perimeter** of the interface. The **vector/multiphase** extension (`u: Ω→ℝ^K`, `W` a multi-well potential vanishing on the `K` phase labels) `Γ`-converges to a **weighted perimeter** `Σ_{i<j} σ_ij · H^{n-1}(∂*{phase i} ∩ ∂*{phase j})`, and the minimizers' triple junctions satisfy a **Young/Herring angle law** `σ_ij / sin θ_k = ...` (surface tensions balance).

**Identification with our regularizers (exact in spirit).** Our loss carries `--length-weight 0.001 · ∫_∂ ds` (the **perimeter** term, the `Γ`-limit) and `--eikonal-weight 0.01 · (|∇φ|−1)²` (forces `φ` to be a signed-distance function, i.e. the *optimal phase-field profile* whose transition layer has the Modica–Mortola shape). So the phase-field ladder is **already instantiated** as geometric regularization, and the multiphase triple-junction angle law is a **prediction about how our junctions should meet** (a rendering prior; §4).

**Honest caveat (the crux of "is it ONE theorem?"):** the Modica–Mortola small parameter is the **spatial interface width `ε`** (how many pixels the transition occupies), governed by the render/SDF smoothing and the length/eikonal weights. It is **not** the logit temperature `τ`. They coincide only under a coupling (§2.4): the annulus width `ε` is tied to `τ` through the boundary logit-gradient. So Ladder A and Ladder B **share the limit object** (a sharp K-partition + its perimeter) but are **driven by different knobs**.

**Verdict B: PROVEN as stated, but it is a distinct limit (`ε→0`, spatial), not the same knob as `τ→0`.**

### 2.3 Ladder C — mirror descent = natural-gradient / Fisher–Rao flow (small parameter: step size / time). **PROVEN.**

**Theorem (Amari natural gradient; Raskutti–Mukherjee 2015, "The Information Geometry of Mirror Descent", IEEE T-IT / arXiv:1310.7780).** Mirror descent with the negative-entropy (log-partition of an exponential family) mirror map is **equivalent to natural-gradient descent in the Fisher–Rao geometry**, i.e. it is the steepest-descent flow on the statistical manifold of the softmax family in its dual (natural) coordinates. On the simplex the Fisher–Rao metric is the Shahshahani metric.

**Identification with our training.** The per-pixel class distribution `p(x) = softmax(ℓ(x)/τ)` is an exponential family in natural parameter `ℓ/τ`. Cross-entropy training of the witness logits is (per-pixel) mirror descent with the entropic mirror ⇒ **its continuous-time limit is a Fisher–Rao natural-gradient flow**. The Fisher metric of this family is
```
I_τ(x) = (1/τ²) · ( diag(p) − p p^T )   (in logit coordinates).
```
This is the *dynamics* view of Ch.2. It connects directly to the caustic (§2.4): the flow moves fastest exactly where `I_τ` is large — on the boundary annulus.

**Verdict C: PROVEN.** MD ≡ NGD on the Fisher–Rao manifold of the softmax family; our CE training is (a preconditioned) such flow.

### 2.4 The three-way statement — what is shared, and the NEW bridge that ties Ladder A to Ladders B/C

Here is the technical heart, and the genuinely new (for us) rigorous content.

**The shared foot.** All three ladders have the **same object at the bottom**: the sharp `K`-partition `{L⋆(x)=k}` together with its **weighted perimeter energy** on the codim-1 stratum. Ladder A reaches it by `τ→0` (tropical), Ladder B by `ε→0` (phase-field), Ladder C flows *on the family* toward it. And they all live on the **same one-parameter family** `p_τ = softmax(ℓ/τ)`.

**The bridge (NEW, PROVABLE): the Fisher metric of `softmax_τ` is a caustic sitting on the Maxwell set.** Consider the boundary between the top-2 classes; locally the relevant coordinate is the margin `m(x) = ℓ_top1(x) − ℓ_top2(x)`. The 2-class softmax gives `p_top1 = σ(m/τ)`, and the Fisher information along the margin direction is
```
I_τ(x) = (1/τ²) · p_top1(1 − p_top1) = (1/τ²) · σ(m/τ)(1 − σ(m/τ)).
```
This function of `x`:
- **VANISHES** in cell interiors (`|m| ≫ τ ⇒ σ(1−σ) → 0`) — the "dark" strata (argmax stable, no score leverage);
- **PEAKS** on the Maxwell set (`m = 0 ⇒ σ(1−σ) = 1/4`, scaled `1/(4τ²)`) — a **bright rank-1 ridge = the caustic = the annulus**;
- has **spatial width** `Δx ≈ (transition of σ over m/τ∈[−1,1]) / |∇m| ≈ 2τ / |∇m|`.

So: **the Fisher-information field is literally an optical caustic — a bright fold of the metric — that lies on the margin-zero set and whose width scales as `τ/|∇m|`.** As `τ→0` it collapses (as a `1/τ`-scaled bump, integrating to a surface measure) onto the tropical hypersurface. **This is the rigorous identity between the caustic (Fisher/phase-field, Ch.2/4) and the tropical variety (Ch.1): they are the same set at two temperatures.**

**Our MEASURED 0.978 is exactly this bridge — and it is near-tautological (which is why it is trustworthy).** We measured `Fisher-curvature ↔ (−margin)` Pearson **0.978** (n600). The formula above shows `I_τ` is a *deterministic monotone function of `−|m|`* (`σ(m/τ)(1−σ(m/τ))` is maximal at `m=0`, decreasing in `|m|`). So the 0.978 is not a lucky empirical correlation; it is the **2-logit softmax formula read off the data**, confirming the caustic sits on the margin-zero set with the predicted profile. This has a hard NO-FAKE consequence:
- **KEEP:** "the margin field is the frozen-scorer Fisher surrogate; the annulus IS the caustic; capacity leverage lives there." (PROVEN, tautological.)
- **DROP:** "margin = UNIWARD steganographic cost." We measured the *pixelwise* margin↔UNIWARD-cost correlation at ≈ near-zero (n6, confirmed 2×); UNIWARD is an RGB-texture cost, the margin is a logit object. The unity, if any, is **metric-level (Fisher / Jacobian), not a scalar-map identity** — do not use raw UNIWARD as a margin-saliency proxy.

**So what IS the three-way "equivalence," stated honestly?**
> The one-parameter family `p_τ = softmax(ℓ/τ)` is *simultaneously* (A) a **Maslov/tropical deformation** in `τ` (semiring dequantization: `τ→0` gives the hard tropical partition), (C) equipped with a **Fisher–Rao metric** under which CE training is **natural-gradient flow**, and whose **Fisher caustic is the `τ`-smoothed Maxwell set**; and its `τ→0` limit partition is *also* the (B) **`Γ`-limit minimizer of the phase-field perimeter energy** when the spatial length/eikonal regularizer is present. The **object** (sharp partition + perimeter + the annulus-as-caustic) is one; the **three ladders reaching it are three different limits** (`τ`, step-size, `ε`).

That is a true, non-trivial, and for the caustic-bridge *newly-articulated* unification. It is **not** the stronger claim that a single scalar limit simultaneously drives all three — and the paper must not assert that.

---

## §3. THE HONEST VERDICT — proven / partial / conjecture (the load-bearing NO-FAKE call)

| # | Claim | Status | Basis / caveat |
|---|---|---|---|
| 1 | `softmax_τ / logsumexp_τ → argmax/max` as `τ→0` is **Maslov dequantization** (tropical/max-plus limit). | **PROVEN (textbook)** | Litvinov 2007; Kolokoltsov–Maslov 1997. Exact error bound `Φ_τ−Φ_0∈[0,τ ln K]`. Our `τ` curriculum IS this. |
| 2 | The hard argmax partition is a **tropical hypersurface complement = power/Laguerre diagram**; boundary = codim-1 tropical variety; triple junctions = codim-2 vertices. | **PROVEN for affine logits; PARTIAL (local) for the deep scorer** | Exact if `ℓ_k` affine. For EfficientNet logits it holds in the **tangent/linearized** sense on the smooth codim-1 stratum — which is where all flips live. |
| 3 | The argmax partition is a **stratified (Whitney/CW) space** stratified by number-of-tied-branches (Maxwell-set stratification); residual is codim-1. | **PROVEN (generic singularity theory) + MEASURED** | 97.7–98.5% error codim-1; junctions 0.027% px / ~1–2% flips (probe `a4c66f2f`). |
| 4 | `m = top1−top2` is the **exact scalar distance-to-flip**; the codim-1 stratum is observationally a binary facet. | **PROVEN given our measurement** | `gap13 ≥ gap12` at all 118M px (min diff 0.0); eq `scalar_top1_top2_margin_is_exact_distance_to_flip_v1`. |
| 5 | Phase-field (Modica–Mortola / vector Allen–Cahn) `Γ`-converges to **weighted perimeter** with Young/Herring junction angles. | **PROVEN (textbook)** | Modica–Mortola 1977; Baldo/Fonseca–Tartar/Sternberg. Our `length`/`eikonal` regularizers instantiate it. **DIFFERENT small parameter (`ε`, spatial).** |
| 6 | Mirror descent (entropic mirror) ≡ **natural-gradient / Fisher–Rao flow**; our CE training is such a flow. | **PROVEN (textbook)** | Amari; Raskutti–Mukherjee 2015 (arXiv:1310.7780). |
| 7 | The **Fisher field is a caustic on the Maxwell set**, width `~τ/|∇m|`, collapsing to the tropical variety as `τ→0`; this unifies Ch.1 ↔ Ch.2/4. | **PROVEN (new articulation for us) + MEASURED** | 2-logit softmax formula `I_τ = (1/τ²)σ(m/τ)(1−σ(m/τ))`; our `Fisher↔(−margin)` 0.978 is this, near-tautologically. Annulus width `~1px` = caustic at operating `τ`. |
| 8 | The three ladders (τ-tropical / ε-phase-field / step-Fisher) share **one limit object and one family**. | **PROVEN (they share the foot)** | The sharp partition + perimeter + softmax_τ family are common to all three. |
| 9 | The three ladders are **the same single limit / one theorem**. | **FALSE — do not claim** | Three distinct small parameters (`τ`, `ε`, step). A `l7` leg adds a **fourth** (`p→∞`, `Lᵖ→L^∞`). Muon is a preconditioner, outside the continuum limit. |
| 10 | Our specific `CE→tau→l7→Muon` trajectory **is** discretized natural-gradient along the Maslov deformation ending at the `Γ`-limit partition. | **CONJECTURE / heuristic bridge** | Well-motivated, matches measured stage behavior (FEED-gi: CE→tau realized `d_seg −21.6%`, junctions `−14%`, code eff-rank 25.8→21.9), but not one theorem; `l7`/Muon break the clean continuum story. |
| 11 | margin = UNIWARD steganographic cost. | **FALSE (measured near-zero) — do not claim** | Unity is metric-level (Fisher/Jacobian), not a scalar-map identity. |

**The single most important honest sentence for the paper:** *the hard↔soft duality is a rigorous statement that softmax→argmax is Maslov dequantization and that the Fisher caustic is the τ-smoothing of the tropical Maxwell set (both PROVEN and both matching our measured 0.978 and 1px annulus); it is NOT a claim that one limit simultaneously realizes the tropical, phase-field, and optimization limits — those are three limits sharing one object.*

---

## §4. Engineering nexus — what the singularity/duality lens actually buys us (honest EV)

The lens is MEANS; the pointer moves only via a byte-closed `evaluate.py` n600 row through the real decode. Ranked by expected `Δ(score-term)/byte realized-through-R`:

1. **Stratified (per-stratum) capacity allocation — HIGH EV, largely already banked.** Allocate representation capacity **by codimension**: zero to codim-0 interiors (Fisher-dark), essentially all to the codim-1 annulus (the caustic — KKT margin-saliency waterfill on `m`), a thin junction-specific correction to codim-2. This is the rigorous justification for the directional/curvelet basis being *prior to* capacity (basis on the tangent field of the codim-1 stratum) and for the annulus being "the whole game" (97.7–98.5%). *EV: already the design; the lens confirms it and forbids spending on interiors.*

2. **Maslov view of the `τ` curriculum + a principled `τ`-floor — MODEST-but-real EV.** The curriculum `τ:1.0→0.05` is a *controlled dequantization*: track the smooth (softmax) minimizer continuously as you cool toward the hard (tropical) target, avoiding direct optimization of the non-smooth `argmax` (numerical continuation / homotopy along the Maslov deformation). Two concrete, testable outputs:
   - **`τ`-schedule = Fisher–Rao geodesic** (Ladder C): the *natural* cooling rate is the one that keeps steps of equal Fisher length; a geodesic-`τ` schedule should beat an ad-hoc cosine. *EV: cheap Tier-2 A/B off a per-stage checkpoint; owed to #205, not a step-change.*
   - **`τ`-floor set by the caustic width vs R-noise:** from §2.4 the annulus width is `~τ/|∇m|`. Cooling past the point where the caustic is thinner than R's uint8 quantization / pixel scale gives no scored benefit and courts instability — it over-sharpens below the noise floor. This **derives a `τ`-floor** from the measured boundary gradient and the R-noise scale, and it **unifies with the "keep boundary/minority margin > R-noise" rule**. *EV: turns the `0.05` floor from a guess into a measured quantity; free.*

3. **Catastrophe-aware / junction-aware rendering — LOW-MED EV (correctly dominated).** At codim-2 junctions the SDF cannot be smooth (medial-axis singular set), so `|∇φ|=1` is the wrong constraint there → **junction-aware eikonal relax** (down-weight the eikonal penalty near detected triple points). Additionally, Modica–Mortola predicts junctions meet at **Young/Herring angles** set by per-class-pair surface tensions — a *rendering prior* for how three boundaries should meet. *EV: honestly LOW — junctions are 0.027% px / ~1–2% flips and car-corner-dominated; this is a WEAK-BANKED refinement, not a lane-residual fix. Do not oversell.*

4. **Along-tangent (anisotropic/curvelet) basis on the codim-1 stratum — HIGH EV (the decisive banked lever, elsewhere).** The lens says the object to represent is the **zero-level-set of one scalar `m`** (a curve, codim-1), whose optimal sparse basis is curved/parabolic-scaling (curvelet). This chapter *grounds* why (the residual is a codim-1 singular curve), the measured `−48%` directional lever is booked in the θ* stack, not here.

5. **Non-lever, but paper-level:** the tropical/power-diagram identity makes the **rate** story crisp — a (locally) polyhedral partition is described by its sites+weights (the ~8-dim lane-orbit sufficient statistic), i.e. the counted payload is the *combinatorial* tropical data, and the generic tropical-expansion is free in inflate.py (rule 118). This is the Ch.1↔rate connection; it belongs to the rate chapter but the singularity lens supplies its justification.

---

## §5. Triality sync + references + closer

**Triality (all three legs agree):**
- **DAG:** simplex probe `a4c66f2f` (FEED-03t), CE→tau `FEED-gi`, annulus/Morse-Smale `post_muon_application_plan_optimal_form_20260630`. All consistent with §1–§2.
- **equations:** `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` (§1.3) + the tau homotopy / eikonal / length regularizers (§2.2) + the Fisher-caustic formula `I_τ=(1/τ²)σ(m/τ)(1−σ(m/τ))` (§2.4, candidate for a new `EmpiricalAnchor` with residual = pred-vs-measured 0.978). No drift: every claim here is already a measured DAG row or a textbook theorem.
- **this memo:** the WHY (singularity + Maslov + Fisher-caustic) behind those rows.

**Canonical references (real, checked):**
- Maslov dequantization / tropical limit: G. L. Litvinov, "The Maslov dequantization, idempotent and tropical mathematics: a brief introduction," *J. Math. Sci.* (2007); V. Kolokoltsov & V. Maslov, *Idempotent Analysis and Its Applications* (1997). Supporting/modern: arXiv **2601.09775** "Disclosing the Transformer as a Tropical Polynomial Circuit" (2026).
- Modica–Mortola / multiphase `Γ`-convergence: Modica–Mortola 1977 (conj. De Giorgi); vector case Baldo, Fonseca–Tartar, Sternberg; survey G. Alberti, "Variational models for phase transitions: an approach via `Γ`-convergence."
- Mirror descent = natural gradient / Fisher–Rao: S. Amari (natural gradient); Raskutti & Mukherjee, "The Information Geometry of Mirror Descent," IEEE T-IT 2015 / arXiv:**1310.7780**.
- Singularity/catastrophe theory of `max`-functions / Maxwell sets: Thom; Arnold, *Catastrophe Theory* / boundary singularities; min/max-type functions.

**Closer (means, not ends).** This chapter is the keystone because it makes the paper's central word — "duality" — *precise and honest*: two textbook limits (`τ→0` Maslov, `ε→0` Modica–Mortola) whose object coincides, a training flow (natural-gradient) on the connecting family, and one new-for-us rigorous bridge (the Fisher caustic = the `τ`-smoothed tropical Maxwell set, verified by our own 0.978 and 1px annulus). It does **not** claim a single unifying limit, and it explicitly retires the margin=UNIWARD overclaim. It moves **no exact pointer** (0.19110 UNMOVED); its engineering payload is the per-stratum allocation law, the Maslov-derived `τ`-floor, and the (correctly dominated) junction lever — each owed to #205 as a measured through-R A/B, never asserted.
