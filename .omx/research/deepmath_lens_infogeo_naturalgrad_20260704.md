# Deep-math lens — Chapter 2: Information Geometry, Fisher–Rao, Natural Gradient, Mirror Descent, Bregman/Dually-Flat

**Paper:** *Amortizing the Argmax: the Separatrix as a Unified Differential-Geometric Object* · **Chapter 2 (the "soft view").**
**UTC:** 2026-07-04T055912Z · **git:** `ab8fd252d` · **task #284** · **author lens:** Amari · Nielsen · Raskutti–Mukherjee · Bernstein–Newhouse.
**Tags:** `[deep-math research memo / research-signal]` · `score_claim=false` · `promotable=false` · `ready_for_exact_eval_dispatch=false`.
**NO GPU · $0 (online + OSS + our own measured cache) · MEANS, not the END.** Pointer **0.19110 UNMOVED** — a memo moves no pointer. #205 read-only (untouched). NO-FAKE: every number below is either our measured cache (cited to artifact) or a literature result (cited to arXiv); conjectures are labelled; the commissioning prompt's "198:1" is flagged as **unsourced** and replaced with the measured value.

---

## 0. One-paragraph thesis (the soft half of the hard↔soft duality)

The frozen SegNet head is a **categorical exponential family**: its softmax is the maximum-entropy exponential-family model with the logits as natural parameters. Everything Chapter 1 said sharply (tropical / argmax / power-diagram) has a **smooth dual** here: the argmax boundary Σ (the separatrix) is where the exponential family's **Fisher information** concentrates; our measured **margin↔Fisher Pearson 0.978** is that Fisher metric *read through the top-1−top-2 logit gap*; the measured **9.56:1 boundary anisotropy** is the metric's eigenstructure certifying Σ is codim-1; **Muon's −32% d_seg win** is a **weight-space** metric-aware (spectral / approximate-natural-gradient) step; and the whole **CE→τ→Muon curriculum is a mirror-descent path** on the dually-flat categorical manifold whose **τ→0 limit is the Γ-convergence bridge** to the sharp phase-field flow of Chapters 3–4. The soft and sharp separatrix are the two ends of **one mirror-descent homotopy**.

---

## 1. The Fisher metric on the frozen-scorer softmax — formalized

### 1.1 The categorical exponential family and its Fisher information (PROVEN, exact)

Per pixel, the SegNet head emits logits `z ∈ ℝ^K` (K=5) and a distribution `p = softmax(z)`. This is the categorical/multinomial exponential family with **natural parameter** `z` (identified up to the `∑`-gauge `z → z + c·1`, which softmax quotients out). The Fisher information matrix **in the natural (logit) coordinates** is the classical

```
    F(z) = Cov_p[one-hot(y)] = diag(p) − p pᵀ          (K×K, rank K−1, null space = 1)
```

Its trace is exactly the quantity our co-location test measured and called "Fisher curvature":

```
    tr F(z) = ∑_k p_k(1 − p_k) = 1 − ∑_k p_k²  =  1 − ‖p‖²   ( = the Gini / collision-entropy gap )
```

**This is not a proxy or an analogy — it is the identity.** `tools/colocation_fisher_stress_anisotropy_test.py` recomputed the frozen SegNet logits (argmax mismatch vs the cached authority = **0.0**; margin |Δ|max = **4.8e-7**, float32 ULP) and evaluated `1 − ∑ p_k²` — i.e. the exact trace of the categorical Fisher matrix in natural coordinates. It also measured `corr(tr F, ‖F‖₂) = 0.997`: the trace tracks the spectral norm because near a two-class boundary the matrix is effectively rank-1 (see §1.3), so trace ≈ top eigenvalue. **Verdict: PROVEN.** (Artifact: `experiments/results/colocation_test_20260629T160343Z/colocation_results.json`.)

### 1.2 Why margin ↔ Fisher ≈ 1 — the logit-collinearity derivation (PROVEN)

On the annulus the two competing classes dominate; write the **margin** `m = z_(1) − z_(2) ≥ 0` (top-1 minus top-2). In the two-class reduction `p_(1) = σ(m)`, `p_(2) = 1 − σ(m)`, all other `p_k ≈ 0`, so

```
    tr F  =  ∑_k p_k(1−p_k)  ≈  2 σ(m) (1 − σ(m))  =  ½ sech²(m/2)
```

a **deterministic, monotone-decreasing** function of the margin (peak `½` at `m=0`; decays like `2e^{−m}` for large `m`). Therefore `tr F` and `(−margin)` are functionally **collinear through a fixed sigmoidal link** — both are pushforwards of the *same* logit gap. This is why:

- Pearson(`tr F`, `−margin`) = **0.978** in the small-margin band (the band restricts to where `½ sech²(m/2)` is near-linear in `m`);
- Spearman(`tr F`, `−margin`) = **0.908** over all pixels (the monotone link, globally);
- the residual from perfect collinearity is exactly (a) the sigmoid nonlinearity over a wide margin range and (b) K>2 corrections when three+ classes compete (tri-junctions).

**Consequence for the codec (already exploited):** the cheap top-1−top-2 **margin field is a byte-faithful surrogate for the full Fisher metric** — the v2 loss weights the annulus by `margin` and *never has to carry the full K-logit tensor*. This is `_live_margin_weight` in the trainer.

### 1.3 NO-FAKE correction — what the 0.978 is and is NOT

The MEMORY shorthand "margin = Fisher = UNIWARD (0.978)" invites a pixelwise misread. The truth (measured twice, `uniward_margin_pixelwise_nearzero_...20260703`):

- **0.978 is Fisher ↔ (−margin)** — collinear *by construction* (both are functions of the same logits).
- **Direct pixelwise margin ↔ UNIWARD-cost Pearson ≈ +0.04** (n6; agent's independent measure +0.012) — **near-zero**. UNIWARD cost is an RGB wavelet-residual texture map; it is unrelated to the logits pixelwise. The steg↔detector "unity" is **metric-level / directional** (the Fisher metric = the SegNet Jacobian's sensitive directions = the steganalyst's low-cost directions), **NOT** a scalar-map correlation. **Do not** wire the raw UNIWARD cost map in as a drop-in margin-saliency proxy.

### 1.4 The anisotropy is the metric's eigenstructure — and "198:1" is unsourced

Pull the output-space Fisher back to image/witness coordinates through the SegNet Jacobian `J`: `G = Jᵀ F J`. Along the boundary tangent the argmax is ~constant (`∂/∂t` small); across the normal it flips (`∂/∂n` large). So `G` at Σ is dominated by **one eigenvalue** (the normal), i.e. Σ is a **rank-≈1 ridge = codim-1** — the differential-geometric certificate that the separatrix is a hypersurface. Measured on the margin field's structure tensor:

| Quantity | Measured | Geometric meaning |
|---|---|---|
| grad-proj ratio **across:along** | **9.56 : 1** | direct analog of the deep-math ≈7:1; normal-vs-tangent sensitivity |
| structure-tensor eigenvalue **λ₁/λ₂** (aggregate) | **37.8 : 1** | overshoots because λ₂→0 at straight boundaries → even *thinner* 1D ridge |
| λ₁/λ₂ lane-class boundary only | 28.5 : 1 | same, lane-specific |

> **HONEST FLAG (NO-FAKE).** The commissioning brief cites "198:1 annulus anisotropy." I could not source that number to any measurement in our cache (`grep` of the colocation JSON and all research/memory finds **no 198**). The measured anisotropy is **9.56:1 (gradient projection)** and **37.8:1 (structure-tensor eigenvalue)**. I ground Chapter 2 in those. A "198:1"-scale figure would only appear if one reports the *condition number of the full pixel-space pullback* `G` including its large **null space** (`κ(G) → ∞` formally, since `G` is rank-deficient) — a different, geometry-degenerate quantity that should not be quoted as "the anisotropy." Use 9.56:1 / 37.8:1.

**Design corollary (validated):** the v2 distortion energy should be **annulus-localized** (margin-weighted = Fisher-weighted), **anisotropic** (~9–10:1 cross:along = a Finsler/tangent-oriented term), and **natural-gradient** (preconditioned by the metric). This is *the same object* as the measured "ALL-CLASS DIRECTIONAL Fourier/curvelet basis = −48% d_seg, ~0 byte" lever — the anisotropic basis and the Fisher-natural loss are two reads of one metric.

---

## 2. Is the categorical Fisher–Rao distance a better seg loss than CE / τ-softplus?

### 2.1 The closed form (PROVEN geometry; Nielsen 2403.10089)

The categorical simplex Δ^{K−1} with the Fisher metric is **isometric to the positive orthant of a sphere** under the square-root embedding `p ↦ (√p_1,…,√p_K)`; the Fisher metric is `¼` the round metric, so the **Fisher–Rao geodesic distance has a closed form**:

```
    d_FR(p, q) = 2 · arccos( ∑_k √(p_k q_k) )        (= 2× the Bhattacharyya angle; BC = ∑√(p_k q_k))
```

Unlike CE/KL, `d_FR` is a **true metric** (symmetric, triangle inequality) and is **bounded** (`∈ [0, π]`). Nielsen (2403.10089, *Approximation and bounding techniques for Fisher–Rao distances*) treats exactly this multinomial/Bhattacharyya case and its bounds — it is the canonical reference for using `d_FR` / Bhattacharyya as a training-time distance.

### 2.2 Honest EV vs CE and τ (CONJECTURED — clean $0 A/B candidate)

- **CE = KL = the Bregman divergence of the entropy potential** is already the "canonical / natural" loss of this exponential family (§4), and softmax+CE already yields the natural-gradient-simplified `(p − y)` update. So CE is not a naive baseline to beat — it is the info-geometrically preferred *divergence*.
- The genuine difference: `d_FR` is **bounded and symmetric**; CE **blows up** on confident-wrong pixels (`−log p_true → ∞`). On the shallow-margin annulus and under R's uint8 noise, a confidently-wrong flip contributes an unbounded CE gradient that can destabilize neighbouring correct pixels; `d_FR`'s bounded arccos may give a **gentler, better-conditioned** boundary gradient — a plausible d_seg win *specifically on the flip annulus*.
- But our actual objective is `d_seg = argmax-mismatch count` (a 0–1 count), for which CE, τ-softplus, and `d_FR` are **all surrogates**. The tuned τ-softplus already sharpens the softmax toward argmax. Whether `d_FR` beats a *tuned* τ+margin stack is genuinely unknown.
- **Verdict: CONJECTURED, MEDIUM EV.** A clean, cheap through-R n600 A/B (swap the CE term for `2·arccos(BC)` against the soft GT logits, keep everything else) is the honest test. It is **not** obviously dominant; it is a well-posed candidate.

---

## 3. Muon = natural gradient / mirror descent? — formalize the equivalence and the gap

### 3.1 What Muon actually is (Bernstein–Newhouse 2024; Jordan)

Muon = momentum + **orthogonalize the update matrix** (Newton–Schulz → all singular values → 1) = **steepest descent under the spectral (Schatten-∞) norm** on each weight matrix (Bernstein & Newhouse, *Old Optimizer, New Norm* / *Modular Duality in Deep Learning* 2410.21265; Jordan's Muon writeup). The "norm you pick = the optimizer" framework: Adam ≈ steepest descent under a max/RMS norm; Muon ≈ under the operator norm induced when inputs/outputs are RMS-normed.

### 3.2 The equivalence and the gap (FALSE FRIEND at the literal level; genuine cousin)

- **Literal "Muon = the categorical natural gradient" is a FALSE FRIEND.** The categorical natural gradient preconditions by `F(z)^{-1} = (diag p − ppᵀ)^{-1}` in **output/logit space** (a per-pixel K×K object). Muon preconditions by the **spectral geometry of the weight matrices** — a *different manifold* (weight space, not the output statistical manifold). They are not the same metric.
- **But Muon is a genuine metric-aware / weight-space approximate-natural-gradient step.** Spectral-norm steepest descent = one-step Shampoo (Bernstein), and Shampoo is a Kronecker-factored **Gauss–Newton / Fisher** approximation. So Muon is an *approximate-natural-gradient in weight space* — the honest statement is "metric-aware, spectral, second-order-flavoured," not "the softmax Fisher–Rao natural gradient."

### 3.3 Does this explain the measured −32%? (PARTIAL / CONJECTURED attribution)

Measured: Muon descended d_seg **~32% more than AdamW** from an identical stage-4 fork, gap widening (`muon_vs_adamw_from_stage4_convergence_arm_20260622`). The consistent mechanism: the boundary Hessian is **ill-conditioned** (κ≈19; the anisotropic §1.4 annulus makes correlated, elongated curvature); AdamW's **diagonal** preconditioner cannot decorrelate it; Muon's polar/spectral flattening is a **κ-buster** (`O(ln 1/ε)` vs `O(κ ln 1/ε)`).

- The κ-busting mechanism is real and the −32% is a real measurement. **But attributing the −32% to "natural-gradient-ness" is CONJECTURAL** — it is *equally* explained by plain spectral conditioning of an anisotropic weight-space Hessian, with **no** appeal to the output-space Fisher. Info geometry gives the *language* (metric-aware step) and a *why* (anisotropy → conditioning), not an independent proof.
- **Sharper version?** The Muon deep-dive (`muon_deep_dive_..._20260703`) already localizes the real EV: Muon **cannot self-reduce its step near a minimum** (NS normalizes magnitude), so the info-geo-correct move is to **anneal Muon's LR** (constant-Fisher-arc-length step → decaying Euclidean step near the basin) and **warm-start its momentum** from the incoming AdamW gradient-EMA (avoid the wild first orthogonalized step). **MD-Decoupling** (`--optimizer md`, in-tree) regulates the *relative* update magnitude (anti-collapse) — a complementary magnitude-drift fix, SPECULATIVE at our 60–230K scale. None of these switch the optimizer; they schedule the metric-aware one we already measured to win.
- **Verdict: Muon = metric-aware weight-space step (PROVEN framing); Muon = categorical natural gradient (FALSE FRIEND); −32% ← natural-gradient-ness (CONJECTURED, equally explained by κ-busting).** Keep Muon; sharpen its *schedule*.

---

## 4. Dually-flat / Bregman structure — is the curriculum a mirror-descent / geodesic path?

### 4.1 CE is literally mirror descent on the categorical simplex (PROVEN)

The categorical family is **dually flat**: the natural (θ = logits) and expectation (η = probabilities) coordinates are Legendre-dual through the log-partition potential `A(z)=log∑e^{z_k}` and its conjugate (negative entropy) `A*(p)=∑p_k log p_k`. The **Bregman divergence** of `A*` is the **KL divergence = the CE loss**. Softmax is exactly the **mirror map** `∇A` linking θ↔η. Therefore:

> **Training with the CE loss and softmax IS mirror descent with the (negative-)entropy mirror map on the categorical simplex.** (Exact, textbook — Amari; Nielsen; Beck–Teboulle.)

### 4.2 The rigorous backbone (Raskutti–Mukherjee 2015, PROVEN)

*The Information Geometry of Mirror Descent* (arXiv 1310.7780, IEEE T-IT): **mirror descent induced by a Bregman divergence ≡ natural-gradient descent on the Riemannian manifold in the dual coordinate system.** For the exponential family this is the steepest descent along the Fisher–Rao manifold; MD with log-likelihood loss asymptotically attains the Cramér–Rao bound. **This is the theorem that makes "the curriculum is mirror descent = natural gradient" not a metaphor.**

### 4.3 The curriculum as a path (PARTLY PROVEN / PARTLY CONJECTURED)

- **CE stage:** mirror descent in the natural (θ) affine chart — **proven** (§4.1).
- **τ-softplus anneal:** scaling logits by `1/τ` **deforms the mirror map / the metric** (temperature = inverse dual-flatness scale). As `τ↓`, softmax→argmax, the entropy barrier `A*` steepens, the Fisher metric `diag(p)−ppᵀ` degenerates onto the boundary — the path moves toward the **sharp** (Chapter 1/3) regime. Interpreting the anneal as a **reparametrization of the same dually-flat geometry** is principled; that the resulting trajectory is a **Fisher–Rao geodesic** is **CONJECTURED** (it is a *homotopy* of metrics, not obviously a geodesic of a single one).
- **Muon finisher:** a **weight-space** natural-gradient polish (§3), a different manifold — it "finishes" the output-space mirror path by second-order-conditioning the *parameters*.
- **Net verdict:** "CE = mirror descent" PROVEN; "curriculum traces a geodesic" CONJECTURED; the *useful* consequence is the **derived τ(t) schedule** of §6.

---

## 5. THE ONE CLAIM for the duality theorem (Chapter 3) — the mirror-descent ↔ Γ-convergence bridge

This is Chapter 2's single contribution to the paper's central duality theorem. State it as a two-part claim, proven half + conjectured half, so it is NO-FAKE:

> **Claim 2.★ (the soft→sharp bridge).** Consider the temperature-`τ` softmax **free energy** per pixel, `F_τ[φ] = ⟨distortion of the soft partition softmax(φ/τ)⟩ + τ·H(softmax(φ/τ))` (the CE/entropy-regularized objective the CE→τ curriculum minimizes). Then:
> 1. **(PROVEN, Raskutti–Mukherjee + §4.1)** For each fixed `τ`, gradient descent on `F_τ` via softmax+CE is **mirror descent = natural-gradient flow of the distortion in the Fisher–Rao geometry** of the categorical family. The soft separatrix (`softmax` level set `p_(1)=p_(2)`) evolves by this Fisher–Rao gradient flow.
> 2. **(CONJECTURED — the Chapter-3 hand-off)** As `τ→0`, the entropy term `τ·H → 0` and `F_τ` **Γ-converges** to a **sharp-interface perimeter functional** on the argmax partition (the Modica–Mortola / Allen–Cahn mechanism with the entropy barrier playing the double-well role, K-class). The Fisher–Rao mirror-descent flow of part 1 then **converges to the phase-field / mean-curvature (perimeter-minimizing) flow of the Morse–Smale separatrix** — the sharp object of Chapters 3–4.

**Therefore the soft (information-geometric, mirror-descent, Fisher–Rao) view of the separatrix and the sharp (Morse–Smale / perimeter / tropical) view are the two ends of ONE `τ`-annealed mirror-descent homotopy.** The CE→τ curriculum is not an engineering schedule bolted onto a sharp target; it **is** the Γ-convergence path realizing the duality. The proven half (MD ≡ natural gradient) and the exact endpoints (softmax free energy at `τ=1`; argmax perimeter at `τ=0`) are solid; the Γ-limit is the theorem Chapter 3 must close. Our **measured** anchors that the bridge must reproduce: Fisher↔margin 0.978, anisotropy 9.56:1, flip-mass 96.8% in a 2px band, τ-anneal empirically sharpening d_seg.

---

## 6. Engineering nexus — candidate levers, ranked by EV toward a lower exact n600 d_seg through R

Unit throughout = **Δ(d_seg-term)/byte realized-through-R on the n600 numpy-fp32 authority** (the calibration common unit). All are MEANS; none moves the pointer until byte-closed via #202.

| # | Lever | What info-geo says | EV verdict | Test (all $0 unless noted) |
|---|---|---|---|---|
| **1** | **Anneal Muon's finishing-stage LR + warm-start its momentum** | Muon = metric-aware step that **cannot self-reduce** near the basin; natural-gradient wants constant *Fisher-arc-length* step ⇒ decaying Euclidean LR. Warm-start avoids the wild first orthogonalized step. | **HIGHEST** — sharpens our *measured* −32% winner; not a new optimizer. | T2 warm-started off the per-stage Muon ckpt; cosine/linear Muon-LR decay + momentum seed. Already the Muon deep-dive's #1/#2. |
| **2** | **Fisher–Rao-arc-length-uniform τ(t) schedule** | The optimal anneal keeps the natural-gradient step constant in **Fisher–Rao arc length** (constant "information velocity") ⇒ a *derived* τ(t) replacing the hand-tuned ramp; critical-slowing near stage transitions is the Fisher-degeneration signature. | **MEDIUM-HIGH** — principled T0-derivable schedule; cheap to test. | T0 derive τ(t) from `∫√(g_FR) dτ = const`; T2 A/B vs current ramp off per-stage ckpts. |
| **3** | **Fisher–Rao / Bhattacharyya-angle seg loss** `2·arccos(∑√(p_k q_k))` vs CE/τ | A **bounded, symmetric metric** loss; gentler boundary gradient than CE's unbounded `−log p` on confident-wrong flips under R-noise. Nielsen 2403.10089 = the closed form + bounds. | **MEDIUM** — clean candidate; NOT obviously > tuned τ+margin (CE is already the canonical Bregman loss). | T1/T2 through-R n600 A/B: swap CE term for `d_FR` to soft GT, hold else fixed. |
| **4** | **Margin field = Fisher volume element (already live)** | `√det G` on the annulus ∝ margin sharpness ⇒ the `_live_margin_weight` allocator **is** the Fisher-natural per-pixel weight (Pearson 0.978). | **FRAMING/CONFIRMATORY** — no new build; validates the existing allocator as info-geo-optimal + licenses adding the **anisotropy** (Finsler ~9–10:1 cross:along) term on top. | already measured; the anisotropy term = the −48% directional-basis lever. |
| **5** | **Output-space natural-gradient bolt-on** (precondition by `F^{-1}`) | The softmax+CE gradient `(p−y)` is **already** the natural-gradient-simplified update ⇒ an explicit output-space `F^{-1}` preconditioner is largely **redundant** and costs a per-pixel K×K solve. | **LOW / DON'T BUILD** (honest negative — saves time). | none — reasoned out. |

---

## 7. Honest verdict ledger (proven / conjectured / false-friend / framing)

| Idea | Verdict | Ground |
|---|---|---|
| Frozen softmax = categorical exp-family; measured "Fisher curvature" `= 1−∑p²` = **trace of the categorical Fisher matrix in logit coords** | **PROVEN (exact identity)** | colocation JSON; textbook |
| **margin ↔ Fisher 0.978** = logit-collinearity (`tr F = ½ sech²(m/2)`, monotone in margin) | **PROVEN** | derivation §1.2 + measured 0.978/0.908 |
| margin ↔ UNIWARD pixelwise (the mis-shorthand) | **FALSE FRIEND** (≈+0.04, near-zero; unity is metric-level) | measured twice, n6 |
| **9.56:1 / 37.8:1 anisotropy** = metric eigenstructure certifying Σ codim-1 | **PROVEN (measured) + framing** | colocation JSON |
| "**198:1** anisotropy" (from the brief) | **UNSOURCED — do not quote**; measured is 9.56:1 / 37.8:1 | grep of all cache = no 198 |
| Muon = the **categorical natural gradient** | **FALSE FRIEND** (weight-space spectral ≠ output-space Fisher–Rao) | Bernstein–Newhouse; §3.2 |
| Muon = **metric-aware / approximate-natural-gradient in weight space** | **PROVEN framing** | Bernstein (Muon≈1-step Shampoo≈Gauss-Newton) |
| −32% d_seg ← "natural-gradient-ness" | **CONJECTURED** (equally = plain κ-busting) | measured −32%; attribution open |
| **CE + softmax = mirror descent** on the categorical simplex | **PROVEN (exact)** | Bregman/Legendre; §4.1 |
| **MD ≡ natural gradient in dual coords** | **PROVEN (theorem)** | Raskutti–Mukherjee 2015 |
| curriculum CE→τ→Muon traces a **geodesic** | **CONJECTURED** (it's a metric homotopy; the useful part is the derived τ(t)) | §4.3 |
| **τ→0 = Γ-convergence softmax-free-energy → perimeter** (the bridge) | **CONJECTURED** (Ch.3's job; proven half = MD≡NG + exact endpoints) | §5 |
| Fisher–Rao/Bhattacharyya loss > CE for d_seg | **CONJECTURED** (needs through-R A/B) | §2.2; Nielsen 2403.10089 |

---

## 8. Key papers cited

- **Amari**, *Information Geometry and Its Applications* (2016) — natural gradient, dually-flat exp-families, Fisher metric.
- **Nielsen, F.** — *Approximation and bounding techniques for the Fisher–Rao distances between parametric statistical models*, arXiv **2403.10089** (multinomial / Bhattacharyya closed form + bounds). Also *pyBregMan* (2408.04175) for Bregman-manifold tooling.
- **Raskutti, G. & Mukherjee, S.** — *The Information Geometry of Mirror Descent*, arXiv **1310.7780**, IEEE T-IT 2015 (MD ≡ natural gradient in dual coords — the backbone of §4–5).
- **Bernstein, J. & Newhouse, L.** — *Old Optimizer, New Norm is All You Need* + *Modular Duality in Deep Learning*, arXiv **2410.21265**; **Jordan, K.**, *Muon: An optimizer for hidden layers* (blog). Muon = spectral-norm steepest descent.
- Supporting: **Beck & Teboulle** (mirror descent / Bregman proximal); **Papyan–Han–Donoho** PNAS 2020 (neural collapse = ETF, the K=5 head geometry); **Modica–Mortola / Allen–Cahn** Γ-convergence (the τ→0 sharp-interface limit, Ch.3 hand-off).

---

## 9. One-line summary for the campaign

The frozen softmax **is** a categorical exponential family; our measured **0.978 = the exact trace of its Fisher matrix read through the margin**, its **9.56:1 anisotropy = the codim-1 eigenstructure**, **Muon = a metric-aware weight-space (not Fisher–Rao) step whose −32% is κ-busting**, **CE+softmax = mirror descent = natural gradient (Raskutti–Mukherjee)**, and **the τ-anneal is the mirror-descent Γ-convergence homotopy from this soft Fisher–Rao flow to the sharp perimeter flow of Chapter 3** — that homotopy is Chapter 2's single load-bearing contribution to the duality theorem. All MEANS; pointer 0.19110 UNMOVED.
