# Papers-checked: "The Geometry of Noise" (arXiv 2602.18428) — deep-dive + honest fork

- **Date:** 2026-07-14
- **Paper:** *The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning*
- **Authors:** Mojtaba Sahraee-Ardakan, Mauricio Delbracio, Peyman Milanfar (Google). Submitted 2026-02-20.
- **Verdict (one line):** **DOMINATED-bookmark.** Different regime (diffusion/high-codimension/stochastic-Gaussian) than our diffusion-free codim-1 deterministic argmax fit; the ONE transferable germ (metric-preconditioned descent that tames a boundary singularity) is something we have ALREADY measured (margin = Fisher surrogate, L1). No new lever, no new $0 probe. Bookmarks to the Fisher-metric / margin-as-natural-gradient facet + LEVER-4 margin-saliency capacity routing (task **#268**).
- **Sources read:** arxiv.org/abs/2602.18428 + arxiv.org/html/2602.18428 (full text incl. app. refs) via WebFetch [MEASURED — fetched]; intuitivepapers.ai/geometry-of-noise explainer; EqM lineage repo github.com/raywang4/EqM (MIT); WebSearch for critiques/forward-cites.

---

## 1. FULL MATH (restated with hypotheses; DERIVED from fetched text)

The paper answers: *when the diffusion noise level `t` is treated as a latent random variable, what landscape is a noise-**agnostic** (time-invariant, "blind") field descending, and why does it stay stable near the data manifold where the naive gradient diverges?*

### 1.1 Marginal Energy (Eq. 1 / Eq. 8)
Marginalize the noisy likelihood over an unknown noise-level prior `p(t)` (uniform on [0,1]):
```
E_marg(u) = -log( ∫ p(u|t) p(t) dt )   =  -log p(u)
```
This is the negative log of the **marginal** noisy density (all corruption levels mixed). The blind field is claimed to descend `E_marg`, not any single-`t` energy.

### 1.2 The Energy Paradox — the 1/b(t) singularity normal to the manifold (Eq. 9, 12)
Gradient of the marginal energy (posterior-over-`t` expectation of the per-level score):
```
∇_u E_marg(u) = E_{t|u}[ (u − a(t) D*_t(u)) / b(t)^2 ]      (Eq. 9/11)
```
where `a(t)`,`b(t)` are the forward-process signal/noise schedule coefficients and `D*_t` is the optimal denoiser. **Divergence result (Eq. 12):** as `u → X` (a data point `x_k` on the manifold),
```
lim_{u→x_k} ‖∇_u E_marg(u)‖ = ∞ ,   scaling O(1/b(t)) or worse.
```
So the marginal-energy landscape is an **infinitely deep potential well** on the manifold — a naive gradient-descent field is unstable there.

### 1.3 The geometric key (the "geometry of noise" claim; concentration)
**Distance-to-manifold IS the noise level.** For data on a `d`-dim `C²` manifold embedded in `D`-dim ambient space, adding isotropic Gaussian noise pushes a sample to radius `r` where the codimension concentrates the shell, giving the estimator
```
σ̂ = r / √(D − d)
```
i.e. the posterior `p(t|u) → δ(t̂)` concentrates so the blind network can **read off the noise level from geometry** rather than being told it. **Load-bearing hypothesis: codimension `D − d > 2`** (Lemma 6, App. B.2), plus `C²` manifold, continuous/bounded density. Two proven regimes: **Regime I** high-dim global concentration (`D ≫ d`, shell separation); **Regime II** local proximity (any dim, near-manifold `p(t|u)→δ`).

### 1.4 Riemannian resolution — conformal metric compensation (Eq. 14, Sec 5.1)
The learned autonomous field decomposes:
```
f*(u) = λ̄(u)·∇E_marg(u)  +  (Transport Correction)  +  (Linear Drift)      (Eq. 14)
                └ natural-gradient term ┘
```
with effective gain
```
λ(t) ≜ (b(t)/a(t)) · ( d(t)a(t) − c(t)b(t) ).
```
`λ` acts as a **conformal metric** that **vanishes at exactly the rate `∇E_marg` diverges**, so the product `λ̄·∇E_marg` stays bounded. Mechanically: the flow is a Riemannian **preconditioned** gradient flow — "descend an infinitely steep slope while shrinking your stride to zero at the same rate," converting the singular well into a **stable attractor**. This is the paper's central theorem.

### 1.5 Parameterization determines stability — Jensen gap (Sec 6, Table 2)
The stability depends entirely on the **prediction target** via a gain `ν(t)`:
- **Noise-prediction (ε / DDPM-DDIM):** `ν(t) ∝ 1/b(t)` → diverges as `t→0`. Residual posterior uncertainty is multiplied by a singular gain → drift error `Δv = |ν(t)|·‖f* − f*_t‖ → ∞`. The "Jensen gap" (mismatch between harmonic mean of noise levels and the true level) is the amplified quantity. Structurally **unstable** when blind.
- **Velocity / flow-matching (v, EqM):** `ν(t) = 1` (bounded). Uncertainty is absorbed into smooth geometric drift, `Δv → 0`. Bounded-gain is **necessary** (not strictly sufficient) for blind stability.
- **Empirical support (cited, Sun et al. CIFAR-10):** blind flow-matching FID **2.23** vs blind noise-prediction FID **40.90** — ~18× gap, matching the theory.

---

## 2. AUTHOR-OSS HARVEST (MEASURED — fetched repo)
- **The paper itself ships NO code** (purely theoretical; confirmed by WebFetch of full text + two searches).
- **Lineage OSS = Equilibrium Matching (EqM):** `github.com/raywang4/EqM`, **MIT license**, official PyTorch. This is the noise-agnostic generative model the paper theorizes about (single time-invariant gradient field of an implicit energy landscape; FID 1.90 ImageNet-256).
  - Files: `train.py`, `models.py` (EqM-B/2, EqM-XL/2 DiT-class nets), `sample_gd.py` (NAG-GD / vanilla-GD optimization-based sampler), `sample_ddp.py`, `train_utils.py`, `wandb_utils.py`.
  - **Reusable-to-us assessment:** NOTHING transfers to a coord-INR fitting a frozen SegNet argmax. The "generic" part (GD-on-a-learned-energy sampler) presupposes a *learned energy field over a stochastic generative manifold*; we fit a **deterministic** target with no energy-sampling loop. Critically, **there is no standalone natural-gradient / metric-preconditioned optimizer exposed** — the conformal metric in the paper is an *emergent property of the learned field*, not a reusable optimizer module. So even the one conceptually-interesting mechanism is not packaged as harvestable code.
- **Harvested pattern (not a link):** the only durable takeaway is conceptual (§1.4) — *precondition boundary-singular descent by a metric that vanishes at the singularity rate* — and we already implement its analog (margin-saliency).

## 3. CRITIQUES + FORWARD CITATIONS (MEASURED — searched)
- **None found.** Paper is 2026-02-20; only explainers exist (intuitivepapers.ai, two Medium posts, a Google-authored X/Twitter thread). No replication, rebuttal, or peer critique surfaced in two targeted searches. The strongest independent-ish datapoint is the *cited* Sun et al. CIFAR-10 blind-FID gap (2.23 vs 40.90), which corroborates the parameterization claim. Treat the paper as **un-contested theory**, INFERRED-stable but not yet stress-tested by the community.

---

## 4. HONEST FORK to our live crux (DERIVED)

**Our regime (V9·CGauge witness):** diffusion-FREE. No noise process, no `t`, no denoising, no generative sampling. A single deterministic nonlinear coord-INR fit at **n=1** to a **frozen SegNet argmax partition**. Binding residual = temporal sub-pixel advection PHASE (flicker floor ~0.0053, L85/L86) + Gibbs ringing; basis has a measured 3.2× along-tangent frequency deficit (#497/#502 curvelet cure).

### The regime divergence is decisive on THREE axes (why this is DOMINATED, not a lever):

1. **No noise / no diffusion.** The entire paper is *about the geometry of Gaussian corruption integrated over noise levels*. We inject no noise and integrate over no `t`. The marginal-energy object `E_marg = -log∫p(u|t)p(t)dt` has **no referent** in our loss — our objective is `100·d_seg + √(10·d_pose) + 25·bytes/N` on a fixed target. The paper's headline result (blind ⇒ read `t` from geometry) answers a question we never ask.

2. **Codimension is the OPPOSITE regime.** The paper's load-bearing hypothesis is **high codimension `D−d > 2`** (concentration `σ̂ = r/√(D−d)` needs a big shell). Our binding geometry is the **codim-1 separatrix** — the argmax class boundary is a hypersurface (codimension 1) in the 2D image domain, and our ~8-dim lane manifold is a *low-codim* boundary structure, not a high-codim point-cloud shell. In our regime `D−d` is small, so the paper's concentration proofs (Regime I) do not apply. Regime II (local proximity, any dim) is closer but it is a statement about noise-posterior collapse near the manifold — still requires a noise posterior we don't have.

3. **Deterministic vs stochastic; amortization vs sampling.** Our "error" (advection phase, Gibbs) is a *deterministic* reconstruction residual, not stochastic estimator variance. The paper's instability is *sampling-time error amplification* under a singular gain; we have no sampler.

### The ONE genuine resonance (already ours; do NOT inflate):
The paper's §1.4 mechanism — **precondition a boundary-singular gradient flow by a metric that vanishes at the singularity rate** — is a sibling of our measured **unified level-set flow** facet (L1): *distortion lives on the codim-1 separatrix in the frozen-scorer **Fisher metric**; Fisher curvature ↔ (−margin) Pearson **0.978** ⇒ the **margin field IS the Fisher surrogate**; UNIWARD steg-cost = same metric read as cost.* Near our argmax boundary the margin → 0 and `∂d_seg/∂field` blows up — a boundary singularity structurally analogous to the paper's `1/b(t)` well. Our answer is the SAME shape: **margin-saliency capacity routing** (LEVER-4, KKT waterfill on margin-saliency, `boundary_routing.py`) = a margin-conditioned (natural-gradient-flavored) preconditioner. So the paper is **independent theoretical GROUNDING** for why margin-preconditioned descent is the right move near the separatrix — it is NOT a new mechanism for us.

A **faint second resonance** (mention, don't inflate): the paper's "parameterization determines blind stability" (velocity stable, ε catastrophic) rhymes with our measured *activation/parameterization determines stability* result — fixed-β hosc DIVERGES (tanh saturation → vanishing grad → AdamW random-walk → d_seg RISES) while `step_basis`/annealed-hosc is stable (L6, DAG FEED-ly). Same *shape* ("wrong parameterization near the singularity destabilizes descent"), **different mechanism** (activation saturation vs singular-gain amplification). Not actionable — it is a coincidence of framing, not a shared theorem.

### Bookmark + next-probe:
- **Target cluster:** the Fisher-metric / margin-as-natural-gradient facet of the unified level-set flow (**L1**) + **LEVER-4 margin-saliency capacity routing**, whose exact through-R measurement is the already-owed **task #268** (`msal_uni` was INERT as a texture proxy; the exact `S_R` version is owed). **NOT** #497/#502 (curvelet basis = along-tangent frequency, orthogonal to noise geometry). **NOT** L85/L86 (temporal advection phase, orthogonal).
- **$0 next-probe:** **NONE unlocked.** The paper adds theoretical justification for an *already-owed* measurement (#268 exact margin-saliency through R); it does not open a new free probe and does not touch d_seg/d_pose/bytes. Do not spawn work off this — it is a MEANS-neutral read.

### Bottom line
No lever. The paper is a clean, un-contested piece of diffusion theory whose central geometric object (noise integrated over levels, high-codim concentration, singular-gain parameterization) does not exist in our diffusion-free, codim-1, deterministic-argmax regime. Its transferable germ (metric-preconditioned boundary descent) is a re-derivation-from-another-field of what we already measured as the margin=Fisher-surrogate facet. Pointer UNMOVED (0.18804/0.19108); this read moves nothing and is honestly logged as a DOMINATED-bookmark.
