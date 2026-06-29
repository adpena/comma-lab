# Jitter path-space: deep-math + OSS map of the per-frame SegNet-argmax jitter wall

**UTC:** 2026-06-29T22:16Z · **Authority:** $0 deep-math + online/OSS subagent (NO GPU, NO paid dispatch, NO training).
**Status:** `[macOS/advisory · research-signal · MEANS]` · pointer contest-CPU **0.19110 UNMOVED**.
**NO-FAKE:** every arXiv ID was web-checked this session (verification ledger at the end); where a number is an
interpolation/model vs a measured row I say so; the door is bounded HONESTLY both ways (open *iff* the noise-law
entropy is low + the ringing fraction is correctable — narrow if the residual is white).
**Reframe (operator 2026-06-29):** the budget-gate "negative" (FEED-kb) refuted ONE configuration — *uniform
whole-bulk store of per-frame samples via warp* (rate 0.118, S=0.26). It did NOT close the space. This memo
OPENS the space: it maps the many paths through the jitter and ranks them by EV toward a lower exact score.

---

## 0. The crux in one screen (from the DAG: F4/jc/jd + jq/jz/kb)

The contest collapses to ONE number with a measured pass-line:

> **S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489**, and with the SDS-TSC base (pose ~875 B → pose-term
> ~0.025; SDF+movables → base rate ~0.0021): **sub-0.15 ⟺ d_seg ≤ 1.23e-3** (sub-0.19 ⟺ ≤ 1.63e-3).

The binding term is **d_seg**, and the binding obstacle is the **per-frame SegNet-argmax JITTER**:

- warp-only through-R total d_seg ≈ 0.0076–0.0080; **bulk** (Road/sky/hood) per-frame jitter floor ≈ 0.0023–0.0037 (2–3× budget); lane flip ≈ 0.39 → 0.0023; movables ≈ 0.0005 (FEED-jq/jz/kb).
- The floor is **intrinsic texture-dependent per-frame SegNet decision noise**, NOT warp error: the static hood (warp-free identity regime) STILL carries 32% of bulk flips (FEED-kb M1 smoking gun). Exact GT poses do not beat the proxy (floor does not drop).
- It is **reproducible-by-storage** (PR95 stores per-frame partitions → d_seg ~6e-4) → so it is a **RATE question, not a hard Bayes-error wall** (A1 RESOLVED).
- The REFUTED config: naive margin-keyed dither = 177,926 B/600 → rate **0.118** (≈ PR95 whole archive); jitter "only moderately annulus-localized, median flip margin 0.37"; S = 0.26.
- **The one open door (FEED-kb):** a TRAINED content-aware generator emitting the jitter from a *compact conditioned code* — must beat PR95's rate. UNTESTED.

This memo's job: give that "open door" a **principled mathematical form** (not a black box), the **literature** it draws from, the **quantitative bar** it must clear, and the **highest-EV $0 test** that decides it.

---

## 1. THE LEAD LENS (operator unlock): jitter = a STOCHASTIC-FRONT TERM in the level-set action

The level-set "Crystal Cathedral" frame (FEED-jx) already says the witness φ is the viscosity solution of a
PDE and minimizing the indirect-RD action **S_τ** over φ IS the codec. The jitter is NOT an external wall — it
is the **stochastic (fluctuating-front) term of that same PDE.** This single object subsumes the noise-process
(axis 1), the annulus factorization/waterfill (axis 4), and the stat-mech interface (axis 5).

### 1.1 The term and its Euler–Lagrange / SPDE form (clean, not a patch)

Deterministic v2: the per-class boundary is the zero level set of φ, advected by the screw warp **V** (drift):
∂φ/∂t + V·∇φ = 0 (transport), with the eikonal |∇φ|=1 and length regularizers as the static energy.

Add the stochastic front. The canonical model is the **stochastic Allen–Cahn equation**, whose *sharp-interface
limit* is **stochastic mean-curvature flow** — a deterministic geometric law (mean curvature / our drift)
**plus a stochastically-perturbed normal velocity**:

> **dφ = [ V·∇φ + κ|∇φ| ] dt + σ(x,φ)·|∇φ| ∘ dW(t)**   (front velocity = drift + diffusion)

where W is a (generally **colored**, not white) noise process on the front and σ(x,φ) is the **local noise
amplitude**. In the contest this is exactly: deterministic drift = the ego-screw warp (free, the safe bulk),
stochastic normal-velocity diffusion = the per-frame argmax jitter on the boundary. This is a genuine
Euler–Lagrange/SPDE term, not a bolt-on: it is δS/δφ for the action with an added Gaussian noise potential
(Onsager–Machlup action of the perturbed flow).

**Grounding (verified real):** sharp-interface limit of stochastic Allen–Cahn = stochastically-perturbed mean
curvature flow (Springer 10.1007/978-981-10-0849-8_4); **Funaki 1995** (Probab. Theory Relat. Fields 102:221–288)
— the explicit interface SDE limit; **stochastically-perturbed MCF by colored noise** (arXiv 1811.04265);
large-deviations action functional of the perturbed flow (arXiv 1604.02064); 1D interface fluctuations, singular
regime (arXiv 2508.15319, 2025). The Onsager–Machlup / large-deviations action is the rigorous form of "the
jitter is a term in S_τ."

### 1.2 The RATE = entropy of the LAW, not the samples (the budget reframe)

The budget-gate stored the **per-frame samples** of the front (0.118). The SPDE says we only need to store the
**law** of the fluctuation: the **noise amplitude field σ(x)**, its **correlation length / spectrum**, and a
**seed**. The decoder integrates the SPDE deterministically (fixed seed → bit-identical) and reproduces a front
with the *correct statistics*. Rate(law) = #(params describing σ + correlation structure) + per-frame driving
coefficients, which can be ≪ #(per-frame sample bits) **iff the field is spatially/temporally correlated**
(low-rank, power-law spectrum). A Gaussian random field is fully specified by its power spectrum + a phase seed
(spectral-synthesis method); a correlated front is a few low-k modes.

**This is where the door opens or narrows — and it is decided by a measurable entropy**, developed in §3.

### 1.3 Emergence (falls out, not imposed) — couples to the Fisher metric

The measured jitter structure is PREDICTED by the term's coupling, it is not assumed:

- **σ ∝ 1/margin** (noise amplitude ∝ inverse decision margin). The SegNet flip-susceptibility is the softmax
  variance p(1-p), maximal at margin 0 → σ is largest exactly on the thin boundary annulus → **annulus
  concentration is automatic** (this is the Fisher-metric coupling: the output Fisher F=diag(p)−ppᵀ peaks at the
  boundary, FEED-jc rank-K−1; the noise rides the high-Fisher directions). This is **fluctuation–dissipation**:
  the flip variance (dissipation) ∝ the susceptibility (response) ∝ the *free* margin field.
- **spatial correlation ∝ content/texture** → the front is a *correlated elastic interface* (Edwards–Wilkinson/
  KPZ roughening; random-field-Ising pinning by the texture → Barkhausen-like correlated avalanches), giving a
  **power-law displacement spectrum** → sparse in the contour-aligned wavelet/Fourier basis.
- "median flip margin 0.37, only moderately annulus-localized" is exactly a *finite-temperature rough interface*
  (not razor-thin, not white) — consistent with a correlated SPDE front, which is the compressible regime.

**Consequence:** the decoder can compute σ(x) from its OWN margin field (free) and only needs the *residual*
correlation structure + seed. The safe-bulk-waterfill (axis 4) is then **derived**, not designed: it is the
drift/diffusion (Hodge-like) split of the front velocity — deterministic drift = free safe bulk (high margin,
σ≈0, never coded); stochastic diffusion = the annulus law (the only counted term).

---

## 2. The THREE-PART decomposition of the jitter (coordinator add — changes the verdict)

The "intrinsic floor" is not monolithic. Peel it into three mechanisms with very different rate costs:

**(a) GIBBS / RINGING / ALIASING — DETERMINISTIC, CORRECTABLE (not stored as samples).**
Band-limited reconstruction of a sharp class boundary oscillates near the edge and crosses the argmax threshold →
flips. Three band-limiters stack: R's bicubic↓ low-pass, **SegNet's stride-2 efficientnet stem** (a known
aliasing source), and any truncated witness basis. This is the *shift-variance* of strided CNNs: Zhang 2019
("Making Convolutional Networks Shift-Invariant Again", arXiv 1904.11486, BlurPool) shows strided downsampling
ignores Nyquist → small input shifts cause large output flips; Alias-Free ConvNets (2303.08085) and Frequency
Pooling (2109.11839) extend it. **A real chunk of the "intrinsic" bulk floor is shift-induced aliasing that
oscillates deterministically at the band-limit (amplitude ∝ edge contrast, located at the edge) → correctable
by anti-ringing, NOT by storing per-frame bits.**
*Anti-ringing toolkit (signal-processing, real):* oversampling render≥320 to push Nyquist past the 2px lane
(= F1's measured lever; SDF@192≈hard@384); windowed-sinc/Lanczos / apodized downsample (reduce Gibbs);
deconvolution (invert the known low-pass); **sigma-delta / noise-shaping dither** (move the ringing error OUT of
the decision band — the margin-keyed dither IS a crude noise-shaper). Coheres with F1 (the 1-Lipschitz SDF
survives R *because* it has no Gibbs — it is anti-ringing by construction), graphics AA (anti-aliasing =
anti-ringing), and curvelet DM2 (oriented basis avoids ringing at the curved boundary).

**(b) STOCHASTIC-FRONT fluctuation — STORE THE LAW (cheap if correlated).** §1: σ-field + correlation +
seed. Prior art for the *segmentation* case is exact and strong: **Stochastic Segmentation Networks**
(Monteiro et al., NeurIPS 2020, arXiv 2006.06015) model **spatially-correlated aleatoric segmentation
uncertainty as a LOW-RANK GAUSSIAN over the logit field** (mean + low-rank covariance factor) and sample
coherent label-map hypotheses. That is *literally a compact parametric law of the boundary jitter* — a rank-r
covariance is r·(#annulus-pixels) numbers, far fewer than per-frame argmax bits, and the published finding is
that low rank already captures the spatially-coherent uncertainty.

**(c) GENUINE SegNet ALEATORIC noise — irreducible, possibly SMALL once (a)+(b) are peeled.** The part of the
flip that is white given *everything* (pose, content, margin, neighbors). To MATCH the specific GT realization of
white noise costs its full entropy; but if (a) is corrected and (b) is law-coded, the *residual* (c) rate could
be ≪ 0.118. **Honest caveat (NO-FAKE):** the FEED-kb hood smoking gun (32% of bulk flips in the warp-free
identity regime) shows a real (c)-like component exists; the open question is whether it is mostly (a)
shift-aliasing of the static hood edge (correctable!) or true (c) — *this is the measurement, §4*.

**The verdict to produce:** the **(a)/(b)/(c) split**. The budget-gate stored ALL of it as samples (0.118).
If (a) is largely correctable and (b) is law-cheap, **the residual irreducible-(c) rate could be ≪ 0.118 →
budget opens.** If (c) dominates and is white, the door is narrow (≈ PR95, ~0.19, no sub-0.15 via this term).

---

## 3. The RD curve, not a point (axis 3) — the QUANTITATIVE BAR (my key contribution)

S is a surface over (d_seg, d_pose, rate). With base pose-term 0.025 + base rate 0.0021, the operating curve is
**S(D) = 100·D + 0.027 + rate_jitter(D)**, where D is achieved d_seg and rate_jitter is the jitter-coding rate.

**Marginal trade (interpretable units).** Reducing d_seg by Δ fixes Δ·N_pix flips (N_pix = 600·512·384 ≈
1.18e8 pixel-evals); the d_seg term drops 100·Δ; the rate term rises 25·B/37.5e6. Fixing K flips is worth it iff
**B < ~1.27·K bytes ≈ 10.2 bits per flip.**

**The naive/PR95 line.** From the measured point (reduce d_seg by 0.00177 cost 177,926 B → ~1.0e8 B per unit
d_seg), the naive dither RD line runs from (D=0.0030, rate=0, S=0.327) toward the full-store end. PR95 — which is
already an arithmetic-coded HNeRV latent — sits at (D≈6e-4, rate≈0.118, **S≈0.20**). **The naive RD line
asymptotes at PR95: storing more samples just walks you to PR95 ≈ 0.19–0.20 and no further.**

**The bar for sub-0.15 (robust, model-level).** At the low-d_seg end (D≈6e-4, capacity-limited by the SDF
carrier), S<0.15 requires **rate_jitter < 0.063** vs PR95's 0.118 → **a ~1.87× compression of the jitter payload
at equal d_seg**; S<0.13 requires ~2.6×. Equivalently: **the structured jitter code must beat PR95's
already-arithmetic-coded latent by ~1.9–2.6×**, by exploiting the structure PR95 ignores: (i) the rank-8 ego-
motion coherence of the annulus (98.6% of variance, FEED-it), (ii) the 1-D contour structure of the boundary
(flips live on a curve, not a 2-D field — the naive dither coded a 2-D field), (iii) temporal AR coherence
frame-to-frame, (iv) the low-rank Gaussian law (SSN), (v) σ∝1/margin computable free at the decoder, and (vi)
peeling the deterministic (a)-ringing entirely.

**This is the non-pessimistic headline:** the "wall" is a **concrete, plausible, measurable compression-factor
target (~1.9–2.6× over PR95)**, not a closed door. The endpoints are both > 0.19; the question is whether the
*structured-code* RD curve dips into the interior below 0.15. Five independent structure handles, each worth a
modest factor, multiply.

**Honest bound (NO-FAKE):** the 1.0e8-B/unit slope and the linear interpolation are advisory (one measured anchor
+ the PR95 endpoint); the 1.87–2.6× bar is robust to that (it is a ratio of rates at fixed d_seg). Whether the
five handles *actually* deliver ~2× is the empirical question — but a ~2× win over a *generic latent* by switching
to a *geometry+pose-conditioned, boundary-1D, low-rank-Gaussian* code is well within what structured coding
routinely achieves; this is not a Hail-Mary.

---

## 4. Info-theoretic floor (axis 2) — verdict

The right object is the **rate–distortion function R_j(D) of the per-frame argmax field under d_seg (Hamming)
distortion** (an indirect-RD / CEO problem: the "source" is the boundary location, the per-frame texture is the
noisy channel). **There is NO hard floor above 0** — the field is reproducible-by-storage (PR95 reaches 6e-4),
so R_j(0⁺) is finite, not ∞. The floor question is purely the *shape/slope* of R_j(D).

The decisive scalar is the **conditional entropy H(flip | free conditioning)** where free conditioning = {pose,
canonical content, neighbor frames, margin field}:
- If **H(flip | free) ≪ H(flip)** → the flip is mostly explained by cheap side info → conditioned SPDE
  generator emits it for few bytes → **R_j(D) is steep → door OPEN**.
- If **H(flip | free) ≈ H(flip)** (white given everything cheap) → R_j(D) ≈ the naive line → **door NARROW**.

Two structural facts already argue OPEN (not proof): (1) the annulus motion is **rank-8 / 98.6% ego-coherent**
(FEED-it) → the *location* is highly conditional-compressible; (2) SSN's published result is that segmentation
aleatoric uncertainty is **low-rank** (spatially correlated, not pixel-white). The residual (c) after removing
rank-8 motion + (a) ringing is the unknown — that is the precursor probe FEED-kb spawned, sharpened in §6 below.

---

## 5. Re-rank of the OTHER vehicles given the jitter wall

The jitter is a **SegNet property, not a vehicle property** — it hits *every* per-frame-partition approach
equally. So the re-ranking collapses:

- **PR95-recode + sub-linear seg lever (l235):** PR95 *is* the "store all per-frame samples" baseline (0.118,
  S≈0.20). It already arithmetic-codes a generic latent → it is the **parity bank + the number to beat by
  ~1.9–2.6×**. The l235 lever broke the d_seg plateau (0.00374→0.00247) but overfits without FiLM-v2 — it lowers
  d_seg but does NOT change the rate of the jitter payload, so alone it lands ≈0.19, not sub-0.15. **Not a
  separate sub-0.15 path; it is the fallback bank.**
- **Quotient codec / weight-requant / base-rate levers:** attack the BASE rate (already ~0.002) — they do not
  touch the binding jitter term. **Dominated for sub-0.15.**
- **v2-witness as a stochastic-front law-coder:** this is *not* a separate vehicle from PR95 — it is precisely
  "PR95's per-frame payload, but coded as a structured stochastic-front LAW (σ-field + low-rank-Gaussian +
  contour-1D + rank-8 motion) instead of a generic HNeRV latent." **It IS the ~1.9–2.6× jitter compressor PR95
  needs.** That unification is the strongest framing: every vehicle converges on the same binding lever, and the
  SPDE-law form is the principled instance of it.

**Conclusion:** v2 (in its stochastic-front law-coding form) remains the best EV *because* it is the only framing
that directly attacks the binding term with structure the parity bank ignores. No other vehicle dominates — they
all hit the same SegNet jitter; the others either *are* the baseline (PR95) or miss the binding term (quotient).

---

## 6. RANKED paths forward (by EV toward a lower exact score)

Tags: **$0** = free CPU/MLX measurement now · **GPU** = needs the training run · **design** = math/spec work.

### #1 (highest-EV $0 — the decisive gate) — the (a)/(b)/(c) jitter-decomposition + law-entropy probe
**One $0 measurement that produces the budget verdict.** On the cached gt frames + the v2 SDF witness, through R
+ frozen CPU-torch SegNet (the established authority; NO-FAKE SegNet(gt)==lstars self-check):
1. **Peel (a):** re-render the witness at render≥320 with a windowed/Lanczos (anti-aliased) downsample and a wide-
   ramp SDF; measure the flip-count drop vs naive bicubic. The drop = the **correctable ringing fraction**.
   (Also: a BlurPool-style low-pass before the argmax as an upper bound on shift-aliasing flips.)
2. **Peel coherent motion:** remove the rank-8 ego-warp of the annulus (already measured; FEED-it).
3. **Fit the law (b):** fit a low-rank Gaussian (SSN-style) AND a spectral/correlation-length model to the
   *residual* flip field; report the **variance-explained vs parameter-count curve** and the **implied law-rate**
   (params + per-frame coeffs + seed) in bytes.
4. **Bound (c):** the residual unexplained variance = the white aleatoric tail; its rate = its entropy.
**Verdict it yields:** the (a)/(b)/(c) split + the implied total jitter rate vs **0.118** and vs the **0.063
bar**. If rate ≪ 0.118 → **door OPEN, route GPU**; if (c) dominates white → **door NARROW, bank PR95 + rethink.**
This sharpens/subsumes the precursor probe FEED-kb spawned ("predictable vs white") with the right mechanistic
split and the right basis. **EV: this single test decides the whole vehicle. Do it first.**

### #2 ($0/design) — the contour-displacement power-spectrum / structure-function measurement
Measure C(r)=⟨[h(x+r)−h(x)]²⟩ of the boundary displacement *along the contour* + the temporal AR(1) coefficient.
A power-law C(r) (correlated EW/KPZ interface) ⇒ sparse in the contour-wavelet basis ⇒ cheap law (door open); a
flat C(r) (white) ⇒ narrow. Gives the **right basis** for the #1 law-fit and a clean stat-mech read of the regime.
(Real grounding: GRF spectral synthesis — store spectrum + phase seed; EW/KPZ roughening exponent.)

### #3 (GPU — the pointer-mover, gated on #1) — the conditioned stochastic-front residual generator
The actual lane-survival + bulk-jitter training run, but **specified as the SPDE law-coder**, not a black box:
a few-step **deterministic flow-matching ODE residual-correction** (GNVC-VD 2512.05016 init-from-prior; OT-NFM
2604.06413 one-step; LieFlow 2512.20043 deterministic) conditioned on {pose, σ=1/margin, openpilot centerline
(Wyner-Ziv, Whang 2106.02797 / Özyılkan-Ballé 2310.16961)}, trained on the **temporally-smoothed** target
(DreamSmooth 2311.01450) with the irreducible (c) stored as a **margin-keyed annulus dither** (PICM-Net
2512.20070). **Success criterion is now QUANTIFIED by §3: the coded jitter payload must be ≤0.063 rate at
d_seg≈6e-4 (≥1.9× over PR95).** Per CLAUDE.md: resumable + per-stage checkpoints + EMA-shadow + explicit operator
GPU steer; deterministic seed for bit-identical decode.

### #4 ($0/design) — soft/weighted gauge for the front (well-posedness)
A hard-argmin canonical front can be discontinuous across the pose manifold (Dym et al. 2402.16077, verified
theorem) → the residual generator's target is discontinuous → harder to code. Use a **soft/weighted gauge** so
the stochastic-front law is continuous. Cheap design constraint, prevents a real failure mode. (Codec-sweep #3.)

### #5 (design/contribute-back) — formalize the stochastic-front term in the canonical equations registry
Add S_τ → S_τ + Onsager–Machlup action of the perturbed front as a canonical equation (Funaki/stochastic-MCF
grounding), with the rate = entropy-of-the-law and the (a)/(b)/(c) split as named sub-terms. Makes the level-set
eureka and the jitter resolution ONE object in the registry; feeds the GPU-run design and any paper.

---

## 7. Synthesis (memory-worthy, 1 paragraph)

The per-frame SegNet-argmax jitter — the binding wall to sub-0.15 (everything else is solved: rate ~0.002, pose
~875 B, sub-0.15 ⟺ d_seg ≤ 1.23e-3) — is best modeled NOT as an external wall but as the **stochastic
fluctuating-front term of the level-set action** (stochastic Allen–Cahn → stochastic mean-curvature flow; Funaki
1995; 1811.04265): front velocity = deterministic ego-screw drift (free safe bulk) + stochastic normal diffusion
(the jitter, on the high-Fisher boundary annulus, σ∝1/margin by fluctuation–dissipation). The budget-gate's
refuted 0.118 stored the **samples**; the SPDE says store the **LAW** (σ-field + correlation/spectrum + seed),
and the jitter splits three ways — **(a) Gibbs/aliasing** ringing from band-limited reconstruction + SegNet's
stride-2 stem (DETERMINISTIC, correctable by oversampling/windowing/anti-aliasing; Zhang BlurPool 1904.11486;
the hood smoking gun may be largely this), **(b) a correlated stochastic front** (store the LAW cheaply; the
published prior art is Stochastic Segmentation Networks 2006.06015 = a low-rank Gaussian over the logit field =
exactly a compact aleatoric law), and **(c)** a possibly-small white aleatoric tail. The RD-curve math gives the
**precise, non-pessimistic bar**: the endpoints are both >0.19 (warp-only 0.33; full sample-store = PR95 0.20),
and **sub-0.15 ⟺ code the jitter payload ≥~1.9× (sub-0.13 ⟺ ~2.6×) below PR95's already-arithmetic-coded latent
at equal d_seg** — achievable by the structure PR95 ignores (rank-8 ego-coherence + 1-D contour + temporal AR +
low-rank-Gaussian + free σ-field + ringing-peel). The decisive **$0** test is the (a)/(b)/(c) decomposition +
law-entropy fit (oversample to peel ringing → remove rank-8 motion → fit a low-rank/spectral law to the residual
→ report variance-explained-vs-params + implied rate vs 0.118 and vs the 0.063 bar): it decides door-open
(route GPU on the SPDE-law generator) vs door-narrow (bank PR95) before any GPU spend. means≠ends; pointer
0.19110 UNMOVED.

---

## 8. Verification ledger (NO-FAKE)

**VERIFIED real this session (web search):**
- Stochastic Allen–Cahn / stochastic mean-curvature flow sharp-interface limit: Springer 10.1007/978-981-10-0849-8_4; **Funaki 1995**, Probab. Theory Relat. Fields 102(2):221–288 (cited across the literature); stochastically-perturbed MCF by colored noise **arXiv 1811.04265**; large-deviations upper bound **arXiv 1604.02064**; 1D interface fluctuations singular regime **arXiv 2508.15319** (2025) + Springer ARMA s00205-025-02121-z.
- **Stochastic Segmentation Networks** (Monteiro et al., NeurIPS 2020), **arXiv 2006.06015** — low-rank-Gaussian spatially-correlated aleatoric segmentation uncertainty (the "store the law" prior art). ValUES validation framework **arXiv 2401.08501**.
- Aliasing/shift-variance: **arXiv 1904.11486** (Zhang, BlurPool, "Making Convolutional Networks Shift-Invariant Again"); Alias-Free ConvNets **arXiv 2303.08085**; Frequency Pooling **arXiv 2109.11839**.
- Gaussian-random-field spectral synthesis (store power spectrum + phase seed; correlation length ↔ spectral peak) — standard method (multiple cosmology/GRF refs).
- Re-used from the codec sweep (already verified there): 2512.05016, 2604.06413, 2512.20043, 2311.01450, 2512.20070, 2106.02797, 2310.16961, 2402.16077.

**Honestly bounded / not overclaimed:** the RD slope (1.0e8 B/unit d_seg) is from ONE measured anchor + the PR95
endpoint → the linear interpolation is advisory; the **ratio bar (1.9–2.6× over PR95)** is robust to it. The
(a)/(b)/(c) split is a HYPOTHESIS with strong structural support (hood smoking gun + SSN low-rank + BlurPool
aliasing) — it is exactly what test #1 measures; it is NOT yet measured. Funaki/stochastic-MCF is the RIGOROUS
PDE home for the term, but applying it to a *discrete argmax* front (vs a smooth phase field) is a modeling
choice, honest. **Door verdict is genuinely two-sided:** OPEN iff (a) correctable + (b) low-rank → residual
rate ≪ 0.118; NARROW iff (c) white dominates. **Pointer 0.19110 UNMOVED; every row here is a research MEANS, not
an exact-eval row.**

**Sources:**
- https://link.springer.com/chapter/10.1007/978-981-10-0849-8_4
- https://arxiv.org/pdf/1811.04265
- https://arxiv.org/pdf/1604.02064
- https://arxiv.org/pdf/2508.15319
- https://arxiv.org/abs/2006.06015
- https://arxiv.org/pdf/2401.08501
- https://arxiv.org/pdf/1904.11486
- https://arxiv.org/pdf/2303.08085
- https://arxiv.org/pdf/2109.11839
- https://github.com/kstoreyf/gaussianfield
