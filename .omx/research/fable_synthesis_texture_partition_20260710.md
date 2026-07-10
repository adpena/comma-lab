# FABLE SYNTHESIS — THE TEXTURED POWER DIAGRAM ON THE SCORER'S OBLIGATION MATRIX — 2026-07-10

**Task (respawn, predecessor lost to session limit — no partial artifact existed; fresh derivation.)**
Synthesis over five convergent measured findings + external math/OSS pulls: derive the ONE structure
they jointly determine. `$0 · read+derive+research · no launches · run dirs read-only.`
**Pointer contest-CPU 0.19110 UNMOVED — everything here is MEANS** `[macOS-CPU advisory ·
research-signal · NON-PROMOTABLE]`. Remaining gap to sub-0.15 = **0.0411 S**.

**STORES CONSULTED:** MEMORY.md CURRENT-STATE (L1/L11/L17/L68/L74/L75/L-v8) ·
`deepmath_amortizing_argmax_paper_draft_20260704.md` (#284 ch.1–6: power diagram · tropical/Maslov ·
caustic/Fisher · curvelet · screw · τ=ε=ħ) · `palette_artifact_probe_20260710.md` ·
`v8_macro_rate_pass_20260710.md` · `resize_exploit_flip_solver_20260709.md` ·
`upstream_scorer_alldim_reread_20260710.md` · DAG FEED-chern-nonnormal + FEED-06h (spike-guard
confound; variance-not-mean killer) + FEED-alldim + FEED-macro-rate · `DUAL_CHAIN_BRIEF_385_20260710.md`
· P1–P12 philosophies (`design_philosophies_eightfold_20260709` — noisefloor, falsifier, floorfirst,
NO-PROXY, negatives-carve) · `src/tac/through_r/flip_inverse.py` (the exact composite/adjoint) ·
`src/tac/canonical_equations/{palette_realization_ceiling,posenet_luma_chroma_asymmetry}_20260710.py`
· `frame0_chromahf_dofs_20260710.md` (UNIT C — landed mid-synthesis, folded in §A2′).
**Boundary-honesty:** `segnet_texture_perception_20260710` is **NOT landed** (not consulted).
`frame0_chromahf_dofs_20260710.md` (UNIT C, commit 428ed04e9) **landed CONCURRENTLY mid-synthesis**
— caught at DAG-append time and FOLDED IN (§A2′): it CONFIRMS frame_0 seg-freedom at n600 (8.5e-9)
and **REFINES** the chroma-HF placement — the exact pose-null holds at the yuv6/384 plane only; a
naive camera-res chroma dither leaks 50% into luma through the 2.28× downsample.

**Label discipline:** every claim below is tagged **MEASURED** (a real n600/n≤600 artifact) /
**DERIVED** (exact from code or mathematics) / **CONJECTURED** (pre-registered, falsifiable, NOT
registered as an equation) / **CITED** (external, verified source).

---

## 0. The one structure (answer-first)

> **The scored object is a textured Laguerre tessellation carried on the scorer's obligation
> matrix:** `W = (G, ξ, T)` — power-diagram **generators** `G` of the frame_1 argmax partition
> (the ~8-dim ego-transported scene manifold), the **ego-screw** `ξ(t)` (banked, 7.2 KB), and a
> per-class family of **stationary, SegNet-legible texture measures** `T = {t_c}` that hold each
> cell inside its argmax basin. `T` is not decoration: partition-without-texture has a MEASURED
> all-palette floor of **d_seg 0.0416** (= 4.16 S — 101× the remaining gap), while textured
> realization measures **0.0048** (8.7× below). The scorer's channel geometry factorizes WHERE the
> three components live: a 2×2 **obligation matrix** (frames × frequency-bands) in which frame_0
> owes only pose, frame_1 chroma-HF owes only seg, frame_1 luma alone is doubly priced — and
> frame_0⊗chroma-HF is **exactly unscored** (a dead subspace). The corrected indirect-RD floor
> gains a texture-rate term that is O(1) per video (statistics, not pixels: synthesize
> deterministically at decode, rule-118 free) and a texture-legibility distortion term
> `100·d_seg*(T)` that is now **the floor-dominating uncertainty of the whole campaign**, bracketed
> MEASURED between 1.6e-7 (GT canary) and 0.0048 (witness).

Each of the five inputs is one face of this object: the palette probe measures `T`'s necessity; the
scorer re-read measures where `T` (and `ξ`) are cheap; the macro-rate pass measures `G`'s rate and
proves the residual is `G`-coverage (patches), not coding; the flip solver is the exact first-order
boundary calculus of the SAME object at the R interface; FEED-chern is its training-time tangent
(covers = charts of `G`; non-normal transients = the flow that fits `T` and `G` jointly).

---

## A. The textured-partition unified object (corrected sufficient statistic, dimension, floor)

### A1. Why the partition alone is NOT the sufficient statistic (MEASURED)

The old indirect-RD framing (L74; S_floor ≈ 0.118 rate-dominated) treated the frame_1 argmax
partition `L*` as the scored sufficient statistic: ship the boundary geometry, realize cells with
any per-class colour, d_seg → boundary noise. The palette probe **refutes the realization half**:

- Realizing the PERFECT `L*` as flat per-class colour floors at **d_seg 0.0416** (zero-R-mixing,
  best palette = per-pair scene mean; every other palette worse; abstract palette catastrophic
  0.504). R contributes only 14%. **MEASURED, n600, all-palette-scope.**
- Root cause: SegNet's argmax is **context/texture-dominated** — constant-colour tiles decode
  Undrivable 195/216, Road 1/216, Lane 0/216. Road and Lane **cannot win argmax on colour at all**.
  **MEASURED.** (External confirmation that this is generic CNN behaviour, not a SegNet quirk:
  the Geirhos texture-bias lineage — **CITED**, §D.6.)
- The trained witness, which renders **textured** RGB, realizes **0.0048** — 8.7× below the flat
  floor and 3.4 orders above the GT canary 1.6e-7. **MEASURED.**

So the scored statistic is the **pair** `(L*, T)`: the partition AND per-class texture sufficient
for the frozen SegNet to re-derive the partition. Formally, per class `c` the texture must place
cell interiors inside the argmax basin: `argmax_k SegNet_k(x)|_{Ω_c} = c` on 1−δ of pixels — a
**basin-membership** constraint, not a fidelity constraint. `T` is an equivalence class (any
SegNet-metamer of the class texture works), which is what makes it CHEAP (§A3).

### A2. The obligation matrix — where the scorer prices each component (DERIVED from code + MEASURED)

From the all-dimension re-read (`modules.py:70–109`, `frame_utils.py:51–78`, all verified
line-by-line; Jacobian MEASURED):

| block | pose obligation | seg obligation | consequence |
|---|---|---|---|
| frame_0 ⊗ luma (+chroma-LF) | YES (f1/f0 = 0.86×) | **none** (`x[:,-1]`; seg_delta=0.0 MEASURED) | pure pose carrier — render as coarsely as pose allows |
| frame_0 ⊗ chroma-HF (above 2×2 yuv grid) | **none** (box-average null, EXACT) | **none** | **dead subspace** — spend nothing, gain nothing |
| frame_1 ⊗ luma | YES (11.1×/plane MEASURED) | YES (texture) | the ONLY doubly-priced block — joint design lives here |
| frame_1 ⊗ chroma-HF **at the yuv6/384 plane** | **none** (EXACT null — MEASURED op-level 3.4e-6) | YES (SegNet reads raw RGB at full 384×512) | the cheap home for `T` — seg texture at ~zero pose cost, **384-band-designed only** |

**DERIVED:** the object `W = (G, ξ, T)` therefore factorizes over this matrix — `ξ` and frame_0's
minimal pose-render occupy row 1; `T` preferentially occupies block (f1, chroma-HF-at-384); `G`'s
boundary placement and whatever texture chroma cannot carry occupy (f1, luma). The NEW content vs
the predecessor framing: **T is not spatially free — it has a canonical cheap embedding**, and the
witness/v8 currently price NONE of this (both render frames symmetrically; FEED-alldim RANK 1/2).

**A2′ — the sibling refinement (UNIT C, MEASURED, folded in):** (a) frame_0 seg-freedom CONFIRMED at
n600 (d_seg 8.5e-9 with random-noise f0); efficient f0 operating point = **luma-only** (−67% bytes,
√10·pose 0.180 UPPER bound — a trained f0 pose-carrier sits far below). (b) The chroma-HF pose-null
is a property of the **yuv6/384 plane**: a naive camera-res chroma dither leaks **50% into luma**
through the 2.28× no-AA bilinear downsample (luma/chroma 1.075 at the scored plane) and pays ~luma
pose cost. The IDEAL lever — a 384-grid-aligned luma-null chroma pattern — is exactly pose-null with
MEASURED seg authority **Δd_seg 2.73e-3** (order of the whole mod32cap d_seg) at zero pose cost.
(c) The two levers are **ORTHOGONAL by construction** (disjoint scorer terms). **Synthesis
connection to §B:** the flip solver's bit-exact `D` kernel is ALSO the accessibility tool here —
render-side chroma texture must be the **pre-image of the desired 384-plane pattern through the
known linear chain** (solve `D·x = pattern` on the extracted operator), which is the same exact
resize calculus used for flip fixing. One kernel, two consumers.

**CONJECTURED (pre-registered falsifier, ranked row #1 in §E), NARROWED by UNIT C:** chroma-HF can
MOVE argmax pose-free (authority 2.73e-3, MEASURED) — but whether 384-band-designed chroma texture
alone provides Road/Lane/Movable **basin legibility** (i.e., cuts the 0.0416 flat-paint floor ≥2×)
is still UNMEASURED. If it fails, `T` must sit in (f1, luma) and pay the 11.1× pose price — the
placement law weakens to "chroma-first, luma-fallback."

### A3. Dimension and rate of the textured object (DERIVED, with MEASURED anchors)

`T` is per-class **stationary** (a texture process, not a pixel map) and **ego-advected** (the same
screw `ξ` that transports `Σ_t = H_t(Σ_0)` transports the texture field's frame — #284 ch.5). So
its payload is a per-VIDEO constant, not per-frame:

- Parametrize `t_c` = a Portilla–Simoncelli-class joint-statistics vector (steerable-pyramid
  auto/cross-correlations + marginals — the classical minimal sufficient statistics a frozen CNN's
  early layers read; **CITED** §D.1–2). |t_c| ≈ 700–1000 floats → quantized **~1–4 KB/class**;
  5 classes ⇒ **5–20 KB one-time** ⇒ rate term `25·bytes/37.5e6` ∈ **[0.003, 0.013] S**.
- The SYNTHESIZER is deterministic, seeded, scorer-free (PS synthesis is fixed-point iteration on
  pyramid statistics — no SegNet at decode; strict-scorer-rule clean) ⇒ **rule-118 FREE** code.
  Encode-side, where the scorer IS available, `t_c` is *chosen* so the synthesized texture is
  SegNet-legible (metamer-style optimization against the frozen scorer — plenoptic's exact use
  case, **CITED** §D.1). Deterministic-reproducibility spine applies (seeded, numpy-fp32 reference).
- `G`: dominant-only geometric rate **0.061 → ~0.0585 S** (MEASURED + R2 ξ-charge DERIVED);
  residual enemy 0.074 is **coverage, not coding** (both re-coding axes MEASURED dead) — in this
  synthesis's language: the residual is *missing charts of G*, exactly the Chern-covers reading
  (FEED-chern), and `T` does not inflate it (T attaches to cells, not boundaries).

**Corrected indirect-RD floor (the honest bracket):**

```
S_floor(W) = R(G) + R(ξ) + R(T) + 100·d_seg*(T) + √(10·d_pose*)
           = [0.0585..0.074-coverage-dependent] + 0.005 + [0.003..0.013] + 100·d_seg* + ~0.018
d_seg*(T) ∈ [1.6e-7 (U1 GT canary, MEASURED) , 0.0048 (witness, MEASURED)]
```

The rate-side correction is small (+[0.003, 0.013]). The distortion-side correction is the finding:
**the floor's dominant uncertainty is ONE measurable scalar — the texture-legibility gap
`d_seg*(T)` — spanning 0.00002 S to 0.48 S** (the latter = 12× the remaining gap). The old floor
statement "0.118, rate-dominated" survives only under the CONJECTURE that synthetic texture can
approach GT-legibility; the campaign's binding unknown has moved from geometry (mapped) and pose
(banked) to texture legibility. This is what the palette probe, read jointly with the witness row,
actually proves.

**Per-class waterfill (MEASURED shape):** flat-misread mass concentrates on Movable 0.367 /
MyCar 0.147 / Lane 0.138 (palette per-class table) — `T`'s byte and legibility budget should
waterfill by this vector, not uniformly (P-philosophy: waterfill-or-justify).

---

## B. The nonlinear flip-solve — closing the 0.594 prediction-vs-realized gap (DERIVED + CONJECTURED)

The #391 solver established the EXACT linear outer structure (kernel bit-exact 5.7e-14; adjoint
verified; ΔS-per-flip uniform 8.477e-7; 98.98% of flips in the 5px annulus) and measured
**verify = 0.594** with 82 collateral flips under the `s0=1` step model. The gap is not the resize
— it is the missing **inner Jacobian**. Derivation of the correct step law:

### B1. First-order step law (DERIVED)

For a residual flip at seg-pixel `p` with margin deficit `μ(p) = z_wrong(p) − z_gt(p) > 0`, a
camera perturbation `δx_cam` changes the margin by

```
δm(p) = ⟨ ∇_x m_p , D · Π_deadzone(δx_cam) ⟩ ,   m_p := z_gt(p) − z_wrong(p)
```

where `∇_x m_p` = the SegNet input-gradient of the margin at `p` (ONE backward pass — this IS the
margin-saliency field #141, evaluated per flip), `D` = the exact bilinear-down matrix (extracted,
bit-exact), and `Π_deadzone` gates by uint8 headroom. The optimal direction and step are

```
d*(p) = Dᵀ ∇_x m_p |_footprint      (≤ 2×2 camera taps × 3 ch = ≤12-dim cone)
α*(p) = μ(p) / ‖P_R ∇_x m_p‖ · (1 + O(κ·μ))
```

`P_R` = projection onto the through-R reachable subspace. The `s0=1` model replaced `∇_x m_p` with
the identity — it ignored (a) the **stem filter bank** (stride-2 conv attenuates and spatially
spreads a single-pixel RGB step; effective logit gain ≠ 1), (b) the part of `∇m` in `ker(D)` ∪
dead-zone (unreachable), (c) activation curvature over 16-LSB steps. All three push realized < 
predicted — the measured ~40% over-prediction has the right sign and plausible magnitude.
**DERIVED** (the law is exact to first order; its accuracy is what row E-3 measures).

### B2. Newton/secant correction + cluster QP (DERIVED spec, CONJECTURED lift)

- **Per-flip secant:** after the first-order step, re-evaluate `m(x+αd)` on a receptive-field crop
  (1–2 extra forwards) and update `α ← α + (μ − Δm)/⟨∇m, d⟩` — a 1-D Newton on the margin along
  `d*`. Handles activation switching (the true second-order effect; the deep Hessian never needs
  forming).
- **Cluster QP for collateral:** build the footprint-overlap graph over targeted flips; per
  connected cluster solve `min ‖δx‖²  s.t.  J δx ≥ μ + ε (targets),  J' δx ≥ −m_j (guard set =
  small-positive-margin neighbours inside the union RF)`. `J` rows = per-flip gradients. Clusters
  are small (flips are annulus-sparse); each QP is ≤ tens of variables after footprint
  restriction. This is trust-region SQP on the margin field — iterate linearize→QP→verify.
- **Cost (M5 Max, $0):** one backward per (pair, flip-cluster) on CPU-torch at 384×512 ≈ 1–2 s;
  the 512-flip/346-pair verify set re-runs in ≈ 15–30 min.

**CONJECTURED (pre-registered):** true-gradient first-order alone lifts verify 0.594 → ≥0.80;
+secant+QP → ≥0.90 with collateral ≤ 10. Falsifier in §E-3. Connection to A: the flip solver is
the boundary-jitter calculus of `W` (the 14% + cheap-fraction mass); the interior-texture flips it
cannot reach (0.65% unreachable, #149 wall) are exactly `T`'s domain — the two instruments
partition the residual by mechanism.

---

## C. The pseudospectral instrument — buildable spec (DERIVED; upgrade of #318)

**What failed:** #318's calibrate-by-DE stability analysis is mean-field LINEAR (von-Neumann on the
SPECTRUM): it checks the asymptotic abscissa `α(M) = max Re λ`. The ep300 bump (3.4×) and the
ep108–114 variance events occurred with α "stable" — they are **transient growth of a NON-NORMAL
operator** (FEED-chern: `H = A + D₀ + εE`, transport skew + symmetric landscape + non-normal
coupling), plus a stochastic component the deterministic analysis cannot see (FEED-06h: the guard
tripped on batch VARIANCE, not the mean).

**The exact computable object (the spec):** at each stage-boundary checkpoint (τ switch, lane-band
start, l7 in/out, Muon switch, any spike-guard event), instrument the **preconditioned minibatch
training operator** `M = P·H` (P = AdamW diag / Muon polar factor at that step; H = Gauss-Newton
of the coupled loss), matrix-free:

1. **Rectangular Arnoldi, k ≈ 100–200:** k HVPs (autograd Hessian-vector products through the real
   coupled loss at B=8–32) → the (k+1)×k Hessenberg `H̃_k`. Toh–Trefethen: `Λ_ε(H̃_k) ⊆ Λ_ε(M)` —
   the projected pseudospectrum is a **certified inner approximation** (no over-claim by
   construction). Run pseudopy/dense-SVD on the small `H̃_k` (trivial cost) for the ε-pseudospectrum
   over ε ∈ 10^{−4..0}·‖M‖.
2. **The four-number readout per boundary:**
   - `α` — spectral abscissa (what #318 already checks; asymptotic),
   - `ω` — **numerical abscissa** `λ_max((M+Mᵀ)/2)` (~20 Lanczos HVPs; the EXACT t→0⁺ growth rate:
     `d/dt‖δθ‖ ≤ ω‖δθ‖`, sharp) — **the ep110 signature is ω>0 ∧ α<0**,
   - `K` — Kreiss constant `sup_ε α_ε/ε` from the projected pseudospectrum (certified LOWER bound
     on peak transient amplification `sup_t‖e^{tM}‖ ≥ K`),
   - `σ_B(ω)` — dispersion of ω across ~8 minibatch draws (the FEED-06h stochastic axis a
     deterministic pseudospectrum structurally misses).
3. **Cost on M5 Max:** HVP ≈ 2× a gradient; ~(k+20+8·20) HVPs ≈ 10–25 min per boundary, CPU/MLX,
   $0. Non-normality certificate: `‖H̃H̃ᵀ−H̃ᵀH̃‖_F/‖H̃‖²` reported alongside.
4. **Retro-validation ($0, pre-registered — §E-4):** replay saved checkpoints bracketing the
   MEASURED ep300 bump and the ep108–114 window vs two quiescent boundaries. The instrument must
   separate them (elevated ω / K / σ_B at events, flat at quiescent) — else the non-normal-transient
   reading of FEED-chern is refuted at this formulation and the variance-only explanation stands.

This is buildable now (tools exist: MLX/torch HVPs; pseudopy for the dense small matrix), and it is
the FEED-chern "queued lever" made concrete. Equations leg for its LAW is **owed WITH the build**
(FEED-chern precedent) — not registered here.

---

## D. External math/OSS pulls (verified; licenses to confirm at pull-time)

1. **plenoptic** (Flatiron/Simoncelli lab; PyPI `plenoptic`) — PyTorch Portilla–Simoncelli model +
   **metamer synthesis against arbitrary frozen models**. Exactly the encode-side machinery for
   `t_c`: extract PS statistics per class, synthesize metamers, and (their core API) optimize a
   stimulus to match a frozen model's response — our "SegNet-legible texture" fit is a metamer
   problem. https://docs.plenoptic.org / https://pypi.org/project/plenoptic/ (MIT-family — verify).
2. **TetsuyaOdaka/texture-synthesis-portilla-simoncelli** + **SteerablePyramid** (numpy PS
   synthesis) — the DECODE-side candidate: scorer-free, seedable, deterministic; port under our
   numpy-fp32 reference discipline. https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli
   (license: check repo).
3. **pseudopy** (andrenarchy — Python EigTool; Trefethen–Embree algorithms) + **mpseudo**
   (parallel, mpmath precision) + **Pseudospectra.jl** (RalphAS) — for §C's dense small-matrix
   pseudospectra + Kreiss. Internal instrument only (not shipped) so any OSI license is fine;
   pseudopy license = check (GPL-family suspected). https://github.com/andrenarchy/pseudopy ·
   https://pypi.org/project/mpseudo/ · https://github.com/RalphAS/Pseudospectra.jl
4. **Semi-discrete OT / Laguerre fitting:** **geogram** (BrunoLevy; exact power diagrams + SDOT) ·
   **pysdot** (sd-ot) · **nyorem/sdot** · Bourne et al. **"Inverting Laguerre tessellations:
   recovering generators from cell volumes and centroids via OT"** (arXiv 2406.00871, ESAIM M2AN
   2025; MATLAB-SDOT + Laguerre-Polycrystalline-Microstructures repos) — the exact machinery to FIT
   v8's generators `G` to the GT partition (store-generators-not-boundaries, L-v8), including the
   mass-matched (AHA logit-offset) form. https://github.com/BrunoLevy/geogram ·
   https://github.com/sd-ot/pysdot · https://arxiv.org/abs/2406.00871
5. **Coding-for-machines / semantic-texture codecs (framing precedent, V2 originality check):**
   Satisfied Machine Ratio (arXiv 2211.06797) · Semantics-Guided Generative Image Compression
   (arXiv 2505.24015) · Scalable human+machine feature-fusion coding (arXiv 2405.09152) · Semantic
   compression with multimodal foundation models (arXiv 2509.05925) · ICM-with-SAM-edges (ICIP
   2024). All ship semantics and/or synthesize texture for a TASK model — **none makes the frozen
   scorer's OWN metamer statistics the counted payload with a deterministic decode-side
   synthesizer** (rule-118 split). V2 originality (L16) survives this sweep. **CITED.**
6. **Texture-bias lineage:** Geirhos et al. 2019 (ImageNet CNNs texture-biased) · Hermann et al.
   NeurIPS 2020 (origins of texture bias) · arXiv 2509.20234 (dissent: "local shape sensitivity,
   not texture bias"). Confirms the palette finding is generic CNN decision geometry; the dissent
   paper is a DESIGN CAVEAT for `t_c` — the legibility statistics may need mid-frequency local
   SHAPE structure (PS cross-scale phase terms carry some), not Gram-marginals alone. **CITED.**

---

## E. RANKED next measured rows (each with a pre-registered falsifier; P7)

1. **Chroma-HF texture-legibility probe ($0, CPU-torch, n600 through-R) — 384-BAND-DESIGNED per
   UNIT C.** On the flat-paint floor arm (4b, per-pair mean, d_seg 0.0416): add per-class texture
   in three arms — 384-plane luma-null chroma (pre-imaged through the exact `D` kernel, NOT a
   camera-res dither) / luma-HF only / both — at 2–3 amplitudes (seeded PS-noise or GT-texture
   transplant). *Falsifies A2-placement:* band-designed chroma fails to cut the floor ≥2× while
   luma succeeds ⇒ chroma-preferential placement REFUTED (T must pay the luma pose price; UNIT C's
   2.73e-3 authority = jitter-only, not basin legibility). *Also decides* the v8 carrier design.
   **Cheapest, most decisive.**
2. **PS-metamer per-class texture arm ($0–cheap, encode-side plenoptic fit + decode-side numpy
   synthesis, n600 through-R).** Fit `t_c` per class from GT-masked frames; synthesize; paint cells;
   measure realized d_seg vs 0.0416 and vs witness 0.0048; count quantized `|t_c|` bytes.
   *Falsifies A3-rate:* PS-textured paint fails to reach ≤0.010 (≥4× below floor) ⇒ "O(1KB)
   statistics suffice" REFUTED at this formulation ⇒ corrected floor's R(T) term rises / textured
   realization needs the trained-witness regime only. Either outcome pins `d_seg*(T)` tighter than
   the current 3.5-order bracket.
3. **Nonlinear flip-solve v2 ($0, CPU-torch, the 512-flip verify set + real-witness re-run).**
   True `∇m` backward + secant + cluster QP (§B). *Falsifies B:* verify ≤ 0.594 (no lift) refutes
   the missing-Jacobian explanation; ≥0.85 confirms and upgrades the flip-fix frontier into a
   deployable deterministic corrector (annulus half of the residual). Re-run the ledger on
   mod32cap ep650 for the honest interior-texture-dominated cheap fraction (#149 caveat).
4. **Pseudospectral retro-validation ($0, saved checkpoints).** Build §C's four-number instrument;
   run on ep300-bump + ep108–114 checkpoints vs two quiescent boundaries. *Falsifies C/FEED-chern:*
   no separation (ω, K, σ_B flat across events) ⇒ non-normal-transient reading refuted at this
   formulation; variance-only (FEED-06h) explanation stands. Separation ⇒ a canary-predicting
   launch gate the spectrum cannot supply.
5. **Semi-discrete OT generator fit (v8 support; $0 CPU).** Fit Laguerre generators to GT partitions
   per frame (pysdot/geogram; Bourne inversion for volumes+centroids); measure generator count vs
   Road/Lane residual coverage (40% off-curve). *Falsifies the power-diagram parametrization for
   lanes:* OT-fit generators fail to cut off-curve coverage below ~20% ⇒ lanes need curved charts
   (the band), generators insufficient — bounding R5's reachable share of the 0.074 enemy.

Ordering rationale: 1 gates both the placement law AND the v8 carrier; 2 pins the floor-dominating
scalar; 3–4 upgrade existing instruments; 5 feeds increment-2. Rows 1–2 jointly convert the
"texture-legibility gap" from the campaign's largest unknown into a measured curve.

---

## Verdict + scope

**verdict_scope: FORMULATION-level synthesis + characterisation — no kill, no score.** The unified
object (§0/§A) is DERIVED from measured anchors; its two load-bearing conjectures (chroma-HF
legibility; PS-statistics sufficiency) are pre-registered with falsifiers (§E-1/2). The flip step
law (§B) and instrument spec (§C) are DERIVED with their accuracy/validation rows owed. Honest
boundaries: sibling memos not landed; `d_seg*(T)` bracket spans 3.5 orders; every number here is
advisory — **pointer 0.19110 UNMOVED (means).**

## Triality legs
- **DAG:** FEED-fable-synthesis (appended this landing).
- **Equations:** `textured_power_diagram_sufficient_statistic_v1` +
  `scorer_obligation_matrix_factorization_v1` + `flip_margin_step_law_v1` (registered,
  `src/tac/canonical_equations/textured_power_diagram_20260710.py`; conjectures NOT registered —
  statuses honest per #284 discipline). §C's law owed WITH the instrument build.
- **DSL:** N/A (derivation/characterisation; no trainer lever — the §E rows route the next
  build-wave, one line appended to `DUAL_CHAIN_BRIEF_385_20260710.md`).
