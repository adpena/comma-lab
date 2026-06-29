# Graphics-AA ↔ Astronomy ↔ inverse-codec cross-check — existence proofs + draw-froms for v2

**UTC:** 20260629T184900Z · **evidence grade:** `[macOS research-signal]` (online+OSS+papers literature
mine; NO code run, NO measurement) · **score_claim:** false · **promotable:** false ·
**pointer:** contest-CPU **0.19110 UNMOVED**. Everything here is **MEANS** toward a byte-closed exact row,
not an end. No numbers in this memo are contest scores; the only measured numbers cited are from our own
prior `[macOS research-signal]` probe (F1 / `r_survival_probe`).

> **Mission framing.** This memo de-risks the v2 "witness" codec (SDS-TSC) by (a) finding the canonical
> graphics/astronomy literature that an **existence-proof cross-check** for our measured F1 SDF-survival
> claim, and (b) banking directly-transferable deterministic techniques for the **lane carrier** and the
> **margin-conditional residual coder**. Per the operator standing rule
> `terminal-conclusion-needs-existence-proof-crosscheck`: F1 ("R is interpolation-dominant; an SDF carrier
> survives R because bicubic/bilinear is exact on its 1-Lipschitz ramp") is exactly the kind of claim that
> a 20-year-old graphics result either confirms or breaks. **It confirms it** — with the same caveat F1
> already flagged.

**rule-118 legend (used per finding):** `FREE` = generic deterministic algorithm, legal to expand inside
`inflate.py` uncounted. `COUNTED` = learned or video-derived payload, must live in `archive.zip` and is
sized by the rate term. `FORBIDDEN` = a video-derived table smuggled into "code" to dodge the rate term.

**Scope of this revision:** Tasks 1–5 are the graphics-AA ↔ astronomy cross-check (SDF carrier + residual
coder + topology codec). Tasks 6–7 (added) are the **third v2 pillar** — the Hodge/Helmholtz pose-warp and
the level-set transport PDE — so the memo now covers all three pillars as **ONE continuous-PDE inverse
renderer**. The framing-lens spine (below) is the "why it coheres" pass.

---

## FRAMING LENS — the spine (phenomenology + causation; brief, grounded, flagged)

Two complementary statements of *why* an inverse-codec is the right object. Both are partly **our framing**
(flagged) over real technical cores (cited).

**(i) Phenomenological / dependent-origination (relational, not inherent).** The contest scores a
**task-relational** statistic (the frozen scorer's argmax/pose reading), never the inherent RGB. So the
optimal code stores the **relational dependent-arising statistic, not the inherent-pixel `svabhāva`** — the
witness is *empty of inherent pixels*; its only content is how it is read. **Technical core (cited):** this
is the **indirect / remote rate-distortion** ("coding for machines") statement — store a **sufficient
statistic** (Fisher–Neyman) for the scorer's decision, not the image. **Empirical content:** our prior F2
measured **~86% of the RGB Jacobian directions live in the scorer's Fisher nullspace** (discardable without
changing the reading) — the measured "emptiness" (the `[macOS research-signal]` anātman). **Flag:** the
Madhyamaka vocabulary (`svabhāva` / `anātman` / `pratītyasamutpāda`) is **our rhetorical lens**, not a cited
result; the cited cores are sufficient statistics + indirect-RD / information bottleneck (Tishby).

**(ii) Causal (intervention-equivalence).** The witness is the **minimal intervention-equivalent input under
the only intervention that matters** — the frozen-scorer reading. All inputs that produce the same scorer
output are **observationally/interventionally equivalent**: they form the **causal-equivalence fiber**, and
the codec is the **quotient by that fiber** (task #155 quotient codec; the level-set quotient of Task 7 —
all `φ` sharing a zero-crossing + argmax are one equivalence class). The **Fisher metric `G = E[JᵀFJ]`** is
the **local causal-sensitivity structure**: its **nullspace = acausally-empty directions** (interventions
the scorer cannot feel). **Technical core (cited):** interventional/observational equivalence (Pearl
do-calculus), sufficient statistics, and the Fisher metric as the natural metric on the model manifold
(Amari information geometry). **Flag:** "causal-equivalence fiber = quotient codec" is **our framing** over
those cited cores. The two lenses meet at one operational rule: **code the fiber-transversal sufficient
statistic (the small-margin / births-deaths boundary events); discard the nullspace.** That single rule is
what Tasks 1–7 each instantiate.

---

## TASK 1 — Existence proof: graphics AA validates the SDF lane carrier

**Canonical source.** Green, C. (Valve), *"Improved Alpha-Tested Magnification for Vector Textures and
Special Effects,"* ACM SIGGRAPH 2007 Courses, pp. 9–18, DOI 10.1145/1281500.1281665 (PDF on
realtimerendering.com / steamcdn). The founding paper of distance-field text/vector rendering. Companion:
the libGDX / mattfife explainers of the same technique.

**What the literature actually proves (vs folklore).**
- **PROVEN (the mechanism).** A distance field is a *smooth, continuous* signal that is **locally linear
  through the edge** (a true Euclidean SDF has `|∇φ|=1`). Bilinear/bicubic interpolation **reproduces a
  linear ramp exactly** (bilinear of an affine function is that affine function), so the **0.5-crossing
  (the recovered edge) is placed at sub-texel precision** regardless of the magnification factor. By
  contrast, interpolating a *hard bitmap* (a step) "just produces murky gray — no useful information about
  where the true edge is" (libGDX). This is **identically** our F1 §1.3 mechanism: R upsamples bicubic,
  the SDF's linear margin `m=φ_top1−φ_top2` crosses zero linearly, the zero-crossing reconstructs at
  sub-pixel precision. **Confidence: HIGH.** The graphics result is the 20-year-old existence proof for
  F1's measured `sdf@192 ≈ hard@384` headroom. `[rule-118: the mechanism is a property of the algorithm,
  not a payload — informational]`
- **FOLKLORE / where the guarantee is only LOCAL.** The "bilinear is exact on the SDF" guarantee holds
  **only where `|∇φ|` is ~constant**. Chlumský 2018 (already grounded in our `r_survival_physics` memo)
  states it precisely: SDF "interpolation provides accurate reconstruction only where the rate of change
  is more or less constant." A real SDF is only *piecewise* linear; the guarantee **breaks at gradient
  discontinuities** — the **medial axis**, **sharp corners**, and **where two edges meet**. So the clean
  "1-Lipschitz ⇒ interpolation-exact" story is rigorous *per-edge-segment* and **folklore if extrapolated
  to corners/junctions**.

**Documented failure modes — and do they match F1's caveat? YES, exactly.**
1. **Monochrome-only** (Green's stated drawback): one SDF channel can't carry arbitrary color. **Not our
   problem** — we don't reconstruct color, we take `argmax_k φ_k` of K per-class fields.
2. **Sharp corners round off** (single-channel SDF) — the Chlumský 2018 raison d'être. This is **F1's last
   holdout**: F1 §4.4 notes "the lane's dashes have corners; a per-class single SDF will round dash-ends,"
   and measured **lane flip 3.19% @ render-192 while every other class < 0.2%**. The corner-rounding
   failure mode of graphics SDF **is** the residual lane error in F1. **Match: exact.**
3. **Thin features below texture Nyquist vanish.** F1 §1.2 measured the same: a 2px lane goes sub-Nyquist
   once witness render res `r ≲ 192` (the cliff). **Match: exact.**

**The honest caveat F1 already carries, restated with graphics backing.** The classic "a level-set is
stable under smoothing" result (and the clean SDF-interpolation guarantee) is a **single-region** statement
(low-curvature boundary, shift ≈ ½·κ·σ²). Our metric is a **K-class argmax with a thin minority class (lane,
0.59% area, 2px, with corners) competing** — exactly where the single-region theorem weakens. F1 measured
this directly (heat-kernel SDF was *worse* than hard on the thin lane — §6b). So graphics **strengthens** the
SDF carrier choice **and pinpoints the residual to corners + thin-class competition** → which is precisely
what Task 2 (MSDF) attacks.

**Verdict (Task 1):** **Existence proof CONFIRMS the SDF lane carrier.** The graphics literature is the
independent, decades-old corroboration of F1's interpolation-exactness mechanism, and its documented failure
modes (corner rounding, sub-Nyquist thin features) are *identical* to F1's measured lane residual — not a
contradiction but a map to the next fix. `[rule-118: SDF rasterization in inflate.py = FREE; the stored lane
boundary descriptor = COUNTED]`

---

## TASK 2 — MSDF draw-from (sharp corners = lane dash-ends)

**Canonical source.** Chlumský, V. (2018), *"Improved Corners with Multi-Channel Signed Distance Fields,"*
Computer Graphics Forum 37(1), DOI 10.1111/cgf.13265; OSS `github.com/Chlumsky/msdfgen` (+ Master's thesis,
CTU Prague 2015). Verified shader idiom (awesome-msdf): `median(r,g,b) = max(min(r,g), min(max(r,g), b))`.

**The algorithm (extracted).** Store **three signed-distance channels** R,G,B. Each channel encodes the
distance to a **subset of the contour's edges** (the edges are 2-colored along the outline so that any two
edges meeting at a sharp corner land in *different* channels). At reconstruction, the per-pixel distance is
the **median of the three channels**. Why the median recovers a corner: a corner is the **intersection of
two half-planes** (two edges). Near the corner, two channels carry the two edges' distances and the third
is the "odd one out"; the **median selects the consensus**, so the reconstructed iso-line follows the *sharp
intersection* instead of the rounded single-channel distance. A single SDF, lacking the second edge, rounds
the corner. **Confidence: HIGH (this is the established, deployed result for font corners).**

**rule-118 classification.** The MSDF **generation** (edge coloring + per-channel distance rasterization +
median reconstruction) is a **deterministic geometric algorithm operating on a vector outline, with NO
machine-learning / data-driven component** (confirmed: "all assets generated by the msdfgen utility … no
ML"). → **FREE in inflate.py.** What is **COUNTED** is the stored **lane boundary descriptor** (the vector
contour / control points of the lane geometry — video-derived). A minimal-dependency deterministic MSDF
rasterizer in `inflate.py` needs only: (i) an edge list with per-edge channel assignment, (ii) per-pixel
per-channel min signed distance, (iii) median combine, (iv) ramp→membership. All numpy-expressible, ≪100
LOC. **`[FREE algorithm + COUNTED descriptor]`. NOT FORBIDDEN** as long as the descriptor is an honest
compact contour, not a per-frame argmax table hidden as "code."

**Interaction with our argmax-after-R metric (the honest complication).** MSDF's median trick was designed
for a **single 2-region field recovered by ONE threshold** (glyph inside/outside). Our metric is a **K-class
argmax (top-2 margin)**, not a single threshold. So MSDF does **not** drop in trivially. Two viable adapt-
ations: (a) apply MSDF **only to the lane class's boundary** and feed the corner-preserved lane distance into
the K-class argmax; or (b) per-class multi-channel encoding. Option (a) is the cheap, targeted fix for the
named residual. **Whether MSDF's corner gain survives R + the K-class argmax is UNMEASURED** — graphics
proves it in the glyph-threshold setting; transfer to argmax-after-R is *plausible, not proven*.
**Confidence: MEDIUM.**

**What this CHANGES in the v2 build (actionable).** Extend `tools/r_survival_probe.py` with an **MSDF lane
carrier** and measure lane-flip% @ render-192 against the single-SDF baseline **3.19%** (F1 §2.1). If MSDF
materially cuts the dash-end flips, the lane carrier becomes: *MSDF-encoded lane contour (tiny COUNTED
descriptor) + FREE deterministic MSDF rasterizer in inflate.py.* This is the named attack on F1's "last
holdout" (the thin lane with corners), the ~8-dim lane-orbit long-tail localized to sub-wall (B).

---

## TASK 3 — Astronomy difference imaging ↔ margin-conditional residual coder

**Canonical sources.** Alard & Lupton (1998), *"A Method for Optimal Image Subtraction,"* ApJ 503, 325 —
**OIS**: solve a least-squares **convolution kernel** (basis of Gaussians × polynomials, spatially varying)
to **PSF-match** a reference to a new image, then subtract → code only the residual. Zackay, Ofek & Gal-Yam
(2016), *"Proper Image Subtraction—Optimal Transient Detection, Photometry, and Hypothesis Testing,"* ApJ
830, 27, **arXiv:1601.02655** — **ZOGY**: closed-form, statistically-optimal Fourier-space difference `D`
and matched-filter detection score `S`, **proven optimal in the background-dominated limit**. OSS: HOTPANTS
(Becker, the Alard-Lupton implementation), ISIS (Alard), `github.com/pmvreeswijk/ZOGY`.

**Structural transfer (direct).** The astronomy pattern is *align reference by motion → PSF-match →
subtract → code only the residual*. The v2 pattern is *warp the canonical static scene by ego-pose → (skip
PSF) → subtract from target → code only the lane-survival residual*. The mapping is 1:1:
- reference image ↔ **canonical static-scene generator** (FREE, deterministic);
- registration/alignment ↔ **stratified per-class ego-pose WARP** (Road=ground-homography, sky=rotation,
  hood=identity) — we already do this;
- residual ↔ **the ONE learned term: the lane-survival residual through R** (the binding path).

**The decisive insight (why our problem is the EASY version of astronomy's).** ZOGY's headline guarantee:
*for accurately registered, adequately sampled images, proper subtraction leaves **NO deconvolution /
subtraction artifacts.*** In astronomy the hard part is that reference and new image have **different seeing
(different PSFs)** — hence the fitted convolution kernel (Alard-Lupton) or the Fourier PSF terms (ZOGY).
**F1 proved we DON'T have that problem:** R is **interpolation-exact**, not a diffusive PSF (the heat-kernel
framing was *falsified* — §6b of `r_survival_physics`). Our "reference" (deterministic render) and "new"
(witness through R) pass through the **same** interpolation operator R on the **same** grid → **there is no
PSF mismatch to fit.** So we **legitimately SKIP the PSF-matching / deconvolution step** that is the bulk of
OIS/ZOGY machinery → a **cleaner, smaller residual**. ZOGY's artifact-free precondition ("accurately
registered + adequately sampled") is **exactly our regime** when (i) render res ≥ 192 (adequately sampled,
F1 Nyquist) and (ii) the ego-pose warp is accurate (accurately registered). **Confidence: HIGH** that the
skip is justified; **the residual-inflating risk migrates entirely to REGISTRATION accuracy** (ego-pose
warp error), which is exactly where ZOGY's "resilient to registration errors" extension would be the hedge
if needed. → **Invest bytes/compute in warp accuracy, NOT in a PSF kernel.**

**Optimal-coding insight (the matched filter / ZOGY `S` statistic → bit allocation).** The matched filter
(North 1943; Turin 1960; the LIGO detection statistic; ZOGY's `S`) is the **optimal detector of a known-shape
signal in (background-dominated Gaussian) noise**: correlate the residual with the known template, normalized
by the local noise. For a **rare-event residual against a known template**, this says: **the information is
concentrated where the matched-filter score is large** — i.e. where a flip is *detectable*. For us the
"flip-detectability" score is the **SegNet decision margin** `m = φ_top1−φ_top2`: the d_seg test statistic
is dominated by **small-margin boundary pixels (the annulus)**. So the principled **margin-conditional
residual coder** = *allocate residual bits ∝ the matched-filter / margin statistic*, concentrating the COUNTED
payload on the small-margin lane-boundary annulus and spending ≈0 in the high-margin interior. This formalizes
"margin-conditional residual coder" with an optimality argument, not a heuristic. **Confidence: MEDIUM-HIGH**
(the matched-filter optimality is exact for Gaussian noise / known template; our "noise" is the R+SegNet flip
process, an approximation of that model).

**rule-118 classification.** subtraction + matched-filter/margin SCORING + residual DECODE = **FREE** generic
algorithm in inflate.py. The deterministic template (canonical render + ego-pose warp) = **FREE**. The coded
**lane-survival residual payload** (video-derived) = **COUNTED**. No table smuggling.

**What this CHANGES in the v2 build (actionable).** Build the residual coder as **register → subtract →
code-residual**, but **omit PSF-matching/deconvolution** (justified by F1's interpolation-exactness). Allocate
the residual bit budget by a **margin/matched-filter weight** so bits land on the small-margin annulus. Treat
**ego-pose warp accuracy** as the primary residual-size lever (the ZOGY "accurately registered" precondition).

---

## TASK 4 — Other directly-transferable deterministic techniques (ranked)

Ranked by transfer value to the **lane carrier** + **residual coder**. rule-118 status per item.

1. **Matched filtering (North/Turin; LIGO; ZOGY `S`)** — **HIGH.** The optimal-detection principle for the
   residual coder's bit allocation (folded into Task 3). `[FREE algorithm; COUNTED residual]`. The single
   most principled knob for *where* to spend residual bits.
2. **MSDF corner recovery (Task 2)** — **HIGH** for the lane carrier specifically (the named dash-end fix).
   `[FREE rasterizer + COUNTED contour]`. UNMEASURED in argmax-after-R → next probe.
3. **Coverage / analytic anti-aliasing + MSAA (edge-only supersampling)** — **MEDIUM-HIGH.** Analytic/
   coverage AA computes **fractional pixel coverage at edges** — this *is* F1's measured **wide ramp**
   (half-width ≳5px, saturating; F1 §2.3), and "conservative rasterization" (never quantize a thin lane's
   coverage to 0) is the principled cure for the sub-Nyquist lane. MSAA's *edge-only* supersampling = the
   **spatial localization** principle for the residual coder (spend resolution/bits on the boundary annulus
   only). `[FREE: coverage computation]`. Directly reinforces the wide-ramp + annulus-localization spec.
4. **DLSS Frame-Gen + TAA temporal reprojection / motion-vector warp (NVIDIA; DLSS 2 = TAAU)** — **MEDIUM.**
   This is **the forward of our inverse codec**: DLSS-FG generates intermediate frames from 2 frames +
   optical-flow + motion-vectors + depth; TAA accumulates **sub-pixel-jittered** samples across frames using
   motion vectors to **beat single-frame Nyquist** (temporal supersampling). Our *warp-canonical-by-ego-pose
   + per-pair-generate + temporal-delta* is the **deterministic inverse** — we *know* the motion (ego-pose)
   and the scene (canonical), so we generate deterministically what DLSS estimates. **Draw-from:** the
   TAA/DLSS *jittered temporal accumulation* idea = render the 2 frames of a pair at sub-pixel offsets and
   accumulate to resolve the thin lane below single-frame Nyquist. **Caveat:** our metric scores **per-frame
   argmax** and a pair is only **2 frames** → very limited accumulation depth; transfer is conceptual, the
   *warp* is the reusable generic primitive. `[FREE: motion-vector reprojection/warp; the DLSS NETWORK
   weights would be COUNTED and we would NOT ship them]`. Validates the warp+generate paradigm as the
   established graphics forward model.
5. **SExtractor background+threshold+deblend (Bertin & Arnouts 1996, A&AS 117, 393)** — **MEDIUM.** The
   segmentation analog: **background** = bicubic-spline of a gridded low-res mode (≈ our smooth deterministic
   scene); **detection** = threshold above background (≈ argmax-above-margin); **deblend** = multi-threshold
   merge tree (≈ superlevel-set filtration — bridges to Task 5). Draw-from: *background-model → subtract →
   threshold/partition* structure, and a principled deblend. `[FREE algorithm]`. Lower direct value because
   our partition is given by the frozen SegNet, not re-derived — but the merge-tree structure is the Task-5
   bridge.

---

## TASK 5 — The convergence audit (verified, not asserted)

**The claim under audit:** are these *literally the same math* — (a) persistent homology / sublevel-set
filtration of the **margin field** (our Morse-Smale codec, task #180), (b) astronomical transient/deblend
detection, (c) the fluid **Kelvin-circulation invariant** (conserved except at vortex reconnection = topology
change), and (d) "irreducible information = the topological events (births/deaths of partition components)"?

**Verdict: (a)=(b)=(d) are a LITERAL mathematical identity. (c) is a STRUCTURAL ANALOGY, not the same
theorem.** Honest split, per NO-FAKE and "verify positives too."

**The precise unifying statement (a=b=d).** Let `f` be a scalar field on a domain (for astronomy `f` = flux
or density; for us `f` = SegNet decision margin `m = φ_top1 − φ_top2`, or a per-class logit). Consider its
**superlevel-set filtration** `{f ≥ t}` as `t` descends. **Morse theory** (Milnor 1963) says the topology of
these sets changes **only at critical points of `f`**: a **maximum births** a connected component, a **saddle
merges** two components (a death). The **persistence diagram** (Edelsbrunner-Letscher-Zomorodian 2002,
*Topological Persistence and Simplification*) is precisely the catalog of these **(birth, death)** critical-
point pairs; **persistence = death − birth = the "prominence"** of a feature.
- **Astronomy is literally this construction.** DRUID (2024, **arXiv:2410.22508**, *"Source Detection and
  Deblending … with Persistent Homology"*) is explicit: filtration from highest to lowest intensity; **"birth
  = the brightest pixel of the component," "death = the trough/saddle where two complexes meet."** SExtractor's
  multi-thresholding deblend (Bertin-Arnouts 1996) builds the **same merge tree** ("converts the light into
  trees, branches = bright areas within fainter objects") — that *is* the 0-dim persistence of the superlevel
  filtration. DisPerSE (Sousbie 2011, MNRAS 414, 350, **arXiv:1009.4015**) builds the **discrete Morse-Smale
  complex** of the cosmic density field (filaments = ascending 1-manifolds joining maxima through saddles) and
  denoises by **persistence**. So {our margin-field codec, astro deblend, cosmic-web skeleton} are **three
  instances of one construction** with different `f`. **Confidence: HIGH — this is an identity, with primary
  sources.**
- **"Irreducible info = topological events" (d) follows.** By Morse theory the **partition (argmax) topology
  is fully determined by the critical points of the margin field and their Morse-Smale connectivity**; the
  persistence diagram is a **minimal sufficient statistic for that topology.** The **low-persistence pairs are
  the small-margin, flip-prone features** — exactly the d_seg-binding set. So *coding the topological events
  (persistence diagram) + deterministic Morse-Smale reconstruction* is a legitimate **indirect-RD sufficient
  statistic** for the partition. **This strengthens task #180. Confidence: HIGH for the topology; MEDIUM with
  a caveat below.**

**(c) Kelvin circulation — analogy, NOT identity (flagged).** Kelvin's theorem (Thomson 1869; Helmholtz vortex
theorems): circulation `Γ = ∮ v·dl` around a *material* loop is conserved (`dΓ/dt = 0`) for ideal/barotropic
flow, and **breaks only at viscous vortex reconnection — a topology change of the vortex lines**. The shared
*theme* with persistence is genuine: **"a quantity is invariant/smooth except at discrete topological events
(reconnection ↔ births/deaths)."** But Kelvin circulation is a **conservation law on a smooth time-evolving
flow**, whereas persistence is a **static filtration catalog of one field**. They are **not the same theorem**;
equating them would be a NO-FAKE violation. The honest unification is a **Morse-theoretic meta-principle**:
*topology changes are the discrete events that carry the irreducible information / break the otherwise-smooth
invariant* — realized as a **literal identity** for {persistence, astro deblend, cosmic web, our codec} and as
an **analogy** for Kelvin reconnection. **Confidence: HIGH that it is only an analogy.**

**The caveat on (d) (NOT-PESSIMISTIC, but honest).** Persistence catalogs the topology of **level sets** of a
*fixed* field; but `d_seg` depends on the **geometry of the zero-margin crossing** (sub-pixel placement under
R — F1's whole point), not only on which components exist. So the persistence diagram is a **necessary**
structural sufficient-statistic for the partition's topology but **not the whole story** — the geometry of the
boundary (where exactly the zero-crossing lands through R) must be coded too (that is what the SDF/MSDF carrier
+ the margin-conditional residual handle). **Topology (persistence) and geometry (SDF zero-crossing) are
complementary halves of the sufficient statistic.** This is the clean way the Task-5 codec and the
Task-1/2/3 carrier+residual compose, with no double-counting and no signal loss.

**What this CHANGES in the v2 build (actionable).** Task #180's Morse-Smale codec is on solid, primary-source
footing: **code the persistence diagram of the margin field (the topological events; low-persistence pairs =
the flip-prone binding set) — COUNTED, tiny — and reconstruct the partition topology via a deterministic
Morse-Smale expansion in inflate.py — FREE.** Pair it with the SDF/MSDF carrier (geometry half) and the
margin-conditional residual (Task 3). The astronomy OSS (DRUID, DisPerSE) are reference implementations of the
*algorithm* (FREE), reusable as design oracles. `[rule-118: persistence/Morse-Smale algorithm = FREE; the
persistence-diagram of the actual video-derived margin field = COUNTED]`.

---

## TASK 6 — 4D-fluid Hodge/Helmholtz parameterization of the pose-warp (the THIRD pillar)

**Canonical sources.** Helmholtz–Hodge decomposition (HHD): any vector field = **curl-free** (gradient of a
scalar potential = the divergent/source part) + **divergence-free** (curl of a vector potential = the
rotational/solenoidal part) + a **harmonic** remainder. Applied to ego optical flow: *Spatial Reasoning for
Robot Navigation Using the Helmholtz–Hodge Decomposition of Omnidirectional Optical Flow* (IEEE, 2010). The
ego-flow physics: **Longuet-Higgins & Prazdny 1980,** *"The interpretation of a moving retinal image,"* Proc.
R. Soc. Lond. B 208, 385–397 — the canonical split of the motion field into a **translational** term and a
**rotational** term. Lie/geometric-algebra fact: `so(4) ≅ so(3) ⊕ so(3)` (the unique low-dim exceptional
splitting); a 4D rotation generator is a **bivector / 2-form with C(4,2)=6 components**; in 3D the rotation
2-form is Hodge-dual to the axis vector (so(3) ≅ ℝ³ — why Rodrigues axis-angle works).

**The claim, verified with a deep-math caveat.** The Longuet-Higgins–Prazdny field is
`v(x) = (1/Z)·A(x)·t + B(x)·ω`, where `t` = camera translation, `ω` = angular velocity, `Z` = scene depth.
Two structural facts are **EXACT** and ground the operator's depth-stratified warp:
- **Translational term scales as `1/Z`** → it is the **dominant, depth-modulated** part on near surfaces
  (Road, small `Z`) and **vanishes as `Z → ∞`** (horizon/sky). Pure forward translation gives a **radial
  (curl-free / pure-divergence) zoom around the focus of expansion** — the "divergence = zoom from forward
  translation" claim.
- **Rotational term `B(x)·ω` is depth-INDEPENDENT** → it is the **only part that survives at `Z → ∞`**
  (sky). Hence **sky = rotation-only `H = K R K⁻¹`** is *exactly* the `d → ∞` (or `t/Z → 0`) limit of the
  plane-induced homography `H = K(R − t nᵀ/d)K⁻¹` — and that is precisely what `tools/measure_pose_warp_dseg.py`
  already encodes (`pose_to_homography`, the `t nᵀ/d` term → 0). **Hood = identity / harmonic** (rigidly
  camera-attached, zero relative flow). **Confidence: HIGH** that the *depth-stratification* is the correct,
  cited physics.
- **CAVEAT (NO-FAKE, the deep-math honesty).** The clean mapping *divergence ⇔ translation, curl ⇔ rotation*
  is **EXACT only for (forward translation → curl-free radial) and (roll about the optical axis →
  divergence-free rotation)**. **Yaw/pitch** produce image-plane fields with **both** divergence and curl
  components, so HHD's div/curl split is **not identical** to the translation/rotation split for general
  motion — it is an *approximation* that is exact at the two extremes and on the two depth limits (`Z` small,
  `Z → ∞`). The **depth-dependence facts above are exact regardless**; it is only the div↔translation /
  curl↔rotation *labeling* that is approximate for yaw/pitch. **Flag this; do not over-claim the HHD
  identity.**

**Compressibility — the real win (quantified, structural).** Per-class homographies cost, per pair, roughly:
Road = full plane-homography (~8 DOF), sky = rotation-only (3), hood = identity (0) ≈ **~11 params/pair →
600 pairs ≈ 6,600 params**. The Hodge/screw parameterization instead stores **per pair only the 6-DOF ego
twist `(t, ω) ∈ se(3)`** — which is **already the stored-pose sidecar** (the d_pose targets ARE 6 PoseNet
scalars; the warp reuses them at **≈0 marginal bytes**) — **plus a STATIC scene descriptor stored ONCE for
the whole clip** (road plane normal+depth `(n, d)`, sky-at-∞, hood mask) ≈ **a handful of params total**. The
per-class warps are then **DERIVED** from `(t, ω)` + static geometry via the Longuet-Higgins–Prazdny /
plane-homography formula. So the warp is **not new payload at all** — it is a **FREE deterministic expansion**
of the already-stored ego-pose + a tiny static descriptor. Net: **~6,600 per-pair warp params → ~0 marginal
(reuse pose sidecar) + O(few) static.** **Confidence: MEDIUM-HIGH on the structure; the absolute byte win
depends on the pose sidecar already being stored (it is, Quantizr-style).**

**Bivector / Lie-so(3) parameterization — what it buys.** (1) **Minimal coordinates:** rotation has 3 DOF
(so(3) is 3-dim) → 3 numbers (axis-angle / bivector / quaternion-imaginary) vs 9 matrix entries with 6
orthonormality constraints — fewer bytes and **no off-manifold drift** (the `expmap` stays exactly on SO(3);
`tools/measure_pose_warp_dseg.py::_expmap_so3` already does Rodrigues). **Numerical-robustness: HIGH
confidence** (matrix renormalization drift is a well-known failure; axis-angle/quaternion avoid it). (2)
**Shared across classes:** because rotation is depth-independent, the **one** `ω` bivector serves **all**
classes (store once) — only the depth/plane selector differs. (3) **Composability** via the Lie bracket /
Baker–Campbell–Hausdorff (twists compose) — useful if multi-frame warp chaining is added. The absolute byte
saving from 3-vs-more on `ω` is small; the **structural win is reuse + robustness, not raw count.**

**rule-118.** Longuet-Higgins–Prazdny flow + plane-homography + `expmap` = **FREE** generic algorithm in
inflate.py. The **6-DOF ego-pose stream** = **COUNTED** (but already the existing pose sidecar — not new).
The **static scene descriptor** `(n, d, hood-mask)` = **COUNTED but tiny** (video-derived geometry, stored
once). **NOT FORBIDDEN** as long as the descriptor is honest geometry, not a per-frame warp table as "code."

**$0-measurable — concrete extension of `tools/measure_pose_warp_dseg.py`.** The tool already: fits
`(s_t, s_r, pitch)` calibration on road classes (`fit_calibration`), builds `pose_to_homography`, and reports
**per-class d_seg** for `warp` vs `persist`. Extension (**"Hodge/screw mode"**, $0, n=96 cached `L*`):
1. **Single-twist derivation:** derive ALL per-class warps from the **one** 6-DOF `(t, ω)` per pair +
   static `(n, d)`: Road = full `K(R − t nᵀ/d)K⁻¹`; Sky = `K R K⁻¹` (the `d → ∞` limit, **t-term dropped**);
   Hood = `I`. Measure per-class d_seg and compare to **independently-fit per-class homographies** — if
   equal, the single-twist compression is **d_seg-free** (the headline falsifier).
2. **Sky divergence-null test:** add the translational (`t`-) term back on the **sky** class and confirm it
   **HURTS** (predicts the depth-independence: translation flow must vanish at infinity). A clean
   sign-of-effect falsifier for the physics.
3. **Roll-vs-yaw HHD check:** decompose `ω` into roll (optical-axis) vs yaw/pitch and verify the div/curl
   labeling is clean for roll, mixed for yaw — quantifying the CAVEAT above.
**Confidence: HIGH that this is $0-measurable today** (the tool is ~90% there; the extension is a derivation
mode + two ablations).

---

## TASK 7 — Level-set transport + eikonal PDE (the continuous form that unifies Tasks 3 & 6)

**Canonical sources.** Osher & Sethian 1988, *"Fronts Propagating with Curvature-Dependent Speed: Algorithms
Based on Hamilton–Jacobi Formulations,"* J. Comput. Phys. 79, 12–49 — the **level-set method**: a front is the
**zero level set** of `φ`; it moves by the **transport/advection PDE** `∂φ/∂t + V·∇φ = 0`; `φ` is kept a
signed distance by **eikonal reinitialization** `|∇φ| = 1` (Fast Marching / Fast Sweeping, Sethian; viscosity
solutions). Boundary segmentation level sets (Chan–Vese 2001, *Active Contours Without Edges*) and
optical-flow-advected segmentation level sets are the task/semantic precedent.

**Is the formulation sound? YES.** "Maintain an SDF (eikonal reinit) while advecting its zero level set by a
velocity field (transport)" **is literally the Osher–Sethian algorithm** — canonical, well-posed (viscosity
solutions), with mature OSS (scikit-fmm, the FMM/FSM family). Our three pillars **compose into exactly this
one system, with no new machinery invented**:
- **SDF/MSDF carrier (Tasks 1–2)** = the `φ` representation + the eikonal `|∇φ|=1` constraint (the carrier
  IS a reinitialized level-set function).
- **Hodge/screw warp (Task 6)** = the **advection velocity `V`** = the ego-flow derived from the stored pose
  (`V = (1/Z)A t + B ω`). Transport `∂φ/∂t + V·∇φ = 0` carries the previous frame's lane SDF forward to
  **predict** the next frame's lane SDF.
- **Residual coder (Task 3)** = the **subtraction**: code only where the PDE-predicted front diverges from
  the true front. **This is Alard–Lupton difference imaging in continuous-PDE form** — the transport PDE is
  the "registration / forward-prediction of the reference," the residual is the "image subtraction."

**Does it make the lane residual SMALLER? Plausible, and it is the SAME claim as Tasks 3 & 5 (UNMEASURED).**
If the front is **mostly PDE-predicted** (advect prior lane SDF by the ego-flow), the residual lives **only**
where prediction fails: (a) **topological births/deaths** (new dash segments entering, occlusion changes) and
(b) **model error** (movables, calibration/registration error). This is **exactly** the convergence spine:
- *"code only at births/deaths"* (Task 7 transport residual)
- = *"code the topological events / low-persistence pairs"* (Task 5 persistence)
- = *"code only the transient residual against the template"* (Task 3 astronomy / matched filter).
Three independent fields say **the deterministic predictor (transport/warp/template) explains the bulk; the
COUNTED payload is the topological-event residual.** Temporal coherence (lane geometry is ~rigid under
ego-motion) is why the transport prediction should be good and the residual small. **Confidence: MEDIUM** —
the formulation is sound and well-posed; the residual-shrink is a *physically-motivated, cross-field-consistent
prediction that is not yet measured.* The falsifier: extend the warp probe (Task 6 #1) to a **2-step
transport** (advect lane SDF p→p+1, measure residual lane-flip% vs the persist/static baselines).

**rule-118.** Level-set transport + eikonal reinit + FMM/FSM = **FREE** generic PDE solver in inflate.py
(deterministic, viscosity-solution, decode-time compute is cheap and within the 30-min budget). The **velocity
`V`** comes from the FREE warp (Task 6) + COUNTED-but-existing pose. The **coded births/deaths residual** =
**COUNTED** (video-derived, but minimized to the topological events). The whole v2 codec is then:
*FREE level-set transport renderer driven by the (existing) pose sidecar + a tiny static scene descriptor,
plus a small COUNTED births/deaths lane residual.*

---

## Consolidated actionables for the v2 build (means → the next byte-closed exact row)

1. **Lane carrier (Task 1+2):** keep the 1-Lipschitz SDF (graphics existence-proof CONFIRMS it), **add an
   MSDF (multi-channel) encoding for the lane class** to preserve dash-end corners; **measure** MSDF-lane
   flip% @192 vs single-SDF **3.19%** by extending `tools/r_survival_probe.py`. FREE rasterizer + COUNTED
   compact contour.
2. **Residual coder (Task 3):** structure = **register(ego-pose warp) → subtract → code residual**, **SKIP
   PSF-matching** (F1: R is interpolation-exact, so ZOGY's artifact-free precondition already holds at
   render ≥192). **Allocate residual bits by a matched-filter / margin weight** → bits land on the
   small-margin lane annulus. Make **warp accuracy** the primary residual-size lever.
3. **Topology codec (Task 5):** task #180 confirmed as a literal instance of astronomical persistent-homology
   deblending — code the **persistence diagram of the margin field** (topological events) + deterministic
   Morse-Smale reconstruction; compose with the SDF/MSDF geometry half (no double-count).
4. **Annulus localization (Task 4):** coverage-AA + MSAA-edge-only confirm the **wide-ramp + boundary-only
   bit-spend** spec from two independent fields.
5. **Pose-warp as Hodge/screw (Task 6) — the highest-leverage parameterization change:** stop storing
   per-class homographies; store the **single 6-DOF ego twist `(t, ω)`** (reuse the existing pose sidecar,
   ≈0 marginal bytes) + a **tiny static scene descriptor `(n, d, hood-mask)`**, and **derive** all per-class
   warps by the depth-stratified Longuet-Higgins–Prazdny / plane-homography formula (FREE in inflate.py).
   **$0 falsifier:** add "Hodge/screw mode" to `tools/measure_pose_warp_dseg.py` (single-twist-derived
   per-class warps vs independently-fit; sky-divergence-null ablation).
6. **Unify as ONE level-set transport renderer (Task 7):** SDF/MSDF carrier = `φ`; Hodge warp = advection
   `V`; residual = births/deaths subtraction. The whole codec = *FREE level-set transport driven by the
   (existing) pose + tiny static descriptor, + small COUNTED births/deaths lane residual.* **$0 falsifier:**
   2-step transport residual probe (advect lane SDF p→p+1, measure residual lane-flip% vs persist/static).

## Honest caveats / NO-FAKE
- Pointer **0.19110 unmoved**; nothing here is a score. All cross-checks are literature + our own prior
  `[macOS research-signal]` F1 probe. The MSDF-in-argmax-after-R gain and the persistence-codec rate are
  **UNMEASURED** — the named next $0 probe (extend `r_survival_probe`) is the falsifier.
- The graphics SDF guarantee and the single-region level-set-stability theorem are **local**; the binding
  case is the **thin minority lane class with corners in K-class argmax**, exactly where they weaken — which
  is why MSDF (corners) + residual (geometry) + persistence (topology) are needed *together*, not any one
  alone.
- The Kelvin-circulation unification is an **analogy**, not an identity — explicitly flagged so it is never
  cited as proof.
- Task 6: the *div↔translation / curl↔rotation* HHD labeling is **exact only for forward-translation and
  roll** (and on the `Z`-small / `Z→∞` limits); yaw/pitch mix the two — flagged. The **depth-stratification
  itself (translation ~1/Z, rotation depth-independent) is exact** (Longuet-Higgins–Prazdny). The compress-
  ibility win assumes the pose sidecar is already stored (it is).
- Tasks 6–7 residual-shrink is **UNMEASURED**; the two named $0 probes (Hodge/screw mode + 2-step transport)
  on `tools/measure_pose_warp_dseg.py` are the falsifiers.
- The phenomenology/causation framing-lens vocabulary (Madhyamaka terms; "causal-equivalence fiber =
  quotient codec") is **our framing** over cited cores (sufficient statistics, indirect-RD/IB, Pearl
  interventional equivalence, Amari Fisher metric) — flagged, never cited as a derived result.

## Wire-in hooks (Catalog #125)
1. sensitivity-map: ACTIVE — corner-rounding (MSDF) and registration-error (residual) are new per-axis
   sensitivity rows for the lane. 2. Pareto: ACTIVE — MSDF contour bytes ↔ lane-flip reduction is a
   rate↔distortion arm; persistence-diagram bytes ↔ topology fidelity is another. 3. bit-allocator: ACTIVE —
   "allocate residual bits by matched-filter/margin weight on the annulus." 4. cathedral autopilot: N/A
   (literature memo, not archive-deployable). 5. continual-learning: this memo + the DAG FEED. 6.
   probe-disambiguator: three named $0 probes are the disambiguators — MSDF lane carrier (extend
   `tools/r_survival_probe.py`, Task 2); Hodge/screw warp mode + sky-divergence-null ablation (extend
   `tools/measure_pose_warp_dseg.py`, Task 6); 2-step level-set transport residual (Task 7).

## Primary citations
- Green, C. 2007. *Improved Alpha-Tested Magnification for Vector Textures and Special Effects.* ACM SIGGRAPH
  2007 Courses 9–18. DOI 10.1145/1281500.1281665.
- Chlumský, V. 2018. *Improved Corners with Multi-Channel Signed Distance Fields.* Computer Graphics Forum
  37(1). DOI 10.1111/cgf.13265. OSS: github.com/Chlumsky/msdfgen (Master's thesis, CTU Prague 2015).
- Alard, C. & Lupton, R.H. 1998. *A Method for Optimal Image Subtraction.* ApJ 503, 325. OSS: HOTPANTS,
  ISIS.
- Zackay, B., Ofek, E.O. & Gal-Yam, A. 2016. *Proper Image Subtraction—Optimal Transient Detection,
  Photometry, and Hypothesis Testing.* ApJ 830, 27. arXiv:1601.02655. OSS: github.com/pmvreeswijk/ZOGY.
- Bertin, E. & Arnouts, S. 1996. *SExtractor: Software for source extraction.* A&AS 117, 393.
- Sousbie, T. 2011. *The persistent cosmic web and its filamentary structure – I.* MNRAS 414, 350.
  arXiv:1009.4015. OSS: DisPerSE.
- DRUID 2024. *Source Detection and Deblending in Astronomical Images with Persistent Homology.*
  arXiv:2410.22508.
- Edelsbrunner, H., Letscher, D. & Zomorodian, A. 2002. *Topological Persistence and Simplification.*
  Discrete Comput. Geom. 28, 511. Morse theory: Milnor, J. 1963. *Morse Theory.*
- North, D.O. 1943 (matched filter); Turin, G.L. 1960. *An Introduction to Matched Filters.* IRE Trans.
  Information Theory.
- NVIDIA DLSS 3 / Frame Generation (GeForce technical docs); DLSS 2 = TAAU. Kelvin/Thomson 1869 *On Vortex
  Motion*; Helmholtz vortex theorems; viscous vortex reconnection.
- Longuet-Higgins, H.C. & Prazdny, K. 1980. *The interpretation of a moving retinal image.* Proc. R. Soc.
  Lond. B 208, 385–397. Helmholtz–Hodge decomposition of ego optical flow (IEEE 2010, omnidirectional-flow
  robot navigation). `so(4) ≅ so(3) ⊕ so(3)`; rotations as bivectors (geometric algebra; Lie theory).
- Osher, S. & Sethian, J.A. 1988. *Fronts Propagating with Curvature-Dependent Speed: Algorithms Based on
  Hamilton–Jacobi Formulations.* J. Comput. Phys. 79, 12–49. Chan, T. & Vese, L. 2001. *Active Contours
  Without Edges.* IEEE TIP 10(2). Fast Marching/Fast Sweeping eikonal solvers (Sethian); OSS scikit-fmm.
- Framing-lens cores: Fisher–Neyman sufficient statistics; Tishby et al. information bottleneck / indirect
  rate-distortion; Pearl interventional (do-calculus) equivalence; Amari information geometry (Fisher metric
  on the model manifold).
