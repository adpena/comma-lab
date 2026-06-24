# The d_seg source as a MANIFOLD — topology × manifold × rate-distortion × Morse × information-geometry (deep synthesis, 2026-06-23)

**Source:** operator 2026-06-23 — *"Love the topology analysis integrate with manifold and other deep
math analysis and extend and expand and deepen and integrate and bridge."* Extends
`frozen_partition_topology_ego_deformation_20260623.md` (af64e924, MEASURED) with the manifold / n-width /
Morse / persistent-homology / information-geometry lenses, JOINT across the full space. Authority
`[analysis]` over MEASURED anchors; all score math via `tac.contest_score`; pointer UNMOVED 0.19110.

---

## 0. The one-paragraph thesis
The d_seg source (the per-frame argmax partition the scorer demands) is a **stratified set**: a thin
**high-persistence stratum** (the coarse scene contour — LINEAR effective rank **4.07**, a smooth low-dim
deforming manifold, already free) glued to a diffuse **low-persistence stratum** (the class-1 islands,
0.72% of pixels, ~31/frame — LINEAR effective rank **52.9/60**, ≈ full). The binding d_seg residual lives
in the diffuse stratum. **Every frozen-instance / explicit / linear code is bounded by the LINEAR rank 53 →
exhausted (no $0 sidecar, measured).** The ONE quantity nobody has measured — and the entire sub-0.15
question — is the **NONLINEAR intrinsic dimension `m`** of that diffuse stratum: a neural generator beats the
linear n-width iff the islands lie on a low-dim *nonlinear* manifold (`m ≪ 53`). The topology probe measured
the LINEAR width (closing sidecars); `m` is the OPEN door (opening trained generators). This memo derives
why, and turns it into a $0 measurement + a falsifiable plateau-prediction for the from-scratch sweep.

## 1. The fiber/quotient picture (what the scorer actually asks for)
SegNet's argmax is invariant to any input move that doesn't cross a decision boundary. The set of frames
with a FIXED argmax partition is an **argmax cell** (a polytope-like region of ℝᴺ). The d_seg-sufficient
statistic is the cell ID; the optimal code is over the quotient `ℝᴺ / (argmax-polytope × pose-null)`
(#155 Dubois lossy-lossless). **d_seg=0 ⟺ the decoded frame lands in GT's cell.** So the d_seg "source"
we must encode is the *sequence of cell boundaries* across the 600 scored frames — equivalently, the
**moving zero-level-set of the SegNet margin field** `m(x) = z_(1) − z_(2)` (top minus runner-up logit).

## 2. Topology → manifold: the stratification, made precise
The probe's two-scale split is a **Whitney stratification** of the boundary source:

| stratum | pixel mass | LINEAR eff-rank | persistence | manifold reading |
|---|---:|---:|---|---|
| coarse contour | 99.3% | **4.07** (top-1 mode 46%) | HIGH (stable 573-modal) | smooth ~4-dim submanifold; Whitney-embeds in ℝ⁹ → ~9 numbers/frame; **the decoder already nails it** |
| class-1 islands | 0.72% | **52.9/60** | LOW (573/600 distinct) | diffuse; LINEAR-full-rank; the d_seg-binding residual |

**Persistent-homology reading (the deepest bridge):** the islands are the birth/death of connected
components in the super-level-set `{m > 0}` — i.e. the **H₀ features of low persistence**. Persistent
homology's core theorem: low-persistence features are perturbation-unstable "noise," high-persistence are
"signal." So the probe RE-DERIVED persistent homology — coarse = high-persistence signal (rank 4), islands
= low-persistence noise (rank 53). This is the SAME object as the "shallow flips" finding (66.5% of flips
<0.5 logit): **low margin ⟺ low persistence ⟺ near a topological critical point of `m`.** Three independent
findings (topology rank, shallow margin, flip-residual rank 547 a3061) are ONE fact: **the d_seg residual
is the low-persistence stratum of the frozen margin field.**

## 3. Manifold → rate-distortion: why D(H) has a fast head and a slow tail
The RD function of a source is governed by its intrinsic dimension and the decay of its width spectrum
(Kolmogorov n-width `d_n` = best n-dim approximation error):
- **Coarse stratum:** rank 4, fast `d_n` decay → D(H) drops FAST; a handful of bits buy it. The frontier's
  d_seg 6e-4 IS mostly the coarse contour being correct. **Saturated early.**
- **Island stratum:** LINEAR rank 53, FLAT `d_n` → D(H) drops SLOWLY; bits needed ∝ rank. **This is the
  d_seg tail** — the disproportionate capacity the residual demands.

→ **The manifold explanation of the capacity cliff:** d_seg(params) is NOT one power law. It is
`d_seg(p) ≈ A·(fast head, saturates) + B·(slow island tail ∝ p^−γ with small γ)`. Pruning a co-adapted net
below the frontier destroys the slow-tail island capacity first → the cliff. The α≈0.71 single-exponent fit
is an average of a saturated head + a slow tail — which is exactly why extrapolating it (the retracted
closed-form S*) was unreliable.

## 4. The LINEAR-vs-NONLINEAR n-width split (THE sub-0.15 crux)
The probe measured **LINEAR** effective rank (PCA-style covariance spectrum). Linear rank 53 closes:
- explicit per-flip sidecars (a3061, 1.27 B/flip break-even — coding a flip ≈ its worth),
- partition stores (#52, 524 KB rate-dominated),
- fixed-basis / linear codecs (bounded by `d_n`, which is flat here).

It does **NOT** close trained generators. A nonlinear map can represent a source with LARGE linear n-width
but SMALL **nonlinear manifold width** `δ_m` (best error over m-dim *nonlinear* manifolds) — the textbook
neural-codec advantage. The islands are class-1 (a single semantic class: lane markings / small road
objects) generated by a LOW-DIM physical process (ego-motion + road geometry + a few movers). So their
NONLINEAR dimension `m` is plausibly `≪ 53` even though their LINEAR rank is 53. **The R²=0.23
ego-explanation was a LINEAR regression on 6-dim pose — a nonlinear function of ego+scene could explain far
more.** This is the gap between "no $0 sidecar" (true, linear) and "trained generator open" (true,
nonlinear) — and it is the precise reason the convergence points to TRAINING, not a contradiction in it.

## 5. Whitney embedding → the latent-dim PREDICTION (falsifiable, actionable)
A dimension-`m` manifold embeds in ℝ^(2m+1). HNeRV's latent is **28-dim**. So:
- if the islands' nonlinear `m ≤ ~13`, the 28-dim latent SUFFICES to carry them → d_seg can keep dropping
  with capacity → **sub-0.15 reachable by a trained generator**;
- if `m > ~13`, the 28-dim latent is the BOTTLENECK → d_seg SATURATES regardless of decoder width → the
  plateau is real and the fix is a **larger latent / per-frame side-channel for the island stratum**, not a
  wider decoder.

**Prediction:** the observed d_seg plateau capacity is set by `2·m+1` vs the latent dim — NOT by total
params. This reframes the from-scratch sweep: sweep the **latent dim** (and a class-1-targeted side-channel)
as a first-class arm, not just base_channels. It also predicts WHERE FINER/WIRE can help: a high-frequency
basis raises the decoder's nonlinear expressivity at fixed latent — it helps iff the bottleneck is the
DECODER, not the latent. The two screens (latent-dim sweep + FINER/WIRE) jointly localize the bottleneck.

## 6. Information geometry → why quantization is fatal at the boundary (unifies the quant findings)
The natural metric on output space for d_seg is the **Fisher metric of the SegNet softmax**. Near the
zero-margin level-set the softmax is steep → Fisher metric BLOWS UP → small RGB/weight perturbations =
large class-probability moves = flips. So uniform-bit quantization (uniform in Euclidean RGB/weight space)
is wrong: it injects equal noise everywhere, but the noise is AMPLIFIED on the boundary band. The
**Fisher-optimal** allocation spends bits ∝ √(Fisher) = more bits on the boundary — exactly the measured
WRQ "spend more bits on boundary weights" + the int5-cap "shallow flips are quant-fragile." Information
geometry unifies: shallow-margin (§2) ⟺ large Fisher (§6) ⟺ quant-fragile (qaxis) ⟺ low-persistence (§2).
**One geometric object — the high-curvature boundary band of the margin field — is the load-bearing fact
behind every d_seg finding this session.**

## 7. The grand bridge (patterns-of-patterns)
| lens | the SAME object, seen differently |
|---|---|
| topology | low-persistence H₀ stratum (islands born/die) |
| manifold | LINEAR-rank-53 diffuse set; nonlinear `m`=? |
| rate-distortion | flat-`d_n` slow tail of D(H) |
| Morse | near-critical points of the margin field `m` |
| info-geometry | high-Fisher-curvature boundary band |
| RGB-vs-task carrier | the bits RGB wastes on interiors vs the islands that need them |

Every lens names the **boundary band of the frozen SegNet margin field, low-persistence stratum**. That is
THE source. The frontier spends its bits on RGB everywhere (interiors free in d_seg but paid in rate);
sub-0.15 spends bits on this object and nothing else.

## 8. What this CHANGES (actionable, deep-math-grounded)
1. **$0 NOW — measure the NONLINEAR intrinsic dimension `m` of the class-1 island stratum** (TwoNN / MLE /
   local-PCA / small-AE bottleneck on the 600-frame class-1 masks + the margin field in the boundary band).
   `m ≪ 53` → trained generator open (sub-0.15 live); `m ≈ 53` → latent/side-channel is the lever, not
   decoder width. Decisive for the sweep design. (spawned; $0/CPU, no MPS contention.)
2. **From-scratch sweep redesign:** sweep **latent-dim** (Whitney `2m+1` arm) alongside base_channels +
   the INT4-co-adapted arm; add a **class-1-targeted capacity term** (route bits to the 0.72% region per
   the probe's reusable prior). The objective is the boundary band, not RGB fidelity.
3. **FINER/WIRE interpretation sharpened:** a high-freq basis helps iff the bottleneck is the DECODER's
   nonlinear width, not the latent. The latent-dim sweep + FINER/WIRE jointly localize it.
4. **Loss design:** weight the d_seg loss by √(Fisher) / inverse-margin (boundary-band emphasis) — the
   info-geometry-optimal training signal; folds into the live generator d_seg campaign (Lever-5 lineage).
5. **The task-space carrier (#171)** is the manifold-optimal carrier: encode the margin-field boundary band
   + the 6-dim pose, on the quotient — the D(H) curve for the task-space source is the one with the cheap
   coarse head AND the only honest shot at the island tail (a generator on the right manifold).

## 9. Honest ledger
- MEASURED (af64e924): coarse rank 4.07, island rank 52.9, ego R²=0.23, byte/S vertices (0.73 / 0.84).
- REASONED (this memo): persistent-homology reading, fast-head/slow-tail D(H), linear-vs-nonlinear n-width,
  Whitney latent-dim prediction, Fisher boundary-band unification. These are deductions FROM the measured
  ranks + standard theory — each is a hypothesis until the §8.1 nonlinear-`m` measurement + the trained
  sweep confirm.
- NOT claimed: no score moved; pointer UNMOVED 0.19110. The synthesis sharpens the TRAINING design; it does
  not itself lower S. The END is a byte-closed exact row from the redesigned sweep.

## 10. THE REPRESENTATION-LEVEL REFRAME (operator 2026-06-23: "pixel is not the correct level; everything is compressible — imagination + divergent thinking + tradeoffs")
The §2–§4 conclusion "islands are full-rank content-noise" is a **PIXEL-LINEAR-LOSSLESS artifact, not a
property of the source.** Three corrections, each an operator knob:

**(a) "Pixel is the wrong level" = rank is BASIS-DEPENDENT.** `rank_B(X)` depends on basis B; pixel-B is a
bad B for this source. The right B makes the islands sparse. Candidate levels (each an "imagination"):
- **spectral** (FINER/WIRE/Daubechies): the boundary is high-frequency in pixels, few coefficients in a
  Gabor/wavelet basis — the EXACT thing the in-flight FINER/WIRE screen tests.
- **contour / Fourier-descriptor:** a class-1 island is a closed curve ≈ O(10) params (Daubechies CO-LEAD's
  multi-scale partition); 31 islands → ~310 numbers, temporally AR-codeable as they drift smoothly.
- **motion-compensated:** the R²=0.23 ego-explanation was a LINEAR regression on 6-dim pose — in the
  NONLINEAR ego-compensated frame the islands are near-static; the residual after warp is what's left.
- **level-set / margin-field:** code the smooth `m(x)` in the boundary band (a CNN output, low-frequency in
  the right region) instead of the discrete partition — the flips are just `sign(m)` crossings.
- **semantic / scene-graph:** class-1 = specific objects; store an object list + ego → RENDER. Near-zero
  bytes if the scene is structurally simple.

**(b) "Everything is compressible" = Kolmogorov complexity ≪ linear rank.** `K(x)` = shortest program
generating x; the islands are generated by a LOW-complexity process (ego plyline + road geometry + a few
movers) → low `K` even at linear-rank 53. Linear rank sees only 2nd-order (covariance) structure; `K` sees
ALL structure. The gap IS "imagination" (finding the generating program). **A neural generator is an
approximate Kolmogorov compressor** — it learns the program. So "everything is compressible" ⟹ a generator
compresses the islands ⟹ TRAINING is the lever (re-confirms the §7 convergence from a new direction) AND the
generator must OUTPUT in the right level (a pixel-output generator re-wastes the bits; a task-level-output
generator does not).

**(c) "Tradeoffs" = the TASK distortion metric, not pixel-MSE.** Indirect-RD: d_seg needs only the ARGMAX
right — the pixel-distortion budget at fixed d_seg is the ENTIRE argmax cell (huge). The "incompressibility"
was measured pixel-lossless (wrong metric); on the task metric the islands have enormous slack. The quotient
codec (#155) is this exactly: code `ℝᴺ/(argmax-polytope × pose-null)`, lossy-in-pixels, lossless-in-task.

**Synthesis:** the islands compress under (right basis) × (generative program) × (task distortion) =
imagination × divergent-thinking × tradeoffs. The "full-rank wall" was an artifact of picking the worst
corner of all three (pixel basis, explicit code, lossless metric) — which is precisely the FROZEN-INSTANCE
corner, which is why frozen-instance is exhausted and TRAINING (a generator, in the right level, on the task
metric) is open. **This is the deepest statement of why sub-0.15 is a training problem AND what to train:
not a pixel-RGB generator (the frontier's wasteful corner) but a generator that outputs the task-level
representation (boundary band / contour / margin-field / quotient) under the task metric.** = #171 task-space
carrier, now derived from first principles + the operator's three knobs.

**§8.1 measurement GENERALIZED (the operator's reframe, made empirical):** don't just measure ONE nonlinear
`m` — measure the islands' intrinsic dimension ACROSS representation levels {pixel-linear, spectral/DCT,
Fourier-descriptor contour, nonlinear-motion-compensated, small-AE generative}. The level where `m`
collapses is the basis to build the generator in. This is a representation-level SEARCH — the operator's
"imagination + divergent thinking" as a $0 measurement.

## 6-hook wire-in
#1 sensitivity-map: ACTIVE — the boundary-band / class-1 region is the canonical d_seg capacity-routing
prior. #2 Pareto: the fast-head/slow-tail D(H) decomposition bounds the achievable region. #3 bit-allocator:
Fisher-√ allocation (§6). #4 cathedral: N/A (analysis). #5 continual-learning: this memo + the
representation-level intrinsic-dimension measurement JSON. #6 probe-disambiguator: the multi-level `m` test
IS the "which representation level compresses the islands" disambiguator (the operator's reframe).
