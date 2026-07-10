# Papers-checked ledger — arXiv 2601.20498

**Verdict: NOT-RELEVANT (no lever, no grain). On-domain paper, off-topic for our stack.**

Date: 2026-07-10 (operator-supplied link, anti-re-research protocol).
Cost: $0 (abstract + method verified via WebFetch; no PDF deep-read warranted).

## What it is

**"Spectral Diffusion Models on the Sphere"** — Mari, Brutti, Durastanti.
arXiv **math.PR** (Probability), cross-listed **stat.ML**. (Real paper, NOT an
off-domain typo — unlike the operator's prior three supplied IDs.)

Method: a **generative diffusion model** (score-based SDE) for data living on the
2-sphere, formulated directly in the **spherical-harmonic (spectral) domain**. Core
technical result: the spherical discrete Fourier transform maps spatial Brownian
motion to a **constrained, non-isotropic** Gaussian process, so the forward/reverse
diffusion SDEs pick up geometry-dependent covariance constraints, and **spatial vs
spectral score-matching are NO LONGER EQUIVALENT** on the sphere (they are in flat
Euclidean space). They derive the modified forward/reverse SDEs and characterize the
induced noise covariance.

## Verdict vs OUR stack (verdict_scope: PARADIGM-level not-relevant, high confidence)

Our problem is **indirect rate-distortion / coding-for-machines**: a deterministic,
byte-closed, non-RGB task-space WITNESS that amortizes the SegNet argmax partition
(d_seg) + stored se(3) ξ pose sidecar (d_pose), scored by `100·d_seg + √(10·d_pose) +
25·bytes/N`. The whole game is a **deterministic generator + minimal counted payload**,
NOT a stochastic sampler.

This paper is a **stochastic generative model** on a **sphere manifold**. Every axis
of contact fails:

- **Not a compression method.** No rate term, no byte budget, no evaluator-equivalent
  witness. A diffusion sampler is the opposite of our deterministic-decode /
  bit-identical-inflate discipline — sampling noise is disqualifying for a byte-closed
  archive.
- **Wrong geometry.** Our manifold is the **codim-1 separatrix** (argmax boundary
  annulus) in image/frame space + the ~8-dim lane-orbit + se(3) screw for pose. The
  paper's manifold is the literal 2-sphere (directional/geospatial/CMB-style data).
  Spherical harmonics ≠ our anisotropic-curvelet-on-a-curved-edge basis.
- **The one superficial echo is not a transferable lever.** "Spectral ≠ spatial
  score-matching on a curved manifold" rhymes with our **"basis-match is PRIOR to
  capacity"** intuition (the directional/curvelet basis matters because the target
  geometry is curved). But that is an ANALOGY, not a method we can lift: they are
  matching a *diffusion score* on a *sphere*; we are matching an *argmax partition* on
  a *codim-1 image separatrix*. No equation, kernel, or code crosses over. Citing it as
  a lever would be a false-transfer (the papers-checked "GRAIN" bar is not met — a grain
  must sharpen a crux we already hold; this does not).

No V2-originality collision (L16): our carrier does not overlap "diffusion on sphere."
No relevance to #385 / the dual-chain WALL / v7.5 / v8 per-class carriers.

## Grains / levers

**None.** Not CONFIRMS (it does not independently re-derive anything we hold), not GRAIN,
not WATCH (no plausible future contact — we are not going to run a stochastic spherical
sampler in a byte-closed inflate). Terminal NOT-RELEVANT at PARADIGM scope.

## Proposed L55 hook phrase (operator applies; I do not edit MEMORY.md)

`SphericalDiffusion(2601.20498)=NOT-RELEVANT (spherical score-based generative SDE, math.PR; stochastic sampler ⊥ our deterministic byte-closed witness; spectral≠spatial-on-curved-manifold only ANALOGIZES basis-match-before-capacity, no transferable method)`
