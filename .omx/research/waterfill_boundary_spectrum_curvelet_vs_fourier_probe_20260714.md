# $0 probe: boundary-spectrum waterfilling — oriented (curvelet) vs isotropic (Fourier) capacity gain (2026-07-14)

**VERDICT: GO — de-risks #502 (genuine localized curvelet/shearlet frames).** On OUR measured
frozen-SegNet boundary annulus, an orientation-adaptive allocation needs **~1.7–2.0× LESS rate**
than orientation-blind (isotropic Fourier) allocation to reach the same distortion; equivalently
isotropic distortion is **12–47% higher** at matched budget (gap grows with budget). The boundary
margin spectrum is **strongly anisotropic (30–68× max/min orientation energy)**. This is a
**linear-basis UPPER bound** on what a trained curvelet witness would realize — it is NOT a d_seg
row and does not touch the pointer (UNMOVED 0.18804 bank / 0.19108 submittable).

**Means. Pointer UNMOVED. This grounds a P0 build (#502) in a measured number, not a vibe.**

---

## Why this probe (theory anchor)

Torralba & Weiss (arXiv 2607.07470, deep-read in
`papers_checked_queued_batch_scaling_contrastive_ptrm_rsi_fork_20260714.md`) prove the sinusoid /
isotropic-Fourier first layer is the optimal contrastive representation **ONLY under GLOBAL
translation-invariant stationarity** (Thm 2.1, Peligrad–Wu: stationarity is the ONLY reason the
alignment matrix `B` and covariance `Σ` co-diagonalize in the global Fourier basis). Their Eq-4
waterfilling then allocates capacity across the generalized eigenvalues of the (B,Σ) pair.

Our d_seg residual lives on a **codim-1 ORIENTED boundary annulus** (memory L66, ~4.7% area) — a
**non-stationary, locally-oriented** manifold. That VIOLATES the theorem's precondition, so global
sinusoids are provably sub-optimal for us. This probe quantifies *how much* by running the paper's
waterfilling on OUR measured boundary spectrum under two bases.

## Method (all MEASURED from existing gt caches; numpy only; no training, no dispatch)

Script: `experiments/waterfill_boundary_spectrum_probe.py` (reusable; ~2s on n96, ~3s on n600).

1. **Boundary annulus**: from cached SegNet argmax `lstars` (H=384, W=512) build the inter-class
   edge set (a pixel whose 4-neighbour has a different class label), dilate by 2 px. Caches used:
   `experiments/results/mlx_fleet_gt_cache/gt_n96.npz` and `gt_n600.npz` (both carry `lstars` +
   `margins` at scorer resolution). Edge fraction ≈1.3% raw → ~4.7% dilated (matches L66).
2. **Local orientation**: Sobel gradient of the cached `margins` field at each boundary pixel gives
   the boundary NORMAL direction (margin changes fastest across the class boundary); tangent = +90°.
3. **Tangent-frame spectrum**: per boundary patch (32×32, Hann-windowed), 2D FFT power accumulated
   into a polar (radius r, angle-relative-to-normal a) histogram — the angle coordinate is rotated
   by the local normal (no pixel rotation), so every patch's boundary is aligned the same way.
   `a`-bin 0 = wavevector along-normal (variation ACROSS the boundary), last `a`-bin = along-tangent
   (variation ALONG the boundary: dashes/curvature). Averaged over 11.5k (n96) / 36k (n600) patches.
4. **Gaussian reverse water-filling** (the canonical closed form of the paper's Eq-4 waterfilling)
   on the tangent-frame spectrum `σ²(r,a)`, minimizing `D = Σ σ²_k · 2^(−2 P_k)` s.t. `Σ cost_k·P_k = B`:
   - **ORIENTED (curvelet/steerable analog)**: each `(r,a)` cell allocates rate freely (cost 1/cell)
     — it can put capacity exactly where the boundary energy is.
   - **ISOTROPIC (global Fourier)**: orientation-blind — all `a`-cells at a radius share ONE rate
     `P_r`, and funding radius r costs `a_bins` (must pay EVERY orientation equally), so capacity is
     wasted on the empty (diagonal) orientations. Energy-preserving; no free ε knob.

This isolates the ONE mechanism the theorem predicts: an isotropic basis must spend capacity
uniformly over orientation; an oriented basis concentrates it — the win = the boundary's orientation
anisotropy converted to rate.

## Measured results

**The boundary margin spectrum is strongly, robustly anisotropic** — per-orientation energy is
**bimodal**, peaking at BOTH the along-normal and along-tangent axes and near-empty at the diagonals
(n600, a_bins=12, energy fraction per orientation bin, normal→tangent):

```
0.266  0.117  0.068  0.039  0.008  0.007  0.006  0.009  0.033  0.075  0.117  0.253
```

Max/min orientation-energy anisotropy: **41× (n600)**, 47–53× (n96), 34–68× across patch/bin choices.
Participation ratio ≈ 6.7–9.7 effective cells out of 112–168 (highly concentrated).

**Capacity gain (headline), n600, 36k patches:**

| metric | value |
|---|---|
| capacity ratio B_iso/B_orient to reach D≤0.70 | **1.99×** |
| capacity ratio B_iso/B_orient to reach D≤0.50 | **1.74×** |
| capacity ratio B_iso/B_orient to reach D≤0.40 | **1.72×** |
| distortion ratio D_iso/D_orient @ budget=1.0 | 1.12 |
| distortion ratio D_iso/D_orient @ budget=2.0 | 1.18 |
| distortion ratio D_iso/D_orient @ budget=4.0 | 1.27 |
| distortion ratio D_iso/D_orient @ budget=9.0 | 1.43 |

**Robustness** (all runs): capacity ratio **1.7–2.0×**, invariant across n96↔n600, seeds 0/1/2
(anisotropy 47→53→47×; ratios identical to 2 d.p.), a_bins ∈ {8,12,16} (1.77–1.85×), patch ∈ {32,48}
(1.77–1.85×). The oriented win GROWS with rate budget (finer structure ⇒ anisotropy bites harder).

## Corroboration of the 3.2× / −48% findings

- **Direction: YES, strongly corroborated.** The measured boundary spectrum is highly anisotropic and
  the oriented allocation is uniformly, robustly better — exactly what L25 (3.2× along-tangent deficit)
  and the −48% self-orient/directional-basis finding predict. Isotropic Fourier provably under-serves
  the oriented boundary.
- **Magnitude: consistent ballpark, NOT a direct match (honest).** My capacity number (isotropic needs
  ~1.7–2.0× the rate ⇒ ~12–47% higher distortion at matched budget, growing to ~47% at high budget) is
  the SAME SIGN and comparable order to the −48% d_seg reduction. But they are DIFFERENT quantities: −48%
  is a trained-witness d_seg delta (flagged PROXY/over-credited in L25); mine is a **linear-basis
  capacity upper bound**. The 3.2× is the residual's along-tangent frequency deficit; my 30–68× is the
  max/min orientation anisotropy of the boundary margin spectrum — related but not the same statistic.
  Do NOT equate the numbers; they agree in sign and rough scale.

## Honest caveats (what this is NOT)

1. **UPPER bound, not a realized d_seg.** This is a linear rate-distortion capacity comparison of two
   frequency partitions on a measured spectrum. A trained curvelet witness must still be built and its
   d_seg measured through-R on the exact bytes — the realized gain will be ≤ this bound (training
   imperfection, the through-R uint8/resize wall, non-linearity). This de-risks #502; it is not #502's result.
2. **Signal = the SegNet margin field**, chosen because its gradient defines the boundary normal and it
   is the scorer's own decision surface. A different choice of "signal" (raw label field, per-class
   indicator) could shift magnitudes; the anisotropy direction is robust to the choice.
3. **Binning-dependent anisotropy magnitude** (34–68× as a_bins grows) — expected, since finer
   orientation bins resolve the concentration better. The *capacity ratio* (1.7–2.0×) is stable, which
   is the load-bearing number for #502.
4. **n96 and n600 agree tightly**, so n is not a concern here; reported headline is n600 (best n).

## De-risk verdict for #502 (genuine curvelet/shearlet build)

**GO.** There is a real, robust, non-trivial capacity gain (~1.7–2.0× rate efficiency; distortion
reduction growing to ~30–47%) from an orientation-adaptive basis on our measured boundary spectrum,
grounded in the Torralba–Weiss theorem (isotropic Fourier is optimal ONLY under stationarity we do not
have). #502 should be built as a genuine localized oriented frame (curvelet/shearlet/steerable),
targeting per-orientation capacity allocation — NOT more isotropic Fourier features. The realized d_seg
gain must be MEASURED through-R (this probe is the pre-build upper bound, per the optimal-form + no-fake
disciplines). Sisters: #497 (alt-to-Fourier), #277 / #25 (the 3.2× / basis-over-credited findings),
and #157/#336 bit-allocator (waterfilling ≡ our bit allocator — the per-orientation split this probe
computes is directly the allocator's target).

**Pointer UNMOVED. This is a MEANS row (de-risk + grounding), not a score-mover.**
