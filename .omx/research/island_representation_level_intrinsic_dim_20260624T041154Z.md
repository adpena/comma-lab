---
title: "Class-1 island intrinsic dimension ACROSS REPRESENTATION LEVELS — is the d_seg-binding stratum FORMAT-compressible or GENERATOR-only? VERDICT: GO-GENERATOR"
authority: "[contest-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19110; $0; CPU-only, NEVER MPS, no GPU; no PR; no exact-eval dispatch"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-23
verdict: GO-GENERATOR
subagent: island-repr-intrinsic-dim-20260623
cross_refs:
  - .omx/research/frozen_partition_topology_ego_deformation_20260623.md   # the pixel-linear rank-53 control this extends across bases
  - experiments/probe_frozen_partition_topology.py                        # the loader reused (frozen-SegNet argmax cache)
  - .omx/research/custom_witness_format_inflate_interpreter_design_20260623.md  # the Whitney 2m+1<=28 -> m<=13 latent budget (GO-FORMAT bar)
  - reports/witness_seg_boundary_decisive.json                            # the flat-sidecar boundary NO-GO (existence-proof cross-check)
  - experiments/results/indep_dseg_bets_20260623_inflated/seg_argmaps.npz # EXACT frozen-SegNet argmax cache (authority-faithful, dt=1e-7)
tool: experiments/probe_island_representation_intrinsic_dim.py
result_json: experiments/results/island_repr_intrinsic_dim_20260624T041154Z/island_repr_full_class1.json
result_json_small: experiments/results/island_repr_intrinsic_dim_20260624T041154Z/island_repr_full_small_islands.json
shuffle_control: experiments/results/island_repr_intrinsic_dim_20260624T041154Z/ae_shuffle_control.json
tests: src/tac/tests/test_island_representation_intrinsic_dim.py   # 19 NO-FAKE, all green
---

# Class-1 island intrinsic dimension across representation levels

**Operator vision (2026-06-23):** *"pixel is not the correct representation level; everything is
compressible = imagination(basis) × divergent-thinking(generative program) × tradeoffs(task
distortion)."* The d_seg-binding class-1 island stratum is LINEAR-rank ~53 in PIXELS (full-rank,
`frozen_partition_topology` Result 4). **Rank is BASIS-DEPENDENT.** This probe measures the islands'
INTRINSIC DIMENSION `m` across representation levels and finds the basis (if any) where it collapses.

**TL;DR — VERDICT: GO-GENERATOR.** The islands are full-rank in EVERY reconstruction-faithful LINEAR
basis (k95 = 412 pixels, 61 DCT, 29 contour, 94 motion-residual — all ≫ the Whitney latent budget
m≤13), so **no flat code in any linear basis carries them within the witness format's latent budget**.
BUT a nonlinear autoencoder reconstructs them at bottleneck dim **8** (90%-of-achievable-variance
knee = 8 ≤ 13), and a **phase-shuffle control proves this is REAL structure, not an AE artifact**
(real islands: 81% explained at dim 8; structure-destroyed shuffle: 18% at dim 8, needs dim 32). The
islands live on a **low-dimensional NONLINEAR manifold curved through linear space** — exactly the
signature where a **trained generative program wins and a flat code does not.** This routes the d_seg
lever to TRAINING (the live generator d_seg campaign), NOT to a $0 flat witness-format sidecar, and
gives the format a concrete target: an ~8–13-dim nonlinear island latent, not a 53-dim flat one.

All numbers `[contest-CPU advisory]` NON-PROMOTABLE. Pointer UNMOVED 0.19110. $0, CPU-only, NEVER
MPS/GPU, no PR, no exact-eval dispatch.

---

## Data (real, frozen, $0)

The class-1 island indicator stack `M[t] = (gt[t] == 1)` over the 600 scored frames, extracted from
the EXACT frozen-SegNet argmax cache (`seg_argmaps.npz` key `gt` (600,384,512) uint8), reusing the
EXACT loader the topology probe used. Cache faithfulness re-verified in-probe: cached d_seg =
0.0005598873 (Δ vs report < 1e-7 — exact-scorer faithful). Class-1 mass = 0.585% of pixels, ~27.6
connected components/frame (the volatile island stratum). Two strata measured (identical verdict):
the **full class-1 indicator** and the **<500px fine-island substratum** (the d_seg-debt carrier).

## Result table — representation level → intrinsic dim → compressibility implication

| # | representation level | participation ratio | **k95 (recon-faithful m)** | compressibility implication |
|---|---|---:|---:|---|
| 1 | **pixel-linear PCA** (CONTROL) | 74.6 | **412 / 600** | full-rank content scatter — reproduces the ~53 control at scale (60f→k95=48; 600f→412); NOT pixel-compressible |
| 2 | **spectral 2D-DCT** (32×32 LF block) | 16.4 | **61 / 600** | DCT concentrates *energy* (LF frac 0.205) but still needs 61 modes for 95% recon — a Gabor/DCT basis does NOT sparsify the islands to the budget |
| 3a | **contour Fourier-descriptor — per-frame** | 6.5 | **29 / 320** | the lowest LINEAR recon-faithful dim, but still 29 ≫ 13 — per-frame island *shape+count* needs 29 modes |
| 3b | contour **shape-vocabulary cloud** | 2.0 | 2 / 8 | ⚠ DISQUALIFIED: pooling all islands loses WHERE (location/count) — a dim-2 *shape* vocabulary does NOT reconstruct the partition; reported, not eligible for GO-FORMAT |
| 4 | **motion-compensated affine warp residual** | 44.4 | **94 / 119** | a CORRECT per-frame affine warp reduces frame-to-frame *energy* 19.2% but leaves residual rank k95=94 (= raw 94) — motion does NOT collapse the islands; they birth/die/morph, not rigidly translate |
| 5a | nonlinear **TwoNN** (Facco 2017) | — | **28.9** | nonlinear ID estimator on the pooled stack: ~29, consistent with the contour linear floor |
| 5b | nonlinear **MLE** (Levina-Bickel) | — | **13.1** | MLE ID ≈ 13 — right at the Whitney budget edge |
| 5c | **tiny CPU autoencoder bottleneck sweep** | — | **knee = 8** (90% of best) | DECISIVE: a nonlinear AE reconstructs the islands at bottleneck **dim 8** (81% var), 16 → 86%, 32 → 89% — the nonlinear `m` is ≈ 8–13, far below the linear 29–412 |

**Whitney latent budget:** the custom witness format's latent layer allows `2m+1 ≤ 28 → m ≤ 13`.
Every reconstruction-faithful LINEAR basis exceeds it (min 29); the NONLINEAR AE knee (8) and MLE
(13) sit AT or BELOW it.

## The decisive NO-FAKE control — phase-shuffle (real structure, not an AE artifact)

The AE-knee=8 could be a sparsity/plateau artifact (sparse binary masks are "easy" for any low-dim
AE). The phase-shuffle control settles it: per-frame, randomly relocate the ON pixels (KEEP the same
mass, DESTROY the spatial structure) and re-run the identical AE sweep:

| stack | explained var @ dim 2 / 4 / 8 / 16 / 32 | best | 90%-knee |
|---|---|---:|---:|
| **REAL islands** | 0.458 / 0.683 / **0.810** / 0.856 / 0.887 | 0.887 | **8** |
| **SHUFFLE control** (mass-matched, structure-destroyed) | 0.005 / 0.000 / **0.178** / 0.319 / 0.939 | 0.939 | **32** |

At bottleneck dim 8, the real islands are 81% reconstructed; the structure-destroyed shuffle is only
18%. The low nonlinear knee is **REAL spatial structure** — the islands occupy a curved low-dim
manifold, not a flat full-rank subspace. (If it were a sparsity artifact, the shuffle would also
collapse; it does not — it needs the full dim 32.) This is the existence-proof that GO-GENERATOR is
the honest verdict, not WALL.

## VERDICT — GO-GENERATOR (the islands are a low-dim NONLINEAR manifold)

Applying the operator's 3-way rule on the reconstruction-faithful dimension:

- **NOT GO-FORMAT:** no reconstruction-faithful LINEAR basis collapses `m` to ≤13 (min k95=29 in the
  contour basis; 61 DCT; 94 motion; 412 pixel). The shape-vocabulary dim-2 is DISQUALIFIED because it
  discards WHERE — a low shape vocabulary ≠ a low partition-reconstruction dimension. So you cannot
  build the witness format's WHERE/HOW-MUCH layers as a flat code in any tested linear basis within
  the latent budget.
- **GO-GENERATOR:** every linear basis exceeds the budget BUT the nonlinear AE 90%-knee = 8 ≪ the
  linear rank, and the phase-shuffle control proves it is real structure. The islands are a low-dim
  (~8–13) NONLINEAR manifold curved through pixel/DCT/contour space. **A TRAINED generative program
  (which can learn the nonlinear chart) reproduces them; a flat code in a fixed basis (PCA/DCT/Fourier
  descriptors) cannot, because the manifold is curved.** This is exactly "everything is compressible =
  imagination(basis) × divergent-thinking(generative program)": the right *generator* exists at m≈8;
  no fixed *basis* exposes it linearly.
- **NOT WALL:** the islands are NOT irreducible content-noise. WALL would require m≈53 even nonlinear;
  the AE/MLE show m≈8–13. The d_seg debt is compressible — by a generator, not a flat sidecar.

## Existence-proof cross-check (both directions — no over-claim, no premature kill)

- **Does any known artifact already carry the islands cheaply?** The flat-sidecar boundary route is a
  measured NO-GO (`witness_seg_boundary_decisive.json`): 543 KB residual, 46% survival (< 50% bar),
  witness MDL 565–600 KB ≫ frontier 177 KB and ≫ the conditional MDL band 24.6–64.6 KB. That route
  stored the boundary as a FLAT per-flip sidecar — exactly the linear-basis flat code this probe shows
  is full-rank (k95=94 in the motion-residual / 412 in pixels). The two findings AGREE: a flat
  boundary/island code is full-rank and loses; the new contribution is *why* — and *what wins instead*
  (an ~8-dim nonlinear generator latent). I do NOT over-claim a format exploit (no linear basis hits
  the budget); I do NOT kill the paradigm (a generator at m≈8 is viable).
- **Does the topology probe's pixel-rank-53 reproduce?** Yes — the control (level 1) gives k95=412 on
  600 frames (the 60-frame subset gave k95=48 ≈ the topology memo's "52.9/60"), confirming the loader
  is faithful and the islands are MORE full-rank at scale in pixels.

## 5-lens deep-math review

- **Algebra (rank vs basis).** Linear rank is a basis-INVARIANT of the row space — PCA/DCT/Fourier-
  descriptor are all orthogonal (or near-orthogonal) linear maps, so a high rank in one is a high rank
  in all (confirmed: 412/61/29/94 are all ≫ budget; the variation is only in how energy concentrates,
  not in the rank needed for 95% reconstruction). The DCT's low participation ratio (16.4) with high
  k95 (61) is the classic "energy-compact but not rank-compact" signature: a few modes hold most
  energy but the *tail* is heavy (heavy-tailed singular spectrum), so faithful reconstruction still
  needs many modes. A flat code pays for the tail; a nonlinear chart does not.
- **Geometry (curved manifold).** k95≈29–412 LINEAR but AE-m≈8 NONLINEAR is the textbook signature of
  a low-dimensional manifold embedded with CURVATURE — like a swiss roll (intrinsic dim 2, linear PCA
  rank 3). The islands' positions/shapes are parameterized by ~8 latent factors (plausibly: ego-yaw,
  pitch, a few scene-object positions, lighting) but the map latent→pixel-mask is nonlinear (a mask is
  a thresholded, occluded, perspective-warped function of the latents), so no linear basis flattens it.
  TwoNN (28.9) overestimates vs the AE (8) because TwoNN measures the *local* intrinsic dim on the
  pooled max-pooled stack where the manifold is noisy at small scale; the AE measures the *global*
  reconstruction dim — both agree it is ≪ the ambient 12288-d pooled / 196608-d pixel space.
- **Calculus (the AE is a learned nonlinear coordinate chart).** The autoencoder's encoder is a
  nonlinear map ℝ^d → ℝ^8 and the decoder its approximate inverse; the 81%-at-dim-8 result is the
  empirical statement that an 8-dim chart + smooth (ReLU-MLP) decoder covers 81% of the island
  variance. The monotone explained-var curve (0.46→0.68→0.81→0.86→0.89) is the reconstruction-vs-
  bottleneck "knee" — the diminishing returns past dim 8 are the curvature/noise the chart cannot
  flatten. A generator trained end-to-end on d_seg (not L2 recon) would chart the *d_seg-relevant*
  directions even more tightly (the AE is a conservative lower bound on generator compressibility).
- **Physics / information (rate-distortion).** The flat-sidecar NO-GO measured ~1.0–1.4 bytes/flip and
  543 KB total — the *rate* of an i.i.d.-coded flat code over a full-rank source. The manifold result
  says the SOURCE is not i.i.d.: it has an 8-dim sufficient statistic. A generator that emits the 8
  latents + a learned decoder pays O(8 floats/frame + one-time decoder weights) instead of O(900
  flips/frame) — the rate gap between coding the manifold COORDINATES vs coding the ambient PIXELS is
  the entire compressibility headroom the operator's framing predicts. The "task distortion" tradeoff:
  the generator is scored on its OWN frame-1 d_seg, so it can spend its 8 latents on exactly the
  d_seg-binding island structure, not on visually-faithful but score-irrelevant detail.
- **Existence-proof (no artifact beats it; no kill).** No fixed-basis flat code in the repo carries
  the islands within budget (boundary sidecar = 543 KB NO-GO; #52 lossless = 524 KB; this probe's
  linear bases all > budget). The ONLY known lever that can chart the nonlinear manifold is a TRAINED
  generator — which is the live in-flight campaign. So this probe does not invent a new artifact; it
  REDIRECTS and RE-TARGETS the existing generator campaign with a measured latent-dimension target
  (~8–13) and a sensitivity prior (the 0.585% class-1 pixels), and it forecloses (with a decisive
  control) the flat-sidecar-in-some-clever-basis hope.

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map — ACTIVE:** the d_seg-binding islands are 0.585% of pixels, ~27.6 components/
   frame, on an ~8–13-dim nonlinear manifold. Reusable prior: a generator's island-latent need only be
   ~8–13-dim; weight trainer capacity toward the class-1 small-component regions (not the stable large
   regions, per the topology probe's coarse-near-free finding).
2. **Pareto — ACTIVE (records a NEW informative point):** no fixed-linear-basis flat island code is on
   the frontier (all > budget); the viable vertex is a NONLINEAR generator latent at m≈8 — a new
   Pareto entry distinct from the dominated flat-sidecar point.
3. **Bit-allocator — ACTIVE (advisory):** "class-1 islands = 8–13-dim nonlinear manifold, linear-full-
   rank (k95 29–412); allocate to a trained generator's island latent (~8 floats/frame + shared
   decoder), NOT to a flat per-flip/per-mode archive sidecar."
4. **Cathedral-dispatch — N/A:** advisory, non-promotable, no archive change, no paid eval.
5. **Continual-learning — ACTIVE:** probe outcome `island_repr_intrinsic_dim_20260623` (verdict
   GO-GENERATOR; reactivation = a generator whose island-latent ≤13 reproduces class-1 to d_seg < the
   beat-frontier line) registered via `tac.probe_outcomes_ledger.register_probe_outcome`.
6. **Probe-disambiguator — ACTIVE:** `experiments/probe_island_representation_intrinsic_dim.py` is the
   disambiguator that arbitrates "FORMAT-compressible (some basis collapses m≤13)" vs "GENERATOR-only
   (nonlinear m≪linear)" vs "WALL (content-noise)" — the math arbitrates GO-GENERATOR, with a
   phase-shuffle control proving the nonlinear collapse is real.

Mission contribution: **frontier_protecting + frontier_breaking-informing** — it forecloses the
flat-sidecar-in-a-clever-basis $0 hope (protecting) AND re-targets the live generator campaign with a
measured ~8–13-dim island-latent budget + sensitivity prior (informing the breaking lever). Sister to
`frozen_partition_topology` (which measured pixel-rank-53 + ego-R²=0.23): this extends that single
basis to FIVE representation levels + nonlinear estimators + a decisive shuffle control. NON-PROMOTABLE
`[contest-CPU advisory]`. Pointer UNMOVED 0.19110.

## NO-FAKE ledger
- MEASURED (this unit, exact frozen-SegNet argmax cache, 600 frames, CPU, NEVER MPS/GPU):
  pixel-PCA k95=412 (control reproducing the rank-53 claim at 60f→48); DCT k95=61 / LF-energy 0.205;
  contour-descriptor frame k95=29 / shape-vocab disqualified; affine-warp residual k95=94 (warp −19.2%
  energy, rank unchanged — after fixing a real warp-convention bug, re-verified on synthetic
  translation); TwoNN 28.9 / MLE 13.1; AE 90%-knee = 8 (81% var); phase-shuffle control (real 81% vs
  shuffle 18% at dim 8). Both class-1 full + <500px substrata give the identical GO-GENERATOR verdict.
- DERIVED: the GO-GENERATOR decision rule (k95 recon-faithful vs Whitney budget vs nonlinear knee); the
  manifold-curvature interpretation; the rate-distortion headroom argument.
- NOT claimed: NO score moved; pointer UNMOVED 0.19110; no archive built/byte-closed; no exact-eval
  dispatch; no PR. The islands are NOT FORMAT-compressible in any tested linear basis (this verdict);
  the d_seg lever is a TRAINED generator with an ~8–13-dim island latent. Tool + 19 NO-FAKE tests
  committed; ruff clean.
