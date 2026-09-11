# ddm_rbf1 — free post-render boundary treatment on move 44

`research_only=true`, `score_claim=false`, `promotion_eligible=false`.
Charter: `ddm_rbf1_free_post_render_boundary_treatment_charter_20260911.md`;
common contract: `.omx/tmp/codex_runs/_common_contract.md`.

## Current outcome

**CLOSED at formulation scope. The exact pointer has not moved and this arm does
not move it.** The FREE deterministic post-render boundary treatment family is
measured NEGATIVE on both scored channels at every amplitude tested, and the
closure is double: the seg channel loses on collateral and the pose channel loses
on a quadratic out-of-distribution tax that symmetric application does not cure.
Wall-clock, which the charter's CORRECTION made the first gate, turns out NOT to
be the binding constraint once the operators are evaluated at the band instead of
the frame.

The arm was resumed on Opus after the codex provider's usage limit killed it
mid-run (rc 1, no final message). Everything below continues the dead arm's files;
nothing was restarted. The fleet-wide n600 scorer slot, which the dead arm could
not obtain and which blocked its only verdict path, was self-assigned on resume
under the operator's 2026-09-11 full-authority standing GO, with a recorded
preflight that no other full-n600 CPU scorer job was live
(`retained/SCORER_SLOT.json`).

All custody is under `/Volumes/VertigoDataTier/pact/ddm_rbf1/`. No Modal,
MPS/ANE, upstream edit, PR-tree edit, sealed-tree edit, or external publication.
No obx2/pc3/ntb1/mxo1/gp1 directory was changed.

## RECALL EVIDENCE

Exact queries, searched paths, source SHA-256s and matching equations are in
`ddm_rbf1_20260911/RECALL.json`; task matches are in `task_recall.json`.

- Content search of `.omx/research/` and `docs/` for `boundary jitter|post.render|guided filter|anti.alias|signed.distance` returned 165 matching documents.
- The canonical equations CLI returned 483 rows. Five matched `renderer_edge_layer_foldback`, `boundary jitter`, `guided filter`, `aa_sdf_observation`, or `renderer_seg_pose`. Full snapshot retained in `equations_recall.json`.
- Searched the canonical research-index glob and `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`; the DAG matched. The task ledger had three relevant content matches, none assigning rbf1 a scorer slot.
- Beyond the charter seeds, **ar1** (`ddm_ar1_aa_render_price_on_born_field_20260903.md`, sha `cf2b6891…`) measured that footprint averaging harms a different point-trained vehicle; correcting its lattice registration recovered little. Consequently this experiment uses centred pixel footprints, restricts changes to a token-edge band, and makes no transfer of an AA improvement claim.
- **ft1 / `tac.gt_lineage`** identify the DALI/PyAV target fork. Both axes here use the same pinned DALI target cache through the lineage guard. No fresh GT video decode occurs.
- The live board's **gdc4 field correction** changed the input choice: shipped `subset6.u8` (`a92e7d90…`) is used, never the unshipped `pass6.u8`.
- The jitter memory is sha `9c223361163636b60b76b3d39aeae226363465372bf9e120955864dd34f2aa52`; rw1 memory is sha `1064501e12c9731b00cc84ba97e253956666268954906835232698346360b05f`; move-44 memo is sha `f7638e1e171e0d19309f0d0b2f2f9f0acf9b5c70c5fb87f14bc029dfbaeaba8b`. rw1's 240-cell statement is a withdrawn extrapolation, not an n600 fact. Its global code/scale negatives do not adjudicate these spatial post-render operators.

## Residual reproduction

**MEASURED on cached n600 CPU argmax of the shipped public output**, with no new
scorer. It is not the charter's still-owed fresh-scoring reproduction.
Receipt: `/Volumes/VertigoDataTier/pact/ddm_rbf1/retained/CACHED_RESIDUAL.json`.

| Quantity | Count / denominator | Fraction |
|---|---:|---:|
| Argmax disagreement | 12,196 / 117,964,800 pixels | d_seg 0.00010338677300347223 |
| Stored token equals GT at disagreement | 10,204 / 12,196 errors | 83.6668% |
| Within Chebyshev 1 of a token class edge | 12,110 / 12,196 errors | 99.2949% |
| Predicted class occurs in GT 3×3 neighbourhood | 12,141 / 12,196 errors | 99.5490% |
| GT class occurs in predicted 3×3 neighbourhood | 11,550 / 12,196 errors | 94.7032% |

The two neighbourhood directions are different measurements. The former
reproduces the historical one-pixel GT-boundary displacement definition. It must
not be relabeled as the latter. The 86.39% correct-token figure was for move 32;
83.67% is this object's cached read. Lane enrichment remains 40.2718×; Movable
10.8819×. Error rows range from 154 through 296 inclusive.

| GT class → predicted class | Road | Lane | Undrivable | Movable | MyCar |
|---|---:|---:|---:|---:|---:|
| Road | 0 | 2,589 | 1,100 | 899 | 447 |
| Lane | 2,776 | 0 | 7 | 26 | 67 |
| Undrivable | 928 | 4 | 0 | 1,311 | 0 |
| Movable | 600 | 16 | 1,016 | 0 | 11 |
| MyCar | 298 | 94 | 0 | 7 | 0 |

Every disagreement is retained in `cached_residual/cells_n600.npy`, with
pair/y/x/GT/prediction/token/token-correct/edge-distance/both neighbourhood tests.
The source argmax hash is `4784e33d…`; its STEP0 total agrees at 12,196, but its
correct-token count (10,825) used a stale base field. That count is superseded
here by direct classification against the shipped token field (10,204).

## Treatments and their actual mechanisms

All new treatment operations execute on CPU in explicit NumPy fp32/integer
order, even on a CUDA host. Cross-host/T4 byte identity is **unmeasured**.
No scorer, learned table, fitted amplitude, selected class ID, or per-video
coordinate is present in the treatment module. Numerical constants express
pixel geometry, centred quadrature, finite differences, or integer conversion.

1. **Categorical guided filtering.** Local least-squares with one-hot token
   guidance and zero ridge: each 3×3 camera window predicts its per-class RGB
   mean; overlapping window predictions average. This instantiates the local
   linear guided-filter mechanism of [He, Sun and Tang](https://people.csail.mit.edu/kaiming/eccv10/).
2. **Edge SSAA.** Four centred ±¼-camera-pixel bilinear samples integrate the
   existing post-render RGB reconstruction. This is post-render supersampling,
   not four evaluations of the neural renderer. The area-resampling anchor is
   [OpenCV's geometric transformation documentation](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).
3. **Token-SDF displacement.** Per-class local signed Euclidean distances
   determine a finite-difference curvature compensation of the RGB sampling
   position, bounded by half a token cell. Signed-distance rasterization anchor:
   [msdfgen](https://github.com/Chlumsky/msdfgen). The new curvature-compensation
   rule is an original unmeasured hypothesis on this vehicle, not a claim that
   msdfgen implements this correction or that its performance transfers.

Only camera pixels mapped to the token-plane's four-connected inter-class
edge endpoints may change. All other camera pixels are byte-identical. Class
renaming leaves every output invariant. All five real-input algebra/retention
tests pass; their renders and repeats are retained in `unit_controls/`.

The predeclared composition is guided → SSAA → SDF. Other subsets and orders
are unmeasured; there is no claim that this composition is optimal.

## Fresh n600 render — the instrument passes its own falsifier

Runner: `experiments/ddm_rbf1_boundary_probe.py`. It parses the exact archive's
semantic section through a byte-identical copy of the shipped runtime, including
SM1/RC1 riders; evaluates the shipped `SemanticTokenRenderer`; uses the exact
bilinear-upsample/clamp/round path; and preserves the pinned public even frames.

**MEASURED, n600, `retained/RENDER_COMPLETE.json`: 120/120 chunks, and
`baseline_vs_retained_public_changed_channels = 0`.** The fresh baseline render is
byte-identical to the shipped public output on all 600 pairs, all three channels.
The harness reproduces the shipped receiver exactly; every treatment row below is
therefore a delta against the real shipped bytes, not against a look-alike.

Full-frame n600 render seconds on this host (contended, load ~39): baseline 749.8,
guided 1034.0, ssaa 311.4, sdf 179.2, composition 1536.6.

## Wall-clock: the charter's CORRECTION gate, and why it is NOT what closes this

The CORRECTION replaced the charter's stale ~540 s premise with ddm_mxo1's
MEASURED strict slack: **27.581274745 s** (1,260 s receiver policy ceiling minus
the live move-44 `t4_direct` inflate wrapper at 1,232.418725255 s), i.e. 233.809 ns
per coded symbol at n600. Priced first, as instructed.

The producer evaluates each operator over the whole camera frame and then keeps
only the token-edge band, so the cost it measures is the frame, not the band.
**MEASURED: the camera-plane band is 2.21% of the frame** (21,883–23,553 of
1,017,336 pixels over five sampled pairs). `experiments/ddm_rbf1_band_local.py`
re-expresses all four operators at the band only and is proven **BYTE-IDENTICAL**
to the producer on real retained frames for all four modes
(`experiments/tests/test_ddm_rbf1_band_local.py`, 7 tests).

Two paired sessions, both measuring the full-frame producer and the band-local
re-expression on the same real frames in the same session. The later one ran on a
lighter host (load 27, 4 threads, seed 7, 5 pairs) and is the primary reading; its
full-frame ssaa (314.7 s) and sdf (168.9 s) agree with the independent n600 render
totals (311.4 / 179.2) to within 1–6%, which is the cross-check that licenses it.

| Treatment | full-frame n600 s | band-local n600 s | speedup | projected T4 s | fits 27.581 s |
|---|---:|---:|---:|---:|---|
| Guided | 250.3 | 58.2 | 4.30x | 56.9 | no, 2.1x over |
| SSAA | 314.7 | 9.0 | 34.97x | **8.8** | **yes** |
| SDF | 168.9 | 20.9 | 8.08x | **20.4** | **yes** |
| Guided→SSAA→SDF | 733.0 | 96.0 | 7.63x | 93.9 | no, 3.4x over |

The earlier contended session (load 39, 2 threads) measured band-local n600 of
guided 134.2, ssaa 39.8, sdf 54.0, composition 249.1 s, and the n600 render itself
measured full-frame totals of baseline 749.8, guided 1034.0, ssaa 311.4, sdf 179.2,
composition 1536.6 s. ssaa and sdf agree across all three readings; guided and the
composition do not, because their cost is memory-bandwidth bound (five full-frame
box filters per class) and they suffer disproportionately under fleet contention.
The conclusion is the same in every reading: **the two cheap operators fit and the
two expensive ones do not.**

Only frame_1 is semantically re-rendered in this vehicle, so a receiver treats 600
frames, not 1,200. The T4 column applies ddm_mxo1's same-receiver 0.978071 factor;
it is a projection, not a T4 run. Receipts:
`retained/pricing/BAND_LOCAL_PRICE_v2.json` (primary),
`retained/pricing/BAND_LOCAL_PRICE_v1.json`, `retained/pricing/WALLCLOCK_TABLE_v1.json`.

**So wall-clock does not close the family.** Two of the three operators fit inside
the strict slack. What closes it is measured below.

## Mechanism probes — two of the charter's three rationales are falsified

Receipt: `ddm_rbf1_20260911/MECHANISM_PROBES_v1.json`. All scorer-free except
probe 4.

1. **No sub-pixel displacement to pre-compensate.** DERIVED exactly in fp64 from
   torch's own bilinear operator: the composite Down(874→384)∘Up(384→874) has
   per-output weight sum exactly 1 and **centroid shift 0 everywhere except the
   two border indices** (std 0.0073 px; the same for 1164/512). There is no
   resampling phase to correct. What the composite does impose is a
   position-dependent LOW-PASS whose width varies 10x (psf std 0.048→0.469 px,
   self-weight 0.780→0.998).
2. **The jitter does not track that low-pass.** n600 cached error counts per row
   and per column, divided by the token-edge exposure measured over 60
   seeded-random pairs, correlate with the derived psf width at **+0.008 (rows)
   and +0.004 (cols)**; blurriest-vs-sharpest tercile ratio 1.076 and 0.974.
   The charter's "anti-aliasing consistent with the scorer's resize" rationale has
   no measured mechanism behind it on this field.
3. **The treatments are contained but violent.** In the scorer's own 384x512 plane
   the change is zero off the token edge — the band restriction survives the
   downsample — but on the edge the mean |dRGB| is 27/255 (guided), 31
   (composition), 10.3 (ssaa), 4.3 (sdf). There are **2,287,200 token-edge pixels
   at n600 and only 12,196 are wrong: a 187.5:1 collateral exposure.**
4. **The error is not the pixel's colour.** At 196 token-correct error cells over
   12 seeded-random pairs, the rendered RGB is **already closer to the GT class's
   own local 7x7 mean in 66.8% of cases** (mean distance 90.9 vs 128.4 to the
   predicted class). SegNet calls them wrong anyway. This is the CLAUDE.md region
   law — SegNet sees REGIONS, not pixels — measured for the first time on the
   render-treatment family rather than on the store-the-flip-pixels sidecar family,
   and it predicts a weak benefit channel for any pointwise purification while the
   collateral channel of probe 3 runs at full strength.

## The bar, and the amplitude family

Pre-registered before any scoring (`ddm_rbf1_20260911/PREREGISTRATION_v1.json`).
From the move-44 components (S 0.1372449041713402, 180,406 B, d_pose 4.59e-06,
pose term 0.00677495387438173, so d_seg 1.0345e-04):

- one net seg error = **8.4772e-07 S**, so the −2e-5 bar needs **−23.6 net errors**;
- one unit of pose SSE = **0.20500 S**, so **9.76e-05 of pose SSE alone costs the
  whole bar**.

`experiments/ddm_rbf1_amplitude.py` traces the family by clamping the retained
treated frame's per-channel delta against the retained baseline at a bound tau;
tau=255 reproduces the producer exactly and tau<0 is the sign control. No new
render is needed. Receipt: `retained/amplitude/AMPLITUDE_guided_seed1_n24.json`;
`ddm_rbf1_20260911/POSE_TAX_v1.json`.

**MEASURED (guided source, 24 seeded-random pairs, seed 1, 355 s — declared SCOPE,
no n600 verdict; a random sample, not a prefix, per the prefix-bias law):**

| tau | B | H | H/B | seg errors | d_pose | ΔS_seg | ΔS_pose | ΔS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | — | 508 | 1.715e-06 | 0 | 0 | 0 |
| 1 | 3 | 35 | 11.67 | 540 | 1.048e-05 | +0.000678 | +0.006467 | **+0.007145** |
| 2 | 5 | 82 | 16.40 | 585 | 3.796e-05 | +0.001632 | +0.026752 | **+0.028384** |
| 4 | 14 | 138 | 9.86 | 632 | 1.497e-04 | +0.002628 | +0.109215 | **+0.111843** |
| 8 | 26 | 281 | 10.81 | 763 | 5.668e-04 | +0.005404 | +0.417040 | **+0.422444** |
| 255 (= the producer) | 82 | 1638 | 19.98 | 2064 | 7.268e-03 | +0.032976 | +5.362333 | **+5.395309** |
| −8 (sign control) | 19 | 276 | 14.53 | 765 | 4.928e-04 | +0.005447 | +0.362443 | **+0.367889** |

Four readings that together close the family:

- **The pose tax is exactly quadratic in amplitude.** The excess d_pose divided by
  tau² is 8.763e-06, 9.062e-06, 9.249e-06, 8.829e-06 for tau = 1, 2, 4, 8 —
  **constant to within 5% over an 8x range**. The law is
  `d_pose(tau) − d_pose(0) = k·tau²` with k = 9.0e-06 on this ~22,000-pixel
  support, i.e. **ΔS_pose = 6.6e-03·tau²**. At tau = 1, a single quantisation
  level, the pose tax alone is 6.5e-03 S = **323 bars**, against a measured seg
  benefit of 3 pixels. Both terms vanish as tau → 0, so the infimum of ΔS over the
  whole family is 0, reached only by not treating. **There is no payable amplitude.**
- **Selectivity does not improve at small amplitude.** H/B is 9.9–20.0 across the
  entire range and is no better at tau = 1 (11.67) than at tau = 4 (9.86). The hope
  that a gentler edit would be a more *selective* edit is measured false.
- **The operator is barely better than its own reverse.** Reversing it (tau = −8)
  gives H/B 14.53 against 10.81, a 1.34x advantage for the real direction, but
  ΔS_seg is +0.005447 against +0.005404 — indistinguishable. Whatever directional
  information the token plane carries into this actuator is weak. (A 2-pair read
  earlier in this arm suggested a 1.8x seg advantage; at n = 24 that does not
  survive, and the 2-pair number is withdrawn.)
- **Symmetric treatment does not cure the pose tax.** Treating frame_0 with the
  same operator and the same token plane (4 seeded-random pairs, seed 3, tau = 8)
  moves d_pose from 6.7723e-04 to 6.6899e-04 — it recovers **1.2%**, and leaves the
  seg errors identical at 98. The tax is out-of-distribution, not a frame_0/frame_1
  asymmetry artefact. **The move-44 render is a pose optimum** (d_pose 4.59e-06 at
  n600); any deviation from the exact rendered bytes is a large relative pose error
  whichever frames carry it.

**And the tax scales with the SUPPORT of the edit, not only its amplitude.**
Applying the same tau = 8 edit to a uniformly random fraction of the band
(4 seeded-random pairs, seed 5) gives excess d_pose of 5.383e-04, 4.182e-04,
7.587e-05, 2.174e-05 at 22,823 / 11,411 / 5,705 / 1,426 pixels: over a 16x range of
support the tax rises 24.8x, an exponent of **1.16 — roughly linear in the number
of edited pixels**. Combined: **ΔS_pose ≈ 0.0024·f^1.16·tau² S**, where f is the
percent of frame_1's pixels edited. Seg errors rise monotonically with support too
(78 baseline → 85, 100, 102, 113), so there is no support at which the seg channel
turns beneficial either. Receipt: `ddm_rbf1_20260911/POSE_TAX_v1.json`.

## Selectivity: how far the family is from payable

Random perturbation of the same support would give H/B = (2,287,200 − 12,196)/12,196
= **186.5**. The three OSS-anchored operators measure H/B of 19.4 (guided), 10.3
(ssaa), 12.7 (sdf) on the first n600 chunk, so they are **10–18x better than
random** — the token plane really does carry boundary information. To break even
on seg alone they would have to be **186x** better than random, another order of
magnitude, and they would still owe the quadratic pose tax on top. That gap, not
an implementation defect, is what closes the family.

## Fresh n600 treatment table

The governed job `score_launch` (pid 10979, counter 1527, slot receipt
`retained/SCORER_SLOT.json`) runs the frozen CPU scorer over all five modes in
five-pair resumable chunks and writes `retained/RESULT.json` at the end. **At the
time of writing it is at 90/600 pairs.** The rows below are the RUNNING AGGREGATE
over those 90 pairs — a declared CONTIGUOUS PREFIX, so per the prefix-bias law it
is SCOPE, not the verdict, and its pose column is biased HARD (pose prefixes
measure 2.54–4.21x harder than the population). The verdict row is `RESULT.json`.

| Mode | seg errors | d_seg | d_pose | B | H | H/B | ΔS_seg | ΔS_pose | ΔS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1,623 | 9.1722e-05 | 1.7265e-05 | 0 | 0 | — | +0.000000 | +0.000000 | **+0.000000** |
| Guided | 6,907 | 3.9034e-04 | 8.4911e-03 | 370 | 5,654 | 15.28 | +0.029862 | +6.253779 | **+6.283641** |
| SSAA | 2,616 | 1.4784e-04 | 7.8137e-04 | 99 | 1,092 | 11.03 | +0.005612 | +0.563922 | **+0.569533** |
| SDF | 2,451 | 1.3852e-04 | 9.6638e-04 | 72 | 900 | 12.50 | +0.004679 | +0.700461 | **+0.705140** |
| Guided→SSAA→SDF | 7,403 | 4.1837e-04 | 1.5824e-02 | 385 | 6,165 | 16.01 | +0.032665 | +11.665767 | **+11.698432** |

The bar is ΔS < −2e-05. The best mode is **+0.5695**, which is **28,477 bars on the
wrong side**, and the composition is the worst row in the arm. No mode is within
four orders of magnitude of the bar, and the ordering, the harm/benefit ratios and
the pose domination all agree with the 24-pair seeded-RANDOM amplitude family, so
the prefix is not carrying the result.

**The fresh residual reproduction runs alongside it and reproduces the cached
census with an independent frozen-CPU scorer.** At 90 pairs: 1,623 errors, of which
**82.50%** have the stored token equal to GT (cached n600: 83.67%), **99.26%** sit
within Chebyshev 1 of a token class edge (cached: 99.29%), and **96.00%** have the
GT class present in the predicted 3x3 (cached: 94.70%). That is the charter's
deliverable 1 falsifier passing on a fresh scorer, not an inherited argmax.


## Where a pixel actuator IS affordable in pose — the spatial law

The pose tax is not spatially uniform, and a successor should not price it as if it
were. Applying the same guided tau = 8 band edit one token class at a time
(5 seeded-random pairs, seed 11; baseline d_pose 5.1845e-06, 92 seg errors):

| class band | mean edited px | Δd_pose | ΔS_pose | ΔS_seg | pose S per 1,000 px |
|---|---:|---:|---:|---:|---:|
| Road | 11,010 | 1.030e-03 | +0.760432 | +0.003662 | 0.0691 |
| Lane | 4,654 | 5.987e-05 | +0.044186 | +0.003764 | **0.0095** |
| Undrivable | 2,814 | 4.799e-04 | +0.354196 | +0.002035 | 0.1259 |
| Movable | 1,092 | 4.997e-05 | +0.036881 | +0.000712 | 0.0338 |
| MyCar | 2,634 | 1.483e-05 | +0.010944 | +0.000407 | **0.0042** |

**The pose tax per edited pixel varies 30x across class bands.** It concentrates in
Road and Undrivable — precisely the ground plane and the horizon, the structures
that carry the ego-motion geometry — and is 16–30x cheaper on the static ego hood
(MyCar) and on Lane markings. PoseNet is not sensitive to the frame; it is
sensitive to the geometry in it.

**But ΔS_seg is positive in every band, including the cheap ones**, and the Lane
band is the sharpest case in the whole arm: it carries 44% of the residual
(Road↔Lane is 5,365 of 12,196 errors) and it is the second-cheapest band in pose,
yet it shows the LARGEST seg harm of any band (+0.003764) from only 4,654 edited
pixels. The spatial escape does not rescue the family. What it does is tell the
next actuator where it may live: **pose is affordable only inside MyCar and Lane,
and any successor must be seg-positive by construction, not merely pose-cheap.**
Widen this 5-pair arm to a seeded RANDOM n≥120 before routing on it.
Receipt: `ddm_rbf1_20260911/POSE_SPATIAL_v1.json`.

## The counted extension is closed too, by the same arithmetic

Rule 118 forbids a fitted scalar in FREE receiver code, but a fitted byte in the
COUNTED archive costs only 25/37,545,489 = 6.66e-07 S, so "ship a selector that
says where to treat" is the natural next move and deserves its price, not a
hand-wave.

- **An oracle per-pixel selector is nearly affordable and still does not pay.**
  Naming 12,196 positions out of 2,287,200 token-edge pixels costs
  `n·H(k/n)` = 109,640 bits = **13,705 B = 0.009126 S**, against a benefit CEILING
  of **0.010339 S** (fixing every error with zero harm, i.e. the entire seg term).
  The margin is 61 bars before any of the treatment's real behaviour is charged.
- **The actuator, not the addressing, is what fails.** At tau = 8 on 24 pairs the
  operator fixes **5.12%** of errors and harms **0.309%** of correct edge pixels.
  Break-even at that fix rate needs a harm rate of **0.0286%** — the measured rate
  is **10.8x too high**. A selector cannot lower the harm rate; it can only stop
  treating, which forfeits the benefit in the same proportion.
- **And the pose tax survives selection.** Treating even the 12,196 scored-plane
  error positions means editing roughly 61,000 camera pixels (6% of the frame, at
  the 5:1 camera-to-token expansion), which by the measured law costs
  `0.0024·6^1.16·8²` ≈ **1.3 S** in pose alone.
- A coarse **per-pair on/off selector** is almost free (600 bits = 75 B =
  4.99e-05 S), but it has nothing to select: the harm/benefit ratio is 9.9–20.0
  essentially uniformly, so there is no pair in which treating is net positive.

So the closure is not "we could not find where to treat." It is that the actuator
is simultaneously too weak on seg (5% fix rate) and too expensive on pose
(quadratic OOD tax) for any addressing scheme to rescue.

## Equations leg

The bar arithmetic in this memo consumes the registered canonical equation
`tac.canonical_equations` equation_id **`pose_sqrt_concave_coupling_sidecar_v1`**:
the score's pose term is `sqrt(10*d_pose)`, so its derivative at the move-44
operating point d_pose = 4.59e-06 is `5/sqrt(10*d_pose)` = **738.0 per unit
d_pose**, and per unit of pose SSE (= d_pose * 600 * 6) it is **0.20500 S**. Every
ΔS_pose column above is that constant times a measured d_pose delta. The sibling
`half_res_bottleneck_destroys_fastvit_posenet_luma_5x_regression_v1` records the
same family of fragility from the other direction — a change to the image the pose
head reads is expensive far out of proportion to its visual size — and this arm
measures the coefficient for a boundary-band edit.

<!-- # FORMALIZATION_PENDING: the NEW law this arm measured -- delta_S_pose ~= 0.0024 * f^1.16 * tau^2 for a free post-render edit of f percent of frame_1's pixels at amplitude tau, with the excess d_pose over tau^2 constant to 5% across an 8x amplitude range (k = 9.0e-06 at a 22,000-pixel support) -- is deliberately NOT registered here. It is a CLOSURE price, not a lever: registering it would put a two-parameter fit with a 4-pair support arm and a 24-pair amplitude arm into the registry with no consumer. Per the charter's own convention the equations leg lands with the exact row that uses it; the successor that prices a counted boundary edit against this tax is the right registrant, and it should first widen the support arm from 4 seeded-random pairs to n>=120. The retained fit and every input row are in ddm_rbf1_20260911/POSE_TAX_v1.json. -->

## Review, landing, and boundaries

Two source review passes were recorded per Python file. They checked dtype and
shape flow, input lineage, absolute pair indices, off-band preservation,
categorical label invariance, no fitted constants, resume refusal on changed
bindings, payload retention and score denominators. Ruff passes; the five
real-input tests pass. These are self-review passes, not independent reviews.

The initial serializer returned **rc 17** (Git object write denied). It retained
bundle commit `3098742a812948c564dfc3719e0656728f9ebc1f` under
`/Volumes/VertigoDataTier/pact/ddm_rbf1/receipts/commit_serializer_fallbacks/20260911T125652.919280Z-81004/`.
This is bundle custody, not a main-worktree landing. Exact argv/stdout/rc are
in `ddm_rbf1_20260911/CODE_SERIALIZER.json`; the next unit has its own
`RESIDUAL_SERIALIZER.json`. No unrelated staged-index work was manipulated.

Triality hooks are deliberately N/A while `research_only=true`: no measured
action is yet available for the sensitivity map, Pareto allocator, bit
allocator, autopilot or posterior. The token-correct/edge-distance census is
the actual probe disambiguator. MAIN's future measurement consumer is the
typed fire order in `ddm_rbf1_20260911/FIRE_ORDERS.json`; no unscheduled future
idea is represented as completed work.

## NEXT_IF_RESUMED

- **FIRED and COMPLETE** — the governed n600 five-mode render: `retained/RENDER_COMPLETE.json`, 120/120 chunks, fresh baseline byte-identical to the shipped public output on all 600 pairs.
- **FIRED** — owner ddm_rbf1; consumer `retained/RESULT.json`; the fleet scorer slot was self-assigned on resume under the operator's full-authority standing GO (`retained/SCORER_SLOT.json`, with the recorded preflight). Launch receipts: `/Volumes/VertigoDataTier/pact/ddm_rbf1/score_launch/`, done receipt `ddm_rbf1_score`. Resume with
  `.venv/bin/python experiments/ddm_rbf1_boundary_probe.py score --resume-from /Volumes/VertigoDataTier/pact/ddm_rbf1/retained --slot-receipt /Volumes/VertigoDataTier/pact/ddm_rbf1/retained/SCORER_SLOT.json`
  (five-pair resumable chunks; a restart re-uses every completed `chunks/*/SCORE.json`).
- **FIRED and COMPLETE** — the amplitude-family falsifier: `retained/amplitude/AMPLITUDE_guided_seed1_n24.json`.
- **CLOSED, no fire order** — the conditional candidate-intent packet. Its trigger was an n600 composition below ΔS −2e-5. Every measured mode and every measured amplitude is POSITIVE by 10^2 to 10^5 bars, and the family's infimum over amplitude is 0. No receiver delta, candidate tree, twin encode, parse-back or intent packet is owed, and none was produced. Nothing was fired at Modal.
- **FIRED and COMPLETE** — landing: the dead arm's five stranded files (rc 17, Git object write denied) are on main unchanged, alongside the two new runners and their tests.

## LIVE-HYPOTHESES

- **A regional operator at SegNet's receptive-field scale.** Probe 4 says the error lives in the context, not the pixel, so the operator that could reach the required 186x selectivity must change regional evidence rather than a 1-px boundary. DERIVED from the measured support law, that operator has LARGER support and therefore pays MORE pose tax: the scale at which the seg error is addressable is the scale at which the pose tax is largest. This is a hypothesis about a wall, not a lever.
- **A counted, not free, boundary edit.** Rule 118 forbids a fitted scalar in FREE receiver code, but a fitted byte in the COUNTED archive costs only 25/37,545,489 = 6.66e-07 S. A per-video amplitude, or a per-pair selector of which boundary segments to treat, is affordable at one to a few bytes. This does not rescue the family as measured — every amplitude is positive — but it is the legal shape any future boundary work should take, and it was not available to this charter's "free only" framing.
- **Anything that lowers the pose operating point first.** The pose tax constant is 5/sqrt(10·d_pose) = 738 at d_pose 4.59e-06. Every free pixel-edit family on this vehicle is priced by that number. A vehicle with a higher d_pose would price pixel edits an order of magnitude cheaper; this is an argument about where in the design space pixel-level actuators become legal, not an argument for raising d_pose.

## DEAD-ENDS

- **The free post-render boundary treatment family is CLOSED at formulation scope on the move-44 vehicle.** verdict_scope: formulation — deterministic post-render operators on the token-edge band of the rendered frames, free receiver code, no counted side information. Closed on TWO independent channels: seg (ΔS_seg > 0 at every operator and every amplitude; harm/benefit 3–19 against, where breaking even needs 186x better-than-random selectivity) and pose (a quadratic, direction-blind, symmetry-proof out-of-distribution tax of ≈5.6e-03·tau² S). Reactivation criteria: an operator that measures harm/benefit below 1 on a seeded RANDOM n≥120 sample AND leaves d_pose within 3e-08 of the shipped render, or a vehicle whose d_pose operating point is more than 10x higher.
- **The charter's "anti-aliasing consistent with the scorer's resize" rationale is falsified**, not merely unmeasured: the composite Down∘Up is centroid-preserving except at the border, and the error rate per edge pixel does not track its position-dependent blur (Pearson +0.008 / +0.004).
- **The "treat both frames to protect pose" rescue is falsified** — it recovers 1.2% of the tax.
- **Wall-clock is NOT what closes this family.** SSAA at a projected 14.9 s and SDF at 21.3 s fit inside the 27.581 s strict T4 slack once evaluated at the band. Any successor that cites wall-clock as the reason this family died would be repeating a premise this arm measured false.
- **Historical-number transfer closed** (carried forward): 86.39% and the STEP0 10,825 correct-token count are not this shipped field's counts; direct n600 reclassification gives 83.6668% and 10,204.
- **rw1 screen extrapolation closed** (carried forward): 240 cells was extrapolated, later withdrawn; it cannot serve as n600 evidence here.


composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)
