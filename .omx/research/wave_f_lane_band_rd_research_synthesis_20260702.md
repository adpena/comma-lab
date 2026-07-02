# Wave-F R1 — the OPTIMAL lane-band rate-distortion code (full online + OSS + papers synthesis)

**Status:** RESEARCH SYNTHESIS (2026-07-02). Advisory/build-only; pointer 0.19110 UNMOVED (moves only via a
byte-closed n600 exact row). Companion to the design authority
`.omx/research/wave_f_optimal_lane_band_rd_code_design_20260702.md` (the L1-L5 framing). Operator directive:
*"deep math + full online research + OSS authority, explore ALL techniques/algorithms/methods, optimal solution
for optimal score lowering; we have rate budget to give and can trade off optimally for d_seg and d_pose."*

## TL;DR verdict

**L1-L5 is CONFIRMED by first-principles rate-distortion theory** — it is the operational decomposition of ONE
exact objective (the distributed-indirect / Wyner-Ziv Lagrangian below). No surveyed technique BEATS the framing;
the survey **REFINES** it in five concrete, OSS-backed ways and adds ONE dual-axis lever (SE(3) continuous-time
ego-trajectory) that lets the granted rate budget lower **d_pose** as well. It also produces ONE clean KILL
(don't build a learned VQ / deep task-quantizer at this scale — dominated). The recommended code composes six
stages, each on the residual of the prior, and lands an estimated **~1-4 KB @ n600 (rate +0.0007 to +0.003)**
from the naive 220 KB (+0.147) — turning the whole question into "does the band lower d_seg through R at all?",
which is the real (unmeasured) gate, not the coding.

## The one objective (the deep-math backbone — this is L1-L5 unified)

The single most load-bearing result of the survey: **distributed indirect source coding with decoder side
information** (Rate-Distortion Analysis of Distributed Indirect Source Coding, PMC12385534) gives the EXACT
Lagrangian our problem minimizes:

```
minimize   Σ_i [ I(W_i ; X_i)  −  I(W_i ; Y) ]  +  λ · E[ d(T, T̂) ]
```

Map to our lane-band code term-by-term (this IS the "everything is one object"):

| Symbol | Our object |
|---|---|
| `X_i` | raw per-pair lane geometry (camera-frame coeffs) |
| `W_i` | the **counted** lane codes we ship in archive.zip (the quantized payload) |
| `Y`   | **decoder side-information = the stored ego-screw ξ** (already paid for d_pose, `tac.lie` #193/#194) |
| `T, T̂` | task-oriented reconstruction = **SegNet argmax partition** → `d(T,T̂)` = **d_seg-through-R** |
| `λ` | the rate/distortion knob (KKT: allocate until `∂d_seg/∂byte = 25/(100·37,545,489) ≈ 6.66e-9`) |

Two theorems make L1-L5 rigorous rather than heuristic:

1. **The `−I(W_i ; Y)` term IS L1.** For sources **conditionally independent given the side-info Y**, the paper's
   main result collapses the per-encoder rate to `R_m ≥ I(X_m ; W_m) − I(W_m ; Y)`. Reading it in our variables:
   *the rate you pay for the lane code is reduced, byte-for-byte, by its mutual information with the ego-pose.*
   That is exactly "reuse the stored ξ to warp per-pair; ship only what ξ can't predict" — **motion-compensated
   prediction, formalized.** L1's "DOMINANT lever" claim is theoretically earned.
2. **The solver is a distributed Blahut-Arimoto** (paper's Eq. 29 / three-step iteration), NOT a hand-tuned sweep —
   satisfies the CLAUDE.md "prefer solvable math over arbitrary sweeps" mandate. Encoders alternate
   (quantize→bottleneck→Bayes-decode) minimizing source-info AND side-info-redundancy jointly.

**Coding-for-machines equivalence (why we don't reconstruct geometry, only the partition):** the task-RD survey
(RD Theory in Coding for Machines, arXiv 2305.17295) proves `R_X(D;T) = R_Y(D;T)` — direct-to-task and
model-splitting achieve the SAME optimal rate, and *"the only consideration that affects rate for machines is the
task and its distortion metric."* Multiple geometries that yield the identical SegNet argmax must cost the identical
bits. This is the license for L3: quantize to `d_seg`-tolerance, never to geometric error. Its Theorem 3/4 further
prove supervised (task-label-driven) optimization **strictly** beats reconstruction-error proxies — so the quantizer
must be driven by `∂d_seg/∂coeff`, not by MSE on the coeffs.

## Ranked techniques (by expected S-reduction; CONFIRM / REFINE / BEAT / KILL vs L1-L5)

### #1 — [L1, DOMINANT · REFINED] Ego-motion factorization via a shared **SE(3) continuous-time B-spline**, used as Wyner-Ziv decoder side-info — *the only dual-axis lever (helps d_pose too)*

- **CONFIRM + REFINE.** L1 says "reuse ξ to warp per-pair." The refinement: represent ego-motion ONCE as a
  **cumulative cubic B-spline in SE(3)** (sparse control points, C², locally controllable — Sommer et al.
  "Jacobian Computation for Cumulative B-Splines on SE(3)" arXiv 2201.10602; Lovegrove "Spline Fusion"). Evaluate
  pose/velocity at any pair timestamp deterministically at decode (**FREE** in inflate). Compose with `tac.lie`
  #193/#194.
- **Why this is the #1 lever AND the answer to "spend rate to help d_pose too":** the SE(3) spline control points
  ARE the decoder side-info `Y` in the objective above. They are **dual-paid**: they are the d_pose payload AND the
  lane world→camera warp. Because ego-motion is smooth, **N ≪ 600** control points reproduce the whole trajectory,
  so this *replaces 600 independent 6-DoF pose vectors with a sparse spline* — fewer counted bytes for equal/better
  d_pose. And spending marginal rate on more/better-placed control points **lowers d_pose (better reconstruction)
  AND lowers lane rate (larger `−I(W;Y)` subtraction)**. This is the single lever where the operator's granted
  budget pays on both distortion axes at once.
- **Rate impact:** per-pair counted payload collapses O(600×50) → O(static world model + N spline control points +
  tiny per-pair innovation). Largest single reduction.
- **OSS/refs:** cumulative SE(3) B-spline (basalt / Spline-Fusion lineage); Wyner-Ziv side-info theory
  (distributed video coding, MCFI side-info generation) is the classical analogue but the *object* here is the
  ego-trajectory, not interpolated frames.

### #2 — [L3, task-RD quantization · REFINED with a named mechanism] Sensitivity-driven **reverse-water-filling** bit allocation ≈ **Variational Bayesian Quantization**

- **CONFIRM + REFINE.** L3 says "quantize each coeff to its `∂d_seg/∂coeff`." The refinement names the exact
  operational realization: **Variational Bayesian Quantization** (Yang, Bamler, Mandt, ICML 2020, arXiv 2002.08158)
  — *"a novel extension of arithmetic coding to the continuous domain"* that allocates **bits per coefficient by
  posterior uncertainty**, is **plug-and-play at any rate from ONE representation** (no per-rate retrain), and
  separates model/quantization. Bits-per-coeff `∝ ½ log(sensitivity/λ)₊` = **reverse water-fill** where the
  "sensitivity" is the margin-saliency `∂d_seg/∂coeff` map (#141), NOT posterior variance of a reconstruction.
- **This is where the granted rate is SPENT OPTIMALLY** — KKT stationarity `∂d_seg/∂byte = 25/(100·37.5M)`:
  sub-tolerance precision is deleted (a coeff whose perturbation never moves the SegNet argmax past the R-downsample
  ~1-2.27px tolerance gets ~0 bits); the freed budget is placed where it flips the partition (the codim-1 boundary
  annulus). The task-based-quantization literature (Deep Task-Based Quantization, arXiv 1908.06845; Model-Aware RD
  limits, arXiv 2602.12866; Feature-Preserving RDO for image coding for machines, arXiv 2408.07028) proves the
  optimal strategy for MSE-on-task is VQ of the MMSE **task** estimate and uses **importance maps** (= our
  saliency) — the same structure.
- **Rate impact:** ~6-16× over float64 alone (each scalar 64 bits → ~4-10 task-relevant bits).

### #3 — [L2, temporal prediction · REFINED: AR(1) → optimal temporal transform] **FPCA / Karhunen-Loève (or fixed DCT) basis** on the world-frame coefficient time-series

- **REFINE (mild BEAT of naive AR(1)).** L2 proposes AR(1) on the residual. The 600-pair sequence of world-frame
  coeffs is a **functional time-series**; the **Karhunen-Loève / functional-PCA** basis is *"the best basis in the
  sense that it minimizes total mean-squared error"* (Kosambi-Karhunen-Loève; Cramér-KL harmonic PCA of functional
  time series, ScienceDirect S0304414913000793) — the min-MSE-per-bit temporal transform. AR(1) is the single-tap
  special case; the KLT/DCT captures slowly-varying **road-curvature** structure AR(1) misses. Ship only the
  low-order temporal components.
- **rule-118 discipline:** a **fixed DCT** over the temporal axis is *generic-computable at decode* (FREE) and is
  near-KLT-optimal for a smooth stationary source — **prefer fixed DCT** so no learned basis is counted. A
  learned/video-derived KLT basis would be COUNTED (ship the basis vectors) and is only worth it if it's a handful
  of components; default to DCT to stay clean.
- **Rate impact:** ~5-10× on the temporal axis (post-factorization innovations are near-zero and low-rank in time).

### #4 — [L3 tail · CONFIRM, named OSS] **Range / rANS entropy coding with a fitted Laplacian prior** — `constriction`

- **CONFIRM.** Coefficient innovations are Laplacian (classic; Lloyd-Max-for-Laplacian literature). Entropy-code the
  quantized innovations with **bamler-lab/constriction** (already in-repo per #152): **range coder** (FIFO — matches
  the temporal-causal/autoregressive innovation stream) or **rANS** for i.i.d. codes, *within 0.1% of the Shannon
  entropy*, Python+Rust, deterministic. Understanding-ANS (arXiv 2201.01741) is the canonical write-up.
- **Quantizer detail:** the scalar quantizer under the entropy coder should be **Lloyd-Max / companding matched to
  the Laplacian** innovation pdf (log-concave → symmetric optimal quantizer; arXiv 1212.2144) **BUT read in the
  sensitivity-warped metric** (companding curve = `∂d_seg/∂coeff`) — i.e. #2 and #4 are the same quantizer:
  Lloyd-Max in the task metric, entropy-coded.
- **Rate impact:** ~1.5-3× over the quantized representation.

### #5 — [L4, inter-line correlation · CONFIRM] Code ego-lane centerline + **lateral offsets** for the other 4 lines

- **CONFIRM.** The 5 lanes are ~parallel; the offsets are near-constant → tiny. This is the "ordered structured
  representation" lesson from **MapTR/MapTRv2** (arXiv 2208.14437 / 2308.05736): model lanes as an *ordered
  point-set with permutation-equivalence* / instance queries — borrow the **parameterization** (Frenet/arc-length
  centerline + offset), NOT the perception network. AV-planning evidence that this is the compressive form: a
  degree-5 polynomial represents a 5-s trajectory in **8.7-40.8% of the point-sample space** (Akaike-optimal;
  arXiv 2407.13431), and openpilot itself emits lanes/path as BEV curves.
- **Rate impact:** ~2-5× on the 5-lines axis.

### #6 — [L5, dash phase = ξ · CONFIRM] Dash phase = ego-forward-distance, derived from the SE(3) spline → **FREE**

- **CONFIRM.** Per the dash-gap FP memo, dash phase IS a component of the ego translation, already in the #1 spline.
  Only period/duty ship (near-constant, entropy-coded to a few bytes). Composes with the range-dependent dash gate.

### KILL — Learned VQ / deep task-based quantizer / CompressAI entropy-bottleneck as the *primary* codec

- **DOMINATED at 30k-scalar scale.** Deep task-based quantization (arXiv 1908.06845) and end-to-end learned
  entropy models (CompressAI: Ballé entropy-bottleneck / Gaussian-conditional) approach the indirect-RD limit **but
  require a learned codebook/prior** — counted video-derived bytes + parse-back complexity. At 30k scalars a
  well-fitted **parametric (Laplacian) prior + task-tuned Lloyd-Max quantizer + range coder** captures essentially
  all the structure; a learned prior's overhead is not amortized. Answering the operator's #4 directly: **fixed /
  parametric prior wins at this scale; a learned prior is over-engineering.** Keep CompressAI/VBQ-style learned
  priors as a *reactivation path* only if a measured residual shows structured, non-Laplacian, high-entropy tails
  the parametric prior can't reach.

## The recommended optimal code (composition, in order — each stage on the residual of the prior)

```
raw per-pair camera-frame lane geometry  (~240 KB float64, ~220 KB brotli, +0.147)
  │
  ├─(1) SE(3) cumulative B-spline ego-trajectory  → decoder side-info Y (dual-paid w/ d_pose)
  │       warp per-pair geometry to world/ground frame  (−I(W;Y): motion-compensated prediction)
  │
  ├─(3) fixed-DCT (≈KLT/FPCA) over the 600-pair temporal axis  → keep low-order components  [FREE basis]
  │
  ├─(2) task-RD quantize each surviving coeff by ∂d_seg/∂coeff  (reverse water-fill / VBQ; KKT operating point)
  │
  ├─(5) inter-line: ego-lane + lateral offsets (near-constant)
  │
  ├─(6) dash phase from the spline (free); ship period/duty only
  │
  └─(4) range/rANS entropy-code the Laplacian innovations  (constriction, #152)  → archive.zip
                          estimated ~1-4 KB @ n600  →  rate +0.0007 to +0.003
```

**Rate budget arithmetic (byte targets):** rate term `= 25·bytes/37,545,489`. To keep the rate cost under the
band's plausible d_seg win, target: +0.005 → 7,509 B · +0.01 → 15,018 B · +0.02 → 30,036 B. The composed code
targets ~1-4 KB, i.e. rate cost ~1-2 orders of magnitude below the +0.005 floor — so **the coding is not the
binding constraint; the d_seg win is.**

## Where the granted rate budget ALSO buys d_pose (the ξ dual-use, made concrete)

Stage (1) is the answer. The SE(3) B-spline control points are simultaneously (a) the d_pose payload and (b) the
lane side-info `Y`. Spending marginal rate there:
- **lowers d_pose** — denser/better-placed control points reconstruct the ego-trajectory more faithfully (√(10·d_pose)
  term); and
- **lowers lane rate** — a better `Y` enlarges the `−I(W;Y)` subtraction in the objective (better prediction →
  smaller lane innovations).
This is the ONLY surveyed lever that improves both distortion axes with the same bytes. The tradeoff (control-point
count vs d_pose vs lane-innovation size) is a **measurable 1-D knob**, not a guess.

## Honest risks (adversarial)

1. **The d_seg WIN is unmeasured through-R at n600 byte-closed.** Even a perfect code NETS negative if the band's
   realized d_seg improvement < rate cost. The optimal code drives rate to ~1-4 KB precisely so this reduces to
   "does the analytic band lower d_seg AT ALL through R?" (design cites ~0.00087 analytic-lane authority; erasure
   ~20.8% of lane mass). **That is the real gate — measure it first, before over-investing in the codec.**
2. **Task-RD quantization needs an accurate, STABLE `∂d_seg/∂coeff` across 600 pairs.** Mis-estimated sensitivity
   either wastes bytes (too fine) or spills d_seg (too coarse). Mitigate: estimate sensitivity **through R on real
   n600**, never a proxy (per the surrogate-vs-exact NO-FAKE rule).
3. **SE(3) control-point count vs d_pose is a real tradeoff** — too-sparse raises d_pose. Measurable; don't
   over-sparsify blindly.
4. **rule-118 hygiene:** a learned KLT/FPCA basis or learned prior is COUNTED (video-derived). Default to the
   **fixed DCT + parametric Laplacian prior** so the counted payload is ONLY the coefficients, and the transform +
   entropy-decode + spline-eval + warp are all generic-algorithm FREE in inflate. Do not smuggle a learned basis
   into inflate.py as "code."
5. **Decode-consistency + determinism gates (Wave-E) must be PRESERVED** — the new code's decoded render must equal
   the training render bit-exact through R; default-off byte-identical must stay 7/7; entropy-decode + spline + DCT
   + warp all O(1)/pixel within the 30-min budget.

## OSS pointers (draw from; nothing to contribute back)

- **bamler-lab/constriction** — range + rANS entropy back-end, Python+Rust, <0.1% above entropy (already #152). [stage 4]
- **Variational Bayesian Quantization** (Yang/Bamler/Mandt, ICML 2020, arXiv 2002.08158) — sensitivity/posterior-driven
  variable-rate quantizer, plug-and-play any rate. [stage 2 mechanism]
- **CompressAI** (InterDigital) — entropy-bottleneck / Gaussian-conditional modules — REACTIVATION ONLY if a measured
  residual defeats the parametric prior. [KILL default]
- **MapTR / MapTRv2** (hustvl, arXiv 2208.14437 / 2308.05736) — borrow the ordered-point-set / permutation-equivalent
  lane parameterization (representation, NOT the perception net). [stage 5]
- **Cumulative SE(3) B-spline** (Sommer et al. arXiv 2201.10602; Lovegrove Spline-Fusion; basalt) — the ego-trajectory
  parameterization for the dual-use ξ. [stage 1]
- **Lloyd-Max-for-Laplacian companding** (arXiv 1212.2144) — the scalar quantizer, read in the task-warped metric. [stage 2/4]
- **Distributed indirect source coding RD** (PMC12385534) + **RD for Coding-for-Machines** (arXiv 2305.17295) — the
  unifying objective + the coding-for-machines equivalence theorems. [backbone]

## Bottom line for the build agent

Build the six-stage composition above. It is CONFIRMED-optimal by the distributed-indirect Wyner-Ziv Lagrangian
(one solvable objective, not a sweep). The two highest-leverage, non-obvious refinements are (1) the **SE(3)
continuous-time ego-trajectory** as the dual-paid side-info (helps d_pose AND lane rate) and (2) **VBQ-style
reverse-water-fill task-RD quantization** driven by `∂d_seg/∂coeff`. Ship a **fixed DCT + parametric Laplacian +
constriction range coder** to stay rule-118-clean and skip the dominated learned VQ. Expected ~1-4 KB @ n600
(+0.0007…+0.003 rate). **Then the whole thing lives or dies on the through-R n600 byte-closed d_seg measurement —
run that gate before polishing the codec.**

## Sources

- [Rate-Distortion Analysis of Distributed Indirect Source Coding (PMC12385534)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385534/) — the unifying objective (backbone)
- [Rate-Distortion Theory in Coding for Machines (arXiv 2305.17295)](https://arxiv.org/html/2305.17295v2) — coding-for-machines equivalence + task-sensitivity theorems
- [Model-Aware RD Limits for Task-Oriented Source Coding (arXiv 2602.12866)](https://arxiv.org/pdf/2602.12866)
- [Feature-Preserving RDO in Image Coding for Machines (arXiv 2408.07028)](https://arxiv.org/html/2408.07028) — importance maps = saliency
- [Deep Task-Based Quantization (arXiv 1908.06845)](https://arxiv.org/abs/1908.06845)
- [Variational Bayesian Quantization (arXiv 2002.08158)](https://arxiv.org/abs/2002.08158) — sensitivity-driven variable-rate quantizer [stage 2]
- [constriction — entropy coders (GitHub bamler-lab)](https://github.com/bamler-lab/constriction) + [Understanding ANS (arXiv 2201.01741)](https://arxiv.org/pdf/2201.01741) [stage 4]
- [MapTRv2 (arXiv 2308.05736)](https://arxiv.org/abs/2308.05736) / [MapTR (ar5iv 2208.14437)](https://ar5iv.labs.arxiv.org/html/2208.14437) — lane parameterization [stage 5]
- [Jacobian Computation for Cumulative B-Splines on SE(3) (arXiv 2201.10602)](https://arxiv.org/pdf/2201.10602) — ego-trajectory [stage 1]
- [Spline-Based Trajectory Representation (IJCV, Lovegrove Spline-Fusion)](https://link.springer.com/article/10.1007/s11263-015-0811-3)
- [Out-of-Distribution Trajectory Prediction via Polynomial Representations (arXiv 2407.13431)](https://arxiv.org/pdf/2407.13431) — poly = 8.7-40.8% of point-space (compressive)
- [Cramér-Karhunen-Loève / harmonic PCA of functional time series (ScienceDirect S0304414913000793)](https://www.sciencedirect.com/science/article/pii/S0304414913000793) + [Kosambi-Karhunen-Loève theorem (Wikipedia)](https://en.wikipedia.org/wiki/Kosambi%E2%80%93Karhunen%E2%80%93Lo%C3%A8ve_theorem) — optimal temporal basis [stage 3]
- [Companding quantizer for Laplacian source (arXiv 1212.2144)](https://arxiv.org/pdf/1212.2144) — scalar quantizer [stage 2/4]
- [Wyner-Ziv distributed video coding + MCFI side-information](https://onlinelibrary.wiley.com/doi/abs/10.1002/9781118705957.ch8) — classical side-info analogue [stage 1]
- [Distortion-constrained compression of vector maps (ResearchGate 221000706)](https://www.researchgate.net/publication/221000706_Distortion-constrained_compression_of_vector_maps) — RD-optimal polyline approximation + quantization + arithmetic coding
