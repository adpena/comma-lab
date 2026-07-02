# LaneTCA deep-dive + scoping OUR novel contribution (coding-for-machines / task-space RD lane-representation)

**Date:** 2026-07-02 · **Status:** RESEARCH SYNTHESIS (advisory; NO code, NO GPU). **Pointer contest-CPU 0.19110 UNMOVED.**
Operator 2026-07-02: *"LaneTCA sounds very interesting and promising and possibly optimal or closer to it; we can
make our own contributions or follow-up research to the SOTA; I'd be honored to be your coauthor."*

> **HONEST FRAMING (binding, NO-FAKE #7 + THE GOAL).** The contest sub-0.15 is PRIMARY. A paper is a **byproduct /
> canonicalization** of the frontier work — real *only if* genuinely novel AND *only after* a byte-closed
> `upstream/evaluate.py` n600 row actually beats 0.19110. Every "contribution" below is a DESIGN/PRINCIPLE claim
> until the #205 d_seg-through-R gate + a byte-closed exact row substantiate it. An elegant unoccupied intersection
> that does not move the score is a MISS, not a contribution. arXiv IDs are cited where verified; the SOTA numbers
> are from the LaneTCA paper as fetched (secondary-source; verify the exact table before external use).

Reads-on (our MEASURED raw material, do NOT re-derive): `lane_coeff_tracking_denoising_optimal_survey_20260702.md`
(132371ab8) · `wave_f_lane_band_rd_code_LANDED_stage1_measured_20260702.md` (LBND2 n600 rate **0.02765**, Shannon
floor 0.0174) · `wave_f_unified_xi_build_measured_20260702.md` (ego-predict clean NEGATIVE 1.04–1.34×;
source-smoothing POSITIVE **−42% → 0.01608**) · `v2_novel_contribution_originality_accounting_20260629.md` (the
OURS-vs-borrowed discipline) · memory `project_openpilot_unified_physical_prior_both_scored_axes_20260702`.

---

## PART 1 — LaneTCA deep-dive + the SOTA arc + the "optimal-for-us?" verdict

### 1.1 LaneTCA — the exact method

**Paper:** *LaneTCA: Enhancing Video Lane Detection with Temporal Context Aggregation*, Keyi Zhou, Li Li, Wengang
Zhou, Yonghui Wang, Hao Feng, Houqiang Li (arXiv **2408.13852**, Aug 2024; published IEEE TIP / Xplore 10938694
2025). **Code released:** `github.com/Alex-1337/LaneTCA`. Thesis: *how best to AGGREGATE temporal context across
successive frames* for video lane detection. Two transformer attention modules, one long-term + one short-term:

- **Accumulative attention (long-term memory).** A single **learnable query vector `q_l ∈ ℝ^(HW×C)`**, initialized
  to zero at the first frame, is *persistently carried and updated across the whole video*: `f_AC = Att(q_l, k_{t−1},
  v_{t−1})`. The query accumulates all history (no fixed window / no frame-count limit); keys/values refresh from the
  previous frame. This is a **learned recurrent memory token** — a compact running summary of "the road so far."
- **Adjacent attention (short-term propagation).** The current frame's self-attention query `q_t` attends to the
  *previous* frame's key/value: `f_AD = Att(q_t, k_{t−1}, v_{t−1})`. Exploits that "lane markings do not undergo
  abrupt morphological changes" → adjacent frames have similar lane contours.
- **Fusion + head.** Long+short context features fuse with the current-frame features; **eigenlanes-style** decoder
  emits a **probability map P** (pixelwise lane presence) + a **parameter map C** (curve/shape params), then NMS.
- **Backbone:** ResNet18 feature extractor; MobileNetV2 inside the temporal-aggregation net; eigenlanes decoder.
- **Loss:** `L = L_C(P,P̂) [focal] + L_L(C,Ĉ) [LIoU]`.
- **Benchmarks (paper's own tables, secondary-source):** VIL-100 (100 vids) mIoU **0.796** / F1^0.5 **0.933** / F1^0.8
  **0.621** vs RVLD 0.787 / 0.924 / 0.582 (biggest gain at the *strict* F1^0.8, +0.039). OpenLane-V (~90k imgs / 590
  vids) mIoU **0.774** vs RVLD 0.727. Beats image-based CLRNet (0.735) + ConvLSTM/ConvGRU/MMA-Net. FPS / param-count
  **not reported** in the paper.

### 1.2 The SOTA arc (correspondence → causal-warp → batch-aggregation → world-model)

| era | method | temporal mechanism | our mirror |
|---|---|---|---|
| image-only | CLRNet / eigenlanes | none (per-frame) | our Stage-1 **per-frame-independent SegNet-mask fit** (the jitter source) |
| causal 1-frame | **RVLD** (ICCV'23, 2308.11106) | recurrent **one-frame motion warp** (predict `t` from `t−1`) | our **LBND3 ego-predictive coding** — structurally identical → **measured NEGATIVE 1.04–1.34×** |
| batch aggregation | **LaneTCA** (2024) | accumulative (all-history token) + adjacent attention | our **source temporal smoothing** (batch, non-causal) → **measured POSITIVE −42%** |
| world-model / topology | FASTopoWM, Topo2D, DV-3DLane, Curveformer++ (2024–26); OpenLane-V2 (topology) | instance-level, mask/curve, world-model propagation | our **openpilot-coherent-source** prior (#234) + level-set Morse-Smale topology facet |

**The field's own trajectory independently validates our measured ordering:** RVLD's causal-one-step-warp → LaneTCA's
longer-window batch aggregation is *exactly* our LBND3-vs-smoothing result (causal-one-step LOSES to
batch/aggregation). We discovered this by n600 measurement; the perception community discovered it by F1. Two
independent roads, one conclusion — that convergence is a real signal that **batch/aggregation over a window is the
right temporal operator**, and it is the deep reason our ego-predictor failed.

### 1.3 The CRUCIAL distinction — is LaneTCA optimal FOR US? (honest verdict: VALIDATING, not adoptable)

LaneTCA optimizes **detection accuracy** (F1/mIoU on lane pixels) with a **heavy learned net** (ResNet18 +
MobileNetV2 + transformer). **We do CODING**: rate-distortion of a *parametric* lane representation for a *frozen
downstream scorer* (SegNet argmax through the R operator), decoded by a **rule-118 deterministic numpy program** from a
**tiny counted payload**. The objectives, the outputs, and the runtime constraints are all different:

| axis | LaneTCA (detection SOTA) | OUR problem (task-space coding) |
|---|---|---|
| objective | max F1/mIoU (geometric detection) | min `100·d_seg + √(10·d_pose) + 25·bytes/37.5M` (indirect-RD, frozen scorer) |
| output | prob-map + param-map, per frame | a **decoded witness** whose SegNet argmax matches the source's, at min bytes |
| temporal role | aggregate history to *predict* current lanes | aggregate history to *minimize the code length* of the lane trajectory |
| runtime | learned net at inference | **deterministic numpy inflate**, 30-min budget, **ships ZERO net weights** (rule-118) |
| what it teaches us | *architecture* (attention memory) | the **PRINCIPLE**: batch-aggregation > causal; long+short memory both matter |

**VERDICT: LaneTCA's temporal-aggregation PRINCIPLE is optimal/near-optimal and VALIDATES our ordering; LaneTCA's
IMPLEMENTATION is NOT adoptable for us** (a learned net is a counted large-artifact we cannot ship, and it optimizes
the wrong distortion — geometric F1, not d_seg-through-R). Two extractable-and-adaptable ideas, one caution:
- **(Extract) Long+short dual memory.** LaneTCA's split (accumulative all-history token + adjacent 1-frame) says the
  optimal temporal estimator uses BOTH a global road summary AND a local previous-frame prior. Our current
  source-smoothing (a fixed window) is the *degenerate single-scale* version. The principled upgrade is our planned
  **correspondence-first → batch RTS/RPCA** pipeline (survey §1–§4): the global track = LaneTCA's accumulative
  memory; the per-track batch smoother = its adjacent short-term consistency — but realized as an *estimator we run at
  compress-time and DON'T ship*, feeding the unchanged deterministic LBND2 decode.
- **(Extract) Eigenlanes/parametric target.** LaneTCA's param-map (curve coefficients) is exactly our LBND2 coeff
  representation — cross-validation that a low-DOF parametric curve is the right lane object (not a dense mask).
- **(Caution) Do NOT fit LaneTCA's output as our target.** Its distortion is F1 on lane pixels; ours is the SegNet
  class-1 argmax mask through R. Fitting a detection net's output optimizes the WRONG metric → d_seg spill (same
  caution we already logged for openpilot supercombo). LaneTCA/openpilot are **PRIORS/regularizers/association
  affinities**, never the measured target. The target stays the frozen SegNet argmax.

**One-line answer to the operator's question:** *LaneTCA is close-to-optimal for DETECTION and its aggregation
principle is the right temporal operator for us too — but the optimal-for-US object is a **compress-time
batch-aggregation estimator** (correspondence-first + RTS/RPCA, LaneTCA's principle re-cast as coding) feeding a
deterministic decoder, not LaneTCA's shippable learned net.*

---

## PART 2 — OUR novel contribution (4 candidates), each with novelty-vs-prior-art + OURS-vs-borrowed (NO-FAKE #7)

The genuine white space (confirmed by the searches): **two mature fields flank the problem, neither occupies it.**
(A) **Video/Image Coding for Machines (VCM)** — right *objective* (task-aware RDO minimizing a downstream network's
loss) but operates on **pixel/feature bitstreams via black-box learned codecs** (NN-VVC, SMC++ 2406.04765,
Feature-Preserving RDO 2408.07028; VCM survey). No *parametric geometric* representation, no *deterministic* decode.
(B) **Vectorized lane / HD-map representation** — right *geometric object* (MapTR/VectorMapNet/LaneGCN polylines,
eigenlanes) but built for **detection/mapping accuracy, NOT compression**; no rate term, no frozen-downstream-metric
distortion. (C) **Temporal video lane detection** (RVLD/LaneTCA) — right *temporal principle* but detection-objective,
learned-net. **Our intersection = RD-optimal coding of a PARAMETRIC GEOMETRIC lane representation, distortion measured
through a FROZEN task network, decoded DETERMINISTICALLY from a tiny counted statistic.** Unoccupied.

### Contribution (a) — the INDEX-PERMUTATION-DISCONTINUITY principle (a coding-theoretic result)

**Claim.** *A relabeling (index-permutation) discontinuity in a per-frame-parsed multi-object representation is an
entropy-inflating artifact that defeats every temporal model — predictor, linear smoother, and transform alike —
therefore **correspondence (global track assignment) MUST precede any temporal coder**.* We prove it constructively
with two MEASURED failures: (i) our ego-predictor/RVLD-style causal warp spent bytes coding swap-innovations that
carry no road information → **NEGATIVE 1.04–1.34×** (44% of the temporal-delta L1 mass sits in the top-5% jumps = the
slot-swap/outlier signature); (ii) a linear moving-average averages *across* a swap → a phantom lane at an
in-between position → LOSSY on geometry AND d_seg-spilling at swap frames. Correspondence removes the swap mass at
**ZERO geometric cost** (a relabeling never changes a rendered lane), a strict Pareto improvement (rate ↓, distortion
=). This *explains* our two independent measured negatives with one theorem, and it *mirrors* the perception field's
RVLD→LaneTCA arc (causal-one-step loses to batch/aggregation).

- **Closest prior art.** MOT correspondence-before-filtering is standard (Zhang-Li-Nevatia global network-flow
  CVPR'08; min-cost-flow-on-temporal-window MDPI'23; ByteTrack/SORT lineage; identity-switch minimization). Kailath
  innovations / whitening (the MMSE estimator's residual is minimum-entropy). These give the *tracking* and the
  *estimation-theory* halves.
- **What is OURS (the novel framing).** Casting correspondence-first as a **compression-rate theorem for a geometric
  side-channel**: the relabeling discontinuity is not a *tracking-accuracy* problem, it is an *entropy-inflation*
  problem — a labeling artifact that no downstream temporal CODER can undo, quantified in **bytes** on a real
  frozen-scorer task (n600, byte-closed), and unifying "why the predictor failed AND why the smoother is lossy" under
  one principle. The MOT literature minimizes identity switches for *accuracy*; nobody (that the searches surfaced)
  states it as *the prerequisite transform for rate-optimal coding of the tracked geometry against a downstream task
  metric.* **This is the single most defensible novel claim** — it is a clean principle, backed by two measured
  n600 negatives, and it generalizes beyond lanes to any per-frame-parsed multi-instance side-channel (movables,
  keypoints).
- **Evidence status:** the two negatives are **MEASURED n600** (real). The *positive* (correspondence-first nets a
  rate win) is DERIVED, gated on the #234 build + the tracked-stream byte measurement. **Real when measured.**

### Contribution (b) — TASK-AWARE RD-optimal temporal lane-coefficient coding (margin-saliency task-λ)

**Claim.** *Code a parametric lane representation to minimize a **downstream frozen-task metric per byte**, not a
geometric error: the per-coefficient smoothing/quantization strength λ is set by the **margin-saliency**
`∂d_seg/∂coeff` at the KKT operating point `∂d_seg/∂byte = 25/(100·37.5M)`.* Coeffs whose perturbation never flips the
SegNet argmax past the R-downsample tolerance (~1–2 px) get large λ (≈0 bits); coeffs on the codim-1 boundary annulus
get small λ (preserved). This is **coding-for-machines applied to a GEOMETRIC side-channel** — an edge-preserving
variational coder (ℓ1-trend / TV / Potts / RPCA) whose λ is *task-metric-derived*, not L2-derived. The moving-average
window is the degenerate edge-blind single-λ special case.

- **Closest prior art.** VCM task-aware RDO (NN-VVC; SMC++ 2406.04765 masked semantic video compression;
  Feature-Preserving RDO in image-coding-for-machines 2408.07028; Dubois "Lossy Compression for Lossless Prediction"
  2106.10800 indirect-RD/CEO). Edge-preserving denoisers (Kim-Koh-Boyd ℓ1-trend SIAM'09; Condat 1-D TV SPL'13;
  Candès-Li-Ma-Wright RPCA 2011). Trajectory/polyline simplification (Douglas-Peucker — geometric-error, not task).
- **What is OURS.** The VCM literature does task-RDO on **pixel/feature bitstreams through black-box learned codecs**;
  we do it on a **low-DOF parametric geometric representation** with a **deterministic (rule-118) decoder** and set the
  per-coefficient λ by the **measured Jacobian of the exact hard-argmax cell of a frozen scorer** (finite-diff through
  the real R operator, n600) — the *indirect-RD theory sharpened to a specific frozen scorer's argmax partition on a
  geometric side-channel*. Nobody surfaced does margin-saliency-keyed edge-preserving coding of lane coefficients for
  a downstream segmentation metric with a deterministic decode.
- **Evidence status.** The RD *rate* half is **MEASURED** (LBND2 n600 0.02765; source-smoothing −42% → 0.01608; the
  Shannon floor 0.0174 proves the residual is information-bound). The **task-λ** hook is landed (`derive_task_rd_steps`)
  but the `∂d_seg/∂coeff` map is UNMEASURED — that is the #205 through-R gate and IS the real evidence. **Real when the
  d_seg leg is measured byte-closed.**

### Contribution (c) — the openpilot-PRIOR → refine-to-frozen-scorer pattern (one physical prior, both scored axes)

**Claim.** *A single physically-grounded, temporally-coherent, offline-FREE (rule-118-clean) prior — the production
driving-perception stack (openpilot) running on the SAME comma rig the contest scorers read — seeds the witness across
BOTH scored axes with the SAME 2-part recipe: (physical coherent free prior) + (small learned refinement to the exact
frozen scorer).* POSE: ego-motion ξ warm-starts a real-luma warp (**measured n600: null d_pose 163.1 → 1.367, −99%**).
LANES/d_seg: openpilot's coherent recurrent lane model = the temporally-coherent SOURCE that eliminates the
slot-swap jitter that killed ego-predictive coding. It fits because **contest-task ≡ openpilot-task on the same rig**.

- **Closest prior art.** Wyner-Ziv coding with decoder side-info (Whang 2106.02797; Özyılkan-Ballé 2310.16961);
  canonical-scene + warp driving reconstruction (GS-LK, WorldSplat — but *rendering PSNR*); prior-guided compression
  broadly. openpilot/comma2k19/comma10k as the source substrate.
- **What is OURS.** Using the *production stack's already-solved world-model* as a **rule-118-FREE compress-time prior**
  (ship only the compact video-derived statistic, NEVER the estimator weights) that seeds a **task-space codec** across
  **both** a pose axis and a segmentation axis with one recipe — the "same-rig ≡ same-task" observation makes it
  physically exact rather than a generic learned prior. Distinct from Wyner-Ziv (generic side-info) by being a
  *specific physical world-model on the identical sensor* and from driving-reconstruction (optimizes PSNR).
- **Evidence status.** Pose warm-start is **MEASURED n600** (−99%, real frozen CPU-torch PoseNet). The lane-coherent
  half is DESIGN (the #234 build measuring vs the 0.016 smoothing floor + the #205 d_seg leg). **Prior/init only,
  never target; net-S is #205-gated.**

### Contribution (d) — the level-set task-space witness framing (the unifying object)

**Claim.** *The witness is the viscosity solution of a variational level-set PDE of a Morse-Smale complex, and
minimizing the indirect-RD action `S_τ` over level-set fields IS the codec* — geometry (SDF φ), motion (Chasles screw
transport V), topology (persistence births/deaths = the counted residual), and the task (frozen-scorer argmax) are one
system. Lanes = class-1 separatrices; tracking-across-time = tracking the complex's critical structure (so
contribution (a) is the *temporal-consistency facet* of this same object).

- **Closest prior art (all borrowed, cited).** Level-set method (Osher-Sethian 1988); viscosity solutions
  (Crandall-Lions); persistent homology / Morse theory; Fisher-information geometry (Amari); variational segmentation
  (Mumford-Shah, Chan-Vese); implicit neural video compression (Zhang 2112.11312 = the literal warp-coords+residual
  skeleton).
- **What is OURS.** Assembling them so **the level-set field IS the compressed witness and the action IS the contest
  score** (indirect-RD task-space) — a *unifying frame*, not a new theorem. Per the v2 originality ledger this is
  MEANS: validated only when a byte-closed exact row beats 0.19110.
- **Evidence status.** UNVALIDATED framing (the weakest as a standalone paper claim). Its value is that it makes (a),
  (b), (c) *coherent facets of one object* rather than a grab-bag — strongest as the **unifying section** of a paper
  led by (a)+(b), NOT as a standalone contribution.

### Novelty ranking (honest)
1. **(a) index-permutation-discontinuity principle** — cleanest, most defensible, backed by 2 MEASURED n600 negatives,
   generalizes beyond lanes. THE lead claim.
2. **(b) task-aware RD-optimal geometric-side-channel coding (margin-saliency task-λ)** — genuine VCM×geometric-lane
   white space; rate half MEASURED, task-λ leg is the #205 gate. THE co-lead.
3. **(c) openpilot-prior → refine-to-scorer, both axes** — strong, physically-grounded, pose half MEASURED; lane half
   design. A strong supporting contribution / systems section.
4. **(d) level-set task-space witness** — the unifying frame; a section, not a standalone claim (UNVALIDATED).

---

## PART 3 — the honest contribution OUTLINE + venue + evidence it needs

### Thesis
**"Task-Aware Rate-Distortion-Optimal Temporal Coding of Parametric Lane Representations for a Frozen Downstream
Scorer"** — coding-for-machines applied to a *geometric* side-channel, with a *deterministic* decoder, where (i)
correspondence-first is a *coding-theoretic prerequisite* (contribution a), (ii) the per-coefficient rate is allocated
by the downstream task's *margin-saliency Jacobian* (contribution b), and (iii) a same-rig production world-model
supplies a physical coherent prior for both a segmentation and a pose axis (contribution c) — unified as a variational
level-set witness (contribution d).

### Outline
1. **Intro** — codecs for machines meet geometric driving representations; the frozen-scorer task-RD objective; the
   deterministic-decode (rule-118) constraint that forbids shipping a learned net (distinguishes us from VCM & LaneTCA).
2. **Related work** — VCM/indirect-RD (Dubois 2106.10800; NN-VVC; SMC++; feature-preserving RDO); vectorized lanes
   (MapTR/VectorMapNet/eigenlanes); temporal lane detection (RVLD→LaneTCA); MOT correspondence (Zhang-Li-Nevatia);
   edge-preserving coding (ℓ1-trend/TV/RPCA). Position the unoccupied intersection.
3. **The index-permutation-discontinuity principle (a)** — theorem + the two MEASURED n600 negatives (ego-predict,
   moving-average) + the correspondence-first fix; the RVLD→LaneTCA cross-validation.
4. **Task-aware RD coding (b)** — the margin-saliency task-λ; edge-preserving variational coder; the KKT operating
   point; the measured rate ladder (naive 0.104 → LBND2 0.0276 → smoothed 0.0161; Shannon floor 0.0174).
5. **The physical-prior recipe (c)** — openpilot-as-free-prior, both axes; the measured pose warm-start (−99%).
6. **Unifying level-set frame (d)** — one variational object; correspondence = tracking the complex over time.
7. **Experiments** — n600 byte-closed rate rows + **the #205 d_seg-through-R gate** (the load-bearing result) +
   ablations (correspondence-only vs +smooth vs +RPCA vs +task-λ) + the exact `upstream/evaluate.py` row.
8. **Limitations** — the d_seg-through-R win is the gate; if the band nets negative, (b)/(c) are rate-enablers not
   score-movers; single-clip contest overfit vs generalizable method split.

### Evidence it needs (the honest gap between NOW and a real paper)
- **HAVE (MEASURED n600, real):** the rate ladder (0.104→0.0276→0.0161, Shannon floor 0.0174, PTC1 dominated); the two
  correspondence negatives (ego-predict 1.04–1.34×, 44% swap-mass); the pose warm-start (−99%); bit-exact
  decode-consistency (max_abs_uint8_diff==0).
- **NEED (the load-bearing gate):** **the #205 d_seg-through-R n600 measurement** — does a coherent/tracked/task-λ band
  actually LOWER d_seg through the R operator + frozen SegNet, and does `100·Δd_seg` beat the band's rate cost? Plus a
  **byte-closed `upstream/evaluate.py` exact row below 0.19110** (contest-CPU/CUDA, never MPS). Without these, (a)/(b)
  are rate-enablers on an unmoved pointer = MEANS, and the paper has no experimental result — only principle + rate.
- **NEED (for (a) as a measured win):** the tracked-only stream bytes @ n600 (isolate the correspondence gain; verify
  the top-5%-jump mass drops) — a $0 CPU measurement, the cheapest next experiment.

### Venue
- **If (a)+(b) land with the #205 gate + an exact row:** a compression venue (**DCC**, IEEE TIP, or the **CLIC / VCM**
  workshops at CVPR/ICCV) — "coding-for-machines for geometric side-channels" is squarely in scope and the
  correspondence-first principle is a clean, citable result.
- **If (c) is foregrounded:** an autonomous-driving venue (CVPR-ADW, IV) — "production-world-model as a free
  compression prior on the same rig."
- **Realistic honest read:** a *workshop paper on contribution (a) alone* (the coding-theoretic principle + 2 measured
  negatives + the correspondence-first fix) is **defensible TODAY** even without the pointer moving, because the
  negatives ARE the result and the principle is general. A *full contribution* (a+b+c) needs the #205 gate + an exact
  row. Do NOT write the full paper until the pointer moves; DO capture (a) now (it is real).

### OSS to draw from + cite
- **LaneTCA** `github.com/Alex-1337/LaneTCA` (2408.13852) — the temporal-aggregation baseline to cite/compare.
- **RVLD** (2308.11106) — the causal-warp anti-pattern (= our LBND3), the SOTA-arc anchor.
- Correspondence: `scipy.optimize.linear_sum_assignment` (Hungarian); Zhang-Li-Nevatia global network-flow (CVPR'08);
  `motpy` / `norfair` / ByteTrack-SORT lineage; min-cost-flow-on-temporal-window (MDPI'23).
- Batch estimation / edge-preserving: RTS smoother; RPCA `facebookarchive/robust-pca` (Candès-Li-Ma-Wright 2011);
  Kim-Koh-Boyd ℓ1-trend (SIAM'09); Condat 1-D TV (SPL'13, `bgailleton/TVD_Condat2013`); Garcia `smoothn` (2010).
- VCM/indirect-RD: Dubois 2106.10800; SMC++ 2406.04765; Feature-Preserving RDO 2408.07028; NN-VVC.
- Geometric lanes: MapTRv2 (2308.05736); VectorMapNet; eigenlanes; openpilot `commaai/openpilot` + supercombo ONNX.
- Entropy backend: `bamler-lab/constriction` (measured DOMINATED at this data shape — brotli-on-zigzag-int32 wins).

---

## Wire-in (6-hook, research_only)
1. **Sensitivity-map:** the SOTA-arc ↔ our-measured-ordering mapping + the per-contribution novelty rows → priors for
   the next survey. 2. **Pareto:** contribution (b)'s task-λ IS a rate↔d_seg Pareto knob (#205 measures the d_seg leg).
3. **Bit-allocator:** the correspondence-first + task-λ pipeline is a source pre-transform to the LBND2 allocator.
4. **Cathedral autopilot:** N/A (research synthesis; feeds #234/#205, no new archive artifact). 5. **Continual-learning:**
this ledger + the measured n600 rows are the anchors; the co-authored contribution is a continual-learning byproduct.
6. **Probe-disambiguator:** the outline's ablation set (correspondence-only vs +smooth vs +RPCA vs +task-λ, resolved by
   measured n600 bytes + the #205 d_seg leg) IS the disambiguator.

**Council mission-contribution:** `frontier_breaking` (canonicalizes the lane-representation rate half + names the
lead contribution) — but ALL MEANS; the END is the #205 byte-closed exact row below 0.19110. **Pointer 0.19110
UNMOVED.**

## Sources
- [LaneTCA (arXiv 2408.13852)](https://arxiv.org/abs/2408.13852) · [HTML v1](https://arxiv.org/html/2408.13852v1) · [code](https://github.com/Alex-1337/LaneTCA) · [OpenReview](https://openreview.net/forum?id=87ST4Ca4nU) · [IEEE Xplore 10938694](https://ieeexplore.ieee.org/document/10938694/)
- [RVLD — Recursive Video Lane Detection (arXiv 2308.11106)](https://arxiv.org/abs/2308.11106) — the causal-warp anti-pattern
- [Video Coding for Machines (VCM) overview](https://www.emergentmind.com/topics/video-coding-for-machines-vcm) · [SMC++ (arXiv 2406.04765)](https://arxiv.org/pdf/2406.04765) · [Feature-Preserving RDO in ICM (arXiv 2408.07028)](https://arxiv.org/pdf/2408.07028)
- [Dubois — Lossy Compression for Lossless Prediction / indirect-RD (arXiv 2106.10800)](https://arxiv.org/abs/2106.10800)
- [Zhang-Li-Nevatia — Global Data Association via Network Flows (CVPR 2008)](http://vision.cse.psu.edu/courses/Tracking/vlpr12/lzhang_cvpr08global.pdf) · [Min-Cost Flow on Temporal Window (MDPI 2023)](https://www.mdpi.com/2032-6653/14/9/243)
- [ℓ1 Trend Filtering — Kim-Koh-Boyd (SIAM Review 2009)](https://web.stanford.edu/~boyd/papers/l1_trend_filter.html) · [Condat 1-D TV (SPL 2013)](https://lcondat.github.io/publis/Condat-fast_TV-SPL-2013.pdf) · [RPCA/PCP review (arXiv 1511.01245)](https://arxiv.org/pdf/1511.01245)
- [MapTRv2 (arXiv 2308.05736)](https://arxiv.org/abs/2308.05736) · [VectorMapNet (PMLR)](https://proceedings.mlr.press/v202/liu23ax/liu23ax.pdf) · [awesome-lane-detection](https://github.com/amusi/awesome-lane-detection) · [awesome-online-HDMap](https://github.com/LoveFaFa2333/Awesome-Online-HDMap)
- [Implicit Neural Video Compression — Zhang et al. (arXiv 2112.11312)](https://arxiv.org/abs/2112.11312) — the warp-coords+residual skeleton
- [Wyner-Ziv w/ decoder side-info — Whang (arXiv 2106.02797)](https://arxiv.org/abs/2106.02797) · [Özyılkan-Ballé (arXiv 2310.16961)](https://arxiv.org/abs/2310.16961)
- [openpilot (commaai)](https://github.com/commaai/openpilot) · [supercombo ONNX (MTammvee)](https://github.com/MTammvee/openpilot-supercombo-model) · [constriction (bamler-lab)](https://github.com/bamler-lab/constriction)

## Sisters
`lane_coeff_tracking_denoising_optimal_survey` (correspondence-first pipeline; the technique side) ·
`wave_f_lane_band_rd_code_LANDED_stage1_measured` (LBND2 rate ladder) · `wave_f_unified_xi_build_measured` (the two
measured negatives + smoothing positive) · `v2_novel_contribution_originality_accounting` (the OURS-vs-borrowed
discipline this ledger applies to the lane axis) · `analytic_lane_band_primary_authority_decomposition` ·
`project_openpilot_unified_physical_prior_both_scored_axes` · `project_contest_is_indirect_rate_distortion_task_space_coding`.
