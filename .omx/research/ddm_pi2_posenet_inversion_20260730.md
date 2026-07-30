---
schema: ddm_pi2_posenet_inversion.v1
date_utc: 2026-07-30
arm: ddm_pi2
axis: "[macOS-CPU advisory — frozen CPU-torch authority; MPS = gradient device only, NEVER a score]"
pointer: "0.1910828242 [contest-CPU] UNMOVED"
score_claim: false
promotable: false
pointer_moved: false
research_only: true
council_predicted_mission_contribution: frontier_protecting
verdict_scope: FORMULATION
consumes: [ddm_ph2_pantheon_convocation_20260730, ddm_pfs1_posefield_and_recompose_20260729,
  frozen_scorer_exact_factorization_20260715, segnet_recursive_fractal_factorization_20260715,
  null_subspace_rate_measure_20260717, ddm_deferral_queue_ledger_20260729]
consumers: [QA47, QA48, QA52, ddm_deferral_queue_ledger_20260729, v10_SPEC_row12_pose_in_burn]
tokens: [no-triality, p0-ledger-ok, magnitude-ok]
---

# ddm_pi2 — PoseNet per-dim inversion (the SegNet-head-analog for PoseNet)

## §0 HEADLINE (pointer honesty first)

**Pointer `0.1910828242 [contest-CPU]` UNMOVED.** Everything here is `[macOS-CPU advisory]`; MPS was
used ONLY as the gradient device (`patch_scorer_for_mps`, forward drift vs CPU 1.1e-5), every realized
Δpose measured through the frozen **CPU-torch** PoseNet on the exact upstream weights
(`upstream/models/posenet.safetensors`). Forward-equivalence positive control: my differentiable
forward reproduces the exact upstream PoseNet on a real pair BIT-EXACTLY (max_abs 0.0 on CPU; yuv6
helper vs upstream max_abs 0.0). This is MEANS — the white-box PoseNet inversion that aims QA47/QA48/QA52.

**Charter (operator 2026-07-30, ledger QA51):** *"Invert PoseNet the way you inverted SegNet's head —
per-dim null space, response function, cheapest input that moves each dim."*

**The one-line result:** unlike SegNet's head (well-conditioned rank-4, argmax decision), PoseNet's scored
head is a **rank-6 (full) linear map with ONE dominant singular direction (σ1/σ6 = 24.8)**, and the scored
output is **dominated by a single dim — p0, forward translation** — whose input-Jacobian is ~50× larger
than every other dim; the rotation dims p3-p5 are **input-near-null**. Steering is **realization-limited by
the uint8 lattice, NOT sensitivity-limited** (identical to SegNet's boundary flips). **THE QA47 PRE-ANSWER:
the image-space steering rank does NOT mirror the warp-param rank-1 law — a FIXED k≤8 shared basis is
FALSIFIED** (per-pair Jacobian directions near-orthogonal, mean cos 0.08, k90 ≈ 43/53); the surviving cheap
path is a **receiver-DERIVED per-pair basis** (the receiver has the frozen net, rule-118 free) with
p0-dominance ⟹ ~1-2 effective coefficients/pair.

Receipts (SSD, certify-or-block): `/Volumes/VertigoDataTier/pact/ddm_pi2_20260730/`
`{algebra_receipt.json, smoke_receipt.jsonl, atlas_summary.jsonl (318 rows / 53 pairs), fields/ (53×
6-dim Jacobian stacks fp16), svd_receipt.json, cheapest_receipt.json}`. Tool:
`experiments/ddm_pi2_posenet_inversion.py` (ruff-clean, 2 review passes). Deterministic: rerun regenerates
from the frozen weights + gt_n96 cache. Pair set: 53 stratified real pairs from `gt_n96.npz`
(17 highest-‖pose‖ hard-core + 24 tail + 12 random controls, seed 20260730).

## §1 DELIVERABLE 1 — THE EXACT ALGEBRA (weight inspection; no forward)

**(a) The final head is EXACTLY linear, rank-6, ONE dominant direction — the pose analog of SegNet's
rank-4.** `hydra.final_layer['pose']` is `Linear(32→12)`; the 6 SCORED dims (`compute_distortion` uses
`h.out//2 = 6`) are the EXACT linear map `pose6 = W6·b + bias6` of the 32-dim post-`res_layer` ReLU feature
`b`. MEASURED singular values of `W6` (6×32):

| σ1 | σ2 | σ3 | σ4 | σ5 | σ6 | cond σ1/σ6 |
|---|---|---|---|---|---|---|
| **8.769** | 0.988 | 0.700 | 0.512 | 0.420 | 0.353 | **24.8** |

- **rank = 6 (full)** — NO scored-dim collapse in the head itself. Contrast SegNet: rank-4 (argmax needs
  only differences of 5 logits), **well-conditioned σ1/σ4 = 1.74**. PoseNet's head is **rank-6 but
  ill-conditioned — σ1 dominates by ~9×** ⟹ the 32-dim feature `b` maps to the score along essentially ONE
  strong axis. This is the head-level shadow of the p0-dominance measured in §2.
- **head-null = 26 of 32 feature dims** are EXACTLY score-invisible (orthogonal to row-space(W6) after the
  frozen net) — a large feature-space null, but AFTER `N_pose`, so not directly an input/rate lever.
- The 6 scored rows and 6 unscored rows (dims 6-11, discarded by the scorer) occupy well-separated
  subspaces: min principal angle 62.3° (not degenerate).

**(b) The composed input-side resize null is EXACT (shared with SegNet).** `A_pose ≡ A_seg` (the same
bilinear downsample (874,1164)→(384,512), modules.py:73 == :109). Separable bilinear downsample is full
output-row-rank ⟹ **ker dim = 874·1164 − 384·512 = 820,728 DOF/ch/frame = 80.7% of camera DOF EXACTLY
invisible to BOTH scorers** (frozen memo B1). The `rgb_to_yuv6` 2×2 chroma box-average adds a further
POSE-ONLY null above 2px@(384,512) (frozen memo B4) — confirmed empirically in §2 (pose is 99.3% luma).

**(c) Per-dim f16 output-quantization floors (from the real 600 banked pose targets).** d_pose = mean over
6 dims of `(pose − gt)²`; a single-dim error `e` contributes `√(10·e²/6)` to S.

| dim | |value| ≤ | f16 ulp @ max | S contrib of one f16-ulp error |
|---|---|---|---|
| **p0** (fwd trans) | **34.68** | **0.03125** | **0.0403** ⟵ LARGE |
| p1 | 0.118 | 6.1e-5 | 7.9e-5 |
| p2 | 0.065 | 6.1e-5 | 7.9e-5 |
| p3 | 0.033 | 3.1e-5 | 3.9e-5 |
| p4 | 0.021 | 1.5e-5 | 2.0e-5 |
| p5 | 0.035 | 3.1e-5 | 3.9e-5 |

**Finding: pose-OUTPUT storage is f16-safe for dims 1-5 but MARGINAL for p0** — at |p0|≈35 the f16 spacing
is 0.031, and a single-dim f16 rounding alone can cost **0.040 in S** (matches pfs1 §4.3's razor-sharp
dim0 lattice: 0.011 rounding → +10 d_pose at a ridge point). Any pose carrier must code p0 with an
offset/scale (store `p0 − mean` at fine step), never raw f16. Dims 1-5 raw f16 is free.

## §2 DELIVERABLE 2 — PER-DIM RESPONSE ATLAS (MEASURED, 53 pairs, ∂pose_i/∂input)

Input-Jacobian ∂pose_i/∂(RGB) via MPS backprop; spatial at resized (384,512), channel/frame/realization at
camera-res. Medians over 53 pairs × 2 frames.

**Magnitude ordering (the dominant structural fact):**

| dim | median ‖J‖ (resized) | vs p0 | read |
|---|---|---|---|
| **p0** (fwd translation) | **0.0675** | 1× | **the whole game** |
| p1 | 0.00089 | 1/76 | minor |
| p2 | 0.00105 | 1/64 | minor |
| p3 | 0.00031 | 1/215 | **input-near-null** |
| p4 | 0.00016 | 1/426 | **input-near-null** |
| p5 | 0.00029 | 1/233 | **input-near-null** |

**p0's input-Jacobian is 50-400× larger than every other scored dim.** The rotation dims p3-p5 barely
respond to the input on this video (and their VALUES are tiny too, §1c) — an INPUT-side confirmation of the
pfs1/sc1 translation-dominant / rotation-inert picture (pfs1: "output dims 3-5 are not raw metric
rotations", e_p per-dim std [0.82, 0.12, 0.24, 0.010, 0.007, 0.029]).

**Channel (luma vs chroma), per dim, MEASURED:**
- **99.3% LUMA / 0.7% chroma (camera-res, all dims)**; 99.1-99.6% luma at resized level. **PoseNet reads
  pose almost entirely from luminance structure — it is chroma-blind** (consistent with the 2×2 chroma
  box-average, frozen memo B4). ⟹ **any pose steering must live in luma**; chroma carries no pose.

**Frame split (f0 vs f1), per dim:**
- **~52% of the pose Jacobian energy is on frame_0** (0.504-0.534 across dims). Since frame_0 is
  structurally seg-free (SegNet reads `x[:,-1]` only, frozen memo B3), **~half the pose response lives on
  the seg-free frame** = the FREE steering surface QA47 targets, now measured per-dim.

**Spatial support (per-dim bands, median; centroid_row 0=top/far, 1=bottom/hood):**

| dim | FAR/top | MID | NEAR/hood | centroid_row |
|---|---|---|---|---|
| **p0** | 0.089 | 0.521 | **0.385** | **0.621** |
| p1 | 0.187 | 0.612 | 0.200 | 0.507 |
| p2 | 0.172 | 0.570 | 0.260 | 0.535 |
| p3 | 0.207 | 0.686 | 0.106 | 0.467 |
| p4 | 0.157 | 0.599 | 0.245 | 0.525 |
| p5 | 0.141 | 0.677 | 0.174 | 0.484 |

**Far-field law, per-dim result (REFINES qa43/qa45):** p0 (the dim that matters) is **NOT far-field
dominated** — its support centroid is **62% down the frame (mid-road + near/hood band)**, with the sky/far
top carrying only 8.9%. This is the classic optical-flow scaling: under forward translation, near-camera
points move most. The other dims are more mid-band centered. So the "bidirectional far-field law" as a
per-dim ∂pose/∂pixel statement is **near-field/ground-weighted for translation**, not far-field.

**Frequency:** the |grad| MAGNITUDE envelope is LF-clustered (57-69% LF), but the SIGNED field carries
substantial mid-freq — only **1/3 of the frame_0 signed Jacobian energy is low-freq** (camera-res
lowfreq_frac median 0.331; §4). Pose reads edge/texture structure, not just a smooth far-field envelope.

**Linearity radius (secant vs tangent, uint8 quanta) — REALIZATION-LIMITED:** the min-L2 steering atom
(§4) amplified by mult ∈ {0.25,0.5,1,2,4} of its L2-optimal amplitude, realized through camera-res uint8,
gives realized/tangent-predicted Δp0:

| mult | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| gap (median over 5 pairs) | ~0.00 | ~0.02 | **0.11-0.26** | 0.28-0.55 | 0.48-0.84 |

**The tangent OVER-predicts realized Δpose by ~3-10× at the L2-optimal amplitude; the realized response
only converges to the tangent as amplitude grows well past L2-optimal.** The min-L2 atom's per-pixel RMS is
~0.018/px (0-255 units) — **sub-uint8-LSB**, so most of it rounds away. **This is the SegNet
realization-limited story, now MEASURED for PoseNet: pose steering is bounded by the uint8 lattice, not by
sensitivity** — the same reframe that made SegNet d_seg "a coding problem, not a sensitivity problem".

## §3 DELIVERABLE 3 — CROSS-PAIR SHARED-BASIS RANK (THE QA47 / PH-1 PRE-ANSWER)

Per dim, the 53 unit-normalized per-pair Jacobian DIRECTIONS → cosine-Gram → eigen energy curve. PH-1's
prediction (ph2 §1): *"image-space steering rank mirrors the warp-param rank-1 law (98.06% dim-0), k≤8 →
≈12-64 B/pair."* MEASURED:

| dim | ‖J‖med | top1 energy | fixed-k=8 energy captured | k90 (of 53) | mean pairwise cos |
|---|---|---|---|---|---|
| **p0** (load-bearing) | 0.0675 | **0.108** | **0.36** | **43** | **0.083** |
| p1 | 0.00089 | 0.208 | 0.42 | 42 | 0.158 |
| p2 | 0.00105 | 0.140 | 0.34 | 44 | 0.109 |
| p3* | 0.00031 | 0.309 | 0.55 | 38 | 0.245 |
| p4* | 0.00016 | 0.294 | 0.54 | 38 | 0.245 |
| p5* | 0.00029 | 0.276 | 0.62 | 39 | 0.213 |

*p3-p5 are input-near-null (§2); their unit directions are fp16-quantization-noise-contaminated → their
rank is NOT load-bearing. The trustworthy answer rests on **p0** (dominant signal, fp16-clean).

**PH-1's fixed k≤8 shared-basis prediction is FALSIFIED.** For p0: a fixed k=8 global basis captures only
**36%** of the per-pair Jacobian-direction energy; you need **k90 = 43 of 53** to reach 90%. The per-pair
directions are **near-orthogonal (mean cos 0.083)** — each scene's ∂p0/∂pixel pattern is essentially
different. **Cross-dim relation (honesty rail, image vs param space):** the 6 image-space per-dim Jacobian
directions have top1 energy **0.353** — vs the warp-PARAM `p_star` SVD dim0 = **0.9698** (pfs1). **The
warp-param rank-1 law does NOT transfer to image space.** They are different objects: in param (warp) space
the OPTIMAL correction is ~1-D (forward translation); in image (pixel) space, steering each pose dim needs
a distinct high-rank spatial pattern.

**What survives (the QA47 re-aim, NOT a kill):** a FIXED global basis fails, but the receiver HAS the
frozen PoseNet (rule-118 FREE) ⟹ it can compute the per-pair input-Jacobian ITSELF at decode. So QA47's
carrier must use a **receiver-DERIVED PER-PAIR basis**, not a shipped fixed basis — and because p0
dominates (§2), the effective per-pair rank is ~1-2, so **~1-2 coefficients/pair** suffice (even cheaper
than PH-1's 12-64 B). QA47's coeff-vs-d_pose sweep should be run **in the per-pair receiver-derived
Jacobian basis**, NOT a fixed global one. verdict_scope: FORMULATION (fixed-basis formulation falsified;
receiver-derived-per-pair formulation is the surviving, cheaper path).

## §4 DELIVERABLE 4 — CHEAPEST INPUT PER DIM (MEASURED, 5 pairs, camera-res uint8)

- **Minimal-L2 steering input** per unit Δp0 = J0/‖J0‖² (closed form): median ‖input‖ ≈ **10-30** (0-255
  units, spread over the whole frame), RMS ~0.018/px — sub-LSB (⟹ realization-limited, §2).
- **Family decomposition of the p0 frame_0 Jacobian (medians):** frame0_frac 0.52 · **chroma_frac 0.007**
  (luma-only) · **lowfreq_frac 0.33** (a pure low-freq/far-field carrier captures only 1/3 of the pose
  response — INSUFFICIENT alone; the other 2/3 is edge/mid-freq structure).
- **Steerability by dim:** p0 is the only meaningfully steerable dim; p1/p2 minor; **p3-p5 input-near-null**
  (no cheap carrier — the dim barely responds to frame_0).
- **Coded-bytes B/unit:** NOT cleanly measurable here — at small target Δpose the uint8 realization death
  (§2) drives the LF-carrier realized Δpose to ~0, so B/unit → garbage (1e6-1e36). HONEST SCOPE: the cheap
  coded-carrier price is meaningful only for p0 at supra-L2 amplitude, and a LF-only carrier is
  insufficient (1/3). The clean carrier metric is the receiver-derived per-pair Jacobian (§3) priced as
  ~1-2 coefficients/pair; the actual coeff-vs-d_pose curve is QA47's job (not resolved here).

## §5 THE PER-DIM NULL (does it change the composed grammar?)

**Yes, one structural simplification, already partly exploited:** p3, p4, p5 (rotations) are input-near-null
(‖J‖ 200-426× below p0) AND value-near-null (|val| ≤ 0.035, f16-free) AND the warp receiver already ships
`s_r=0` (R=I) making dims 3-5 inert (pfs1 §4). So the composed grammar can treat pose as **effectively p0
(+ minor p1/p2)** — a 1-3 dim OUTPUT surface, with rotations a genuine 3-dim null in BOTH input-response
and value. This is an INPUT-side confirmation that pose coding should spend its bytes on p0 (fine step,
offset-coded, §1c) and near-nothing on rotations. The head-null (26/32 feature dims) is a feature-space
null after N_pose — not a direct rate lever.

## §6 CONTRAST WITH THE SEGNET INVERSION (the two frozen scorers, side by side)

| | SegNet head | PoseNet head |
|---|---|---|
| final map | Conv2d k=3 → 5 logits, LINEAR | Linear 32→12, LINEAR |
| scored rank | **4** (argmax = differences) | **6** (MSE = values) |
| conditioning | well-cond σ1/σ4 = 1.74 | **ill-cond σ1/σ6 = 24.8 (one dominant axis)** |
| decision | discontinuous argmax; flip-distance field | quadratic MSE; smooth |
| dominant structure | Lane normals largest (all 4) | **p0 (fwd translation) dominates 50×** |
| input null | resize 80.7% + argmax-interior 95% | resize 80.7% (SAME A) + chroma HF + rotation dims |
| limiting regime | **realization-limited (uint8 lattice)** | **realization-limited (uint8 lattice) — SAME** |
| steering basis | 4-dim head projection (exact) | receiver-derived per-pair (fixed k≤8 FALSIFIED) |

**The unification:** both scorers are **realization-limited by the uint8 camera-res lattice, not
gradient-limited** — the sharp statement (from SegNet, now confirmed for PoseNet) that both d_seg and
d_pose optimization are CODING problems against a known frozen oracle, not sensitivity problems.

## §7 Wire-in (#125), scope, custody

- sensitivity-map: the per-dim ‖J‖ ordering + spatial bands are new advisory saliency rows · Pareto N/A
  (no new (S,byte) point) · bit-allocator: informs pose-carrier coding (p0 offset-fine, dims1-5 f16-free,
  rotations near-zero) · cathedral N/A · continual-learning: this memo + ledger QA51→FIRED + QA47 amend ·
  probe-disambiguator: the PH-1 k≤8 falsifier IS the disambiguator (FIRED).
- verdict_scope: FORMULATION (fixed-basis QA47 formulation falsified; receiver-derived-per-pair survives).
  No paradigm/family kill. All rows n≤53 real gt_n96 pairs, macOS-CPU fp32 advisory, MPS grad-only; NOT
  contest-CPU/CUDA; no score claim. n600 generalization owed where a carrier consumes these numbers.
- [no-triality] [p0-ledger-ok] [magnitude-ok] — measurement arm; no DSL lever or canonical-equation surface
  changed. Tool `experiments/ddm_pi2_posenet_inversion.py`; receipts on SSD (certify-or-block; gitignored,
  deterministic-regenerable from frozen weights + gt_n96).

## STORES CONSULTED

CLAUDE.md; AGENTS.md; docs/operating_manual_craft_handoff.md; upstream/modules.py + frame_utils.py (pinned,
read-only); ddm_ph2_pantheon_convocation_20260730 (PH-1/PH-5, QA47/QA51); ddm_pfs1_posefield_and_recompose
+ d2_price_receipt (warp-param rank-1, FD lattice); frozen_scorer_exact_factorization_20260715 (A_pose≡A_seg,
frame_0 seg-free, chroma box-avg); segnet_recursive_fractal_factorization_20260715 (the rank-4 precedent);
null_subspace_rate_measure_20260717; ddm_deferral_queue_ledger_20260729 (QA47/QA51); gt_n96 cache;
tac.differentiable_eval_roundtrip + tac.torch_mps_compat + tac.scorer_targets.
