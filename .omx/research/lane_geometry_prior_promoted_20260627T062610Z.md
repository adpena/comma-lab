# Lane-geometry prior PROMOTED (#145 cross-ref / #144 polynomial-fill / #137 road-lane sidecar) — FEED-di

**UTC:** 2026-06-27T06:26:10Z
**Lane:** `lane_geometry_prior_137_144_145`
**Authority:** `[macOS-CPU advisory]` research-signal — `score_claim=false`, `promotable=false`,
`ready_for_exact_eval_dispatch=false`. $0 CPU-only. NOT a byte-closed row. Frozen CPU-torch SegNet
argmax partition (cached `lstars`, NO surrogate, NO new scorer pass; bit-exact per FEED-db).
**GPU untouched** (levelset descent pid 72600/72602 + review subagent ran concurrently; this task
read only `lstars` from `gt_n96.npz`, ~150 MB, no GPU/MPS, no scorer fleet).

<!-- FORMALIZATION_PENDING: lane-orbit intrinsic-dim + recon-d_seg decomposition is a fresh
empirical-finding memo; the canonical equation (lane-class-1 structured rate = centerline_poly +
width_poly + dash_periodic, ground-frame) is queued for tac.canonical_equations registration once a
byte-closed witness row confirms the predicted d_seg benefit. -->

---

## 0. TL;DR (answers to the 3 questions)

1. **Lane-orbit intrinsic dimension (MEASURED, n96, frozen CPU-torch L*):** the lane class-1 region is a
   **very low-order polynomial manifold** — per lane line: centerline ≈ **2–3 coeffs** (median degree
   1–2 to a 1 px centerline band) + width(forward) ≈ **2 coeffs** + dash (period, phase) ≈ **2 params**
   ≈ **~7 floats/line**; ~**4.5–5 lane lines/frame** → **~30–40 video-derived floats/frame** reconstruct
   the ENTIRE lane class-1 region. This matches the CLAUDE.md "~8-dim lane-trajectory coords per frame"
   frontier claim.
2. **Ground-plane vs image-plane:** **NOT a decisive coeff-count win on THIS contest segment** — the
   comma2k19 RAV4 segment is mostly **STRAIGHT** (lane lines are already near-straight in the image:
   deg-1 centerline residual median **0.82 px**). At the in-tree horizon (row 174) image is marginally
   cheaper; at the **IPM-optimal horizon (row ≈188)** ground is marginally cheaper (deg-1 residual
   **0.70 vs 0.82 px**). The openpilot "ground ≪ image" advantage manifests under **road curvature**,
   which this segment largely lacks. **The one decisive ground-frame win is the DASH model** (below).
3. **Is the geometry prior a real complementary d_seg lever?** **YES.** The polynomial centerline+width
   band captures the lane *shape* so well that it MISSES only **d_seg 0.00046** of true lane pixels
   (false-negatives) — **already BELOW the witness target 0.00087**. The entire band-reconstruction
   d_seg (0.0044) is **90% false-positives from filling DASH GAPS** (0.00396). Dashes are periodic in
   **ground meters (constant period)** but a **chirp in image rows** → modeling them is cheap in the
   ground frame. So the lever = give the witness the lane SHAPE deterministically (free, ~30 floats) +
   a dash model; the witness's learned capacity then handles only the residual annulus. COMPLEMENTARY
   to the lane-edge loss lever (loss focuses gradient on flips; geometry reduces the dimension the
   witness must represent).

---

## 1. openpilot lane geometry — research (cited; CONFIRMED vs INFERRED)

### 1.1 The driving model lane representation — ground/calibrated frame, low-order along forward distance
- **CONFIRMED** (openpilot `common/transformations/camera.py`, fetched 2026-06-27 via `gh api
  repos/commaai/openpilot/contents/openpilot/common/transformations/camera.py`): the **road frame is
  `x→forward, y→left, z→up`** (line 84 `# road : x->forward, y -> left, z->up`). Lane lines / path are
  parametrized **along the forward axis**. `get_view_frame_from_road_frame(roll,pitch,yaw,height)`
  builds the extrinsic `view_from_road = view_frame_from_device_frame · (R_euler · diag([1,-1,-1]))`
  with translation `[0, height, 0]`.
- **CONFIRMED** (supercombo model, comma.ai blog "How openpilot works in 2021"
  https://blog.comma.ai/openpilot-in-2021/ + MTammvee/openpilot-supercombo-model README): path / left
  lane / right lane are predicted as **trajectories over 33 timestamps, quadratically spaced to 10 s /
  192 m**, in **bird's-eye / calibrated (road) coordinates relative to ego**, each as **mean + std**.
  So the canonical lane representation is a **low-dimensional curve in the ground frame** (forward →
  lateral), NOT a per-pixel image structure.
- **CONFIRMED (historical)** the *older* openpilot model emitted lane lines directly as **3rd-order
  polynomials** `y = a·x³ + b·x² + c·x + d` (lateral as a cubic in forward distance) — the modern
  supercombo replaced the explicit polynomial with the 33-point trajectory, but both encode the same
  **cubic-class low-order ground curve**. (Polynomial-path lineage: Mankaran Singh, "End-to-End Motion
  Planning…", https://mankaran32.medium.com/end-to-end-motion-planning-with-deep-learning-comma-ais-approach-5886268515d3.)

### 1.2 The ground homography / IPM (medmodel frame)
- **CONFIRMED** (openpilot `common/transformations/model.py`): `MEDMODEL_INPUT_SIZE=(512,256)`,
  `medmodel_fl=910.0`, `MEDMODEL_CY=47.6`, `medmodel_intrinsics=[[910,0,256],[0,910,47.6],[0,0,1]]`;
  `medmodel_frame_from_calib_frame = medmodel_intrinsics · get_view_frame_from_calib_frame(0,0,0,0)`;
  `calib_from_medmodel = inv(medmodel_frame_from_calib_frame[:,:3])`. The "ground_from_medmodel_frame"
  the prompt names is this `calib_from_medmodel` (image-homogeneous → calibrated ray frame; intersect
  the road plane z=0 at camera height for the BEV ground point). IPM = projecting image pixels onto the
  flat road plane via intrinsics+extrinsics (EmergentMind "Inverse Perspective Mapping (IPM)").
- **CONFIRMED (in-tree)** `src/tac/camera.py`: scorer-resolution `COMMA_INTRINSICS` fx=400.3, fy=399.5,
  cx=256.0, cy=192.0 (910 scaled 1164→512 / 874→384); `COMMA_EXTRINSICS` height=1.2 m, pitch=−0.02 rad;
  native comma2k19 `fx=fy=910, cx=582, cy=437` (comma2k19 `camera_intrinsics.txt`,
  https://github.com/commaai/comma2k19/blob/master/notebooks/camera_intrinsics.txt). `VANISHING_POINT`
  row 174 (the empirically-calibrated horizon used by this measurement's IPM).

### 1.3 comma10k class-1 = lane markings (CONFIRMED)
- **CONFIRMED** (commaai/comma10k README): 5 classes — `1 #402020 road`, **`2 #ff0000 lane markings`**
  ("don't include non lane markings like turn arrows and crosswalks"), `3 #808060 undrivable`,
  `4 #00ff66 movable`, `5 #cc00ff my car`. The SegNet 0-based contest argmax = `[road 0, lane 1, …]`
  (matches `tac.camera` + DAG class-order). So **class index 1 = lane markings (lines only)** — a clean
  geometric target for a polynomial prior (crosswalks/arrows excluded by the labeling convention).

---

## 2. Measured lane-orbit intrinsic dimension (n96, $0, frozen CPU-torch L*)

Method: read only cached `lstars` (96×384×512 int64 = frozen CPU-torch SegNet argmax). Class-1 mask →
cluster pixels into lane lines by **BEV lateral** (groups a line's dashes across forward distance into
one curve — the openpilot-aligned grouping). For each line fit IMAGE `u=poly_D(v)` and GROUND
`lateral=poly_D(forward)` (reprojected), scored by **centerline RMS in image pixels** (the marking
width is a separate term, not curve-fidelity). IPM = small-angle flat-ground:
`forward = H·fy/(v−v_h)`, `lateral = −(u−cx)·forward/fx`, `v_h=174`, `H=1.2`.

| quantity | value |
|---|---|
| class-1 pixel fraction / frame | mean **0.6%** (~1180 px), max 0.83% |
| connected components / frame (dashes fragment lines) | mean **30** (20–42) |
| **fraction of class-1 in margin<2 (flip-prone band)** | **mean 0.94, median 0.96** |
| lane-line clusters / frame (BEV-grouped) | mean 4.5, median 5 |
| IMAGE deg-1 centerline residual px (straightness) | median **0.82**, p90 1.60 |
| GROUND deg-1 reproj residual px (v_h=174) | median 1.18 (worse — far-field IPM noise) |
| GROUND deg-1 reproj residual px (**v_h≈188, IPM-optimal**) | median **0.70** (beats image) |
| centerline min-deg @1.0 px: IMAGE / GROUND | median **1 / 2** (2.56 / 2.86 coeffs/line) |
| centerline min-deg @1.5 px: IMAGE / GROUND | median **1 / 1** (2.17 / 2.41 coeffs/line) |
| measured marking width px | median 6 (p10 2 far, p90 9 near) |
| **GROUND structured floats / frame** (centerline deg3 + width deg1, all lines) | median **30** |

**The 94% finding** (class-1 ≈ flip-prone margin<2 band) is a strong structural confirmation: the lane
class IS essentially the boundary orbit FEED-dd/dg identified (lane IoU 0.263) — there is almost no
"interior" lane to waste capacity on; the lane class is all edge.

### 2.1 Reconstruction-d_seg (the decisive test) and its FP/FN decomposition
Rasterize a **compressed representation** = ground deg-3 centerline + width(forward) deg-1 band per
line, union → reconstructed class-1 mask → disagreement vs true L* class-1:

| component | d_seg | px/frame | meaning |
|---|---|---|---|
| **false-negative** (missed true lane px) | **0.00046** | 91 | shape captured — **BELOW target 0.00087** |
| **false-positive** (band fills dash gaps) | **0.00396** | 777 | the band is continuous; lanes are DASHED |
| total band recon d_seg | 0.00442 | 868 | ≈ witness plateau 0.0032 |
| cluster coverage of class-1 px | 1.00 | — | clusters span ~all class-1 rows |

**90% of the reconstruction error is dash-gap false-positives.** The polynomial centerline+width model
already reaches FN d_seg 0.00046 (below the witness target). The dominant residual is the **DASH
ON/OFF pattern**, which is the cheap, principled ground-frame win (next).

### 2.2 Why ground beats image only marginally HERE, decisively for dashes
- **Centerline:** a straight flat-road lane line projects to a near-straight IMAGE line (all parallel
  ground lines meet at the VP; each individual line is straight in image). Since this segment is mostly
  straight (image deg-1 residual 0.82 px), the ground frame's "undo perspective" advantage on the
  centerline is small (and an uncalibrated horizon even hurts via far-field noise). At the IPM-optimal
  horizon (v_h≈188) ground edges out image (0.70 vs 0.82). **Verdict: ground ≈ image on centerline for
  THIS video; ground is the safer parametrization that STAYS low-order under curvature (generalizes).**
- **Dash period (the real ground win):** road dashes are a **fixed length in world meters** → **constant
  period in forward distance**, but a **chirp in image rows** (period shrinks toward the horizon).
  Modeling dashes costs **2 params (period+phase) in the ground frame** vs a high-order / non-stationary
  model in the image frame. INFERRED from geometry (not separately fit here); this is the load-bearing
  ground-plane advantage for the 90%-FP dash residual.

---

## 3. Proposed integration into the level-set witness (COMPLEMENTARY to the lane-edge loss lever)

The witness is a coordinate-INR / level-set amortizing the SegNet argmax partition. Add a **deterministic
lane-geometry channel** that supplies the lane SHAPE for free, reallocating capacity onto the hard
residual annulus:

1. **Stored per-frame lane parameters `θ_lane(t)` (VIDEO-DERIVED → COUNTED in archive.zip):** for each of
   ~5 lines: ground centerline coeffs `(a0,a1,a2,a3)` + width coeffs `(w0,w1)` + dash `(period, phase)`
   ≈ **~7 floats/line × ~5 = ~35 floats/frame**. Stored with temporal-delta (lanes move slowly:
   consecutive lateral-offset |Δ| ~0.8 m, smooth) + AR/brotli coding.
2. **Generic rasterizer in inflate.py (FREE — the "compile nonlinear d_seg" rule):** `θ_lane(t)` →
   ground centerline poly → reproject to image (the in-tree IPM, a fixed deterministic transform) →
   band of width `w(forward)` gated by the dash on/off computed in the **ground frame** (constant
   period) → image lane indicator `B_lane(x,y,t)`. This is generic algorithm (allowed free per
   contest rule 118 + CLAUDE.md "inflate.py is a FREE interpreter"); only the coeffs are counted.
3. **Witness consumption — two compatible options:**
   - **(a) prior/bias (lighter):** add `B_lane` as an additive bias to the class-1 level-set logit, so
     the partition is pulled toward lane mass where `B_lane=1`. The witness still renders all classes;
     the prior just shapes the lane head.
   - **(c) deterministic-lane + residual (stronger):** render the lane class-1 region DIRECTLY from
     `B_lane` (FN already 0.00046 < target) and let the witness model ONLY the non-lane classes + the
     all-class boundary annulus residual. This DIRECTLY reallocates the bytes/capacity off the lane
     shape (FEED-da/CLAUDE.md "reallocate the SAME bytes onto the task-space manifold").
   - Either way, FiLM-condition the witness lane head on `θ_lane(t)` (the per-frame coeffs) so it
     learns the residual *given* the shape.

**Complementarity with the lane-edge loss lever (FEED-dc/de, the in-review #1 lever):** orthogonal axes.
- *loss lever:* up-weights realized margin-hinge on fragile lane-class-1 flips → focuses GRADIENT.
- *geometry prior:* gives the witness the lane SHAPE (low-order curve + dash) → reduces the DIMENSION
  the witness must learn. Net: the prior removes the bulk lane mass deterministically (free shape, FN
  below target), the loss + remaining capacity crack the residual annulus. Not redundant; multiplicative.

### 3.1 Predicted d_seg benefit + byte cost (advisory, NOT byte-closed)
- **d_seg:** the lane class-1 contribution is a chunk of the 0.0032 plateau (lane = 19% of all-class
  flips + induces adjacent road/hood boundary flips; lane IoU 0.263 is THE unstable orbit). A
  deterministic lane shape with FN 0.00046 + dash model could remove most of the lane-attributable
  d_seg, predicted plateau **0.0032 → ~0.002–0.0025** `[macOS-CPU advisory; needs byte-closed witness
  row to confirm]`. The FN floor (0.00046) is the structural existence-proof that the shape is not the
  wall — the dash + the non-lane all-class edges are.
- **bytes (COUNTED):** ~35 floats/frame × 600 = 21k floats; temporal-delta + AR coding (slow lane
  motion, constant dash period) → estimated **~1–2 KB** → rate-score `25·1500/37.5M ≈ 0.001`. Net
  strongly positive (predicted d_seg drop ×100 ≫ rate add ×25·tiny). Consistent with FEED-dd "rate is
  NOT the binding constraint."

---

## 4. Honest caveats / NO-FAKE
- **n96 only** (not n600); straight-segment finding may not hold for any curved sub-segment of the full
  600 — the ground-frame centerline advantage would GROW on curves (so the recommendation is
  conservative: ground frame chosen for generalization even though image ≈ ground here).
- **Ground-vs-image centerline is a near-tie on this video** — I do NOT claim ground ≪ image in
  coeff-count (the prompt's literal "structural win" hypothesis is **only partially confirmed**: ground
  wins decisively for the DASH period, marginally for the centerline at calibrated horizon, not at all
  at the in-tree horizon). The robust structural win is "lane orbit = ~7-float/line polynomial+dash
  manifold," frame-agnostic.
- **Dash period is INFERRED from geometry** (constant in world meters), not separately fit in this pass
  — a quick $0 follow-up (autocorrelation of class-1 on/off along forward distance) would confirm and
  give the period distribution; queued.
- **recon-d_seg is on the continuous band without the dash model** — the 0.0044 is the upper bound; the
  FN 0.00046 is the achievable floor once dash is modeled. Both measured bit-exact on the frozen L*.
- This is a **PRIOR / structural lever**, not a witness output by itself — the predicted d_seg benefit
  requires a byte-closed witness row to verify (the next step, not this memo).

## 5. Observability surface
- Per-line: centerline coeffs, width coeffs, deg-1 residual (straightness), min-deg per tolerance — all
  printed per run and decomposable per line / per frame.
- Reconstruction d_seg decomposed into FP (dash) / FN (shape) — diff-able vs the true L* mask.
- Scripts (scratch, reproducible): `measure_lane_orbit_v3.py` (intrinsic dim + recon), V_H sweep, FP/FN
  decomposition. Inputs: `experiments/results/mlx_fleet_gt_cache/gt_n96.npz` `lstars` member only.

## 6. Wire-in (6-hook)
1. sensitivity-map: lane class-1 = the binding d_seg orbit (94% in margin<2) — feeds the lane-edge
   saliency the loss lever already consumes. 2. Pareto: lane payload ~1–2 KB (rate non-binding). 3.
   bit-allocator: θ_lane coeffs are a new counted section (temporal-delta+AR). 4. cathedral: the
   inflate.py lane rasterizer is the FREE generic interpreter path. 5. continual-learning: this memo +
   DAG FEED-di. 6. probe-disambiguator: N/A (single measured verdict; ground-vs-image tie is reported,
   not arbitrated by a probe).

**Pointer UNMOVED 0.19110.** `[macOS-CPU advisory]` research-signal; no score claim; no byte-closed row.
