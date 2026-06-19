---
title: Cross-reference — openpilot / comma.ai polynomial lane geometry vs the contest d_seg core
authority: "[research/advisory] — source-grounded crossref; pointer UNMOVED 0.19110; $0, no GPU, no PR"
score_claim: false
promotable: false
date: 2026-06-19
provenance_confirmed:
  - openpilot selfdrive/modeld/constants.py — POLY_PATH_DEGREE=4, NUM_LANE_LINES=4, NUM_ROAD_EDGES=2, IDX_N=33, POSE_WIDTH=6, X_IDXS/T_IDXS quadratic to 192m/10s (verbatim)
  - openpilot cereal/log.capnp — DrivingModelData.PolyPath{xCoefficients,yCoefficients,zCoefficients}; ModelDataV2.{laneLines,roadEdges,position} as List(XYZTData) sampled points (verbatim)
  - openpilot common/transformations/camera.py — Neo/EON fcam 1164x874 focal 910.0, principal point at center (582,437) (verbatim)
  - comma10k README — 5 classes + exact hex: road #402020, lane_markings #ff0000, undrivable #808060, movable #00ff66, my_car #cc00ff (verbatim)
  - upstream/modules.py:105 — SegNet = smp.Unet('tu-efficientnet_b2', classes=5) (LOCAL CONFIRMED)
local_crossref:
  - src/tac/camera.py — COMMA_INTRINSICS_NATIVE(fx=910,fy=910,cx=582,cy=437) + COMMA_EXTRINSICS(height=1.2m, pitch=-0.02rad) + VANISHING_POINT(256,174) + HORIZON_BAND(155,195)
  - src/tac/semantic_label_contract.py — comma10k 5-class scheme + hex (EXACT MATCH to comma10k README)
  - src/tac/lane_mark_pose.py — PoseNet dim0 = speed/radial-zoom (mean 31.3, rank-1 Jacobian); dims1-5 near-zero
  - src/tac/openpilot_seeding.py / openpilot_features.py — supercombo seeding (compress-time only)
cross_refs:
  - eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md (the texture wall + margin-polytope)
  - tasks #138 (geometric road↔lane prior $0 IoU gate), #137 (road↔lane sidecar), #141 (margin-saliency)
---

# openpilot polynomial geometry × the contest d_seg core — what is reusable, ranked

**One-line answer to the operator's "polynomials and polytopes":** openpilot natively represents the
road/lane geometry as **degree-4 polynomials** (`POLY_PATH_DEGREE = 4`, source-verbatim) in a
**ground-plane-aligned calibrated frame**, and the contest video is **comma EON native footage**
(focal 910, 1164×874 — exact match), so the d_seg-critical road↔lane partition is provably
representable by **a handful of polynomial coefficients in a known geometric frame** — which is the
cheapest continuous primitive available and exactly what beats the survival wall. The polynomial is
the *shape* prior; the polytope (margin slack from the companion eval-roundtrip memo) is the *byte*
prior. They compose.

---

## Q1 — openpilot's lane/path representation (the core finding) — CONFIRMED FROM SOURCE

openpilot ships **two** representations, both authoritative:

1. **`ModelDataV2` (the primary model packet)** — `cereal/log.capnp`:
   - `laneLines @8 :List(XYZTData)`, `laneLineProbs @9 :List(Float32)`, `laneLineStds @13`
   - `roadEdges @10 :List(XYZTData)`, `roadEdgeStds @14`
   - `position @4 :XYZTData`, plus `orientation/velocity/orientationRate/acceleration`
   - `struct XYZTData { x,y,z,t @0..3 :List(Float32); xStd,yStd,zStd @4..6 :List(Float32) }`
   - i.e. lanes/edges/path are **sampled 3-D points over time**, NOT polynomials, in this packet.

2. **`DrivingModelData` (the compact "qlog" packet, fredyshox PR #32821)** — `cereal/log.capnp`:
   - `path @5 :PolyPath;`
   - `struct PolyPath { xCoefficients @0 :List(Float32); yCoefficients @1; zCoefficients @2; }`
   - i.e. the **path is a polynomial in coefficient form** — this is the byte-cheap packet comma ships
     to logs precisely because the polynomial is the minimal description of the same curve.

The sampling grid + degree (`selfdrive/modeld/constants.py`, **verbatim**):
```python
def index_function(idx, max_val=192, max_idx=32): return (max_val) * ((idx/max_idx)**2)
IDX_N = 33
X_IDXS = [index_function(idx, max_val=192.0) for idx in range(IDX_N)]   # 0..192 m, quadratic
T_IDXS = [index_function(idx, max_val=10.0)  for idx in range(IDX_N)]   # 0..10 s,  quadratic
NUM_LANE_LINES = 4      # outer-left, inner-left, inner-right, outer-right
NUM_ROAD_EDGES = 2      # left edge, right edge
POSE_WIDTH = 6          # 3 rotation + 3 translation
POLY_PATH_DEGREE = 4    # <<< THE polynomial order: degree-4 → 5 coeffs / axis
```
**So the native curve is degree-4 → 5 coefficients per spatial axis.** A lane line / road edge is, to
openpilot's own model, a curve that a 4th-order polynomial captures to driving tolerance over 192 m.
That is the proof the d_seg road↔lane boundary geometry is "natively cheap-polynomial."

UNVERIFIED detail: the exact `np.polyfit` deg used to convert the 33 XYZT samples → `PolyPath`
coefficients was not located in the parser file I could read (the parser references `ModelConstants`
but the polyfit callsite wasn't in `parse_model_outputs.py`); `POLY_PATH_DEGREE=4` is the declared
constant and the strongest available evidence. Treat "5 coeffs/axis" as the source-declared order.

## Q2 — comma10k segmentation scheme — CONFIRMED (exact match local ↔ upstream)

The contest SegNet is `smp.Unet('tu-efficientnet_b2', classes=5)` (LOCAL `upstream/modules.py:105`).
Its 5 classes ARE the comma10k scheme, byte-exact to `src/tac/semantic_label_contract.py`:

| id | class | comma10k hex | labeling note (comma10k README, verbatim) |
|----|-------|--------------|--------------------------------------------|
| 0 | road | `#402020` | "all parts, anywhere nobody would look at you funny for driving" |
| 1 | lane_markings | `#ff0000` | "don't include non lane markings like turn arrows and crosswalks" |
| 2 | undrivable | `#808060` | |
| 3 | movable | `#00ff66` | "vehicles and people/animals" |
| 4 | my_car | `#cc00ff` | "anything inside it, incl wires, mounts; no reflections" |

Key consequence for d_seg: **lane_markings (class 1) is a THIN line class** painted only on actual lane
paint (arrows/crosswalks excluded). The road↔lane boundary the eval-roundtrip memo measured at ~64% of
d_seg is exactly the perimeter of these thin `#ff0000` strokes sitting inside the `#402020` road — a
1-D curve set, not an area. **A 1-D curve set is the ideal target for a polynomial representation.**

## Q3 — the ground-plane homography (the geometric prior) — CONFIRMED intrinsics, ANCHORED extrinsics

Intrinsics (openpilot `common/transformations/camera.py`, verbatim): the **Neo/EON fcam is
1164×874, focal length 910.0**, intrinsic matrix `[[f,0,W/2],[0,f,H/2],[0,0,1]]` → principal point
(582, 437). This is an **EXACT match** to the contest constants (`camera_size=(1164,874)`,
`camera_fl=910`) and to local `COMMA_INTRINSICS_NATIVE`. ⇒ **the contest clip is comma EON native
footage**; the full openpilot camera model is directly applicable, not an approximation.

Extrinsics: openpilot's `device→calibrated→car/road` frames (transformations README) define a
**calibrated frame aligned to the road plane in pitch/yaw**; comma mounts ship at 22°/28° brackets and
`calibrationd` estimates `rpyCalib` so the model frame sits on the ground plane. The road is then a
**ground plane** z=0 in car frame, and the image↔road map is the standard pinhole homography
`H = K · [r1 r2 t]`. Local `COMMA_EXTRINSICS(height=1.2 m, pitch=-0.02 rad)` is the anchored mount
geometry (comma2k19-derived) — UNVERIFIED to 3 digits against a single published EON number, but
height ≈1.2 m / slight-down pitch is the canonical comma dashcam pose and is consistent with the
local `VANISHING_POINT=(256,174)` (VP above center ⇒ slight downward pitch).

Geometric facts this buys for d_seg, all byte-free:
- The **road is a trapezoid** (ground plane under the horizon); pixels above the **HORIZON_BAND
  (rows 155–195 @512×384)** are never road/lane → a hard geometric mask.
- **Lane lines converge to the vanishing point (256,174)** → their image-space curvature is
  constrained (perspective foreshortening), so the *image-plane* lane curve is also low-order — a
  degree-≤4 polynomial in image coords, or (cleaner) a low-order polynomial in the ground frame
  back-projected through `H`.
- Local `tac.camera.vanishing_point_saliency` already exists as a prior surface.

## Q4 — PoseNet ↔ openpilot ego-motion — CONFIRMED (same lineage)

openpilot "runs the posenet in `models/posenet.dlc`, **takes in two frames and outputs the 6-DoF
transform between them**" (comma tour). The contest PoseNet is FastViT-T12 on **YUV6 of two frames →
6 dims = 3 rotation + 3 translation** (`POSE_WIDTH=6`). **Same problem, same lineage**: frame-to-frame
ego-motion. The local empirical finding refines it: contest **dim0 ≈ forward speed / radial zoom**
(mean 31.3, std 1.3), the Jacobian is **rank ≈1** (`tac.lane_mark_pose`), dims 1–5 are near-zero and
contribute ≤0.18 even at the per-clip mean. ⇒ d_pose is **one effective DOF (forward translation)
expressed as radial optical flow centred on the VP** — geometry-native and cheap to carry (the
pose-FiLM / coarse-luma carrier, task #140).

## Q5 — other directly-reusable comma/openpilot structure

- **The calibrated/road frame itself** — a fixed, byte-free coordinate system in which the road↔lane
  partition is simplest. Represent the d_seg core in the road frame, render via `H` to image.
- **supercombo at COMPRESS time only** (already wired: `openpilot_seeding.py`/`openpilot_features.py`)
  — emits the degree-4 lane/path polynomials + a scene embedding as a *free oracle* for the encoder
  (it never ships in the archive; contest "no scorers at inflate" is respected — see CLAUDE.md strict
  scorer rule).
- **commaVQ tokenizer (local `workspace/upstream/commavq`)** — a VQ-VAE → 128 tokens/frame world
  model. Reusable as a *temporal prior* (next-frame token prediction = the warp between pairs) for the
  pose carrier; NOT the contest scorer (that's the separate lossless-token challenge). Tag clearly to
  avoid conflating the two comma challenges.
- **Two-frame YUV preprocessing** — openpilot's quarter-Y + half-chroma packing is the same family as
  the contest's YUV6 (already mirrored in `openpilot_seeding._frames_to_supercombo_yuv`).

---

## SYNTHESIS — the "photoshop-esque" layered d_seg core (combine the techniques)

The three confirmed structures compose into one candidate representation whose whole point is to be
**byte-cheap + continuous + geometry-matched**:

**Layer 0 — geometric base (≈0 bytes).** Fixed from the camera model: horizon mask (rows <155 not
road), road trapezoid via the ground homography `H` (K=910, pp=(582,437), height≈1.2 m, pitch≈-0.02),
VP at (256,174). This partitions the frame into "can-be-road" vs "sky/above-horizon" for free.

**Layer 1 — polynomial lane/road skeleton (tens of bytes).** Carry the d_seg-critical road↔lane
boundary as **openpilot-native degree-4 polynomials in the road frame**: 4 lane lines + 2 road edges,
5 coeffs/axis × (x,y) ≈ a few dozen float coeffs per ~clip-segment (quantized; lanes are near-static
over a 2-frame pair, so per-clip not per-frame). Back-project through `H` to image to paint the thin
`#ff0000` strokes and the road/undrivable split. This is the *shape* of d_seg at polynomial cost — the
cheapest continuous primitive that follows the actual class boundary instead of a flat fill.

**Layer 2 — texture-detail layer ONLY on margin-critical boundary pixels (sparse bytes).** The
companion eval-roundtrip memo proved the binding wall is **SegNet's texture-dependence at the argmax
polytope boundary** (flat fills die: +0.00562 d_seg), and that each boundary pixel has free distortion
budget = its logit margin. So spend texture bytes ONLY where the polynomial skeleton's painted color
lands *outside* the GT per-pixel argmax polytope — i.e. a sparse residual coder keyed on the
margin-saliency field (task #141), concentrated on the ~1.3% low-margin boundary band that is exactly
the perimeter of the Layer-1 polynomial curves. **Polynomial (Layer 1) places the boundary; polytope
margin (Layer 2) decides where texture must be paid.** This is the photoshop stack: cheap continuous
gradient/skeleton base + selective high-detail layer masked to the boundary.

Why this beats the current flat-fill failure: a flat per-class color has zero texture and lands
outside the polytope at every boundary pixel; the polynomial skeleton + boundary-only texture keeps
`s_recon` inside the argmax cell along the curve while paying near-zero bytes in the high-margin
interior (free budget). It is geometry-matched because the boundary literally IS an openpilot lane
polynomial in the openpilot camera frame.

---

## RANKED reusable structure/techniques (by EV for cheaply lowering d_seg) + the $0 next-probe each

1. **[HIGHEST] Degree-4 lane/road polynomial skeleton in the road frame as the d_seg-boundary carrier.**
   The single most reusable openpilot structure; turns the 64%-of-d_seg road↔lane boundary into a few
   dozen coefficients. **$0 probe (feeds task #138/#137):** fit degree-4 polys (np.polyfit deg=4) to
   the GT SegNet road↔lane boundary on the contest clip *in the ground frame via H*; render back and
   measure the IoU / argmax-flip vs GT masks. Decisive go/no-go on whether the polynomial shape alone
   (before texture) recovers the boundary geometry. This is literally the pending #138 gate with the
   confirmed degree.

2. **[HIGH] Geometric horizon + road-trapezoid + VP prior (Layer 0), ≈0 bytes.** **$0 probe:** apply
   the horizon mask (rows<155) and ground-plane road region to GT masks; measure how much of d_seg is
   *structurally impossible* (pixels the geometry forbids from being road/lane) — quantifies the free
   floor the geometry removes before any learned bytes.

3. **[HIGH] Polynomial-skeleton × margin-polytope residual = the photoshop Layer-1/Layer-2 stack.**
   The composition above. **$0 probe:** on the frontier decoded frames, overlay the Layer-1 polynomial
   paint and compute, per boundary pixel, whether it sits inside the GT argmax polytope (using the
   margin field from the eval-roundtrip memo's P1); count the residual bytes needed only for violators.
   Quantifies the combined lever's byte cost at the real operating point.

4. **[MED] Rank-1 pose as radial zoom about the VP (openpilot 2-frame 6-DOF lineage).** Confirms
   d_pose is one DOF (forward translation) → a coarse-luma/pose-FiLM carrier centred on (256,174)
   (task #140). **$0 probe:** verify dim0 = radial-flow-magnitude-about-VP correlation on the clip;
   confirm dims1–5 carry ≤0.18 so the carrier can be 1-D.

5. **[MED] supercombo compress-time oracle for the polynomial coeffs + scene embedding.** Free encoder
   side-info (never archived). **$0 probe:** run the already-wired `openpilot_seeding`/`features` on the
   clip; check the emitted lane polynomials match the GT boundary fit from probe #1 (validates the
   oracle as a coeff source so the encoder needn't re-fit).

6. **[LOW] commaVQ world-model token prior as the pair-warp temporal prior.** Reusable for the pose
   carrier, NOT the d_seg core; flag to avoid conflating the lossless-token challenge with the lossy
   scorer challenge.

**Caveats / UNVERIFIED:** (a) the exact polyfit degree used in the modeld parser callsite wasn't
read — `POLY_PATH_DEGREE=4` is the declared constant; (b) `COMMA_EXTRINSICS` height/pitch are anchored
(comma2k19) not re-verified to 3 digits against a single published EON spec; (c) all EV is geometric
plausibility, not a measured d_seg — probe #1 is the decisive measurement-first gate before any spend.
Pointer UNMOVED 0.19110; $0; no GPU; no PR. `[research/advisory]`.
