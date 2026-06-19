---
title: comma.ai / openpilot ecosystem — domain hints, tricks, techniques for the frozen contest instance
authority: "[research / advisory] — pointer UNMOVED 0.19110; no PR; $0; source-grounded (every fact cited)"
score_claim: false
promotable: false
date: 2026-06-19
operator_directive: "domain hints and tricks and techniques from openpilot and comma ai repos"
scope_note: "BROADER ecosystem deep-mine. COMPLEMENTS (does NOT duplicate) sister thread #145 (comma_openpilot_crossref_polynomial_geometry_20260619T014433Z.md) which owns lane-polynomial geometry + comma10k classes + ground-plane homography. This doc owns: exact camera/calibration provenance, the comma2k19 GT-pose unlock, the preprocessing-convention exploit, the kinematic pose null-space, the HEVC source-encode lineage, and the ranked $0-exploit synthesis."
provenance:
  - "upstream/frame_utils.py, upstream/modules.py, upstream/README.md, upstream/public_test_segments.txt (CONFIRMED contest source)"
  - "/Users/adpena/openpilot_research/openpilot @ ee54e8209 (2026-04-24): common/transformations/{camera,model,orientation}.py, selfdrive/locationd/calibrationd.py, selfdrive/modeld/{constants.py,fill_model_msg.py,parse_model_outputs.py}, system/loggerd/loggerd.h, common/constants.py (CONFIRMED, file:line cited)"
  - "github.com/commaai/comma10k README + HF commaai/comma10k-segnet model card + YassineYousfi/comma10k-baseline (CONFIRMED via web)"
  - "github.com/commaai/comma2k19 README + HF commaai/comma2k19 tree + arXiv 1812.05752 (CONFIRMED via web)"
cross_refs:
  - .omx/research/frozen_instance_exploit_catch_up_then_surpass_vcm_20260619.md (the lens this serves)
  - .omx/research/eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md (the operator chain)
  - .omx/research/comma_openpilot_crossref_polynomial_geometry_20260619T014433Z.md (sister #145 — disjoint scope)
---

# comma.ai / openpilot domain tricks for the frozen contest instance

**One-line headline (the single biggest find): the contest video is a known comma2k19 segment whose EXACT
per-frame ego-motion ground truth (ECEF positions/velocities/orientations) is PUBLICLY DOWNLOADABLE — the
d_pose target is recoverable to within the dataset's own ~0.6 m / 0.20-0.25° noise floor, at ~0 archive
bytes.** Provenance: `upstream/public_test_segments.txt` = `b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc`,
which is the comma2k19 **RAV4** route (dongle `b0c9d2329ad1606b` = RAV4, confirmed in comma2k19 README), and
the HF `commaai/comma2k19/compression_challenge/` tree literally contains
`b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc`. The matching `global_pos/` GT arrays for that segment
are in the same dataset.

Pointer UNMOVED 0.19110. This is research/advisory; every "exploit" below is a candidate $0 probe, not a score.

---

## 0. The frozen provenance — what the contest actually is (all source-confirmed)

| Fact | Value | Source |
|---|---|---|
| Camera = openpilot **"neo"/EON** config | `CameraConfig(1164, 874, 910.0)` | `common/transformations/camera.py:54` (`_neo_config`); contest `frame_utils.py:11-12` (`camera_size=(1164,874)`, `camera_fl=910.`) — **byte-identical** |
| Intrinsics K (exact) | `[[910,0,582],[0,910,437],[0,0,1]]` | computed from the above (principal point = W/2,H/2) |
| Video = comma2k19 RAV4 segment | `b0c9d2329ad1606b\|2018-07-27--06-03-57/10` | `upstream/public_test_segments.txt`; comma2k19 README ("dongle_id of the RAV4 is `b0c9d2329ad1606b`") |
| Source capture | **20 Hz, H.264**, 1200 frames / 1-min segment | arXiv 1812.05752 ("captured at 20Hz and compressed with H.264") → **1200 frames = exactly the contest's 600 pairs × 2** |
| comma road-cam HEVC (device logger) | 5 Mbit/s, GOP 20 (EON `main_road_encoder`) | `system/loggerd/loggerd.h:39` (`bitrate=5'000'000,.gop_size=20`) |
| Mount calibration (constant for this video) | height **1.22 m**, roll **≡0** (image NOT roll-corrected), pitch ∈ [-0.0907, 0.17] rad, yaw ∈ [-0.0691, 0.0691] rad | `selfdrive/locationd/calibrationd.py:6,38,42-45` |
| SegNet = openpilot comma10k segnet | `tu-efficientnet_b2` Unet, 5 cls, input `(512,384)` = openpilot `SEGNET_SIZE` | `upstream/modules.py:103`; `common/transformations/model.py:6` (`SEGNET_SIZE=(512,384)`) |
| SegNet scores **last frame only** | `x[:, -1, ...]` | `upstream/modules.py:108` |
| PoseNet = openpilot 2024+ backbone | `timm fastvit_t12`, 12-ch in (2 frames × YUV6), scores first 6 of 12 dims (MSE) | `upstream/modules.py:66,105` |
| Pose 6-dim = [v_fwd, v_lat, v_vert, ω_roll, ω_pitch, ω_yaw] | trans m/s, rot rad/s, calibrated frame [fwd,right,down] | `selfdrive/modeld/fill_model_msg.py:186-191`; `cereal/log.capnp:2282-2285`; order proven by `calibrationd.py:189,202-204` |

**Why this matters:** the contest is NOT a generic "compress a video" task. It is "compress ONE comma2k19
RAV4 highway segment, scored by two frozen openpilot networks whose entire training distribution, label
grammar, camera geometry, and ego-motion physics are public and KNOWN." Every domain fact below converts a
piece of that known structure into bytes we don't have to spend.

---

## 1. Camera / calibration geometry → the scene is a KNOWN homography (free d_seg structure)

**Domain facts (source):**
- The camera is the fixed openpilot neo mount: K = `[[910,0,582],[0,910,437],[0,0,1]]`, height 1.22 m,
  roll≡0, pitch/yaw within the tight calibration window above (`camera.py`, `calibrationd.py`).
- Ground-plane homography: a road point at distance *d* ahead lands at image row `v = 910·1.22/d + 437`
  (pitch≈0). **Computed (this session):** horizon = row **437**; 100 m ahead → row 448; 50 m → 459;
  20 m → 493; 10 m → 548; 5 m → 659. (Camera frame H=874.)
- Vanishing point of the forward axis sits ON the principal point (582, 437) at nominal calibration; the
  calibration window shifts it by at most ~±0.069·910 ≈ ±63 px horizontally / a few px vertically. (Computed
  from `get_view_frame_from_calib_frame` + `vp_from_ke`, `camera.py:96-118`.)

**The $0 exploit (→ task-space/quotient code #155 + dominated-rung #153):**
The *spatial layout* of the 5 SegNet classes is geometrically pinned by this homography and is near-static
across all 600 frames:
- **Rows < ~437 (above horizon) ≈ sky/undrivable** (the SegNet's confident `undrivable`, gray 124). Class is
  near-certain a priori → the SegNet argmax almost never flips there → these pixels need almost no texture
  fidelity (high-margin interior; see §3). A scene-prior class-map for the top band is a near-zero-byte
  d_seg structure.
- **The road trapezoid is geometrically bounded** by the homography (a wedge from ~row 437 widening down to
  the bottom, centered on the VP). The `road` interior (gray 41) is the dominant, maximally-confident class.
- **The ego-hood band at the bottom is fixed** (the RAV4 hood; `my car` gray 90; comma10k labels the visible
  hood/body). Static across the whole video → a constant region.
The exploit is a **geometry-derived margin/class prior**: instead of letting the renderer "discover" the
class layout, seed it (interior class certain, only the trapezoid edges + horizon line uncertain). This is
the indirect-RD "code the sufficient statistic, not the pixels" move: the homography IS most of the d_seg
sufficient statistic for the interior. Concretely it tells the bit-allocator WHERE the cheap (interior) vs
expensive (boundary) pixels are, geometrically, before any forward pass.

**Caveat (NO-FAKE):** the homography pins the *static-scene* class geometry, but `movable` objects (other
cars, gray 161) appear at data-dependent positions on the road and break the static prior — those are not
geometry-predictable. The prior is a *floor* on certainty for the static classes (road interior, sky, hood),
not a full partition. Realizing it still requires §3's texture inside the boundary polytopes.

---

## 2. The comma2k19 GT-pose unlock → a near-free d_pose code (HIGHEST EV)

**Domain facts (source):**
- The contest video = comma2k19 RAV4 segment `b0c9d2329ad1606b|2018-07-27--06-03-57/10` (§0).
- comma2k19 ships, per segment, `global_pos/{frame_positions, frame_velocities, frame_orientations,
  frame_times, frame_gps_times}` — ECEF camera position (m), ECEF velocity (m/s), and the Hamilton
  quaternion rotating ECEF→local-camera-frame [forward,right,down] (comma2k19 README; arXiv 1812.05752).
  These are at the **same 20 Hz** as the video, RAV4 accuracy 0.6 m N/E, 0.20° roll/pitch, 0.25° yaw.
- The contest PoseNet output IS this ego-motion: 6 dims = [v_fwd, v_lat, v_vert, ω_roll, ω_pitch, ω_yaw],
  calibrated frame, the same [fwd,right,down] convention (`fill_model_msg.py:186-191`).

**The $0 exploit (→ cheap pose code #140):**
The d_pose ground truth (the PoseNet output on the GT frames) is a **smooth, ~1-2-DOF, publicly-known signal**:
1. **It's recoverable.** Download the comma2k19 `global_pos/` arrays for this exact segment. The PoseNet's
   6-dim output on GT pairs is the per-frame velocity/angular-velocity; the comma2k19 GT velocity (rotate
   ECEF→local) + differentiated orientation give the same physical quantities at 20 Hz. We can fit the
   PoseNet's per-pair output to a tiny smooth code (below) and *verify the d_pose it produces against the
   real frozen PoseNet on GT* — no training, $0.
2. **It's ~1-2 effective DOF.** For a fixed-mount highway dashcam (the entire comma2k19 premise — 20 km of
   CA-280 highway): `v_fwd` is the only large, smooth channel (0…~35 m/s); `ω_yaw` is the only nonzero small
   rotational channel (path curvature; |·| typically <0.05 rad/s, hard cap 1, `paramsd.py:75`); the other 4
   (`v_lat, v_vert, ω_roll, ω_pitch`) are constant-zero-plus-noise OR fixed linear functions of `v_fwd` given
   the frozen calibration (`v_vert ≈ -v_fwd·tan(pitch_mount)`, etc.). Source: `pose_kf.py` obs-noise scales
   (trans 0.5 m/s, rot 0.05 rad/s) = the precision floor; quantizing finer is wasted bits.
3. **It's temporally low-order.** comma represents the ego trajectory itself as a **degree-4 polynomial**
   (`fill_model_msg.py:93`, `POLY_PATH_DEGREE=4`; `constants.py` plan structure). Over a 60 s segment the
   600 per-pair pose vectors are a handful of smooth curves → code as low-order temporal polynomials /
   first-order deltas with residuals at the noise floor.

**Concrete pose code (the #140 actuator, sized):** for each of the 6 dims, fit a low-order polynomial (or
constant + linear) over the 600 pairs; quantize residuals at the per-dim noise floor (0.5 m/s trans, 0.05
rad/s rot). Realistic size: ~5 poly coeffs × 6 dims + sparse residuals ≈ low-hundreds of bytes, vs a per-pair
6-float dump (600×6×4 = 14.4 KB). And the d_pose this incurs is bounded BELOW by the dataset's own GT noise.
At the operating point ∂S/∂d_pose ≈ 85.8 (√-fragility), getting d_pose right is high-leverage — and we have
the answer key.

**Caveat (NO-FAKE):** the contest scores the **frozen PoseNet's output on our RECONSTRUCTED frames** vs its
output on GT frames — NOT the comma2k19 GT pose directly. So the comma2k19 arrays are a *prior / verification
oracle*, not a drop-in. The real lever is: (a) the comma2k19 GT tells us the pose trajectory is smooth and
~1-2 DOF, so a tiny pose-aware carrier suffices; (b) we still must make the reconstructed frame_pair PRODUCE
that pose through the real PoseNet (the YUV6/luma-texture path of §4). The unlock is the strong, free,
verified PRIOR on the pose signal's shape, which collapses the pose search space dramatically. It does not
zero d_pose by itself.

---

## 3. comma10k label grammar → the margin-map (which d_seg pixels are cheap vs fragile)

**Domain facts (source — exact 5-class table, gray↔hex verified by BT.601 arithmetic):**

| argmax-relevant value | gray | hex | class | meaning |
|---|---|---|---|---|
| road | 41 | `#402020` | **road** | all drivable surface |
| lane | 76 | `#ff0000` | **lane markings** | painted lines only (NOT turn-arrows/crosswalks) |
| my-car | 90 | `#cc00ff` | **my car** | ego hood/body + interior |
| undrivable | 124 | `#808060` | **undrivable** | catch-all INCLUDING **sky**, buildings, vegetation, curb |
| movable | 161 | `#00ff66` | **movable** | vehicles + people/animals (dynamic) |

(Source: comma10k README; YassineYousfi/comma10k-baseline `LitModel.py`; HF `commaai/comma10k-segnet` card.)
**MUST-VERIFY:** the README class *order* (`road,lane,undrivable,movable,my-car`) differs from the
gray-ascending order (`road,lane,my-car,undrivable,movable`) used in baseline code; the frozen SegNet's
argmax-channel→class index must be confirmed empirically (which output channel lights up on known road
pixels). Positions 3-5 (undrivable/movable/my-car) are the ambiguous ones.

**Inductive biases (source):**
- Baseline trains with **plain CrossEntropy, NO class weighting** (`LitModel.py`) → training is dominated by
  the majority class (road) → **road interiors are maximally confident; minority classes (esp. lane markings,
  the rarest/thinnest) are fragile.**
- EfficientNet-B2 **stride-2 stem** halves resolution at input → structures below ~half-res are invisible
  (thin lane lines, distant small `movable`) — the model's blind spot AND its fragile band.
- comma10k labeling rules create rule-exception fragile zones: turn-arrows/crosswalks are labeled `road` not
  `lane` (frequent labeler error → low margin); curb/shoulder is the subjective road↔undrivable line.

**The $0 exploit (→ margin-map #141 + dominated-rung #153):**
This is the qualitative map of where d_seg lives, BEFORE any forward pass:
- **Cheap (high-margin interior, near-zero d_seg cost — the dominated rungs):** road-surface interior,
  sky/undrivable expanse (top band, §1), hood-band center (§1). These argmax-flip rarely → spend minimal
  texture bytes here. Combined with §1's geometry, the prior says: the top ~third of the frame and the hood
  band are almost free.
- **Fragile (low-margin, the d_seg-critical 1.3% — where bytes must go):** lane-marking pixels (rarest,
  thinnest, unweighted, below-stem-resolution); road↔undrivable curb/shoulder edges; all `movable` object
  silhouettes (data-dependent, the §1 prior's blind spot); the hood-top silhouette; turn-arrow/crosswalk
  rule-exception zones.
This is exactly the §5.2 margin-polytope lever of the eval-roundtrip memo, now with a *semantic* prior on
where the boundary band IS (lane lines + object silhouettes + curb), so the boundary-residual coder can be
seeded geometrically/semantically rather than discovered.

---

## 4. Preprocessing conventions → chroma is cheap, luma-edges are the currency

**Domain facts (source):**
- The contest's `rgb_to_yuv6` packing (`frame_utils.py:74-78`: `[y00,y10,y01,y11,U_sub,V_sub]`) is
  **structurally identical** to openpilot's production `frames_to_tensor` (`compile_modeld.py:49-58`): 4 luma
  phase channels (the 2×2 quad at full res) + U,V **already 2×2-box-averaged + 4:2:0-subsampled** before the
  model sees them. Both stack **2 frames** (12 ch; `IN_CHANS=6*2`, `N_FRAMES=2`).
- GT decode is **BT.601 limited-range** (`yuv420_to_rgb`: `(y-16)*255/219`, chroma bilinear-upsampled) →
  RGB-uint8 round → then `rgb_to_yuv6` re-derives full-range YUV. All steps `.clamp_(0,255)`.
- PoseNet normalizes `(x-127.5)/63.75`; SegNet uses **NO normalization** (raw [0,255] YUV6, last frame only).
- FastViT-T12 = RepMixer (depthwise-conv, **local-to-mid receptive field**, not global attention) — pose is
  recovered from **local mid/high-freq luma parallax** across the 2-frame pair.

**The $0 exploits:**
1. **Chroma compression is task-safe (dominated-rung #153).** Chroma reaches both models already at 1/4 the
   spatial budget (4:2:0 + 2×2 box-average, applied BEFORE the model). Aggressively quantize/decimate chroma
   → near-zero task cost. The contest's own preprocessing has pre-declared chroma as low-priority.
2. **Luma 2×2-quad edges are the pose currency.** FastViT's local conv RF means the smallest task-relevant
   luma scale is ~the 2×2 quad at 512×384 (≈256×192 chroma grid). Preserve local luma gradient/edge energy
   in textured static structure (road texture, lane edges, buildings) — that is what produces correct pose.
   Over-smoothing/blocking that kills local luma edges hurts pose more than uniform global shifts.
3. **Out-of-gamut clipping is FREE** (every Y/U/V step `.clamp_(0,255)`) — the renderer can push values past
   the clamp at no score cost (PR95's `sigmoid·255` already exploits the symmetric version: makes the inflate
   clamp a no-op so only `round` bites).
4. **SegNet sees only the LAST frame, unnormalized** (`x[:,-1,...]`). So d_seg fidelity is entirely about
   frame_1's luma boundary geometry; frame_0 matters ONLY for pose. This splits the budget: frame_0 = pose
   carrier (coarse luma + the §2 pose code is enough), frame_1 = pose carrier AND the d_seg boundary frame
   (needs the §3 boundary texture). Asymmetric per-frame byte allocation is a free structural win.

**Caveat (NO-FAKE):** the contest goes through an RGB-uint8 intermediate (`yuv420_to_rgb`→`rgb_to_yuv6`), NOT
openpilot's raw NV12 path — so optimize bytes against the contest's exact two-step transform, not against raw
NV12. The conventions are shared in *structure* (what's cheap/important), which is what transfers; the exact
byte path is the contest's own (verified from `frame_utils.py`).

---

## 5. The pose null-space (physically-impossible perturbations) → free pose budget

**Domain facts (source):** roll≡0 (image not roll-corrected, `calibrationd.py:6`); pitch/yaw within the tight
calib window; the vehicle stays on the road plane; non-slipping highway kinematics. EKF process noise ranks
smoothness: orientation Q=0.001² (smoothest) ≫ velocity Q=0.01² ≫ angular-vel Q=0.085² (`pose_kf.py:48-53`).

**The $0 exploit (→ pose null-space, sister of #47/#140):** the 6-dim pose collapses to ~1-2 free DOF because
4 dims are kinematically null for THIS fixed-mount road video:
- `v_vert ≈ -v_fwd·tan(pitch_mount)` — deterministic function of `v_fwd` (fixed mount).
- `v_lat ≈ v_fwd·tan(sideslip)`, sideslip≈0 on highway → ≈ fixed small offset of `v_fwd`.
- `ω_roll ≈ 0`, `ω_pitch ≈ 0` (suspension only; bounded by the static pitch window).
Set these 4 to their (small, fittable) per-video means/linear-maps; spend the pose code's bytes on `v_fwd`
and `ω_yaw` only. d_pose contribution of the 4 null dims is then only at the noise-floor scale. Combined with
§2, this is why the pose code is ~hundreds of bytes, not 14 KB.

---

## 6. HEVC / source-encode lineage → entropy-coding context

**Domain facts (source):** comma's device road-cam logger encodes HEVC at **5 Mbit/s, GOP 20**
(`loggerd.h:39`); comma2k19 captured at 20 Hz H.264 (arXiv). The contest `0.mkv` is the remuxed comma2k19
segment.

**The (LOW-EV) exploit:** this is context, not a strong lever — the contest archives the decoder+latents
(the INR), not a re-encode of the raw frames, so the source HEVC params don't directly bound our archive. The
one usable fact: the source was a lossy 5 Mbit/s / GOP-20 encode, so the GT frames already carry HEVC
blocking/quant structure at GOP boundaries — our reconstruction need not (and cannot, cheaply) reproduce
sub-quant HEVC artifacts, and the SegNet/PoseNet were effectively trained-and-scored on such artifacts, so
matching the *statistical* texture (not exact HEVC blocks) is sufficient (consistent with §3/§4).

---

## 7. RANKED synthesis — $0 exploits by EV for lowering S (flag rate vs d_seg vs d_pose)

S = 100·d_seg + √(10·d_pose) + 25·B/B₀. Ranked by expected leverage for THIS frozen instance:

| # | Exploit | Axis | Mechanism (frozen-exact) | EV / why | Maps to |
|---|---|---|---|---|---|
| **1** | **comma2k19 GT-pose prior + cheap pose code** | d_pose (+rate) | Download the exact `global_pos/` arrays for `b0c9d2329ad1606b\|2018-07-27\|10`; the 6-dim pose is smooth, ~1-2 DOF, degree-≤4 temporal; code as poly+residual (~hundreds of B) vs 14 KB raw. ∂S/∂d_pose≈85.8 → high-leverage AND we have the answer key. | **HIGHEST** — a free, verified prior on the entire pose signal; collapses the pose search space. | #140 cheap pose code; #155 task-space |
| **2** | **Geometry+semantic margin-map → boundary-only byte spend** | d_seg (+rate) | §1 homography pins static-class layout (horizon row 437, road trapezoid, hood band) + §3 comma10k margin prior (road/sky/hood interior = high-margin = cheap; lane lines + silhouettes + curb = the 1.3% fragile band). Spend boundary-residual bytes only there. | **HIGH** — d_seg is the dominant term (×100); this says *where* the cheap vs expensive pixels are a priori. | #141 margin-map; #153 dominated-rung; #155 |
| **3** | **Chroma decimation + per-frame asymmetric budget** | rate (+safe d_seg/d_pose) | §4: chroma already 1/4-budget (4:2:0+2×2 box) → compress hard, task-safe. SegNet uses last frame only → frame_0 = coarse pose carrier, frame_1 = boundary frame. Out-of-gamut clip is free. | **MED-HIGH** — pure rate savings at ~0 task cost; structural, no training. | #153 dominated-rung |
| **4** | **Pose null-space collapse (4 of 6 dims kinematically fixed)** | d_pose (+rate) | §5: `v_vert,v_lat` = fixed linear maps of `v_fwd`; `ω_roll,ω_pitch`≈0. Code only `v_fwd`+`ω_yaw`; set the rest to per-video means. | **MED** — shrinks the pose code further; folds into #1. | #140; #47 null-space |
| **5** | **Top-band / sky scene-prior class seed** | d_seg | §1+§3: rows < ~437 ≈ undrivable(sky), near-certain → near-zero-byte class structure for the top third. | **MED** — a concrete dominated rung (large area, high margin). | #153; #155 |
| **6** | **Luma-edge-preserving texture target (FastViT local RF)** | d_pose/d_seg | §4: preserve local luma gradient energy in textured static structure (the pose currency); informs the continuous-texture generator's loss to weight local-edge fidelity. | **MED** — directs the (already-running) texture-generation axis. | eval-roundtrip §5.1 texture wall |
| **7** | **Verify argmax-channel→class mapping** | correctness | §3 must-verify: README order ≠ gray-ascending order; confirm empirically which channel = road. | **CORRECTNESS** — wrong mapping silently breaks any class-prior exploit. | gate for #2/#5 |
| **8** | **HEVC source-statistics context** | rate | §6: match statistical texture not exact HEVC blocks; low direct lever (archive holds INR not re-encode). | **LOW** | context |

### The two headline probes (most likely to move S soonest, both $0/local)
- **P_pose (exploit #1+#4):** download comma2k19 `global_pos/` for the exact segment, fit the smooth ~2-DOF
  pose code, run the **real frozen PoseNet** on GT pairs to get the d_pose target, and measure the d_pose a
  reconstruction conditioned on that code incurs. Decisive for a near-free pose slot. (Verification oracle is
  the public answer key — the strongest single domain unlock found.)
- **P_seg (exploit #2+#5):** compute the geometry-derived static-class map (homography) + the SegNet per-pixel
  margin field on the frontier frames; quantify the byte cost of coding ONLY the fragile boundary band (lane
  lines + silhouettes + curb), with the interior/sky/hood seeded by the scene prior. This is the
  eval-roundtrip memo's Lever-D economics, now semantically/geometrically seeded.

### Honest bounding (NO-FAKE)
- None of these is a measured score; pointer UNMOVED 0.19110. Each is a candidate $0 probe; the exact-eval CPU
  row is the only authority.
- The comma2k19 GT is a PRIOR/oracle, not a drop-in — the contest scores the frozen PoseNet on our
  *reconstructed* frames, so we still must make the frames produce the pose (the luma-texture path). The
  unlock is the free, verified shape of the answer, which collapses the search — not a zeroing of d_pose.
- The §1 geometry prior pins STATIC-scene classes; `movable` objects are data-dependent and break it — the
  prior is a certainty floor on road/sky/hood, not a full partition.
- §3 channel-order must be verified before any class-prior exploit is trusted.
- Sister #145 owns the lane-polynomial geometry detail; this doc stays on the broader ecosystem + the
  comma2k19/preprocessing/kinematics unlocks. Cross-check #145 before building the §1/§2 actuators.
