---
title: The openpilot world model as a FREE generative prior for the v2 witness codec — scope + a decisive $0 lane-residual measurement
authority: "[macOS research-signal] / advisory — $0, no GPU, no paid dispatch, no training. score_claim=false; promotable=false. Pointer UNMOVED contest-CPU 0.19110. This is MEANS toward a byte-closed exact row, never an end."
date: 2026-06-29
subagent: openpilot-worldmodel-prior
builds_on:
  - .omx/research/comma_openpilot_crossref_polynomial_geometry_20260619T014433Z.md  (#145 lane-poly/homography)
  - .omx/research/comma_openpilot_domain_tricks_20260619T035417Z.md                 (#156 ecosystem mine + comma2k19 GT unlock)
  - .omx/research/openpilot_comma_repo_wider_exploit_sweep_pose_cereal_hood_20260617T192718Z.md (#158-ish pose/cereal/hood)
  - memory project_contest_source_is_known_comma2k19_rav4_segment_pose_gt_downloadable_20260619
new_measurement: tools/measure_lane_polynomial_shape_floor.py  (oracle polynomial lane-shape FLOOR on cached frozen-SegNet argmax)
---

# openpilot world model as a free generative prior for v2 — scope + decisive $0 lane measurement

**Operator hypothesis under test.** The frozen contest SegNet + PoseNet are openpilot-lineage readouts of
the same latent (scene geometry + ego-trajectory) that openpilot's driving/world model encodes; therefore
openpilot is a FREE generative prior and each v2 learned term collapses to a small residual against
openpilot's own prediction. The v2 binding term (F4) is the **lane-survival d_seg residual**:
sub-0.15 ⟺ trained-through-R lane d_seg ≤ **1.23e-3**; sub-0.19 ⟺ ≤ **1.63e-3**.

**HEADLINE (audited, decisive for the binding term).** I measured the *necessary-condition* floor at $0:
the BEST a polynomial lane carrier can do is to be fit DIRECTLY to the frozen SegNet argmax-lane (an
ORACLE — openpilot's own lanes can only be worse). That oracle floor is **lane d_seg ≈ 0.0021–0.0034**
(per-dash best 0.00214; bridged-continuous deg-4 0.00341) — **1.7×–2.8× ABOVE the 1.23e-3 sub-0.15
threshold and at/above the 1.63e-3 sub-0.19 threshold.** Because this is SHAPE-ONLY (no R) and the R
operator can only ADD error, the floor is a **lower bound on the through-R polynomial-carrier lane d_seg**
→ **a robust NEGATIVE: an openpilot/poly lane skeleton ALONE does NOT collapse the lane-survival residual.**
BUT the same measurement is a strong POSITIVE for openpilot's *real* role: the polynomial nails the lane
**centerline to sub-pixel (0.50–0.81 px median residual)**, recovering **~64%** of the lane d_seg
(0.005885 drop-all-lanes → 0.00214). So openpilot's lane model is a **free positional PRIOR that shrinks
the trained-through-R witness's residual job by ~64%**, not a replacement for it. This INDEPENDENTLY
CONFIRMS the existing repo finding (the ~8-dim nonlinear "lane-orbit manifold" needs a trained generator;
DECISIVE 2026-06-23 + CLAUDE.md class-1 IoU 0.263).

Every number below is `[macOS research-signal]` advisory; the only authority is `upstream/evaluate.py` on
byte-closed bytes. Pointer UNMOVED 0.19110.

---

## TASK 2 (highest value) — the binding-term measurement, in full

### What I measured + how (provenance)
- **Data:** `experiments/results/mlx_fleet_gt_cache/gt_n96.npz`, key `lstars` = the cached FROZEN-SegNet
  argmax over 96 GT last-frames, shape (96, 384, 512), int64, classes 0–4. Class fractions I verified
  match CLAUDE.md canonical order EXACTLY: 0=Road 22.95%, **1=Lane 0.589%**, 2=Undrivable 49.33%
  (top/sky, mean row 95), 3=Movable 1.56%, 4=MyCar 25.57% (bottom, mean row 334). Lane is class **1**.
- **Method (`tools/measure_lane_polynomial_shape_floor.py`):** per frame, lane = (lstars==1); bridge
  dashed strokes (vertical morphological closing) → connected components ≈ lane lines; fit a polynomial
  (deg≤4) per component `col=poly(row)` (or `row=poly(col)` by extent — the openpilot-native form);
  rasterize at the ORACLE per-component width (the width minimizing that component's XOR — best case);
  metric = **symmetric XOR(true lane, poly lane) / (384·512)**, which is EXACTLY the lane contribution
  to d_seg under perfect substitution of the poly lane for the true lane (FP pixels flip TO lane, FN
  pixels flip AWAY). Deterministic; n=96.

### Results
| variant | lane d_seg FLOOR | IoU | centerline resid (px) | vs 1.23e-3 | vs 1.63e-3 |
|---|---|---|---|---|---|
| drop ALL lanes (baseline) | 0.005885 | 0 | — | ✗ | ✗ |
| deg-4 bridged-continuous (oracle width) | 0.003414 | 0.51 | 0.81 | ✗ | ✗ |
| deg-4 **per-dash** (oracle width, no gap-overpaint) | **0.002144** | 0.67 | 0.50 | ✗ | ✗ |
| deg-4 continuous + **oracle PER-ROW width** | 0.00292 (FP 0.00056 / **FN 0.00237**) | — | — | ✗ | ✗ |
| deg sweep 1→4 (bridged) | 0.00366→0.00341 (monotone, saturating) | 0.48→0.51 | 0.89→0.81 | ✗ | ✗ |

### Adversarial audit of the negative (per the not-pessimistic discipline)
I tried three ways to OVERTURN the negative before accepting it:
1. **Dash-gap overpaint?** Bridging a continuous poly over dashed strokes adds false-positives in the
   gaps. Removing the bridge (per-dash fit) DID help (0.00341 → **0.00214**) — but still ✗.
2. **Crude fixed width?** I added an ORACLE *per-row* width search (the theoretical best for a
   centerline + arbitrary width profile). Floor = 0.00292, and it is **FN-DOMINATED (0.00237 missed
   lane pixels vs 0.00056 overpaint)** → the residual is NOT width-tuning; it is that the true lane
   pixels at many rows are a **ragged, multi-pixel, non-centered set** a smooth curve cannot enclose.
3. **Degree too low?** Sweeping deg 1→4 only moves the floor 0.00366→0.00341 (saturating) — higher
   order does not help; the centerline is ALREADY sub-pixel at deg-1. The error is boundary raggedness,
   not curvature.
**Verdict: the negative survives the audit and is ROBUST** — and it is robust in the strong direction
(shape-only floor ≥ 0.00214 > 1.23e-3, and through-R only adds error, so the through-R polynomial-carrier
lane d_seg is necessarily above the sub-0.15 threshold). Confidence: **HIGH** that a polynomial AS THE
LANE CARRIER cannot reach sub-0.15 alone; this does NOT bound a TRAINED generator (which learns the
ragged boundary — the v2 plan, untouched).

### What it means for the operator hypothesis (the refinement)
- **openpilot/polynomial lanes = a FREE POSITIONAL PRIOR, not a residual-collapser.** They place the
  centerline to sub-pixel for free (compress-time) and recover ~64% of the lane d_seg. The remaining
  ~0.00214 (the thin ±1px ragged boundary/width = the "8-dim lane-orbit manifold") is what the
  trained-through-R witness term must still encode. To cross 1.23e-3 from the polynomial base, the
  learned residual must fix only ~0.00091 (≈43% of the polynomial's residual) instead of starting from
  0.005885 — **the prior shrinks the trained job by ~64%.** rule-118: the polynomial coeffs are
  VIDEO-DERIVED → COUNTED (tiny, see Task 4); the rasterizer is GENERIC → FREE in inflate.py.
- This is fully CONSISTENT with — and an independent confirmation of — the v2 plan: store the deterministic
  geometry (lane centerline + pose-warp), and spend the trained INR ONLY on the lane-survival residual
  (project_gr_unified_action..., the v2 grok). My measurement quantifies WHY the trained term is
  irreducible and HOW MUCH openpilot shrinks it.

### The remaining (Q-source) measurement + its blocker
The oracle floor used a poly fit to SegNet's OWN argmax. The untested step is whether **openpilot's own
lane prediction agrees with SegNet's argmax-lane** (so we can use openpilot coeffs without re-fitting).
That needs running supercombo on the contest frames. **Blocker (confirmed this session):** `onnxruntime`
is NOT installed in the venv and `supercombo.onnx` is NOT present locally (no `/workspace/openpilot/...`).
**Cheapest path to run it (≈$0, ~30 min, no GPU — supercombo runs on CPU onnxruntime):**
`uv pip install onnxruntime` → download `supercombo.onnx` (~30 MB) from
`github.com/commaai/openpilot/blob/v0.9.7/selfdrive/modeld/models/supercombo.onnx` (path + version pin
already wired in `src/tac/openpilot_seeding.py`, `SUPERCOMBO_VERSION_PIN="v0.9.7"`) → feed the cached
`gt_f1` frames through `_frames_to_supercombo_yuv` (compress-time only) → slice the lane-line head →
project through the EON homography (Task 3) → XOR vs `lstars==1`. Note: even a PERFECT agreement only
*sources the centerline for free*; it does NOT move the 0.00214 floor (that's a property of the SegNet
lane class, not of the coeff source). So Q-source is a BYTE/EFFORT optimization, not a d_seg lever — the
d_seg verdict above already stands without it.

---

## TASK 1 — world-model output inventory (what openpilot emits; downloadable vs run-required)

openpilot ships TWO model artifacts; both are CONFIRMED-from-source in the prior memos (#145/#156/#158).
Confidence HIGH (source-cited); rule-118 noted per item.

**A. supercombo (the vision+temporal "world model", ~30 MB ONNX).** Single net, two consecutive frames +
recurrent state in. Outputs (`cereal/log.capnp` `ModelDataV2`, `selfdrive/modeld/constants.py`):
- **laneLines** (`NUM_LANE_LINES=4`: outer-L, inner-L, inner-R, outer-R) + `laneLineProbs` + `laneLineStds`
  — as `XYZTData` sampled 3-D points over the quadratic grid `X_IDXS` (0→192 m) / `T_IDXS` (0→10 s),
  `IDX_N=33`.
- **roadEdges** (`NUM_ROAD_EDGES=2`) + `roadEdgeStds`.
- **position / orientation / velocity / orientationRate / acceleration** (the planned ego trajectory).
- **pose** head (the 6-DoF frame-to-frame transform) — sliced `[5755:5761]` in `openpilot_seeding`.
- compact **`DrivingModelData.PolyPath{xCoefficients,yCoefficients,zCoefficients}`** — the SAME path as
  degree-4 polynomial COEFFICIENTS (`POLY_PATH_DEGREE=4` → 5 coeffs/axis), the byte-cheap log form.
- monocular **depth** is NOT a first-class supercombo output (depth is implicit in lane/edge/position z);
  there is no shippable per-pixel depth head. Geometric depth comes from the ground-plane homography
  (Task 3), not the net. Confidence MED-HIGH (absence-of-evidence; the model card lists no depth raster).
- rule-118: supercombo is GENERIC-pretrained (not derived from THIS clip) but is LARGE neural weights →
  if shipped in inflate.py it is AMBIGUOUS/likely COUNTED. **Use it as a COMPRESS-TIME analyzer only**
  (its per-clip OUTPUTS are the counted payload, not the net). See Task 4.

**B. commaVQ (a separate VQ-VAE generative world model, local `workspace/upstream/commavq`).** 128
tokens/frame; next-token prediction = the temporal warp. Reusable as a *temporal prior* for the pose/warp
carrier; it is the OTHER comma challenge (lossless tokens) — do NOT conflate with the lossy scorer task.
Confidence HIGH (file present); rule-118: generic tool, but not needed for the binding term.

**Downloadable vs run-required (CONFIRMED, #156):**
- **DOWNLOADABLE (no model run):** the comma2k19 segment GT — `global_pos/{frame_positions,
  frame_velocities, frame_orientations, frame_times}` (ECEF, 20 Hz) for the exact contest segment
  `b0c9d2329ad1606b|2018-07-27--06-03-57/10` (HF `commaai/comma2k19/compression_challenge/`). This is
  the EGO-MOTION GT (Task 3), not the lane outputs.
- **RUN-REQUIRED:** the supercombo lane-line / road-edge / pose HEADS for THIS clip (comma2k19 does not
  ship supercombo lane outputs; it ships pose GT + CAN/IMU). Run supercombo on the frames (CPU
  onnxruntime, the Task-2 blocker path). Confidence HIGH.

---

## TASK 3 — pose + depth + intrinsics exploits (measured where $0)

**Intrinsics K (CONFIRMED, HIGH).** EON/neo `CameraConfig(1164,874,910.0)` ⇒
K_native = `[[910,0,582],[0,910,437],[0,0,1]]` (`common/transformations/camera.py` `_neo_config`;
byte-identical to contest `frame_utils.py` `camera_size=(1164,874)`, `camera_fl=910`). At the SegNet grid
512×384 (uniform 0.44× resize), K_512 = `[[400.3,0,256],[0,399.5,192],[0,0,1]]` — EXACTLY local
`tac.camera.COMMA_INTRINSICS` / `tac.calibrated_geometry` (CAMERA_FX=400.3, pp=(256,192)). Mount: height
**1.22 m**, roll≡0, pitch/yaw in the tight `calibrationd` window. rule-118: a GENERIC constant → FREE.

**Pose seed (MEASURED this session, HIGH).** I SVD'd the cached `gt_poses` (96×6, the frozen-PoseNet
targets): **99.9% of trajectory energy is in the FIRST singular vector**; dim0 (forward speed) mean 33.0
m/s, std 1.02; the other 5 dims have std ≤ 0.038 and ≈0 mean. dim0 variance is **~700× the next**. ⇒ the
d_pose target is effectively **RANK-1 (forward speed)**, confirming the kinematic null-space (#156/#158).
Consequence: openpilot/comma2k19 ego-motion GT (or even just the rank-1 speed trajectory) seeds the stored
pose for ~free; a low-rank temporal code is strictly Pareto-better than the iid F4 sidecar (474 B), though
F4 is already tiny — so this is a **byte-refinement, not a d_pose lever** (honest). rule-118: per-clip
pose = VIDEO-DERIVED → COUNTED (hundreds of bytes); homography/SE(3) decode = GENERIC → FREE.

**Per-class depth to close F1's +15% warp gap (SCOPED, MED).** The ground-plane homography from K_512 +
height 1.22 m + the calibrated pitch gives EXACT metric depth for the Road plane (row v ↔ distance d:
`v = fy·h/d + cy`), which is the calibration F1 needs for the Road=ground-homography warp; sky/hood need
no depth (rotation-only / identity). comma2k19 + `LiveCalibrationData.extrinsicMatrix`
(view_frame_from_road_frame) pins the pitch to remove the +15% residual. `$0` next step: recompute the F1
per-class warp with the homography-derived depth and re-measure the calibration residual (uses
`tac.calibrated_geometry`; no openpilot run needed). rule-118: GENERIC homography → FREE.

---

## TASK 4 — compliance boundary (the crux) + clean-path byte estimate

Two paths; rule-118 = GENERIC algorithm/tool FREE in inflate.py, VIDEO-DERIVED/LEARNED payload COUNTED in
archive.zip, per-frame table smuggled as "code" FORBIDDEN.

- **(a) openpilot as a COMPRESS-TIME ANALYZER [CLEAN — RECOMMENDED everywhere it suffices].** Run
  supercombo OFFLINE (encoder side, never in inflate). Store only its compact per-clip OUTPUTS — lane/edge
  polynomial coeffs + the pose stream — as the VIDEO-DERIVED COUNTED payload. inflate.py contains only
  GENERIC code: read coeffs → project through the (generic, constant) EON homography → rasterize the lane
  centerline → hand to the trained witness as its conditioning/base. This is squarely legal (the
  rasterizer is the openpilot polynomial+homography ALGORITHM, free; the coeffs are the counted statistic).
- **(b) ship supercombo.onnx INSIDE inflate.py as a "generic tool" [AMBIGUOUS — AVOID].** supercombo is
  ~30 MB of LARGE neural weights. Contest rule 118 says large neural-net weights MUST be in archive.zip
  and ARE counted; the "generic-pretrained-not-derived-from-this-clip" defense is exactly the ambiguous
  ruling we must never depend on. **Recommendation: never need path (b).** Path (a) sources every exploit
  above (lanes, pose, depth) from compact compress-time outputs, so the ambiguous ruling is moot.

**COUNTED-byte estimate for the CLEAN path (a).** From my per-frame fit: ~8.7 components × (deg+1) coeffs
≈ 43 deg-4 coeffs/frame → ~65 B/frame raw, ~30–53 KB across 600 frames if stored iid. **But the lanes are
near-static in the ground frame and slowly-varying over the 60 s clip** (highway), so the iid number is an
over-estimate. Two compressions, both clean: (i) store lane geometry in the GROUND/bird's-eye frame (a few
parallel lane lines as low-order polynomials, ~tens of coeffs TOTAL) + reuse the already-stored pose →
project per-frame for free in inflate.py; or (ii) temporal low-rank coding (the #158 finding) — the per-
frame coeffs are ~rank-few → entropy-code to ~5–15%. Either yields **lane GEOMETRY ≈ 0.5–5 KB for 600
frames** (rate term 25·B/37.5M ≈ 0.0003–0.003 — tiny, consistent with F4's ~3.2 KB rate half). **The
dominant counted cost is the LEARNED ragged-boundary residual (the ~0.00214 the polynomial leaves), which
is NOT estimable here — it needs the trained generator.** Honest: geometry is cheap; the residual is the
real budget.

---

## TASK 5 — deep-math framing (brief, grounded)

This is a textbook **Wyner–Ziv source coding with side information at the decoder** (Wyner & Ziv 1976):
inflate.py's decoder has FREE side information Y = the openpilot world-model prediction (lane centerline,
ego-pose), which is CORRELATED with the source X = the frozen-SegNet argmax partition. WZ says the rate
needed is the CONDITIONAL R_{X|Y}(D), not R_X(D) — we code only the **conditional residual** X−E[X|Y].
My measurement quantifies the correlation precisely: the side-info Y (polynomial centerline) explains ~64%
of the lane d_seg, leaving the conditional residual ~0.00214 (the ~8-dim lane-orbit manifold) to code —
which is exactly the trained-through-R witness term. This is also the cooperative-receiver / Atick–Redlich
+ Tishby IB lens (code the task-latent, not RGB): openpilot's latent IS an estimate of the shared
geometry+ego latent the two frozen scorers read out, so coding openpilot's latent + a residual is the
indirect-RD (coding-for-machines) optimal structure. I do NOT overclaim the dependent-arising framing
beyond this: openpilot's latent is a PRIOR/side-info that shrinks the conditional rate, NOT a zeroing of
the learned term.

---

## SYNTHESIS — top exploit per v2 term + the single most decisive next $0 step

| v2 term | top openpilot exploit | role | confidence |
|---|---|---|---|
| **lane d_seg (binding)** | lane polynomial centerline (compress-time, CLEAN) as WZ side-info → trained residual codes only the ~0.00214 ragged boundary | PRIOR that shrinks the trained job ~64%; NOT a collapser | HIGH (measured) |
| **d_pose** | comma2k19 / rank-1 ego-motion GT seeds the stored pose | byte-refinement of F4's 474 B sidecar; d_pose already solved | HIGH (measured rank-1) |
| **rate** | ground-frame lane geometry + pose reuse → 0.5–5 KB/600 | tiny COUNTED geometry; clean rule-118 | MED (estimate) |
| **F1 warp calib (+15%)** | homography-derived per-class depth (Road plane) | closes the warp calibration gap, FREE | MED (scoped) |

**The single most decisive next $0 step:** NOT more lane scoping (the binding-term verdict is settled —
polynomial alone can't reach sub-0.15; a trained residual on the polynomial base is required). The highest
$0 value is to **wire openpilot's lane centerline as the WITNESS's conditioning/base and measure the
trained-through-R lane d_seg of (polynomial base + small learned residual)** — i.e. does conditioning the
v2 witness on the free centerline let the trained residual reach ≤1.23e-3 with a smaller model than from
scratch. That is a TRAINING measurement (needs the v2 witness loop), so the $0 precursor is: recompute the
F1 per-class warp with homography depth (closes +15%, no model run) AND, if a quick CPU run is affordable,
the Q-source supercombo-vs-SegNet agreement check (install onnxruntime + 30 MB supercombo) to confirm
openpilot can SOURCE the centerline for free (byte saver, not a d_seg lever).

## 6-hook wire-in
1. sensitivity-map: ACTIVE — quantified lane d_seg recovered by the centerline prior (~64%) + residual (~0.00214).
2. Pareto: ACTIVE — openpilot prior is strictly Pareto-better (shrinks trained-residual rate at equal d_seg target).
3. bit-allocator: ACTIVE — geometry (0.5–5 KB clean) vs learned ragged-boundary residual split.
4. cathedral autopilot: N/A — advisory research memo, non-promotable.
5. continual-learning: this memo + #145/#156/#158 + the DECISIVE-8-dim-manifold finding form the
   lane-prior anchor chain; the new tool is a reusable floor probe.
6. probe-disambiguator: `tools/measure_lane_polynomial_shape_floor.py` IS the disambiguator (does a
   polynomial-shape carrier collapse the lane residual — answered NO, robustly).

## Honest bounds (NO-FAKE)
- No score; pointer UNMOVED 0.19110. All numbers `[macOS research-signal]` advisory; only `evaluate.py`
  on byte-closed bytes is authority.
- The floor is SHAPE-ONLY (no R) + ORACLE (fit to target) → a LOWER bound; through-R is harder, so the
  NEGATIVE for a polynomial CARRIER is robust. It does NOT bound a TRAINED generator (the v2 plan).
- n=96 (gt_n96), not the full 600; class fractions match the n96 cache and CLAUDE.md. Re-run on gt_n600
  to firm the number (the tool takes `--gt-cache .../gt_n600.npz`).
- Q-source (openpilot lanes vs SegNet agreement) is UNMEASURED (onnxruntime + supercombo not local); it is
  a byte/effort optimization, not a d_seg lever, so the binding-term verdict stands without it.
