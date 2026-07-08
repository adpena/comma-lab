# DESIGN — Task-space-native pose: Morse-Smale-stratified parallax warp (2026-07-08)

STORES CONSULTED: FEED-poseresearch/FEED-posehard/FEED-sdfresearch (sub015 DAG, 2026-07-08) ·
pose_representation_deepresearch_20260708 (560b16634) · pose_sidecar_reuse_assessment_20260708
(f8f8f4479) · `src/tac/boundary_math/xi_pose_coder.py` (the store-nothing screw = ground homography,
_CAMERA_HEIGHT_M=1.22, `H=K(R−t nᵀ/d)K⁻¹`, n=[0,−cos p,−sin p], d=1.22) · `src/tac/lie/` (se3/so3/
screw_blend/se3_bspline) · `src/tac/boundary_math/warp_real_luma_frame0.py` · `src/tac/depth_motion.py`
· openpilot geometry reconciliation #325/#326/#327 · comma2k19 ego-GT #158 · tasks #194 (per-class
warp / canonicalize-to-ground), #238 (real pose MEASURED), #248 (pose-carrier ladder). Author: main
(operator design riff 2026-07-08). Pointer **0.19110 UNMOVED** — this is a DESIGN + a $0 measurement
plan; nothing here is a score until byte-closed through upstream/evaluate.py.

## 0. THE CLAIM (operator riff, worked through)
There is a pose carrier NATIVE to the SDF-level-set-witness + Morse-Smale framing that is **better than
Quantizr/HNeRV on rate** (~1–2 KB vs their ~270 KB) at equal-or-better d_pose, because **the depth field
the parallax warp needs is already stratified by the exact argmax partition the witness computes for
d_seg.** The store-nothing screw was not wrong — it was a plane-only warp that is EXACT for the ground
plane and sets depth-parallax γ(p)≡0 everywhere else. The fix is surgical and free-per-region.

## 1. WHY store-nothing was RIGHT-for-the-ground-plane, wrong off-plane (measured mechanism)
`xi_pose_coder.py` warps by the ground homography `H(ξ)=K(R−t·nᵀ/d)K⁻¹`, d=1.22 m. Irani–Anandan–
Weinshall: real flow `u(p)=u_planar(H;p)+γ(p)·e`, γ(p)=depth-scaled parallax. A homography reproduces
`u_planar` EXACTLY and sets γ≡0. So the screw is **exact for pixels ON the ground plane** (road, lane
markings — they lie on the plane at d=1.22 geometry) and **exact for pixels at infinity** (sky: γ→0
because parallax vanishes at ∞, only rotation remains). It is WRONG only for OFF-PLANE 3-D structure:
movable vehicles (finite depth ≠ ground) and vertical structure (buildings). PoseNet loses the parallax
cue that separates forward-translation from rotation → d_pose caps ~2.5 (MEASURED, FEED-posehard).
The defect is spatially CONCENTRATED in exactly the regions the partition already isolates.

## 2. THE UNIFICATION (rigorous, not a decorative identity chain)
- The image partition `P(x)=argmax_c(φ_c(x)+b_c)` has separatrices = tie loci (tropical/RAG; CONFIRMED).
- The 3-D scene projects to this partition; depth Z(p) is a function on the image.
- **Depth is smooth WITHIN a class-region and jumps only at OCCLUSION separatrices** — a LABELED SUBSET
  of the RAG edges (near-class abutting far-class). Coplanar boundaries (road↔lane) carry NO depth jump;
  occlusion boundaries (road↔movable, ground↔sky-at-horizon, hood↔road) DO. The witness's tie loci give
  the CANDIDATE depth-edge set for free; which edges are occlusion vs coplanar is a per-edge bit.
- Therefore depth is a PER-MORSE-CELL model, and the cells are FREE (already computed for d_seg). This
  is the task-space-native structure: **one partition serves both d_seg (the labels) and d_pose (the
  per-cell depth-warp).** The SAME twist ξ warps the partition (d_seg) and IS the pose (d_pose) — the
  dual-use screw (CLAUDE.md unified-flow; tac.lie).

## 3. THE PER-REGION WARP LAW (what falls out)
Per Morse cell, warp frame0→frame1 under ego-twist ξ by the depth-appropriate model:
- **Road + Lane (coplanar ground):** `H_ground(ξ)` — CLOSED-FORM from openpilot calibration (d=1.22,
  pitch), ZERO stored depth params. This is the existing `xi_pose_coder` path, now correctly SCOPED to
  ground pixels only. Dominant image area → near-free.
- **Sky / Undrivable-at-∞:** rotation-only `R(ξ)` (translational parallax vanishes). Zero params.
- **MyCar / hood:** identity / fixed near-field (IoU 0.994 static, #139). Zero params.
- **Movable islands (1.56% area):** each island gets ONE stored inverse-depth scalar → parallax
  `(1/Z)·K·t`. openpilot's model outputs lead-vehicle distance; comma2k19 has it (#158). ~a few bytes.
- **Vertical structure (the off-plane part of "undrivable"):** one coarse depth model (e.g. depth ∝
  1/row via horizon, or a low-order per-region field). A handful of scalars.
Full flow: `u(p)=K[ω]×K⁻¹p + (1/Z(p))K·t`, with Z(p) supplied PIECEWISE by the partition. This is
plane+parallax with the parallax structure handed over by the segmentation — the thing openpilot +
our witness already produce.

## 4. BYTE ACCOUNTING (why it beats Quantizr/HNeRV)
Quantizr/HNeRV get low pose as a BYPRODUCT of full-frame RGB reconstruction (~270 KB archive; the pose
blob is a FiLM input welded to a ~214 KB mask + ~56 KB generator — pose_sidecar_reuse_assessment).
This carrier stores: ξ trajectory (store-nothing, derive-H free per #257) + per-edge occlusion bits +
sparse off-plane inverse-depths (movable islands + coarse structure) ≈ **~1–2 KB**. Same twist, same
partition (both already paid for d_seg). If d_pose lands low, this DOMINATES Quantizr on rate by ~100×
at equal pose — the "better than Quantizr" the operator sees. It is Option A (depth-consistent warp)
made TASK-SPACE-NATIVE: depth free-per-region (partition-keyed + openpilot-calibrated) instead of a
generic learned dense depth net.

## 5. RISKS (honest; each has a measurement)
- **R1 — PoseNet appearance vs motion sensitivity (THE make-or-break):** does PoseNet read correct
  motion from a geometrically-correct warp of a TASK-SPACE (non-photometric) f0, or does it need real
  luma appearance? If motion suffices → fully store-nothing-class (warp a cheap f0). If appearance
  needed → store real f0 (Option B bytes) but STILL derive f1 by geometric warp (cheaper than storing
  both). Decided by M3 below.
- **R2 — movable-depth accuracy:** movable is sparse (1.56%); PoseNet is a global 6-DOF regressor. Is
  the off-plane parallax mass large enough to matter, and is a per-island scalar accurate enough?
  Decided by the |t_forward| scatter + M2.
- **R3 — occlusion holes:** the warp exposes disocclusion at depth edges. These coincide with the
  separatrix (Morse-Smale) → localized, inpaintable from the partition. Measure hole area; likely small.

## 6. THE DECISIVE $0 MEASUREMENT LADDER (read-only through frozen CPU-torch PoseNet; no GPU/paid)
- **M1 (control):** ground-homography-only warp of real_f0 → PoseNet d_pose. Should REPRODUCE the ~2.5
  cap (confirms the plane-only defect is the whole story).
- **M2 (the test):** per-region warp of real_f0 — ground `H(ξ)` + sky `R(ξ)` + hood id + movable
  stored-inverse-depth + coarse structure → PoseNet d_pose. **Does it collapse ≪ 2.5?** If yes, the
  parallax stratification is the fix and Option A is viable at ~1–2 KB.
- **M3 (store-nothing viability):** same per-region warp of the TASK-SPACE witness f0 (not real luma) →
  d_pose. Decides R1: store-nothing-class vs store-real-f0 (Option B bytes).
- **Free companion (0 compute):** scatter per-pair d_pose vs |t_forward| from stored ξ + fraction of
  parallax mass that is off-plane — confirms the root cause and bounds R2 before any warp is built.
Ladder is byte-honest: M1/M2/M3 measure d_pose only; the RATE claim (~1–2 KB) is byte-closed separately
through the L13 format before any exact row.

## 7. RELATION TO EXISTING WORK (build-on, don't rebuild)
This COMPLETES #194 (per-class warp / canonicalize-to-ground / dual-quat blend — the per-region law is
its missing depth stratification), consumes #325–327 (calibrated ground homography), #158 (comma2k19
ego-GT + lead depth), tac.lie (ξ), `warp_real_luma_frame0.py` (the warp op M2/M3 extend), and
`depth_motion.py` (existing depth-motion surface — audit for reuse). It is the top rung of the #248
pose-carrier ladder and the concrete form of Option A from FEED-poseresearch. Canonical equation
`morse_smale_stratified_parallax_dpose_v1` COUNCIL-FLAGGED (anchor owed to M2/M3 + byte-close).
