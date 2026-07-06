# openpilot world-model → witness lane alignment plan (task #325)

**Date:** 2026-07-06 · **Author:** main + mining agent `a7d4cc52b2aabf68d` · **Scope:** $0, shadow-safe
(no GPU, no training, no dispatch). Sister of #145/#138/#156 (openpilot geometric road↔lane prior) and
the analytic-lane-band decomposition (`analytic_lane_band_primary_authority_decomposition_20260701`).

## Headline verdict (honest, up front)

**We have already internalized ~90% of openpilot's usable lane/camera math** — in our own MIT-clean
reimplementation that is in several places *better* (RD-optimal). The web mine CONFIRMS our constants
and pins the one genuinely-missing piece. The actionable gap is **surgical, not an import**:

1. **DONE (this task):** replaced the shape-PCA proxy in
   `src/tac/witness_curriculum/eased_targets.py::oriented_width_eased` with the analytic
   **vanishing-point tangent** `normalize(VP − centroid)` (openpilot's forward-road-converges-to-VP
   geometry). Weights-free, rule-118-clean. Default `tangent_mode="vp"`; `"pca"` kept for the A/B.
2. **Open (per-clip):** verify the vanishing point / pitch for the actual contest clip (the only real
   calibration unknown; the VP-tangent is designed to be robust to it).

We do **NOT** import supercombo, its 33-point lane format, or any learned weights (COUNTED + forbidden
per rule-118; and unnecessary — the geometry is analytic).

## What the mine established (with citations)

### Camera intrinsics — CONFIRMED, we already have them exactly
`openpilot/common/transformations/camera.py` `_neo_config = CameraConfig(1164, 874, 910.0)` (road cam).
Probed `upstream/videos/0.mkv` → **1164×874** = *exactly* `_neo_config`. So contest intrinsics are pinned:
**K_native = [[910,0,582],[0,910,437],[0,0,1]]**. Our `src/tac/camera.py::COMMA_INTRINSICS_NATIVE`
(fx=fy=910, cx=582, cy=437) already matches; scorer `COMMA_INTRINSICS` (fx=400.3, cx=256, cy=192 @
512×384) = native rescaled by 512/1164=0.4399 (910·0.4399=400.3 ✓). openpilot `model.py`
`SEGNET_SIZE=(512,384)`, `medmodel_fl=910` corroborates. **Our camera model is verified correct.**

### Ground-plane homography / road frame — we already have the equivalent
openpilot `get_view_frame_from_road_frame(roll,pitch,yaw,height)` builds the 3×4 extrinsic; project via
`K @ view_from_road @ [x,y,z,1]`. Our equivalents (already implemented):
- `boundary_math/lane_sdf_component.py::{image_to_ground, ground_to_image_row}` — flat-ground IPM
  (`forward = cam_h·fy/(v−v_h)`), openpilot's calibrated-road projection with the `z=0` plane baked in.
- `boundary_math/xi_pose_coder.py::homographies_from_xi` = `K(R − t·nᵀ/d)K⁻¹`, the plane-induced
  homography driven by the ego twist ξ. H derived, never stored (rule-118).
- `calibrated_geometry.py::homography_to_pose` — Faugeras/Lustman H→SE(3) (the *inverse* openpilot omits).

### Lane parametrization — openpilot's is WORSE than ours for our purpose
openpilot lane lines are **not a polynomial**: `lane_lines` shape `(4, 33, 2)` = per line, per fixed
forward distance `X_IDXS[k]=192·(k/32)²`, the net predicts `(y,z)` in calibrated `[Forward,Right,Down]`
(+ MDN stds). **Getting those 33 points requires supercombo (learned weights).** Our `LaneLine`
(`lane_sdf_component.py`) stores `centerline_coeffs` (lateral = poly(forward), deg ≤ 3, ~7 floats/line +
dash period/phase/duty) — a **more compact, RD-optimal, weights-free** representation fit directly from
the SegNet lane mask. Importing openpilot's format would be a regression.

### Ego-kinematics / dash-phase — we already model it
openpilot `modelV2.position` at `T_IDXS` = forward-distance-vs-time = the ego screw. Our
`ego_xi_trajectory.py::{advect_slot_matrix, advect_centerline_coeffs}` + `xi_pose_coder.py` realize
"dash phase advected by ego ξ"; `lane_sdf_component._fit_dash` matched-filters period/phase. LBND3
(ego-comp P-frame band coding) already on record as an honest negative (fit jitter, not ego sweep,
dominates).

## The implemented fix — exact math + measurement

**Replacement direction field** (`oriented_width_eased`, default `tangent_mode="vp"`): a lane marking
lies along the forward-road direction; under perspective, all forward-parallel ground lines converge at
the vanishing point, so **`tangent(u,v) = normalize(VP − (u,v))`**, `VP=(256,174)` (=
`camera.VANISHING_POINT`, openpilot road-cam rescaled). Degenerate fallback to shape-PCA only when the
centroid sits on the VP (direction undefined).

**MEASURED n600 (adversarial, before wiring):** PCA-axis vs VP-direction over 11,805 lane components
(19.7/frame): agree within 15° for **83.4%** (median 3.5°), and the VP-tangent FIXES the noisy **7.5%**
tail (>30°, 3.9% >45°) where a short/near-square-dash shape-PCA picks a spurious axis. **MEASURED n600
(after wiring):** at matched added-area (0.00871 vs 0.00866) VP and PCA give the same n_components
(21.3 vs 21.1) → VP is manifold-preserving to the same degree (the operator's decisive concern) — a
strict robustness upgrade on the tail, no regression.

*First-order refinement (documented, not yet wired):* for a curved lane, step forward in the ground
frame from `image_to_ground(u,v)`, reproject with `ground_to_image_row`, and use that image tangent
(follows curvature exactly, still weights-free). Deferred because it introduces the flat-ground +
pitch/height calibration dependence the zeroth-order VP-direction is robust to; gated on per-clip VP
verification.

## Transfers cleanly / needs calibration / not worth it

- **Transfers cleanly (already ours):** native + scorer intrinsics (CONFIRMED by 1164×874); flat-ground
  IPM; plane-induced homography from ξ; lane-as-polynomial; ego advection of the band. No import needed.
- **Needs contest-clip calibration (the honest unknown):** this clip's pitch/height/roll. Mitigations:
  the VP-tangent depends only on VP *direction* + is scale-invariant → robust to height/mild-pitch error
  (lowest risk). To harden: estimate VP per-clip from lane convergence or the SegNet road/sky horizon,
  then recover pitch/yaw with openpilot's closed form `get_calib_from_vp` (`yaw=atan(vp_x)`,
  `pitch=−atan(vp_y·cos yaw)`, ~10 LOC, MIT, weights-free). Height is least-constrained; only needed for
  the dash-phase↔ego-distance cross-check, not the tangent.
- **NOT worth importing:** supercombo net + weights (learned → COUNTED + forbidden, and unnecessary); the
  33-point `X_IDXS` format (our polynomial is more compact); MDN/desire/lead/planner heads;
  `coordinates.py` (ECEF/geodetic — irrelevant); live `calibrationd` (we do one-shot offline VP instead).

## License / rule-118 boundary

- **openpilot is MIT.** `common/transformations/` is pure math, freely reusable. Our reimplementations are
  independent, not verbatim copies.
- **rule-118:** the openpilot ALGORITHM (euler→rot, IPM `forward=h·fy/(v−v_h)`, plane homography,
  `get_calib_from_vp`) is a **generic deterministic prior → FREE in inflate.py** (no bytes). openpilot
  LEARNED WEIGHTS (supercombo) are a large video-derived artifact → **COUNTED + forbidden** in archive.zip
  and must not be smuggled into inflate.py "code." The whole point of the analytic tangent is that it needs
  **zero** learned weights.

## Cross-refs
Files cited — openpilot: `common/transformations/{camera,model,orientation}.py`,
`selfdrive/modeld/{constants,parse_model_outputs,fill_model_msg}.py`. Ours:
`src/tac/witness_curriculum/eased_targets.py`, `src/tac/boundary_math/{lane_sdf_component,
analytic_lane_render_band,xi_pose_coder,ego_xi_trajectory,lane_headstart}.py`,
`src/tac/{camera,calibrated_geometry,lane_mark_pose}.py`. Memories:
`ladder_costate_optimal_difficulty_gradient_lane_movable_20260706`,
`analytic_lane_band_primary_authority_decomposition_20260701`,
`project_openpilot_unified_physical_prior_both_scored_axes_20260702`.
