# openpilot cross-surface geometry audit (#325 verify+extend) — 2026-07-06

Read-only, $0/shadow-safe. NO edits, NO PR, NO GPU. Every openpilot claim cites
file+symbol from the local checkout `/Users/adpena/openpilot_research/openpilot`
@ `ee54e82` (commaai/openpilot). Every "impact" is tagged MEASURED (verified
against source + arithmetic) vs INFERRED (needs an n600 witness row to confirm).

## Canonical openpilot values (source of truth, cited)

| Quantity | openpilot value | file:symbol |
|---|---|---|
| road cam (native) | 1164×874, f=910 | `common/transformations/camera.py:_neo_config` |
| principal point | `(width/2, height/2)` = (582, 437) native | `camera.py:CameraConfig.intrinsics` |
| SegNet size | (512, 384) | `common/transformations/model.py:SEGNET_SIZE` |
| camera height | **1.22 m** | `selfdrive/locationd/calibrationd.py:HEIGHT_INIT` |
| calibrated pitch | nominal ~0; limits [-0.0907, 0.17] rad | `calibrationd.py:PITCH_LIMITS` |
| VP↔calib | `yaw=atan(vpx'); pitch=-atan(vpy'·cos yaw); roll=0` | `camera.py:get_calib_from_vp` |
| frames | road[Fwd,Left,Up] · device[Fwd,Right,Down] · view[Right,Down,Fwd] | `camera.py:75-88` |
| road→view extrinsic | `rot_from_euler(rpy)·diag([1,-1,-1])`, trans=[0,height,0] | `camera.py:get_view_frame_from_road_frame` |

Derived (MEASURED arithmetic): scorer `cx=582·512/1164=256.0`, `cy=437·384/874=192.0`,
`fx=910·512/1164=400.27`, `fy=910·384/874=399.82`. Horizon row at pitch p (from cy=192):
`v_h = 192 − fy·tan(p)` → p=0→192, p=0.010→188, p=0.045→174.

---

## Ranked findings (most witness-impactful first)

| # | finding | file:symbol | our value | openpilot / correct | verdict | impact (axis, size) | UPSTREAMABILITY | fix |
|---|---|---|---|---|---|---|---|---|
| 1 | horizon row for lane IPM | `lane_sdf_component.py:_V_HORIZON` (→`analytic_lane_render_band` via import) | `174.0` (=saliency VP y) | geometric cy=**192**; module's OWN note says **188** IPM-optimal (FEED-dj) | CORRECTION+OPTIMIZATION | d_seg via lane band: `below=rows>v_h+1` rasterization cutoff (174 rasterizes 14 extra near-horizon rows = far-field FP, the "whole enemy") + poly-fit forward reparam + dash-period scaling. MEASURED: cutoff/geometry; INFERRED: Δd_seg size (needs n600 174 vs 188 vs 192) | OURS-BETTER-FOR-COMPRESSION-ONLY | measure the 3; adopt the winner as `_V_HORIZON` for IPM, decoupled from saliency VP |
| 2 | camera height (plane distance) | `lane_sdf_component.py:_CAM_H=1.2`; `camera.py:COMMA_EXTRINSICS.height=1.2` | `1.2` | **1.22** (openpilot `HEIGHT_INIT`; matches our own `xi_pose_coder._CAMERA_HEIGHT_M` + `warp_real_luma_frame0.CAMERA_HEIGHT_M`) | CORRECTION (FIX-OURS) | d_seg: 2nd-order (round-trip-invariant band placement; +1.67% forward scale). Rate: lane-frame(1.2)↔ego-frame(1.22) mismatch injects ~1.67% into predictive-coding advection innovation (tiny). MEASURED value; INFERRED negligible d_seg | CONFIRMED-MATCH after fix | unify all three to 1.22 |
| 3 | pitch / horizon encoding split | `camera.py:COMMA_EXTRINSICS.pitch=-0.02`; `VANISHING_POINT=174`; live `warp_real_luma_frame0.eon(pitch=0.0)`; `lane_sdf._V_HORIZON=174` | 4 mutually-inconsistent encodings: pitch −0.02→horizon 184/200, VP 174→p=0.045, v_h 188→p=0.010, warp p=0→horizon 192 | ONE pitch; openpilot nominal calib pitch ≈0 (horizon 192) | CORRECTION (FIX-OURS) | d_seg (lane IPM) + d_pose (homography plane normal n(pitch)). INFERRED | FIX-OURS | pick ONE p consistent with chosen v_h (or estimate per-clip, #7) |
| 4 | scorer focal fy | `camera.py:COMMA_INTRINSICS.fy=399.5` (→`lane_sdf._FY`) | `399.5` | `399.82` (910·384/874); fx 400.3 vs 400.27 also | MINOR CORRECTION | d_seg negligible (round-trip invariant, 0.08%). MEASURED | FIX-OURS (trivial) | nudge fy→399.82; cosmetic |
| 5 | calibrated_geometry K + pose order | `calibrated_geometry.py:CAMERA_PP=(582,437)`,width=512,height=384,fx=910; pose=[ω,t] | native pp with 512×384 label = Frankenstein K; rotation-first order | scorer-res needs pp≈(256,192),f≈400 OR relabel native; `xi_pose_coder` uses translation-first [ρ,ω] | CORRECTION (FIX-OURS), LATENT | OFF live path (grep: no importers). If ever fed 512-res H → wrong pose. Order-mismatch if interop. INFERRED | N/A (off-path; the H→SE(3) Faugeras is a STANDARD-CV-RESULT openpilot doesn't ship/need) | leave dormant or relabel; do NOT wire into live path |
| 6 | dash period search | `lane_sdf_component.py:_fit_dash` period∈[3,25]m | data-driven matched filter | MUTCD US = 3.05 stripe + 9.14 gap = **12.19 m** period | OPTIMIZATION | rate (fix period→store only phase) + fit stability. COUPLED: fitted period absorbs v_h/cam_h scale error, so only valid AFTER #1/#2. INFERRED | OURS-BETTER-FOR-COMPRESSION-ONLY | after fixing IPM geometry, prior-init/narrow the search |
| 7 | per-clip VP/pitch estimation | (absent) hardcoded 174/pitch | none | openpilot ships `camera.py:get_calib_from_vp` (weights-free) | OPTIMIZATION | removes 174/188/192 guesswork; self-calibrates v_h from fitted lane convergence or road/sky boundary; rule-118 FREE. INFERRED | STANDARD-CV (adopt openpilot's own util — not an upstream) | reimplement get_calib_from_vp on fitted VP |
| 8 | SE(3)/frame conventions | `xi_pose_coder.homographies_from_xi` n=[0,−cos p,−sin p], xi=(ρ,ω); `ego_xi_trajectory` yaw about y(down) | — | view y=down → n(p=0)=[0,−1,0]=up ✓; twist=Sola/tac.lie ✓; warp asserts bit-identity | CONFIRMED-CORRECT | none — internally consistent + openpilot-frame-consistent | CONFIRMED-MATCH | DO NOT change |

### Cluster summary
Findings 1–4 are ONE root cause: the lane-IPM geometry (`lane_sdf_component`
`_CAM_H`/`_FY`/`_V_HORIZON`, imported wholesale by `analytic_lane_render_band`)
drifted from the openpilot-correct pose-warp geometry (`warp_real_luma_frame0`:
1.22, f from 910-native, pitch 0). The pose/ξ path is openpilot-correct; the lane
path is the divergent one. Highest-EV single action: reconcile the lane IPM to
ONE openpilot-consistent geometry AND measure v_h∈{174,188,192} at n600 (#1 is
the only finding with plausibly-material d_seg; #2–#4 are consistency/cosmetic).

---

## Upstream contribution candidates (operator review; NO PR)

Honest bar first: **there is essentially no strong UPSTREAM-CANDIDATE here.** The
pieces where "we're better" are better *for our byte/RD objective on a fixed clip*,
not for openpilot's live control needs; the pieces openpilot lacks (H→SE(3)) we
implement *worse* and off-path. Classified:

- **NOT upstreamable — OURS-BETTER-FOR-COMPRESSION-ONLY:** the RD-compact
  ~7-float polynomial+dash lane rep (`lane_sdf_component.LaneLine`) and the
  empirical per-clip v_h tuning (#1). openpilot's planner NEEDS its learned
  33-point lane MDN + per-point stds (supercombo) — a fixed-clip polynomial with
  a hand-tuned horizon is strictly worse for real-time driving. Do not draft.
- **NOT a candidate — STANDARD-CV, and ours is buggy/off-path:** the Faugeras/
  Lustman H→SE(3) decomposition (`calibrated_geometry.homography_to_pose`).
  openpilot deliberately gets pose from locationd/EKF, not planar-H decomposition
  (grep: no H-decomposition anywhere in `common/`,`selfdrive/`). Reimplementing a
  1988 result they don't want ≠ contribution.
- **NOT a candidate — it's already theirs:** per-clip VP calibration (#7) is
  literally `camera.py:get_calib_from_vp`. We'd be *adopting*, not upstreaming.

### REFACTOR-CANDIDATE (their code, behavior-preserving shortening)
- **`common/transformations/camera.py:get_view_frame_from_road_frame` vs
  `get_view_frame_from_calib_frame`** — near-duplicate 3-liners differing only by
  `.dot(np.diag([1,-1,-1]))` (road=calib axes flipped). (b) awkwardness: two
  functions, duplicated `view_frame_from_device_frame.dot(...)` + `hstack` trans.
  (c) sketch: `get_view_frame_from_road_frame(r,p,y,h)` could call the calib one's
  rotation and post-multiply the constant `diag([1,-1,-1])`. (d) **HARD BLOCKER:**
  there is **NO dedicated test file for `camera.py` geometry** (only
  `tests/test_coordinates.py` + `test_orientation.py`; grep confirms camera funcs
  are untested). Cannot assert behavior-preserving without a gating test →
  **this is a blocker, NOT a green light.** (e) LOC saved: ~2. Marginal + ungated
  → do NOT pursue without first adding the conformance test.

No other openpilot geometry function read (intrinsics, `get_calib_from_vp`,
`vp_from_ke`, `normalize/denormalize`, `img_from_device`) is convoluted enough to
warrant a refactor — they are already tight and math-clear. Honest verdict: **no
compelling upstream contribution from this audit.**

---

## do NOT do (looks like a fix, is wrong)
1. **Do NOT import supercombo / any openpilot learned weights** (rule-118: COUNTED
   in archive.zip + forbidden). The lane MDN, calib EKF, etc. are learned artifacts.
2. **Do NOT re-touch** the lane-tangent VP fix in `eased_targets.oriented_width_eased`
   (already fixed to VP=(256,174)).
3. **Do NOT change** the SE(3) twist ordering or `plane_normal` sign (#8): internally
   consistent AND `warp_real_luma_frame0` asserts bit-identity — changing breaks the
   xi_pose_coder strict-parity gate for zero benefit.
4. **Do NOT "fix" fy→399.82 expecting a d_seg move** (#4): round-trip invariant, cosmetic.
5. **Do NOT set v_h=192 blindly** (#1): 192 is the *zero-pitch geometric* horizon; the
   real scene horizon sits lower-row (~188). Measure 174 vs 188 vs 192.
6. **Do NOT hardcode dash period 12.19 m before fixing cam_h/v_h** (#6): the fitted
   period currently absorbs the geometry scale error; hardcoding first would bake in the bias.
7. **Do NOT wire `calibrated_geometry.py` into the live path to "unify"** (#5): it is the
   buggy off-path module; the live ξ/warp path is the openpilot-correct one.
8. **Do NOT draft an openpilot PR from this audit** — no finding is generically better
   for openpilot's own real-time purpose (see Upstream section).

## Provenance
openpilot @ `ee54e82`. Repo files read: `camera.py`,`lane_sdf_component.py`,
`xi_pose_coder.py`,`analytic_lane_render_band.py`(head),`ego_xi_trajectory.py`,
`calibrated_geometry.py`,`lane_mark_pose.py`,`warp_real_luma_frame0.py`(grep).
`[macOS-CPU advisory]` research-signal; no score claim; all Δd_seg/Δd_pose sizes
INFERRED pending n600 through R + frozen CPU-torch scorer.

---

## MEASURED ADDENDUM (task #327, 2026-07-06) — finding #1 (v_horizon) FALSIFIED by n600

The audit's finding #1 (v_h=174 is the "wrong" lane-IPM horizon → far-field FP → adopt
188/192) was INFERRED. The $0 n600 measurement (`tools/measure_lane_ipm_vhorizon_reconciliation.py`,
real GT argmax, band-vs-GT lane, `[macOS-CPU advisory]`) **OVERTURNS it** — 174 is OPTIMAL:

| v_h | recall | precision | FP_far | FP_total | band_err |
|-----|--------|-----------|--------|----------|----------|
| **174** | **0.5475** | **0.6585** | **0.00040** | **0.00198** | **0.00462** |
| 188 | 0.4966 | 0.4663 | 0.00160 | 0.00355 | 0.00649 |
| 192 | 0.4462 | 0.4300 | 0.00128 | 0.00371 | 0.00695 |

Raising the horizon makes recall, precision, far-field FP, AND band-error ALL worse: the
near-horizon rows 174–188 contain REAL lane (the inferred "false positives" were true lane),
and the higher-v_h forward reparam degrades the fit everywhere. **VERDICT: leave `_V_HORIZON=174`
for the lane IPM — no change.** (Physically bounded below by the horizon; sub-174 unpainted-sky
rows would be genuine FP, so 174 ≈ the principled floor.)

**cam_h 1.2 vs 1.22 (finding #2):** ZERO effect on the band (Δband_err = 0.00e+00 — fit↔render
cancels, exactly the predicted 2nd-order). Unifying to 1.22 is a cosmetic cross-module consistency
fix (the ξ/pose path already uses 1.22 and is CONFIRMED-CORRECT), NOT a d_seg lever.

**Reframe of finding #3 (pitch/horizon "split"):** NOT an inconsistency to collapse — 174 (lane-IPM
saliency VP, MEASURED-optimal) and 192 (zero-pitch geometric horizon for the CONFIRMED-CORRECT pose
projection) are TWO CORRECT values for TWO DIFFERENT roles. Do not force them to one.

Net: the openpilot cross-surface audit produced NO score-moving lane change (174 optimal, cam_h zero-
effect, pose path already correct) + NO upstream contribution. A clean measurement-prevents-regression
outcome — the inferred lever was real reasoning, killed by the n600 row before it became a bad edit.
