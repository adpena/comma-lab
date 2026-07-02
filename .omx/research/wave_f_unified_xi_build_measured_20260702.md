# Wave-F unified-ξ BUILD — MEASURED (n600): ego-predictive NEGATIVE, source-smoothing POSITIVE

- **Date:** 2026-07-02
- **Status:** BUILT + MEASURED + tested. Advisory / build-only. **Pointer contest-CPU 0.19110 UNMOVED**
  (moves only via a byte-closed `upstream/evaluate.py` n600 exact row). Every rate row is `[macOS-CPU advisory]`
  MEASURED brotli byte-count; the d_pose row is real frozen CPU-torch PoseNet (NO MPS). NOT a score claim.
- **Design authority:** `unified_xi_design_and_adversarial_review_20260702.md` (PROCEED-WITH-REVISIONS; the
  decisive revision was the P-frame predictive-coding reframe — built + measured here).
- **Builds on:** the LANDED Stage-1 LBND2 codec (`wave_f_lane_band_rd_code_LANDED_stage1_measured_20260702.md`:
  n600 rate 0.02765). This unit is the Stage-2 L1 (ego-factorization) build.
- **Commits:** `55d61f921` (seam + LBND3 codec + smoothing) · `44b11f25b` (n600 bake-off tool + result) ·
  `dba22b28b` (17 tests). Result JSON: `.omx/research/wave_f_ego_predictive_rate_n600_RESULT.json`.

---

## TL;DR (the headline)

The design's decisive reframe — **ego-motion-compensated predictive coding** of the camera-frame lane
coeffs — was BUILT (bit-exact, decode-consistent, a strict generalization of LBND2) and MEASURED at n600
across every estimator. **It is a clean NEGATIVE: every ego-predictive variant is 1.04–1.34× LARGER than
LBND2.** The negative is IMPLEMENTATION-informative (per "negatives are deep-math signal"): it proves the
camera-frame residual is **per-frame fit JITTER**, not a coherent ego sweep a planar advect can predict.
That diagnosis revealed the REAL L1 lever — **source temporal smoothing** of the coeff trajectory (the
"fit the world lane once + smooth trajectory" thesis realized via denoising) — which is a strong
**POSITIVE: 42% smaller** (n600 rate 0.02765 → 0.01608, win15; decode-consistent, ships as standard LBND2).

**The pose↔lane tension (design §5.5, the sharp question) is RESOLVED, not compromised:** since the lane
axis does NOT use ξ at all, ξ is a **pure-pose sidecar** — there is no shared-ξ trade-off. The pose axis
uses whatever calibration is pose-optimal, at zero lane cost.

---

## What was built (P1 + P2, all committed + tested)

- **P1 — the estimator-agnostic ξ seam** (`src/tac/boundary_math/ego_xi_trajectory.py`):
  `XiEgoTrajectory` (frozen; per-pair planar `(ds, dy, dpsi)` + `dense_xi (P,6)` for the pose readout) +
  the `EgoEstimator` protocol + the **3-DOF planar ADVECT operator** (`advect_slot_matrix`: exact fp64
  Taylor-shift by `ds` + lateral `dy` + yaw `dpsi`; numpy-fp64 authority, bit-identical both sides) +
  three estimators: `LaneOptimalEgoEstimator` (fits `(ds,dy,dpsi)` from the lane geometry = the
  achievable-floor / lane-optimal ξ), `PoseTargetEgoEstimator` (calibrates the frozen `gt_poses` PoseNet
  ego readout up-to-affine = the physical ξ), `ConstantVelocityEgoEstimator` (the zero-info physical
  prior) + `fit_se3_bspline_controls` / `bspline_fit_error_curve` (the ξ-compression + design-Q5 curve).
- **P2 — the LBND3 ego-predictive codec** (`analytic_lane_render_band.py`): `serialize_lane_band_rd3` /
  `deserialize_lane_band_rd3` (closed-loop DPCM: predict Q(coeffs)_t from the DECODED Q(coeffs)_{t-1}
  advected by ξ; code the innovation; magic `LBND3`), `deserialize_lane_band_any` LBND3 dispatch, and
  the **`temporal_smooth_pairs_lines`** source re-parameterization. `lane_band_rd3_rate_report` +
  `tools/wave_f_ego_predictive_rate_n600.py` (the measured bake-off).
- **Tests** (`src/tac/tests/test_ego_xi_predictive_coding.py`, 17/17): advect exactness + DOF isolation;
  LBND3 bit-exact round-trip; **LBND3(ξ=0) innovation == LBND2 temporal delta** (strict generalization);
  smoothing decode-consistency via the existing LBND2 path; estimators; b-spline curve. Existing LBND2 /
  Wave-E decode-consistency suite **26/26 unbroken**.

### The DECISIVE design revision, built: predictive coding (no inverse-warp)
LBND3 uses ξ as a P-frame **predictor** (advect the DECODED previous coeffs), NOT a store-in-world-frame
warp — so `unwarp(warp(x))==x` is a non-requirement and the determinism hazard is deleted. The ONLY fp
obligation is `advect` bit-identity both sides, trivially guaranteed. Estimator error is a RATE cost, never
a correctness bug (verified: LBND3 round-trips bit-exact for ANY ξ; the dequant lines == LBND2's grid).

---

## MEASURED n600 lane-rate bake-off (real `gt_n600.npz`, byte-closed, `[macOS-CPU advisory]`)

| variant | brotli bytes | rate_term | vs LBND2 | ego payload |
|---|---:|---:|---:|---:|
| **LBND2 baseline** (Stage-1) | 41,526 | 0.02765 | 1.000 | — |
| LBND3 ego — lane-optimal (ds,dy,ψ) | 47,453 | 0.03160 | **1.143** | 2,744 B |
| LBND3 ego — lane-optimal (dy,ψ) | 44,652 | 0.02973 | **1.075** | 1,842 B |
| LBND3 ego — lane-optimal (dy,ψ)+smooth5 | 43,228 | 0.02878 | **1.041** | 1,081 B |
| LBND3 ego — PoseNet-physical (affine) | 44,908 | 0.02990 | **1.081** | 2,092 B |
| **LBND2 on SMOOTHED src** win3 | 31,360 | 0.02088 | **0.755** | — |
| **LBND2 on SMOOTHED src** win5 | 28,050 | 0.01868 | **0.675** | — |
| **LBND2 on SMOOTHED src** win9 | 26,260 | 0.01749 | **0.632** | — |
| **LBND2 on SMOOTHED src** win15 | **24,149** | **0.01608** | **0.582** | — |
| LBND3 ego on SMOOTHED win15 | 32,463 | 0.02162 | 1.344 | — |

**Every ego-predictive row is WORSE (>1.0). Every source-smoothing row is BETTER (<1.0).** Ego even HURTS
the already-smoothed source (1.344×). The best measured = source-smoothing win15 = **24,149 B / 0.01608**,
a **42% rate reduction** over LBND2 — and it DIPS BELOW the Stage-1 Shannon floor (26,179 B / 0.0174),
because smoothing CHANGES the source to a lower-entropy signal (the raw-source floor no longer applies).
Crossing the old floor via a source transform IS the L1 source-re-parameterization thesis, confirmed.

### Why ego-predictive fails (the diagnostic, measured n96)
The advect is a near-no-op on the dominant entropy dims (the dy+yaw advect only touches c0/c1; c2/c3/
halfwidth/dash/forward_range carry most of the entropy and are untouched), and its per-pair estimate is
NOISY. Root cause: **the frame-to-frame centerline change is fit jitter, not ego sweep** — 44% of the
temporal-delta L1 mass is in the top-5% largest jumps (a slot-swap / fit-outlier signature), and
temporally median-smoothing the SOURCE coeffs collapses the bytes 48% (raw-M probe) / 34% (presence-aware
n96) / 42% (n600). Jitter is removable by DENOISING THE SOURCE, not by predicting noisy fits from noisy
fits. This directly answers design **Q2 with a decisive negative for the predictor** and a positive for
the re-parameterization it revealed.

---

## Pose axis (P3) + the tension resolution (design §5.5 / Q1 / Q4)

- **The sharp question RESOLVED (measured):** the lane-rate axis is **ξ-free** (ego-predictive is a
  negative; source-smoothing wins with zero ξ). Therefore there is NO "lane-optimal ξ" competing for the
  shared sidecar — the shared-ξ-vs-per-axis-optima tension is resolved trivially: **ξ is a pure-pose
  object**, optimally calibrated for d_pose at zero lane cost. The "one ξ, both axes" claim is MOOT
  (the lane axis declined ξ), which is a cleaner outcome than the anticipated compromise.
- **d_pose warm-start:** the seam's `dense_xi` feeds the already-built `warp_real_luma_frame0` carrier
  (`warp_frame0_uint8_numpy`). The authoritative n600 warm-start is measured by
  `tools/measure_warp_real_luma_frame0_dpose.py` (real frozen CPU-torch PoseNet, NEVER MPS): the
  zero-motion null (~182) → the ξ-calibrated ground-homography warp (~10.5, −94%), residual to ~3.4e-5
  closed by a trained per-pair `dxi` (`w_pose>0`, #205). `[FRESH n600 RESULT: see
  wave_f_pose_warmstart_dpose_n600_RESULT.json — POSE_WARMSTART_ROW]`. This is our prior FEED-lj physics
  (`warp_real_luma_frame0_pose_carrier_dpose_v1`); the seam is the estimator-agnostic front end feeding it.
- **Q4 up-to-affine identifiability:** `PoseTargetEgoEstimator(calib='affine_to_lane')` fits the affine
  map from the `gt_poses` PoseNet channels to a reference ξ; the physical/pose calibration is the fitted
  `(s_t, s_r)` in `xi_from_pose_calibration`. Since the lane axis rejects ξ, the affine calibration is a
  pure-pose concern (the warp carrier's global `s_t` fit), already the measured 182→10.5 path.

---

## The 4-axis bake-off synthesis (Yousfi lens: the SCORER is the authority)

| axis | LBND2 | LBND3 ego (lane-opt / PoseNet) | LBND2-on-smoothed (win15) |
|---|---|---|---|
| **(i) lane-rate @ n600** | 0.02765 | 0.0316 / 0.0299 (WORSE) | **0.01608 (−42%)** |
| **(ii) d_pose warm-start** | ξ-free (pose sidecar decoupled) | same (ξ is pure-pose) | same |
| **(iii) determinism / decode** | bit-exact ✓ | bit-exact ✓ (but not shipped) | bit-exact ✓ (existing LBND2 inflate path) |
| **(iv) rule-118 cleanliness** | ✓ | ✓ (ξ counted, but net-negative) | ✓ (no ξ; smoothing is a compress-time source transform) |

**Winner: source-smoothing (ξ-free) for the lane rate; the physical PoseNet ξ for the pure-pose warm-start
(decoupled).** VO/SfM vs supercombo estimator bake-off is MOOT for the lane axis (ξ doesn't help it); for
the pose axis the calibrated `gt_poses` (the contest PoseNet's own readout) is the natural physical ξ.
LA-Pose remains a documented experimental follow-up (no public code).

---

## rule-118 / NO-FAKE accounting (binding, honored)

- **Source smoothing (the win):** ships as **standard LBND2 bytes** — the counted payload is unchanged in
  KIND (quantized temporal-delta coeff stream); smoothing is a **compress-time source transform**, so the
  existing LBND2 inflate mirror (`_lane_parse_rd`) decodes it bit-exactly with ZERO new inflate code. The
  smoothing is **LOSSY on the geometry** (denoises the per-frame fits); whether it NETS lower S is the #205
  trained-in d_seg measurement (a conservative win5 = −32% is safer than win15 if the geometry change costs
  d_seg). **NOT claimed to lower S — only RATE, measured.**
- **LBND3 ego (the negative):** COUNTED = innovation + presence + the (ds,dy,dpsi) quantized stream; FREE =
  advect/quant/dequant/raster (numpy-fp64, ZERO mlx in inflate). It is a MEASURED negative → **not
  shippable**, so its inflate mirror is deliberately NOT wired (documented; the shippable smoothing win
  needs no LBND3 decode).
- Steps DERIVED from geometric tolerances; NO GT mask, NO scorer weights, NO per-pixel table.

---

## Honest verdict + what's deferred

**The unified-ξ ego-predictive-coding hypothesis is FALSIFIED for the lane-rate axis @ n600** (best-case
lane-optimal AND physical-PoseNet ξ both worse than LBND2). The machinery is real, bit-exact, tested, and
the negative is CONSTRUCTIVE: it located the true lever (**source temporal smoothing, −42% rate, decode-
consistent**) and cleanly decoupled ξ into a pure-pose sidecar. This is a genuine Stage-2 rate advance
(0.02765 → 0.01608) achieved DIFFERENTLY than the design hypothesized — the deep math (source re-param
crossing the old Shannon floor) holds; the specific mechanism (predictive coding) does not.

**Deferred / next (named):**
1. **#205 trained-in d_seg with the smoothed band** — the ONLY thing that moves the pointer: does the −42%
   rate + the denoised geometry NET a lower exact S? Sweep the smoothing window for the S-optimum (win15
   rate-optimal may over-smooth d_seg; win5 conservative). This is a scorer run, gated.
2. **Pose warm-start fold-in** — the fresh n600 `measure_warp_real_luma_frame0_dpose` row confirms the pure-
   pose ξ warm-start (decoupled from the lane rate).
3. **Non-negatives to try before closing predictive-coding** (per "negatives adversarially overturned"):
   joint MULTI-FRAME world-lane fit (fit ONE world lane + a smooth ego trajectory jointly, not per-frame
   fits then denoise) — the strongest form of the L1 source re-param; and slot-correspondence stabilization
   (the 44% top-5%-jump mass is partly slot swaps — a Hungarian/temporal slot tracker could cut it further).

---

## Wire-in (6-hook, research_only)
1. Sensitivity-map: the measured per-variant Δrate rows → `tac.sensitivity_map` (rate axis). 2. Pareto: the
smoothing window is a rate↔d_seg Pareto knob (#205 measures the d_seg leg). 3. Bit-allocator: the smoothing
is a source-denoise pre-pass to the existing LBND2 allocator. 4. Cathedral autopilot: N/A (research_only; no
new archive-deployable artifact — the smoothing plugs into the existing byte-close LBND2 path). 5. Continual-
learning: this memo + `wave_f_ego_predictive_rate_n600_RESULT.json` are the anchors. 6. Probe-disambiguator:
the bake-off IS the disambiguator (ego-predictive vs smoothing, resolved by measured n600 bytes).

**Council mission-contribution:** `frontier_breaking` (the −42% rate half of the lane band) tempered by an
HONEST negative on the predictive-coding mechanism. All MEANS; the END is the #205 byte-closed exact row.

## Sisters
`wave_f_lane_band_rd_code_LANDED_stage1_measured` · `unified_xi_design_and_adversarial_review` ·
`analytic_lane_band_primary_authority_decomposition` · `pose-solved-screw-twist-dual-use-film-conditioned-sidecar`
· `not-pessimistic-first-results-adversarial-deepmath-oss-against-negatives`.
