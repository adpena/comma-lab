# Unified-ξ — estimator-agnostic ego-trajectory: DESIGN + ADVERSARIAL REVIEW (Yousfi lens)

- **Date:** 2026-07-02
- **Status:** DESIGN + ADVERSARIAL REVIEW (advisory / design-only). **NO code edits, NO build, NO GPU** in this unit.
- **Axis discipline:** every quantitative row below is `[prediction]` / `[research-signal]` / `[derived]` — **NO score claim**. MPS-never. Pointer **contest-CPU 0.19110, UNMOVED** (moves only via a byte-closed `upstream/evaluate.py` n600 exact row). `research_only: true`.
- **Feeds:** the Wave-F Stage-2 L1 build (lane ego-factorization) + #205 (witness θ* pose config) + #191 (openpilot-seeded witness) + #206/#221 (pose residual). This memo **gates the build** (parent reviews → surfaces to operator for GO).
- **Verdict up front:** **PROCEED-WITH-REVISIONS.** The unified-ξ is the right object and the estimator-agnostic seam is sound; the single decisive revision is to **reframe L1 from "store-in-world-frame" (needs an exact-invertible warp) to "ego-motion-compensated PREDICTIVE coding" (needs only a bit-identical predictor)** — this removes the build-design's flagged "main build risk" and is textbook P-frame motion compensation. Four smaller revisions + five ranked open questions below.
- **Provenance (measured, do NOT re-derive):** `wave_f_optimal_lane_band_rd_code_design_20260702.md` (L1–L5), `wave_f_lane_band_rd_code_LANDED_stage1_measured_20260702.md` (Stage-1: rate 0.1041→0.02765, Shannon floor 0.0174, residual = ~11KB ego-sweep + ~15KB fit-jitter, **L1 = source re-parameterization**), `pose_in_training_levers_survey_20260702.md` (H1 openpilot ξ-init co-#1; warp-real-luma −94%; VO/SfM + supercombo + LA-Pose estimators; LDM Thm 1 up-to-affine), `wave_f_lane_band_build_design_20260702.md` (R2 primitive inventory + the two headline findings). In-tree APIs grepped, not invented: `tac.lie.se3_bspline`, `tac.boundary_math.warp_real_luma_frame0`, `tac.boundary_math.lane_sdf_component`.

---

## 1. The estimator-agnostic architecture

### 1.1 The ONE seam

There is exactly **one** shared object — the ego trajectory `ξ_ego(t)` — and exactly **one** interface between "how we estimate it" (swappable) and "what consumes it" (fixed). The seam is a typed, frozen descriptor:

```
XiEgoTrajectory  (frozen dataclass; the counted-tiny sufficient statistic)
  control_poses : np.ndarray  (M, 4, 4)  fp64 SE(3) matrices  # the B-spline knots — THE payload
  n_pairs       : int                    # 600
  n_segments    : int  = M - 3            # cubic B-spline segment count
  knot_time_of_pair : np.ndarray (n_pairs,)  # pair index -> spline time t in [0, M-3]
  calib         : CommaCalib              # fx=fy=910, cx=582, cy=437, native (1164,874), h=1.22 (FIXED, not stored)
  estimator_id  : str                     # "vo_sfm_homography" | "supercombo_posehead" | "raft_ipm" | "la_pose_invdyn"
  provenance    : dict                    # git sha, seed, estimator version, upstream sha  (compress-time only)
```

- **Producer side (compress-time, FREE):** an `EgoEstimator` protocol — `estimate(frames_or_video) -> dense_xi : (n_pairs, 6)` (per-pair relative twist, `tac.lie` convention `ξ=(ρ,ω)` translation-first). Any estimator implements this ONE method. The dense per-pair trajectory is then **fit** to `M` SE(3) control poses (`fit_se3_bspline(dense_xi, M) -> control_poses`), which IS the compression. The B-spline `tac.lie.se3_bspline` already exists (cumulative uniform-cubic, Sommer 2020, MLX + numpy oracle, parity-gated).
- **Consumer side (fixed, two readouts of the same field):**
  1. **Pose warm-start (d_pose):** `xi_at(pair) = se3_bspline_eval_numpy(control_poses, knot_time_of_pair[pair])` → `warp_frame0_uint8_numpy(frame0, xi, GroundHomographyGeom.eon())` (already in `warp_real_luma_frame0.py`; `H = K(R − t·nᵀ/d)K⁻¹`, measured d_pose 182→10.53 = −94%). frame0 is seg-free (`modules.py:108`) → warping it CANNOT disturb d_seg.
  2. **Lane ego-factorization (d_seg-lane RATE):** the SAME `xi_at(pair)` advects the metric ground-frame lane coeffs (`lane_sdf_component`: `lateral(m)=polyval(centerline, forward(m))` is already ground-relative) → used as the **predictor** for the Stage-2 L1 residual coder (§2 reframe).

### 1.2 Knot count M vs ξ-DOF (MEASURED, not asserted)

- Payload = `M × 6` floats (each 4×4 SE(3) control pose has 6 intrinsic DOF; store as the 6-vec `log_se3` per knot, not the 16 matrix entries). For a ~60 s / 1200-frame / 600-pair drive, ego-motion is smooth: near-constant forward speed (dim-0 = 99.8% of PoseNet signal), slow yaw at curves.
- **M is CHOSEN by a MEASURED fit-error-vs-M curve, per the "calibrate parametrization" discipline** — NOT by a guessed number. The bracket to sweep offline ($0, CPU): `M ∈ {8, 12, 16, 24, 32, 48}`. For each M, measure (a) the ξ-fit residual through R on d_pose (does the B-spline reproduce the dense warp's −94%?) and (b) the lane-innovation entropy after ego-compensation. Pick the smallest M whose fit residual is below the argmax-band tolerance (~1 px @ 384, ~2.27 px per ancestor findings) AND whose incremental rate is negligible. Order-of-magnitude `[prediction]`: `M ~ 16–48` → `96–288` fp values → **hundreds of bytes to ~1–2 KB counted** (dominated by the range-coded innovation, not the control poses). This is the `[prediction]`; the MEASURED fit curve is the authority.

### 1.3 rule-118 accounting (binding; honored)

| Object | rule-118 class | Where |
|---|---|---|
| The estimator (supercombo / VO-net / RAFT / LA-Pose encoder) | **FREE** (compress-time external code/tool; "compile the generator") | offline, never ships |
| ξ = the B-spline control poses (`M×6` floats) | **COUNTED** (video-derived sufficient statistic) | `archive.zip`, ~hundreds of B–2 KB, ONE stream, dual-attributed to pose+lane |
| The ego-compensated lane innovation stream | **COUNTED** | `archive.zip` (Stage-2 L1 residual) |
| ξ evaluation + warp + advection at decode | **FREE** (generic numpy oracle) | `tac.lie._se3_numpy` + `se3_bspline_eval_numpy` + `homography_from_xi_numpy` in inflate.py |
| Estimator weights (supercombo / VO-net) | **NEVER SHIP** (large video-derived artifact, counted AND unnecessary) | — |
| Euler / quaternion form of ξ | **FORBIDDEN** (C2 guard, Zhou 2019 discontinuity) — keep the continuous Lie/matrix form | — |

**The clean boundary:** a FREE offline estimator produces a COUNTED tiny ξ; a FREE generic numpy decoder expands ξ into the pose warp + the lane predictor. This is exactly the "compile the generator, count only the video-derived payload" discipline. **NO scorer weights, NO GT-argmax table, NO estimator net ships.**

---

## 2. Bit-exact warp/unwarp determinism (the deferred hard part — the DECISIVE design move)

### 2.1 The hazard the build-design memo flagged

R2 (`wave_f_lane_band_build_design_20260702.md` §3b / §4-L1) named the "main build risk": if L1 **stores lane geometry in a fixed world frame**, decode must **`_unwarp` (world→camera)**, and the gate `unwarp(warp(x)) == x` must hold **bit-exact on the quantized coefficient grid**. Warping a polynomial curve under an SE(2) planar rigid motion is a coefficient re-composition (re-expand `polyval` under `(Δs, Δy, Δψ)`), which in fp64 is NOT exactly invertible (catastrophic cancellation + polynomial recompose) — a real determinism fragility.

### 2.2 The reframe that REMOVES the hazard (predictive coding, not lossy transform)

**Do NOT store in the world frame. Use ξ as a PREDICTOR for ego-motion-compensated coding of the CAMERA-frame coeffs — the exact P-frame construction of every video codec.**

```
Encode (compress):
  pred_t = advect(camera_coeffs_{t-1}, xi_at(t))        # ξ-advected previous pair's coeffs (numpy-fp64)
  innov_t = camera_coeffs_t - pred_t                     # small residual (near-zero when ego is well-estimated)
  quantize(innov_t) -> range-code                        # the COUNTED stream (Stage-1 codec, upgraded predictor)

Decode (inflate):
  pred_t = advect(decoded_camera_coeffs_{t-1}, xi_at(t)) # SAME numpy-fp64 function, SAME inputs
  camera_coeffs_t = pred_t + dequantize(innov_t)         # exact reconstruct of the quantized camera coeffs
```

Why this is bit-exact and hazard-free:
- **No inverse-warp is ever required.** There is no `world→camera` step, so `unwarp(warp(x)) == x` is a non-requirement. The only determinism obligation is **`advect` is bit-identical on both sides** — trivially guaranteed because both sides call the SAME numpy-fp64 `advect` on the SAME inputs (the decoded/quantized previous coeffs + the bit-identical ξ). This is the standard "the predictor operates on already-reconstructed samples" rule that makes P-frames deterministic.
- **ξ is bit-identical both sides by construction:** ξ = the stored control poses, evaluated by `se3_bspline_eval_numpy` + `_se3_numpy` (fp64; the `warp_real_luma` docstring already certifies this numpy-fp64 path is "bit-identical across hosts"). MLX/fp32 is TRAINING-ONLY; decode is numpy-fp64 == compress-side numpy-fp64.
- **The only fp path is `advect`.** Fix its operation order once in a single shared function used by both compress and decode (rounding order pinned). The innovation stream is INTEGER (range-coder over quantized ints) — no fp there.
- **End-to-end gate unchanged:** the Wave-E `max_abs_uint8_diff == 0` (numpy-fp32 authority, NEVER MPS) over the full n600 decode→raster→composite→R still stands; the reframe only changes what the coder predicts, not the raster/composite (those stay the FREE generic algorithm).

### 2.3 Residual determinism risks (post-reframe) + the tests that pin them

| Risk | Post-reframe status | Test |
|---|---|---|
| `unwarp(warp(x))==x` fp fragility | **ELIMINATED** (no unwarp) | n/a |
| `advect` fp order drift compress-vs-decode | contained (one shared fn) | `advect_compress(q,ξ) == advect_decode(q,ξ)` bit-for-bit on the quantized grid |
| ξ-eval host drift | contained (numpy-fp64 oracle, host-portable) | `se3_bspline_eval_numpy` cross-host parity (already the warp_real_luma authority) |
| fp32 MLX training vs fp64 decode | irrelevant (decode is fp64; MLX is train-only, parity ≥0.9997) | existing MLX↔numpy parity gate |
| Round-trip through quantization | the innovation is quantized+coded; `pred` uses the DECODED (quantized) previous coeffs (closed loop) | Stage-1 `pose_carrier_real_bytes` roundtrip assert + Wave-E `max_abs_uint8_diff==0` |

**Fixed-point vs fp64:** keep fp64 for the ξ geometry (the numpy oracle, host-portable) and the `advect` polynomial; keep INTEGER for the innovation quantization + range-coder. Do NOT introduce fixed-point for the geometry — the numpy-fp64 path is already the certified bit-identical-across-hosts authority (`warp_real_luma_frame0.warp_frame0_native_numpy` docstring). The closed-loop predictor (predict from DECODED, not original, previous coeffs) is what makes the quantization interaction exact.

---

## 3. The Yousfi-lens calibration protocol ("compare + test against deep math")

**Yousfi's principle applied:** *the SCORER is the authority; the comparison metric is the MEASURED EFFECT THROUGH R, never the estimator's intrinsic accuracy.* An estimator with a lower VO reprojection error is a **proxy win** Yousfi would reject if it does not lower S. So every estimator is scored by what the frozen scorers actually read back.

### 3.1 Build order (estimator-agnostic FIRST, then the first two estimators)

1. **The seam** (`XiEgoTrajectory` + `EgoEstimator` protocol + `fit_se3_bspline`) — one interface, so estimators are hot-swappable.
2. **Estimator (i) — VO/SfM with KNOWN comma calibration** [build first; classical, no net weights to worry about]. Options, ranked internally by robustness on THIS footage: RAFT-flow → ground-**homography decomposition** → ξ (`raft_pose.py` / `raft_radial_pose.py` already in-tree; `homography_from_Rt` = `K(R − t·nᵀ/d)K⁻¹` already in `warp_real_luma_frame0`), OR direct VO (DPVO / DROID-SLAM / COLMAP / ORB-SLAM3 with `fx=fy=910, cx=582, cy=437`). Scale from the ground plane `h=1.22` (metric).
3. **Estimator (ii) — openpilot supercombo pose head** [build second]. Run supercombo's ego-motion head OFFLINE on the YUV frames (ONNX mirror `MTammvee/openpilot-supercombo-model`; weights stay offline — FREE; only the output ξ ships).
4. **LA-Pose Stage-1 inverse-dynamics encoder** = documented **experimental follow-up** (no public code repo yet; project page `la-pose.github.io`). Revisit as OSS matures. NOT in the first bake-off.

### 3.2 The 4-axis measured-through-R metric (all $0, CPU, n600, offline)

Every estimator's ξ is scored on four axes — this is the whole comparison, and **it costs $0** (no paid GPU): the estimator runs offline, the ξ fit is offline, byte-close serialization is CPU, d_pose is the frozen **CPU-torch** PoseNet through R, determinism is the bit-exact gate.

| Axis | What it measures | How (all offline/CPU) | Authority |
|---|---|---|---|
| **(i) lane-rate collapse** | ego-compensated innovation stream bytes @ n600 | byte-close the Stage-2 L1 innovation with this ξ predictor; `pose_carrier_real_bytes` | MEASURED bytes |
| **(ii) pose warm-start** | d_pose after `warp_frame0` by this ξ | frozen CPU-torch PoseNet through R (the `cpu_verdict_d_pose` path) | MEASURED d_pose `[contest-CPU advisory]` |
| **(iii) determinism / decode-legality** | `advect` bit-exact both sides + 30-min budget | the §2.3 tests + inflate wall-clock | pass/fail |
| **(iv) rule-118 cleanliness** | estimator offline, only ξ ships, no net weights | audit the archive manifest | pass/fail |

**The winner is the ξ that minimizes the joint S-contribution it controls:**

```
ξ* = argmin_ξ  [ 25 · bytes_lane_innovation(ξ) / 37_545_489  +  √(10 · d_pose(warp_frame0(ξ))) ]
     subject to ξ affine-calibratable-to-contest-PoseNet (LDM Thm 1, up-to-affine — VERIFY, §5)
```

(d_seg from the band is the SEPARATE #205 trained-in measurement; the ego ξ controls lane RATE + pose here.) This is a **real convex-ish two-term RD/pose objective on MEASURED points** — no predicted-ΔS band asserted; the winning ξ is the one whose measured rows minimize the sum. Deep-math optimum = the ξ that is simultaneously the best ego-motion-compensation predictor (minimizes lane innovation entropy) AND the best warp-real-luma warm-start (minimizes d_pose) — these MAY differ (the pose↔lane tension, §5) → the bake-off MUST report both terms for the shared ξ vs the per-axis optima.

### 3.3 Dispatch discipline

The entire bake-off is **$0, offline, CPU, n600** → no MVP-first paid-smoke gate needed, no lane-dispatch claim. It fully honors "allergic to non-n600 / toys" (all rows n600, real gt) and "measurement-first." No paid GPU until #205's trained-in d_seg run, which is a separate gated dispatch.

---

## 4. Deep-math grounding

- **Ego-factorization is a SOURCE RE-PARAMETERIZATION, not a coding transform (MEASURED, Stage-1).** Stage-1's per-dim breakdown proved the residual is INFORMATION-bound at 26 KB Shannon floor in the CAMERA frame: ~11 KB is the centerline curvature sweeping every frame because the ego drives forward, + ~15 KB per-frame fit jitter. A perfect entropy coder on camera-frame coeffs CANNOT cross the floor. L1 changes the SOURCE: the ego-advected predictor (§2) makes pair-t's coeffs predictable from pair-(t−1) + ξ, so the coded quantity becomes the tiny INNOVATION, not the full curve. This is P-frame motion compensation lifted from pixels to lane-coefficient time-series.
- **Morse-Smale ξ-advection of the whole complex.** The argmax partition is a Morse-Smale complex (separatrices = class boundaries; lanes = class-1 separatrices). `ξ_ego(t)` is the vector field that advects the ENTIRE complex frame-to-frame via the ground-homography pushforward. The **pose warp** (frame0 luma) and the **lane predictor** (class-1 separatrix coeffs) are two READOUTS of the SAME advection field — which is precisely why one stored ξ serves both terms.
- **Up-to-affine calibration to the contest PoseNet.** Per the pose survey (LDM Thm 1), the physical metric ξ ↔ the contest-PoseNet 6-vector is identifiable **up-to-affine** → the trained residual is a 6-DOF affine calibration + a small nonlinearity, NOT ξ-from-scratch → a tiny, well-conditioned, fast-converging problem. This is WHY the warm-start works (start at the physical ξ → −94% from step 0 → refine only the affine map). **FLAGGED for verification (§5): this identifiability is cited, not yet measured for the frozen FastViT PoseNet.**
- **Dual-axis, one object.** One ξ, both contest terms: d_seg-lane RATE (via advection/prediction) + d_pose (via warp-real-luma). rule-118: ξ counted ONCE, marginal attributed across both terms. This is the "the math falls out and fits perfectly" unification — the ego-motion is the shared latent that the source (lanes) and the scorer (pose) both read.

---

## 5. ADVERSARIAL REVIEW (Yousfi + skunkworks; assumption-challenge axis)

Every assumption stress-tested. Classification per the HARD-EARNED-vs-CARGO-CULTED discipline.

### 5.1 "The world-frame road geometry is ~STATIC on 0.mkv" — **CARGO-CULTED (as literally stated); the reframe fixes it.**
- **Road curvature:** on curves the world-frame curve is NOT static — its curvature evolves. **Occlusion:** cars occlude lane pixels → missing/short observations. **Lane changes:** the ego crosses a lane → the L4 slot canonicalization (ego-lane = min|lateral@near|) RE-INDEXES; a "static" world lane suddenly belongs to a different slot. **FOV entry/exit:** lanes enter/leave the frustum → birth/death of `LaneLine` slots (the finest-scale erasure the witness already suffers).
- **Verdict:** the world-frame is NOT globally static — it is **piecewise-smooth with structural events**. The "store the static world lane once" framing (design-authority §L1) is the FRAGILE version. The **predictive-coding reframe (§2) is ROBUST to all four**: curvature → small nonzero innovation; lane-change → one larger innovation spike (or a slot re-assignment flag); occlusion → a coded hold/gap; FOV birth/death → the presence bitmap Stage-1 already carries. The reframe converts "static assumption" (false) into "good predictor + code the innovation" (true and event-tolerant). **HARD-EARNED refinement — adopt the reframe.**

### 5.2 "Monocular ξ is observable up-to-scale" — **HARD-EARNED (scale resolved), with one degenerate case flagged.**
- ξ from a monocular estimator is metric-scale-ambiguous in general. Here scale is FIXED by the KNOWN camera height `h=1.22 m` + the dominant ground plane (homography decomposition gives metric translation). Dashcam regime: forward-dominant (dim-0 = 99.8%) = the well-conditioned VO regime; the classic failure modes (unknown calib, rotation-only, texturelessness) are largely absent — recoverability HIGH.
- **Degenerate case (flag):** stopped at a light / pure-rotation segments → translation scale is unobservable there. Mitigation: the B-spline smooths across such gaps (few knots, slow forward), and a per-segment robustness check flags low-parallax windows. **HARD-EARNED with a named per-segment guard.**

### 5.3 "The bit-exact warp survives quantization" — **HARD-EARNED after the reframe (was the top risk).**
- Pre-reframe (store-in-world-frame): fragile — `unwarp(warp(x))==x` on the fp grid is not guaranteed. Post-reframe (predictive coding): **the exact-inverse requirement is DELETED** — only `advect` bit-identity both sides is needed, guaranteed by the shared numpy-fp64 function on identical inputs + the closed-loop predictor (predict from decoded, not original, coeffs). This is the single most important adversarial improvement: it converts R2's "main build risk" into a non-issue. **HARD-EARNED — this is the decisive revision.**

### 5.4 "Estimator error introduces scorer-visible d_seg/d_pose artifacts" — **REFUTED (fault-tolerant by construction) — a KEY Yousfi property.**
- If ξ is mis-estimated: the lane predictor is worse → the innovation is LARGER → **more bytes, but the camera coeffs are still reconstructed EXACTLY** (the innovation carries the full residual; prediction is lossless). So estimator error costs **RATE, never d_seg correctness** — the scorer never sees prediction error as distortion. For pose: a wrong ξ → a worse warp → worse d_pose warm-start, but the trained affine-cal residual (§4) recovers it → estimator error costs **warm-start convergence, never final d_pose correctness**.
- **Verdict:** the estimator-agnostic design is **fault-tolerant** — a bad estimator degrades gracefully into a rate/convergence cost, never a correctness bug. This is exactly the property Yousfi's "scorer is the authority" lens demands: the scorer's distortion terms are immune to estimator error by construction. **HARD-EARNED — and it is WHY estimator-agnostic is the right architecture (any estimator is safe to try; the bake-off just picks the cheapest).**

### 5.5 "One ξ serves both axes — free" — **HARD-EARNED for STORAGE; UNVERIFIED for OPTIMALITY (the sharpest finding — MEASURE, don't assume).**
- **Storage:** ξ counted ONCE regardless of how many consumers read it → the rate win (ONE ξ vs 600×6 poses vs a separate lane-ego stream) is REAL and HARD-EARNED.
- **Optimality (the tension):** the **pose-optimal ξ** (minimizes d_pose warm-start; weighted toward forward-translation dim-0, the ground homography reads the whole plane) and the **lane-optimal ξ** (minimizes lane-innovation entropy; weighted toward yaw + forward-curvature, the SE(2) advection of the (forward,lateral) coords) MAY DIFFER — they weight the 6 DOF differently. The "one ξ, both axes, FREE" claim conflates "counted once" (true) with "jointly optimal" (unverified).
- **Resolution:** ξ is stored ONCE (the shared default; storing two ξ's doubles the tiny rate and the affine-cal likely closes the gap since PoseNet is eff-dim ~4). But the bake-off (§3.2) **MUST report d_pose AND lane-rate for the shared ξ vs the two per-axis optima** to confirm no meaningful S loss. If `S(shared ξ) − min(S(ξ_pose), S(ξ_lane))` is negligible → the "one object" claim is confirmed MEASURED; if not → store the shared ξ + a tiny per-consumer affine refinement (0 extra shared bytes; a training-time head). **UNVERIFIED-OPTIMALITY — the #1 open question (§6).**

### 5.6 Two carried-over findings (from the build-design + LANDED memos) — reconciled.
- **"Reuse the stored pose ξ (#206) → ~0 marginal"** — **CARGO-CULTED, already falsified (R2 finding #2).** `scorer_targets.py` stores the PoseNet **6-vector OUTPUTS** `(n_pairs, 6)`, NOT a metric se(3)/planar ego twist usable to warp geometry. The metric ξ is a **NEW COUNTED stream** — the "~0 marginal" claim is FALSE and must be MEASURED. This design already treats ξ as a new counted payload (§1.3). Honesty flag resolved.
- **"The byte-close 5th block does not exist" (R2 finding #1)** — **RESOLVED: it EXISTS in THIS worktree.** Grep of the worktree: `serialize_lane_band_rd` / `LBND2` = 22 hits in `analytic_lane_render_band.py`; `_lane_parse_rd` / 5th-block = present in `tools/levelset_byte_close_and_eval.py`; 26/26 tests per the LANDED memo. R2's "does not exist" was measured against the OLDER main-checkout git state (`/Users/adpena/projects/pact`), not this worktree. **The Stage-1 LBND2 codec + 5th block are LANDED here.** Stage-2 L1 (this design) extends the predictor; it is NOT building the 5th block from scratch. (Revision: reconcile the worktree-vs-main-checkout state before the build agent starts, so it edits the LANDED code, not a phantom.)

### 5.7 Assumption-Adversary summary
| Assumption | Class | Action |
|---|---|---|
| Predictive-coding reframe (P-frame motion comp) | HARD-EARNED | ADOPT (decisive revision) |
| ξ stored once = rate win | HARD-EARNED | ADOPT |
| Estimator error = rate/convergence cost, not artifact | HARD-EARNED | ADOPT (fault-tolerance) |
| Known-calib metric scale resolution | HARD-EARNED (+ rotation-only guard) | ADOPT + guard |
| World-frame STATIC | CARGO-CULTED | REPLACE with predict+innovation |
| Reuse #206 pose ξ ~0 marginal | CARGO-CULTED | REJECTED — ξ is a NEW counted stream (measure) |
| LDM Thm 1 up-to-affine for frozen FastViT | INFERRED (cited, unverified) | VERIFY — measure affine-cal residual through R |
| One ξ jointly OPTIMAL for both axes | UNVERIFIED-OPTIMALITY | MEASURE the pose↔lane tension (shared vs per-axis) |

---

## 6. Ranked open questions (the bake-off resolves them; all $0/offline/n600)

1. **[highest EV] Is the shared ξ jointly (rate+pose)-optimal, or Pareto-dominated by two ξ's?** Decides the "one object, both axes, free" claim. Measure `S(shared ξ)` vs `min over per-axis optima`. If negligible gap → unification CONFIRMED measured.
2. **Does the ego-compensated predictor actually collapse the 26 KB camera-frame Shannon floor toward the research ~1–4 KB?** Stage-1's residual was information-bound at 26 KB in the camera frame; L1 must RE-PARAMETERIZE the source. Measure the innovation entropy after ξ-advection — this is the whole rate thesis.
3. **Which estimator wins the 4-axis bake-off — VO/SfM-homography vs supercombo pose head?** (i) lane-rate × (ii) d_pose × (iii) determinism × (iv) rule-118. Robustness on THIS footage decides; LA-Pose is a later follow-up.
4. **Is LDM Thm 1 up-to-affine identifiability REAL for the frozen FastViT-T12 PoseNet?** Measure the affine-cal residual through R (does a 6-DOF affine map close warp-alone d_pose 10.53 → the ~0.018 term?). If the map is NOT affine → the residual head is bigger than claimed.
5. **What is the MEASURED fit-error-vs-M curve (knot count)?** Pick the smallest M below the argmax-band tolerance; confirm the ~hundreds-of-bytes-to-2KB `[prediction]` for the ξ payload.

---

## 7. Recommendation + wire-in

**PROCEED-WITH-REVISIONS.** The unified-ξ estimator-agnostic architecture is sound, the seam is clean, rule-118 is honest, and the deep math (source re-param + Morse-Smale advection + up-to-affine cal + dual-axis) is coherent. Build it, with these **five revisions**:

1. **REFRAME L1 to ego-motion-compensated PREDICTIVE coding** (§2) — predict camera-frame coeffs from ξ-advected previous coeffs; code the innovation. Removes the exact-inverse determinism hazard (R2's "main build risk"). **Decisive.**
2. **Build the estimator-agnostic ξ seam FIRST, then VO/SfM-homography (i), then supercombo (ii)**; bake off on the $0/offline/n600 4-axis measured-through-R metric (§3). LA-Pose Stage-1 encoder = documented experimental follow-up.
3. **MEASURE the pose↔lane ξ-tension** (§5.5) — report shared-ξ d_pose + lane-rate vs the two per-axis optima; confirm the "one object" claim MEASURED before asserting it.
4. **VERIFY (don't assume) up-to-affine identifiability** for the frozen FastViT PoseNet — measure the affine-cal residual through R (§6 Q4).
5. **Reconcile the worktree-vs-main-checkout state** so the build agent extends the LANDED LBND2 5th block (present in this worktree), not a phantom (§5.6).

Rule-118 honesty flag stays live: ξ is a **NEW COUNTED stream** (not a free reuse of #206). Every rate/pose row is MEASURED byte-closed / through-R, never asserted. Pointer **0.19110 UNMOVED** — this design moves nothing until the bake-off's measured rows + #205's trained-in d_seg + gate-T5 byte-closed `upstream/evaluate.py` n600 row.

### Canonical-vs-unique decision per layer
- **ξ engine (SE(3) B-spline + warp):** ADOPT `tac.lie.se3_bspline` + `warp_real_luma_frame0` (verified, parity-gated, numpy-fp64 authority). No fork.
- **Entropy coder:** ADOPT `pose_trajectory_entropy` (verified reversible constriction path; PR103-silver-grade).
- **Bit allocation:** ADOPT `frontier_exact_bitalloc` (exact KKT, arbitrary sensitivity).
- **Ego-estimator + advect predictor:** FORK/NEW (no canonical exists; planar SE(2)⊂SE(3) sub-case + the P-frame predictor). This is the ~10% genuinely-new logic.
- **Raster/composite:** ADOPT (already the FREE rule-118 generic algorithm).

### Observability surface
Per-estimator 4-axis rows (lane-bytes / d_pose / determinism / rule-118), shared-ξ-vs-per-axis-optima S decomposition, fit-error-vs-M curve, innovation-entropy-vs-Shannon-floor, `advect` bit-exact diff, Wave-E `max_abs_uint8_diff`. All machine-readable JSON at byte-close time; cite-chain = ckpt sha + upstream sha + estimator id + seed.

### 6-hook wire-in (research_only)
1. Sensitivity-map: N/A-emit-now (the per-ξ Δrate/Δd_pose marginals are `[prediction]`; filled by the bake-off's measured rows → `tac.sensitivity_map` then). 2. Pareto: ACTIVE-as-design — the §3.2 joint (rate+pose) objective is the Pareto point the shared ξ occupies. 3. Bit-allocator: ACTIVE-as-design — the ξ B-spline + innovation KKT allocation are the pose+lane bit-allocator primitives. 4. Cathedral autopilot: N/A (research_only; no archive-deployable artifact here). 5. Continual-learning: this memo + the bake-off rows are the anchors the #205/#191 pose+lane config consumes. 6. Probe-disambiguator: the estimator bake-off IS the disambiguator (VO vs supercombo, resolved by measured-through-R, never proxy).

**Council mission-contribution:** `frontier_breaking` — the unified ξ is the shared latent that lets ONE tiny counted stream serve BOTH the lane-rate collapse (toward the ~1–4 KB target that makes the band net-positive) AND the pose warm-start (−94% from step 0). All MEANS; the END is a byte-closed exact row below 0.19110.

---
*Advisory / design-only. Pointer contest-CPU 0.19110 UNMOVED — moves only via a byte-closed `upstream/evaluate.py` n600 exact row. MPS-never; every quantitative row here is `[prediction]`/`[derived]`.*
