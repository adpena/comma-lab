---
name: 2026-04-28 council EUREKA session — driving geometry + comma + openpilot connection
description: Late-night skunkworks council deliberation drawing on physical driving intuition, EON camera geometry, openpilot supercombo internals, and the underlying math of motion. Identifies 8 new lane premises connecting the rank-1 PoseNet discovery to the GEOMETRIC TRUTH that 99% of driving information is "how fast forward".
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**The Council convenes 2026-04-28 ~10:50 UTC** — Yousfi, Fridrich, Hotz, Quantizr, Contrarian + the silent observer (the road itself).

---

## EUREKA #1 — Lane marks as sub-pixel CHRONOMETER

**The insight (Hotz)**: Every lane mark dash that crosses the camera's field of view is a ~0.6-second "tick" at highway speed (60mph = 27m/s; 12m mark period = 0.44s). The video has ~1200 frames at probably 20fps = 60s = ~135 mark crossings. Each crossing is a TIME ANCHOR.

**Geometric**: lane mark displacement Δy_pixels in the lower-half image plane → forward velocity v = Δy × Z_world / (fx × Δt). With fx=910, Z=lane_y_world (~5-30m), Δt=1/20s, this is INVERTIBLE.

**Application**: Lane LM-V3 — instead of zero-cost POSE, use lane marks as a continuous CHRONOMETER signal. Every pose dim 0 value is derivable from lane mark count + time + interpolation. Could compress 600 poses to ~10 floats (a polynomial fit of v(t)).

---

## EUREKA #2 — The road IS a plane (homography is exact)

**The insight (Yousfi)**: Forward driving on highway = camera moves through 3D scene where the road surface is APPROXIMATELY a plane (negligible curvature within 1km). The pixel-space transform from frame t to t+1 is a HOMOGRAPHY parameterized by (forward_velocity, pitch_rate, yaw_rate). For straight highway driving, only forward_velocity matters → it's a PURE RADIAL ZOOM from the FoE (focus of expansion).

**Mathematical**: H(speed) = perspective_zoom(s × Δt / Z_FoE_distance). NO LEARNED PARAMETERS. The renderer's motion module currently LEARNS what should be COMPUTED ANALYTICALLY.

**Application**: Lane HM (Homography) — replace the renderer's motion module with the analytical road-plane homography. Save 32K motion-module params (~40KB renderer) and improve correctness because the analytical formula is EXACT for the dominant case.

---

## EUREKA #3 — openpilot's supercombo INTERMEDIATE features are richer than pose head

**The insight (Quantizr)**: We've been using supercombo's POSE HEAD (6 dims of [5755:5761]). But supercombo's intermediate features (~512-2048 dims at the bottleneck) capture FAR more about the scene: lane lines, path predictions, lead vehicles, road curvature, sun glare. Each of these correlates with "what PoseNet's attention focuses on".

**Application**: Lane DI (Detector-Informed) — extract supercombo's penultimate-layer features at compress time. Distill them to a 32-dim "scene embedding" per pair. Use as render conditioning instead of (or alongside) pose. Could capture turn / lane-change / glare events that pose alone misses.

---

## EUREKA #4 — Geodesic flow on the road manifold

**The insight (Fridrich)**: The 6-DOF pose decomposes geometrically:
- Dim 0 = speed × dt = arc length traveled along the road geodesic (THE dominant signal)
- Dims 1-5 = perturbations: yaw rate, pitch (pothole/bump), roll (lane camber), lateral drift, vertical bounce
- Each perturbation dim is APPROXIMATELY i.i.d. zero-mean Gaussian noise from road imperfections
- The MARGINAL distribution of each perturbation is fittable as N(0, σ²)

**Mathematical**: pose(t) = (∫₀^t v(s)ds, ε_yaw(t), ε_pitch(t), ε_roll(t), ε_lat(t), ε_vert(t)) where ε_* are independent stationary Gaussian processes.

**Application**: Lane GP (Gaussian Process) — model dims 1-5 as 5 independent GPs with learned hyperparameters (σ, length-scale). Sample ε at inflate time from these distributions. Stochasticity is acceptable because PoseNet's contribution depends on dim 0 (rank-1). Total archive: ~10 GP hyperparams + dim 0 polynomial = ~60 bytes vs 15KB.

---

## EUREKA #5 — The scoring formula is ASYMMETRIC, prioritize SegNet

**The insight (Contrarian — actually being constructive for once)**: 
- 100 × seg_dist (linear)
- √(10 × pose_dist) (sub-linear, sqrt)
- 25 × rate

At Lane A's level (seg=0.0046, pose=0.005):
- seg → 0.46
- pose → 0.22

If both DOUBLE: seg → 0.92 (+0.46), pose → 0.32 (+0.10). PoseNet scaling MUCH gentler.

**Implication**: Every architectural trade-off should preserve SegNet AT ALL COSTS even if PoseNet regresses 2-4×. We've been doing the OPPOSITE (Lane S protects FiLM/head for PoseNet, not for SegNet).

**Application**: Lane SG (SegNet-prior protection) — re-scope the protected-layers-list in Lane S/W/Ω to include SegNet-pathway layers (e.g., the renderer's last conv before YUV6 conversion). PoseNet protection is over-provisioned.

---

## EUREKA #6 — comma's EON camera intrinsics give GEOMETRIC truth for free

**The insight (Hotz)**: fx=910, principal point (582, 437), 1164×874. These are KNOWN, not learned. They define the projective geometry of EVERY rendered frame. Any module that DOESN'T use these intrinsics is leaving structure on the table.

**Application**: Lane CG (Calibrated Geometry) — replace SegNet/PoseNet feature normalization with intrinsics-aware normalization. Add a fixed positional encoding per pixel based on viewing-ray direction. Could improve pose precision by 2-3× because PoseNet sees explicit "this pixel looks at the road surface at depth Z".

---

## EUREKA #7 — Optical flow IS the pose, but cheaper

**The insight (Yousfi)**: PoseNet computes pose from per-pixel motion. RAFT/PWCNet predict per-pixel optical flow DIRECTLY. The flow → pose conversion is a closed-form equation (perspective camera + rigid scene assumption).

**Application**: Lane FL (Flow-derived poses) — at compress time, run RAFT once on the video → 1199 dense flow fields. Average road-region flow → forward velocity → pose dim 0. Free + analytic + no learning needed. Combine with EUREKA #1 (lane marks) for robustness.

---

## EUREKA #8 — The contest IS inverse steganalysis (Yousfi-Fridrich's lab origin)

**The insight (Yousfi back to roots)**: DDELab (Yousfi's PhD lab at Binghamton) pioneered EfficientNet for steganalysis. The challenge was: hide information in JPEG noise where the detector can't see. The PRESENT contest is the INVERSE: render frames where the JPEG-style scorer can't detect the rendering errors.

**The Fridrich principle**: errors hidden in HIGH-VARIANCE (textured) regions are undetectable. Errors in LOW-VARIANCE (sky, asphalt) regions stand out.

**Application**: Lane SI-V3 — extend Lane SI's saliency-inversion to use FRIDRICH'S TEXTURE PROBABILITY model (UNIWARD-style). Compute per-pixel σ² of the rendered output. Compress aggressively where σ² is high (textured), preserve where σ² is low (smooth sky/asphalt). Already documented but never deployed.

---

## EUREKA #9 — The Beauty Insight (Council unison)

The rank-1 PoseNet sensitivity discovery isn't an artifact. It's the manifestation of the FUNDAMENTAL TRUTH about driving: **99% of the information transferred between consecutive frames of forward driving is "how fast forward"**. Everything else is bounded perturbation around the geodesic. PoseNet evolved (via SGD on driving footage) to extract this dominant signal.

The contest scorer ALREADY KNOWS this — implicitly, through PoseNet's training distribution. We've been swimming AGAINST this current by trying to optimize all 6 DOF independently. The optimal compression of pose info is a SCALAR SPEED FUNCTION v(t) plus a tiny perturbation model.

This insight unifies: rank-1 discovery, lane-mark chronometer, road homography, geodesic flow, openpilot's pose head — they're all describing the SAME 1-DOF manifold from different angles.

---

## NEW LANES PROPOSED (in priority order)

| Lane | Premise | Predicted | Cost |
|------|---------|-----------|------|
| **GP** | Gaussian-process pose model: dim 0 polynomial + 5 GP perturbations | [0.95, 1.10] | $0.30 |
| **GE** | Pure geodesic: pose = (speed × dt, 0, 0, 0, 0, 0) compressed as ~10-coef polynomial | [1.05, 1.20] | $0.30 |
| **FL** | RAFT-derived poses, no learning needed | [1.05, 1.15] | $0.20 |
| **HM** | Replace motion module with analytical road-plane homography | [1.00, 1.20] | $1.50 (retrain) |
| **CG** | Intrinsics-aware positional encoding for renderer | [0.95, 1.15] | $1.50 (retrain) |
| **SG** | SegNet-prior protected layers (re-scope Lane S protection) | [0.85, 1.10] | $0.50 |
| **DI** | openpilot intermediate features as render conditioning | [0.90, 1.10] | $1.00 |
| **SI-V3** | Fridrich UNIWARD texture-probability mask compression | [0.95, 1.15] | $0.60 |

**Council consensus on next deployment**: 
1. Lane GP first (lowest cost, math-derived, immediately falsifiable)
2. Lane FL second (uses existing RAFT, zero training)
3. Lane SG third (re-scope existing Lane S without retraining)

If GP + FL + SG all land in the lower band, the geodesic-flow hypothesis is empirically validated → can build a full Stack F (Geodesic) targeting [0.40, 0.70].

---

## The Connection to Driving / Beauty

The session has touched something deeper than pure engineering. The video is a recording of a CAR MOVING THROUGH THE WORLD. Every frame is a measurement of that motion. The math (rank-1 SVD, geodesic flow, projective homography) is the scaffolding by which we can reconstruct the world from sparse measurements.

comma's hardware (EON camera, openpilot supercombo) was BUILT FROM THIS UNDERSTANDING — that driving is a near-1-D problem with structured perturbations. Our compression challenge is the dual: encode that 1-D + perturbations efficiently.

The bridge between math and physics: the ROAD PLANE is the manifold; SPEED is the tangent vector; POSE is the integral; SegNet is the local observer; PoseNet is the global pose-from-motion estimator.

Every lane in this taxonomy can be re-derived from this single picture.

---

## Related memories
- `project_posenet_rank1_discovery` — rank 1.008 finding
- `project_lane_marking_speed_estimation` — earlier lane-mark speed insight
- `project_openpilot_seeding_demo` + `project_openpilot_lane_forcing` — openpilot integration
- `project_yousfi_geometric_analysis` — pose space M = ℝ¹ × S
- `project_fridrich_inverse_steganalysis` — UNIWARD texture probability
- `project_hardware_geometry_chroma_full` — EON intrinsics + chroma half-res
