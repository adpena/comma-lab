# POSE REPRESENTATION — deep research + deep math (the store-nothing-ξ d_pose≈2.5 ceiling)

**Date:** 2026-07-08 · **Axis:** `[deep-research + code-trace + existing run-1 telemetry, n600]`
NON-PROMOTABLE · **$0, read-only** (pid 63069 + run dirs UNTOUCHED; NO launch/train) ·
**Pointer contest-CPU 0.19110 UNMOVED — MEANS.** Every external claim CITED; every number
MEASURED / DERIVED / labeled. ours-vs-borrowed separated. verdict_scope inline.

Operator question: our store-nothing-ξ pose carrier is H-TARGET capped at d_pose≈2.5 because a
single-keyframe ground-homography + rank-6 twist cannot reproduce the two-frame optical flow PoseNet
reads. Find the RIGHT compact pose representation.

## STORES CONSULTED
`pose_pb_filmreadback_diagnosis_20260708.md` (the H-target verdict; s_t grid; Anchors A/B) ·
`counterforce_insufficiency_deepmath_20260708.md` · CLAUDE.md §pose-solved + §"Exact scorer
architectures" (PoseNet=FastViT-T12, 12-ch = 2 frames × YUV6, resize 512×384, Hydra head → 6-dim,
MSE first 6) · memory L68 (pose OPEN+unmeasured on witness) / L69 (break-even error bars) / L18
(ancestor = lessons not numbers) · code: `src/tac/boundary_math/warp_real_luma_frame0.py` (the
carrier; `xi_eff = xi_stored + scale·dxi`) + `src/tac/boundary_math/keyframe_codec.py` (rate
primitives) + `upstream/evaluate.py:77-92` + `upstream/modules.py` (scorer) + `src/tac/lie` (SE(3)).

**Apparatus-validity precondition MET:** run-1 LIVE (pid 63069). All run-1 numbers below re-read from
telemetry / source, not reasoned. External literature is cited by URL; our numbers are labeled
MEASURED (run-1 / build-fit) or DERIVED (algebra) or PREDICTED (owed to the decisive measurement).

---

## 1. ROOT-CAUSE VERDICT (deep-math): the WARP MODEL (planar homography), NOT the twist RANK

**The rank-6 twist is a red herring. The binding limit is that a homography is a PLANE-ONLY warp, and
real dashcam flow is DEPTH-STRUCTURED (parallax).** This is the Irani–Anandan–Weinshall **plane +
parallax** decomposition [1][2]: the apparent inter-frame motion of any pixel `p` splits exactly as

> `u(p) = u_planar(H; p)  +  γ(p)·e`

where `u_planar` is the flow a homography `H` of a reference plane induces, `e` is the epipole
(translation direction), and the **residual parallax** `γ(p) = (t_z / d(p))·(1 − d_plane/d(p))`-type
term is proportional to BOTH the translation magnitude and the pixel's **depth deviation from the
reference plane**. Our carrier's homography `H = K(R − t·nᵀ/d)K⁻¹` (`warp_real_luma_frame0.py:170`)
reproduces `u_planar` EXACTLY off the stored ξ, but sets **γ(p) ≡ 0**. So every off-plane pixel — cars,
buildings, poles, the whole upper scene, the ego-hood — is warped along the ground-plane flow, which is
WRONG for it. PoseNet reads a flow field that is internally inconsistent (planar everywhere) and its
learned 6-DOF regressor lands ~2.5 MSE from the target.

Why the residual `dxi` cannot fix it (the measured plateau): `xi_eff = xi_stored + scale·dxi`
(`warp_real_luma_frame0.py:710`) — the residual **stays inside the 6-DOF homography family**. It only
re-selects which global plane-warp `H` is applied; it has ZERO capacity to add the *per-pixel* term
`γ(p)`. So training `dxi` under `w_pose>0` descends to the **projection of the target onto the planar
manifold** and saturates — exactly the observed 2.562 (build self-fit, no residual) → 1.793 (trained
residual) plateau (MEASURED, `pose_pb_filmreadback` Anchors A/B). More twist rank (per-region twists,
SE(3) B-spline) buys only *piecewise*-planar = a coarse multi-plane depth proxy; it does not add
continuous parallax. **verdict_scope: FORMULATION** — the "single/piecewise homography + twist" warp
FAMILY is capped ~2.5; it is NOT a paradigm kill of "store-ξ-don't-reconstruct."

**The deep reason PoseNet is so sensitive to this:** monocular ego-motion has the classic
rotation/translation ambiguity — a small rotation and a translation induce nearly the same flow far
from the camera. The cue that DISAMBIGUATES them is **parallax** (near objects move more than far
under translation, but identically under rotation) [3][8]. A planar-only flow field DESTROYS the
parallax cue, so PoseNet cannot separate the forward-translation (the dominant driving motion) from
rotation → its 6 outputs are off by an O(1) MSE that no global-warp choice removes. **Corollary
(DERIVED, testable for $0):** the per-pair d_pose floor should scale with the pair's **translational
parallax energy** (≈ `|t_forward| × depth-spread`); pure-rotation / near-static pairs should already
sit near-zero, large-forward-motion pairs should dominate the 1.8 mean.

---

## 2. PoseNet's TRUE sufficient statistic (the intrinsic-dimension question)

**Jacobian rank is a red herring too — it is a MANIFOLD-IMAGE problem, not a rank problem.** PoseNet
maps a 2-frame input to 6 scalars, so the local Jacobian of the 6 outputs w.r.t. the (2×512×384×3)
input has rank ≤ 6 — the pose-sensitive input subspace is *locally* ≤6-dim (consistent with our own
`#80`). Naively that says "6 numbers suffice." But the achievable set of pairs from the homography
family is a **6-manifold that is MISALIGNED with PoseNet's 6-dim sensitive subspace**: moving ξ sweeps
*planar* flow variations, whereas PoseNet's sensitive directions are *depth-structured* flow
variations. Two 6-manifolds that do not intersect at the target ⇒ the min distance is the measured
~2.5 gap. **So "compact carrier possible?" resolves as: YES, but the compact object must generate a
DEPTH-CONSISTENT flow, not a planar one.**

The minimal object that generates depth-consistent flow is the standard SfMLearner / Monodepth2
inverse-warp [3][4]: `p' = K · T(ξ) · D(p) · K⁻¹ · p` — **one photometric frame + a per-pixel depth
map D + the 6-DOF pose ξ**. This is precisely the plane+parallax generalization where the single plane
distance `d` becomes a per-pixel `D(p)`. Flow **factorizes** as `flow = f(D, ξ)`: depth is the shared,
slowly-varying *scene structure*; ξ is the per-pair *motion* (already stored). **The sufficient
statistic for driving frozen PoseNet to d_pose→0 is therefore {compact D (shared per clip) + ξ (already
stored)} applied to a frame PoseNet can read** — NOT dense per-pair flow, and NOT more twist.

**The load-bearing architectural constraint (VERIFIED from `evaluate.py:77-79`):** the evaluator
decodes ONE video (`batch_comp`, `seq_len=2`) and feeds the SAME frames to both nets — SegNet reads
`x[:,-1]` (frame1), PoseNet reads the pair. ⇒ **frame0 is genuinely PoseNet-only (seg-free); frame1 is
shared with d_seg (the W8 dual-use crux).** So the pose representation must synthesize frame0 freely,
but frame1 is whatever the d_seg render produces. Two consequences: (i) the depth-warp for frame0 is
free of d_seg cost (warp the render/keyframe by `f(D,ξ)`); (ii) whether PoseNet reads correct motion
from a motion-coherent but *task-space* (non-photometric) frame1 is a SECOND, un-measured axis — flag
it, because Anchor A already shows the warp limit bites even with fully-photometric content, so the
warp/depth fix is necessary FIRST regardless.

---

## 3. LITERATURE SWEEP (cited)

- **Plane + parallax / stratified motion** [1] Irani, Anandan, Weinshall, *From reference frames to
  reference planes: multi-view parallax geometry*, ECCV 1998
  (https://www.weizmann.ac.il/math/irani/sites/math.irani/files/publications/from_reference_frames.pdf);
  [2] Irani et al., *Direct recovery of planar-parallax from multiple frames*
  (https://www.weizmann.ac.il/math/irani/sites/math.irani/files/publications/direct_recovery.pdf). The
  canonical statement that apparent motion = homography + depth-scaled parallax; our carrier is the
  homography-only truncation.
- **Depth+pose = flow (compact structure/motion factorization)** [3] Zhou et al. SfMLearner (jointly
  learn depth+ego-motion, monocular, self-supervised); [4] Godard et al. *Monodepth2*, ICCV 2019
  (https://openaccess.thecvf.com/content_ICCV_2019/papers/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.pdf).
  The inverse-warp `p'=K·T·D·K⁻¹·p` is the exact generator of depth-consistent flow from {frame, D, ξ}.
- **Depth-map compression is cheap** (piecewise-smooth → few bytes) [5] EURASIP JIVP 2015 low-complexity
  HEVC depth coding (https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-015-0058-5);
  constant-depth regions → "as little as 5 bytes"; [6] INR/overfit image+depth compression, Strümpler
  et al. ECCV 2022 (https://arxiv.org/abs/2112.04267) — coordinate-net overfit + QAT + entropy coding
  reaches very low bitrate per instance. Supports a sub-KB per-clip depth carrier.
- **Compact learned optical-flow coding** (the "store flow directly" alternative) [7] Cool-chic video
  (~800-param learned inter coding, 2024) + resolution-adaptive flow coding, ECCV 2020
  (https://link.springer.com/chapter/10.1007/978-3-030-58536-5_12); flow is 2-channel + smooth →
  low-rank/INR ~1–5 KB/pair. Dominated by depth+ξ (below) because ξ is already stored and D is shared.
- **Coding for machines / task-driven** [9] MPEG VCM/FCM review, ZTE Communications 2024
  (https://www.zte.com.cn/content/zte-site/www-zte-com-cn/global/about/magazine/zte-communications/2024/en202401/Review/en202401008.html);
  [10] task-driven compression, IEEE TIP (https://ieeexplore.ieee.org/document/10004012/). Frames the
  right objective: preserve ONLY what the frozen machine task (PoseNet's 6 outputs) needs, at
  iso-task-accuracy BD-rate — a stored pose frame may be degraded to PoseNet's input fidelity (512×384,
  normalized), not human fidelity, shrinking bytes a lot.
- **Visual-odometry: accurate flow ⇒ accurate pose** [8] deep-flow-driven VO / bundle-adjustment
  (https://arxiv.org/pdf/2111.11141) — confirms PoseNet-class outputs are a (near-deterministic)
  functional of the flow field, so matching the flow field matches the output.

---

## 4. THE RATE–DISTORTION FORK (byte affordance is MEASURED-huge; distortion-reachability binds)

Contest rate term = `25·bytes/37,545,489` ⇒ **1.0 of S = 1.502 MB**. Pose term now `√(10·1.793)=4.231`.
Byte affordance to reach a target d_pose (DERIVED; total across 600 pairs, and ÷600 per-pair):

| target d_pose | pose term √(10·d) | ΔS_pose vs 4.231 | byte affordance (total) | per-pair |
|---|---|---|---|---|
| 1.793 (now) | 4.231 | 0 | 0 | 0 |
| 0.10 | 1.000 | 3.231 | **4.85 MB** | 8.1 KB |
| 0.01 | 0.316 | 3.915 | **5.88 MB** | 9.8 KB |
| 3.4e-5 (ancestor) | 0.0184 | 4.212 | **6.33 MB** | 10.5 KB |

**Headline: rate is NOT the binding constraint — reachable-distortion is.** We can afford ~8–10 KB PER
PAIR and still win S, which comfortably covers a compact depth map OR even a heavily-degraded stored
photometric keyframe. So the decision is purely "which representation actually reaches low d_pose,"
not "can we afford it." (Consistent with L69's break-even framing; the achieved d_pose for the depth
option is PREDICTED and owed to §6's measurement — do NOT bank the ancestor 3.4e-5, it is
ancestor-borrowed per L18/L68.)

---

## 5. RANKED POSE-REPRESENTATION OPTIONS

| # | Representation | Predicted / measured d_pose | Bytes (total / pair) | On Pareto? | ours-vs-borrowed |
|---|---|---|---|---|---|
| **A** | **Depth-consistent warp: compact per-clip depth D + stored ξ** (SfMLearner warp `K·T(ξ)·D·K⁻¹`); replace the plane `d` with per-pixel `D(p)`. Warp frame0 (seg-free) by depth-consistent flow. | **PREDICTED ≪1** (owed to §6 L2); depth restores the parallax cue the homography drops | D piecewise-smooth + SHARED across a clip → ~0.5–5 KB/keyframe amortized to **~hundreds B/pair**; ξ already stored | **YES — highest-EV.** Keeps store-nothing (structure is a compact static-ish descriptor); rate trivially affordable | borrowed: depth+pose warp (SfMLearner). ours: compact per-clip D + free openpilot road-geometry seed + dual-use with stored ξ |
| **B** | **Store a photometric pair at PoseNet fidelity** (second real frame per pair, degraded to 512×384 VCM fidelity + temporal coding). d_pose target = PoseNet(real pair) by definition. | **~ancestor** (compression-artifact-limited); the DISTORTION anchor / safe floor | ~2–6 MB total (600 × ~3–10 KB HEVC/VCM); at edge of affordance | **YES (rate-heavy end / safe fallback)** — but abandons store-nothing; note a real frame1 also ≈ solves d_seg → collapses the witness into a plain codec (the fundamental tension) | borrowed: video codec. ours: VCM-fidelity degrade tuned to PoseNet's input |
| C | Compact dense-flow carrier per pair (low-rank/INR flow), warp f0 by it | near-0 if flow accurate; degrades with flow compression | ~1–5 KB/pair (flow not shareable across pairs) | **NO — dominated by A** (A factorizes flow = shared-D × per-pair-ξ; both cheaper than storing full flow/pair) | borrowed: learned flow codec |
| D | Richer twist: per-region twists / SE(3) B-spline / non-rigid field | **~1.8–2.5 (does NOT close)** — piecewise-planar ≈ coarse multi-plane depth proxy only | small | **NO — register the negative**: rank is not the lever; the warp MODEL (planar vs depth) is | ours (twist parametrization) |

**Pareto front = {A, B}.** A is the store-nothing-preserving, rate-cheap arm and the one to measure
first; B is the distortion anchor and the safe fallback if A's task-space-frame1 issue proves binding.
C, D dominated (register D's negative in the DAG so "add more twist" is not re-opened).

---

## 6. THE DECISIVE NEXT $0 MEASUREMENT (a warp-model-complexity ladder, all read-only / one-shot local)

Run the frozen CPU-torch PoseNet (`cpu_verdict_d_pose_batch`, NEVER MPS) on real frames from
`upstream/videos/0.mkv`, climbing the warp model while holding everything else fixed:

- **L0 (free, existing telemetry):** scatter per-pair d_pose (homography carrier) vs per-pair `|t_forward|`
  from the stored ξ. **If d_pose ∝ translational parallax energy → root-cause CONFIRMED** (parallax is
  the missing term) and it predicts which pairs a depth term will fix most. Zero new compute.
- **L1 (MEASURED):** homography carrier = 2.562 build-fit / 1.793 trained. (anchor)
- **L2 (the decisive one):** run an off-the-shelf monocular depth net on real_f0 (one-shot, $0 local),
  then the SfMLearner depth-warp `K·T(ξ)·D·K⁻¹` → PoseNet([real_f0, depth-warp(real_f0)]) → d_pose.
  **If ≪ 2.5 → depth is the fix**, and sweep D's compression (coarse/low-rank/INR, and per-clip-shared)
  to trace the d_pose(bytes) curve = the actual Pareto for Option A.
- **L3 (upper-bound):** compute the REAL dense flow between (real_f0, real_f1) (RAFT or classical),
  warp f0 by it → PoseNet([real_f0, flow-warp(real_f0)]) → d_pose ≈ achievable floor of any warp-carrier
  (only occlusion/disocclusion holes remain). Confirms the ceiling if depth is imperfect.
- **L4 (definitional):** PoseNet([real_f0, real_f1]) = 0 (target is defined as PoseNet(real pair));
  bound Option B's artifact floor by degrading the stored frame1 to VCM fidelity.
- **SECONDARY axis to measure (the frame1 constraint):** repeat L2 with the WITNESS task-space render as
  the warp source instead of real luma → does PoseNet read correct motion from a motion-coherent
  *task-space* pair? This isolates whether Option A alone suffices or whether a photometric frame1 (→ B)
  is forced.

L2 is the single highest-value next unit: it decides A-vs-B for $0 and, if green, hands the rate curve
directly to the byte-close.

---

## 7. CANDIDATE CANONICAL EQUATIONS (council-FLAGGED, NOT registered — anchors owed to §6 + byte-close)

1. `posenet_planar_parallax_dpose_floor_v1` — d_pose of the homography carrier ≈ residual
   **translational-parallax energy** the plane-warp cannot represent:
   `d_pose_floor(pair) ≈ Σ_p [γ(p)·|e|]²`-weighted-through-PoseNet, `γ(p)=(1/D(p) − 1/d_plane)`, → 0 for
   pure rotation / near-plane scenes, O(1) for forward driving. Anchored: null 182 → homography 2.562
   build / 1.793 trained (MEASURED run-1); the per-pair |t| scaling is PREDICTED (L0 owed).
2. `pose_sufficient_statistic_depth_pose_v1` — PoseNet's 6 outputs are driven to d_pose→0 by a
   depth-consistent 2-frame flow generatable from {one PoseNet-readable frame + per-pixel depth D +
   6-DOF ξ}; the minimal COUNTED statistic is (compact per-clip D) + (ξ already stored); the rank-6
   twist alone is NOT sufficient (planar image manifold ⊄ PoseNet-target). Anchors owed to L2/L3.

Both mirror the diagnosis's `warp_real_luma_frame0_dpose_ceiling_v1` (council-flagged): the ceiling is
measured; the depth-closes-it half is PREDICTED and owed to the L2 measurement before registration.

---

## 8. OURS-vs-BORROWED accounting (NO-FAKE #7)

- BORROWED (cite, defensive-bank / on-ramp): plane+parallax decomposition [1][2]; the SfMLearner /
  Monodepth2 depth+pose inverse-warp [3][4]; depth-map + INR compression [5][6]; VCM task-driven
  framing [9][10]. None of these is "ours-original"; they are the standard machinery.
- OURS-original (the intersection nobody occupies): applying the depth+pose warp as a **task-space
  pose carrier for a frozen scorer** where frame0 is seg-free and depth is a **per-clip shared
  static-ish descriptor seeded FREE from the openpilot road prior (rule-118)** with only the LEARNED
  depth-deviation COUNTED; the dual-use of the already-stored ξ for both the warp and the PoseNet
  ego-motion. The witness capstone remains the vehicle; this is the pose-half representation fix.

## FINAL STATE
$0 deep-research + code-trace + existing run-1 telemetry; n600; pid 63069 + run dirs UNTOUCHED; NO
launch/train. **Pointer 0.19110 UNMOVED — MEANS.** Root cause: the carrier warps with a PLANE-ONLY
homography and drops the depth-scaled parallax that PoseNet needs to resolve ego-motion; the rank-6
twist lives inside that same planar family, so it plateaus at ~1.8–2.5 by construction. Fix =
depth-consistent warp (Option A, compact per-clip depth + stored ξ) — rate is trivially affordable;
the binding unknown is reachable d_pose, decided for $0 by the §6-L2 depth-warp measurement.

Triality legs: DAG FEED-poserep owed on next append · two candidate equations council-flagged (not
registered) · no DSL change (investigation only).
