# Alternative IN-TRAINING pose levers for the task-space level-set witness — full survey + ranking

- **Date:** 2026-07-02
- **Author:** POSE-LEVER RESEARCH subagent (research/synthesis only; NO code edits, NO GPU)
- **Axis discipline:** every row below is `[prediction]` / `[research-signal]` — **NO score claim**. MPS-never. The pointer is **contest-CPU 0.19110, UNMOVED**; nothing here moves it. Levers become real only when a byte-closed `upstream/evaluate.py` n600 exact row measures them.
- **`research_only: true`** — this is a planning/survey ledger. It feeds the #205 witness pose config + #221 (w_pose>0 FT) + #227. It authorizes NO dispatch.
- **Feeds:** #205 (witness θ* launch), #221 (pose-residual FT), #227.
- **Provenance:** builds on measured findings in `project_pose_solved_screw_twist_dual_use_film_conditioned_sidecar_20260701.md` + `project_gr_unified_action_full_witness_architecture_20260629.md`, honoring the ancestor-vehicle discipline (`feedback_ancestor_vehicle_findings_are_lessons_not_transferable_hnerv_pr95_abandoned_20260701.md`).

---

## 0. The baseline these levers must BEAT or COMPLEMENT (our MEASURED ground truth — do NOT re-derive)

The contest defines **`d_pose = MSE(PoseNet(gen_pair)[:6], PoseNet(orig_pair)[:6])`**, PoseNet = **frozen FastViT-T12 on 2-frame YUV6**, term = **√(10·d_pose)**. This shapes EVERYTHING below:

- **We do NOT train a pose head.** PoseNet is frozen and IS the authority. The authority metric is **6-vector Euclidean MSE in PoseNet-output space**, NOT a geodesic distance in SE(3). So any Lie/geodesic/equivariant machinery applies to how WE parameterize/supervise the stored twist **ξ**, not to the scorer.
- **PRIMARY realization (measured, pose gate #206):** **warp-real-keyframe-luma frame0 by the stored twist ξ** carries **−94%** (d_pose 182→10.53, ~0 training; frame0 is seg-free per `modules.py:108`, so warping it can't disturb d_seg). PoseNet reads REAL warped motion → **pose-valid by construction** (retires the flat-SDF-render OOD risk: Jacobian rank **6 FULL**, σ_max 0.076 ≥ real 0.053). The residual 10.53 → ~0.018-term needs a trained FiLM/dxi residual (**REACHABLE per rank-6, UNMEASURED on witness** — the 3.4e-5 magnitude is the ANCESTOR-RGB anchor, not yet witness-validated).
- **seg⊥pose decoupling is FREE (measured):** render-space cos(∂d_seg/∂F, ∂d_pose/∂F) median **5.9e-5**, **99.95% pose-null**, PoseNet Jacobian rank-6 full. → exact freeze-and-add via **disjoint frames + trunk-stopgrad**. **PCGrad-style gradient surgery = FALSE FRIEND** (confirmed below).
- **PoseNet is low-dim:** eff-dim ~4, output rank ~1.008, **dim-0 forward-speed = 99.8%**. Pose is a smooth, low-dim, topology-preserving, near-free facet. The pose stream caps at **~5KB / −0.003 score even at ∞ compression** (`joint_admm_proximal_pose_delta.py`). **d_seg is the binding wall, not pose.** ⇒ the highest-value pose lever is often the one that **frees curriculum/capacity for d_seg while banking pose**, not the one that squeezes the last µ of d_pose.
- **ξ is stored ONCE, dual/triple-use:** SE(3) B-spline ξ_ego(t) (`tac.lie.se3_bspline`, Sommer 2020, already built MLX+numpy) → pose (warp luma) AND lane/separatrix advection (d_seg) AND temporal consistency all fall out of the same vector field.
- **Already built in-tree (BUILD ON, do not reinvent):** `lie/se3_bspline.py`, `lie/screw_blend.py`, `riemannian_pose_optimizer.py` (Bonnabel 2013 Riemannian SGD on SE(3)), `geodesic_pose.py` (Chebyshev rank-1), `lora_pose_v2.py` (learnable-rank 1–6 gated), `pose_delta_codec_v2.py` (arithmetic-coded deltas), `freezing/pose_gradient_stop_after_warmstart.py`, `kl_pose_distill.py`, `raft_pose.py`/`raft_radial_pose.py`, `joint_admm_proximal_pose_delta.py`, `local_acceleration/mlx_yuv6_primitive_parity.py` (**MLX-native differentiable YUV6 — the gradient-reachability plumbing**), `scorer_targets.py`/`pose_from_embedding.py`.

**Gradient-reachability prerequisite (not a lever — a MUST-HAVE enabler for EVERY w_pose>0 loss below):** upstream `rgb_to_yuv6` is `@torch.no_grad()`/in-place → it SEVERS PoseNet gradients. Any pose training loss is a no-op unless differentiable YUV6 is patched in first (PR95/#106 monkeypatch; MLX-native version = `mlx_yuv6_primitive_parity.py`). Confirm this is active before crediting ANY loss-shaping lever.

---

## 1. The lever taxonomy — seven orthogonal classes

The operator's list collapses into seven classes. They are **mostly orthogonal → composable**, NOT competing. **Class H (the openpilot ego-motion WARM-START INITIALIZER) is ELEVATED to a PRIMARY, top-tier candidate** (operator 2026-07-02): it is the pose-analog of the openpilot lane/homography prior we already use for d_seg, and — because ξ is the ONE object — the SAME offline initializer seeds BOTH d_pose AND lane-advection (task #191).

| Class | What it controls | Members |
|---|---|---|
| **H. WARM-START INITIALIZER ★PRIMARY** | seed ξ physically (offline=free) | **openpilot ego-motion → SE(3) B-spline init** (supercombo pose head / VO-SfM with known comma calib / LA-Pose inverse-dynamics / RAFT+homography) — ONE ξ seeds d_pose AND lane-advection |
| **A. REALIZATION** | how the render produces PoseNet-legible motion | warp-real-luma (PRIMARY) · FiLM-SDF (fallback) · SfMLearner photometric-warp |
| **B. LOSS-SHAPING** | how the ξ-residual is supervised | geodesic/Riemannian SE(3) loss · PoseNet feature-distillation · Fisher/natural-grad (rank-6) preconditioning |
| **C. PARAMETERIZATION / RATE** | how ξ is stored | SE(3) B-spline continuous-time (built) · 6D-continuous rotation · learnable-rank low-rank codec |
| **D. ORTHOGONALITY / OPTIMIZER** | keep pose from disturbing d_seg | disjoint-frame freeze-and-add + trunk-stopgrad · two-timescale staged freeze · Muon/orthogonal (maintenance) · ⛔ PCGrad (false friend) |
| **E. CONSTRAINT** | budget pose vs d_seg | KKT/Lagrangian pose-TUBE (hold √(10·d_pose) ≤ ε, spend capacity on d_seg) |
| **G. FOVEATION / TELESCOPIC** | spatial capacity concentration | Telescope hyperbolic-foveation warp (located; a d_seg lever, low pose value) |

---

## 2. Per-lever deep dive

### H. WARM-START INITIALIZER — the openpilot ego-motion prior [★★ PRIMARY, top-tier]

**H1. Offline openpilot/comma ego-motion → SE(3) B-spline ξ init [★★ the highest-EV pose lever, and it is DUAL-AXIS].**

This is the pose-analog of the openpilot lane/homography prior we already use for d_seg — elevated to PRIMARY per operator 2026-07-02 (it is NOT a footnote).

**The pipeline (all compress-time = OFFLINE = FREE, exactly like training):**
1. **Estimate ξ_ego(t) from `upstream/videos/0.mkv` offline** by ANY of (rank internally by robustness on THIS footage):
   - **(a) openpilot supercombo pose head** — supercombo contains a posenet used for velocity/ego-motion estimation from images (comma blog; ONNX at `MTammvee/openpilot-supercombo-model`; `commaai/openpilot` `selfdrive/modeld`). Run offline on the YUV frames → 6-DOF ego-motion per frame.
   - **(b) classical / learned VO-SfM with the KNOWN comma calibration** (fx=fy=910, cx=582, cy=437, h=1.22, RPY; dominant ground plane): RAFT/flow → essential-matrix or ground-**homography decomposition** → ξ; OR direct VO — **DPVO** (Deep Patch VO, NeurIPS 2023), **DROID-SLAM**, **OpenVO** (2602.19035, real-scale monocular DASHCAM ego-motion), **ORB-SLAM3 / COLMAP** (classical, known intrinsics). `raft_pose.py`/`raft_radial_pose.py` already in-tree.
   - **(c) LA-Pose inverse-dynamics latent-action encoder** (2604.27448, located below) — its Stage-1 encoder IS a frame-pair ego-motion extractor; run offline as another ξ estimator.
2. **Initialize the witness SE(3) B-spline ξ** (`tac.lie.se3_bspline`) with the estimated trajectory → **warm-start warp-real-luma so it carries −94% from step 0** (no cold-start).
3. **Train ONLY the affine-calibration-to-contest-PoseNet + a fine residual.** By **LDM Thm 1**, the physical ξ ↔ contest-PoseNet-6-vector is identifiable **up-to-affine** → the residual to learn is a **6-DOF affine calibration + a small nonlinearity**, NOT ξ-from-scratch. This is a tiny, well-conditioned, fast-converging problem.

**Is 0.mkv's ego-motion cleanly recoverable? — YES, ideal conditions.** Dashcam + KNOWN intrinsics (comma EON fx=910, cx=582, cy=437) + KNOWN camera height h=1.22 (gives metric scale via the ground plane) + dominant ground plane (robust homography decomposition) + dominant forward motion (dim-0 = 99.8%, the well-conditioned VO regime). The classic monocular-VO failure modes (unknown calibration, rotation-only degeneracy, texturelessness) are largely ABSENT → recoverability is HIGH.

**rule-118 cleanliness — CONFIRMED clean.** The estimator (supercombo/VO/SfM/LA-Pose net) runs OFFLINE at compress-time = FREE external code/tools (the "compile the generator" discipline). Ship **only the tiny ξ** (counted, ~O(10-48) B-spline DOF = hundreds of bytes; ξ IS video-derived so it CORRECTLY goes in `archive.zip`). Decode-time warp = **generic numpy oracle** `tac.lie._se3_numpy` + the ground homography from stored ξ (free, deterministic). **Do NOT ship supercombo weights or ANY VO-net weights** — those are large video-derived artifacts (counted AND unnecessary): only their tiny OUTPUT ξ ships. This is exactly the FREE-estimator / COUNTED-sufficient-statistic boundary. ✅

**UNIFICATION — ONE prior, BOTH axes (the value multiplier).** Because ξ is the ONE object (the vector field that advects the whole Morse-Smale complex), this SINGLE offline initializer seeds BOTH: **(i) d_pose** (warp-real-luma frame0 by ξ) AND **(ii) the lane/separatrix advection** (Wave-F L1: undo the ξ-flow → store the STATIC world-frame complex once + ξ → d_seg-lanes) = the full **openpilot-seeded witness (task #191)**. One physical prior, both contest terms. This is WHY H1 outranks a pure-pose lever: its EV is d_pose **plus** the d_seg-lane base coordinate.

**Ranking scorecard:** d_pose-reduction **HIGH** (−94% warm-start from step 0 + residual shrinks to affine-cal) × warm-start-**stability VERY HIGH** (start at the physical ξ → sidesteps the Muon+weak-pose cold-start divergence — the single biggest training-robustness win) × **byte near-free** (ship only ξ) × **seg⊥ UNIFIED** (same ξ advects lanes — beyond orthogonal) × **decode-legal CLEAN**. **Verdict: PRIMARY, co-#1 with D1.** It is the FOUNDATION that makes warp-real-luma + the trained residual a warm-start-stable, dual-axis problem. Not competing with D1/D2/E1 — it is the warm-start those training structures wrap around. Build on pose-memory line 27 + tasks #158/#191/#145.

### A. REALIZATION levers

**A1. Warp-real-keyframe-luma by ξ [PRIMARY — the incumbent, SfMLearner-grounded].**
Math: synthesize the pose channel as `frame0` advected by the ground homography `H(ξ)` (K: fx=910, cx=582, cy=437, h=1.22, RPY) so PoseNet sees a real optical-flow field. This IS the SfMLearner/monodepth2 view-synthesis warp (Zhou 2017; Godard 2019) run in reverse: instead of learning pose from photometric loss, we STORE ξ and warp, so PoseNet reads it back. Mechanism: −94% d_pose by construction, no training. Bytes: ξ only (near-free). Orthogonality: PRESERVED exactly (frame0 seg-free). **This is the realization SOTA for a non-photorealistic witness. Nothing below beats it as the realization — they improve its residual, its rate, or its orthogonality.**
- OSS: `SfMLearner` (tinghuiz), `monodepth2` (nianticlabs), `SC-SfMLearner` (geometric-consistency loss). Auto-mask stationary pixels (monodepth2) ≈ our hood-identity stratum.

**A2. FiLM-conditioned SDF render [FALLBACK].** Condition the SDF render on the 6-vec via FiLM. Cheaper-if-it-works but carries the read-back OOD risk (flat render may be OOD for FastViT). Verdict from #206: warp-real-luma preferred; FiLM kept as the residual-closing head on top of the warp (dxi), not the sole carrier. **Complements A1** as the trained residual.

**A3. SfMLearner learned-depth photometric warp [PARTIAL SUBSTITUTE for the FiLM residual].** Warp by pose+DEPTH (not pose+planar-homography) → handles non-ground structure (cars, poles) the homography flattens. Cost: adds a depth field (bytes + a second head). Verdict: **the ground-homography already handles the dominant ground-plane flow; depth is worth it ONLY if the residual after homography-warp is dominated by out-of-plane parallax** — measure the post-warp residual structure first. Lower priority than the FiLM/dxi residual.

### B. LOSS-SHAPING levers

**B1. Geodesic / left-invariant Riemannian SE(3) loss on ξ [COMPLEMENT — condition the residual].**
Math: instead of naive component-wise MSE on the (ω,t) 6-vector, penalize the squared Riemannian geodesic distance on SE(3) with a left-invariant metric, which COUPLES rotation+translation (respects SE(3)=SO(3)⋉ℝ³) and removes the manual β rotation-vs-translation weight (geomstats; "Loss it right" 2401.05396; DPC-Net 1709.03128). Riemannian SGD on SE(3) already in-tree (`riemannian_pose_optimizer.py`, Bonnabel 2013). Mechanism: better-conditioned ξ-residual optimization, fewer local minima than Euler/quaternion. Bytes: 0 (training-only). Orthogonality: preserved. **CAVEAT (important):** the AUTHORITY metric is 6-vec MSE in frozen-PoseNet output space, NOT SE(3) geodesic. Geodesic loss on ξ helps ONLY through the (affine, LDM Thm 1) ξ→6-vec map, and PoseNet is eff-dim ~4 dominated by translation dim-0 (99.8%) → the rotation-translation COUPLING benefit is **small** (rotation dims are near-null). Verdict: **marginal complement**; cheap to add, don't over-invest.

**B2. PoseNet feature-distillation [COMPLEMENT — better target than 6-vec MSE].**
Math: instead of (or alongside) matching the 6-vec output, align the PoseNet PENULTIMATE features (or its input-Jacobian subspace) of the rendered pair to the GT pair (feature-level KD; ResKD 2006.04719; uncertainty-aware 6DoF KD 2503.13053). Already scaffolded: `kl_pose_distill.py`. Mechanism: the 6-vec output is a rank-~4 bottleneck; the penultimate carries a denser, smoother training signal → faster/stabler residual convergence. Bytes: 0. Orthogonality: preserved (only the pose channel). Verdict: **medium complement** — the feature signal is richer than the 6 scalars but the 6-vec IS the exact authority, so use feature-distill as an AUXILIARY term, never a replacement for the 6-vec residual.

**B3. Fisher / natural-gradient (rank-6) preconditioning [COMPLEMENT — cheap here].**
Math: precondition the pose gradient by the inverse Fisher of the PoseNet-output distribution restricted to the rank-6 pose subspace (natural gradient; KFAC/INGD 2312.05705). Normally NGD is intractable (full Fisher inverse), BUT here the pose signal is **exactly rank-6** (measured) → the Fisher restricted to the pose-informative subspace is a **6×6 matrix, trivially invertible**. Mechanism: steepest descent in the frozen-scorer's OWN metric (this IS the GR-unified-action Fisher metric, `project_gr_unified_action`) → the ξ-residual converges along the scorer-natural directions, un-warping the 99.8%/0.2% eigenvalue anisotropy between forward-speed and the tail dims. Bytes: 0. Orthogonality: preserved. Verdict: **strong-cheap complement** — uniquely well-suited because the subspace is tiny. Fold into B1 (natural-gradient on the geodesic loss).

### C. PARAMETERIZATION / RATE levers

**C1. Continuous-time SE(3) B-spline ξ_ego(t) [THE rate mechanism — already built].**
Math: cumulative uniform-cubic B-spline on SE(3) (Kim 1995 / Sommer CVPR 2020 / Lovegrove spline-fusion BMVC 2013 / Furgale 2012). Store O(10–48) control-pose floats for the WHOLE drive instead of 600×6. Mechanism: exploits temporal smoothness of ego-motion → intrinsic ~rank-4/8 → the base coordinate of the task-space quotient; the pose-null fiber gets ZERO rate. Bytes: **near-free (the rate win)**. Dual/triple-use (pose + lane advection + temporal consistency). Orthogonality: preserved. Verdict: **adopt as the ξ store** (already `tac.lie.se3_bspline`); differentiable exp/log gives dξ/du (velocity) and control-pose grads with no custom VJP. Continuous-time survey grounding: 2411.03951.

**C2. 6D-continuous rotation representation [CORRECTNESS guard, not a mover].**
Zhou 2019 (1812.07035): Euler/quaternion are DISCONTINUOUS → 6–14× higher error + slow convergence for NN rotation regression; the 6D Gram-Schmidt representation is continuous. **We are already on the right side of this** — ξ is stored as SE(3) matrices via `se3_bspline` (exp/log), the continuous Lie parameterization. Verdict: **do NOT store ξ as raw Euler/quaternion** in any residual head; keep the matrix/6D-continuous form. Not a new lever — a guard confirming C1 is right.

**C3. Learnable-rank low-rank pose codec [RATE, already built].**
`lora_pose_v2.py` (learnable rank 1–6 gated, prune to effective rank) + `pose_delta_codec_v2.py` (arithmetic-coded deltas, 2.7×). Mechanism: data-driven effective rank of the residual head (don't hard-code rank-1). Bytes: small (the residual, ~few KB). Verdict: **adopt for the trained residual head storage**; complements C1 (spline = base, low-rank = residual).

### D. ORTHOGONALITY / OPTIMIZER levers

**D1. Disjoint-frame freeze-and-add + trunk-stopgrad [★ TOP — realizes our MEASURED free decoupling exactly].**
Math: assign frame0 = the pose carrier (warp-real-luma), frame_last = the seg carrier (SegNet reads `x[:,-1]` only). Train d_seg on the trunk; add the pose channel with a **stop-gradient on the shared trunk** so the pose loss cannot perturb the seg partition. Mechanism: our measurement (cos 5.9e-5, 99.95% pose-null, rank-6 full, frame0-seg-free asymmetry) says the two objectives are ALREADY near-orthogonal → freeze-and-add on disjoint frames makes the additive-S structure **EXACT** (pose contribution provably cannot move d_seg). Bytes: 0. Orthogonality: PERFECT by construction. Verdict: **#1 in-training lever** — grounded in our OWN witness measurement + Tier-1 frozen-scorer geometry; near-zero risk; MLX-native. Scaffold: `freezing/pose_gradient_stop_after_warmstart.py`.

**D2. Two-timescale / staged freeze [★ TOP — the curriculum wrapper for D1].**
Math: standard staged fine-tune (two-stage FT survey): converge d_seg (trunk) → freeze trunk → train pose-only (frame0 warp-residual) at a reduced LR (0.1× base), OR co-train with per-group LR (pose head base LR, trunk 0.1×). Mechanism: pose is smooth/low-dim → it converges fast on a frozen trunk; staging prevents the weakly-driven pose term from perturbing the d_seg basin (the Muon+weak-pose divergence risk noted in pose-solved memory). Bytes: 0. Orthogonality: preserved. Verdict: **#1-tier — the natural curriculum for D1**; every SOTA fine-tune uses it.

**D3. Muon / orthogonalized optimizer [MAINTENANCE, already in use].**
Newton-Schulz orthogonalization (Keller Jordan 2024; bf16-stable vs Shampoo's fp32 coupled-Newton; spectral-flattening 2605.13079). Mechanism: flattens the update spectrum → stable LR. Already the witness optimizer. Verdict: **keep as maintenance**; it is not a pose-specific lever (raising update-orthogonality is a MEANS, d_seg/d_pose are the END). Note the pose-solved-memory caveat: **Muon + a weakly-driven pose term diverged** at high LR → pair Muon with D2 staging (converge trunk before turning on pose), not simultaneous high-LR pose.

**D4. ⛔ PCGrad gradient surgery [FALSE FRIEND — DO NOT USE].**
Project conflicting task gradients (Yu 2020, 2001.06782). Two independent reasons it fails HERE: (1) empirically PCGrad "alone is sometimes useful but not always," often **worse than the baseline / single-task** (NeurIPS reviews; multiple 2024–2025 replications); (2) at our measured cos ≈ **6e-5** the seg/pose gradients are already orthogonal → **there is nothing to project** — PCGrad is a no-op at best, and its renormalization can only inject noise. Verdict: **forbidden for this problem**; D1 (disjoint-frame freeze-and-add) is the correct, exact mechanism.

### E. CONSTRAINT lever

**E1. KKT / Lagrangian pose-TUBE [★ TOP — serves THE GOAL directly].**
Math: reformulate from a weighted sum `S = 100·d_seg + √(10·d_pose) + rate` to a **constrained** program: minimize d_seg (+rate) **subject to √(10·d_pose) ≤ ε** (ε ≈ 0.02, the banked pose contribution). Solve by dual ascent on a single multiplier λ (van Rozendaal 2020 "Lossy Compression with Distortion Constrained Optimization" CVPRW; augmented-Lagrangian trust-region 2025). Mechanism: **pose is near-solved and d_seg is the binding wall** → a constraint (not a fixed weight) holds pose in its cheap banked region and spends ALL remaining capacity + curriculum on d_seg. This is the operating-point-aware move: at PR106-frontier pose_avg the marginal FLIPS (pose 2.71× seg), but as a CONSTRAINT we bank pose at ε and stop chasing its marginal, redirecting to the term that actually gates sub-0.15. Bytes: 0 (a training reformulation). Orthogonality: preserved (composes with D1). Verdict: **#1-tier and mission-aligned** — the single most goal-relevant pose lever, because it turns "pose is solved" into "capacity freed for d_seg." Scaffold: `joint_admm_proximal_pose_delta.py` + `joint_admm_coordinator`.

### F. PRIOR / INIT levers — FOLDED INTO H1 (the elevated initializer)

F1 (RAFT flow → ξ), F2 (supercombo / comma2k19 ego-motion), F3 (ground-homography operator) are the **estimator options + decode operator of H1** — see §2.H. They are no longer a separate low-priority class; they ARE the primary initializer. Discipline unchanged: **warm-start / decode ONLY, NEVER the target** (the contest PoseNet is the sole authority; the sidecar target is `PoseNet(GT-pair)[:6]`, computed offline). `raft_pose.py`/`raft_radial_pose.py` in-tree.

### G. FOVEATION / TELESCOPIC lever — the two operator-flagged papers, LOCATED

**"Telescopic foveation" — LOCATED as a real paper, but it is OBJECT DETECTION, not pose.** The exact work behind our local `telescope_2026` paradigm is **Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection** (Parker Ewen, Dmitriy Rivkin, Mario Bijelic, Felix Heide; **arXiv:2604.06332**, verified by direct fetch). Method: a two-stage detector with a novel **learnable hyperbolic-foveation re-sampling layer + image transformation** that magnifies far-field regions where targets occupy a few pixels (>500 m), giving a **76% relative mAP** gain at ultra-long range at constant compute. **"Telescopic foveation" is NOT a titled pose paper** — it is our LOCAL paradigm name for this hyperbolic-foveation capacity-concentration idea (companions: Foveated-Diffusion 2603.23491, Foveated-Telepresence 2510.19848). **Evaluation for our in-training POSE:** the transferable idea is a deterministic/learnable image-space warp that concentrates resolution on a region. It is differentiable, MLX-portable, and decode-legal IF the foveation warp is deterministic-from-stored-params. **But its value for POSE is LOW** — pose is already low-dim/near-free (eff-dim ~4). Its real value is a **d_seg lever** (concentrate render/loss on the lane annulus / small far-field movables — exactly the finest-scale erasure residual). **Verdict for pose: low priority; route the hyperbolic-foveation warp to the d_seg axis, not pose.** `lapose_foveation_atoms.py` in-tree.

**"LA-pose" — LOCATED and identified precisely (do NOT guess — here is the real cite).** **LA-Pose: Latent Action Pretraining Meets Pose Estimation** (Zhengqing Wang, Saurabh Nair, Prajwal Chidananda, Pujith Kachana, Samuel Li, Matthew Brown, Yasutaka Furukawa; **arXiv:2604.27448**, CVPR 2026; project page https://la-pose.github.io/; **no public code repo found**). It is **latent-action** (not Lie-algebra, not look-around). Exact method: **Stage 1** — an inverse–forward-dynamics model learns **latent actions** from consecutive driving-video frames via a self-supervised inverse-dynamics objective (predict future tokens; Genie-style); a latent action = a learned representation of the camera motion between two consecutive frames (separates translation-speed / rotation-direction with NO supervision). **Stage 2** — a lightweight head on the pretrained encoder predicts relative translation, rotation (quaternion), FoV, and metric scale from the latent action; fine-tuned on limited 3D labels. Result: **>10% higher pose accuracy than feed-forward SOTA on Waymo/PandaSet with orders-of-magnitude less labeled data**. **Evaluation for our in-training pose:** (1) It is a **pretraining paradigm for a pose HEAD** — **NOT an in-training lever for us** (we do not train a pose head; the frozen contest PoseNet IS the authority). (2) Its Stage-1 inverse-dynamics encoder IS a frame-pair ego-motion extractor → a **candidate OFFLINE ξ-estimator for H1(c)** (composition point, offline=free, ship only ξ). (3) Deep insight: LA-Pose's "**latent action = camera motion between two frames**" INDEPENDENTLY VALIDATES our "**pose = the screw twist ξ**" framing — an external SOTA converged on the same object. **Verdict: does NOT beat/replace our stack; it VALIDATES it and OFFERS an offline ξ-estimator option; do not fold the pretraining pipeline itself.**

---

## 3. Ranking

Score = (expected d_pose reduction) × (MLX/w_pose>0 training-compat) × (byte cost, lower=better) × (seg-orthogonality preserved) × (deterministic-decode-legal). "Beats/Complements" is relative to the incumbent **warp-real-luma + FiLM/dxi residual**.

| Rank | Lever | Class | Δd_pose mechanism | Train-compat | Bytes | Seg⊥ | Legal | vs incumbent |
|---|---|---|---|---|---|---|---|---|
| **1** | **★ openpilot ego-motion → SE(3) B-spline ξ INIT** | H1 | −94% warm-start from step0 + residual→affine-cal; **DUAL-AXIS** (seeds d_seg-lanes too) | ✅ offline/free | ~free | **UNIFIED** | ✅ clean | **COMPLEMENT — the warm-start FOUNDATION (co-#1)** |
| **1** | **Disjoint-frame freeze-and-add + trunk-stopgrad** | D1 | makes additive-S EXACT; pose can't move d_seg | ✅ MLX-native | 0 | **perfect** | ✅ | **COMPLEMENT (enables clean composition)** |
| **2** | **KKT / Lagrangian pose-TUBE** | E1 | banks pose at ε → frees capacity for d_seg (the wall) | ✅ dual-ascent | 0 | ✅ | ✅ | **COMPLEMENT (mission-critical)** |
| **3** | **Two-timescale staged freeze** | D2 | fast pose convergence on frozen trunk; no basin damage | ✅ | 0 | ✅ | ✅ | **COMPLEMENT (curriculum for #1)** |
| 4 | SE(3) B-spline ξ store (built) | C1 | near-free rate; dual-use | ✅ built | ~free | ✅ | ✅ | COMPLEMENT (the rate half; the store H1 inits) |
| 5 | Fisher / natural-grad rank-6 precond | B3 | scorer-natural residual descent; cheap (6×6) | ✅ | 0 | ✅ | ✅ | COMPLEMENT (residual conditioning) |
| 6 | PoseNet feature-distillation | B2 | denser target than rank-4 6-vec bottleneck | ✅ built | 0 | ✅ | ✅ | COMPLEMENT (auxiliary) |
| 7 | Geodesic/Riemannian SE(3) loss | B1 | couples R+t, no β; but authority is 6-vec MSE | ✅ built | 0 | ✅ | ✅ | COMPLEMENT (marginal — rot dims near-null) |
| 8 | Learnable-rank low-rank codec (built) | C3 | data-driven residual rank | ✅ built | small | ✅ | ✅ | COMPLEMENT (residual store) |
| 9 | Muon / orthogonal optimizer (in use) | D3 | stable LR; maintenance | ✅ in use | 0 | n/a | ✅ | NEUTRAL (pair with D2, not solo high-LR) |
| 10 | SfMLearner learned-depth warp | A3 | handles out-of-plane parallax | ✅ | +depth bytes | ✅ | ✅ | CONDITIONAL (only if post-homography residual is parallax-dominated) |
| 11 | Telescope hyperbolic-foveation warp (located) | G1 | low pose marginal (pose already low-dim) | ✅ | small | ✅ | ✅ | LOW for pose — route to d_seg (lane annulus) |
| 12 | LA-Pose inverse-dynamics encoder (located) | H1(c) | offline ξ-estimator option; validates ξ=latent-action | ✅ offline | ~free | ✅ | ✅ | COMPLEMENT (an H1 estimator; not a head-trainer for us) |
| 13 | SE(3)-equivariant pose head | — | frozen PoseNet ⇒ equivariance N/A to scorer | — | — | — | — | **NOT APPLICABLE (we don't train a head)** |
| ⛔ | **PCGrad gradient surgery** | D4 | no-op at cos 6e-5; often worse than baseline | — | — | — | — | **FORBIDDEN (false friend)** |

---

## 4. TOP levers to fold into the #205 witness pose config

0. **★ openpilot ego-motion → SE(3) B-spline ξ INITIALIZER (H1) [do this FIRST — it de-risks everything below].** Estimate ξ_ego(t) offline from `0.mkv` (supercombo pose head / VO-SfM with known comma calib / RAFT+homography / LA-Pose inverse-dynamics) → init `tac.lie.se3_bspline` → warp-real-luma carries −94% from step 0. Then train only the affine-cal-to-PoseNet + fine residual (LDM Thm 1). rule-118 clean (offline estimator free; ship only ξ; generic numpy decode). **DUAL-AXIS:** the same ξ seeds d_seg-lane-advection (task #191). Highest EV: it makes the pose problem a warm-start affine-cal instead of a cold-start divergence risk, AND seeds the d_seg lane base coordinate.
1. **Disjoint-frame freeze-and-add + trunk-stopgrad (D1).** frame0 = warp-real-luma pose carrier; frame_last = seg carrier; stop-gradient the shared trunk under the pose loss. Realizes the MEASURED free decoupling EXACTLY (cos 6e-5, 99.95% pose-null, frame0-seg-free). Near-zero risk, MLX-native, 0 bytes. The structural backbone that makes "warp + trained residual" a clean composition, not a tug-of-war.
2. **KKT / Lagrangian pose-TUBE (E1).** Constrain √(10·d_pose) ≤ ε≈0.02 via dual ascent; minimize d_seg (+rate) subject to it. Because pose is near-solved and **d_seg is the binding wall**, this is the single most GOAL-aligned pose lever — it converts "pose solved" into "capacity + curriculum freed for the term that gates sub-0.15." van Rozendaal-2020-grounded.
3. **Two-timescale staged freeze (D2) + Fisher-rank-6-preconditioned geodesic residual (D2∘B3∘B1).** Converge the trunk on d_seg → freeze → train the frame0 warp-residual at 0.1× LR with a natural-gradient (6×6 Fisher) step on a left-invariant SE(3) geodesic loss. Fast, stable closure of the warp's residual; the rank-6 natural gradient is uniquely cheap here.

**Prerequisite for all:** MLX-native differentiable YUV6 active (`mlx_yuv6_primitive_parity.py`) so PoseNet gradients aren't severed.

## 5. Does anything BEAT warp-real-luma + FiLM/dxi?

**Honestly, no lever BEATS it as the REALIZATION.** Warp-real-luma makes PoseNet read real motion BY CONSTRUCTION — it is the strongest realization for a non-photorealistic witness (SfMLearner/monodepth2-grounded, #206-measured −94%). The **openpilot ego-motion initializer (H1)** does not beat it either — it is the **warm-start that MAKES it carry −94% from step 0** (and uniquely also seeds the d_seg-lane axis). Every ranked lever above **complements** the warp:
- D1/D2/E1 change HOW/WHEN/UNDER-WHAT-CONSTRAINT we train around it,
- B1/B2/B3 change how the residual is SUPERVISED,
- C1/C3 change how ξ is STORED,
- F warm-starts it.
The only near-substitute for the FiLM/dxi RESIDUAL is A3 (SfMLearner learned-depth warp), and only if the post-homography residual turns out to be out-of-plane-parallax-dominated (measure first). **Recommendation:** keep warp-real-luma as realization; wrap it in D1+D2+E1; supervise its residual with B1∘B3(+B2); store ξ via C1(+C3).

## 6. Honest risks + forbidden

- **B1 geodesic-loss gain is likely marginal** — authority is 6-vec MSE in frozen-PoseNet space, and PoseNet is eff-dim ~4 dominated by translation dim-0 (99.8%), so the R+t coupling benefit is small. Cheap to add; don't over-invest.
- **The 3.4e-5 pose magnitude is ANCESTOR-RGB, NOT witness-validated.** Warp-alone = d_pose 10.53 (term 10.2). The ~0.018 pose term REQUIRES the trained residual (reachable per rank-6, UNMEASURED on witness). #221 (w_pose>0) measures it. Do not cite 3.4e-5 as settled for the witness.
- **Muon + weak-pose high-LR diverges** — always stage (D2) so the trunk converges before pose turns on; never simultaneous high-LR pose with Muon.
- **⛔ FORBIDDEN:** PCGrad (D4, false friend). comma2k19/RAFT/supercombo as the pose TARGET (F — init/prior ONLY; the contest PoseNet is the sole authority). SE(3)-equivariant heads (N/A — we don't train the scorer). Storing ξ as raw Euler/quaternion (C2 guard — keep the continuous Lie form). Any score claim from a non-`upstream/evaluate.py` surface (MPS/proxy — all rows here are `[prediction]`).
- **Foveation-for-pose is low-value** — pose is already low-dim; route foveation capacity to d_seg (the lane annulus).

## 7. OSS to draw from (all clean-license, reference-only)

- **Realization/warp:** `tinghuiz/SfMLearner`, `nianticlabs/monodepth2` (auto-mask stationary), `JiawangBian/SC-SfMLearner-Release` (geometric consistency).
- **SE(3) B-spline / continuous-time:** Sommer CVPR-2020 derivation (already re-implemented fresh in `lie/se3_bspline.py`; basalt is MPL-2.0 — do NOT copy), `306327680/SE3-pose-interpolation-using-bspline` (reference math only).
- **Geodesic/Riemannian:** `geomstats` (SE(3) left-invariant geodesic loss + `geodesic_regression`), DPC-Net (1709.03128).
- **Rotation continuity:** Zhou 2019 6D representation (1812.07035) — reference for the residual-head param.
- **Optimizer:** `KellerJordan/Muon` (in use), natural-gradient KFAC/INGD (2312.05705) for the rank-6 restriction.
- **Constrained RDO:** van Rozendaal 2020 CVPRW (distortion-constrained lossy compression) — the pose-tube pattern.

## 8. Wire-in / unified-Lagrangian hooks (per CLAUDE.md 6-hook discipline; `research_only`)

1. **Sensitivity-map:** N/A-emit-now — this survey ranks levers; the per-lever Δd_pose/byte marginals are `[prediction]`, to be filled by #221's measured rows (feeds `tac.sensitivity_map` then).
2. **Pareto constraint:** ACTIVE-as-design — **E1 (pose-tube) IS a Pareto-constraint reformulation** (√(10·d_pose) ≤ ε as a feasible-set boundary); to be added to `tac.pareto_*` when #205 wires it.
3. **Bit-allocator:** ACTIVE-as-design — C1/C3 (SE(3) B-spline + low-rank residual) are the pose-stream bit-allocator primitives (~5KB cap known).
4. **Cathedral autopilot dispatch:** N/A (`research_only`; no archive-deployable artifact here).
5. **Continual-learning posterior:** this ledger + the #221 measured rows are the anchors the next pose-config decision consumes.
6. **Probe-disambiguator:** the A1-vs-A2-vs-A3 realization choice + the "is the post-homography residual parallax-dominated?" question are the live disambiguators → resolved by #206/#221 measured-through-R d_pose, never by proxy.

**Council mission-contribution:** `frontier_breaking` — the TOP-3 (D1+E1+D2) are the pose-side enablers that let the witness spend its full budget on the binding d_seg wall toward sub-0.15. All MEANS; the END is a byte-closed exact row below 0.19110.

---

## Sources

- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019 — https://arxiv.org/abs/1812.07035
- *Loss it right: Euclidean and Riemannian Metrics in Learning-based Visual Odometry* — https://arxiv.org/pdf/2401.05396
- *Computing CNN Loss and Gradients for Pose Estimation with Riemannian Geometry* — https://arxiv.org/pdf/1805.01026
- geomstats — https://arxiv.org/pdf/1805.08308
- DPC-Net: Deep Pose Correction — https://arxiv.org/pdf/1709.03128
- Sommer et al., *Efficient Derivative Computation for Cumulative B-Splines on Lie Groups*, CVPR 2020 — https://openaccess.thecvf.com/content_CVPR_2020/papers/Sommer_Efficient_Derivative_Computation_for_Cumulative_B-Splines_on_Lie_Groups_CVPR_2020_paper.pdf
- Lovegrove et al., *Spline Fusion*, BMVC 2013 — https://www.bmva-archive.org.uk/bmvc/2013/Papers/paper0093/paper0093.pdf
- *Continuous-Time State Estimation Methods in Robotics: A Survey* — https://arxiv.org/pdf/2411.03951
- Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad) + NeurIPS reviews — https://arxiv.org/pdf/2001.06782 · https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Review.html
- Keller Jordan, *Muon* — https://kellerjordan.github.io/posts/muon/ ; *Spectral Flattening Is All Muon Needs* — https://arxiv.org/pdf/2605.13079
- *Structured Inverse-Free Natural Gradient Descent (KFAC/INGD)* — https://arxiv.org/html/2312.05705v3
- van Rozendaal et al., *Lossy Compression with Distortion Constrained Optimization*, CVPRW 2020 — https://openaccess.thecvf.com/content_CVPRW_2020/papers/w7/van_Rozendaal_Lossy_Compression_With_Distortion_Constrained_Optimization_CVPRW_2020_paper.pdf
- SfMLearner (Zhou 2017) / Monodepth2 (Godard 2019) / SC-SfMLearner — self-supervised photometric-warp pose lineage
- **LA-Pose** — Wang, Nair, Chidananda, Kachana, Li, Brown, Furukawa, *LA-Pose: Latent Action Pretraining Meets Pose Estimation*, CVPR 2026 — https://arxiv.org/abs/2604.27448 · project page https://la-pose.github.io/ (no public code repo found) — **LOCATED + verified**
- **Telescope ("telescopic foveation")** — Ewen, Rivkin, Bijelic, Heide, *Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection*, arXiv:2604.06332 — https://arxiv.org/abs/2604.06332 — **LOCATED + verified (an object-DETECTION paper, not pose; "telescopic foveation" is our local paradigm name)**
- openpilot supercombo — https://github.com/commaai/openpilot (`selfdrive/modeld`) · ONNX mirror https://github.com/MTammvee/openpilot-supercombo-model · comma2k19 / supercombo posenet — https://ar5iv.labs.arxiv.org/html/2206.08176
- VO/SfM tooling for the offline ξ estimator: DPVO (Deep Patch VO, NeurIPS 2023) · DROID-SLAM · OpenVO (monocular dashcam real-scale ego-motion) — https://arxiv.org/pdf/2602.19035 · ORB-SLAM3 / COLMAP (classical, known intrinsics)
- Foveated Diffusion 2603.23491 / Foveated Telepresence 2510.19848 (companion foveation lineage — d_seg-oriented)
