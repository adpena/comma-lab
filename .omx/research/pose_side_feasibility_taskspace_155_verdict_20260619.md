---
title: "POSE-SIDE feasibility probe (#155 from-scratch task-space rep) — VERDICT"
authority: "[contest-CPU advisory] — pointer UNMOVED 0.19110; no PR; $0; REAL frozen scorers, NO synthetic fixtures"
score_claim: false
promotable: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-19
verdict: "RED for a separable cheap pose code; pose is BUNDLED with the full-frame reconstruction (not a free extra cost, not a cheap extra slot)"
segment: "b0c9d2329ad1606b|2018-07-27--06-03-57/10 (comma2k19 RAV4, verified)"
producer: "experiments/probe_pose_side_feasibility_taskspace_155.py"
result_json: ".omx/research/pose_side_feasibility_taskspace_155_20260619T180012Z.json"
gt_pose_npz: "experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz"
cross_refs:
  - .omx/research/comma_openpilot_domain_tricks_20260619T035417Z.md (the prior this tests — exploit #1 P_pose)
  - "memory project_contest_source_is_known_comma2k19_rav4_segment_pose_gt_downloadable_20260619.md"
---

# Pose-side feasibility for the from-scratch task-space rep (#155): the answer key, measured

**One-line headline:** the comma2k19 GT-pose prior is REAL and verified, but the contest's d_pose
target is the FROZEN PoseNet's per-pair output — which is **dominated by irreducible per-pair NN
jitter, not the smooth physical trajectory**. A cheap standalone pose code therefore CANNOT hold
d_pose; pose is realized only through a **near-full-resolution frame reconstruction** — so for #155
pose is **bundled with the d_seg frames the rep reconstructs anyway**, not a separable cheap slot.

**Verdict:** `RED_EVEN_VECTOR_FLOOR_NEEDS_MANY_BYTES` for a separable cheap pose code; the operative
finding for #155 is the L3 reframe (pose is bundled, ~0.091 pose-term for free with a full-frame
rep). `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer UNMOVED 0.19110. $0, no GPU, no PR.

## What was measured (3 layers, all REAL frozen scorers, NO-FAKE)

### Downloaded the verified comma2k19 GT (the answer-key prior)
- Source: `commaai/comma2k19` `data/demo-00000-of-00003.parquet` `log.global_pose__frame_*` — the
  EXACT segment `b0c9d2329ad1606b|2018-07-27--06-03-57/10` (no 9 GB chunk needed; the demo parquet
  is 75 MB and contains all RAV4 segments incl. ours). Saved `comma2k19_gt_pose_raw.npz`.
- Shape verified: 1200 frames @ 20 Hz = exactly the contest's 600 pairs × 2; positions/velocities
  (ECEF, 3) + orientations (unit quaternion, 4). Physical speed **31.7 ± 0.87 m/s** (highway), and
  **extremely smooth: consecutive-frame Δspeed std = 0.0099 m/s** — corroborating the "~1-2 DOF,
  smooth" prior on the *physical* signal.

### L1 — information floor (vector-level): the PoseNet target is JITTERY, not smooth
The d_pose reference is `T = PoseNet(GT_pairs)[:6]` ∈ ℝ^{600×6} (the cached `pose` target, what
reconstructed frames must reproduce). Fitting a per-dim Chebyshev poly + noise-floor residual:

| code | bytes | d_pose | pose term |
|---|---:|---:|---:|
| 0-DOF constant mean | ~12 | 0.2636 | 1.62 |
| deg-4 poly | 60 | 0.1483 | 1.22 |
| deg-8 poly | 108 | 0.1254 | 1.12 |
| deg-16 poly | ~204 | 0.1094 | 1.05 |
| deg-32 poly | ~396 | 0.1008 | 1.00 |
| deg-64 poly | ~780 | 0.0943 | 0.97 |
| **exact per-pair vector, noise-floor quantized (full dump, thousands of B)** | ~thousands | **0.0040** | **0.20** |

**Decisive fact:** the PoseNet's `v_fwd` output has **consecutive-pair Δ std = 1.11 m/s** — larger
than the whole signal's std (1.26 m/s) — while the *physical* comma2k19 speed Δ std is 0.0099 m/s
(100× smoother). corr(PoseNet v_fwd, physical speed) = only **0.72**. So the PoseNet output is
dominated by per-pair NN measurement noise, NOT the smooth physical trajectory the prior describes.
No smooth temporal code can reproduce that jitter: even a 780-byte deg-64 poly floors at d_pose
0.094 (pose term 0.97), and even the *exact* per-pair dump (noise-floor quantized) only reaches
0.0040 — still **~12–130× worse** than the frontier's pose anchor (3.4e-4 loose / 3e-5 tight).
The 5 small dims (v_lat, v_vert, ω_roll/pitch/yaw, std 0.007–0.036) are below their physical noise
floor → ~constant (2 B each); the cost is entirely `v_fwd`'s jitter.

→ **No cheap (≤400 B) vector-level pose code holds d_pose. The pose sufficient statistic is NOT
the smooth physical trajectory; it is the jittery PoseNet output, which is not cheaply codeable.**

### L2 — realizability through frames (REAL PoseNet, the contest quantity)
Measured realized d_pose = `MSE(PoseNet(degraded_GT_pair), PoseNet(GT_pair))` over 16 real GT
pairs, sweeping carrier resolution × luma-bits × {edge-preserving bicubic, edge-killing block}:

| carrier | grid | realized d_pose | pose term |
|---|---|---:|---:|
| bicubic div16 | 54×72 | 64.0 | 25.3 |
| bicubic div8 | 109×145 | 0.222 | 1.49 |
| bicubic div4 | 218×291 | 0.0355 | 0.60 |
| **bicubic div2** | 437×582 | **0.00094** | **0.097** |
| bicubic div1 (= GT frame, sanity) | 874×1164 | **0.000000** | 0.000 |

- **Sanity passes:** the undegraded GT frame (div=1) gives realized d_pose = 0.000000 → the harness
  is exact (real PoseNet on GT == target).
- **Edges are the pose currency** (corroborates the memo): bicubic (edge-preserving) beats block
  (edge-killing) at div=2 by ~16× (0.00094 vs 0.015). But even half-res bicubic (div=2) only
  reaches 0.00094 — **2.7× the frontier's loose anchor** — and div=4 explodes to 0.036.
- **Pose needs near-full luma resolution.** There is no cheap (≤400 B) carrier that realizes
  frontier-grade d_pose; the carrier must retain near-full-resolution luma the PoseNet reads.

### L3 — frontier-recon anchor (the operative #155 finding)
The bc20 HNeRV basin (a from-scratch full-resolution frame rep) achieves, MEASURED on torch-CPU,
**d_pose = 0.000831 (pose term 0.0912)** with the SAME full-frame reconstruction that gives its
d_seg = 0.00378 (best_meta.json). This matches the bicubic-div2 L2 point exactly — i.e. the HNeRV
decoder is ~half-resolution-effective for pose.

→ **In a from-scratch rep, pose is BUNDLED with the full-frame reconstruction.** It is neither a
free extra (the rep must reconstruct near-full-res luma) nor a separable cheap slot (L1/L2 RED).
To push pose from ~0.091 toward the frontier's tight anchor (0.017), the rep must improve
**full-frame luma fidelity** — there is no pose-code shortcut.

## The decisive feasibility answer (operator's question)
> Can the d_pose-critical content be coded in ~hundreds of bytes while holding d_pose ≤ frontier?

**No.** The d_pose-critical content is the PoseNet's jittery per-pair output, not a smooth
hundreds-of-bytes trajectory. Even the exact per-pair vector dump (thousands of bytes) lands 12×
above the frontier, and realizing pose through frames needs near-full luma resolution. The
comma2k19 GT is a real, verified PRIOR (it told us the physical signal is smooth and confirmed the
camera/segment), but it is NOT a drop-in: the contest scores the frozen PoseNet on reconstructed
frames, and that target is jitter-dominated.

**What this means for #155:** do NOT design a separate cheap pose component. Pose comes bundled with
the full-frame reconstruction the rep needs for d_seg anyway, at ~0.091 pose-term "for free" at
bc20-grade. The pose axis is improved (toward the 0.017 frontier anchor) by improving full-frame
luma fidelity, not by adding a pose code. This is consistent with the frontier being rate- and
d_seg-dominated (CLAUDE.md S_floor 0.118; pose is the small term).

## Honest bounding (NO-FAKE)
- Pose is a SMALL term (~0.017 at the frontier). This probe answers whether a from-scratch rep can
  add pose CHEAPLY/standalone (no), not a frontier nudge. Pointer UNMOVED 0.19110.
- The comma2k19 GT is a PRIOR/oracle, not a drop-in — we did NOT fake d_pose=0 from a stored
  vector (the eval has no pose-vector input; verified the contest reduction in code + tests).
- L1/L2 are torch-CPU advisory (CLAUDE.md "local CPU + MLX GPU good"); the only authority is an
  exact byte-closed `upstream/evaluate.py` CPU row, which this probe does not produce or claim.
- The PoseNet-jitter finding (Δ-std 1.11 vs physical 0.0099) is the load-bearing measured fact and
  is reproducible from the cached n600 pose target + the downloaded comma2k19 GT.

## Wire-in (6-hook, per Catalog #125)
- #1 sensitivity-map: ACTIVE — the d_pose axis is bundled with full-frame luma fidelity (informs
  that pose bytes are not separable; the bit-allocator should not reserve a pose slot).
- #2 Pareto constraint: ACTIVE — pose-term floor ~0.091 at bc20-grade frame fidelity is a measured
  point on the rate↔d_seg↔d_pose surface (pose is not an independent free axis).
- #3 bit-allocator hook: ACTIVE (negative) — do not allocate a standalone pose code; allocate to
  full-frame luma.
- #4 cathedral autopilot: N/A — advisory probe, no archive-deployable artifact.
- #5 continual-learning posterior: N/A — advisory; no exact row to seed the posterior.
- #6 probe-disambiguator: ACTIVE — this IS the disambiguator between "cheap pose code" (RED) and
  "pose bundled with frames" (the operative interpretation).
