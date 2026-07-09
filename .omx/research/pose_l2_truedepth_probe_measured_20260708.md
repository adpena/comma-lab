# L2 TRUE-DEPTH POSE PROBE — the true-depth-flow formulation is FALSIFIED, MEASURED (2026-07-08)

**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE` · **$0, CPU-torch, read-only** (crucible run-1 pid 63069 +
run dir UNTOUCHED; EMA snapshot-copied; NO launch/train/paid/GPU/MLX). **Pointer contest-CPU 0.19110
UNMOVED — MEANS.** Checkpoint = crucible **run-1** EMA (`levelset_witness_ema_mlx.npz`, ep200, n_pairs=600,
params=117527, self_orient, w_pose=1.0). Positive control reproduced (`d_pose([gt_f0,gt_f1])` = 5.8e-12,
n24 → instrument trusted). Harness `pose_l2_truedepth_probe.py` REUSES `pose_mladder.py` /
`train_witness_realized_through_R_mlx` (twr) exactly: same `renders_n24.npz` witness renders, same
`gt_n24.npz`, same through-R frozen CPU-torch PoseNet authority, same `xi_from_pose_calibration`
(S_T=0.16, S_R=1.0, pitch=0.02) as A0. **n=24 target (n8 confirms; n600 OWED).**

**Depth model (full online authority, provenance recorded):** `depth-anything/Depth-Anything-V2-Small-hf`
(HF revision `5426e4f0f36572d16453bbda7a8389317b1bef99`, 24.8 M params, transformers 5.13.0), inverse-depth
output, **scale+shift aligned to the calibrated ground plane** (linear disparity fit `s·i+b ≈ 1/Z_plane` on
GT-Road pixels below horizon; d=1.22 m, K focal 910 pp (582,437)). Ground-aligned depth is sane: road
Z≈2.3 m (r700) / 8.5 m (r550), sky clamped 400 m. Flow = the true-depth parallax the plane homography
misses: δ(p)=proj(K(R·Z_true·ray+t)) − proj(K(R·Z_plane·ray+t)) with the SAME (R,t)=exp_se3(ξ) as the
homography → **δ=0 on-plane, = true parallax off-plane** (median magnitude **3.91 px** native).

STORES CONSULTED: `pose_stratified_texture_probe_measured_20260708` (A1T falsification + triangulation) ·
`pose_mladder_depthwarp_measured_20260708` (A0 1.685 / A2 1.486 / A2+ 1.223; the harness) ·
`pose_taskspace_native_morse_smale_depth_warp_design_20260708` (§0c consistent-generated-pair; §5 R1
appearance risk; the M1/M2/M3 ladder this probe completes) · `pose_representation_deepresearch` (560b16634,
warp-model root cause) · CLAUDE.md L80 class order (MEASURED).

---

## HEADLINE — the true-depth-flow formulation is **FALSIFIED** (verdict_scope: FORMULATION / this off-the-shelf mono-depth carrier + run-1 ckpt). Decision row **2→3**: the wall is APPEARANCE, not flow-model crudeness. **Do NOT buy paid stored-depth (Option-A).**

Two independent, mutually reinforcing results, both on the **validated inverse-warp apparatus** (base =
exact homography inverse warp, bit-identical to `warp_frame0`, reproducing the self-fit **by
construction**):

1. **TRUE dense depth does NOT beat the plane homography — it slightly HURTS.** On real luma, the plane
   homography self-fit already reaches d_pose **0.878** (n24 median); adding the true mono-depth parallax
   field gives **1.296** — *worse*, net-neutral-to-harmful per-pair (helps 2–3 pairs, hurts ~15). The last
   untested flow-refinement (a full per-pixel true-depth warp, the thing A1T's synthesis said was "owed")
   is measured and it is **not the fix**. Flow is NOT the bottleneck on real content.
2. **Witness (task-space) luma is pose-blind (~167) regardless of flow.** Warping the cartoon witness
   render — by the plane homography (166.8) OR by true depth (171.8) — is **invisible** to PoseNet (no
   texture to advect → reads the zero-motion null). The gap between the real-luma self-fit (~0.9) and the
   deployable store-nothing carrier (~1.7–1.8, run-1) is **APPEARANCE**, not geometry.

**Therefore the pose wall is photometric CONTENT, not flow-model crudeness.** The cure is a dedicated
**joint pose-descent RUN where the render co-adapts** pose-readable content (R1-class, #238) — the only
measured path to low pose — NOT a paid stored-true-depth Option-A, which this $0 probe **falsifies as a
lever before any spend**.

## THE MEASURED TABLE (n24 median, warp,src ordering = deployment self-fit; `[macOS-CPU advisory]`)

| arm | frame0 luma | flow model | d_pose median (warp,src) | mean | (src,warp) | note |
|---|---|---|---:|---:|---:|---|
| positive control | gt_f0/gt_f1 real pair | — | **5.8e-12** | — | — | apparatus valid |
| INVHOMOG self-fit | real | homography (exact) | **0.878** | 20.2 | 8.61 | the real-luma flow reference |
| **HPLAN_REAL** (validity gate) | real | homography (δ=0) | **0.878** | 20.2 | 8.61 | == INVHOMOG (bit-parity) ✓ |
| **L2_REAL** | real | **TRUE mono-depth** | **1.296** | 23.6 | 11.9 | true depth ≈ / slightly WORSE than plane |
| HPLAN_WITNESS | witness | homography | **166.8** | 165.5 | 170.2 | cartoon → pose-blind |
| **L2_WITNESS** | witness | **TRUE mono-depth** | **171.8** | 170.0 | 173.1 | true depth does NOT rescue witness |
| RAFT ceiling (INVALID) | real | raft_small backward | 187.9 | 190.4 | — | half-res flow degenerate (mag 0.49 px) — see caveat |

Baselines (prompt): real pair ~0 ✓; homography self-fit ~2.5 (n8 anchor; **n24 median 0.878** here, cleaner
n) ; flat A0 1.685.

**Per-pair distribution (bimodal, n24):** HPLAN_REAL — 15/24 pairs `< 2`, 6 in `[2,20)`, **3 catastrophic**
(97.6, 169.1, 184.1). L2_REAL — 14/24 `< 2`, 7 in `[2,20)`, **3 catastrophic** (162.6, 168.6, 189.4). The 3
catastrophic pairs fail for BOTH plane and true depth (homography self-fit itself breaks there — large/ill-
conditioned motion, not a depth deficiency). True depth does not touch them. The median is set by the
working pairs; the heavy tail (mean ~20–24) is the self-pair synthesis failing on hard-motion pairs.

## WHICH DECISION ROW FIRED (prompt table)

| L2-REAL | L2-WITNESS | prompt verdict | fired? |
|---|---|---|---|
| ≪1 | ≪1 | flow was everything → GREENLIGHT paid Option-A + depth-bytes frontier | **NO** |
| ≪1 | high | true flow needs real appearance → witness needs photometric content / joint-descent (#238) | **~YES** (real self-fit 0.88 ≪ deployable 1.7; witness 172 high) |
| high | — | PoseNet scene-dependent beyond flow → joint-descent or pose-blocked | **YES** (L2 doesn't beat plane; tail floors high) |

**Fired: row 2 sharpened by row 3.** The two rows agree on the actuator: **joint pose-descent, not stored
depth.** The row-2 reading (appearance is the axis) and the row-3 reading (true flow doesn't help beyond the
plane) are the SAME conclusion from two directions — the store-nothing/witness geometric carrier cannot
carry pose; a co-adapting render (or real photometric content) is required.

## WHY TRUE DEPTH DOESN'T HELP (mechanism, reconciled with the ladder)

- **Off-plane parallax mass is tiny (~0.5%, Rung-0) AND the plane homography is already a near-perfect flow
  for the median real-luma pair (0.878).** The depth correction δ (median 3.91 px) is real and applied
  across the frame, but it (a) can't fix the 3 catastrophic hard-motion pairs, and (b) on the easy pairs the
  homography is already at the disocclusion/appearance floor, so the mono-depth correction only adds noise
  (imperfect ground-plane alignment sprays spurious parallax onto the road) → net slight regression.
- This CLOSES the reformulation A1T left open ("correct scene flow needs the true per-clip DEPTH field,
  which the geometric carrier does not have"). The true depth field, supplied by off-the-shelf SOTA-small
  mono-depth + ground-plane metric alignment, is now MEASURED and it **does not collapse d_pose**. The
  binding constraint was never the flow model's fidelity — it is the pair's photometric content.

## DISOCCLUSION / HOLES (honest accounting)

The apparatus is the **inverse** homography warp (base covers every target pixel → **hole-free by
construction**; off-frame source → persist fallback = src[target], the same non-gameable accounting as
`warp_frame0`). This is why HPLAN reproduces the homography self-fit exactly. (A first attempt used a
FORWARD scatter warp; it produced 37–42% disocclusion holes at a ~5.5 m forward step and could NOT
reproduce the homography floor — invalid apparatus, discarded. The inverse-warp pivot is what makes the L2
numbers load-bearing.) The δ parallax is added as a source-coord offset sampled at the homography source,
so off-plane content that a real f1 reveals is not fabricated — where true depth demands content absent
from f0, the warp degrades gracefully to the homography value.

## RAFT CEILING — INVALID, not load-bearing (honest caveat)

The intended "perfect measured flow" ceiling (raft_small backward flow f1→f0, inverse-sampled) returned
d_pose ~188 because half-res raft_small produced a **degenerate ~0.49 px** median flow (it found almost no
motion — a resolution/preprocessing failure, not a real result). It is discarded. **It is not load-bearing:**
HPLAN_REAL = 0.878 already IS the real-luma flow reference (the homography is a near-exact flow for the
median pair), so the "best-possible-flow on real luma" bound is already established at ~0.9. A valid dense-
flow ceiling (full-res RAFT with the weights' own transforms) is OWED but would not change the verdict.

## DEPTH-BYTES FRONTIER — NOT RUN (gated RED, means discipline)

The optional depth-degradation sweep (downsample D 8×/16×/32× → the d_pose(depth-bytes) frontier for a paid
Option-A) was gated on "L2-REAL promising." L2-REAL is NOT promising (true depth doesn't beat the plane), so
storing depth buys nothing → the frontier is moot. Not run. Do LESS but REAL.

## SYNTHESIS + verdict_scope

- **True-depth flow (off-the-shelf mono-depth + ground-plane alignment): FALSIFIED at the FORMULATION
  level** for the store-nothing depth-warp pose carrier on crucible run-1. NOT a family/paradigm kill.
- **Triangulation (now four ways):** A0T (texture×global-H) RAISES d_pose; A1T (texture×per-cell flow)
  RAISES d_pose; A2/A2+ (6/12-DOF solve) floor ~1.2; **L2 (true dense depth) does NOT beat the plane
  homography (1.30 vs 0.88) and does NOT rescue the witness (172).** Every geometric/flow lever on the
  cheap carrier is exhausted. The invariant across all four: **the wall is the pair's photometric content
  (cartoon witness = pose-blind; the deployable carrier's 1.7–1.8 vs the real-luma self-fit's 0.9), not the
  flow model.**
- **NOT refuted (named reformulations that remain open):** (a) a **dedicated JOINT pose-descent RUN** where
  the RENDER co-adapts so the warped pair carries pose-readable content (R1-class, #238 — the ONLY measured
  path to low pose; run-1's dxi already refines 1.99→1.79); (b) pose-as-budget-item + win sub-0.15 on
  d_seg+rate. **Refuted as levers:** paid stored-true-depth (Option-A), store-real-appearance (DEAD:
  rate +573, d_pose 10.4), any post-hoc warp of the fixed render.
- **Budget honesty:** the deployable carrier can't use real luma (rate-dead) and witness luma is pose-blind
  in the geometric self-pair; run-1's co-adapted floor is ~1.79 → contribution √(10·1.79) ≈ 4.2, which
  **kills sub-0.19 for any store-nothing pose carrier.** Sub-0.15 must come from d_seg + rate with pose as a
  budget item, OR from a dedicated joint pose-descent run re-validated through byte-close at n600.
- Owed before any promotable pose number: **n600 + exact-eval**; a valid dense-flow ceiling. All rungs
  `[macOS-CPU advisory]` NON-PROMOTABLE.

## TRIALITY / EQUATION

Canonical equation `morse_smale_stratified_parallax_dpose_v1` gains the MEASURED L2 advisory anchor
(`l2_truedepth_falsified_crucible_run1_20260708`): off-the-shelf true dense mono-depth (ground-plane
aligned) does NOT collapse d_pose (L2_REAL 1.296 ≈/> HPLAN_REAL 0.878, net-harmful) and does NOT rescue the
pose-blind witness (171.8); validity gate HPLAN==homography self-fit (0.878, bit-parity); the wall is
photometric appearance, not flow-model crudeness; Option-A stored-depth REFUTED as a lever, joint
pose-descent the only open path; n600 + exact-eval OWED. No DSL change (investigation/probe, no new trainer
lever fired).

## FINAL STATE

$0 CPU-torch; pid 63069 UNTOUCHED; NO launch/train/paid/GPU/MLX. **Pointer 0.19110 UNMOVED — MEANS.**
Scratch: `pose_l2_truedepth_probe.py` + `l2_n24.json` + `l2_n8.json` + `l2_depths_n24.npz` (Depth-Anything
inverse-depth cache) under the session scratchpad. Depth model HF rev
`5426e4f0f36572d16453bbda7a8389317b1bef99`; raft_small ckpt `01064c6dba73…` (degenerate, discarded).
