# POSE M-LADDER — Morse-Smale stratified parallax warp + pose-space solve, MEASURED (#365)

**Date:** 2026-07-08 · **Axis:** `[macOS-CPU advisory] NON-PROMOTABLE` · **$0, CPU-torch, read-only**
(run-1 pid 63069 + run dir UNTOUCHED; checkpoint SNAPSHOT-copied to scratchpad before use; NO launch,
NO training, NO paid eval, NO GPU). **Pointer contest-CPU 0.19110 UNMOVED — MEANS.** verdict_scope
tags inline. Checkpoint = crucible **run-1** EMA (`levelset_witness_ema_mlx.npz`, epoch 200, n_pairs=600,
params=117527, self_orient, w_pose=1.0). Positive control reproduced every session
(`d_pose([gt_f0,gt_f1] vs gt_poses)` = 1.2e-12–5.8e-12 ≈ 0 → instrument trusted). Harness (scratchpad
`pose_mladder.py`) reuses `tools/pose_frame0_inverse_solve_probe.py` (#249) machinery +
`experiments/train_witness_realized_through_R_mlx.py` (twr) verdict + `tac.boundary_math.warp_real_luma_frame0`
+ the new `tac.boundary_math.stratified_depth_warp`. Deployment-faithful CONSISTENT GENERATED pair
(§0c): frame1 = witness INR render; frame0 = warp(witness's OWN frame0 render); d_pose =
MSE(PoseNet6(gen_pair), PoseNet6(gt_pair)) through the frozen CPU-torch authority (NEVER MLX).

STORES CONSULTED: pose_taskspace_native_morse_smale_depth_warp_design_20260708 · pose_carrier_arms_measured_20260708
(16030e6bf) · pose_taskspace_native_review_20260708 (5711a4fdf) · pose_frame0_inverse_solve_probe_20260703T0810Z
(#249, tool 3f15daa53) · council_pose_carrier_optimal_form_symposium_20260703 (cited R1 0.0011) ·
xi_pose_coder.py / warp_real_luma_frame0.py / lie · CLAUDE.md L80 class order.

---

## HEADLINE (the honest frontier)
**The task-space witness's cheap warp pose-carrier floors at d_pose ≈ 1.2–1.7 — orders above the 0.019
target — and neither depth stratification (A2+) nor a 6-DOF pose-space solve (A2) breaks it.** Every
rung DESCENDS monotonically but SHALLOWLY; the design's core premise (depth-stratified parallax collapses
d_pose) and the extension's "A2 ≈ 0.0011" prediction are BOTH **REFUTED at the formulation level** on this
checkpoint. The binding wall is NOT recoverable off-plane parallax (Rung 0: off-plane mass ≈ 0.5%,
corr(d_pose,|t|) NEGATIVE) — it is the cartoon-pair appearance/flow-consistency versus the real
photometric pair, which no low-DOF (≤12) warp of the generated source escapes.

### The measured (d_pose, bytes) frontier
| rung | mechanism | DOF | bytes/pair | d_pose median | d_pose mean | n | scope |
|---|---|---|---:|---:|---:|---:|---|
| positive control | PoseNet(gt_pair) | — | — | 1.2e-12 | — | 8–24 | apparatus valid |
| **A0** | deterministic global ground-H warp (store-nothing) | 0 | ~1.7 (coded ξ) | **1.685** | 1.734 | 24 | reproduces prior 1.995(n8)/telemetry 1.79 |
| **A2** | per-pair 6-DOF pose-space LM/GN solve over ξ_eff | 6 | ~12 B (6 fp16) | **1.486** | 1.521 | 24 | p90 1.86, max 2.90 |
| **A2+** | + 6 off-plane affine steering DOF (ORACLE GT mask) | 12 | ~24 B | **1.223** | 1.567 | 8 | best-case ceiling |
| — free-pixel P-E (ref, #249) | full-rank free frame0 | ~N | rate-PROHIBITIVE | ~2.7e-7 | — | 3 | existence proof only, NOT shippable (#249 correction) |
| — ancestor RGB anchor | photometric recon | — | — | — | — | — | 3.4e-5 BORROWED, never witness-validated |

Same-8-pairs controlled contrast: **A0 1.580 → A2 1.362 (−14%) → A2+ 1.223 (−10% over A2)** median.
The 12-DOF solve with an ORACLE off-plane mask still cannot break ~1.2. Every number MEASURED through
the real R / frozen CPU-torch PoseNet; `[macOS-CPU advisory]`; n600 + exact-eval owed before any
promotable pose number.

---

## RUNG 0 — free companion (the root cause, bounded up front)
- **corr(d_pose, |ξ_translation|) = −0.446 (n24) / −0.676 (n8)** — NEGATIVE. d_pose does NOT rise with
  ego forward motion. The naive "plane-only warp loses forward-translation parallax → d_pose grows with
  |t|" story is **not what the data shows**; if anything larger-motion pairs are BETTER conditioned and
  near-zero-motion pairs (tiny target) are the ill-conditioned tail.
- **Off-plane finite-depth mass is TINY.** From the authoritative GT SegNet partition (matches CLAUDE.md
  L80: Road+Lane ground = 23.5%, Movable ≈ 1.6%, Undrivable ≈ 49% mostly ABOVE horizon = sky/parallax≈0,
  MyCar/hood ≈ 25% static): off-plane **AREA ≈ 2.7–2.9%**, row-weighted (1/Z) **parallax MASS ≈ 0.5%**.
  → A depth-stratified warp can only touch ~0.5–3% of the flow. **R2 (design §5: "is off-plane parallax
  mass large enough to matter?") answered LOW** — the achievable win from depth stratification is bounded
  small BEFORE any warp is built, and A2+ (below) confirms it empirically (~10%).

## RUNG A0 — plane-only control (reproduces the wall)
Deterministic store-nothing: ξ[p] = `xi_from_pose_calibration(gt_poses[p], s_t=0.16, s_r=1.0, pitch=0.02)`;
frame0 = `warp_frame0_uint8_numpy(witness_f0_render, ξ, geom)`; frame1 = witness_f1_render. **d_pose
median 1.685 / mean 1.734 (n24)** — cleanly reproduces the prior n8 store-nothing (1.995) and run-1's
trained telemetry (1.79). The ~1.7 floor is real and stable at n24.

## RUNG A2 — per-pair 6-DOF pose-space solve (THE decisive rung, per extension)
Per pair, damped Gauss-Newton over ξ_eff (6 DOF), forward-diff Jacobian on the frozen **uint8** authority
(the #249 STE-consistency key: the objective IS the deployed uint8 d_pose), line-search accept on true
d_pose. Init = A0 ξ. **d_pose median 1.486 / mean 1.521 (n24)**, p90 1.86, max 2.90 — only ~12% below A0.
**Robustness (adversarial, ruled out a conditioning artifact):** zero-init, FD 2e-2, 15 iters, low λ →
plateaus identically (1.2–2.1); the solver genuinely DESCENDS (steps accepted, d monotone) to a local min
far above 0, it is not stuck. **This REFUTES the extension's "A2 ≈ 0.0011-class" prediction.** The gap to
#249's ~0 is the parametrization: #249 reached ~2.7e-7 by solving **free pixels** (full-rank 6×N frame0
Jacobian → surjective onto the 6-dim pose output), which is **rate-prohibitive** (#249's own correction:
image-space per-pair stores ≈ 8.6 rate @ n600). A rate-cheap **6-scalar warp** of the fixed cartoon
render spans only a low-rank, appearance-constrained manifold whose reachable PoseNet outputs do NOT
contain the real-pair target. The "6→6 map steers to any target" argument fails because the warp Jacobian
is effectively rank-deficient / the reachable manifold is bounded ~1.2–2.0 from target.

## RUNG A2+ — + off-plane depth-steering DOF (the extension's key remaining test)
6 ξ_eff + 6 off-plane affine flow DOF (`stratified_depth_warp.affine_extra_flow` on the ORACLE GT
off-plane mask — best case; deployment witness-mask would be noisier). **d_pose median 1.223 / mean 1.567
(n8)** — only **−10% over A2** on the same pairs. The off-plane depth steering helps a LITTLE (consistent
with the nonzero-but-tiny 3% off-plane mass) but does **NOT** "collapse orders below the 6-DOF floor" as
predicted. **REFUTED at formulation level**: with an ORACLE partition and 12 DOF the warp family still
floors ~1.2.

## RUNG A1 — forward depth-warp (CUT per envelope priority, ceiling bounded by A2+)
A1-forward (fit the per-cell affine-inverse-depth field from real f0/f1 correspondence, no solve-to-target)
was the FIRST cut per the extension's priority. Its win is upper-bounded by A2+ (which SOLVES the off-plane
flow to the target with an oracle mask — a strict ceiling over any FIT field), so A1-forward ≤ A2+'s ~10%
improvement. Not worth building given the triple-negative. The `stratified_depth_warp` module is landed
and A1-ready (fit-field path) if a future unit wants the exact forward number.

## Rectification / geometry check (review-flagged force)
K = `[[910,0,582],[0,910,437],[0,0,1]]` — single focal, principal at image center, zero skew ⇒ **rectified
pinhole** (openpilot eon calibration; horizon row 437 matches the design's re-derivation). The pinhole
assumption HOLDS. Note A2/A2+ solve DIRECTLY against the PoseNet target, so peripheral off-plane geometry
correctness is NOT load-bearing for the measured rungs (it would only matter for A1-forward).

---

## SYNTHESIS — why pose stays a budget item on this carrier (verdict_scope: FORMULATION)
1. The wall is NOT off-plane parallax (Rung 0: mass ≈ 0.5%, negative |t| corr) and NOT the number of warp
   DOF within a rate-cheap budget (A2 6-DOF and A2+ 12-DOF-oracle both floor ~1.2–1.5). It is the
   **appearance/flow-consistency of the generated (cartoon) pair vs the real photometric pair** — exactly
   the "consistent generated pair floors at homography-model cap" finding of pose_carrier_arms_measured
   (16030e6bf), now shown to persist under a full 12-DOF oracle-depth warp re-solve.
2. Pose IS inverse-solvable to ~0 in PRINCIPLE (free frame0 pixels, #249 P-E) but that is rate-prohibitive
   and adversarial (NOT decoder-reproducible) — the NO-FAKE firewall (#6/#8) forbids shipping it.
3. **R1's cited 0.0011 (council_pose_carrier_optimal_form_symposium) is NOT reproducible by a post-hoc
   warp of this fixed render** (A2/A2+ prove it). If real through byte-close, it was reached by JOINT
   TRAINING co-adapting the RENDER itself (f0/f1 renders move so the homography-warped pair hits the pose
   target), NOT by a cheap post-hoc ξ carrier. Reconciles with the coordinator's "1.79 plateau is
   joint-training pathology" — the cure is a dedicated joint pose-descent RUN, not a post-hoc carrier;
   the crucible run-1 checkpoint (w_pose=1.0, epoch 200) simply did not descend pose (floors ~1.7).

## HONEST NEXT STEP (aimed at the exact score)
- Pose on the cheap task-space carrier stays ~1.2–1.7 (contribution √(10·1.2)≈3.5 — dominates S). Sub-0.15
  cannot come from a post-hoc warp pose carrier on this checkpoint. Either (a) a dedicated joint
  pose-descent training run (re-validate R1's 0.0011 through byte-close at n600 — the ONLY measured path
  to low pose, and it is a RUN not a carrier), or (b) accept pose as a budget item and win sub-0.15 on
  d_seg + rate. Do NOT build A1-forward (bounded ≤ A2+'s ~10%). Do NOT store real appearance (measured
  DEAD: d_pose 10.4 + rate +573, pose_carrier_arms_measured).
- Owed before any promotable pose number: n600 + exact-eval; these advisory rungs are `[macOS-CPU advisory]`.

## TRIALITY / EQUATION
Canonical equation `morse_smale_stratified_parallax_dpose_v1` REGISTERED (src/tac/canonical_equations/)
with the MEASURED advisory anchor (A0 1.685 / A2 1.486 / A2+ 1.223, n8–24, this checkpoint); n600 +
exact-eval marked OWED in the anchor notes. Prototype module `src/tac/boundary_math/stratified_depth_warp.py`
(bit-parity to the A0 authority proven, maxdiff 0). No DSL change (investigation/carrier-measurement, no
new trainer lever fired).

## FINAL STATE
$0 CPU-torch; pid 63069 UNTOUCHED; NO launch/train/paid/GPU. **Pointer 0.19110 UNMOVED — MEANS.**
Scratch: `pose_mladder.py` + `renders_n24.npz` + `a2_n24.jsonl` + `a2plus_n8.jsonl` +
`mladder_{a0,a2,a2plus}_*.json` under the session scratchpad.
