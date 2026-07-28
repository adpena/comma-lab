# DAG FEED — ddm_sc1 seeded scene-carrier (solve-first ledger + e_p probe FIRED n600)

**Pointer:** 0.1910828242 [contest-CPU] UNMOVED. `score_claim=false · promotion_eligible=false`.
Axis `[macOS-CPU frozen-scorer advisory]`. Base `main@d9dc77114a`.

## What landed
1. **SOLVE-FIRST LEDGER (12 DOFs):** 9/12 SEEDED/SOLVED/DERIVE-FREE; TRAINED bucket = {support-geometry
   realization (the crux, UNBUILT), 706-param quotient residual, ξ/VOP events}. Pose = TERMINAL solve,
   not trained. Memo `.omx/research/ddm_sc1_seeded_scene_carrier_20260728.md`.
2. **e_p PROBE FIRED, n600, real PoseNet** (`experiments/ddm_sc1_ep_probe.py`; receipt
   `/Volumes/VertigoDataTier/pact/ddm_sc1_20260728/ep_probe_receipt.json`): paint BOTH frames from GT
   argmax (flat comma10k-luma), through R → PoseNet → b_p; e_p = b_p − t_p_local.
   - e_p **rank-1** (SVD energy 0.9986) → AR-int5 **2,039 B** (≈9% of PR130 23,054 B; AGREES with ar1's
     independent rank-1 t_p proxy 2,064 B).
   - global-affine R² **0.384** (<0.5, dim0 ρ≈0.58): flat-luma paint carries ego-motion only PARTIALLY.
   - d_pose paint-uncorrected **1.962** (√(10·) = 4.43): paint base is pose-DEAD without a field — field
     is MANDATORY but CHEAP (~2 KB).
   - **VERDICT: pose leg feasibility-bounded at ~2 KB, MEASURED n600 — NOT the binding constraint.**
     Closes ar1 open question #2 (pose survival of realized base).

## RECALL DISCREPANCY reported (not proceeded-on)
Charter's "SEED = W_seg 0.024124510 @ 138,031B" conflates two states. RE-VERIFIED: ws3 arbitration
`selected_warm_start=W_joint` (KEEP_WJOINT); W_seg reformed-opening formulation-stopped (SEG_REGRESSION).
**Arbitrated seed = W_joint (d_seg 0.070519 @ 138,801B)**, not the best-seg W_seg (0.024 @ 138,031B).

## Seeded-start realized triple (BEFORE training; banked ws2 endpoints)
- W_joint (ARBITRATED): d_seg 0.070519 · raw d_pose 36.62 · 138,801 B · advisory S 26.28.
- W_seg (best-seg, stopped): d_seg 0.024125 · raw d_pose 146.36 · 138,031 B · advisory S 40.76.
- Solve-first alone is nowhere near the bar: pose un-terminal-solved at seed (dominates advisory S);
  near-solved seg not yet realized (crux row 10). Rate at seed already 0.0924 (25·138.8KB/37.5M).

## Descent slope + wall-clock (banked ws3, honest confound)
ws3 W_joint 4-step window: total_distortion_term_delta −0.09377/step (seg −0.00759, pose-term +0.01585).
da1 D5 pose: −1.26%/epoch, ~0.095 S left at zero bytes. **Projection to 0.172/0.15 is DOMINATED by the
UNBUILT support-geometry realization (row 10), NOT descent-step count** — the ws3 slope is on the
advisory objective with RAW un-terminal-solved pose, so it cannot be projected to the shipped-S bar.

## Next (named, not fired — one-n600-slot spent on e_p)
`tools/launch_ddm_joint_descent.py --bounded-smoke --resume-from <W_joint receiver-closed archive>`
(memory-preflight + resume mandatory). Binding lever = BUILD the range(A)-only, gauge-fixed,
coarse-quantized contour-support realization carrier (row 10 / ar1 crux #1), not another descent window.
