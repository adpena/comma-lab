# Current Focus -- 2026-04-13T10:00:00-05:00

## Scores
- **Official best**: 1.33 (dilated_h64)
- **Target**: beat Quantizr at 0.60
- **Projection**: 0.57 (seg=0.30 + pose=0.14 + rate=0.13)

## Active Experiments

### GPU Lane — Asymmetric Warp Renderer (primary attack vector)

1. **asym_v3_longer_tight** (RUNNING, detached Modal T4)
   - 20K epochs, batch=8, pose_boundary=0.01, no supervision layers
   - At epoch 11000/20000 as of 2026-04-13
   - Resume: /results/asym_v3_longer_tight/renderer_best.pt

2. **asym_v4_supervised** (READY TO LAUNCH)
   - Path A: PoseNet supervision (Layer 1) + RAFT flow (Layer 2)
   - resume from asym_v3_longer_tight/renderer_best.pt
   - Command: `modal run ... --tag asym_v4_supervised --variant supervised --resume-from /results/asym_v3_longer_tight/renderer_best.pt`

3. **asym_v4_raft_only** (READY TO LAUNCH)
   - Path B: RAFT flow only (isolates Layer 2 contribution)
   - Command: `modal run ... --tag asym_v4_raft_only --variant raft_only --resume-from /results/asym_v3_longer_tight/renderer_best.pt`

### Volume prerequisites (confirmed present)
- posenet_targets.bin: ✓ on volume
- raft_flow.pt: ✓ on volume

## Code State
- Round 19 complete: 4 council issues fixed (double PoseNet fwd, p1_pose_sup_weight, ego_flow reuse, logging cache)
- Round 20 complete: MED-4 DCT device fix, MED-5 n_written assert, HIGH-5 W→(W-1) flow normalization (council 4-1)
- Round 21 complete: 66 pre-existing test failures fixed → 672 passed, 0 failed
- Round 22 complete: flow_compress CUDA, even_pairs guard, warp_quality telemetry
- Round 23 complete: ego_flow.py W→(W-1) normalization
- Round 24 complete: Lightning deploy crash fix — strip/reinject paths
- Round 25 complete: .pt hardening (raft_flow/pose_targets fail-fast); Kaggle script path fix
  - tac/deploy/{modal,kaggle,lightning}/__init__.py created — wheel now discovers subpackages
  - 3 Kaggle kernel bundles committed (experiments/kaggle_kernels/asym_warp_*)
  - GPU scorer proxy: ~0.8-0.9 (asym_v3_longer_tight at ep ~11K/20K)

## Kill/Promote
- Kill: seg > 0.8 for 200 consecutive epochs in Phase 2
- Promote: score < 0.70 proxy at any checkpoint
- Target: score < 0.60 (beats Quantizr)

## Deadline
- May 3, 2026 (20 days remaining as of 2026-04-13)
