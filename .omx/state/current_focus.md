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
- CLAUDE.md: "multiple contenders → multiple paths" added as non-negotiable rule
- Deploy script: variant= param, deployment_manifest.json, Path A + B templates

## Kill/Promote
- Kill: seg > 0.8 for 200 consecutive epochs in Phase 2
- Promote: score < 0.70 proxy at any checkpoint
- Target: score < 0.60 (beats Quantizr)

## Deadline
- May 3, 2026 (20 days remaining as of 2026-04-13)
