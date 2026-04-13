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
  - test_hardening: 9 missing non-postfilter variants added to exclusion set, non-postfilter skip added
  - test_tac_lossless_{arithmetic,baseline,evaluate}: 8 env-sensitive data_files assertions fixed via os.path.basename()
- CLAUDE.md: "multiple contenders → multiple paths" + 5-member non-conservative council (Yousfi, Fridrich, Contrarian, Quantizr, Hotz)
- Deploy infrastructure: deploy_config.py (provider-agnostic canonical flags)
  - Modal: imports from deploy_config, manifest saved before training
  - Lightning: variant-aware (base/supervised/raft_only), env-var injection (no shell injection)
  - Kaggle: 3 kernel specs (asym_warp_base/supervised/raft_only), launcher reads ASYM_VARIANT
  - All 8 critical/high adversarial review issues fixed (--output-dir crash, __init__.py, UPSTREAM_ROOT, PYTHONPATH, dead code, injection, uv mandate, flag dedup)

## Kill/Promote
- Kill: seg > 0.8 for 200 consecutive epochs in Phase 2
- Promote: score < 0.70 proxy at any checkpoint
- Target: score < 0.60 (beats Quantizr)

## Deadline
- May 3, 2026 (20 days remaining as of 2026-04-13)
