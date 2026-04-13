# Current Focus -- 2026-04-13T21:30:00Z

## Scores
- **Renderer baseline (ep 12400)**: auth=1.0000 — seg=0.210, pose=0.692, rate=0.100
  - proxy at ep12400 was 0.6019 (proxy→auth gap ~0.40, calibration artifact)
- **asym_v4_supervised ep19999 [NEGATIVE RESULT]**: auth=1.7900 — seg=0.5664, pose=1.1188, rate=0.1004
  - REGRESSION: 7600 more epochs of PoseNet+RAFT supervision made things worse
  - best_score=0.6019 was never beaten → periodic checkpoint, not best
  - both seg and pose regressed vs ep12400 baseline
- **CPU postfilter best**: ~1.89 auth (dilated_h64)
- **Target**: beat Quantizr at 0.60
- **Path**: cut pose_dist from 0.0479 → 0.016 (3× reduction needed)

## Active Experiments

### GPU Lane — Asymmetric Warp Renderer (A/B comparison)

1. **asym_v4_supervised** (COMPLETE — NEGATIVE RESULT)
   - Path A: PoseNet supervision (Layer 1) + RAFT flow (Layer 2)
   - Resumed from ep12400 (renderer_best_v3.pt, proxy=0.6019)
   - auth ep19999: 1.7900 (seg=0.5664, pose=1.1188) — REGRESSION vs baseline
   - Volume: tac-asymmetric-results /asym_v4_supervised/

2. **asym_v4_raft_only** (RUNNING, Kaggle T4, kernel comma-lab-asym-warp-raft-only)
   - Path B: RAFT flow only (isolates Layer 2 contribution)
   - Resumed from ep12400 (renderer_best_v3.pt — in Kaggle dataset v3)
   - Volume: /kaggle/working/ checkpoints every 500 epochs + timeout save at 8.5h
   - Dataset: adpena/comma-lab-private-assets v3

### KILLED
- asym_v3_longer_tight: PERMANENTLY KILLED (base variant, served as seed for v4 runs)
- asym_v4_raft_only (Modal): stopped at ep13000 — replaced by Kaggle run

### Volume prerequisites (confirmed present)
- posenet_targets.bin: ✓ on tac-asymmetric-results volume
- raft_flow.pt: ✓ on tac-asymmetric-results volume (600 pairs, 0 null frames, mean=10.89px)
- renderer_best_v3.pt: ✓ in Kaggle dataset v3 (ep12400, confirmed)

## Code State
- Rounds 26-28 complete: Kaggle DX hardened
  - kaggle_asym_warp_launcher.py: slimmed to 54-line bootstrap stub
  - runner.py: RESUME_FROM support, LFS retry, double-script bug fixed, post-install verify
  - build_kaggle_kernels.py: launch_policy stripped, RESUME_FROM preamble injected
  - 34 tests, all passing
- Dataset v3 uploaded: tac-1.0.0 wheel + raft_flow.pt + posenet_targets.bin + renderer_best_v3.pt
- Kaggle kernel v1 pushed with RESUME_FROM=/kaggle/input/comma-lab-private-assets/renderer_best_v3.pt

## Paranoia Audit (2026-04-13)
- Fridrich ✓: renderer_best_v3.pt confirmed ep12400, eval_every=200 in config
- Yousfi ✓: training script has graceful timeout save (renderer_epochN_timeout.pt)
- Quantizr ✓: raft_flow.pt has 0 null frames, mean=10.89px, max peak=277px — real signal

## Kill/Promote
- Kill supervised: proxy doesn't improve past 0.6019 by ep17000
- Kill raft_only: same criteria
- Promote: any checkpoint scores < 0.60 proxy
- Target: score < 0.60 (beats Quantizr)

## Deadline
- May 3, 2026 (~20 days remaining)
