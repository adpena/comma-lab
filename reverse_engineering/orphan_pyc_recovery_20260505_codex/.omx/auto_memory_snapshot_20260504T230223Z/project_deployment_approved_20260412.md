---
name: Deployment Approved — Asymmetric Warp Renderer (2026-04-12)
description: 37 commits, 50+ bugs, 16 review rounds. Skunkworks team unanimous DEPLOY. Lightning T4 training approved.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Approved deployment (2026-04-12 late evening)

37 commits this session. 50+ bugs fixed. 16 review rounds. Zero known issues.
Skunkworks team (Yousfi, Fridrich, Quantizr, Contrarian) signed off unanimously.

### Deployment command
```bash
LIGHTNING_USER=<user> bash scripts/lightning_deploy_asymmetric.sh
```

### Training config
```
--pair-mode asymmetric --epochs 10000 --batch-size 4 --lr 2e-4
--embed-dim 6 --base-ch 36 --mid-ch 60 --motion-hidden 32
--max-flow-px 20.0 --max-residual 20.0 --gate-reg-weight 0.1
--seg-boundary 0.005 --pose-boundary 0.02 --rho-growth 1.02
--target-bytes 200000 --eval-every 200 --even-pairs-only
--device cuda --seed 42 --max-hours 48
```

### Model: 287K params, ~140KB FP4
### Architecture: AsymmetricPairGenerator (warp paradigm, Quantizr-inspired)
### Kill criteria: gate_mean>0.7@500, seg_hard>0.1@2000, pose_dist>0.5@2000

### Key decisions
- Decode on target (DALI on Lightning, no precomputed frames)
- Ship full model (renderer + motion) in archive
- Non-overlapping pairs match scorer exactly
- Fridrich 3-phase curriculum (soft→tempered→STE)
- SegNet loss on frame_t1 only
- Even-pairs-only training
