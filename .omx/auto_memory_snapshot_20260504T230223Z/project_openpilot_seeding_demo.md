---
name: openpilot supercombo model for compress-time seeding (demo idea)
description: Use openpilot's lane/path/calibration models at compress time to seed zoom scalars, speed estimates, camera extrinsics, and depth maps. Contest-compliant. Demo-worthy for paper.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Demo Concept

Run openpilot supercombo model on the competition video at compress time:
1. Lane detection → lane marking positions → inter-frame displacement → zoom scalar init
2. Path planning → per-frame speed estimate → direct zoom scalars (no gradient opt needed)
3. Calibration daemon → refined extrinsics → better FoE position
4. Lead car detection → depth prior → depth-aware parallax in residual path

Then show: openpilot-derived zoom scalars match gradient-optimized zoom scalars
within tolerance. This validates the physical interpretation of the rank-1 PoseNet
discovery and demonstrates zero-archive-cost motion estimation.

## Contest Compliance
Unlimited compute at compress time per rules. openpilot model weights NOT in archive
(only the resulting zoom scalars or masks are). If zoom is computed at inflate time
from masks (the lane marking approach), even the scalars aren't needed.

## Paper Value
- Connects autonomous driving stack to video compression
- Demonstrates cross-domain knowledge transfer
- Shows that the semantic masks carry enough information for both
  appearance (SegNet) and motion (PoseNet) — the dual-use insight
