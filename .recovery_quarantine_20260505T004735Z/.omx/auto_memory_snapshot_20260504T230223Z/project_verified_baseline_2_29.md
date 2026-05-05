---
name: VERIFIED contest-CUDA baseline 2.29 — first reproducible-from-saved-artifacts score
description: 2026-04-27 verified contest-CUDA via experiments/contest_auth_eval.py on Vast.ai RTX 4090 (Oregon, NVDEC-good). Rebuilt full-res 384x512 CRF=50 archive from saved components (renderer + poses + freshly-generated masks). Beats comma baseline_fast (4.40) by 1.92 points.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**The first reproducible contest-CUDA score we've measured this session, anchored to deterministic artifacts.**

| Metric | Our 2.29 | comma baseline_fast (verified same day) |
|---|---|---|
| Final score | **2.290** | 4.400 |
| PoseNet dist | 0.247 | 0.382 |
| SegNet dist | 0.0026 | 0.00946 |
| Rate (unscaled) | 0.0185 | 0.0598 |
| Archive bytes | 693,857 | 2,243,917 |

**Reproducibility recipe (deterministic from current `submissions/baseline_dilated_h64_0_90/` repo state):**
1. On a CUDA box with NVDEC + DALI + ffmpeg-master (in_primaries support):
   ```
   .venv/bin/python experiments/build_baseline_archive.py \
       --device cuda --crf 50 \
       --output /tmp/archive.zip
   ```
2. Eval:
   ```
   .venv/bin/python experiments/contest_auth_eval.py \
       --archive /tmp/archive.zip \
       --inflate-sh submissions/robust_current/inflate.sh \
       --upstream-dir upstream --device cuda
   ```
3. Provenance archive_sha256 = `dcdb617ad3686489a1e16d61500f5470e46935ae725eeaf13cc647529d4ca30b` (Oregon RTX 4090, driver 580.95.05, torch 2.5.1+cu124).

**What's NOT in this archive (proven worse today):**
- 48x64 masks (catastrophic 53.60 — `project_baseline_0_9001_lost_archive_test`)
- Half-frame masks (PoseNet=28.7, breaks renderer's motion path — `feedback_half_frame_breaks_posenet`)
- CRF=55 or 63 masks (3.07 / 9.89 — PoseNet sensitivity too high)

**Council Yousfi+Fridrich's verdict on the 2.29 floor (2026-04-27):**
- SegNet contribution 0.26 is at floor — DON'T retrain renderer.
- PoseNet contribution 1.57 is 86% of remaining headroom — pose TTO is THE bet.
- FastViT-T12 Jacobian rank 1.008 (per `project_posenet_rank1_discovery`) means a SCALAR per-pair correction in pose-dim 0 should collapse PoseNet 5-10x.
- ONE BET RECOMMENDATION: pose TTO with warm-start from baseline poses, eval_roundtrip, 500 steps × 600 pairs on 4090. $1.50, 6h. Predicted 0.85-1.10.

**The "0.9001 historical record" is unreproducible from saved artifacts.** The 2.29 IS our true verified floor. Stop quoting 0.9001 anywhere — it does not survive contact with reproducible artifacts.

**Why this matters:** for a month we operated on phantom scores. 2.29 is the first number where (a) the archive bytes are saved + SHA-pinned, (b) the eval pipeline is verified against historical data point (53.61 reproduced exactly), (c) every step is deterministic given hardware + seed.
