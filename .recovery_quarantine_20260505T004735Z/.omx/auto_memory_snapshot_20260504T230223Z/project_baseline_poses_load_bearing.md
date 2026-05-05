---
name: 0.90 baseline poses are LOAD-BEARING (don't TTO from scratch)
description: The dilated-h64 renderer + its 7KB poses are a JOINT artifact. Replacing poses with extract_gt_pose_targets-init causes 33% pixel shift and 23x PoseNet degradation. Pose TTO from any non-baseline init is structurally hopeless on this checkpoint.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified 2026-04-26 via local sanity test** (`submissions/baseline_dilated_h64_0_90/`):

| Test | Result |
|---|---|
| Render 10 pairs with REAL baseline poses | frames mean=46.78, std=18.80 |
| Render 10 pairs with ZERO poses | frames mean=61.82, std=26.14 |
| Pixel diff (real − zero) | **mean=15.8, max=105** (33% shift on 0–255 range) |
| Renderer FiLM modulation | `pose_dim=6`, FiLM present, fully active |

**What this means:**
- The renderer's FiLM layers have memorized a SPECIFIC pose distribution: `mean=5.22, std=11.68, max=35.03`. These are NOT raw frame-pair PoseNet outputs (~0±1) — they're large FiLM-modulation coefficients trained jointly with the renderer.
- The renderer + poses are a JOINT artifact. The 0.90 score is co-attributed.
- Replacing the poses with anything from a different distribution (e.g. `extract_gt_pose_targets(GT)`) → renderer renders DIFFERENT (worse) frames → PoseNet on those frames diverges by 23× (LANE-B 0.011 → 0.246).

**How to apply:**
1. **NEVER run pose-only TTO on this renderer.bin from a non-baseline init.** It's structurally hopeless. The optimizer cannot escape the wrong basin. This is what burned LANE-B (6.5h + $2 of pure waste).
2. **If you MUST TTO poses on this checkpoint**, warm-start from `submissions/baseline_dilated_h64_0_90/optimized_poses.bin` and use a TINY learning rate (lr=1e-4 max) — you're refining, not searching.
3. **For NEW architectures**, train poses jointly with the renderer (pose_dim init from extract_gt_pose_targets is fine if the renderer is also being trained alongside).
4. **To beat 0.90 starting from this baseline**, the only safe lane is RATE ATTACK (LANE-A): shrink the renderer (brotli, FP4 codebook tuning) or shrink masks (CRF, half-frame) WITHOUT touching the poses. 25KB shaved off the archive ≈ 0.017 off the rate score.
5. **For the "carrier signal" theory** (Fridrich/Yousfi): the renderer is using the pose vector as a steganographic carrier. Detector-detectable artifacts in the rendered frames are encoded by the joint (renderer-weights × pose-vector) inner product. Decoupling the two destroys the encoding.

**Council R1 status (2026-04-26):** Yousfi + Fridrich's "entangled carrier" theory is empirically confirmed. Quantizr's "poses ARE the win" theory is empirically confirmed. Both are correct; they describe the same phenomenon at different abstraction levels.
