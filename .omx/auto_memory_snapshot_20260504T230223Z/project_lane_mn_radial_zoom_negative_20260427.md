---
name: Lane M+N (radial-zoom + L∞) auth 2.35 — radial-zoom 1-DOF was a NET LOSS on baseline checkpoint
description: 2026-04-27 Lane M (radial-zoom 1-DOF pose TTO from rank-1 PoseNet Jacobian discovery) + Lane N (Fridrich L∞ pose penalty) combined run on baseline_dilated_h64. Auth 2.35 [contest-CUDA] vs baseline 2.29 = +0.06 REGRESSION. Lane A 1.15 dominates by 1.20 pts. The rank-1 hypothesis is correct for the PoseNet's internal sensitivity but wrong for the renderer's pose input — the baseline renderer was trained on full 6-DOF poses and degrades when fed 1-DOF.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Lane M+N result 2026-04-27 (Spain 4090, 35716364, ~$0.45 + 1.3h):**

| Metric | Lane M+N | Baseline 2.29 | Lane A 1.15 |
|---|---|---|---|
| **Final score** | **2.35** [contest-CUDA] | 2.29 | **1.15** |
| PoseNet dist | 0.269 | 0.247 | 0.005 |
| SegNet dist | 0.0026 | ~0.003 | 0.0046 |
| Rate unscaled | 0.0182 | ~0.0185 | 0.0185 |
| Archive bytes | 682,882 | ~700KB | 694KB |
| Pose contribution | 1.64 | 1.57 | 0.22 |

**Config:**
- `--checkpoint submissions/baseline_dilated_h64_0_90/renderer.bin` (FP32 ASYM, NOT Lane A's checkpoint)
- `--gt-poses-path submissions/baseline_dilated_h64_0_90/optimized_poses.pt` (6-DOF baseline poses as warm-start)
- `--pose-mode radial-zoom` (Lane M: constrains updates to 1-DOF along optical axis from FoE)
- `--linf-pose-weight 1.0 --linf-pose-budget 0.05` (Lane N: Fridrich L∞ penalty, mostly inactive at 0.0001-0.001)
- `--steps 500 --batch-pairs 8 --eval-roundtrip --posetto-noise-std 0.5`

**Why it failed (forensic post-mortem):**

The rank-1 PoseNet Jacobian discovery (`project_posenet_rank1_discovery`) showed that PoseNet's INTERNAL sensitivity is 99.8% rank-1 along dim 0 = scalar radial zoom from FoE (256, 174). The hypothesis was: optimize poses in this 1-DOF subspace and capture nearly all the signal at 6× lower parameterization.

**The hypothesis was a category error.** The rank-1 finding was about PoseNet's sensitivity to ITS OWN INPUT (the YUV6 frames), NOT about the renderer's response to its pose input. The baseline_dilated_h64 renderer was trained on full 6-DOF poses; constraining the input poses to 1-DOF means dims 1-5 fall to zero (or to the GT warm-start values, depending on padding) and the renderer's FiLM-conditioned outputs are off-manifold. The result: rendered frames don't match the YUV statistics PoseNet expects, so PoseNet distortion EXPLODES (0.247 → 0.269, +9%).

**Saved-pose shape pitfall (today's debugging):**

`optimize_poses.py --pose-mode radial-zoom` saves poses as `(N, 1)`. Loader expects `(N, 6)`. Initial fix padded with zeros → above result. A more correct fix would pad with the baseline pose values for dims 1-5 (preserving the warm-start), but the underlying premise (radial-zoom = enough) is still wrong.

**L∞ penalty was inactive:**

`linf=0.0001-0.001` per batch (vs budget 0.05) means the L∞ term contributed ~0% of the loss. The radial-zoom 1-DOF constraint already kept pose updates so small that L∞ had nothing to spread. Lane N component is unevaluated as a result.

**Implications:**

1. **Radial-zoom is dead as a generic pose TTO trick.** The 1-DOF parameterization doesn't help on renderers trained for 6-DOF.
2. **Lane M's prediction was wrong.** Council-Yousfi prediction was [0.85, 1.10]; actual 2.35.
3. **The rank-1 discovery is still valid for OTHER applications:** lane-marking displacement → speed (`project_lane_marking_speed_estimation`), zero-cost pose initialization, etc. Just not as a pose TTO subspace constraint.
4. **The correct radial-zoom TTO would have to** train a renderer JOINTLY with 1-DOF poses (a smaller, simpler renderer that doesn't expect 6-DOF input). That's a different lane (Lane M+).

**What NOT to do:**

- Do not promote Lane M+N's archive (682KB at 2.35) to submission. Lane A's 1.15 dominates.
- Do not retry radial-zoom on dilated-h64 baseline — the renderer's pose input space is fundamentally 6-DOF.
- Do not assume rank-1 sensitivity applies to renderer input. Sensitivity ≠ control.

**Cost of finding:** $0.45 + 1.3h GPU + ~10 min debug overhead (config.env missing on remote, 1-DOF shape pitfall). CHEAP for the empirical evidence that closes the radial-zoom hypothesis.

**Bonus discovery:** The remote bootstrap process is silently dropping `submissions/robust_current/config.env`. Spain remote was missing it; pushed it manually for re-eval. This is a deploy bug — should be added to the canonical bootstrap checklist.
