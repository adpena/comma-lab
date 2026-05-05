---
name: Curriculum difficulty MUST use full score formula, not PoseNet-only
description: Yousfi audit caught that curriculum used PoseNet MSE only for difficulty. SegNet is 77x more important. Must use 100*seg + sqrt(10*pose). This misdirected the entire curriculum toward PoseNet-hard pairs.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

The curriculum difficulty metric was computing per-pair PoseNet MSE only.
SegNet contributes 77x more to the score at our operating point.
A pair with high seg error but low pose error would be ranked "easy"
and undersampled — exactly wrong.

**Why:** SegNet score = 100 * seg_dist. PoseNet score = sqrt(10 * pose_dist).
At seg=0.003, pose=0.002: SegNet contributes 0.30, PoseNet 0.14.
The difficulty should weight SegNet 2x more than PoseNet.

**How to apply:** ALWAYS use the contest formula for difficulty:
  difficulty = 100 * per_pair_seg_disagree + sqrt(10 * per_pair_pose_mse)
NEVER use a single scorer's raw metric as difficulty.
This applies to: curriculum sampling, hard-pair selection, error replay,
and any future difficulty-based technique.
