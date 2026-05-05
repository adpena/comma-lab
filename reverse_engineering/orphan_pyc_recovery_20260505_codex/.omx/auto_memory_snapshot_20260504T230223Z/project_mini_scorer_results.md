---
name: Mini-Scorer Results — SegNet PASSES (98.7%), PoseNet FAILS (R²=0.002)
description: MiniSegNet hidden=32 achieves 98.69% argmax fidelity. MiniPoseNet fundamentally cannot learn 6-DoF regression at 48x64. Workaround: store pose targets directly.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Mini-Scorer Training Results (2026-04-20, local MPS)

### MiniSegNet: PASSES ✓
- hidden=32: 98.69% pixel-wise argmax agreement with full SegNet
- hidden=16: 97.04% (below 98% threshold)
- Archive cost: 87KB (FP16) for h=32, 28KB for h=16
- Suitable for inflate-time SegNet TTO and constrained gen

### MiniPoseNet: FAILS ✗
- R² = 0.002 (threshold was 0.95) — essentially random
- 2-layer CNN + GAP at 48×64 cannot learn 6-DoF pose regression
- Fundamental architecture limitation, not hyperparameter issue

### Workaround for Lane 4
MiniPoseNet not needed if we store GT PoseNet OUTPUTS directly:
- 600 × 6 floats × 4 bytes = 14.4KB (already have this as poses.pt)
- At inflate time: use pixel-level MSE against flow fields derived from poses
- Or: MiniSegNet-only TTO (optimize SegNet, use stored poses as fixed targets)
- No gradient through PoseNet needed — just target matching

### Implications
- Lane 4 (constrained gen): partially unblocked. MiniSegNet provides SegNet gradients.
  PoseNet optimization uses stored targets (no mini-PoseNet gradient).
- Mini-TTO inflate: MiniSegNet-only TTO viable. PoseNet improvement requires
  a different approach (FiLM conditioning, stored targets, or larger mini-PoseNet).
