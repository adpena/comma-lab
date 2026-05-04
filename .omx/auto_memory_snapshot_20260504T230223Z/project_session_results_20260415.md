---
name: Session Results 2026-04-15 — Auth 0.61, Distillation ep900, FP4, Mini-scorer
description: Auth 0.61 contest-compliant via distillation ep300 and pose-space TTO. Distillation proxy 0.338 at ep900 (still converging). FP4 saves 0.085 rate points. Gradient corrections dead (43MB). MiniSegNet passes 98.7%, MiniPoseNet fails.
type: project
originSessionId: docs-agent-20260415
---

## Session Outcome (2026-04-15)

### New Auth Results
- **Auth 0.61 [contest-compliant]** via distillation v2 at epoch 300 (proxy 0.446)
- **Auth 0.61 [contest-compliant]** via pose-space TTO at epoch 300
- Both represent 30% improvement over renderer baseline (0.87)

### Distillation v2 Full Trajectory
| Epoch | Proxy | PoseNet | SegNet |
|-------|-------|---------|--------|
| 0 | 0.807 | 0.0310 | 0.00217 |
| 50 | 0.596 | 0.0170 | 0.00240 |
| 100 | 0.544 | 0.0120 | 0.00230 |
| 200 | 0.493 | 0.0090 | 0.00210 |
| 300 | 0.446 | 0.0070 | 0.00200 |
| 550 | 0.375 | 0.0041 | 0.00112 |
| 680 | 0.364 | 0.0035 | 0.00098 |
| 900 | 0.338 | — | — |

Status: still converging at ep900. No plateau detected.

### Pose-Space TTO
- seg_weight=0 is optimal (pure PoseNet optimization)
- PoseNet: 0.031 → 0.0016 (-94.7%) via 6D FiLM conditioning vectors
- Per-batch: 90-99% PoseNet improvement across all 600 pairs
- Archive cost: 14.4 KB (600 × 6 × float32)
- PoseNet-SegNet orthogonality in FiLM space: conditioning space decouples them

### FP4 Quantization
- Renderer: 297 KB → 170 KB (FP4 custom codebook)
- Rate savings: 0.0079 → 0.0045 (-0.085 points)
- FP4 + CRF30 masks combined: 215 KB. Saves 0.113 vs FP32 baseline.
- No training changes required.

### Gradient Corrections: DEAD
- Measured: 743 KB for 20 frames
- Projected full scale: ~44.6 MB (rate cost ~1.19)
- Not viable. Gradient signal too dense to sparsify efficiently.

### Mini-Scorer Results
- MiniSegNet h=32: 98.69% argmax fidelity. Archive: 87 KB FP16. PASSES.
- MiniPoseNet: R² = 0.002. FAILS fundamentally.
- Workaround: store 600×6 GT pose outputs (14.4 KB) as fixed targets.

### Key Insight: Conditioning-Space TTO
196,608:1 compression of optimization space. FiLM conditioning space (6D) is the
natural optimization surface for PoseNet (6-DoF output). PoseNet and SegNet gradients
are approximately orthogonal in this space. This enables pure-PoseNet optimization
without SegNet tradeoff.

### Updated Projected Floor
Full stack: distilled renderer + pose-space TTO + FP4 + MiniSegNet inflate TTO → auth ~0.25

### Budget
~$9 spent on Vast.ai. ~$15 remaining.
