---
name: Current Track B floor at 1.727
description: Promoted floor is 1.727 via h=64 best-checkpoint int8 selection, beats leaderboard #1 (1.95) by 0.223 — session trajectory 2.01→1.727
type: project
---

Promoted 2026-04-09. Authoritative CPU scorer result.

- PoseNet distortion: 0.03317023
- SegNet distortion: 0.00575544
- Archive bytes (current_workflow): 864,167
- Rate: 0.02301653
- Exact score: 1.727
- Leaderboard #1 is 1.95, we lead by 0.223

**Weights**: `submissions/robust_current/postfilter_int8.pt` (45587 bytes, h=64)
**Architecture**: 3→64→64→3 conv residual, QAT-wrapped (25219 params)
**Training**: 1000 epochs, saliency α=20, QAT+EMA decay=0.997
**Key mechanism**: best-checkpoint selection with int8-quantized evaluation (epoch 826)

**Session trajectory**: 2.01 → 1.99 → 1.945 → 1.92 → 1.845 → 1.762 → 1.727
**LeCun scaling law**: score = -0.159 × ln(h) + 2.382

**Active experiments pushing toward 1.45:**
1. h=96 at 1500 epochs (next scaling point, predicted ~1.67)
2. SegNet h=64 attack on local + bat00 CUDA (the biggest lever)
3. Per-channel int8 on h=64 (free upgrade, proxy running)

**Panel consensus on maximum potential (24 days)**: 1.45 ± 0.10
**Theoretical absolute minimum**: 0.96 (requires perfect reconstruction)
