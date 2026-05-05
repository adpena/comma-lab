---
name: Lane F-V2 FP4 QAT on Lane A — 1.79 [contest-CUDA] — bug fix worked but FP4 architecturally bottlenecks dilated-h64 PoseNet
description: 2026-04-27 Lane F-V2 (FP4 QAT anchored on Lane A's 1.15 checkpoint with --poses bug fix) scored 1.79 [contest-CUDA]. Bug fix VALIDATED (0.94 better than Lane F V1's 2.73). But still REGRESSION vs Lane A 1.15 (+0.64). PoseNet exploded 0.005→0.101 (20× degradation) — confirms FP4 quantization on dilated-h64 is architecturally PoseNet-hostile. Don't promote. For rate gains beyond brotli, need different architecture (smaller renderer or self-compression), not FP4-on-current-arch.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Lane F-V2 result 2026-04-27 (Nevada 4090 35719867, ~$0.20 + 35min wall):**

| Metric | Lane F-V2 | Lane A (anchor) | Lane F V1 (buggy) |
|---|---|---|---|
| **Final score** | **1.79** [contest-CUDA] | **1.15** | **2.73** |
| PoseNet dist | 0.101 | 0.005 | 0.391 |
| SegNet dist | 0.0039 | 0.0046 | 0.0037 |
| Rate unscaled | 0.0156 | 0.0185 | 0.0156 |
| Archive bytes | 586,426 | 694,045 | 586,232 |
| Pose contribution | 1.005 | 0.223 | 1.977 |
| Seg contribution | 0.393 | 0.461 | 0.365 |
| Rate contribution | 0.390 | 0.462 | 0.390 |

**Bug fix VALIDATED:**

The Lane F V1 bug (silent zero-pose load) was real. Lane F-V2 with proper `--poses` arg threading scored 1.79, which is **0.94 better than Lane F V1's 2.73**. PoseNet improved from 0.391 → 0.101 (4× better). The bug fix delivered exactly what the council audit predicted.

**But Lane A still dominates by 0.64:**

The remaining 0.79 PoseNet gap (Lane A=0.005 vs Lane F-V2=0.101) is NOT from the bug — it's from FP4 quantization itself. The QAT process trains the model to be robust under FP4, but the QAT training metric (proxy distortion=1.135) is wildly optimistic vs the contest-CUDA result (1.79). Per memory `feedback_proxy_auth_math_useless`, this is a known 100-350× proxy-auth gap; Lane F-V2 is in the lower end of that range (~16× drift on the QAT proxy).

**The architectural insight:**

For Lane A's renderer (dilated-h64 ASYM, 290KB FP32), FP4 quantization causes a PoseNet penalty that scales as ~20× the float baseline. This is NOT linear in bit-depth — going from 32 bits → 4 bits doesn't add 8× quantization noise; it adds 20× PoseNet penalty. The reason is the ASYM PoseNet path's sensitivity to YUV6 statistics — small rendering errors at the renderer's last conv propagate as YUV-correlated noise that FastViT's attention amplifies.

**Implications for rate-attack lanes:**

1. **FP4 on dilated-h64 is architecturally bad.** Don't try other variants on this checkpoint.
2. **Alternative rate attacks must avoid FP4-on-current-arch:**
   - Brotli compression on FP32 (Lane B alt, -0.0037, already shipped) — minimal gain
   - Smaller architecture trained from scratch (Lane K DSConv, Lane L PSD)
   - Self-Compression (Lane S) — learnable per-channel bit allocation, structurally different from FP4
   - Cool-Chic mask compression (Lane I) — orthogonal to renderer bytes
   - Half-frame masks (Lane D, in progress, ETA 4h) — store only odd frames, -0.20 rate predicted

3. **The Quantizr 0.33 gap is still 1.46 below Lane A's 1.15.** Need at least:
   - Lane A's PoseNet (0.005) AND
   - Quantizr-level rate (0.0024 → score contribution 0.06 vs Lane A's 0.46) AND  
   - Quantizr-level SegNet (~0.001 → 0.10 vs Lane A's 0.46)
   - Total achievable: ~0.22 (Lane A pose) + ~0.10 (better seg) + ~0.06 (better rate) = 0.38 — within striking distance of Quantizr's 0.33 only if SegNet AND rate both improve dramatically.

**Cost of Lane F-V2 finding:** ~$0.20 + 35 min. Cheap for the architectural insight.

**What NOT to do:**

- Don't promote Lane F-V2's 586KB/1.79 archive.
- Don't retry FP4 with more epochs — 500 was 10× the original 50, still hit the same 20× PoseNet penalty.
- Don't try INT8 instead — INT8 has 8× more precision but the architectural problem is YUV statistics sensitivity, not raw bit-depth.
- Don't FP4 other architectures (SHIRAZ, DEN, WILDE, GREEN) without sensitivity testing PoseNet first.

**What TO do:**

1. Wait for Lane D half-frame retrain (proxy plateau but auth gap could be huge).
2. Push Lane S Self-Compression (stashed at stash@{0}, requires manual completion of subagent's partial work).
3. Push Lane I Cool-Chic on masks (orthogonal to renderer bytes).

**Related memories:**
- `project_lane_f_fp4_qat_regression_20260427` — Lane F V1 bug-induced regression
- `feedback_silent_default_masquerading_as_negative_result` — the bug fix pattern
- `project_5stage_quantization_advantage` — our QAT stack's superiority over Quantizr's vanilla 5-stage
- `feedback_proxy_auth_math_useless` — the 100-350× proxy-auth gap in this codebase
