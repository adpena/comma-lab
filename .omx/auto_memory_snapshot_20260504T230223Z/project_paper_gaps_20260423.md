---
name: Paper missing 10 major findings from April 21-23 session
description: QAT vs float, EMA, CRF50 matching, scorer architectures, Yousfi-Fridrich, measurement disasters, variable-rate, FP4 vs ASYM, infra hardening, methodology. All need writeup.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Findings NOT yet in the paper (priority order for writeup)

1. **Yousfi-Fridrich inverse steganalysis framing** — the meta-narrative.
   Contest creator was Fridrich's PhD student. SegNet IS a steganalysis detector.
   Our Fridrich losses ARE detector-informed embedding (Yousfi 2022).
   This frames the ENTIRE paper. Should be in introduction.

2. **The 4 measurement disasters** — cautionary tale + preflight.py solution.
   48x64 masks (score 103→2), wrong archive, overlapping pairs, eval_roundtrip=False.
   Publishable as a "pitfalls" section. Shows the value of preflight validation.

3. **QAT vs float head-to-head** — definitive empirical result.
   QAT from epoch 0 is 8.7x slower. Quantizr trains float then quantizes.
   Directly contradicts the "QAT from start is always better" assumption.

4. **Exact scorer architectures** — SegNet=EfficientNet-B2 U-Net (NOT B4),
   PoseNet=FastViT-T12 (NOT EfficientNet). Vanilla stride-2 stem = blind spots.
   YUV6 chroma subsampling = PoseNet can't see sub-2x2 chroma patterns.

5. **Fridrich ablation** — Phase 1 helps 5-7%, Phase 2 mixed.
   First empirical evidence of phase-specific inverse steganalysis losses.
   Novel contribution: UNIWARD + L∞ adapted for neural video compression.

6. **Difficulty distribution** — 227x ratio, sqrt exploit, 0.932 net improvement.
   Already partially in hard_pair_analysis.md but needs the new data.

7. **CRF50 mask-matched training** — training/deployment mismatch causes 27x regression.
   Universal lesson: always train with deployment-matched data.

8. **EMA as missing ingredient** — 5-15% improvement, was built but never wired.
   Illustrates the "build tools then don't use them" failure pattern.

9. **Variable-rate mask encoding** — framework built, TDD, 8 tests.
   Per-frame CRF allocation via difficulty map. Novel for video compression.

10. **Human-AI collaborative methodology** — 48 commits, 20 tests, 3 adversarial
    review rounds, skunkworks council, cross-disciplinary insights.
    The PROCESS is the innovation. Meta-narrative for best writeup prize.
