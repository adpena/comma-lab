---
name: Session 2026-04-25 Deployment
description: 11 review rounds (64 issues fixed, 7 critical), auth score 2.26, WILDE+SHIRAZ deploying on Vast.ai 4090s
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Session 2026-04-25: DX hardening + QAT pipeline + deployment.

**Auth score**: 2.26 [contest-compliant] via contest_eval.py → upstream evaluate.py
- SegNet: 0.238 (100 × 0.00238)
- PoseNet: 1.566 (sqrt(10 × 0.245))
- Rate: 0.457 (25 × 0.0183) — 670KB archive (float renderer, no Brotli)

**11 review rounds**: 64 issues found and fixed (7 critical).
Key criticals: gradient flow killed (R1), padding_mode not serialized (R4), inline APG TypeError (R8), brotli DQ (R9).

**DX implemented**:
- contest_eval.py: single-command e2e eval matching upstream 1:1
- Half-frame mask auto-duplication with hard error on bad count
- FP4/QAT guard, ffprobe validation, env sanitization
- 7/7 renderer loaders now handle padding_mode/use_dilation
- I4LZ integrity check, strict=True everywhere

**Deploying**: WILDE + SHIRAZ on two Vast.ai 4090s (~$0.25/hr each, ~$4 total).
- WILDE: hinge loss, freeze/unfreeze, error_boost 9→49, Fridrich losses
- SHIRAZ: focal_ste γ=2, curriculum 0.3, Fridrich losses
- Both: base_ch=32, mid_ch=48, use_dsconv, use_dilation, padding_mode=replicate, eval_roundtrip=True

**Why:** Current model was trained with old profile (proven_baseline). WILDE/SHIRAZ incorporate all session fixes + new techniques. Expected auth score: 0.65-1.15.

**How to apply:** After training completes (~8h), download checkpoints → QAT (50+250 epochs) → pose TTO → archive → contest_eval.py. Target: sub-1.0 auth.
