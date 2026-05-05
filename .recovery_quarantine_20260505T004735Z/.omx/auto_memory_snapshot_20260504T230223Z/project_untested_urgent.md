---
name: URGENT — 10 Implemented-but-Untested Techniques + 4 Not-Yet-Built
description: Critical audit (2026-04-20). 10 techniques built but never tested at scale. 4 items not yet built. Must test ALL before submission.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## 10 IMPLEMENTED BUT UNTESTED (must validate or kill with evidence)

1. Per-pair latent codes (16D) — smoke marginal, needs 200+ steps
2. LoRA TTO — smoke 2.8%, needs 1000+ epochs
3. DSConv architecture — never trained, separate experiment needed
4. Stacked postfilter — needs stable renderer output first
5. MiniSegNet inflate TTO — wired up, never e2e tested
6. Constrained gen from noise — council killed for submission, paper only
7. Multi-pass refinement — council says <0.001, keep for polish
8. Null space projection in TTO — validated mechanism, not in any run
9. Per-class SegNet weighting — implemented, low priority at current operating point
10. Feature matching loss — wired in training.py, unused in distillation

## 4 NOT YET BUILT

1. FP4 export of FiLM renderer (296KB→148KB, saves 0.05 score from rate)
2. Optimized embedding baked into ASYM .bin header
3. Submission PR template for commaai repo
4. Final writeup polish for best-writeup prize

## Priority for remaining 13 days
- FP4 export: IMMEDIATE (free 0.05 score)
- Auth eval of pose TTO: IMMEDIATE (after completion)
- Embedding TTO: HIGH (120 bytes, compounds with pose TTO)
- Stacked postfilter: MEDIUM (after renderer stabilizes)
- Submission PR: by April 27 (lock date)
- Everything else: test if time allows, kill with evidence if not
