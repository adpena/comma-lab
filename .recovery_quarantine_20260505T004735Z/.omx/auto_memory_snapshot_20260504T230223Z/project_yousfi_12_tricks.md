---
name: YassineYousfi's 12 Tricks — Complete Implementation Guide
description: Contest organizer's insights — 5 original + 7 supplemental. All must be implemented. He is the expert.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The 12 Tricks

**Original 5 (IMPLEMENTED):**
1. Feature matching loss (PoseNet embeddings) — `tac.losses.posenet_embedding_loss`
2. Frequency-domain shaping (Haar DWT) — `tac.losses.frequency_aware_loss`
3. Even-frame degeneracy — `tac.training` + `compress.sh`
4. Archive pruning — `compress.sh` metadata stripping
5. Aggressive overfitting — `overfit_cpu` profile (10K epochs)

**Supplemental 7 (PARTIALLY IMPLEMENTED — must complete):**
6. Direct gradient attack (TTO) — `tac.tto` exists but not fully exploited
7. SegNet argmax insight — mask2mask paradigm, renderers in tac, NOT YET SCORED
8. Chroma channel exploitation — YUV420 in tac.data, dedicated technique NOT built
9. Neural network info density — FP4 in tac, not yet packaged for submission
10. Multi-model stacking — all components exist, integration pipeline NOT built
11. Extreme SegNet reward — understood, MRS in tac.adaptive
12. Video = mask sequence — entropy coder 239 bytes, FULL PIPELINE NEVER RUN E2E

**Why:** YassineYousfi designed the scoring formula and the contest. His insights ARE the solution. Every trick he suggested must be fully implemented and tested. He is the expert on our council.

**How to apply:** Track implementation status. Any trick marked "not yet" is a priority. The full mask2mask pipeline (trick 12) is the path to sub-0.50.

Full document saved to ~/Downloads/yousfi_tricks_complete_20260411.md
