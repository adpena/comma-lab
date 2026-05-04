---
name: Session End State — 2026-04-12 (58 commits, training live)
description: Marathon session. Asymmetric warp training converging on Modal T4. CRF sweep running. Auth eval fleet on 4 platforms. 21 days to deadline.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Training Status
- **Modal T4**: Asymmetric warp training v2, converging beautifully
  - Epoch 175 metrics: seg_hard=0.016, pose=0.048, loss=0.26
  - Phase 1 (40% = 4000 epochs): learning colors + basic scorer optimization
  - Dashboard: check Modal for app "tac-asymmetric-warp"
  - Results volume: `tac-asymmetric-results/asymmetric_warp_t4/`
  - Est. cost: ~$3.25 for 5.5h

## Active Parallel Work
- CRF sweep running locally (6 CRF values: 30, 32, 34, 36, 38, 40)
- Auth eval fleet ready on Modal + Lightning + Kaggle + CLI

## What Was Built This Session (58 commits)
- AsymmetricPairGenerator (warp paradigm, Quantizr-inspired)
- Fridrich 3-phase curriculum (soft→tempered→STE)
- Full export/inflate pipeline (DPSM + ASYM binary formats)
- 54 CLI flags, full telemetry, replicability manifest
- Modal deploy with T4, periodic volume commits, resume support
- Auth eval on 4 platforms (Modal, Lightning, Kaggle, CLI)
- Canonical AuthEvaluator class (src/tac/eval/)
- DALI mask validation script
- Scorer resolution verification (Eureka 3 — lane open)
- Ego-motion precompute (already implemented, just needs activation)
- Entropy archive fixes (binary search, safe serialization, compress_byte_stream API)
- Gate monitoring, auto-kill, gate regularization
- 7 eureka moments documented
- 21-item research roadmap with 30+ papers
- 10 killed techniques documented with cross-domain notes
- Quantizr deobfuscation (full architecture reverse-engineered)
- Council expanded: Shannon + Bhat advisory, Quantizr adversarial

## Key Numbers
- Our 1.33 is #1 on official leaderboard (Quantizr 0.60 unconfirmed)
- Training target: sub-0.50
- Model: 287K params, ~140KB FP4
- 21 days to May 3 deadline
- ~$47 Modal credits remaining

## Next Steps (council-approved priority)
1. CRF sweep results → improve current 1.33 submission
2. Training convergence → auth eval → first renderer score
3. Writeup/paper infrastructure for best-writeup prize
4. Eureka 2 (odd frames SegNet-free) architectural sketch
5. Entropy coding on renderer weights (polish phase)
