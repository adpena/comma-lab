---
name: Session State 2026-04-14 End (HISTORICAL)
description: Historical snapshot from April 14. Superseded by later sessions.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## STALE NOTICE
This is a historical snapshot from 2026-04-14 (6 days old). Key changes since:
- TTO v5a/v5b ran successfully (auth 0.43/0.41) -- gradient bug fix validated
- Three breakthroughs identified (hinge, latent codes, two-phase TTO)
- Quantizr improved to 0.33 (PR#55)
- SegNet "77:1" finding was WRONG (wrong checkpoint)
- Vast.ai is now primary compute (Modal deprioritized)

## Historical State (April 14)
- auth=0.87 (v5 renderer_best.pt ep12600, asymmetric warp + Lagrangian annealing)
- Leaderboard: #2 behind Quantizr (was 0.60, now 0.33)
- bat00: Tailscale unreachable (may still be down)
- Kaggle: mount path fixed, kernels boot successfully
- Modal: idle, low credits

## Four Paths (April 14 -- some now validated)
1. **Renderer + TTO**: VALIDATED. Auth 0.41 with embedding loss.
2. **Joint pair generator**: Not pursued yet.
3. **Scorer-space generation**: Needs 4 fixes. Low priority.
4. **Kaggle long training**: Fixed, available for bonus compute.
