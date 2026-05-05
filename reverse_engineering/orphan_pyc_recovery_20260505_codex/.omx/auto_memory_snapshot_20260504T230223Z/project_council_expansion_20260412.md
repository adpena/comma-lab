---
name: Council Expansion — Bhat Advisory + Quantizr Adversarial
description: Bhattacharyya added as advisory panel, Quantizr as non-voting adversarial reviewer. Lossless winner rejected.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Council Roster (2026-04-12)

**Tripartite Pact (voting):** Yousfi, Fridrich, Contrarian
**Extended council (voting):** Karpathy, LeCun, Tao, Pareto, Arrow, von Neumann
**Advisory panel:** Bhattacharyya (Bhat) — loss function design
**Adversarial reviewer (non-voting):** Quantizr — challenge PoseNet strategy, rate budgets, architecture

## Key decisions
- Arm C (true Bhattacharyya distance) REJECTED — curriculum dominates
- Lossless commavq winner REJECTED — domain mismatch, council bloat
- A/B test: 2500 epochs baseline vs Fridrich curriculum (soft→tempered→STE)
- Quantizr's role: "How would Quantizr attack this?" as forcing question

**Why:** Quantizr's 480x PoseNet advantage (0.001 vs our 0.482) is the #1 technical gap. Their renderer approach is directly relevant. But their specific choices shouldn't override our Fridrich constrained optimization — different theoretical foundation.
