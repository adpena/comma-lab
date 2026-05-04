---
name: Grand Council Consensus (2026-04-10)
description: Full council debate outcome — approved plan, scoring formula, literature strategy, dissents recorded
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Approved Implementation Plan

Phase 1 (Apr 10-14): Validate PCGrad (Modal) + CRF 35 (Local) + extreme_posenet (Kaggle)
Gate: seg < 0.00590 AND pose < 0.00250
Phase 2a (if gate passes): Implement CAGrad + run
Phase 2b (if gate fails): CRF 35/36 auth eval + extreme_segnet
Phase 3 (Apr 17-20): Spatial attention split OR CRF fine-tuning
Decision point: April 21 — lock submission
Phase 4 (Apr 21-25): Auth eval top-2 candidates
Phase 5 (Apr 25-May 3): Packaging + paper

## Literature Decisions

IMPLEMENT: PCGrad (validate existing), CAGrad (if PCGrad passes gate)
REJECT: Nash-MTL (wrong architecture), Aligned-MTL (convex frontier = weighted sum), FAMO (overkill for 2 tasks)
QUEUE: Spatial attention split (Collier's proposal, after gradient methods)

## Scoring Formula (Arrow+Pareto)

SCORE = (s/s₀)^0.40 * (p/p₀)^0.35 * (r/r₀)^0.25

Properties: scale-invariant, complementary, Pareto-complete, no axis exploitation
For paper Section 6 only — cannot change competition formula

## Key Contrarian Insight

60% probability PCGrad fails gate. The SegNet regression may be fundamental to
the dilated architecture, not fixable by loss reweighting. Prepare Phase 2b.

## Dissents

1. Contrarian: gradient methods won't help, focus on rate
2. Collier: spatial attention split addresses root cause (spatial allocation)
3. Tao: Arrow impossibility "escape" is technically cardinal utility, not ordinal
4. Rubin: wants counterfactual for every experiment (Karpathy says too slow)

**Why:** This is the approved plan for the final 23 days.
**How to apply:** Follow phases strictly. One variable at a time. Lock Apr 21.
