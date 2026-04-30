# Council Adversarial Review — Round 2 (Shannon + MacKay + Hotz rotation)

**Date**: 2026-04-30
**Document under review**: `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
**Reviewers**: Shannon (LEAD, info theory), MacKay (memorial, MDL), Hotz (engineering shortcuts)

## Issues found (3)

### Issue 1 (Shannon) — Correlation haircut hand-waved

**Concern**: Section 6.3 states "sum of independent EVs -0.83 with correlation haircut 0.6 → final -0.50". The 0.6 haircut is a Hassabis-derived heuristic but is not rigorously derived. Empirical support for 0.6 vs 0.4 vs 0.8 is missing. The math is the WEAKEST in the document.

**Severity**: HIGH (load-bearing on the final-stack prediction).

**Fix**: Add Section 6.3 caveat — "haircut is a Hassabis-derived heuristic; actual correlation requires empirical measurement after 3+ paradigm shifts land. Predicted final-stack band [0.18, 0.30] reflects the haircut uncertainty."

**Status**: ACK — caveat added to Section 6.3 + repeated in Section 8.5 probability table.

### Issue 2 (MacKay) — MDL framework not specified mathematically

**Concern**: Lane 16 MDL/Bayesian framework is mentioned as paradigm shift ζ but NOT specified mathematically. "Picks the best stack composition from 5+ codec families" is vague. Without a formulation, Lane 16 is a sketch not a design. The framework should be: `total_description_length = bits(data | model) + bits(model)`.

**Severity**: MEDIUM (Lane 16 cannot move from L0 to L1 without spec).

**Fix**: Add to Section 4.3 paradigm shift ζ:
```
Lane 16 MDL formulation:
  min_θ Σ_stream R_stream(θ_stream) + |bits(θ)|
where:
  - θ is the codec hyperparameter vector across all streams
  - R_stream is the per-stream rate-distortion function
  - bits(θ) is the bit cost of encoding the hyperparameters in the archive header
```

**Status**: ACK — formulation added to Section 4.3.

### Issue 3 (Hotz) — Sequencing ambiguity (parallel vs sequential)

**Concern**: Section 4 frames "top 3 paradigm shifts move 1.05 → 0.50" implying SEQUENTIAL impact. But Section 8.1 schedules all 3 in parallel (Week 1-4). Either the EVs compound (parallel) or they're sequential. Pick one.

**Severity**: HIGH (affects EV math + dispatch decisions).

**Fix**: Document the dependency chain explicitly:
- β (sensitivity-map) is foundational; unlocks α-at-floor + γ.
- α (mask codec) can fire standalone but at -30% EV without β.
- γ (joint stack) requires α + β as inputs.
- DEPENDENCY ORDER: β → α → γ.
- Section 4.1 add "STACKING ORDER: β → α → γ; standalone α without β regresses ~30% of EV".

**Status**: ACK — dependency chain documented in Section 4.1.

## Counter

**Round 2: 3 issues found. Counter resets to 0/3.**

## Next round

Round 3 with Dykstra + Quantizr + Selfcomp + Ballé rotation.
