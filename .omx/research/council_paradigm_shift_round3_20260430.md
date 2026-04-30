# Council Adversarial Review — Round 3 (Dykstra + Quantizr + Selfcomp + Ballé rotation)

**Date**: 2026-04-30
**Document under review**: `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
**Reviewers**: Dykstra (CO-LEAD, Pareto), Quantizr (adversarial), Selfcomp (collaborative), Ballé (modern neural compression)

## Issues found (4)

### Issue 1 (Dykstra) — Joint-ADMM convergence on non-convex R(D) curves

**Concern**: Section 3.1 KKT condition assumes per-stream R(D) curves are convex. Empirically (per chain audit Step 2), they are sparsely sampled and likely non-convex. Joint-ADMM may not converge to a global optimum; could oscillate or land at saddle.

**Severity**: MEDIUM (paradigm shift γ may underdeliver if ADMM doesn't converge).

**Fix**: Lane 10 Joint-ADMM design must include:
- Anderson acceleration (~30 LOC) for stiff problems.
- Restart logic when KKT residual oscillates.
- Adaptive penalty schedule (already in Round 11 Q4B fix).
Document convergence safeguards in Section 4.1 paradigm shift γ.

**Status**: ACK — convergence safeguards documented in Section 4.1.

### Issue 2 (Quantizr) — Self-Compress NN transfer assumption

**Concern**: Section 4.2 paradigm shift ε cites arXiv:2301.13142 with "3% bits + 18% weights remaining". But the original demonstration was on ImageNet classification, not regression-style renderer training. The transfer assumption is load-bearing for the predicted -0.05 to -0.10 score.

**Severity**: HIGH (could nullify paradigm shift ε).

**Fix**: Add caveat to Section 4.2 paradigm shift ε:
- "Self-Compress NN result on classification; renderer is regression."
- "Transfer assumption requires empirical validation on Lane G v3 retrain."
- "Predicted band conservative due to transfer risk; could be -0.02 to -0.05 instead of -0.05 to -0.10."

**Status**: ACK — caveat added.

### Issue 3 (Selfcomp) — Paradigm shift δ conflates two different ideas

**Concern**: Selfcomp's "6th paradigm shift" (joint-trained continuous-canvas mask + renderer) is bundled with paradigm shift δ (Hassabis joint end-to-end). They're related but DIFFERENT:
- Selfcomp's shift: mask-encoder + renderer joint (4-week dev).
- Hassabis's shift: also pose joint (6-month moonshot).
Distinguish to clarify dev cost + EV.

**Severity**: LOW (clarity / cost-budget impact only).

**Fix**: Section 4.2 split paradigm shift δ into:
- **δ1**: Joint-trained mask-renderer (Selfcomp's 6th, 4-week dev, $50 GPU).
- **δ2**: Full {mask + renderer + pose} joint (Hassabis, 6-month moonshot, $100-300 GPU).

**Status**: ACK — split into δ1/δ2 in Section 4.2.

### Issue 4 (Ballé) — Hyperprior cost/EV bands inconsistent across sections

**Concern**: Lane 20 Ballé hyperprior is mentioned at multiple cost/EV bands across Sections 4 / 6 / 8. The numbers don't match (Section 4 says -0.01 to -0.03; Section 6 says -0.005 to -0.025; Section 8 says -0.01 to -0.03). Pick one canonical band and reference it.

**Severity**: LOW (clarity / consistency).

**Fix**: Canonical Ballé hyperprior band = **-0.01 to -0.03 score on streams ≥30KB**. Update Sections 6 and 8 to reference Section 4 canonical.

**Status**: ACK — canonicalized.

## Counter

**Round 3: 4 issues found. Counter resets to 0/3.**

## Next rounds (clean-pass attempts)

Rounds 4-6 will rotate remaining council voices and verify all Round 1-3 fixes incorporated. Counter advances if 0 issues found per round.

**Goal: 3 consecutive clean passes (Rounds 4 + 5 + 6) to clear the gate.**
