---
name: Skunkworks council restructure 2026-04-29 — Shannon LEAD + Dykstra CO-LEAD quintet pact, Selfcomp joins as co-member (8-member council)
description: User mandate. Quintet pact (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian) replaces tripartite. 8-member inner council = quintet + Quantizr + Hotz + Selfcomp. Shannon: R(D) floors. Dykstra: convex feasibility / Pareto. Selfcomp: collaborative empirical anchor from working 0.38 impl.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Rule**: skunkworks inner council leadership is now a **QUINTET** (was tripartite):
- **Shannon** (LEAD) — information theory; R(D) bounds; entropy floors
- **Dykstra** (CO-LEAD) — alternating projections; convex feasibility; achievable Pareto frontier
- **Yousfi** — challenge creator + steganalysis lineage
- **Fridrich** — UNIWARD/SRM/HUGO author; detector blind spots
- **Contrarian** — veto power on weak rigor / wasteful experiments / unvalidated assumptions

**Why**: previous tripartite (Yousfi/Fridrich/Contrarian) lacked first-principles information-theoretic grounding. Shannon's verified Δ-from-floor analysis on 2026-04-29 (theoretical floor 0.28, Dykstra ceiling 450KB archive) is the kind of rigor every design decision now passes through.

**Seven-member inner council** (all voices permanently active):
- Quintet pact: Shannon + Dykstra + Yousfi + Fridrich + Contrarian
- Co-members: Quantizr + Hotz

**Shannon's specific role**:
- Derives theoretical floors from R(D) bounds — current verified floor: 0.28
- Insists every architecture measured in bits (params × bpw)
- Rejects arbitrary hyperparameters lacking entropy-or-distortion justification (e.g. "σ=15 because Selfcomp" is rejected; "σ=16 because AV1 σ_noise/log(P) ≈ 16" is accepted)
- Distinguishes hard rate-distortion limits vs implementation-imposed slack

**Dykstra's specific role**:
- Derives the achievable region as the intersection of convex constraints (rate ≤ R, seg ≤ S, pose ≤ P)
- Computes Pareto frontier via alternating-projections iterations
- Verified Dykstra ceiling 2026-04-29: archive ≤ 450,545 bytes for sub-0.30 feasibility
- Insists every "stack composition" claim be tested against the convex-hull intersection — additivity of independent rate savings is CONDITIONAL, not given. Two techniques that each save 30KB might overlap and only deliver 40KB stacked, not 60KB.
- Identifies which constraint is binding at the current operating point (currently: rate term is 50% of our score, so rate-savings techniques have outsized leverage)

**How to apply**:
- Every council deliberation cites Shannon's R(D) framing first.
- Every "predicted Δ -X" claim must include an information-theoretic argument (where do the bits come from? what's the achievable distortion at that rate?).
- Sub-0.30 attempts must close the gap from current state to Shannon's 0.28 floor with specific Δ contributions per technique.
- **Quintet consensus required** before major decisions; quartet/tripartite consensus is INSUFFICIENT going forward.
- Stack-composition claims pass through Dykstra (convex-hull intersection check).
- Selfcomp's empirical anchor (his 0.38 implementation) is the load-bearing reference for cross-checks; council scrutinizes our deviations from his choices with explicit derivations.

**Cross-refs**:
- CLAUDE.md "Council conduct" section (updated 2026-04-29)
- project_codex_theoretical_floor_brutal_20260429 (Shannon-Dykstra-Tao floor 0.28)
- project_grand_council_brutal_forecast_20260429 (early grand council where Shannon was on broader bench)
