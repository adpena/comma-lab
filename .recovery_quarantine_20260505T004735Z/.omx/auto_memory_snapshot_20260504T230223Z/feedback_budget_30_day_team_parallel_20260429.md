---
name: BUDGET RESET — 6 months dev time + team parallelization (was 4 days solo) + post-contest extension
description: 2026-04-29 PM. User correction (final): "actually we have a six month development time because we will keep pushing after the contest and submitting and polishing and adding to and extending our writeup/paper". 30-day was the team-parallel-up-to-contest budget; 6-month is the full project arc including post-deadline paper polish + extensions. Goal NON-NEGOTIABLE: "optimal lowest theoretical floor possible".
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What changed

CLAUDE.md and prior memories operated under "competition deadline May 3, 2026 + solo work" assumption. The user clarified the actual constraint:

- **Time**: ~30 days (not 4)
- **People**: a team in parallel (not solo Claude orchestrating)
- **Budget**: hundreds of dollars / multi-day H100/A100/4090 OK
- **Implication**: anything previously killed for "not 4-day feasible" is back on the table

## Decisions to RESURRECT (previously deferred for 4-day budget)

1. **Joint ADMM** (Boyd grand council voice) — multi-day convex optimization across rate/seg/pose constraints. Originally killed: "not a four-day plan." Under 30-day budget: feasible.
2. **Wavelet-domain mask coding** (Mallat grand council voice) — was deferred as "post-deadline paper lane." Now: viable redesign for STC.
3. **C3 / coordinate-MLP residual codec** (Schmidhuber rejected for code complexity under deadline) — viable now.
4. **NeRV mask codec** (codex top-2 STC redesign at 44% endorse, 2-3 days) — feasible.
5. **Stored-flow + RAFT residual** (codex top-3 at 35% endorse, 2-4 days) — feasible.
6. **Lane J-IMP** (10-20 cycle full IMP at $25/lane) — now in budget.
7. **Full DARTS-S sweep** (was restricted to 3 configs for cost) — can expand under 30-day budget.
8. **STC max-rigor "ALIVE WITH REDESIGN" (45% endorse, 4-12h)** — feasible AND can run in parallel with the Hybrid AV1+residual top-1 redesign.

## What stays prioritized

1. SC++ v4 / q_faithful_v3 base — already in flight on Modal
2. Ω-W water-filling export — ready
3. Restricted DARTS-S — in flight on Vast.ai
4. LCT — fixed and ready (8/8 tests)
5. Hybrid AV1+residual STC (codex top-1 STC redesign) — IMPLEMENT FIRST in extended budget

## Implication for portfolio probability

Sub-0.30 portfolio probability under 4-day budget: 24-34% (per codex grand council)

Under 30-day budget:
- More base lanes can land at <0.40 (compounding the OR-probability)
- Multiple stack components can be built and validated independently
- Adversarial review per landing has ZERO time cost — discovers bugs before deploy
- The "fix-lands-in-helper-but-not-callsite" bug class can be systematically swept (Sherlock Holmes audit in flight)
- **Estimated revised sub-0.30 probability: 50-65% central, band 35-80%**

## Process changes

1. EVERY landing gets 3-clean-pass adversarial review before dispatch (was sometimes deferred for speed). Now mandatory.
2. EVERY major design decision gets a max-rigor codex consultation (already doing this, now affordable).
3. CUDA validation BEFORE strategic kill or promote — never use MPS results for kills (CLAUDE.md non-negotiable now strictly enforced).
4. Parallel codex sessions: continue dispatching multiple via Pattern A detach.
5. Multiple Modal/Vast.ai lanes can run concurrently — no need to serialize.
6. Recursive Sherlock-Holmes audits for entire bug classes (e.g., the GP `baseline_poses` fix-not-in-callsite was 1 of N similar; sweep them all).

## How to apply

- Stop saying "4 days left"
- Re-evaluate every "killed for time" decision against the new budget
- The grand council's top-5 ranking (4-day) is no longer the operative ranking; under 30-day budget, top-10+ is operative
- Schedule wakeup intervals can be longer (15-30 min for codex sessions, 1-3h for Modal lanes)
- Multi-day Modal lanes are now practical, not "indulgent"

## Cross-refs

- project_grand_council_final_designs_20260429.md (the 4-day council ranking — STILL VALID at the top, but expanded list now relevant)
- project_stc_redesign_verdict_20260429.md (top-3 STC redesigns — ALL three now buildable in parallel)
- feedback_compute_budget_hundreds_of_dollars_20260428.md (earlier $200-500 budget acknowledgement)
