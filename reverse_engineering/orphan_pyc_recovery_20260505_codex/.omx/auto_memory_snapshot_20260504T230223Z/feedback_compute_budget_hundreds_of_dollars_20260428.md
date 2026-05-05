---
name: Compute budget — HUNDREDS of dollars (NOT $25)
description: 2026-04-28 user explicitly clarified budget is "hundreds of dollars" — multi-day H100 / 4090 runs OK, lanes up to $50 affordable, full IMP-style 10-20 cycle experiments fit. Updates the stale $25 hard cap that's been propagating through CLAUDE.md + memory files since 2026-04-15.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The rule

Compute budget is **$200-$500** on Vast.ai (per user 2026-04-28). NOT $25. The $25 hard cap was an early-session estimate; the user has since multiplied the budget. This MUST be reflected in:

- **Lane cost-benefit math**: lanes up to $50/each are now affordable. A $30 IMP-style 10-cycle experiment is in budget.
- **Hardware selection**: H100 ($3.30/hr), A100 ($1.50/hr), RTX 4090 ($0.25/hr) all on the table. Multi-day runs OK.
- **Parallel batch sizes**: 3-5 simultaneous Vast.ai lanes is normal, not extravagant.
- **Subagent prompts**: any reference to "$24 hard cap" or "$25 budget" is OBSOLETE. Update prompts before dispatching.
- **Lane evaluations**: when assessing whether a paper is "applicable", a $40-80 cost is NOT automatically disqualifying.

## Why this matters

Multiple subagent evaluations this session (LTH, CARVE) factored in the stale $25 cap when computing applicability. Specifically the LTH evaluation said:
> "IMP to 80% sparsity: 5 cycles × 16h = $20 — Hardware budget: $25 cap. Full IMP exceeds budget by 1.6-2.4×."

That math is OUT OF DATE. At $200-$500 budget, full IMP (10 cycles × 16h × $0.25/hr = $40) is well in budget. The Lane J-IMP proposal Jack landed today ($25/60h on 4090) is now THE moonshot rate-attack lane.

## How to apply

1. **Before dispatching subagent prompts that mention budget**: replace "$25" / "$24 hard cap" with "$200-$500 budget; multi-day runs OK".
2. **In council deliberations**: cost-prohibitive arguments must clear the $50/lane bar, not the $5/lane bar.
3. **In CLAUDE.md GPU budget section**: the "$25 total ($24 hard cap)" line is STALE. Should be updated to "$200-$500 total budget; per-lane cost cap $50 unless explicitly approved".
4. **In auto-destroy logic in tac.deploy.vastai.cli**: the $24 trigger should be raised to $50 per-instance OR $200 total.

## When to revisit

If the user revises the budget DOWN (unlikely given current trajectory), re-anchor. If the budget revises UP further (e.g., a deadline-week splurge), update accordingly. Always confirm before treating as deployed-budget-truth.

## Cross-references
- `feedback_vastai_cost_paranoia` — historical $25 paranoia (now SUPERSEDED for budget calculations; the cleanup-stale-instance discipline still applies)
- `project_lth_evaluation_RESEARCH_PARK_20260428` — needs revision under new budget (LTH math may now favor a Lane J-IMP dispatch)
- CLAUDE.md "GPU budget and compute resources" section — needs update
- `project_dead_codex_recovery_inventory_20260428` — recovery wave was budget-conscious; future wave can dispatch larger lanes
