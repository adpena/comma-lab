---
name: There is NO $24 Vast.ai cap — user correction 2026-05-01
description: User correction 2026-05-01: "there is no $24 cap I told you that". The "$25 total / $24 hard cap" line in CLAUDE.md "Vast.ai deployment — non-negotiable" is stale or never reflected actual policy. Stop sizing dispatch decisions or "spend % of cap" claims against $24.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The user has explicitly corrected: **there is no $24 cap on Vast.ai.** Don't size dispatches against it.

**Why:** The CLAUDE.md line "$25 total ($24 hard cap in deploy script)" reflects either (a) a historical credit grant that has since been replenished or removed, (b) a deploy-script flag that doesn't reflect actual budget policy, or (c) text the user wants ignored. The user has previously said "we have a budget of one month of development time because we have a team to parallelize" (per `feedback_budget_30_day_team_parallel_20260429.md`) — the budget is meaningfully larger than $24.

**How to apply:**
- Stop including "% of $24 cap" or "$24 hard cap" framings in dispatch memos.
- Stop using $24 as a constraint when ranking dispatches by EV.
- When a dispatch needs cost mention, state the cost itself ($1.65, $2.00, etc.) and let operator decide.
- The Modal-A10G OOM and Vast.ai-4090 cost-per-hour facts remain accurate — just don't bound spend against $24.
- Per CLAUDE.md "Executing actions with care": still confirm before $-spending — but the constraint is "operator approval", not "fits under $24".

**Surfaced from:** project_beta_fisher_dispatch_launch_ready_20260501.md (had "0.8-1.6% of budget" framing), project_lane_19_dispatch_launch_ready_20260501.md, project_lane_17_imp_dispatch_launch_ready_20260501.md, project_shannon_floor_execution_state_checkpoint_20260501.md (Wave 1 summary). All of these contained "16% of $24 cap" or similar text and should be re-stated as just-the-cost.

**Cross-refs:**
- CLAUDE.md "Vast.ai deployment — non-negotiable" (the source of the stale cap claim — leave as-is unless user authorizes the edit)
- `feedback_budget_30_day_team_parallel_20260429.md` (the user's prior expanded-budget signal)
- `project_vastai_balance_65_indulgent_strategy_20260429.md` ($65 balance reference)
