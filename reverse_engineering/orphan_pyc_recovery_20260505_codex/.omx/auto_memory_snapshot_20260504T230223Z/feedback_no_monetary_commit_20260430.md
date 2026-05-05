---
name: NO MONETARY COMMIT — budget caps lifted, dispatch freely on EV merit
description: 2026-04-30 ~10:00 CDT user mandate: "remember there is no monetary commit anymore". Budget caps removed for this session and going forward. Per-dispatch $0.50 cap, $5 single-recovery cap, $15 session cap, $24 IMP dispatch cap, $5 contest-CUDA wave cap — ALL LIFTED. Future agents should spawn GPU work freely on EV merit, NOT gated by dollar thresholds. Still apply due-diligence (predicted band, kill criteria, harvest discipline) — but DO NOT pause for "approve $X" gates anymore.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The rule

User: "remember there is no monetary commit anymore"

## What changes

- **Per-recovery dispatch cap**: REMOVED (was $0.50 default)
- **Single-recovery escalation cap**: REMOVED (was $5)
- **Session budget cap**: REMOVED (was $15)
- **Lane 17 IMP $24/$25 cap discussion**: APPROVED at user level (already running)
- **Contest-CUDA wave budget**: REMOVED (was $5)
- **PSD-LumaSkip $3 dispatch budget**: REMOVED
- **Engineering recovery autonomous mode**: now operates without dollar gates

## What still applies (from CLAUDE.md non-negotiable)

- Pattern A nohup detach for >3min commands (NEVER `Bash run_in_background:true`)
- NEVER claim `[contest-CUDA]` from MPS/CPU (still strict)
- Predicted score bands MUST be documented BEFORE dispatch (still strict)
- Kill criteria MUST be documented BEFORE dispatch (still strict)
- Harvest discipline within 24h for Modal `.spawn()` (still strict)
- Vast.ai destroy-after-harvest (still strict — no idle instances)
- All commits via `tools/subagent_commit_serializer.py` (still strict)
- EMA + eval_roundtrip on training paths (still strict)

## How to apply

When agent encounters a dispatch decision:
- BEFORE: "Cost = $X. If $X > $0.50, pause for user approval."
- NOW: "Cost = $X. Predicted EV = Y. If EV > 0 and predicted band makes sense, dispatch. Document cost in memory."

When chaining experiments:
- BEFORE: "Stop at $15 session cap"
- NOW: "Continue until session-context constraint or natural stopping criterion (kill verdict, all-recoveries-attempted, etc.)"

## Why the rule exists

Multiple sessions have accumulated budget thresholds that were appropriate in earlier session phases when:
- Budget was tight (Modal credits exhausted at $0)
- Vast.ai $25 starting credit was the only budget
- Operational discipline was being established

As of 2026-04-30, the project has:
- Modal credits ~$70 reserve from billing reload
- $500 user-approved discretionary spend
- Vast.ai active credit pool
- Established operational discipline (Pattern A nohup, harvest, destroy-after, etc.)

The budget thresholds are now FRICTION, not guardrails. The user is making the policy call that EV-driven dispatch is more valuable than per-action approval gates.

## Cross-refs

- project_quota_incident_2_recovery_state_20260430_1000.md
- feedback_budget_30_day_team_parallel_20260429.md (the original budget reset)
- feedback_production_hardened_standard_definition_20260430.md
