---
name: Deploy gate CLEARED 2026-04-28 PM — 5 Lane J ready for Vast.ai
description: Round 28 + Round 29 + Round 30 all PASS = 3 consecutive clean passes per CLAUDE.md gate. 5 Lane J lanes ready (J-JBL/J-NWC/J-NWCS/J-NWCS-EC/J-IMP) totalling $48.50 well under $200 budget. Cycle 1 dispatch order: J-JBL first (cheapest+highest-confidence), then J-NWC + J-IMP + J-NWCS in parallel, finally J-NWCS-EC after J-NWCS lands.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Gate clearance verified

Per CLAUDE.md "3 consecutive clean passes required before code is cleared for deployment":
- Round 28 PASS (commit `8c80165b` Lane J-NWCS-EC)
- Round 29 PASS (commits `4af8a68b` ARCHIVE_BYTES + `8c80165b` composition)  
- Round 30 PASS (commits `f94e924b` Round 29 advisory + all preceding)

## Lanes ready for Vast.ai

| Lane | Cost | Wallclock | Predicted band | Wedge | Anchor |
|------|------|-----------|----------------|-------|--------|
| Lane J-JBL | $1.50 | 8h | [0.92, 1.02] | SegNet (38%) | Lane G v3 |
| Lane J-NWC | $5.00 | 24h | [0.92, 1.02] | Rate (44%) | Lane G v3 |
| Lane J-NWCS | $8.00 | 14h | [0.85, 0.98] | Rate (44%) | Lane G v3 |
| Lane J-NWCS-EC | $9.00 | 14h | [0.78, 0.92] | Rate+residuals | J-NWCS output |
| Lane J-IMP | $25.00 | 60h | [0.85, 1.00] | Rate moonshot | Lane G v3 |
| **Total Cycle 1** | **$48.50** | **60h max parallel** | various | — | — |

## Dispatch order

**Wave A (parallel, 4 lanes, $39.50, ~60h max):**
- Lane J-JBL — cheapest, highest-confidence SegNet attack
- Lane J-NWC — independent rate attack (different mechanism than J-JBL)
- Lane J-IMP — moonshot, longest wallclock (60h), START FIRST
- Lane J-NWCS — independent rate attack via sensitivity codec

**Wave B (depends on Wave A):**
- Lane J-NWCS-EC ($9) — composes J-NWCS output with EC corrections after J-NWCS lands

## Deployment mechanics

- All deploy scripts at `scripts/remote_lane_j_*.sh` are gate-cleared + executable
- V6 launcher at `scripts/launch_lane_on_vastai.py` handles full lifecycle
- Per memory `feedback_codex_sandbox_blocks_vastai_dns_20260428` — Vast.ai launches MUST come from parent shell (not subagent)
- Per memory `feedback_bash_harness_kills_long_running_tasks_20260428` — split-phase deploy or run_in_background to avoid 5-min SIGURG kill
- Per memory `feedback_per_instance_verify_pattern_20260428` — `scripts/verify_vast_instances.py` for monitoring + auto-destroy stale

## Stack predictions (per docs/stacking_architecture.md)

If all Wave A + Wave B land cleanly:
- **Conservative**: J-JBL renderer + J-NWC encoder = [0.85, 0.95]
- **Aggressive**: + Lane MOS + Lane EC = [0.65, 0.85]
- **Moonshot**: J-IMP renderer + J-NWCS encoder + J-DCAE mask + J-EFD distill = [0.40, 0.65] — could PASS Quantizr 0.33

## Next session continuation

If session ends mid-deploy:
1. Check `.venv/bin/vastai show instances` for live state
2. `scripts/verify_vast_instances.py` — health check
3. Memory `project_signal_loss_audit_20260428` for the dead-codex pattern (don't repeat — use Claude general-purpose subagents not codex CLI)

## Cross-references
- `feedback_compute_budget_hundreds_of_dollars_20260428` — $200-500 budget context
- `.omx/research/jack_skunkworks_segnet_rate_research_20260428.md` — Jack's full research synthesis
- `docs/stacking_architecture.md` — composition rules
- `project_lane_g_v3_landed_1_05_20260428` — current frontier anchor
