---
name: 19-lane Vast.ai monitoring cadence (2026-04-28 PM mass deploy)
description: 19 lanes deployed in rapid succession at 18:00. Phase2 wakeup at 18:10 (T+10). Then monitoring cadence: T+15 (first verify), T+30, T+60, every 30 min after. Use scripts/verify_vast_instances.py for SSH-based heartbeat probes. Auto-destroy stale (CRASHED/IDLE > 30 min). Memory wedge for crash recovery: replace destroyed lane with same script if priority > 5.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The 19 lanes (18:00 dispatch batch)

| Lane | Instance | Phase1 done | Phase2 due | Cost cap |
|------|----------|-------------|------------|----------|
| J-JBL | 35781543 | ✓ | T+5 | $1.50 |
| J-NWC | 35781594 | ✓ | T+5 | $5.00 |
| J-IMP | 35781603 | ✓ | T+5 | $25.00 |
| J-NWCS | 35781604 | ✓ | T+5 | $8.00 |
| Ω-V2 | 35781754 | ✓ | T+5 | $5.00 |
| SAUG-V2 | 35781755 | ✓ | T+5 | $4.00 |
| W | 35781756 | ✓ | T+5 | $4.00 |
| M-V3 | 35781757 | ✓ | T+5 | $1.50 |
| EC | 35781763 | ✓ | T+5 | $1.00 |
| WC | 35781764 | ✓ | T+5 | $3.00 |
| SZ Phase 2 | 35781765 | ✓ | T+5 | $5.00 |
| EBR | 35781775 | ✓ | T+5 | $2.00 |
| GP | 35781777 | ✓ | T+5 | $1.00 |
| FL | 35781778 | ✓ | T+5 | $1.00 |
| V | 35781781 | ✓ | T+5 | $5.00 |
| T2-DROP | 35781793 | ✓ | T+5 | $1.50 |
| T2-RATIO | 35781799 | ✓ | T+5 | $9.00 |
| EC-V2 | 35781802 | ✓ | T+5 | $1.50 |
| RM | 35781804 | ✓ | T+5 | $1.50 |
| **Total** | — | — | — | **$80** |

Burn rate: $5.10/hr. $200 budget = 39 hours headroom. Most lanes run 8-14h.

## Monitoring cadence

- **T+10** (18:10): phase2 deploy wave (run scp+extract+launch on ready instances)
- **T+15** (18:15): first `verify_vast_instances.py` pass — most lanes still in setup_full.sh
- **T+30** (18:30): second verify pass — most lanes mid-Stage-3 DALI install
- **T+60** (19:00): third verify pass — most lanes should be in lane training
- **Every 30 min after**: continued verify until lanes complete

## Monitoring command

```bash
.venv/bin/python scripts/verify_vast_instances.py --auto-destroy-stale --stale-minutes 30
```

This SSHes into each instance, checks `$LOG_DIR/heartbeat.log` freshness, greps for crash signals, classifies HEALTHY/IDLE/CRASHED/UNREACHABLE/GONE. With `--auto-destroy-stale`, kills CRASHED/IDLE > 30 min old.

## Crash recovery rules

When a lane crashes:
1. Check the lane's `lane.log` and `setup.log` via `vastai logs <id>` for root cause
2. If failure is environmental (NVDEC/DALI/host) — relaunch with same script + new instance
3. If failure is code (typo, NameError, missing import) — fix code first, then redispatch
4. If failure is rate (>$50/instance) — investigate before redispatching
5. Per `feedback_vastai_nvdec_host_variation`: NVDEC failures get auto-handled by Stage 0.5 lightweight probe (saves 5 min DALI install)

## Priority replacement table

If a lane crashes, replace immediately if predicted band overlaps current frontier (1.05) or below:
- HIGH PRIORITY (replace immediately): SZ Phase 2, V (Quantizr replica), J-IMP — all moonshot bands
- MEDIUM PRIORITY (replace if budget): J-JBL, J-NWC, J-NWCS, Ω-V2, SAUG-V2, EBR, WC
- LOW PRIORITY (don't replace if budget tight): RM, T2-RATIO, GP, FL — informational

## Context for future sessions

If session ends mid-monitoring, the cadence + script + recovery rules survive here. The `feedback_per_instance_verify_pattern_20260428` memory entry has the verify script details. The `project_deploy_gate_cleared_20260428_PM` memory entry has the gate-clear context.

## Cross-references
- `feedback_per_instance_verify_pattern_20260428` — verify script details
- `feedback_vastai_nvdec_host_variation` — NVDEC variability handling
- `feedback_vastai_launch_returns_success_before_lane_starts` — phase2 verification needed
- `project_deploy_gate_cleared_20260428_PM` — gate clearance context
- `feedback_compute_budget_hundreds_of_dollars_20260428` — $200-500 budget
