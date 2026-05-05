---
name: Vast.ai spot is UNRELIABLE — multi-instance simultaneous preempt observed (5 instances destroyed at once)
description: 2026-04-30 ~15:30 CDT incident. ALL 4 active Vast.ai instances (HM-S 35885106 + Lane 19 35899850 + 35906669 + 35907873) DISAPPEARED simultaneously. HM-S was at epoch 504/600 (84%) with promising metrics (pose=0.006, seg=0.016) — RUN LOST. Lane 19 retry was actively training — LOST. Net: ~$5-10 burned + multiple in-flight runs LOST. PIVOT: Use Modal/Lightning for long-running training, Vast.ai only for cheap fast (<2h) probes.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What happened (verified 2026-04-30 ~15:30 CDT)

Earlier in session (~12:00 UTC), Vast.ai had 4 active 4090s:
- 35885106 HM-S (epoch 504/600, healthy training)
- 35899850 Lane 19 logit-margin retry (49% util, training)
- 35906669 (~80% util, identity unclear)
- 35907873 (~77% util, identity unclear)

**At ~21:00 UTC**: API check shows ZERO instances active. All 4 destroyed simultaneously.

Subsequent #314 dispatch spawned 35925274 fresh, then API briefly showed 0, then 1 (the new one).

## Root cause analysis

Most likely: **Vast.ai spot preempt swept the host(s) simultaneously**. Vast.ai spot pricing means hosts can be reclaimed when on-demand bidders out-bid us. If multiple of our instances were on the same host (or hosts owned by the same provider), a single bid event destroys multiple.

Less likely: external destroy (no agent action triggered this — the SSH/destroy commands from earlier were SIGURG-144'd before any actual destroys).

## Cost of incident

- HM-S sunk: ~$2.40 (9.5h × $0.26/hr) + LOST contest_auth_eval result
- Lane 19 retry sunk: ~$3+ (running multi-hour) + LOST contest_auth_eval result
- 35906669 + 35907873 sunk: ~$2-4 + LOST contest_auth_eval results
- **Total: ~$8-10 burned, 4 in-flight scientific runs LOST**

## Lessons + new mandate

### MANDATE: Vast.ai spot ONLY for cheap fast dispatches (< 2h training)
- Lane PFP16 build/eval (~10 min) — OK for Vast.ai
- Lane 20 Ballé encode/eval (~40 min) — OK for Vast.ai
- Lane 8 multi-pass quick check (~30 min) — OK for Vast.ai
- HM-S, Lane 17 IMP, Self-Compress NN, joint training (>2h) — **NOT Vast.ai. Use Modal/Lightning.**

### REASSIGN long-running lanes
| Lane | Old platform | NEW platform | Reason |
|---|---|---|---|
| Lane 17 IMP 10-cycle (80h) | Vast.ai 4090 | Lightning H100/L40S | Persistent, no preempt |
| HM-S retrain (5h) | Vast.ai 4090 | Modal A10G | Persistent function call |
| Lane 19 logit-margin (5h) | Vast.ai 4090 | Modal A10G | Persistent |
| Self-Compress NN (8h+) | Vast.ai → Modal/Lightning | — | Heavy training |
| Joint renderer-scorer (12h+) | Vast.ai → Modal/Lightning | — | Heavy training |

### CHECK current state pattern
- `vastai show instances` returns 0 → either preempt or no instances spawned
- The "active_dispatches.md" was STALE — it didn't reflect post-preempt reality
- **NEW RULE**: Before dispatching, ALWAYS run `vastai show instances` to ground-truth current state, never trust `.omx/state/active_dispatches.md` alone.

### Add watchdog
- Spawn a background watchdog cron that polls `vastai show instances` every 15 min
- If active count drops by >1 simultaneously, alert + harvest provenance from local logs
- Auto-update active_dispatches.md to reflect reality

## Cross-refs

- feedback_vastai_nvdec_roulette_pivot_to_modal_20260429.md (similar pattern: Vast.ai unreliability)
- feedback_modal_spawn_result_cache_pattern_20260429.md (Modal also has 24h cache GC, but more reliable runtime)
- feedback_lightning_ai_ssh_credentials_20260430.md (Lightning persistent Studio)
- project_quota_incident_4_recovery_state_20260430_1530.md (concurrent recovery state)
