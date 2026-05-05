---
name: Vast.ai launcher returns "success" BEFORE the lane script actually starts — silent invocation failures
description: 2026-04-28 discovered W/K/OS-V2 deploy failures. SSH succeeded, repo cloned, env set up, BUT lane script `bash scripts/remote_lane_*.sh` never executed. Launcher reported success because clone completed. ~$2.50 burned for 0 work across 3 instances. Need watchdog or post-launch heartbeat verification.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The bug

`scripts/check_vastai.py` / `src/tac/deploy/vastai/client.py` launch path:
1. SSH to instance ✓
2. git clone /workspace/pact ✓
3. `pip install` deps ✓
4. Send `bash scripts/remote_lane_X.sh` command ←— SILENT FAILURE possible here
5. Disconnect
6. Register instance to `vastai_active_instances.json`

If step 4 fails or the lane script exits before forking the heartbeat background process, the instance enters a state where:
- `actual_status` = "running" (container running)
- `gpu_util` = 0 (no Python process holding GPU)
- `mem_usage` ≈ 0 MiB
- NO heartbeat.log on disk
- NO tmux session
- NO `lane_*_results/` directory

But the launcher reports success because steps 1-3 succeeded.

## Symptom evidence (2026-04-28)

| Instance | Lane | Hours idle | $ wasted | Confirmed via |
|----------|------|-----------|---------|----------------|
| 35739770 | W Iceland | 2.2h | $0.75 | No heartbeat.log, no tmux, no results dir |
| 35739771 | K Denmark | 2.2h | $0.75 | Same |
| 35739773 | OS-V2 NC | 2.2h | $0.75 | SSH key denied (worse: never even logged in successfully) |

## Why GPU util API was misleading

`vastai show instances --raw` returns `gpu_util` from a snapshot poll that can be 1-5 minutes stale. **Heartbeat.log freshness is the only ground-truth readiness signal.** Several earlier instance investigations (Lane I, Lane G v3) gave contradictory results between API and heartbeat.

## How to apply

1. **Post-launch verification step** in launch path: poll the instance for heartbeat.log freshness within 5 minutes. If absent, ABORT and destroy.
2. **Watchdog script** that periodically scans every instance in `vastai_active_instances.json` for heartbeat freshness; alert / auto-destroy if stale > 30 min.
3. **Treat heartbeat.log as canonical readiness signal**, not gpu_util API. Update `scripts/check_vastai.py` and any monitoring scripts.
4. Consider adding **Check 41 to preflight**: every `remote_lane_*.sh` must write a heartbeat AND a `lane_started_at_utc.txt` sentinel file as its first action — the launcher polls for the sentinel.

## What this caught earlier

- Lane G v3 instance was idle for 4h+ AFTER auth eval completed at 11:37Z — would have been auto-destroyed by a watchdog, saving ~$1.20 in burn
- Lane V-V2 (in flight today) hasn't been re-checked yet; may be similar pattern

## Cost paranoia rule (already in memory `feedback_vastai_cost_paranoia`)

This finding reinforces it: **destroy idle instances within 30 min of confirmed completion or non-start.** Cost burn is mostly orphaned-instance, not heavy training.

## Cross-references
- `feedback_vastai_cost_paranoia` — cost cap discipline
- `feedback_no_wasted_resources` — every $ must produce measurement
- `feedback_remote_code_parity_required` — heartbeats not tmux sessions
- `feedback_canonical_remote_bootstraps` — provenance + heartbeat as canonical pattern
