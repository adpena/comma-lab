---
name: Per-instance verify pattern — heartbeat freshness + crash-signal grep + auto-destroy
description: 2026-04-28 built `scripts/verify_vast_instances.py` to enforce the watchdog from feedback_vastai_launch_returns_success_before_lane_starts. For each tracked instance: SSH + heartbeat-mtime + grep for Traceback/FATAL/CUDA_ERROR/RuntimeError. Classifies HEALTHY/IDLE/CRASHED/UNREACHABLE/GONE. Optional --auto-destroy-stale stops cost burn. Caught Lane I crash + auto-destroyed in first run.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What it does

`scripts/verify_vast_instances.py` reads `.omx/state/vastai_active_instances.json` and for each instance:

1. **vastai metadata**: confirm instance still exists + grab gpu_util + ssh details
2. **SSH heartbeat probe**: `find /workspace/pact -name 'heartbeat.log' -printf '%T@\n' | sort -n | tail -1` → freshness in minutes
3. **Crash-signal grep**: `grep -E 'Traceback|FATAL|CUDA_ERROR|RuntimeError|out of memory'` on run.log + train.log + auth_eval.log
4. **Classification**:
   - `HEALTHY` — heartbeat < 30 min old, GPU util > 5%
   - `IDLE` — heartbeat > 30 min old OR (GPU util < 5% AND heartbeat > 5 min old)
   - `CRASHED` — crash signal found in any log
   - `UNREACHABLE` — SSH failed
   - `GONE` — vastai show returned no data

## Usage

```bash
# Just report (no destruction)
.venv/bin/python scripts/verify_vast_instances.py

# JSON for piping into watchdog
.venv/bin/python scripts/verify_vast_instances.py --json

# Auto-destroy IDLE/CRASHED instances older than threshold
.venv/bin/python scripts/verify_vast_instances.py --auto-destroy-stale --stale-minutes 30
```

Exit code: 0 if all HEALTHY, 1 if any IDLE/CRASHED/UNREACHABLE.

## What it caught (2026-04-28 first run)

```
✗   35733831 lane_i_overnight                 CRASHED      hb=112.7min util=0%
    crash: Traceback (most recent call last):
✓   35733832 lane_v_overnight                 HEALTHY      hb=0.5min   util=46%
✓   35736027 lane_m_v2_overnight_v2           HEALTHY      hb=0.7min   util=0%
```

Lane I was idle 1.9h after Stage 3 export crashed (parametrize-strip mismatch — separately documented in `project_lane_i_crashed_parametrize_strip_20260428`). Caught it + auto-destroyed in one command. Saved ~$0.28/hr × however long until human notice.

## Hardening this enables

- **Cron / systemd timer**: schedule every 10-30 min on the local machine
- **Cost paranoia enforcement**: per `feedback_vastai_cost_paranoia` "destroy idle instances within 30 min of confirmed completion or non-start" — now mechanizable
- **Tracker hygiene**: `GONE` classification flags stale tracker entries that were destroyed externally

## Future enhancements

- Add SLACK/email notification on CRASHED detection
- Add log-tail capture before destroy (so post-mortem is preserved)
- Add per-lane-script-type heuristics (e.g., auth eval phase legitimately has GPU=0)
- Extend to detect zombie Python processes (training PID alive but no GPU activity)

## Cross-references
- `feedback_vastai_launch_returns_success_before_lane_starts` — the W/K/OS-V2 motivating incident
- `feedback_vastai_cost_paranoia` — cost cap discipline
- `project_lane_i_crashed_parametrize_strip_20260428` — first crash this caught
- `feedback_canonical_remote_bootstraps` — heartbeat as canonical signal
