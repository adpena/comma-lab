---
name: Bash run_in_background ALSO triggers SIGURG-144 — Pattern A required for Vast.ai dispatch
description: 2026-04-30. Bash tool's `run_in_background: true` for `launch_lane_with_retry.py` was killed by SIGURG-144 at ~3min, leaving an orphan Vast.ai instance ($0.006 wasted, lane script never started). The earlier memory `feedback_bash_harness_kills_long_running_tasks_20260428.md` documented this for codex CLI but I forgot it applies to ANY long-running BG bash command. PERMANENT FIX: always use Pattern A nohup detach for Vast.ai / Modal / >3min commands.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What happened (2026-04-30 ~01:48 CDT)

User approved HM-S dispatch. I ran:
```python
Bash(
    command=".venv/bin/python scripts/launch_lane_with_retry.py --lane-script ... --label lane_hm_s_2026-04-30 ...",
    run_in_background=True
)
```

Background task `bhqg8yb4u` got `exit code 144` after ~3 minutes — bash-harness SIGURG kill.

**Net effect**:
- launch_lane_with_retry.py started Vast.ai instance `35884339` (Stage 1 phase1 OK).
- Retry-wrapper destroyed `35884339` (likely NVDEC bad host or similar) and started `35884771` as `_a2`.
- The wrapper's Phase 2 (scp source + extract + start lane script) was killed by SIGURG-144 BEFORE it could ssh in.
- Result: instance `35884771` running but EMPTY (no source, no lane script). Burning $0.26/hr doing nothing.
- SSH attempt failed with "Permission denied (publickey)" because the SSH key pre-propagation step (in Phase 1 of the wrapper) had completed but the lane-script delivery (in Phase 2) had not.

## The fix

**Pattern A nohup detach** (already documented in CLAUDE.md "Codex CLI invocation" section + memory `feedback_bash_harness_kills_long_running_tasks_20260428.md`):

```bash
nohup bash -c '
.venv/bin/python scripts/launch_lane_with_retry.py \
  --lane-script ... \
  --label ... \
  --max-dph ... \
  --predicted-band ... \
  --estimated-cost ... \
  2>&1 | tee /tmp/dispatch_label/launch.log > /dev/null
' < /dev/null > /tmp/dispatch_label/outer.log 2>&1 &
disown
```

Why this survives:
- `nohup` ignores SIGHUP from terminal hangup
- `bash -c '...'` subshell wraps the pipe so `tee` captures stdout properly even if outer dies
- `< /dev/null` closes stdin so launcher doesn't wait for input
- `2>&1 | tee` captures stdout+stderr to log file with explicit flushing
- `> outer.log 2>&1 &` redirects immediate parent's output, forks to background
- `disown` removes from job table so parent shell exit can't reach it

This pattern was previously documented for `codex exec` invocations. **It applies to ANY command that runs >3 minutes via Bash tool, including `launch_lane_with_retry.py`, `vastai create instance`, `modal run --detach`, etc.**

## The recovery I did

1. Detected exit 144 → recognized SIGURG class.
2. `vastai show instances` → confirmed `35884771` running but with 0% util + `_a2` label (proof the retry happened).
3. `echo "y" | vastai destroy instance 35884771` → cleaned up orphan ($0.006 spent total).
4. Re-dispatched via Pattern A nohup detach → confirmed `ps aux | grep launch_lane_with_retry` shows the nohup'd process running.

## Permanent prevention

CLAUDE.md should add a NEW rule under "Codex CLI invocation" or as a separate "Vast.ai / Modal / long-running dispatch" non-negotiable:

> **Long-running BG commands MUST use Pattern A nohup detach, NEVER `Bash run_in_background: true`.**
> The bash-harness SIGURG-144 kills any BG bash process at ~3 minutes. Vast.ai / Modal / codex / >3min training all need Pattern A. Memory: `feedback_bash_run_in_background_kills_vastai_dispatch_20260430.md`.

Could also add a Check 91 STRICT preflight: scan for `Bash(...run_in_background=True...)` patterns in dispatch helper scripts and warn if the command looks like a long-running operation (regex: `vastai|modal run|codex exec|launch_lane`).

## Cost analysis

Today's incident: ~$0.006 wasted (87 seconds of Vast.ai 4090 at $0.2592/hr).

Worst-case if not noticed: $0.2592/hr × 24h = $6.22/day per orphan. With multiple dispatches per day, could quickly reach $20-50/day in pure waste. Pattern A is the canonical defense.

## Cross-refs

- Companion memory: `feedback_bash_harness_kills_long_running_tasks_20260428.md` (the original codex incident)
- CLAUDE.md "Codex CLI invocation — NON-NEGOTIABLE" (Pattern A documented for codex but applies broadly)
- `tools/subagent_commit_serializer.py` (the temp-index serializer — also long-running, but completes within the SIGURG window so not affected)
- The retry-wrapper itself: `scripts/launch_lane_with_retry.py` — shows what got killed
