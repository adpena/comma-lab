---
name: Bash tool harness kills long-running tasks at exit 144 — Vast.ai launcher incompatible
description: 2026-04-28 5 separate launch attempts via scripts/launch_lane_on_vastai.py all failed at exit 144 (SIGURG-like) regardless of run_in_background flag. The launcher CREATED 5 Vast.ai instances (zombie-creating) before being killed. ~$0.30 wasted across 5 zombies. Bash tool harness has aggressive kill behavior incompatible with multi-minute foreground bash subprocesses. Launcher works ALGORITHMICALLY but cannot complete in this session's environment.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The bug

`scripts/launch_lane_on_vastai.py` is a 9-stage Python script that:
1. Searches Vast.ai (~5s)
2. Creates instance (~5s)
3. Waits 90s for boot
4. Verifies SSH ready (up to 3 min poll)
5. Builds tarball (~30s)
6. SCPs tarball (~2-3 min)
7. Extracts on remote (~10s)
8. CUDA probe (~5s)
9. Starts tmux (~10s) + polls heartbeat (up to 8 min)

Total realistic wallclock: 5-13 min. Logic is correct (verified by dry-run + manual stage execution earlier in session).

**But**: the bash tool harness aggressively kills subprocesses at exit code 144 long before stage 9 completes. Even with `--run_in_background=true` and `--timeout=900000` (15 min), the task is killed within seconds-to-minutes:
- Foreground call (15-min timeout): exit 144 after ~14 min (Lane Ω-V2 first try)
- Background call (15-min timeout): exit 144 with NO output captured (Lane Ω-V2 retry)
- Background call: same 144 across 5 launches

But the launcher's Stage 1 (vastai create instance) DID complete before the kill — leaving zombies.

## What this caused (2026-04-28)

5 launches × successful stage-1 creation + harness kill = 5 zombies:
- 35749872 lane_omega_v2 (5.9 min before destroy)
- 35749876 lane_ec_cycle_1 (5.9 min)
- 35749878 lane_saug_v2 (5.8 min)
- 35749892 lane_i_retry (5.4 min)
- 35749893 lane_w_hard_pair (less)

Total cost: ~$0.30 across 5 zombies before manual destroy.

## Why exit 144?

Exit code 144 = 128 + 16 = SIGURG. Probably the Claude Code bash harness sending SIGURG to subprocess on some internal timeout/deadline that's tighter than the configurable `--timeout`. Possibly related to how the harness manages concurrent background tasks.

## Symptom signature

- bash background task with no output (`wc -l logfile` = 0)
- exit code 144
- Underlying Python script created some side effect (Vast.ai instance) but didn't complete its main loop

## Mitigation

The launcher is correct in logic; the issue is harness compatibility. Options:

### Option A — split launcher into harness-compatible phases
- **Phase 1** (instant, parent-foreground): vastai create + register tracker → returns instance ID + SSH details
- **Phase 2** (3-4 min, parent-foreground): SCP tarball + tmux execute → returns success
- **Phase 3** (separate, async): wakeup → run `verify_vast_instances.py` to confirm heartbeat
- Each phase completes in < 5 min, harness-friendly

### Option B — run launcher from external shell (not via bash tool)
- User invokes `python scripts/launch_lane_on_vastai.py ...` directly from their own terminal
- Bash tool harness not involved
- Loses the Claude Code orchestration benefit

### Option C — make launcher run remote-side
- Use `vastai create instance --onstart-cmd '<self-contained bootstrap>'` 
- The onstart command does git-clone + setup + lane execution all on the remote
- Parent's bash tool only sends the create command (~5s)
- Limitation: requires public clone access (we have private SSH-only repo) — would need an HTTPS clone with token

## Decision (this session)

Stop attempting Vast.ai launches from this session. Document the issue. Hardening the launcher is TIER-1 follow-up for next session. Lane G v3 = 1.05 frontier remains the secured win of the day.

## Cross-references
- `feedback_canonical_parent_shell_launcher_20260428` — the launcher code
- `feedback_cycle_1_launch_postmortem_20260428` — earlier failure modes
- `feedback_codex_sandbox_blocks_vastai_dns_20260428` — separate codex sandbox issue
- `feedback_per_instance_verify_pattern_20260428` — verify_vast_instances.py companion
