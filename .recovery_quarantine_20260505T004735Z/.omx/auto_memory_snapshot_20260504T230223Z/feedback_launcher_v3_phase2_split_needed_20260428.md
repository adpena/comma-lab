---
name: Launcher V3 phase2-deploy hits harness 5-min limit — needs further split
description: 2026-04-28 Launcher V3 split-mode (phase1 + phase2-wait + phase2-deploy) validated for first 2 phases but phase2-deploy hits exit 144 harness kill at ~3-5 min. Phase2-deploy SCP of 962MB tarball + extract + nohup-detach exceeds 5-min budget. Next session: split phase2-deploy further (phase2-scp + phase2-launch each ≤2 min).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Validation status

| Phase | Status | Wallclock | Harness fit? |
|-------|--------|-----------|--------------|
| phase1 (create + register) | ✓ FULLY VALIDATED | 10-30s | ✓ |
| phase2-wait (status + ssh ready) | ✓ FULLY VALIDATED idempotent | 4-10 min depending on offer | ✓ if called multiple times |
| phase2-deploy (SCP + extract + launch) | ⚠ HITS HARNESS LIMIT | 3-5 min | ✗ exit 144 |

## phase2-deploy timeline (where it dies)

Stage 0 (sanity check): 5s
Stage 1 (start tarball-build thread): 0s, but build runs 22s in background
Stage 1 (SSH re-verify): 1-30s
Stage 3 (Join tarball): 0-220s
Stage 4 (SCP 962MB): 60-180s ← HARNESS HITS HERE
Stage 5 (extract + CUDA probe): 15s
Stage 6 (launch via subshell-detach): 5s

Total: 80-450s. Worst case >5 min.

## Failed detach patterns (2026-04-28)

1. `tmux new-session -d -s lane '...'` — tmux not installed on PyTorch container, apt-get install slow
2. `setsid nohup ... </dev/null >/workspace/run_lane.out 2>&1 & disown` — SSH held output stream, hung
3. `nohup bash ... </dev/null >/dev/null 2>&1 & disown` — same hang
4. `( bash ... </dev/null >/dev/null 2>&1 & ) && echo launched` — works in theory but combined with SCP exceeds harness

The detach pattern itself isn't the bottleneck — the cumulative SCP + extract + detach exceeds harness.

## Real fix (TIER-1 next session)

Split phase2-deploy into 3 sub-phases:

```
phase2-scp        # SCP tarball (~2-3 min) — fits harness
phase2-extract    # Extract + CUDA probe (~15s) — instant
phase2-launch     # Subshell-detach launch (~5s) — instant
```

Or alternatively: build SMALLER tarball. Currently 962MB (after fixing 5.9GB→962MB earlier). Aggressive prune target: ~150MB by removing:
- experiments/__pycache__ (1.5MB but can be removed)
- experiments/results/lane_*_landed (~700KB each, multiple)
- submissions/robust_current/masks*.mkv (2.3MB)
- non-essential lane scripts

Stretch: 50MB if we strip everything except the lane's own code.

## What does work (for ad-hoc launches)

Each phase invoked SEPARATELY from a regular shell (not via the bash tool harness) works reliably. The issue is harness-specific. For ad-hoc human-driven launches in a regular terminal, the V3 launcher with combined phase2 works fine.

## Cross-references
- `feedback_canonical_parent_shell_launcher_20260428` — original V2 launcher
- `feedback_cycle_1_launch_postmortem_20260428` — broader Cycle 1 launch failures
- `feedback_bash_harness_kills_long_running_tasks_20260428` — harness behavior
- `feedback_codex_sandbox_blocks_vastai_dns_20260428` — separate codex sandbox issue
