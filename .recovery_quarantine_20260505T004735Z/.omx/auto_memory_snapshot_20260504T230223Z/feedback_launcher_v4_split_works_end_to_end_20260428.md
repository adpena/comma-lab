---
name: Launcher V4 split-mode WORKS end-to-end — Lane Ω-V2 deployed on Vast.ai
description: 2026-04-28 launcher V4 split into 5 atomic phases (phase1, phase2-wait, phase2-scp, phase2-extract, phase2-launch) each ≤2-3 min — fits bash tool harness budget. Lane Ω-V2 (instance 35759655) deployed end-to-end successfully. setup_full.sh Stage 0 (GPU probe) confirmed running on remote. First successful Vast.ai deploy via the new launcher today.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## V4 split (the working architecture)

| Sub-command | Wallclock | Atomic action |
|-------------|-----------|---------------|
| phase1 | 10-30s | Search offer + create instance + register tracker |
| phase2-wait | 3-10 min (idempotent, re-callable) | Poll vastai status + ssh-ready |
| phase2-scp | 2-3 min | Build local tarball + SCP to remote |
| phase2-extract | 30s | Extract + CUDA probe |
| phase2-launch | 10s | Subshell-detach lane wrapper via SSH |

Each fits comfortably under harness 5-min budget. Each is INDEMPOTENT or SAFE to retry.

## Validation (2026-04-28 ~16:51 UTC)

Lane Ω-V2 instance 35759655 on RTX 4090 (ssh1.vast.ai:39654, $0.55/hr):
- phase1: ✓ instance created
- phase2-wait: ✓ ssh ready (single call, didn't need retry)
- phase2-scp: ✓ 962MB tarball transferred
- phase2-extract: ✓ extract + CUDA probe OK
- phase2-launch: ✓ subshell-detach launched
- Verified setup_full.sh Stage 0 (GPU probe) running on remote: "NVIDIA GeForce RTX 4090, 580.126.09"

## CORRECTION (2026-04-28 Wave 3 audit) — partial invalidation

Wave 3 ruff F821 audit found this "end-to-end success" claim was MEASURED ONLY on the EXPLICIT phase2-deploy subcommand path. Two other code paths in the same launcher were broken at the time of the original validation:

1. **`cmd_full(args)` had `cmd_phase2(args)` NameError** at line 811 (function had been renamed to `cmd_phase2_deploy` but one caller was missed). Anyone running `python scripts/launch_lane_on_vastai.py full ...` OR the legacy no-subcommand path would crash with `NameError: name 'cmd_phase2' is not defined`.
2. **Legacy combined-launcher dead block** referenced `threading` (never imported). Block was unreachable due to `return cmd_full(args)` above it, but ruff F821 still flagged.

Fixed in commit b0a2e45f (2026-04-28). The V4 split architecture itself was sound; the bugs were in alternate dispatch paths that the original validation didn't exercise. Lesson: "end-to-end" must mean ALL declared dispatch paths execute green, not just the one used during the validation run.

## What this unblocks

- Cycle 1 deploys (Lane Ω-V2 + Lane EC + Lane SAUG-V2 + Lane I retry + Lane W) all now possible
- Lane M-V3-clean deploy (when codex finishes implementation)
- All future research-synthesis lane deploys

## Critical implementation details

1. **Subshell-detach SSH** is the only reliable detach: `( bash run_lane.sh </dev/null >/dev/null 2>&1 & ) && echo launched`. The `( ... & )` subshell exits, leaving the & job orphaned to init.
2. **Tarball excludes** must use SPECIFIC paths not globs: `--exclude=submissions/robust_current/eval_runs` works; `--exclude=submissions/*/eval_runs` does NOT match (fixed earlier today).
3. **actual_status='running'** is necessary but NOT sufficient for SSH ready — sshd takes another 30-180s. Use phase2-wait's separate SSH-poll step.
4. **A100 PCIE Minnesota** ($0.508/hr, rel 0.998) tends to boot faster than the cheapest 4090s ($0.241/hr). For multi-hour training, A100 reliability is worth the +50% cost.

## Cross-references
- `feedback_canonical_parent_shell_launcher_20260428` — V2/V3 history
- `feedback_launcher_v3_phase2_split_needed_20260428` — V3 pain points
- `feedback_bash_harness_kills_long_running_tasks_20260428` — harness behavior
- `feedback_per_instance_verify_pattern_20260428` — companion runtime watchdog
