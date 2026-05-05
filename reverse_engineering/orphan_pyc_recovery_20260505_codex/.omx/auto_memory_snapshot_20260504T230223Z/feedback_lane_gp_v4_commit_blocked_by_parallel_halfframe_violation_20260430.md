---
name: Lane GP v4 commit blocked — parallel-subagent half-frame untracked file fails preflight
description: 2026-04-30. Lane GP v4 KILL deliverables are clean and tested (14/14 + 4/4 existing) but the subagent commit serializer cannot land them because the preflight hook's whole-repo static scan blocks on a half-frame violation in scripts/remote_lane_19_logit_margin.sh:240 — an untracked file owned by another in-flight subagent. Multiple parallel commits (owv2-stack-result, anonymous Check 83 fix, lane_gp_v4_kill) all blocked by the same root cause. Files staged but not committed; awaiting parent agent's parallel-write resolution.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Situation

Lane GP v4 KILL verdict + Check 91 extinction work is COMPLETE:
- 5 council research files at `.omx/research/council_lane_gp_v4_*.md`
- Check 91 `check_pose_basis_fit_kill_acknowledged` STRICT @ 0 in `src/tac/preflight.py`
- 14 regression tests in `src/tac/tests/test_check_pose_basis_fit_kill.py` (14/14 pass)
- Kill markers in `experiments/fit_pose_gp.py` and `src/tac/pose_gaussian_process.py`
- 4 council review rounds (3/3 clean pass)
- Memory entry at `project_lane_gp_v4_killed_basis_fit_infeasible_20260430.md`
- Lane registry at `/tmp/lane_registration_lane_gp_v4.json`

## Blocker

`tools/subagent_commit_serializer.py` cannot acquire the commit. Pre-commit
hook (`tools/preflight_hook.py`) runs `python -m tac.preflight` which
triggers `check_halfframe_archive_uses_trained_profile` (line 410 in
`preflight_all`) which raises `MetaBugViolation` on:

```
scripts/remote_lane_19_logit_margin.sh:240: `--half-frame` present but no
`--profile` in 30-line window. Half-frame archives REQUIRE a renderer
trained with mask_half_sim_prob>0 OR use_zoom_flow=True (memory
feedback_half_frame_breaks_posenet).
```

That file is **untracked** in my working tree (`?? scripts/remote_lane_19_logit_margin.sh`)
— it was created by another in-flight subagent. My commit does NOT include
this file, but the preflight hook does whole-repo static scan, not just
staged files.

## Affected commits (all blocked by same violation)

From `.omx/state/commit-serializer.log` (last 6 entries):

- pid 45481 (`owv2-stack-result`): blocked
- pid 50541 (`anonymous` Lane 7 PSD KILL memo): blocked
- pid 52316 (`owv2-stack-result` retry): blocked
- pid 58812 (`lane_maturity_harness`): blocked
- pid 66110 (`lane_gp_v4_kill` ME): blocked rc=1, files staged, commit failed
- pid 69147 + 69588: lock_timeout (couldn't even reach the preflight)

## Why I am NOT using `PREFLIGHT_HOOK_ENABLED=0`

CLAUDE.md "Review gate" non-negotiable says:
> NEVER use REVIEW_GATE_OVERRIDE=1 when committing .py files.

Memory `feedback_check_64_smoke_proofs_resolved_AND_subagent_serializer_landed_20260429.md`
extends this principle:
> PREFLIGHT_HOOK_ENABLED=0 should never appear again.

My commit includes 4 .py files (preflight.py, test, fit_pose_gp.py, pose_gaussian_process.py).
Per the non-negotiable, I will NOT bypass the preflight hook.

## What needs to happen

Resolution options (in priority order — for parent agent to decide):

1. **Wait for the half-frame violator subagent to finish** and either commit
   their fix or have it cleaned up. Then retry.
2. **Parent agent commits the half-frame fix first** (add `--profile
   lane_19_logit_margin` adjacent to the `--half-frame` line on
   `scripts/remote_lane_19_logit_margin.sh:240`, similar pattern to other
   logit-margin sister scripts).
3. **If the half-frame violator is mid-flight subagent**, parent could
   stash its untracked file out of the way temporarily, let the queued
   commits land (mine + others), then restore and let that subagent's own
   commit run.

## Files needing commit (lane_gp_v4_kill)

```
.omx/research/council_lane_gp_v4_design_20260430.md
.omx/research/council_lane_gp_v4_round1_20260430.md
.omx/research/council_lane_gp_v4_round2_20260430.md
.omx/research/council_lane_gp_v4_round3_20260430.md
.omx/research/council_lane_gp_v4_round4_20260430.md
src/tac/preflight.py                                 # Check 91 added
src/tac/tests/test_check_pose_basis_fit_kill.py      # 14 tests
experiments/fit_pose_gp.py                            # kill marker
src/tac/pose_gaussian_process.py                      # kill marker in docstring
```

All 4 .py files pass: pytest (14/14 new + 4/4 existing pose_gp tests),
Check 91 STRICT against real repo, py_compile, ruff F821.

## Commit message ready (heredoc-safe)

The full commit message body is preserved in the failed-commit log entries
(see pid 66110 in `.omx/state/commit-serializer.log`).

## Status

Lane GP v4 work logically complete; commit pending parallel-write coordination.
