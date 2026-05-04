---
name: Lane 7 PSD kill memo blocked from commit by 2 pre-existing Check 83 false positives (2026-04-30)
description: 2026-04-30 ~03:30 CDT. Lane 7 PSD council REJECT (10/10) verdict landed at .omx/research/council_lane_7_psd_dispatch_review_20260430.md + lane_7_psd_kill_memo_20260430.md + memory project_lane_7_psd_killed_or_deferred_20260430.md. Commit via tools/subagent_commit_serializer.py was BLOCKED by preflight Check 83 (no-MPS-decision) firing on TWO pre-existing files NOT touched by this work. Need separate subagent fix path before Lane 7 docs can land.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## What blocked

`tools/subagent_commit_serializer.py --files .omx/research/council_lane_7_psd_dispatch_review_20260430.md .omx/research/lane_7_psd_kill_memo_20260430.md` returned rc=1. Preflight check 83 (`check_no_proxy_metric_drives_decision`) fired with these violations on pre-existing files:

1. `src/tac/nerv_mask_codec.py:537` — docstring fragment "MPS is FORBIDDEN for any kill/promote decision per CLAUDE.md" inside the NeRVMaskTrainer docstring. **This is the CLAUDE.md rule itself being quoted in a docstring** — Check 83 is over-triggering on its own rule's restatement.
2. `src/tac/tests/test_check_pose_basis_fit_kill.py:3` — module docstring "Lane GP v3 (89.67 [Modal-T4-CPU]) was killed 2026-04-30 per Council #271". **The kill decision was made per Council #271 documented elsewhere**, the Modal-T4-CPU tag is part of the original measurement context.

Both are FALSE POSITIVES of Check 83. They are pre-existing in the repo (not introduced by Lane 7 PSD work).

## Why I did NOT fix them in this session

- My Lane 7 task explicitly creates `.md` files only; modifying `.py` files outside that scope is out of scope.
- CLAUDE.md "review gate" rule: NEVER use `REVIEW_GATE_OVERRIDE=1` for `.py` files. Adding a `[contest-CUDA]` tag to `nerv_mask_codec.py` and `test_check_pose_basis_fit_kill.py` would create a `.py` change that needs proper review.
- CLAUDE.md "PREFLIGHT_HOOK_ENABLED=0 should never appear again" — bypass is forbidden.
- Per Lane 7 prompt: "If hit any blocker, write to memory + return early. Do NOT guess at user intent."

## What's now durable

The Lane 7 REJECT verdict IS recorded:
1. `.omx/research/council_lane_7_psd_dispatch_review_20260430.md` (full per-voice deliberation, 10/10 REJECT)
2. `.omx/research/lane_7_psd_kill_memo_20260430.md` (formal kill memo)
3. `memory/project_lane_7_psd_killed_or_deferred_20260430.md` (memory entry)
4. `/tmp/lane_registration_lane_7_psd.json` (lane registration JSON since `tools/lane_maturity.py` does not exist)

These exist on disk but are NOT yet committed to git.

## Remediation path (separate subagent should do)

Two minimal fixes to add `[contest-CUDA]` tags within ±10 lines of each offending docstring line (the documented Check 83 fix path per the error message). Both are docstring-only edits that change zero runtime behavior:

### Fix 1: src/tac/nerv_mask_codec.py:537
```diff
     * ``device != "mps"`` — refused at construction (PoseNet drift 23x;
+      contest-CUDA verified — see CLAUDE.md "MPS auth eval is NOISE" table:
+      [contest-CUDA] PoseNet 0.245 MPS vs 0.0107 CUDA = 23x drift;
       MPS is FORBIDDEN for any kill/promote decision per CLAUDE.md
       "MPS auth eval is NOISE — NON-NEGOTIABLE").
```

### Fix 2: src/tac/tests/test_check_pose_basis_fit_kill.py:3
```diff
 """Regression tests for Check 91 (pose-basis-fit kill).
 
-Lane GP v3 (89.67 [Modal-T4-CPU]) was killed 2026-04-30 per Council #271
+Lane GP v3 (89.67 [Modal-T4-CPU advisory]) was killed 2026-04-30 per
+Council #271. [contest-CUDA] Lane G v3 = 1.05 remains the standing
+CUDA-verified bar; no GP variant has produced a [contest-CUDA] win
 + Lane GP v4 design verdict (.omx/research/council_lane_gp_v4_design_20260430.md).
```

After those two edits land via the proper review-gate path, my three uncommitted Lane 7 .md files can be committed via `tools/subagent_commit_serializer.py`.

## Or alternatively: refine Check 83's regex

Check 83 should EXEMPT lines whose decision verb is part of a quoted CLAUDE.md rule restatement, OR lines whose decision is documented in a same-paragraph cross-ref to a council .md file (Council #271 reference pattern). This is a Check-83 refinement task that's separate from the immediate Lane 7 unblock.

## Cross-refs

- `.omx/research/council_lane_7_psd_dispatch_review_20260430.md` (Lane 7 council deliberation)
- `.omx/research/lane_7_psd_kill_memo_20260430.md` (Lane 7 kill memo)
- `tools/subagent_commit_serializer.py` (the serializer that gate-checked and blocked)
- `tools/preflight_hook.py` (preflight hook entrypoint)
- `src/tac/preflight.py:6062` (Check 83 strict-raise location)
- `feedback_check_82_83_landed_council_round3_prescription_20260429.md` (Check 83 was added 2026-04-29 PM)
