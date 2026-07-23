---
title: Codex findings - DDM J6A #366 prefire contract hardening
utc: 2026-07-23T08:56:30Z
lane_id: ddm_j6a_366_prefire_contract_hardening
verdict: BLOCKED_POSE_FINISH_CONDITIONING_HISTORY_INSUFFICIENT
verdict_scope: J6A PREFIRE CONTRACT AND ONE-STEP BOUNDED MACOS ADVISORY RE-SMOKE ONLY
research_only: true
execution_allowed: false
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# Outcome

The four J6 contract defects are repaired and regression-guarded. The hardened
ticket is **not READY_TO_FIRE**: its one-step exact n600 re-smoke has only two
run-owned d_seg verdict points, below the typed five-point #383-style
conditioning minimum, so pose remains disengaged and the launcher exits 4 with
`BLOCKED_POSE_FINISH_CONDITIONING_HISTORY_INSUFFICIENT`.

This is the intended fail-closed result. No campaign, paid dispatch, score
claim, promotion, or pointer move occurred. The pointer remains
`0.1910828242 [contest-CPU]`.

# Blocker-by-blocker disposition

| J6 blocker | Repair | Regression |
|---|---|---|
| Pose finish was a string plus immediate 0→1 switch | `PoseFinishEngageConfigV1` and `PoseFinishEngageStateV1` at `src/tac/optimization/direct_description_joint_descent.py:233-464`; launcher use at `tools/launch_ddm_joint_descent.py:1402`. Exact n600 d_seg history, strict-admission count, classification, and monotone engage step persist in every run cursor. | `test_pose_finish_engage_is_typed_rolling_slope_latched_and_checkpoint_roundtrippable` and `test_pose_finish_engage_refuses_binary_first_admission_and_rising_history` at `src/tac/optimization/tests/test_direct_description_joint_descent.py:421-464`. |
| Fire safety was local to last-admitted | `classify_cumulative_fire_gate` at `src/tac/optimization/direct_description_joint_descent.py:133`; applied to stage00 at `tools/launch_ddm_joint_descent.py:1542-1629`. The current fire-ready state is revocable, not OR-latched. | `test_cumulative_fire_gate_catches_locally_safe_second_move` at `src/tac/optimization/tests/test_direct_description_joint_descent.py:465`. |
| Memory measured 8 rather than worst 52 secants and cited stale J3 custody | Sealed memory receipt validation at `tools/launch_ddm_joint_descent.py:233-293`; memory-only all-groups bootstrap at `tools/launch_ddm_joint_descent.py:964-1176`. Ticket binds final sources, J5 baseline/proposal/memory/checkpoint, and the new receipt SHA. | `test_worst_geometry_receipt_must_bind_all_52_stage3_secants` at `tools/tests/test_launch_ddm_joint_descent_j5.py:71`. |
| Target-unmet limits/plateaus could advance and COMPLETE lacked final target | `classify_governed_stage_exit` and `exact_final_target_gate` at `src/tac/optimization/direct_description_joint_descent.py:158-225`; applied at `tools/launch_ddm_joint_descent.py:1880-1905` and `:1972-2001`. | `test_target_unmet_limits_and_plateaus_are_nonpromoting_stops` and `test_schedule_complete_requires_exact_final_target_verdict` at `src/tac/optimization/tests/test_direct_description_joint_descent.py:490-546`. |

# Sealed contract and measured evidence

- Semantic SHA:
  `3ba05e4d8fd2f85475173f0a9e17e668198507350d353a4257aaf196692b98c2`.
- Typed hash:
  `35c929d0031ef3ae3225afdfeb09997619bb70016f27527ea4ce1e7bed31ff47`.
- Ticket file SHA:
  `9aa12699eb53e77351fd0b96eed16d527eaf4a443b177f07512edd5ed6f3d88d`.
- Final consumer/launcher SHAs:
  `76c66ceb...f51` / `361f35e4...051`.
- Worst geometry is MEASURED at pair start 498:
  28 island + 24 Lane = 52 secants, 4.72976016998291 GiB basis,
  15.609344482421875 GiB peak RSS, 19.73121337890625 GiB projected
  versus the 116 GiB ceiling. Receipt SHA:
  `45a0e5d23fc04822c75036fefc78efcfc1503b75bb41838eb4617b6bf4b6a661`.
- Fresh-process checkpoint restore is green; receipt SHA:
  `0e091c54b2a97ce1742b692cd05ca21e0ca609f0f0ab0fad1184da600b555f83`.
- The final one-step re-smoke reproduces proposal archive
  `d4eb1450...d0d` and exact
  `delta_S=-0.002843840398518996`. Cumulative stage00 component/residual
  safety is green. Receipt SHA:
  `436fbb7b53ae6fcd4bc724d495f13181c535e300246e996c9a96dd334437e7a4`.
- Banked R1 `d_pose=0.001610`, score contribution `0.127`, and `7200`
  payload bytes are non-binding comparator/fallback-harvest metadata only.
  The final binding pose floor is the same-vehicle measured J5 proposal
  `d_pose=163.06116431842463`.

# Triality

- **DSL:** RFC8785/SHA-256 ticket contains the typed rolling-slope detector,
  cumulative stage00 fire contract, exact same-vehicle pose floor,
  worst-geometry memory contract, and non-promoting stop law.
- **DAG:** stage00 exact verdict → exact admitted verdict history → typed pose
  detector → pose weight; stage00 + candidate C1/component custody →
  cumulative fire state; pair498 all-groups memory receipt → governor;
  exact target verdict → stage advance → explicit final-target completion gate.
- **Equations:** pose engage requires the #383-derived
  `EMA span=3`, rolling window `3`, hysteresis `3`, and
  `|relative slope| <= 3e-4` on exact n600 d_seg after a strict Seg
  admission. Fire requires
  `d_seg(candidate)<=d_seg(stage00)`,
  `d_pose(candidate)<=d_pose(stage00)`, a strict component descent, and
  cumulative residual-error delta `<0`. A target-unmet limit or plateau maps
  to `STOPPED_BELOW_TARGET_*`, never advance.

# Verification and residual blocker

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest --timeout=180 -q \
  src/tac/optimization/tests/test_direct_description_joint_descent.py \
  tools/tests/test_launch_ddm_joint_descent_j5.py \
  src/tac/optimization/tests/test_pure_priced_realized_objective.py
```

Result: `41 passed in 75.92s`. Ruff, `py_compile`, JSON validation, and
`git diff --check` are green.

The first closeout review found and repaired one additional fail-closed issue:
non-finite final d_seg/d_pose values now classify
`REFUSE_SCHEDULE_COMPLETE_FINAL_VERDICT_INVALID` instead of passing the target
comparison through IEEE NaN ordering. The regression covers both fields.
Three clean passes over 127 tracked Python entities are sealed in
`ddm_j6a_366_three_clean_review_receipt_20260723.json`
(`ee9f46f972820f0e1cd9e917cbd446591f2c1a3a3fbae923cee9c4f5038a749d`).

The exact residual blocker is not code ambiguity: the run owns only stage00
and one admitted exact verdict (`2/5` points). Under this task's no-campaign
authority, no additional trajectory may be manufactured. MAIN must
independently review and land this branch before any later standing-GO
decision; a future governed run must earn the pose latch and all exact stage
targets.

# STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; v7.5/v8 vehicle specs; J2/J3/J5/J6
findings and receipts; sealed J5 ticket and producer artifacts; final consumer,
launcher, tests, lane/subagent/frontier state; delegation inboxes.
