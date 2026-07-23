---
title: DDM J3 #366 full-run mode and ticket reseal DAG FEED
utc: 2026-07-23T03:25:00Z
lane_id: ddm_j3_366_fullrun_mode_and_ticket_reseal
verdict: BLOCKED_REALIZED_DSEG_REGRESSION
research_only: true
score_claim: false
pointer_moved: false
---

# Executable DAG

`J1 semantic program @ 68a8aa97... (declarative only)`
→ `actual J2 lift/receiver/scorer consumer`
→ `receiver-wire audit: 706 named -> 368 effective coordinates`
→ `2*163 island translations + 4*6 Lane wire fields + 3*6 template bytes`
→ `no 2,197-knot expansion absent counted grammar plus measured marginal value/byte`
→ `actual full-run batch-4 timing window: 104.09510249993764 s/step`
→ `ceil(600/4)=150 steps/exposure; three force-group stages; 450 steps`
→ `quarter-quantum LR with decreasing line search`
→ `periodic checkpoint every floor(150/4)=37 steps plus immutable stage checkpoints`
→ `atomic Adam/live/EMA/moment/cursor/archive identity persistence`
→ `same-outdir lock + SSD storage waterfall + memory governor`
→ `strict chunked n600 archive/paint/uint8/R/frozen-scorer stage verdict`
→ `forced rc=23 after step-1 checkpoint @ 6e85c716...`
→ `new process resumes and reaches step 4`
→ `exact d_seg 0.027470296223958333 -> 0.027603208753797744`
→ `exact d_pose 163.0613272814428 -> 163.0613308426994`
→ **`BLOCKED_REALIZED_DSEG_REGRESSION`; no stage advance, launch, score, or pointer movement**.

# FEED

- Typed ticket: `.omx/research/configs/ddm_j3_366_joint_descent_witness_program_20260723.json` (`df8db01f...`; typed hash recorded in ticket custody).
- Full-run implementation: `src/tac/optimization/direct_description_joint_descent.py` and `tools/launch_ddm_joint_descent.py`.
- Preflight: `.omx/research/ddm_j3_366_fullrun_preflight_receipt_20260723.json`.
- Launcher-bound admission: `.omx/research/ddm_j3_366_launcher_memory_preflight_20260723.json` (final typed hash, exact-geometry reuse of measured receipt).
- Ticket delta: `.omx/research/ddm_j3_366_ticket_reseal_diff_20260723.json`.
- Exact smoke truth: `.omx/research/ddm_j3_366_fullrun_smoke_receipt_20260723.json`.
- Schedule equations: `.omx/research/ddm_j3_366_schedule_canonical_equations_20260723.md`.
- Downstream: MAIN reviews the branch and preserves the sealed ticket/build, but MUST NOT fire it while the exact smoke blocker remains.

# Triality

- DSL: RFC8785/SHA-256 sealed `DirectDescriptionJointDescentTypedConfigV1` with `FullRunScheduleV1`.
- DAG: the executable chain above, including exact stage-decision and crash-resume boundaries.
- Equations: `ddm_j3_366_full_run_schedule_v1` in the companion canonical-equations note.

# Honest boundary

The memory preflight is safe (`13.5690185546875 GiB < 116 GiB`), but safety is not efficacy. The four-step warm-start instance regressed both exact advisory components. This is an INSTANCE blocker for the fixed warm-start policy, not a formulation/family/paradigm verdict. The producer's earlier `FULL_RUN_BOUNDED_GREEN` string was invalid because it depended only on memory admission; the implementation now refuses component regression and returns nonzero.

# Verification

- `ruff check`: clean on all four owned Python surfaces.
- `py_compile`: clean on all four owned Python surfaces.
- Focused receiver plus full-run suite: `41 passed`.
- Typed ticket: independently recomputed semantic and typed hashes match the seals.
- Governed dry-run on the final hashes: ADMIT with `execution_allowed=false`.
- Adversarial reviews: 3 clean passes recorded for every modified Python file after fixes.
- Global lane validation: pre-existing repository debt of 110 missing legacy evidence paths; the new J3 lane introduces no missing evidence and is L2.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; v7.5 operating contract; J1/J2 ticket, SPEC, receipts, checklist, and DAG; canonical lane/subagent/frontier surfaces; live operator inbox including the v16 uint8 warning; exact external preflight/smoke roots. Pointer `0.1910828242 [contest-CPU]` unchanged.
