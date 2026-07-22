---
title: DDM receiver-priced member solve v2 at n64
date_utc: 2026-07-22T04:08:36Z
task: 603
master_task: 578
feeds_task: 613
lane_id: lane_ddm_mdl_member_solve_v2_priced_603_20260722
research_only: true
execution_allowed: false
score_claim: false
verdict: MEASURED_FIXED_WIDTH_RECEIVER_RATE_WALL_N64
verdict_scope: FORMULATION negative for the six fixed-record ZIP_STORED Task 603 chart grammar at n64; the wider direct-description and member families remain open
main_landing_review_required: true
---

# Outcome

The v2 solve replaced Task #602's diagnostic full-array zlib objective with the exact byte length of
the six-member archive consumed by the Task #603 receiver. The bounded n64 run completed in 211.42
wall-clock seconds, stopped after rung 1, resumed from its immutable checkpoint, and preserved all
five rung checkpoints.

The result is a measured **fixed-width formulation wall**. Every low/mid/high residual proposal
changed thousands of receiver-consumed coefficient scalars, compiled and decoded successfully, but
changed exact archive size by `0` bytes. Reverse-waterfill therefore rejected every proposal before
spending distortion. This is not the Task #602 identity/reference failure: the selected artifact is
a self-contained 274,664-byte decoder-consumed archive with all six semantic payloads and no source
raw reference.

# Tolerance-versus-bytes curve

| Rung | Allowed escape | Exact archive bytes | Frozen-SegNet membership | Pose completeness | Feasible |
|---|---:|---:|---:|---:|---|
| exact cell | 0.000000 | 274,664 | 0.493605613708 | 1.000000000000 | no |
| 1 | 0.000152 | 274,664 | 0.493605613708 | 1.000000000000 | no |
| 2 | 0.000300 | 274,664 | 0.493605613708 | 1.000000000000 | no |
| 3 | 0.000500 | 274,664 | 0.493605613708 | 1.000000000000 | no |
| 4 | 0.000800 | 274,664 | 0.493605613708 | 1.000000000000 | no |

No #613 knee exists on this syntax: all 15 actual proposal encodes had `delta_archive_bytes=0`.
The exact rate price was `25 / 37,545,489` score units per byte, but there was no positive byte
saving against which a Fisher/margin distortion spend could be admitted.

# Wall decomposition

Each proposal used the maximal deterministic safe-zero collapse in exactly one residual stratum.
It retained any scalar whose zeroing would leave the integer receiver's uint8 domain.

| Receiver stratum | Changed scalars per probe | Changed records | Exact byte delta | Accepted |
|---|---:|---:|---:|---|
| low variation | 24,531 | 8,192 | 0 | 0/5 |
| mid variation | 24,159 | 8,192 | 0 | 0/5 |
| high variation | 23,955 | 8,189 | 0 | 0/5 |

The final membership wall is class-skewed, not a healthy aggregate solution:

| Stratum | Membership | Escape fraction |
|---|---:|---:|
| Road | 0.000000000000 | 1.000000000000 |
| Lane | 0.000000000000 | 1.000000000000 |
| MyCar | 0.000000000000 | 1.000000000000 |
| Movable | 0.000000000000 | 1.000000000000 |
| Undrivable | 0.999986797783 | 0.000013202217 |
| boundary codim-1 | 0.118697769367 | 0.881302230633 |
| cell interior | 0.502205475276 | 0.497794524724 |

The same-batch target-versus-described membership numerator is the measured advisory statistic.
The requested `gt_n600.lstars` crosscheck was 0.999873638153, not 1.0, so the cached raster remains a
separate batch-geometry caveat and is not silently substituted for the current batch16 target cells.

# Receiver and custody closure

- Selected archive: `274664` bytes, SHA-256
  `f3f98457ff8495dfefbfad2fb04549c8936eea15a1087d12c852144b5be5ae35`.
- Receipt: SHA-256 `7015fca367321c89fd35b50402e5dad9550adec6093ad050e70cdeac1e8dc398`.
- Six ZIP_STORED members are parsed and consumed once; unique-home coverage is exactly 274,664 bytes.
- Encode determinism x2, parse/re-encode identity, streaming decode determinism x2, sampled semantic
  no-op honesty, and sampled archive-home fail-closed behavior are green.
- Pose6 has 384/384 exact n64 coordinates and owns a nonempty 512-byte semantic payload.
- Five atomic checkpoints are preserved; the rung-1 stop/resume terminal archive is byte-identical.
- SSD target/cache inputs were read-only. No scorer weights, raw source, candidate claim, bulk copy,
  delete, move, paid dispatch, GPU work, or contest evaluation occurred.

# Blocker delta

1. `N600_MEMBER_SOLVE_COVERAGE`: partial green at the required n64 minimum; n600 selection remains owed.
2. `RECEIVER_CARRIABLE_CODED_MEMBER_PAYLOAD`: red to green at n64 apparatus scope.
3. `COUNTED_ARCHIVE_MDL_INSIDE_SOLVE`: red to green for exact final-ZIP pricing; a new fixed-width rate-gradient wall is measured.
4. `PRE_UINT8_MEMBER_STATE`: remains red; no zero-realization-loss claim is made.
5. `POSE_STREAM_IN_MEMBER_PAYLOAD`: red to green at n64 apparatus scope.
6. `PER_STRATUM_TOLERANCE_FEASIBILITY`: remains red at every rung.

The inherited PRIMARY 19-row launch register stays `8/19` green. Nothing here authorizes PRIMARY
execution, contest replay, promotion, or pointer movement.

# Bounded re-derivation argv

```bash
/usr/bin/env python3 tools/run_direct_description_receiver_priced_member.py \
  --config .omx/research/ddm_receiver_priced_member_n64_603_20260722T040836Z.config.json \
  --output-dir .omx/research/ddm_receiver_priced_member_n64_603_REDERIVE_artifacts \
  --execution-allowed false
```

The output path must be fresh because every receipt is immutable. Measured local runtime was 211.42
seconds, below the delegated ten-minute bound.

# Stores consulted

- `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`.
- `docs/operating_manual_craft_handoff.md` (workflow and handoff authority).
- `.omx/research/direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`.
- Task #603 target, measurement-ladder, and polytope-membership receipts and source modules.
- Task #602 member receipt and Task #603 carrier preflight/blocker register.
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, and both delegated inboxes.
- Project Claude MEMORY top entries and Codex memory registry quick pass; neither was treated as current measurement authority.

0.1910828242 [contest-CPU] — unchanged.
