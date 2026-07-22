---
title: DDM measurement ladder rungs 1-3 landing
utc: 2026-07-22T02:25:04Z
task: 603
lane_id: lane_ddm_measurement_ladder_rungs123_603_20260722
verdict: RUNG1_TO_RUNG3_MEASURED_APPARATUS_SCOPE
verdict_scope: local full-resolution C1 RGB/Pose integer apparatus at n64 and n256 only
research_only: true
execution_allowed: false
main_landing_review_required: true
---

# Outcome

Rungs 1-3 are complete at the delegated apparatus scope. The new counted grammar stores global and
axial coefficients, three target-derived chart-variation strata, and Pose6 codes. Its receiver emits
the actual `uint8 [N,2,384,512,3]` planes; the 8x8 projection is not used. Target files remain
read-only on `/Volumes/VertigoDataTier/pact`, and fitting/measurement retain at most one 12-pair
chunk plus one described chunk.

The rung-2 and rung-3 rows use the exact same 1,095,272-byte archive. Every one of 256 pairs has an
exact agreement row. This is not an archive candidate, scorer result, `d_seg`, `d_pose`, score, or
promotion receipt.

# Measured ladder

| Rung | Pairs | Full target | Archive bytes | RGB pixels exact | Channel values exact | RGB-channel argmax disagreement | Pose6 integer debt |
|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 64 | `2x384x512x3 uint8` | 274,664 | 0.002674818039 | 0.090567164951 | 0.196429928144 | 0 |
| 2 | 256 | `2x384x512x3 uint8` | 1,095,272 | 0.001762549082 | 0.087192485730 | 0.229121357203 | 0 |
| 3 | 256 | exact rung-2 artifact | 1,095,272 | 0.001762549082 | 0.087192485730 | 0.229121357203 | 0 |

`Pose6 integer debt = 0` means the counted Pose6-code stream stores all 6 target ordinal codes per
described pair. It does not mean PoseNet distortion is zero. `RGB-channel argmax` is a tie-first RGB
input-plane apparatus diagnostic; it is explicitly not SegNet argmax.

# Custody and resumability

- Final described archive SHA-256: `1c22f9b1c0911e4c89d151899fd3e8c8a3af2b339b998e6a0765e6f7cd9d3135`.
- Receipt SHA-256: `dac6a705f3efefb8c35f3c735d1aa7a7ad5b31b5ba122d88ef9402f4caa81b87`.
- Six independently framed ZIP-STORED semantic members; all 1,095,272 final bytes have one home.
- 18 semantic no-op samples changed the receiver output; 39 sampled final-ZIP home mutations all
  failed closed.
- Three primary and three stopped/resumed stage checkpoints are preserved. Matching stage hashes are
  `35fe76512d...`, `83ae7f0e1d...`, and `5e631be581...`; terminal archive, bridge, and history are
  bit-identical after disk continuation.
- Parse/re-encode identity and compiler determinism x2 are green.
- The 8,303,507-byte pre-fix scratch tree was content-addressed in the cleanup certificate and
  removed after the corrected durable replacement was harvested.

# Exact bounded MAIN re-derivation argv

Measured wall time in this worktree was 17.34 seconds, below the delegated 10-minute bound:

```text
/usr/bin/env python3 tools/run_direct_description_measurement_ladder.py --config .omx/research/ddm_measurement_ladder_rungs123_603_20260722T015737Z.config.json --output-dir .omx/research/ddm_measurement_ladder_rungs123_603_20260722T015737Z_artifacts --execution-allowed false
```

Outputs are immutable by design; MAIN should use a fresh reviewed output directory when re-deriving.

# Blocker delta and pointer honesty

- `FOUR_RUNG_CELLS_THEN_POSE_MEASUREMENT_LADDER`: `RED -> GREEN_MEASURED_APPARATUS_SCOPE`.
- `N600_SAME_ARTIFACT_ARCHIVE_CLOSURE`: remains `RED_N256_ONLY`.
- Register total: `6/19 -> 7/19` scoped green; 12 remain red.
- Pointer remains `0.1910828242 [contest-CPU]`; no scorer or evaluator was called.

The 12 exact remaining blockers are PRIMARY execution authority, live V8/V9 owner receipts,
canonical resume-registry integration, canonical PRIMARY typed-compiler integration, governed
launcher/memory adapter, heavy-run cleanup/cold-store integration, n600 same-artifact closure,
contest-CPU replay, contest-CUDA replay, healthy completion certificate, externally attested failure
token, and separate SHA-bound operator GO.

# STORES CONSULTED

Delegated authority, `CLAUDE.md`, `AGENTS.md`, project memory top, `PROGRAM.md`, operating manual,
v7.5 operating contract, DDM PRIMARY spec/DAG/equations, predecessor receiver/rung-zero code and
receipts, current lane/task/pointer state, exact C1 target receipt and used SSD chunks, and both
inboxes.

Implementation commits: `579a1c504e`, `918edb4453`. MAIN landing review is required.
