---
schema: dag_feed.v1
utc: 2026-07-22T02:25:04Z
task: 603
lane_id: lane_ddm_measurement_ladder_rungs123_603_20260722
research_only: true
execution_allowed: false
evidence_axis: "[macOS-CPU full-resolution real-plane apparatus]"
---

# DAG FEED — DDM measurement ladder rungs 1-3

## Executed trajectory

`C1 n600 target receipt -> verified 12-pair SSD chunks -> per-chart/per-stratum fit -> exact A(z) ZIP -> integer/uint8 384x512 receiver -> per-pair plane/Pose bridge`

Stage edges are `rung1_n64_full_resolution -> rung2_pair_scaling_n256 -> rung3_same_artifact_quantity_bridge`.
Each edge writes an immutable checkpoint. A stopped run after rung 1 resumed through rungs 2-3 with
the same checkpoint, archive, bridge, and history hashes as the uninterrupted path.

## New typed signals

- `DirectDescriptionMeasurementLadderConfigV1`: SHA-bound target, n64/n256 pair plan, 32x32 chart
  units, execution false, and immutable stage policy.
- `DirectDescriptionChartZV1`: six counted semantic owners: global anchors, axial gradients, three
  target-variation chart strata, and Pose6 codes.
- `DirectDescriptionMeasurementLadderCheckpointV1`: config/DSL/argv/target identity, exact archive,
  full per-pair bridge, continuation cursor, and history.
- `direct_description_plane_quantity_bridge.v1`: exact per-pair hashes/counts plus the measured
  `(archive_bytes, plane-exactness, pose-debt)` tuple.

## Six-hook disposition

1. Sensitivity map: no scorer sensitivity row; RGB/Pose integer apparatus metrics cannot enter it.
2. Pareto constraint: exact archive bytes are counted, but absent `d_seg/d_pose` forbids contest
   Pareto admission.
3. Bit allocator: six stream byte totals are exposed; score-unit value per byte remains unmeasured.
4. Cathedral/autopilot: research-only edge; no dispatch or candidate promotion hook is enabled.
5. Continual learning: blocker state advances by one scoped green row; no score posterior is updated.
6. Probe disambiguator: parse/re-encode, compiler x2, six-stream no-op samples, exact per-pair bridge,
   and stopped/resumed identity disambiguate custody from task efficacy.

## Edge-state delta

`FOUR_RUNG_CELLS_THEN_POSE_MEASUREMENT_LADDER` becomes green only at measured n64/n256 apparatus
scope. `N600_SAME_ARTIFACT_ARCHIVE_CLOSURE`, scorer axes, launch authority, and pointer movement stay
red. MAIN landing review required.
