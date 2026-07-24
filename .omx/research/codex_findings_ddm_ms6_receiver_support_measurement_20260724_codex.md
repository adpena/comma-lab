# Codex findings — DDM MS6 receiver-support measurement

Date: 2026-07-24  
Lane: `lane_ddm_ms6_receiver_support_measurement_20260724`  
Evidence axis: `[macOS-CPU frozen-scorer advisory]`  
Score claim: `false`  
Pointer: `0.1910828242 [contest-CPU]` (unmoved)  
Verdict scope: `INSTANCE_V19C_ENDPOINT_ONE_QUANTUM_SWEEP`

## STORES CONSULTED

- Delegated authority:
  `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/ddm_ms6_receiver_support_measurement_20260724T052034Z.wrapped.prompt.txt`
  at SHA-256
  `14b8bac8a4e3708069461ee1cacaa8f30fcaaf494f17a51f42dcab4583f7463a`.
- The MS5 assignment table at SHA-256
  `20fa2b2ce2bd96b91c64d4e1342109dd7dab399d4769cd372dbf67fbcdf97d8d`
  and its receipt at SHA-256
  `3d0b9fcc738a1092bad495b0dbce2b022451e1442814a7cc274da41e43d455d6`.
- The PF2 receipt at SHA-256
  `85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73`.
- The V19C endpoint archive at SHA-256
  `dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9`.
- Frozen SegNet weights at SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`
  and upstream modules at SHA-256
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`.
- `reports/latest.md`, the lane registry, current sister findings/session
  summaries, CLAUDE.md, AGENTS.md, and the craft handoff manual.

## Finding 1 — the missing foreign key is now executable, resumable, and exact

`tools/measure_ddm_ms6_receiver_support.py` reconstructs the exact PF2 raw-event
addresses, applies one named signed receiver quantum, realizes camera uint8,
uses SegNet's own preprocessing as the composite-R support detector, and runs
batch-32 frozen SegNet only on support-overlap pairs. Every probe is an
immutable SSD checkpoint. Exact event sets are stored as uint32 NPZ arrays and
bound by SHA-256; the committed table carries their hashes and causal
actuator/direction/pair aggregates.

The consumed v2 checkpoint schema also binds the exact SegNet and upstream
module hashes, base archive SHA, seed, deterministic mode, four-thread setting,
and batch-32 scorer geometry at each probe. The earlier v1 checkpoint directory
is preserved but explicitly not consumed because it lacked checkpoint-local
scorer custody.

This is a first rung, not a completed sweep.

## Finding 2 — 49 signed island probes measured real causal joins

The resumable sweep currently includes 49 receiver-effective signed island
probes. The initial bounded smoke completed both signs for
`j2.island.track0.center_x`:

- Negative quantum: 35 support pairs, 10,398 changed camera values, 566
  composite-R cells, and 545 exact PF2 raw events across 15 buckets.
- Positive quantum: 35 support pairs, 10,572 changed camera values, 579
  composite-R cells, and 493 exact PF2 raw events across 13 buckets.
- Aggregate table status: `CAUSAL_ACTUATOR_DIRECTION_JOIN_PARTIAL`.
- Completed probes: 110/748, comprising 49 measured receiver-effective probes
  and 61 explicit infeasible probes.
- Exact joined rows: 31/1,200. The other 1,169 remain fail-closed.

No class-label similarity, neighboring-pixel heuristic, or pair-only
co-membership was used to form a join.

## Finding 3 — 61 requested probes are infeasible on this V19C instance

The V19C nested V15 carrier contains the 163-track G1 worldsheet and six
scorer-solved templates, but no counted Lane program member. Adding the derived
Lane seed would change more than the named coordinate, so all 24 Lane DOFs ×
two signs are explicitly
`INFEASIBLE_RECEIVER_QUANTUM`.

Likewise, the carrier's V13 grammar explicitly forbids mixing its worldsheet
production vocabulary with post-solve G2CS1 correction symbols. The six G2G
addresses × two signs are therefore also explicit infeasible probes. They are
not measured-empty zeros. Closing these 60 rows requires a new receiver grammar
that jointly represents those coordinates around the same SHA-bound base, then
a fresh measurement. One additional signed island probe,
`j2.island.track1.center_x +1 quantum`, failed closed because the derived G1
Movable polygon escaped the scorer geometry; that is a direction-specific
instance boundary, not a grammar omission.

Verdict ladder: this is an `INSTANCE` blocker for the selected V19C base and
current receiver grammar. It is not a `FORMULATION`, `FAMILY`, or `PARADIGM`
negative.

## Finding 4 — MS4 remains held

The measured table covers 31/1,200 rows, so it does not satisfy the MS4
consumer's completeness gate or establish coverage of every required G3 hard
block. `tools/produce_ddm_ms4_metric_custody.py` was not invoked. Pose remains
the already-landed COMPLETE component and was not remeasured.

## Required next action

Resume the SSD checkpoint sweep for the remaining receiver-effective island
and template probes. Separately extend the receiver grammar so Lane and G2CS1
are true isolated one-quantum coordinates on the same base; do not reinterpret
the 60 grammar-infeasible rows or the one geometry-infeasible row as zeros.
Only after G3 hard-block coverage is proved should MAIN review and authorize
the MS4 top24-first rerun.

## Review disposition

Three clean passes completed after the final v2 receipt:

1. implementation and focused-test review (`25 passed`);
2. SHA lineage, advisory-axis, pointer, and MS4 fail-closed review;
3. byte-level revalidation of all 110 checkpoint hashes, all 49 event NPZ
   artifacts, every per-bucket exact event hash, the digest chain, and both
   canonical JSON content hashes.

`main_landing_review_required=true`.
