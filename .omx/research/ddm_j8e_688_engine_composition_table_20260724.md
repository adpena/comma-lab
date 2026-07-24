---
title: "DDM J8E #688 engine composition table"
date_utc: "2026-07-24T15:55:00Z"
lane_id: "lane_ddm_j8e_688_engine_build_20260724"
research_only: true
execution_allowed: false
score_claim: false
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
main_review_required: true
---

# DDM J8E #688 engine composition table

This is deliverable 0 for the consolidated engine build. `ADOPT` means the
typed DDM program/continuation contract must expose and validate the capability
in this landing. It does not authorize campaign execution. `DEFER` names the
missing custody and leaves the hook execution-disabled.

| # | Capability | Disposition | Engine binding / fail-closed reason |
|---:|---|---|---|
| 1 | Box tolerance / two-channel economics | **ADOPT as milestone for descent; stop only for describe/solve** | Descent continues while exact receiver-realized joint `delta S < 0`, with real coder bytes measured every verdict. Crossing `d_seg<=0.00116` emits and checkpoints an R6-candidate milestone; it never stops descent. Per-block MS2R tolerances are proposal-ordering priors for descent and tolerance stops only for the byte-priced describe/solve channel. Stop events are measured marginal progress/hour, EMA/dynamics convergence, or explicit safety budgets. |
| 2 | Visibility typing on the parametrization | **ADOPT** | Every proposal declares `seg-only`, `pose-only(frame_0)`, or `joint`; frame 0 is Seg-free, fine-chroma is Pose-safe, the #580 resize null space is gauge-fixed out, and #401 blind coordinates are excluded. Ambiguous or untyped coordinates refuse compilation. |
| 3 | #517/#518 resume-warmup geometry set | **ADOPT** | Resume boundary occurs before baseline-v0. Beta2 rewarmup length resolves through `adam_v_variance_warmup_length_v1`; fork LR trigger, `ForkHeadSolve`, and margin trust-radius are typed event hooks. Hooks remain execution-disabled unless their required receipts resolve. |
| 4 | Solve interleave | **ADOPT** | `ForkHeadSolve` at step 0, #423 `HeadOffsetSolver`, #342 basin triggers, and MS2 TerminalSolve are real event-graph interleave nodes with callback contracts. Missing metric/solver custody disables the node and records a blocker; a no-op cannot masquerade as a solve. |
| 5 | Validity radius / realized trust | **ADOPT** | v16/v17 validity radius, J4 quarter-quantum cap, shrink ladder, exact receiver verdict, and rollback are admission invariants on every proposal. |
| 6 | px1 update-RMS and Muon | **ADOPT px1; DEFER Muon actuation** | Any optimizer A/B must match realized update RMS from copied state. `Muon`/`MuonWarmStart` remain registered but execution-disabled because the heterogeneous DDM coordinate surface has no matched DDM update-matrix and RMS receipt. No level-set trainer flag is presumed to be a DDM consumer. |
| 7 | Pose verdict gate | **ADOPT** | Consume the J7 five-point pose history as the engage detector. Pose evaluation may be skipped only while the typed gate proves the pose state frozen; every skipped row carries gate reason and liveness. The retired generic `PoseVerdictGate` DSL factory is not treated as evidence of DDM wiring. |
| 8 | Throughput stack | **ADOPT batch-32 verdict + grouped backward; DEFER micro-batch/async actuation** | Exact scorer verdicts retain batch 32 chunking. The final DDM-specific worst-geometry preflight measured the custom grouped backward active with bit-identical fused-R forward and gradient. Micro-batch and async authority eval remain disabled until their own DDM-specific bit-identity, deterministic ordering, and timing receipts exist; #410’s different trainer surface is not silently transferred. |
| 9 | #408 Q1-Q7 telemetry debt | **ADOPT** | Every continuation verdict emits Q1-Q7 state, `lever_engage`, `term_inert`, liveness stamps, event marks, proposal provenance, and confound alarms. Missing required telemetry refuses a promotable verdict. |
| 10 | EMA | **ADOPT** | Inference/verdict uses the EMA shadow. Decay is exactly `0.997`, resolved and cross-checked through `ema_decay_run_geometry_v1`; no fallback literal is accepted and every stage/event checkpoint preserves live plus EMA state. |
| 11 | Costate organ + G3/C1 telemetry | **ADOPT SENSE-only** | Costate organ may rank event predicates and selective unfreeze candidates but has `actuation=NONE`. G3 hard-pair and C1 residual-bucket telemetry feed scorer-recursive proposal aiming. Containment remains exact. |
| 12 | Terminal band | **ADOPT preregistration only** | #400 MC-finisher diagonal mode and `erm_margin_topk` K>1 batching are typed post-descent nodes with `execution_enabled=false`. This build records their budget/metric requirements but does not execute them. |

## Required capstone deltas

| Surface | Disposition |
|---|---|
| GC item 2 | Replace scalar acquisition with the Pareto pair `(g_S, g_L)`; dominance first, stable proposal ID tie-break only. |
| GC item 6 | Add exact charge audit: counted payload, code/runtime boundary, receiver parse-back, and source hashes. |
| GC item 9 | Preregister OP-GC1-5 fit with `execution_enabled=false`; no fit result is implied. |
| DM4 to J5 | Add a typed adapter whose rows include aimed cell, corrected-J row, scorer-derived support footprint, and `seg-only` / `pose-only(frame_0)` / `joint` type. Disabled adapter output must be byte-identical to the pre-adapter consumer. |

## Completeness sweep

`tac.witness_dsl.lever_registry.completeness()` on this worktree reports
438 trainer flags, 363 mapped flags, 75 unmapped flags, 3 stale emitted flags,
and coverage `0.8287671232876712`. This is repository-wide evidence, not a
claim that all 75 unmapped level-set flags belong in DDM.

The activation ledger reports the following campaign-applicable levers as
registered but never fired: `AdamBeta2`, `Beta2WindowRewarmup`,
`EmaDecayCalibrated`, `ForkHeadSolve`, `HeadOffsetSolver`, `MarginStepCap`,
`MicroBatch`, `Muon`, `MuonWarmStart`, `PoseVerdictGate`, `ResumeLRWarmup`,
`TerminalPoseFinish`, and `WarmStartRestoreBoundaryState`. This table adopts
the semantic capabilities above but does not convert a level-set activation
record into DDM measurement custody.

The #405 decision table and DAG/memory grep add no authority to fire those
levers here. They do identify four omission classes that the original #688
brief did not spell out and this landing therefore adds:

1. explicit resume-boundary placement before baseline-v0;
2. DDM-specific throughput custody rather than trainer-level transfer;
3. Q1-Q7/engage/inert/liveness telemetry as compile-required schema;
4. post-descent terminal-band nodes as preregistered, disabled continuations.

## Acquisition custody

Campaign acquisition values must be measured in the campaign. C1 reservation
identities are not prices. The compiler accepts `(g_S, g_L)` only when each
axis names its measurement receipt and evidence axis; missing or incomparable
custody makes the proposal ineligible rather than scalarizing it.

## Directive consumption

| UTC | Source | Disposition |
|---|---|---|
| 2026-07-24T14:45:16Z | operator | Scorer-recursive construction is binding; DM4 resize-footprint + stem-lattice support is the accepted constructor and generic spatial/history menus are excluded. |
| 2026-07-24T15:47:45Z | operator | All twelve composition items and the independent completeness sweep are incorporated above before code changes. |
| 2026-07-24T15:50:30Z | operator amendment | The box is a descent milestone, not a descent stop; measured exact delta-S and wall-clock/dynamics economics control continuation, while tolerance stopping applies only to describe/solve. |
