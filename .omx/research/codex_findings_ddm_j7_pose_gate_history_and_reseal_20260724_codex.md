---
title: Codex findings - DDM J7 #366 pose-gate history and reseal
utc: 2026-07-24T05:10:00Z
lane_id: lane_ddm_j7_366_pose_gate_history_and_reseal_20260724
verdict: BLOCKED_NO_LAUNCHABLE_WS1_START_AND_REALIZED_DSEG_REGRESSION
verdict_scope: J7 BOUNDED MACOS ADVISORY CONTROL AND CURRENT WS1 START-CUSTODY INSTANCE ONLY
research_only: true
execution_allowed: false
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# Outcome

J7 is **not READY**. The exact batch32 pose history is complete at five points,
but the latch classifies `DSEG_STILL_TRENDING` and step 4 triggers
`BLOCKED_REALIZED_DSEG_REGRESSION`. Independently, neither WS1 candidate is a
launchable start: both endpoint rows omit `archive_path` and `archive_sha256`
and neither supplies a live optimizer state. The preregistered four-step
W_seg/W_joint slope comparison is therefore `UNDECIDABLE_FAIL_CLOSED`, with no
selected start.

No campaign, paid dispatch, contest evaluation, score claim, promotion, or
pointer change occurred. The pointer remains
`0.1910828242 [contest-CPU]`.

# Round-1 findings and repairs

1. **Batch custody mismatch.** The consumer hardcoded exact-verdict batch 16
   although J7 requires batch 32. The typed compiler now reads the DSL field,
   restricts it to 16/32, and requires 32 for the J7 semantic SHA.
2. **Exact history suppression.** The launcher first made a bounded
   `--stage-exit-on-stop` verdict due, then reset it to false whenever the prior
   admission was component-safe. That is why the owed history stayed 2/5.
   Bounded stage exits now always emit the exact verdict.
3. **Endpoint/state category error.** The proposed WS1 arbitration assumed
   endpoint metrics plus a byte count were sufficient to initialize the J5
   consumer. They are not. The new resealer refuses any WS1 start without a
   real path, exact SHA/bytes, and receiver-closed parse-back custody.
4. **Stale source commit.** The inherited ticket still named its earlier
   source commit. The resealer now derives and records this worktree’s actual
   base HEAD (`7bdc9dbb9e304b6a2a8edb6d1979dda0bb4567bf`).
5. **Stale regression expectation.** The first focused-suite run found one
   test that still required the J6A semantic/typed hashes from the ticket J7
   was authorized to reseal. The test now checks the current J7 semantic SHA,
   typed hash, and batch32 contract. The post-repair suite is 49/49 green.

# Exact measured disposition

The inherited V15 control produced:

| step | d_seg | d_pose | bytes | decision |
|---:|---:|---:|---:|---|
| 0 | 0.027470296223958333 | 163.06132728121813 | 133941 | baseline |
| 1 | 0.02744209289550781 | 163.0611643144218 | 133936 | strict Seg admission |
| 2 | 0.02744209289550781 | 163.0611643144218 | 133936 | no component descent |
| 3 | 0.02744209289550781 | 163.0611643144218 | 133936 | no component descent |
| 4 | 0.027461522420247395 | 163.0610768919435 | 133936 | `BLOCKED_REALIZED_DSEG_REGRESSION` |

All five are exact n600, batch32, receiver-closed,
`[macOS-CPU frozen-scorer advisory]`, and `score_claim=false`. The detector’s
latest relative slope is `8.063256697554839e-05` with stderr
`1.2070409497686895e-04`; its hysteresis windows do not all satisfy the
non-positive flat condition, so the pose-finish latch stays off.

The fixed warm-start crossover is independently re-derived by the registered
callable as `R*=4.1215446777965665`. No measured slope is reported: the
required start states do not exist in launchable custody. The inherited V15
control is not mislabeled as W_seg or W_joint.

# Directive consumption

| directive | disposition |
|---|---|
| Optimal-form/holistic facet read | Applied: final verdict reports Seg, Pose, rate, per-class custody in the exact run receipt; scope is instance-only. |
| Reverse-waterfill / stop at rate break-even | Not actuated: no new residual or candidate allocator was authorized; no bytes were admitted from proxy EV. |
| Fisher-margin metric and corrected inner Jacobian | Preserved as the WS1 preregistered metric contract; no Euclidean/Fourier substitute was introduced. |
| Curvelet/shearlet residual basis | Not applicable to this control-history/reseal task; no residual basis was built or selected. |
| Pointer unmoved | Satisfied: `0.1910828242 [contest-CPU]`, `pointer_moved=false`. |

# Triality and canonical feed

- **DSL:** ticket semantic `bb30eade...`, typed hash `d8a8bb4f...`,
  `verdict_batch=32`, consumer `72343c91...`, launcher `4207705e...`.
- **DAG:** the executable and custody graph is in
  `ddm_j7_366_pose_gate_history_reseal_DAG_FEED_20260724.md`.
- **Equations:** `ddm_ws1_warm_start_slope_falsifier_v1` remains the callable
  slope authority; its exact R* replays. The J7 resolution is encoded in
  `ddm_train_decision_table_j7_resolution_20260724.json` so later planners
  cannot treat endpoint-only rows as executable states.
- **Continual-learning gate:** canonical probe outcome
  `ddm_j7_ws1_launchability_and_pose_gate_20260724` records a scoped blocking
  `DEFER` until both WS1 starts have receiver-closed archive/live-state
  custody and MAIN has reviewed this landing.

# Remaining exact blockers

1. A WS1 producer must materialize each candidate as a receiver-closed archive
   with exact path/SHA/bytes, or preserve a live optimizer state compatible
   with the J5 consumer. Only then can the preregistered four-step windows run.
2. The inherited-V15 control is below its stage target, the pose latch is not
   engaged, and the latest exact move regressed Seg. It is not fire-ready.
3. MAIN must independently review and land this branch before any future
   standing-GO decision. This branch confers no execution authority.

# STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; operating manual; v7.5/v8 specs;
J3/J5/J6A receipts and findings; WS1 receipt, typed slope spec, train-decision
table, and canonical equation registry; current ticket, consumer, launcher,
tests, exact SSD receipts/checkpoints; canonical lane/subagent/frontier/probe
state; councils, operator authorizations, and both delegation inboxes.
