# Codex findings: G111 live-loop native-v3 integration

Date: 2026-07-27  
Actor: Codex/root with independent Sol xhigh adversarial review  
Verdict scope: trainer live-loop integration only  
Candidate claim: false  
Score claim: false  
Pointer moved: false  
Research only: false

## Executive finding

The committed native-v3 transaction/barrier core at `f29a4efdeb` is sealed, but
the current trainer is not yet a native-v3 state machine. The missing
composition layer is an exactly-once transaction from immutable scorer result
through pure reducer state, controller/history updates, BEST selection,
checkpoint publication, restore, and the next optimizer step.

The most important newly explicit state is O5/BEST. BEST is not merely another
dictionary: it is an irreversible scorer-result side effect whose file writes,
cursor, component values, and lineage must commit exactly once with the verdict
that selected it. A worker-side BEST write or a resume that forgets `_best` can
silently overwrite the true pre-resume witness with a later worse result even
when model/EMA/optimizer tensors themselves resume correctly.

## Current exact blockers

1. **Worker impurity and suppressed failure**
   - `_schedule_async_verdict` workers call `_emit_verdict_row`,
     `_maybe_preserve_best`, telemetry settlement, printing, controller/history
     mutation, and exception suppression.
   - Native-v3 workers must only compute and return an
     `ImmutableVerdictResult`. Typed worker/reducer failures must poison the
     transaction and forbid later submit, checkpoint, BEST, or optimizer update.

2. **Pose counter is mutated in the worker**
   - `_verdict_v` advances `_pose_verdict_count` off the main thread.
   - Reserve the pose-verdict index exactly once on the main thread at submit
     time and carry it in the immutable snapshot.

3. **Self-orient snapshot key type is incompatible**
   - The directional snapshot currently uses integer pair-index keys.
   - The sealed freezer requires canonical string mapping keys. Encode the
     aligned state as a tuple/list or exact string-key mapping and update the
     read path consistently.

4. **O4 history is not complete resume state**
   - `history`, verdict counts/skips, closed-loop verdict journal, pose counter,
     wire histories, and the bounded applied-result journal affect Tail, tau,
     Muon, event decisions, and final result selection.
   - They must be captured and restored before any decision or new submission.

5. **O5 causal selection is not restored**
   - `_best`, fork-EMA clearance, stage-checkpoint inventory, causal ordinal and
     coordinate, and physical BEST/stage reference inventory affect future
     selection and publication.
   - A worse first verdict after resume must never replace a better pre-resume
     BEST.

6. **Restore order mutates before total validation**
   - The current path mutates model/EMA and calls registry/controller restores
     before a total six-owner transaction is staged.
   - Required order is: construct fresh expected topology; load immutable NPZ;
     validate manifest, exact reverse coverage, O4 barrier, and cross-invariants;
     stage O1/O2/O3/O4/O5; reserialize and compare O1-O5; stage O6; publish once;
     only then enable workers or optimizer updates.

7. **Reducer and side-effect publication are conflated**
   - The reducer must return a pure replacement state and must not print, write
     BEST, mutate live controllers, or touch files.
   - Main-thread publication consumes the reduced intent exactly once. O5 BEST
     file publication and its cursor/lineage are the remaining nontrivial design
     crux.

8. **Checkpoint publication is outside the barrier**
   - Every `_do_checkpoint` call must hold
     `QuiescentVerdictTransaction.checkpoint()` from before snapshot through all
     deploy/resume/stage/periodic saves and lineage-tip publication.
   - Native-v3 forbids every `__cl_pend_` payload. The final checkpoint drains
     through the same barrier; do not retain a separate double-join path.

9. **Late-bound state must move before cold-root staging**
   - O3 includes closed-loop state, event state, birth state, ladder scalars,
     w-pose law state, Jacobian basin, pose-gate detector, liveness, previous
     segmentation form, Muon switch, last boundary, freeze/MSC/SG/tail state,
     rollback savepoint, RNG, and hardness state.
   - O4 and O5 state named above must also exist before the cold-root capture.

10. **The current adapter is only a foundation**
    - Same-source expected schemas are not independent authority.
    - A manifest key's mere presence cannot clear a cold-root gate.
    - Production must build the expected schema from freshly constructed live
      topology independently of the captured checkpoint payload.

## Minimal real native-v3 flow

1. Construct all O1-O6 live state and independent expected topology.
2. Stage/validate a cold-root transaction before creating the executor.
3. Create one `ThreadPoolExecutor(max_workers=1)` after successful
   construction/restore and close it in `finally`.
4. At an evaluation boundary, reserve the pose index on the main thread,
   capture canonical detached scorer inputs, and submit through the transaction.
5. Worker computes scorer output only and returns an
   `ImmutableVerdictResult`.
6. Pure reducer updates detached O3/O4/O5 intent in submission order.
7. Main thread publishes the reduced state and side effects once, runs the
   decide-on-previous controller, then permits the next submission/update.
8. `_do_checkpoint` drains through the transaction barrier, captures O4 from
   the barrier, serializes all owners, validates and self-reopens the exact
   arrays, and keeps the barrier held through every file/lineage publication.
9. Restore validates and stages everything before mutation, publishes once,
   then enables the executor.

## Decisive tests owed before fresh v8

- worker-thread purity plus main-thread-only pose-index reservation;
- real-shaped self-orient snapshot acceptance;
- decide-on-previous ordering;
- active-worker checkpoint drains exactly once with zero pending state;
- typed nonfinite/worker/reducer failure permanently forbids later state
  mutation, BEST, checkpoint, submission, and optimizer update;
- continuous versus interrupted exact next step across
  model/EMA/optimizer/O3/O4/O5/RNG;
- restore-before-mutation sentinel;
- worse-post-resume BEST regression;
- AST/source-order dominance guard for every checkpoint dependency;
- native-v3 archive contains no `__cl_pend_` key in any namespace.

## Operational disposition

- `f29a4efdeb`: completed and authoritative for the generic transaction/barrier
  primitive.
- `8b1395525e`: completed and authoritative for G111's typed zero Pareto tail
  floor; it does not authorize a launch.
- Adapter work may land only with an explicit adapter-only verdict.
- `pact-g111-complete-trainable-state-resume-20260727` remains in progress.
- Fresh v8, real n600, G121/G119/G110 composition, and authoritative evaluation
  remain blocked until the real live-loop proof above is green.

