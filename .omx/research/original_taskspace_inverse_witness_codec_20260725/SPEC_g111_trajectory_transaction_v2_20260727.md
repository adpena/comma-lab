# G111 Native-v3 Trajectory Transaction

Date: 2026-07-27  
Status: SPECIFIED; IMPLEMENTATION AND PHYSICAL PROOF OWED  
Lane: `lane_g111_fresh_batch16_v9_training_binding_20260727`  
Authority: original local implementation only; no launch, archive, score, or pointer claim

## Purpose

This is the one-time crash-continuity gate between the original G111
task-space/inverse-solver producer and the score-moving G121 -> G119 -> G110
archive loop. It is not a new representation family and must not become a
permanent local-polish campaign. Once this contract has deterministic code
proof and a fresh v8 two-pass physical dry-start, the next action is the real
full-n600 producer and exact archive scoring against the live `0.172` pointer.

No historical V15/C1 payload, public PR archive, or borrowed learned payload is
admissible. The lineage begins at a new deterministic epoch-zero state produced
by our own current DSL and physical G109 target custody.

## Structural correction

The fourteen previously named domains are a **semantic coverage checklist**,
not a sound key-ownership partition:

- a rollback savepoint contains a historical image of current train state;
- Tail inputs, verdict history, pending work, and reducer progress are one
  transaction;
- fresh lineage is an envelope derived from the complete serialized state;
- duplicate event booleans must be derived from canonical gate state;
- BEST selection state and filesystem publication are related but different
  atomic objects.

Native-v3 therefore uses six disjoint atomic owners and maps all fourteen
semantic domains onto them. The old
`g111_complete_trajectory_state.v1` skeleton cannot establish complete
trajectory proof and remains fail-closed.

## Atomic owners

### O1 `current_train_state`

Owns the exact current optimization coordinate:

- primary live model, EMA, optimizer family, optimizer leaves, epoch, stage,
  and completed-step coordinate;
- protected-island sparse residual, independent AdamW state, logical geometry,
  deterministic support identity, selected-row keysets/dtypes/shapes, and
  support hash when active;
- Polyak arm/start/count and, only when count is positive, the exact heavy mean
  leaf set matching the primary parameter topology.

Activity and expected leaf inventories are derived from the compiled DSL and
freshly constructed model/optimizers, never from checkpoint claims.

### O2 `rollback_savepoint`

Owns the rollback supervisor and an immutable, content-addressed last-good
savepoint:

- rollback count/max, LR scale, last-good coordinate, spike counters,
  exhausted latch, recent-loss window, and re-arm state;
- one savepoint ID and exact inventory/hash of rollbackable primary live, EMA,
  primary optimizer, protected seed, and seed optimizer state;
- savepoint topology and support identity equal to O1.

The savepoint may be embedded in the sidecar for the first closure. It must use
a distinct namespace and sparse selected-row seed form. A later external
content-addressed savepoint is allowed only with atomic retention and recursive
physical custody. A loose path is forbidden.

### O3 `schedule_control_state`

Owns every value that can change a future update, stage, LR, loss weight,
controller decision, or stop:

- all canonical event gates and their exactly-once action latches;
- `last_boundary_epoch`, `prev_seg_form`, `muon_switched`, rewarmup state, and
  any non-derived optimizer-group LR/multiplier;
- event curriculum, tau advance, birth completion, closed-loop controller and
  liveness state;
- ladder lambda/previous-radius/rung values and the state needed to rebuild
  the exact island-weight field;
- TailController state;
- Jacobian-basin cadence/failure/plateau state and active pose-conditioning
  detector state;
- every other active registered controller proven by reverse reachability to
  influence a later update, stage, stop, BEST decision, or output.

Observer-only telemetry may be excluded only by a code-level no-actuation
inventory. Inactive controllers claim no keys.

### O4 `verdict_transaction`

Owns one monotone, bounded, main-thread-reduced verdict journal and every
history consumed by O3:

- canonical submission and application sequence numbers;
- last applied result ID/hash;
- the complete bounded verdict rows used by closed-loop and Tail;
- Tail byte rows;
- lane, annulus, label-floor, pose-conditioning, and other active sensor
  histories;
- async skipped-count and reducer status where future behavior reads them.

Native-v3 chooses a **strict quiescent checkpoint barrier**, not persisted
in-flight replay:

1. acquire checkpoint barrier and stop new verdict submissions;
2. join any worker;
3. worker returns an immutable result and never mutates journals/controllers;
4. main-thread reducer applies completed results in sequence order exactly once;
5. assert `pending_count == 0` and `next_submit_seq == next_apply_seq`;
6. snapshot O1-O6 while the barrier remains held;
7. atomically publish, then release.

Native-v3 rejects every `__cl_pend_*` member. The old pending-snapshot replay
path is legacy-v2 compatibility only and can never produce
`complete_trajectory_proven=true`.

### O5 `causal_selection_state`

Owns causal output selection:

- current BEST metric, epoch, deterministic tie break, and content hash of the
  referenced deploy checkpoint;
- preserved stage-checkpoint inventory with epoch/stage/kind/file SHA;
- causal checkpoint boundary coordinate;
- fork-EMA clearance and any other latch that can suppress or admit BEST.

Publication is a content-addressed atomic commit: after a crash, readers see
either the old complete pointer or the new complete pointer, never a dangling
or mixed pointer. Wall-clock timestamps are provenance, not tie breakers.

### O6 `lineage_envelope`

Owns immutable configuration and physical ancestry:

- semantic schema, compiled DSL identity, seed, target projection and physical
  G109 authority;
- exact static configuration members needed to reconstruct O1-O5;
- immutable cold-root identity;
- checkpoint coordinate:
  `epoch`, `stage`, `boundary_kind`, `optimizer_step_complete=true`,
  `pending_gradients=false`, `pending_verdict_count=0`;
- parent receipt/checkpoint identity and the derived complete-state hash.

Derived physical lineage fields are validated as an envelope over O1-O5 plus
the non-derived O6 members. They are not allowed to self-attest.

## Fourteen-domain coverage map

| Semantic checklist domain | Atomic owner(s) |
|---|---|
| primary model / EMA / optimizer family | O1 |
| protected seed / optimizer / support | O1 |
| fresh root / physical lineage | O6 |
| RNG streams | O3 |
| event gates / duplicate booleans | O3 |
| stage transition / rewarmup | O3 |
| spike rollback / last-good snapshot | O2, cross-checked with O1 |
| ladder | O3 |
| Tail / verdict inputs | O3, O4 |
| verdict journal / sensor histories | O4 |
| pending verdict / reducer boundary | O4 |
| Jacobian basin | O3, O4 |
| Polyak atomic state | O1 |
| BEST / stage bookkeeping | O5 |

The manifest records this complete coverage map so future agents cannot add a
trajectory-affecting object without assigning it to an owner and checklist
domain.

## Native-v3 manifest and reverse coverage

Schema: `g111_trajectory_transaction.v2`.

For serialized arrays `A`, manifest key `M`, atomic owner `c`, and the
independently constructed expected topology:

```text
K = keys(A) - {M} - derived_physical_lineage_keys
O_c = actual key set claimed by owner c
R_c = independently derived required key set for c
E_c = independently derived permitted key set for c

union(O_c) = K
O_i intersect O_j = empty                    for all i != j
active(c)  => R_c subset O_c subset E_c
inactive(c) => O_c = empty
```

Every entry carries canonical key, owner, dtype, shape, byte length, and
SHA-256. Object arrays and pickle are forbidden. Prefixes are namespace guards,
not evidence of completeness.

Validation must reject:

- missing, extra, multiply owned, unknown, or prefix-impostor keys;
- missing/extra owners or semantic-domain coverage;
- activity drift in either direction;
- dtype, shape, length, hash, finiteness, topology, family, or support drift;
- empty wildcard matches and dummy-scalar substitutions;
- unequal parallel-history lengths, non-monotone/duplicate journal sequence,
  or invalid bounded-history truncation;
- Polyak scalar/heavy asymmetry;
- native-v3 pending-verdict payloads;
- a BEST/stage pointer whose content hash cannot be reopened;
- a rollback savepoint whose trainable topology/support differs from O1.

After staged restore, reserialize O1-O5 and require the same canonical keyset
and semantic hashes before permitting a verdict worker or optimizer update.

## Restore transaction

1. Load NPZ into immutable staging. Execute no restore callbacks.
2. Validate schema, coordinate, DSL/target custody, every entry hash/dtype/
   shape, the independently derived owner schemas, reverse coverage, and
   physical lineage.
3. Construct the expected model, optimizers, seed, controllers, registry, and
   journal without starting workers.
4. Stage O1 and O2; validate optimizer/seed/Polyak/savepoint cross-invariants.
5. Restore RNG before any stochastic operation.
6. Stage gates, schedule, tau, birth, ladder, closed-loop, Jacobian/pose, Tail,
   and the verdict transaction.
7. Deterministically rebuild derived island fields, LR/stage values, and
   duplicate booleans from their canonical owners; assert equality to any
   retained duplicate sentinels.
8. Stage and physically reopen O5 references.
9. Reserialize and compare canonical semantic hashes.
10. Publish all staged state atomically, enable workers, and permit the next
    decision/update.

Any failure leaves the cold constructed objects unchanged and exits before the
first update. The immediate implementation may use a single late publish point
inside the trainer, provided no training, verdict submission, or BEST mutation
can happen before it.

Checkpoints are legal only at cold-root, epoch-end, or stage-end quiescent
boundaries. Mid-microbatch checkpoints are forbidden; therefore pair cursor,
partial gradients, and partial losses are intentionally outside this schema.

## Implementation units

1. Add a reusable typed transaction module under
   `src/tac/witness_control/` containing descriptors, entry hashes, owner and
   domain validation, canonical journal/barrier state, and pure NumPy staging.
2. Route the trainer's native-v3 writer through one snapshot transaction.
   Register every active controller before this writer can run.
3. Replace worker-side history/controller mutation with immutable result return
   plus main-thread reducer, then place the strict barrier around every native-
   v3 checkpoint.
4. Add atomic sparse-aware rollback savepoint packing/restoration.
5. Add controller, journal, BEST/stage, and derived-state adapters.
6. Route native-v3 continuation through pre-publish validation and one late
   restore transaction. Preserve legacy-v2 behavior with proof forced false.
7. Emit the physical lineage only after the complete manifest is present in
   the semantic state hash.

## Decisive proof

The implementation is not closed by parser or round-trip tests alone. Required:

1. parameterized delete-one-key across every active owner;
2. unowned key, fake prefix, duplicate owner, unknown owner/domain, and
   activity-drift failures;
3. dtype/shape/hash/nonfinite/history-length/order/duplicate-sequence failures;
4. restore-failure injection at each phase with cold objects unchanged;
5. post-capture live-array/list mutation cannot alter written state;
6. checkpoint during active async verdict joins and applies exactly once,
   emits no pending keys, and preserves one reducer transition;
7. continuous vs interrupted exact next-step state across seed update,
   rewarmup, event transition, ladder refresh, Tail mid-cycle, and pose/Jacobian
   cadence;
8. interrupted immediately before rollback produces the exact continuous
   post-rollback primary/EMA/optimizers/seed/counters/LR/window state;
9. crash between candidate checkpoint and BEST publication exposes only an
   old-complete or new-complete pointer;
10. legacy-v2 seed-off still loads additively but cannot open a fresh root or
    claim complete trajectory proof.

Focused Ruff, `py_compile`, existing resume/lineage/sparse/Polyak tests, and the
new transaction suite must all pass. The four stale governed-DSL test fixtures
must be corrected or explicitly scoped before the literal focused command is
reported green.

## Physical release gate

After code proof, generate a **fresh v8** clean-directory governed dry-start.
It must perform:

- cold root -> at least one update -> quiescent checkpoint;
- process exit -> same-lineage resume -> next update/checkpoint;
- uninterrupted control to the same coordinate;
- exact complete-state and next-decision comparison;
- immutable deploy/resume/receipt custody and storage preflight.

Only then may the real resumable full-n600 G111 producer start. G121 compiles
every immutable stage through the exact public G105 wire, G119 inverse-solves
the conditional pose operand, and G110 performs whole-archive arbitration,
double decode, and exact upstream CPU/CUDA evaluation. The effective pointer
moves only on those exact archive bytes and scores.

## Triality plus task-state leg

DSL:

`G111NativeV3TrajectoryTransaction(owners=6, semantic_domains=14,
checkpoint_boundary=quiescent_epoch_or_stage_end,
pending_verdict_policy=strict_barrier)`

DAG:

`compiled G111 DSL + physical G109 target -> construct all active objects
-> native-v3 barrier -> O1..O6 snapshot -> reverse-coverage/hash validation
-> atomic resume + physical lineage -> fresh v8 two-pass proof
-> n600 G111 -> G121 exact-wire stages -> G119 conditional inverse
-> G110 archive race -> public double decode -> upstream exact score`

Equations:

`union_c O_c = K`, `O_i intersect O_j = empty`, and
`Resume(Checkpoint(x_t)) = x_t` with the same next deterministic transition
`F(x_t)` as uninterrupted execution.

Task-state:

The durable parent remains
`pact-g111-complete-trainable-state-resume-20260727 = in_progress` until code
proof and fresh v8 physical proof close. Implementation children are marked
complete only after their exact patches and tests are committed. The dry-start
and full-n600 tasks remain blocked, not reopened or rediscovered, until that
parent is green.
