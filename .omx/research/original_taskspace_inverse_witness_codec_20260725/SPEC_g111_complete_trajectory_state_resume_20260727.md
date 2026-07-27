# G111 Complete Trajectory-State Resume Closure

Date: 2026-07-27  
Status: PARTIAL IMPLEMENTATION CLOSED; FULL CONTRACT BLOCKED FAIL-CLOSED
Lane: G111 fresh batch-16 V9 semantic-base producer  
Authority: local implementation and deterministic tests only; no score claim

Durable status receipt:
`g111_complete_trajectory_state_resume_blocker_20260727.json`. Device-first
protected-seed custody, semantic cold-root binding, optimizer-family custody,
Polyak atomicity, and the total-manifest validator are implemented and tested.
The remaining controller/journal/snapshot adapters are not implemented; a new
fresh G111 cold-root launch is deliberately refused until the total fourteen-
component manifest exists and validates.

## Objective

Make the active G111 producer crash-resumable as a complete deterministic
dynamical system. A checkpoint filename, model tensor match, or parseable
sidecar is insufficient. For every stateful component that can change a later
update, checkpoint write and resume restore must be symmetric and fail closed
when active state is missing, extra, corrupt, or incompatible.

The fresh producer root must bind the complete initialized trajectory state,
including the protected-island seed module, its independent AdamW state, and
its sparse support identity. No fresh physical dry-start or full-n600 launch is
authorized until this contract is green.

## Constraints

- Preserve `main` as the only source of truth and do not touch unrelated dirty
  files.
- Preserve default-off and legacy behavior outside the fresh G111 contract.
- Preserve public contest custody, NO-FAKE, deterministic reproduction, exact
  parse-back, and stage-checkpoint requirements.
- Never materialize the dense n600 protected-island seed or its Adam moments on
  the host. Select support rows in MLX before NumPy conversion.
- Warm-start may intentionally discard training-only state, but ordinary resume
  must fail closed on partial active state.
- Unknown, missing, duplicated, or prefix-colliding active components are
  integrity errors, not warnings.
- Do not launch training or evaluator work in this implementation unit.

## Required active state

The active-component manifest must cover, at minimum:

1. primary live model, EMA, primary optimizer, and optimizer family;
2. protected-island seed residual, independent AdamW state, support geometry,
   support SHA, exact keysets, dtypes, and logical shapes;
3. fresh producer immutable root and physical checkpoint lineage;
4. RNG streams;
5. event gates and duplicate gate booleans;
6. stage-transition rewarmup boundary;
7. spike rollback controller metadata, counters, LR scale, recent-loss window,
   and the last-good rollback snapshot when one exists;
8. ladder lambda, previous radius, and rung state;
9. TailController state and its verdict input history;
10. verdict journal plus each active label-floor/lane/annulus history used by a
    later controller;
11. pending async verdict identity and reducer-consumption state, or a strict
    checkpoint barrier proving no pending verdict exists;
12. Jacobian-basin failure/cadence state;
13. Polyak scalar and heavy state as one atomic component;
14. BEST pointer and stage-checkpoint bookkeeping where those can change later
    selection or controller behavior.

Any item not active for the concrete G111 DSL must be recorded as inactive by
the derived active-component contract rather than silently omitted.

## Implementation shape

- Reuse and strengthen `ResumeRegistry`; add a typed complete-trajectory
  manifest rather than creating a second uncoordinated checkpoint system.
- Prefer small reusable `Resumable` adapters for scalar mappings, bounded JSON
  state, controller state dictionaries, and atomic coupled components.
- Create one transactional verdict journal consumed by every verdict-driven
  controller. Checkpoints must occur only at a reducer-consistent boundary, or
  persist/replay a pending verdict exactly once.
- Move fresh-root initialization until every trainable component and optimizer
  has been initialized. Hash a namespaced, complete epoch-zero snapshot.
- Make sparse seed packing memory safe by gathering support rows on device and
  serializing only selected rows plus exact logical/support metadata.
- Make Polyak scalar-without-heavy and heavy-without-scalar fatal on ordinary
  continuation.
- Restore gate booleans and controller state before any post-resume decision can
  fire.

## Acceptance criteria

The implementation unit is complete only when all of the following pass:

1. Focused static checks:

   `.venv/bin/ruff check --select F821,F823 experiments/train_levelset_witness_realized_through_R_mlx.py src/tac/witness_control/fresh_producer_lineage_v1.py src/tac/witness_control/sparse_auxiliary_resume_v1.py`

   `.venv/bin/python -m py_compile experiments/train_levelset_witness_realized_through_R_mlx.py src/tac/witness_control/fresh_producer_lineage_v1.py src/tac/witness_control/sparse_auxiliary_resume_v1.py`

2. Focused unit suites:

   `.venv/bin/python -m pytest -q src/tac/witness_control/tests/test_sparse_auxiliary_resume_v1.py src/tac/witness_control/tests/test_fresh_producer_lineage_v1.py experiments/tests/test_levelset_checkpoint_resume.py`

3. New tests prove:

   - sparse seed pack/restore is exact and never requires dense host tensors;
   - support/key/dtype/shape/config tampering fails closed;
   - fresh root changes when any initialized trainable/optimizer/support state
     changes;
   - every active component is present exactly once in the manifest;
   - missing or extra active state fails closed;
   - Polyak scalar/heavy partial state fails closed;
   - continuous two-step and interrupted-after-one-step trajectories are exact
     for the full active G111 state on a small deterministic fixture;
   - boundary tests cover seed update, rewarmup, rollback, event transition,
     ladder refresh, Tail mid-cycle, and pending-verdict handling.

4. A fresh local G111 dry-start may be launched only after code review and the
   above proof. The physical proof must use a new v8 run directory and must
   preserve both uninterrupted and resumed control evidence.

## Explicit non-goals

- No historical V15/C1 payload reuse.
- No proxy score, hypothetical candidate, or pointer movement.
- No claim that the full-n600 producer is complete merely because unit tests
  pass.
- No change to the frozen upstream evaluator or scorer.
