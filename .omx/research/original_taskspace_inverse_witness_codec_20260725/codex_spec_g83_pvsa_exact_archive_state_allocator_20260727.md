# G83 — PVSA exact archive-state allocator

Date: 2026-07-27  
Lane: `lane_g83_pvsa_exact_archive_state_allocator_20260727`  
Authority: deterministic allocation over caller-supplied exact upstream rows  
Status: research-only; no scorer launch, score claim, candidate claim, pointer
mutation, or promotion authority

## Outcome

G83 lands the smallest real allocation seam between G80/G82's compact selected
archive objects and future whole-object exact evaluation. It does not estimate
component deltas or build a candidate. It accepts a finite family in which
every node already has:

1. the actual selected archive bytes and SHA-256;
2. a complete 600-sample `upstream/evaluate.py` component row;
3. an exact contest CPU or contest CUDA axis;
4. one immutable evaluator/runtime/target/file-list custody identity; and
5. an ordered logical-actuator tuple bound one-for-one to exact
   decoder-transition IDs.

The allocator validates the entire state family, prices each node from its own
archive length and component row, applies the nonlinear contest objective,
Pareto-prunes only monotone same-custody axes, selects the globally minimum
exact state, and emits an exact add/remove/rollback route from the current
state. A fresh caller-supplied dynamic frontier snapshot is reopened before,
during, and after allocation. No component threshold exists.

G83 consumes the committed G80 `CompactPVSAArchiveBuildV1` and G82
`TSPPV2ToPVSA1LoweringV1` typed interfaces. It does not edit G78, G80, or G82.
The current PVSA1 wire realizes zero actuators or one G74 actuator. Future
multi-actuator archives enter only through the generic constructor after their
own receiver-closed bytes, validation receipt, and exact upstream row exist.

## The missing identity seam found by adversarial review

The initial implementation bound logical actuator tuples to G80/G82 only by
count. That was insufficient: an exact G74 byte stream could have been
relabeled as another logical actuator while preserving archive SHA and row.

The landed form closes that gap. Every state stores
`selected_decoder_transition_ids`; allocator validation requires each ID to
equal the registered transition for that logical actuator. G80/G82 adapters
derive their transition IDs from the parsed compact wire enum, for example:

```text
pvsa1:1:G74_ROLE_AWARE_PREPAINT
```

Thus archive bytes, parsed actuator type, logical state, transition order, and
component row are one same object rather than adjacent claims.

## Triality

DSL:

```text
Custody =
  exact_axis
  × evaluator_sha
  × upstream_snapshot_sha
  × runtime_tree_sha
  × target_sha
  × file_list_sha
  × context_epoch

State q =
  (archive_bytes_q, archive_sha_q, selected_actuators_q,
   decoder_transitions_q, complete_upstream_row_q, custody_sha)

allocate(Q, A, q_current, dynamic_frontier) -> q_star + exact_route
```

DAG:

```text
G80 compact build or G82 rich-to-wire lowering
  -> public receiver closure and archive validation receipt
  -> exact n600 upstream component row on one contest axis
  -> bind actual archive bytes/SHA + row + custody
  -> validate conditional prerequisites/conflicts/order/transition identity
  -> safe monotone Pareto prune
  -> exact nonlinear global argmin
  -> add/remove/rollback route
  -> compare selected same object to fresh dynamic pointer
```

Equations:

```text
S(q) =
  100*d_seg(q)
  + sqrt(10*d_pose(q))
  + 25*len(archive_bytes(q))/37_545_489

q_star =
  argmin_q (
    S(q),
    len(archive_bytes(q)),
    selected_actuators(q),
    state_id(q)
  )
```

Pareto elimination is valid only when one state weakly improves all three
primitive axes and strictly improves at least one:

```text
d_seg(a) <= d_seg(b)
d_pose(a) <= d_pose(b)
bytes(a) <= bytes(b)
```

All states are prevalidated on the same exact custody and axis. There is no
linearized pose delta, additive actuator-byte estimate, isolated component
threshold, or cross-axis comparison.

## Exact state-family contract

The actuator registry is ordered and typed. Prerequisites must precede the
actuator, conflicts must be symmetric, and a selected state must:

- contain known unique actuator IDs in registry order;
- bind each ID to its exact decoder transition;
- satisfy prerequisites using the already-executed prefix;
- contain no conflict;
- have every strict prefix present as an exact archive state; and
- include one zero-actuator semantic baseline.

The finite family may be sparse away from its strict prefixes. The allocator
does not manufacture an unmeasured combination. Routes use only supplied exact
states:

- `ADD`: append one actuator;
- `REMOVE`: remove the last actuator;
- `ROLLBACK`: return to a strict prefix while dropping two or more actuators.

Breadth-first routing minimizes transition count with a deterministic
tie-break. Every transition recomposes before/after score from that state's own
component row and exact archive length against the same fresh pointer.

## Adversarial composition fixture

The focused suite includes a seven-state structural fixture over actuators
`A`, `B`, and `C`. Its component values and archive lengths are explicitly
synthetic and carry no empirical authority.

It proves both directions that scalar gates miss:

- `B` improves the nonlinear score when added to the baseline, but the global
  optimum excludes `B`;
- `C` harms the score when added to the baseline, but `A+C` is the global
  optimum because the conditional composition changes segmentation, pose, and
  rate together.

Starting at `A+B+C`, the exact shortest route is rollback to `A`, then add
`C`. Starting at `B`, the route removes `B`, adds `A`, then adds `C`. Dominated
states are pruned only by complete `(d_seg, d_pose, archive_bytes)` dominance.

These are algorithmic fixtures, not score measurements. They demonstrate why
one arbitrary acceptable value for segment, pose, or rate is mathematically
invalid for this objective.

## Fail-closed surfaces

G83 rejects:

- missing, empty, or duplicate actuator states;
- missing zero-actuator baseline or strict rollback prefix;
- unknown, out-of-order, dependency-violating, or conflicting actuators;
- logical actuator relabeling of an exact decoder transition;
- archive payload/length/SHA/component-row identity drift;
- nonfinite, negative, partial, proxy, advisory, or non-n600 rows;
- row final-score drift from exact component recomposition;
- stale/foreign evaluator custody;
- contest CPU/CUDA mixing;
- missing, stale, forged, or mutated dynamic frontier snapshots; and
- G80/G82 adapter calls with any nonexact interface type.

An exact evaluation is not considered stale merely because time passes. It is
versioned by its complete custody identity. “Stale custody” means the state row
does not bind the caller-selected evaluator/runtime/target epoch.

## Verification

Focused:

```text
.venv/bin/pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_g83_pvsa_exact_archive_state_allocator_v1.py

16 passed in 0.78s
```

Adjacent:

```text
.venv/bin/pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_pvsa_compact_container_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g82_tsppv2_pvsa1_lowering_v1.py \
  src/tac/witness_dsl/tests/test_dynamic_frontier_target.py \
  src/tac/witness_dsl/tests/test_taskspace_whole_archive_allocator.py

56 passed in 54.45s
```

Static:

```text
ruff check: passed
ruff format --check: passed
python3 -m py_compile: passed
```

The G80 adapter test binds the retained 129,335-byte semantic-only selected
outer archive itself, not a synthetic byte sum. The remaining focused
selection fixtures are labeled structural and do not claim upstream evidence.
The adjacent G82 suite exercises the exact committed lowering, its G80
baseline/actuated archives, receiver equality, and fail-closed custody.

## Honest remaining product debt

G83 is ready to allocate exact rows, but no conditional `Y0|Y1` PVSA actuator
archive or full-n600 public evaluation row currently exists in this seam.
Therefore the next score-producing chain remains:

1. implement the new typed conditional `Y0|Y1` PVSA wire transition and
   deterministic receiver;
2. lower only its counted operand into G80's compact physical grammar;
3. close public `inflate.py` / `inflate.sh`, parse-back, exact expected-video,
   double-decode, and runtime custody;
4. evaluate zero, one, and composed actuator archives through full n600
   `upstream/evaluate.py` on one exact authority axis;
5. pass those same archive bytes and rows to G83 for nonlinear global
   selection; and
6. promote only if the selected exact archive beats the then-current dynamic
   frontier.

The allocation layer deliberately cannot substitute for those owed archives
or evaluations.

## Pointer-delta honesty

At landing, the dynamic pointer resolved to upstream official `0.172` from
pointer SHA-256
`b544fda607eae6bb02871fb145166773e05d2ef8b1a470c12996056e0b26235d`.
That value is an observed landing fact, not a compiled threshold. G83 neither
invoked the scorer/evaluator nor moved the pointer. Exact score claim:
**false**. Candidate claim: **false**. Frontier moved: **false**.
