# tac.boosting parallel-merge runtime hardening - 2026-05-17

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

## Context

The current L5/L5-v2 and stack-of-stacks control plane depends on a reliable
composition abstraction. The untracked `tac.boosting` namespace WIP correctly
documented `|` as sequential composition and `&` as side-by-side parallel
merge, but the runtime implementation had not yet earned that claim.

## Finding

`ComposableBoostingPipeline.__and__` tagged a stage as
`composition_kind="parallel"`, and `build()` allowed overlapping emitted keys
for that tag. However, `run()` still executed the stage list in a plain
left-to-right loop:

```text
state = dict(state)
state.update(stage_output)
```

That meant `(A | B) & C` was only a build-time annotation. At runtime, a
parallel sibling could read the previous sibling's output, and conflicts were
resolved by accidental stage order rather than the declared `merge_policy`.

This is a material stack-of-stacks false-authority class: a future L5-v2 or
Rule #6 composition could appear to test independent byte regions while
actually testing an order-dependent cascade.

## Patch

Runtime execution now groups one sequential root plus all following parallel
siblings into a single execution group.

- Every sibling in the group receives the same pre-group input state.
- Accepted outputs are merged into a group-local output map, then applied once
  to the running state.
- Conflicts honor the incoming stage's concrete `merge_policy`:
  `last_writer_wins`, `first_writer_wins`, `additive`, `concatenate`, or
  fail-closed `explicit`.
- Pareto rejection still records the rejected stage and does not merge its
  output.
- The test-only registry clear helper now clears both the contract registry and
  function registry, preventing stale stage handlers from surviving fixture
  resets.

## Verification

Focused tests:

```text
.venv/bin/python -m pytest src/tac/tests/test_tac_boosting.py -q
87 passed in 0.33s
```

New coverage:

- `test_clear_registry_removes_stage_functions_too`
- `test_parallel_merge_runs_siblings_against_same_input_state`
- `test_parallel_merge_additive_policy_sums_overlapping_numeric_keys`
- `test_parallel_merge_explicit_policy_requires_non_conflicting_keys`

## Score-Lowering Relevance

This does not claim a score improvement. It prevents the new lattice/graph
composition helper from producing misleading evidence. The next score-relevant
use is still a byte-closed stage that emits consumed archive bytes, but now
the `&` primitive is closer to the semantics needed for L5-v2 / Rule #6
stack-of-stacks experiments.
