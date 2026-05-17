# TAC Compress-Time Optimization Determinism + Parallel Hardening - 2026-05-17

## Scope

This landing hardens the new `tac.compress_time_optimization` namespace before
it becomes an authority-bearing score-lowering actuator. It is a concrete
artifact-producing bug hunt on the current WIP partner surface, not a score
claim, dispatch, promotion, or architecture-lock packet.

The namespace is intended to support L5-v2 / Rule #6 / stack-of-stacks
compress-time search. False determinism or order-dependent composition here
would directly corrupt future candidate ranking.

## Findings

### 1. `SeedRequiredViolation` existed but never fired

The decorator docstring, exported error hierarchy, and error-class text all
claimed that deterministic compress-time passes must either pin
`contract.seed` or accept a `seed=` kwarg. The implementation had an inert
B-check: it described `SeedRequiredViolation`, then fell through without
raising.

Failure class:

- deterministic compiler guard presented as present;
- pass functions could later add hidden randomness without any explicit seed
  contract;
- future archive-build or compress-search stages could become
  non-reproducible while still passing decorator registration.

Fix:

- `tac.compress_time_optimization.decorator._check_determinism_invariant`
  now raises `SeedRequiredViolation` whenever `deterministic=True`,
  `contract.seed is None`, and the callable signature lacks `seed`.
- Deterministic fixtures now expose `seed=0` defaults when they intentionally
  rely on signature-level seeding.
- Added a regression test proving rollback removes the partially registered
  pass after the seed violation.

### 2. `&` parallel merge executed as sequential `|`

`ComposableCompressPipeline.__and__` marked a stage as `composition_kind=
"parallel"`, and `build()` allowed overlapping emits for that explicit merge
intent. Runtime `run()` still iterated left-to-right, mutating state after
each pass. That meant the second sibling in `A & B` could see `A`'s output,
so the supposed parallel group silently degenerated into an order-dependent
sequential chain.

Failure class:

- stack-of-stacks experiments become order-dependent;
- sibling passes can accidentally consume prior sibling outputs that are not
  in the contract;
- overlapping emits can be resolved by last-writer accident instead of a
  declared merge policy;
- future L5-v2 / Rule #6 compress-time search can fabricate or suppress
  signal depending on pipeline order.

Fix:

- `ComposableCompressPipeline.run()` now groups each sequential root plus
  trailing parallel siblings.
- Every pass in the group observes the same pre-group input state.
- Accepted outputs are merged only after the group finishes.
- Merge conflicts are resolved by the pass contract's `merge_policy`:
  `last_writer_wins`, `first_writer_wins`, `additive`, `concatenate`, or
  fail-closed for `explicit`.
- Wallclock is still charged conservatively by observed local elapsed time,
  even for logical parallel groups, so smoke budgets cannot undercount.

### 3. Failed re-decoration could erase a valid pass

The decorator allowed idempotent re-decoration with the same contract identity,
but rollback did not distinguish a fresh registration from a pre-existing
valid registration. A second failed decoration with the same contract could pop
the contract registry entry while leaving the old callable in the function
registry.

Failure class:

- valid pass registration can be corrupted by a later bad import/decorator
  attempt;
- registry and function sidecar can diverge;
- preflight may see a different pass set than runtime dispatch.

Fix:

- decorator rollback now removes registry state only for fresh failed
  registrations;
- failed same-contract re-decoration preserves the already valid contract and
  callable;
- regression tests cover both non-callable and deterministic-seed rollback
  against an existing registration.

## Tests Added

- deterministic pass without `contract.seed` or `seed=` signature raises
  `SeedRequiredViolation` and rolls back registry state;
- failed same-contract re-decoration preserves the previous valid registration;
- parallel siblings cannot observe prior sibling output;
- additive merge sums overlapping numeric keys;
- explicit merge policy fails closed on overlapping keys.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_tac_compress_time_optimization.py -q
```

Result:

```text
108 passed
```

## Routing Consequence

This namespace remains a score-lowering infrastructure surface, not a frontier
claim. It is now safer to use as the canonical compress-time composition layer
for:

1. L5-v2 / TT5L side-info effect-curve search;
2. Rule #6 byte-closed bolt-ons;
3. grammar-aware master-gradient operator rows;
4. stack-of-stacks candidate assembly.

Any future compress-time pipeline that emits a candidate archive still needs
byte closure, inflate-consumption proof, component rows, axis labels, result
review, and paired CPU/CUDA custody before promotion.

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
