# Master-Gradient Per-Pair Consumer False-Authority Hardening - 2026-05-17

## Status

- owner: Codex
- evidence grade: local unit tests only; no score claim
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- dispatch_spend: none

## Context

While resuming the per-pair master-gradient optimizer consumer wave, live
source churn was detected in the optimizer/test cluster. Edits were deferred
until a quiet watch window showed stable status and no source mtimes inside the
last minute.

The active WIP added three predicted-only consumer surfaces:

- `tac.optimization.bit_allocator_end_to_end.allocate_per_pair_bits`
- `tac.optimization.field_equation_planner.consume_per_pair_lagrangian_duals`
- `tac.optimization.jacobian_fisher_importance_allocator.allocate_per_pair_fisher_importance`

## Bugs Fixed

1. Non-hex archive IDs were accepted by the new consumer entrypoints as long as
   they were at least 12 characters. That was weaker than the canonical
   master-gradient loader contract and could let malformed custody IDs flow
   into predicted planner envelopes.
2. `allocate_per_pair_fisher_importance(bottom_k=0)` returned all byte indices
   because Python treats `sequence[-0:]` as `sequence[0:]`. The corrected edge
   case now returns an empty tuple.

Both fixes preserve the current 12+ hex prefix convention used by the
master-gradient sidecar filename surface while refusing non-hex custody IDs.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_bit_allocator_per_pair_consumption.py \
  src/tac/tests/test_field_equation_planner_lagrangian_consumption.py \
  src/tac/tests/test_jacobian_fisher_per_pair_consumption.py -q
# 39 passed in 0.37s

.venv/bin/python -m pytest \
  src/tac/tests/test_master_gradient_consumers.py \
  src/tac/tests/test_bit_allocator_per_pair_consumption.py \
  src/tac/tests/test_field_equation_planner_lagrangian_consumption.py \
  src/tac/tests/test_jacobian_fisher_per_pair_consumption.py -q
# 70 passed in 0.42s
```

## Non-Claims

This patch does not create a candidate archive, dispatch a provider job, or
promote any predicted per-pair optimizer output. It only hardens local custody
validation and a byte-index reporting edge case before these predicted surfaces
feed downstream planning.
