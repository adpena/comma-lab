# Meta-Lagrangian Pareto Gate

Date: 2026-05-06
Agent: Codex
Evidence grade: empirical

## Finding

The meta-Lagrangian atom ledger ranked rows by expected total score delta, byte
delta, and atom id, but did not annotate Pareto dominance. A row could stay
high in the planning queue even when another rankable row in the same
replacement scope was no worse on expected total score, bytes, SegNet delta,
PoseNet delta, confidence, and archive-custody readiness.

## Change

`tac.optimization.meta_lagrangian_allocator` now annotates each row with:

- `pareto_scope`
- `pareto_frontier`
- `pareto_dominated_by`
- `pareto_objectives`

Dominance is scope-local by default (`family_group`) so orthogonal families
such as mask, pose, foveation, categorical, and HNeRV recode atoms are not
discarded against each other before stack interaction review.

This is a planning-order improvement only. It does not make any atom
dispatchable, exact-score-promotable, or evidence-grade promotable.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_meta_lagrangian_allocator.py -q
```

No GPU dispatch, remote staging, or lane claim was attempted.
