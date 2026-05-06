# LA-POSE Motion Atom Identity Canonicalization - 2026-05-06

## Context

The LA-POSE motion-atom planner is planning-only score feedback for hard-pair
and geometry-aware archive search. Its records may arrive from JSON, component
response harvesters, or shuffled analysis jobs. Record order is not scientific
identity; `pair_index` is the contest-pair identity.

## Finding

`build_motion_atom_manifest()` previously consumed records in caller order.
That made `record_sha256`, sparse graph construction, atom order, and
meta-Lagrangian rows sensitive to input ordering. Duplicate `pair_index`
records also reached graph construction and the atom ledger, where duplicate
atom IDs could collapse during ranking.

Evidence grade: `empirical` tooling bug, not score evidence.

## Change

- Normalize records, sort canonically by `pair_index`, and compute manifest
  hashes/graphs/ledgers from that canonical sequence.
- Reject duplicate `pair_index` values before graph or ledger construction.
- Add regression tests proving reversed input order emits identical manifest
  identity surfaces and duplicate pair identities fail closed.

## Verification

Focused:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_lapose_motion_atoms.py \
  src/tac/tests/test_lapose_planning_chain.py \
  src/tac/tests/test_meta_lagrangian_allocator.py -q
```

Full preflight:

```text
.venv/bin/python tools/all_lanes_preflight.py --timings
```

## Promotion Status

This is harness and research-DX hardening only. It does not dispatch GPU work,
does not build an archive, and does not make a score claim. Any LA-POSE-derived
candidate still requires charged payload closure, no-op controls, and exact
CUDA auth eval before promotion.
