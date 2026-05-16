# L5 v2 Gate Artifact Predicate Identity Hardening - 2026-05-16

## Summary

Tightened L5 v2 dispatch-readiness evidence so each gate artifact must carry
its own `predicate_id`. The outer evidence row is no longer enough to identify
which predicate passed.

## Failure Class

- `l5_v2_gate_artifact_predicate_id_missing`

Before this hardening, an artifact with a correct `gate_id`, `passed=true`, and
valid semantic fields could omit `predicate_id` while the wrapper evidence row
supplied one. That allowed provenance to be split across two records and made
review brittle. The artifact now fails closed unless its predicate id is present
and matches the evidence row.

## Code Surfaces

- `src/tac/optimization/l5_staircase_v2.py`
- `src/tac/tests/test_l5_staircase_v2.py`

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q
# 165 passed
```

## Interpretation

This is an L5-v2 custody hardening, not a score claim. It reduces the chance
that a stale wrapper row or copied evidence record unlocks gate-probe dispatch
without the artifact itself proving which predicate it satisfied.
