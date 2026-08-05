# NEXT_IF_RESUMED

DY1R NOTE 2026-08-05: this isolated-branch handoff has been consumed by
`.omx/research/ddm_dy1r_20260805/RECEIPT.md`. Use the dy1r receipt and next
file as the current authority.

1. On the next authorized v3 smoke, require `scope_law_resolution` telemetry rows for all declared laws. Treat declared-but-unresolved laws as an inertness failure, not as absent evidence.
2. Extend the migration queue in this order: lane-guard ratchet horizon, deterministic-R decision surface, JD1 live `w_pose` lower/retreat arm, EN1 margin-weight steering, SL2/PE3 switches, m51 governance fire-order, #847 NONE/no-default knobs.
3. Re-run:

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  -m pytest -q src/tac/witness_dsl/tests/test_scope_laws.py \
  src/tac/tests/test_ddm_bp1_boundary_reset_race.py -k 'scope_law or jd3'
```

4. Do not launch scorer, MLX, or training work from this handoff unless the active lane/scorer slot is free and the launcher governor accepts the run.
