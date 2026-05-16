# L5 / Cathedral Autopilot Campaign Blocker Hardening - 2026-05-16

## Scope

Hardened the Cathedral autopilot composition-ranker path so long-burn
campaign rows remain visible for planning, but cannot silently pass into
operator-authorized dispatch as clean candidates.

## Finding

The substrate composition matrix carried Z3/Z4/Z5/C1/C6 campaign rows with
predicted deltas and low smoke/campaign cost estimates. The ranker marked the
serialized ranking as planning-only, but individual campaign candidates had no
machine-readable blockers. In `tools/cathedral_autopilot_autonomous_loop.py`,
operator-authorized le-$5 mode refuses unresolved blockers, so missing blockers
were the difference between "operator must adjudicate" and "candidate is clean
enough for self-authorization."

## Patch

- Added `dispatch_blockers` to `SubstrateRow` and `ParetoRow`.
- Added explicit campaign blockers to Z3/Z4/Z5/C1/C6 rows.
- Propagated blockers into singleton and orthogonal-pair ranked dispatches.
- Added regression coverage ensuring blockers survive into
  `CandidateRow(**as_candidate_row_kwargs())`.

## Evidence

Commands run:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_autopilot_dispatch_ranking.py
.venv/bin/python -m pytest -q \
  src/tac/tests/test_autopilot_dispatch_ranking.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py
.venv/bin/ruff check \
  src/tac/optimization/substrate_composition_matrix.py \
  src/tac/optimization/autopilot_dispatch_ranking.py \
  src/tac/tests/test_autopilot_dispatch_ranking.py
```

Results:

- `27 passed`
- `95 passed`
- `ruff`: all checks passed

## Evidence Boundary

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- This is a dispatch-safety and scientific-rigor hardening patch, not a score
  claim and not a new candidate archive.

## Next Gates

Before any L5/staircase candidate can dispatch as clean:

1. Current operator recipe predeploy must pass.
2. Lane-specific prerequisites must clear, such as Z3 full-main/smoke routing,
   Z4 after Z3 anchor, Z5 after Z4 anchor, C1 probe-v2 architecture lock, and
   C6 full-anchor or explicitly authorized smoke predeploy.
3. Promotion still requires archive/runtime custody plus paired axis-labelled
   CPU/CUDA evidence; the autopilot ranking remains planning-only.
