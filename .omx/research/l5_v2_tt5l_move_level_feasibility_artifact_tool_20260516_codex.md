# L5 v2 TT5L Move-Level Feasibility Artifact Tool

Date: 2026-05-16
Author: Codex
Scope: TT5L / L5 v2 staircase cargo-cult unwind

## Summary

TT5L readiness already required a move-level feasibility artifact at
`.omx/state/tt5l_move_level_feasibility.json`, but the repository did not have
a canonical builder for that artifact and the validator did not bind it to an
underlying proof file. That left the next L5 v2 blocker vulnerable to a
hand-written state row masquerading as proof.

This landing adds:

- `tools/build_tt5l_move_level_feasibility_artifact.py`
- `TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH`
- public `tt5l_move_level_feasibility_status(...)`
- hash/path custody checks for both the move-level proof artifact and the
  score-axis Dykstra sanity artifact
- readiness `command_template` for
  `materialize_tt5l_move_level_feasibility_proof`
- tests covering valid build, missing proof refusal, unproven-payload refusal,
  next-action discoverability, and proof-hash mismatch rejection

## Contract

The builder consumes a solver/proof JSON artifact that must itself assert:

- `predicate_passed=true`
- `move_level_constraint_proof=true`
- finite `residual_max`
- positive finite `residual_tolerance`
- `residual_max <= residual_tolerance`
- exact TT5L constraint set IDs

It also requires the existing score-axis sanity artifact
`.omx/state/dykstra_feasibility_time_traveler_l5.json` to exist and records its
SHA-256. The emitted state artifact is valid only while both underlying files
still match the recorded hashes.

The artifact remains planning custody only:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

It unlocks later side-info proof and timing-smoke planning only after the
move-level proof is materialized; it does not claim a contest score or promote
TT5L.

## Operator Command Shape

```bash
.venv/bin/python tools/build_tt5l_move_level_feasibility_artifact.py \
  --proof-artifact experiments/results/time_traveler_l5_v2/tt5l_move_level_solver_proof.json \
  --proof-command-argv-json '<json-array-from-solver-run>' \
  --output-json .omx/state/tt5l_move_level_feasibility.json
```

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  tools/build_tt5l_move_level_feasibility_artifact.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_build_tt5l_move_level_feasibility_artifact.py
```

Result: all checks passed.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_build_tt5l_move_level_feasibility_artifact.py -q
```

Result: 96 passed.

## Next Frontier Action

Produce the actual TT5L move-level solver/proof artifact in
`experiments/results/time_traveler_l5_v2/`, then run the builder above. If the
status passes, the staircase can advance to the full-frame side-info
consumption proof and then the first-anchor timing-smoke custody artifact.
