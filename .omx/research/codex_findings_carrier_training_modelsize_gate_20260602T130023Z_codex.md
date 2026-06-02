# Codex Findings: Carrier Training Modelsize Gate

UTC: 2026-06-02T13:00:23Z

## Landed

- Hardened `score_aware_carrier_training_plan.v1` so
  `score_aware_training_ready=true` now requires:
  - complete score-aware training stack;
  - verified G3 adjoint;
  - `receiver_closed_modelsize_budget_selected`;
  - non-null `receiver_closed_selected_archive_bytes`.
- Changed the default route for missing/projected/advisory modelsize rows to
  `run_receiver_closed_modelsize_ladder_before_score_aware_training`.
- Preserved advisory selected byte targets for planning, but blocked them from
  becoming training/replay readiness via
  `receiver_closed_modelsize_budget_ladder_missing`.
- Propagated modelsize planner blockers into carrier-training
  `dispatch_blockers` under the `modelsize_budget:` prefix so final-rate,
  Cathedral, and bit-allocator consumers can see exactly why the row is not
  byte-closed.

## Artifact

- `.omx/research/carrier_training_modelsize_gate_proof_20260602T130023Z.json`
  - SHA-256: `6662da92ada3eb62331c1543948982e34c3b923584a1382c153af02915766006`
  - receiver-closed case:
    - `planner_action`: `run_byte_closed_local_replay_gate_before_exact_auth`
    - `score_aware_training_ready`: `true`
    - receiver-closed selected modelsize archive bytes: `40,000`
  - advisory case:
    - `planner_action`:
      `run_receiver_closed_modelsize_ladder_before_score_aware_training`
    - `score_aware_training_ready`: `false`
    - advisory selected modelsize archive bytes: `40,000`
    - receiver-closed selected modelsize archive bytes: `null`
    - blocker: `receiver_closed_modelsize_budget_ladder_missing`

## Verification

- `PYTHONPATH=src .venv/bin/pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_carrier_training_plan.py -q`
  passed: 7 tests.
- `PYTHONPATH=src .venv/bin/pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_modelsize_budget_plan.py src/tac/substrates/_shared/mlx_score_aware/tests/test_carrier_training_plan.py -q`
  passed: 11 tests.
- `PYTHONPATH=src .venv/bin/pytest src/tac/tests/test_nerv_control_surfaces.py src/tac/tests/test_nerv_top_priority_stack_seam.py src/tac/cathedral_consumers/pareto_carrier_fit_consumer/tests/test_pareto_carrier_fit_consumer.py -q`
  passed: 49 tests.
- `PYTHONPATH=src .venv/bin/ruff check ...` passed on the touched files.

## Verdict

GO for SNeRV/HiNeRV local planning with advisory modelsize rows. NO-GO for
score-aware training readiness, local replay readiness, exact dispatch, rank,
promotion, or PR95 beat claims unless the modelsize/fc_dim ladder is
receiver-closed.

## Next Work

Build the actual receiver-closed modelsize/fc_dim ladder producer for SNeRV and
HiNeRV. It must emit receiver-proofed archive-byte rows that satisfy
`receiver_closed_modelsize_budget_selected` before any downstream consumer can
advance from planning to score-aware replay/training readiness.
