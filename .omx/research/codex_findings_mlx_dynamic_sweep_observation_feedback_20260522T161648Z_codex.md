# Codex Findings: MLX Dynamic Sweep Observation Feedback

Date: 2026-05-22T16:16:48Z

## Verdict

PROCEED. The MLX dynamic learned sweep planner now consumes the append-only
`mlx_dynamic_sweep_observation.v1` ledger and suppresses repeat spend rows
without granting score, rank, promotion, or dispatch authority.

## Change

- Added observation feedback to `tac.optimization.mlx_dynamic_learned_sweep`.
- Added `--observation-jsonl` to `tools/plan_mlx_dynamic_learned_sweep.py`.
- Exact contest-axis observations suppress the same
  `candidate_id + sweep_config_id + family` across all planner passes.
- Local/advisory observations suppress only the exact
  `candidate_id + sweep_config_id + optimization_pass_id + family` tuple.
- Planner output records `observation_feedback`,
  `suppressed_observed_sweep_rows`, and summary counts while preserving all
  false-authority fields as explicit `false`.

## Empirical Anchor

Regenerated ignored planning artifacts:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_gap_uleb_dynamic_learned_sweep.json`
- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_gap_uleb_dynamic_learned_sweep.md`

Input observation ledger:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl`

Observed feedback result:

- `observation_row_count = 1`
- `suppressed_observed_row_count = 4`
- Suppressed rows: `prefix_k028 / contest_cpu_exact_candidate / {smoke,micro,intermediate,macro}`
- Suppression reason: `exact_axis_observed_candidate_config_family`
- Latest observed score in feedback summary: `0.1920292853802774 [contest-CPU]`

This is planning hygiene only. It is not a score claim and not a promotion
claim.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_mlx_dynamic_learned_sweep.py`
- `.venv/bin/python -m ruff check src/tac/optimization/mlx_dynamic_learned_sweep.py tools/plan_mlx_dynamic_learned_sweep.py src/tac/tests/test_mlx_dynamic_learned_sweep.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_scorer_response_dataset.py -q`

## Residual Risk

The feedback suppresses repeat candidate/config planning spend. It does not yet
change candidate predictions, fit a posterior, or auto-promote exact results.
Those should remain separate score-authority paths.
