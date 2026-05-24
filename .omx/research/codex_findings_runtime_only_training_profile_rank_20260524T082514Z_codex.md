# Codex Findings: Runtime-Only Training Profile Ranking

UTC: 2026-05-24T08:25:14Z

## Landing

- Representation-training and PR95/HNeRV local-training candidate adapters now
  use local runtime-profile timing as a rank score when no quality/training score
  or rankable auth bridge exists.
- The rank field is explicitly suffixed as a cost signal, for example
  `seconds_per_step_cost_signal_not_score` or
  `seconds_per_epoch_cost_signal_not_score`.
- The rows remain proxy/local research signal only; false-authority fields stay
  false and missing quality scores remain blockers.

## Why This Matters

The native PR95/HNeRV MLX timing smokes emit runtime profiles before they can
emit real source-faithful quality scores. Before this adapter change, those
rows could enter candidate queues with no usable deterministic ordering. This
landing lets queue/autopilot surfaces rank local MLX timing observations for
throughput planning while still refusing score, promotion, rank/kill, and exact
dispatch authority.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py`
- `.venv/bin/ruff check src/tac/optimization/representation_training_probe_integration.py src/tac/optimization/pr95_muon_local_training_integration.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py`

Observed result: 17 tests passed and ruff passed.

Additional empirical check against
`experiments/results/pr95_hnerv_mlx_timing_smokes_20260524T081106Z`:

| Stage | Rank score | Rank score field | Exact ready |
|---:|---:|---|---|
| 1 | `0.34745379199739546` | `seconds_per_step_cost_signal_not_score` | `false` |
| 5 | `0.04001095803687349` | `seconds_per_step_cost_signal_not_score` | `false` |
| 8 | `0.04175970901269466` | `seconds_per_step_cost_signal_not_score` | `false` |

## Remaining Gaps

- Runtime-only ranking is for scheduling/acquisition throughput, not quality.
- Any row used for exact-eval spend still needs byte-closed archive export,
  runtime-consumption proof, receiver proof, and exact CPU/CUDA auth eval.
