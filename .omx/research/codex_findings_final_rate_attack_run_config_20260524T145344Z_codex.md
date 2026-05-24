# Codex Findings - Final Rate Attack Run Config

UTC: 2026-05-24T14:53:44Z

## Scope

This pass followed the operator direction to focus on inverse-steganalysis,
automated final-rate attack work, and PR95/HNeRV MLX substrate training. The
immediate gap was not missing math primitives: the repo already had inverse
action functionals, inverse-action-to-byte-shaving conversion, materializer
queue compilation, MLX acquisition batches, and PR95 MLX matrix queues. The
operational gap was that the final-rate attack runner still required long CLI
invocations instead of a durable file that can be updated, queued, inspected,
and rerun.

## Landed Fix

- `tools/run_byte_shaving_materializer_campaign.py` now accepts
  `--run-config <json>` with schema
  `byte_shaving_materializer_campaign_run_config.v1`.
- The config can define inverse-steganalysis sources, byte-shaving sources,
  MLX effective-spend triage selections, acquisition batching, queue ids,
  byte budgets, materializer contexts, storage/preflight cleanup, staircase/SSH
  controls, and execution bounds.
- CLI scalar flags override config fields; repeatable source flags append to
  configured lists. The emitted run summary records the config path, bytes, and
  SHA-256 so queue runs have custody over the file that defined them.
- Unknown config keys fail closed before any queue or worker is built.

## Authority Boundary

The run config is configuration only. It is not queue authority, score
authority, dispatch authority, promotion authority, or rank/kill authority.
`experiment_queue.v1` remains the queue state authority, and exact CPU/CUDA auth
eval remains the score authority.

## Verification

- `ruff check tools/run_byte_shaving_materializer_campaign.py
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py
  src/tac/tests/test_inverse_steganalysis_action_functional_cli.py
  src/tac/tests/test_byte_shaving_campaign.py -q`
- Result: `47 passed`

## Next Integration

The next bridge should add a checked-in or generated example config for the
current live MLX learned-sweep selection plus materializer contexts, then run it
plan-only and execute bounded local proof-chain rows. In parallel, PR95/HNeRV
MLX matrix queues should emit the same run-config-compatible handoff metadata
so local substrate timing and optimizer evidence flows into the inverse
action/rate-attack loop without bespoke command assembly.
