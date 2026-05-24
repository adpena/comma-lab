# Codex Findings: MLX Acquisition Batch Runner Integration

UTC: 2026-05-24T07:51:04Z

## Scope

The grouped `mlx_acquisition_batch.v1` artifact was usable by the standalone
inverse-action CLI, but the queue-owned materializer campaign runner still
could not consume it directly. This pass wires the grouped MLX operation-set
artifact into the campaign control surface.

## Landed

- Added `--mlx-acquisition-batch` to
  `tools/run_byte_shaving_materializer_campaign.py`.
- Counted MLX acquisition batches as high-level action sources for the runner's
  plan-vs-source exclusivity checks.
- Forwarded each batch path into
  `tools/build_inverse_steganalysis_action_functional.py`.
- Added regression coverage proving the runner accepts MLX acquisition batches
  as first-class high-level action sources.

## Authority Boundary

This remains a local planning path only. The runner still builds an inverse
action functional, campaign plan, materializer backlog/work queue, and local
experiment queue. It does not grant score authority, promotion authority,
rank/kill authority, or paid exact-dispatch readiness.

## Remaining Gaps

- Teach `tools/run_byte_shaving_materializer_campaign.py` to optionally build
  `mlx_acquisition_batch.v1` from strict MLX selections inline, instead of
  requiring a prebuilt batch.
- Implement executable materializers for the new family-agnostic receiver
  contracts.
- Feed native MLX training/export manifests into `mlx_acquisition_batch.v1`.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_byte_shaving_campaign.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  src/tac/tests/test_optimizer_candidate_queue.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  src/tac/tests/test_inverse_steganalysis_action_functional_cli.py \
  src/tac/tests/test_local_training_runtime_profile.py \
  src/tac/tests/test_representation_training_probe_integration.py \
  src/tac/tests/test_local_training_execution_queue.py \
  src/tac/tests/test_mlx_execution_queue.py -q
# 169 passed in 3.03s
```
