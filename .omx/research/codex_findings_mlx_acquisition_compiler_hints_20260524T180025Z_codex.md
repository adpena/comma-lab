# Codex Findings: MLX Acquisition Compiler Hints

- UTC: 2026-05-24T18:00:25Z
- Lane: `codex_mlx_acquisition_compiler_hints_20260524`
- Scope: bridge strict `[macOS-MLX research-signal]` acquisition rows into family-agnostic inverse-action compiler hints without granting score, promotion, rank/kill, dispatch, or exact-eval authority.

## Findings

The MLX acquisition path already grouped strict spend-triage rows into `mlx_acquisition_batch.v1`, but its `selected_operations[]` were high-level `materialize_scorer_response_candidate` placeholders with target `mlx_scorer_response_candidate_v1`. Those rows are useful evidence, but they are not byte-closed materializer operations. Downstream inverse-action planning therefore could stop at source provenance instead of using the newer `operation_set_compiler` bridge.

The landing makes explicit compiler hints first-class:

- `mlx_effective_spend_triage_selection` preserves explicit compiler/target fields from strict-selected rows.
- `mlx_acquisition_batch` emits `operation_set_compiler` only when rows carry concrete target evidence such as archive section, packet member, or tensor operation hints.
- `inverse_steganalysis_acquisition` preserves that compiler hint on the action atom, action cell, and MLX provenance payload.
- `byte_shaving_campaign` treats MLX scorer-response placeholders as non-materializable source provenance, allowing real compiler hints to lower into PacketIR operation sets.

## Authority Boundary

All new compiler-hint payloads stay planning-only and false-authority. Nested truthy authority fields inside compiler operations are rejected before batch construction. Unsupported or missing concrete targets fail closed into the existing `high_level_operation_compiler_required` path.

Codex hardening added one more boundary: strict MLX spend-triage selection now
rejects truthy authority nested inside compiler-hint passthrough fields before
the selection artifact is written, so invalid authority cannot be temporarily
carried into a later batch builder.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_mlx_effective_spend_triage_selection.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_acquisition_batch.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
- `.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_acquisition_batch.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/optimization/byte_shaving_campaign.py src/tac/optimization/mlx_effective_spend_triage_selection.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py`
- `git diff --check -- src/tac/local_acceleration/mlx_acquisition_batch.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/optimization/byte_shaving_campaign.py src/tac/optimization/mlx_effective_spend_triage_selection.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_mlx_effective_spend_triage_selection.py`

## Remaining Gap

This closes the MLX acquisition -> inverse action -> PacketIR compiler handoff. The next closed-loop gap is runner feedback: `run_byte_shaving_materializer_campaign.py` should emit standalone `queue_performance_summary.json` and response-update/replan placeholder artifacts so the next action-functional build can consume runtime evidence directly.
