# Codex Findings: Inverse Scorer Chain Context Parity Smoke

Date: 2026-05-24T16:55:55Z
Agent: Codex
Lane: codex_inverse_scorer_chain_context_smoke_20260524

## Finding

The inverse-scorer artifact-map path could generate one-shot candidate
materializer rows, and the lower-level queue builder could construct chain rows
when handed complete parity context, but the CLI artifact-map path did not yet
prove that generated inverse-scorer contexts could become an executable chain
and clear inflate parity end to end.

## Landed Changes

- Added a regression that starts from `final_byte_artifact_map.fixture.v1`, not
  hand-authored context JSON.
- The test generates `byte_shaving_materializer_contexts.v1`, lowers it into a
  `tools/run_inverse_scorer_cell_candidate_chain.py` work row, and executes the
  generated command.
- The smoke uses a deterministic local inflate runtime to compare the source
  archive and materialized candidate archive output trees.
- The chain clears `candidate_inflate_output_parity_missing` while preserving
  `score_claim=false` and `ready_for_exact_eval_dispatch=false`; exact auth
  remains blocked by `exact_auth_eval_required_before_score_claim`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_byte_shaving_campaign_queue_cli_generates_inverse_scorer_chain_context_and_executes_parity -q`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_inverse_scorer_cell_materializer.py -q`
  passed: 130 tests, one expected duplicate-ZIP warning.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_packet_compiler_sparse_packet_ir.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_inverse_scorer_cell_materializer.py tests/test_pr106_context_recode.py -q`
  passed: 350 tests, one expected duplicate-ZIP warning.
- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/optimization/inverse_scorer_cell_chain.py src/tac/optimization/inverse_scorer_cell_inflate_parity.py`
  passed.
- `git diff --check` passed.
- Current pre-commit rerun also passed:
  `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_packet_compiler_sparse_packet_ir.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_readiness.py tests/test_pr106_context_recode.py -q`
  with 312 tests, plus `tools/review_gate_hook.py` and
  `tools/lane_maturity.py validate`.

## Remaining Work

The next frontier-moving gate is a real local campaign smoke that starts from
MLX/scorer-response sources plus an actual template archive, auto-generates the
inverse-scorer artifact map, materializes a chain candidate, harvests the
chain manifest into exact-readiness rows, and dispatches exact auth only after
the contest-axis blockers clear.
