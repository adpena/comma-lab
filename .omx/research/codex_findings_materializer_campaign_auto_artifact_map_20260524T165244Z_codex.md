# Codex Findings: Materializer Campaign Auto Artifact Map

Date: 2026-05-24T16:52:44Z
Agent: Codex
Lane: codex_materializer_campaign_auto_artifact_map_20260524

## Finding

The final-byte runner could already build an inverse-steganalysis action
functional from high-level scorer/MLX sources and then build a campaign plan,
but it still required a hand-authored `--materializer-artifact-map` before the
inverse-scorer materializer context compiler could unblock byte-closed candidate
rows. That left the path one manual file away from the queue-owned automation
the final-rate attack needs.

## Landed Changes

- `tools/run_byte_shaving_materializer_campaign.py` can now auto-generate
  `materializer_artifact_map.json` for inverse-scorer candidate materialization.
- The generated artifact map can use either an explicit
  `--inverse-scorer-action-functional` with an existing plan or the action
  functional generated from high-level scorer/MLX inputs in the same run.
- The runner forwards the generated artifact map into
  `tools/build_byte_shaving_campaign_queue.py`, which then generates
  `byte_shaving_materializer_contexts.v1` and queue-owned executable work rows.
- Conflict handling fails closed when an operator provides explicit
  `--materializer-contexts` / `--materializer-artifact-map` together with the
  new auto artifact-map flags.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  passed: 25 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py -q`
  passed: 92 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_packet_compiler_sparse_packet_ir.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_readiness.py tests/test_pr106_context_recode.py -q`
  passed: 312 tests.
- `git diff --check`, `tools/review_gate_hook.py`, and
  `tools/lane_maturity.py validate` passed; lane maturity validated 1252 lanes.

## Remaining Work

The next automation gate is a bounded no-paid-dispatch campaign smoke starting
from a real MLX/scorer-response source and a real template archive, using the
new auto artifact map to produce local materializer work without hand-written
context JSON. The follow-up should harvest chain outputs into exact-readiness
bridge rows and only then consider exact auth dispatch.
