# Codex Findings: DFL1 Parity DAG No-Orphan Signal

- UTC: 2026-05-25T01:51:13Z
- Scope: `renderer_payload_dfl1_v1` shell-inflate parity in the byte-shaving materializer queue and harvest bridge.
- Authority: local proof-chain and exact-readiness planning only. No score claim, no promotion claim, no rank/kill authority.

## Finding

DFL1 shell parity was safe at exact-readiness time, but the queue could still lose signal in three ways: a custom sidecar parity output directory was not validated against the scheduler workload root, missing parity follow-up context was silent in experiment metadata, and harvest-only runs with DFL1 rows but no sidecar proof did not inventory the missing proof.

## Landing

- Added explicit DFL1 parity follow-up blockers in `build_materializer_execution_queue` metadata.
- Rejected custom DFL1 parity sidecar output paths outside the scheduler workload root when scheduler preflight gates execution.
- Strengthened the parity DAG step postconditions beyond schema/full-frame flag to include file-list claim, byte/SHA/cmp equality, empty proof blockers, and false dispatch authority.
- Made harvest report DFL1 candidates with missing sidecar proofs instead of returning an empty sidecar report.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_execution_queue_builds_dfl1_parity_followup src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_execution_queue_records_missing_dfl1_parity_context src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_execution_queue_rejects_dfl1_parity_output_outside_workload_root src/tac/tests/test_materializer_chain_harvest_scheduler.py::test_harvest_attaches_dfl1_shell_parity_sidecar src/tac/tests/test_materializer_chain_harvest_scheduler.py::test_exact_readiness_bridge_blocks_dfl1_without_full_frame_parity -q`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py -q`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/materializer_chain_harvest.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py tools/harvest_materializer_chain_candidates.py tools/prove_shell_inflate_parity.py`
- `git diff --check`

## Remaining Boundary

The shell parity proof remains a local exact-readiness input, not score authority. Exact CUDA/CPU auth eval and lane claim discipline are still required before dispatch or promotion.
