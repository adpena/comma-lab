# Codex Findings: DQS1 Storage And Exact-Dispatch Hardening

UTC: 2026-05-24T05:46:06Z
Role: Codex adversarial reviewer and hardening lane
Scope: DQS1 local-first queue, queue-owned storage preflight, artifact-retention git boundary, and paid exact-dispatch authority gates.

## Findings

1. `dqs1_local_first` scheduler preflight reused the action-summary date for storage-plan and cleanup-plan output paths. When a 2026-05-24 queue was built from a 2026-05-23 summary, it could overwrite historical `.omx/research/dqs1_local_first_*_20260523.json` artifacts. Fix: preserve the old lane date for candidate identity but use the current `eureka_run_id` as the preflight artifact id when available.
2. Existing storage-plan overwrites were allowed without a provenance guard. Fix: scheduler preflight now hashes an existing target plan and passes `--expected-output-sha256`; `tools/plan_experiment_storage.py` delegates that guard to the canonical artifact writer.
3. New proactive artifact-retention JSON/journal files were not ignored. Fix: `.gitignore` now treats `*_artifact_retention_*.json` and journals as local queue custody, matching storage and cleanup plan policy.
4. Paid exact-dispatch consumers accepted exact-ready rows without an explicit contest score axis. Fix: `exact_dispatch_authority` supports a required axis, and the materializer dispatch consumer, materializer dispatch plan, and `parallel_dispatch_top_k.py` require `contest_cuda`.
5. `contest_mode=true` was too broad and could be misread as `contest_exact_eval`. Fix: exact-dispatch authority now requires explicit `target_modes=["contest_exact_eval"]`.
6. Truthy pre-result authority fields in candidate payloads could leak proxy authority into pre-dispatch rows. Fix: exact-dispatch authority now blocks truthy forbidden authority fields except the two allowed pre-dispatch facts: `ready_for_exact_eval_dispatch` and `dispatch_packet_ready`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_scheduler_storage_preflight.py src/tac/tests/test_exact_dispatch_authority.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py src/tac/tests/test_materializer_exact_eval_consumer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_dispatch_command_builder_shapes.py tests/test_comma_lab_research_state.py -q` -> 148 passed.
- `.venv/bin/python -m ruff check ...` on touched Python files -> passed.
- `.venv/bin/python -m py_compile ...` on touched scheduler/optimizer/tool modules -> passed.
- `git check-ignore -v` confirms generated `dqs1_proactive_artifact_retention_*.json` and journal files are ignored.

## Authority

No score claim, promotion claim, rank/kill decision, or exact-dispatch authorization is made by this memo. This hardening reduces false authority and signal-loss risk before the next local-first DQS1/materializer tranche.
