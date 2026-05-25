# Codex Findings: Materializer Runner Context Autowire

UTC: 2026-05-25T08:46:20Z
Lane: `codex_materializer_runner_context_autowire_20260525`
Commit status at write time: pending

## Finding

The low-level queue CLI can now generate `byte_shaving_materializer_contexts.v1`
from inline compiler/PacketIR params without a separate artifact map, but the
top-level materializer campaign runner still only requested generated contexts
when an artifact map existed. That left runner-owned campaigns vulnerable to
producing blocked work queues even when the campaign plan already carried enough
inline compiler metadata.

## Landing

`tools/run_byte_shaving_materializer_campaign.py` now requests
`--materializer-contexts-out` whenever no explicit `--materializer-contexts`
file is supplied. If an artifact map exists, it is still passed through as
additional custody hints. If no artifact map exists, the runner lets
`tools/build_byte_shaving_campaign_queue.py` consume inline backlog/compiler
params directly. Context output-root selection is centralized in
`_default_materializer_context_output_root(...)`, preserving storage-preflight
workload-root behavior.

## Proof

Durable proof artifact:
`.omx/research/codex_materializer_runner_inline_contexts_20260525T084604Z/proof_summary.json`

Key assertions:

- `runner_artifact_map_omitted=true`
- `runner_contexts_generated=true`
- `runner_contexts_unblocked=true`
- `runner_work_queue_has_executable_row=true`
- `runner_worker_plans_materializer_step=true`
- `score_authority_false=true`

The proof uses an inline compiler plan and no artifact map. The runner builds
contexts, emits an executable local materializer step in dry-run queue mode, and
keeps all score/promotion/rank/dispatch authority fields false.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_builds_queue_owned_followup_command src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_generates_contexts_without_artifact_map src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_can_auto_generate_contexts_from_artifact_map src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_places_generated_context_outputs_under_workload_root`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`

Latest result: 59 passed.

## Authority Boundary

The runner proof is dry-run queue execution. It proves context generation and
queue-owned executable materializer planning, not a score improvement, not
promotion eligibility, and not exact-auth readiness. Exact eval authority still
requires byte-closed materialization, runtime proof, and contest CPU/CUDA auth
artifacts.

