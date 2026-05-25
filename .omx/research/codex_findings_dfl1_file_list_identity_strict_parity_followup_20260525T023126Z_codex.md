# Codex findings: DFL1 strict parity follow-up audit

UTC: 2026-05-25T02:31:26Z

## Scope

This follow-up preserves the adversarial audit signal after the initial DFL1
strict-parity landing. It keeps the earlier memo immutable and records the
additional fail-open edges closed around forged proof identity, candidate
runtime identity, duplicate file-list entries, queue-state custody, and harvest
provenance.

## Findings

- Forged parity-proof booleans could be trusted without direct comparison of
  actual `file_list_sha256` and `file_list_entry_count`.
- Explicit chain manifests could bypass queue-state provenance when an
  experiment queue state file was supplied.
- Generated materializer campaign execution could fall back to a shared queue
  DB instead of a run-local queue state path.
- DFL1 parity-bearing exact-readiness flows could reuse the source runtime as
  candidate runtime implicitly.
- Duplicate full-frame file-list entries could be silently deduped before
  parity proof generation.

## Landed integration

- DFL1 full-frame parity verification now directly compares actual file-list
  SHA/count values against expected proof values independent of proof-supplied
  boolean flags.
- DFL1 final-byte contexts reject duplicate full-frame file-list entries and
  require an explicit candidate runtime for parity-bearing exact-readiness
  flows.
- Materializer campaign execution defaults to a run-local
  `materializer_execution_queue.sqlite` when `--queue-state` is omitted.
- Harvest with an explicit experiment queue state rejects explicit
  chain-manifest inputs that have no queue work-id provenance.

## Verification

- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  passed after the follow-up adversarial audit fixes: 239 tests.

No score claim, rank/kill decision, promotion authority, or dispatch authority
is created here.
