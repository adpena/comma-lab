# Codex findings: DFL1 file-list identity and strict parity

UTC: 2026-05-25T02:21:53Z

## Scope

This pass hardens the renderer-payload DFL1 final-byte materializer lane and
the inverse-action/PacketIR bridge feeding it, so generated campaign runs do not
require hand-entered full-frame file-list identity fields and do not silently
enter exact-readiness follow-up without the DFL1 shell-inflate parity proof.

## Findings

- DFL1 campaign generation previously accepted a file-list path or inline
  entries while still requiring separate expected SHA/count/source fields. That
  created a drift-prone manual surface where the supplied identity could differ
  from the actual file list used by the parity proof.
- Exact-readiness queue construction could record missing DFL1 parity context
  as metadata while still emitting the harvest/dispatch-plan follow-up. That is
  acceptable for exploratory blocked-signal capture, but generated DFL1
  campaign runs should fail closed when the full-frame receiver proof cannot be
  queued.
- Inverse-action/PacketIR DFL1 hints needed to carry renderer-payload parity
  and output-manifest parameters through to final-byte operation contexts so
  grouped search can emit executable, queue-owned DFL1 work instead of
  orphaned ad hoc context.
- Explicit inverse-action compiler hints could be shadowed by stale concrete
  source provenance because source provenance was rehydrated before the newer
  compiler hint. The compiler hint is now treated as the authoritative
  deterministic materialization handoff when both are present.
- Water-bucket selections used `atom_id` as the rehydration key. Duplicate
  cells with the same `atom_id` could bind the selected bucket to the wrong
  compiler/provenance, so duplicate action-cell atom ids now fail closed.
- The inverse compiler's executable target set drifted from the materializer
  registry for `packet_member_zip_header_elide_v1`; that target now uses the
  executable path consistently.
- The materialization bridge now separates PacketIR-lowering readiness from
  executable work readiness, so "queue consumable" cannot be mistaken for a
  proved executable materializer row.
- The follow-up adversarial audit found five additional fail-open edges:
  forged parity-proof booleans were trusted without direct value comparison,
  explicit chain manifests could bypass queue-state provenance when a state
  file was supplied, generated execution could fall back to a shared queue DB,
  DFL1 parity could reuse the source runtime as candidate runtime implicitly,
  and duplicate full-frame file-list entries could be silently deduped.

## Landed integration

- `tools/run_byte_shaving_materializer_campaign.py` now derives the expected
  full-frame file-list SHA-256, entry count, and source label from either the
  supplied file-list file or the inline ordered entries. Explicit expected
  values are still allowed, but mismatch fails before artifact-map generation.
- Generated renderer-payload DFL1 campaign queues now forward
  `--require-renderer-payload-dfl1-parity-followup` to
  `tools/build_byte_shaving_campaign_queue.py`.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py` exposes
  `require_renderer_payload_dfl1_parity_followup`; when enabled, DFL1
  exact-readiness queue construction fails if the parity follow-up is blocked.
- Inverse-action compiler params and final-byte contexts preserve DFL1
  archive/output/parity identity fields, including `output_manifest`.
- Inverse-action compiler hints take precedence over stale source provenance,
  duplicate action-cell `atom_id` values fail closed, and PacketIR bridge
  summaries expose explicit `packetir_lowering_ready_*` and
  `executable_work_ready_*` counters.
- DFL1 full-frame parity verification now directly compares actual
  `file_list_sha256` and `file_list_entry_count` against the expected proof
  values, independent of proof-supplied boolean flags.
- DFL1 final-byte contexts reject duplicate full-frame file-list entries and
  require an explicit candidate runtime for parity-bearing exact-readiness
  flows.
- Materializer campaign execution now defaults to a run-local
  `materializer_execution_queue.sqlite` when `--queue-state` is omitted.
- Harvest with an explicit experiment queue state now rejects explicit
  chain-manifest inputs that have no queue work-id provenance.

## Verification

- `.venv/bin/python -m ruff check ...` on the touched scheduler, optimizer,
  tool, and test files passed.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  passed: 167 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  passed: 130 tests.
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  passed after the follow-up adversarial audit fixes: 239 tests.
- `.venv/bin/python -m py_compile tools/build_byte_shaving_campaign_queue.py tools/run_byte_shaving_materializer_campaign.py`
  passed.
- `.venv/bin/python tools/lane_maturity.py validate` passed: 1299 lanes.

No score claim, rank/kill decision, or promotion authority is created here.
This is local proof-chain automation only; exact contest CPU/CUDA eval remains
required for any score movement.
