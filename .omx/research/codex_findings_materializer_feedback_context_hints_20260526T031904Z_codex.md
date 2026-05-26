# Codex Findings: Materializer Feedback Context Hints

UTC: 2026-05-26T03:19:04Z

## Verdict

Empirical materializer feedback was carrying usable archive/runtime context, but
`frontier_rate_attack_feedback` was aggregating it into operation-portfolio rows
without preserving the concrete source archive and parser section manifest. That
made the downstream final-byte context compiler re-open already-known blockers
such as `materializer_context_missing:archive_path` and
`materializer_context_missing:section_manifest`.

This is now wired into the bridge: materializer feedback rows emit bounded
context hints, the operation portfolio preserves the best hint in
`evidence_summary`, and materializer backlog `operation_params` carry those hints
into final-byte contexts and the materializer work queue.

## Authority Boundary

Failed receiver/runtime proofs are preserved as observed metadata only. They do
not populate the active `runtime_consumption_proof` context field unless the row
reports receiver contract or inflate parity success. The generated work queue
may run local materializer commands, but score, promotion, rank/kill, GPU launch,
and exact-dispatch authority remain false.

## Empirical Anchor

Generated artifact:
`.omx/research/frontier_rate_attack_feedback_cycle_20260526T_materializer_context_hints_v3/`

The archive-section entropy-recode row now carries:

- `archive_path`:
  `.omx/research/codex_materializer_runner_inline_contexts_20260525T084604Z/source.zip`
- `section_manifest`:
  `.omx/research/codex_materializer_runner_inline_contexts_20260525T084604Z/sections.json`
- active runtime proof: absent
- observed failed runtime proof: preserved

The generated `archive_section_entropy_recode_v1` materializer work row became
local-executable, emitted a byte-closed candidate archive, and was harvested into
a planning-only optimizer source queue. The candidate saved 66 serialized bytes
on the fixture archive, but receiver contract satisfaction is still false and
the harvest has zero dispatch-ready rows.

## Verification

- `ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `PYTHONPATH=. pytest src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
- `tools/run_frontier_rate_attack_feedback_cycle.py` with explicit materializer feedback source
- Executed generated `archive_section_entropy_recode_v1` local materializer command
- `tools/harvest_materializer_chain_candidates.py` over the generated work queue

## Remaining Work

This closes the archive-section entropy-recode context-loss gap. Remaining
queue-visible gaps are still broader: `byte_range_entropy_recode_v1` needs schema
manifest/beam/runtime context, and archive-section header-elide/reorder still
need real compiler/work-queue adapters before they stop at unsupported context
rows.
