# Codex Findings - Family Materializer Sweep Queue

Timestamp: 2026-05-25T10:24:27Z
Agent: Codex
Lane: `codex_family_materializer_empirical_sweep_queue_20260525`

## Finding

Family-agnostic materializers were executable one candidate at a time, and the
empirical sweep runner covered packet header elision, packet recompress, and
tensor factorization. The queue could not yet own a bounded archive sweep as a
normal local work row, and archive-section entropy recode was missing from the
sweep surface. That kept useful rate-positive/receiver-negative signal too
close to leaf scripts.

## Landing

- `tools/run_family_agnostic_materializer_sweep.py` now supports
  `archive_section_entropy_recode_v1` with section manifest, section selection,
  and Brotli quality sweep parameters.
- `byte_shaving_campaign_queue` now recognizes explicit sweep contexts
  (`sweep_archives`, `sweep_archive_specs`, `materializer_sweep_archives`, or
  `materializer_sweep_archive_specs`) and emits
  `tools/run_family_agnostic_materializer_sweep.py` queue work rows.
- Sweep queue rows write a sweep JSON plus observation JSONL, carry typed
  postconditions, and preserve JSONL false-authority checks row-by-row.
- Single-candidate materializer behavior remains the default unless a context
  explicitly declares sweep archive specs.

## Authority Contract

The sweep output is local planning signal only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Rate-positive archive-section recodes with unsatisfied receiver contracts are
kept as repair signal, not promoted or dispatched.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/comma_lab/scheduler/byte_shaving_campaign_queue.py \
  tools/run_family_agnostic_materializer_sweep.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_family_agnostic_materializer_sweep.py
```

Result: passed.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_family_agnostic_materializers.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_family_agnostic_materializer_sweep.py \
  src/tac/tests/test_final_byte_operation_contexts.py \
  -q
```

Result: 241 passed.

## Next Integration

Feed real PR101/PR110/PR95-family archives into queue-owned materializer sweeps
for packet recompress, header elide, archive-section recode, and tensor
factorize. The next planner step should compare these observations as grouped
operation-set priors, not separate per-family anecdotes.
