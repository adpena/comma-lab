# Codex Findings: ITEM 8 Multi-Granularity Sensitivity Tensor

Date: 2026-05-19
Actor: codex
Task: `codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_8`

## Verdict

`multi_granularity_sensitivity` is landed as a bounded, planning-only DuckDB read-model extension for `(pair_id, byte_offset, class_id, axis)` sensitivity cells.

It is intentionally not score authority, promotion authority, or dispatch authority. Normal consumers receive adjacent provenance fields (`evidence_grade`, `source_measurement_axis`, `source_anchor_authoritative`, `score_claim`, `promotion_eligible`, `ready_for_exact_eval_dispatch`, `blocker_reason`) with every queried cell.

## Authority Decisions

- Source tensor must be `per_pair_per_byte_v1`; aggregate `aggregate_per_byte_v1` anchors are skipped.
- Class-specific rows require an explicit class-source sidecar. Missing class attribution records a blocker run and emits zero fake per-class cells.
- Dense materialization is bounded by byte offsets or top-K bytes plus `max_rows`; no default all-archive dense refresh.
- Historical runs are preserved by including `run_id` in the cell primary key. The query helper defaults to the latest unblocked run while allowing `include_all_runs=True`.
- `source_anchor_authoritative=True` is rejected unless the run has contest-axis evidence-grade shape; caller-controlled advisory authority is blocked.

## Operator Surface

`tools/refresh_canonical_duckdb.py` now accepts:

```bash
.venv/bin/python tools/refresh_canonical_duckdb.py \
  --repo-root . \
  --tables multi_granularity_sensitivity \
  --multi-granularity-class-source .omx/state/<class_source>.json \
  --multi-granularity-byte-offsets 0,1,2 \
  --multi-granularity-max-rows 1000000
```

The no-option path only bootstraps/counts the table, preserving normal `--tables all` safety.

## Verification

- `.venv/bin/ruff check src/tac/canonical_duckdb/per_byte_sensitivity_ext.py src/tac/canonical_duckdb/tests/test_per_byte_sensitivity_ext.py tools/refresh_canonical_duckdb.py src/tac/canonical_duckdb/query.py`
- `.venv/bin/python -m pytest src/tac/canonical_duckdb/tests/test_per_byte_sensitivity_ext.py src/tac/tests/test_canonical_duckdb_read_model.py -q`
- temp-repo bounded CLI materialization: 60 rows for `2 byte_offsets x 2 pairs x 5 classes x 3 axes`

## Deferred Consumer Hook

The cathedral autopilot is the canonical consumer, but `tools/cathedral_autopilot_autonomous_loop.py` had active partner WIP in this turn. Per the operator's no-trampling rule, this landing does not edit that file. The intended next hook is a planning-only read of `query_multi_granularity_sensitivity_top_cells(...)` after existing Venn/per-pair difficulty hooks; missing or malformed tensor data must be passthrough and must never grant score or dispatch authority.
