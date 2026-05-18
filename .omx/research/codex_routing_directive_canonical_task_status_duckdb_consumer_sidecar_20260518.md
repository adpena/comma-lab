# Codex routing directive sidecar: DuckDB consumer view for canonical_task_status
# Date: 2026-05-18
# Sister to `codex_routing_directive_canonical_task_status_single_source_of_truth_20260518.md`
# Per operator: "perhaps this is a duckdb candidate too"

## OPERATOR DIRECTIVE (verbatim 2026-05-18)

> *"perhaps this is a duckdb candidate too"*

YES. Per `tac.canonical_duckdb` package pattern (already canonical): JSONL = source of truth (APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110); DuckDB = READ-MODEL CONSUMER refreshed periodically.

## ITEM 10 — Extend `tac.canonical_duckdb` schema with `canonical_task_status` view

```python
# src/tac/canonical_duckdb/canonical_task_status_ext.py
CREATE_CANONICAL_TASK_STATUS_TABLE = """
CREATE TABLE IF NOT EXISTS canonical_task_status (
    task_id VARCHAR NOT NULL,
    source_design_memo VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    owner VARCHAR NOT NULL,
    predicted_cost_usd DOUBLE,
    predicted_delta_s_lower DOUBLE,
    predicted_delta_s_upper DOUBLE,
    actual_delta_s DOUBLE,
    commit_shas VARCHAR[],
    test_status VARCHAR NOT NULL,
    blockers VARCHAR[],
    started_at_utc TIMESTAMP,
    completed_at_utc TIMESTAMP,
    event_type VARCHAR NOT NULL,
    event_timestamp_utc TIMESTAMP NOT NULL,
    event_actor VARCHAR NOT NULL,
    event_notes VARCHAR,
    session_id VARCHAR,
    written_at_utc TIMESTAMP NOT NULL,
    PRIMARY KEY (task_id, event_timestamp_utc)
);

CREATE VIEW IF NOT EXISTS canonical_task_status_latest AS
SELECT * FROM canonical_task_status t1
WHERE event_timestamp_utc = (
    SELECT MAX(event_timestamp_utc) FROM canonical_task_status t2 WHERE t2.task_id = t1.task_id
);
"""
```

Refresh via `tools/refresh_canonical_duckdb.py --tables canonical_task_status` (canonical pattern; mirrors existing per_byte_sensitivity_ext + sister tables).

## ITEM 11 — Cross-table queries unlock dashboard observability

The DuckDB view enables operator queries the JSONL alone cannot:

```sql
-- "Show all pending Codex tasks blocked on probe outcomes"
SELECT task_id, title, source_design_memo, blockers
FROM canonical_task_status_latest cts
LEFT JOIN probe_outcomes po ON po.related_task_id = cts.task_id
WHERE cts.status = 'pending' AND cts.owner = 'codex'
  AND po.verdict IN ('INDEPENDENT', 'KILL', 'DEFER');

-- "Show all tasks landed in last 24h with their commit shas and actual ΔS"
SELECT task_id, title, commit_shas, actual_delta_s
FROM canonical_task_status_latest
WHERE status = 'completed' AND completed_at_utc > NOW() - INTERVAL 24 HOUR;

-- "Cross-reference pending tasks with substrate council verdicts"
SELECT cts.task_id, cts.title, cdp.council_verdict, cdp.deferred_substrate_id
FROM canonical_task_status_latest cts
JOIN council_deliberation_posterior cdp ON cdp.deferred_substrate_id = cts.task_id
WHERE cts.status = 'pending';

-- "Aggregate Codex's spend over time vs frontier movement"
SELECT date_trunc('day', cts.completed_at_utc) AS day,
       SUM(cts.predicted_cost_usd) AS total_predicted_spend,
       SUM(mcil.cost_actual_usd) AS total_actual_spend,
       AVG(cts.actual_delta_s) AS avg_actual_delta_s
FROM canonical_task_status_latest cts
LEFT JOIN modal_call_id_ledger_latest mcil ON mcil.lane_id = cts.task_id
WHERE cts.status = 'completed'
GROUP BY day ORDER BY day DESC;
```

These queries become operator's "dashboard" — observability per harness engineering principles.

## ITEM 12 — HF dataset push (per existing `tac.canonical_duckdb.hf_push` pattern)

```python
# src/tac/canonical_duckdb/hf_push.py extension
def push_canonical_task_status_to_hf(
    duckdb_path: pathlib.Path,
    hf_dataset_id: str = "adpena/pact-canonical-task-status",
    *,
    private: bool = True,  # per Public Disclosure Hygiene non-negotiable
) -> None: ...
```

Per Public Disclosure Hygiene: DEFAULT private (operator can flip to public if desired). Push refreshed daily via cron OR per-Codex-session-end.

## INTEGRATION WITH CATALOG #523

Catalog #523 (`tac.canonical_duckdb`) already has this pattern landed. Extension is ADDITIVE (new table + view + refresh hook + HF push). No conflicts with existing tables (per_byte_sensitivity / continual_learning_posterior / council_deliberation_posterior / probe_outcomes / cost_band_posterior / lane_registry / modal_call_id_ledger / subagent_progress).

## DISCIPLINE

Same as parent directive (Catalog #110/#117/#131/#138/#157/#174/#206/#229/#245/#287/#305/#314/#325/#523).

— Main-Claude (relayed on behalf of operator 2026-05-18)
