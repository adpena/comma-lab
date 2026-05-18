# Codex Findings: Canonical Task Status HF Push

Date: 2026-05-18T23:54:54Z
Agent: codex
Task: `codex_routing_directive_canonical_task_status_duckdb_consumer_sidecar_20260518::ITEM_12`

## Result

Implemented and fired the private Hugging Face Dataset publication path for the
`canonical_task_status` DuckDB read model.

Remote dataset:
`https://huggingface.co/datasets/adpena/pact-canonical-task-status`

Remote verification:

- `repo_info.private == True`
- Files present: `.gitattributes`, `README.md`,
  `data/canonical_task_status.parquet`,
  `data/canonical_task_status_latest.parquet`,
  `metadata/canonical_task_status_hf_manifest.json`
- Remote manifest `status=pushed`, `private_only=True`, `source_rows=77`
- Remote manifest has no `path` keys and no `/Users/adpena` local path leakage
- README states DuckDB/parquet are read models and the source of truth remains
  `.omx/state/canonical_task_status.jsonl`

## Authority

The JSONL ledger remains authoritative. The HF dataset is an observability
read model only.

Uploaded manifest source watermark:

- `source_ledger_path`: `.omx/state/canonical_task_status.jsonl`
- `source_ledger_rows`: `77`
- `source_ledger_latest_event_timestamp_utc`: `2026-05-18T23:45:06.699787Z`
- `source_ledger_sha256`:
  `64e195794d38b8444af44e6e927fd596d938d70632628dc5d5b2d5901b462707`

## Hardening

Adversarial review found four defects before this landing was finalized:

1. Existing HF repos might not be forced private by `create_repo(..., exist_ok=True)`.
2. Direct push from a stale DuckDB file could create false source-of-truth authority.
3. A raw `--hf-public` toggle was too cheap for operator-state data.
4. Uploaded manifests could leak local filesystem paths.

The patch now:

- forces and verifies `private=True` via `update_repo_settings` plus `repo_info`
  before any upload;
- verifies private visibility again after data upload;
- refreshes `canonical_task_status` from the JSONL source when `repo_root` is
  provided and refuses stale DuckDB row/watermark mismatches;
- makes raw `canonical_task_status` HF export private-only;
- writes an uploaded manifest without local path keys;
- uploads a dataset card that labels the dataset as private operator custody.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_canonical_duckdb_read_model.py src/tac/tests/test_canonical_task_status.py`
  - 17 passed
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check src/tac/canonical_duckdb/hf_push.py src/tac/canonical_duckdb/__init__.py tools/refresh_canonical_duckdb.py src/tac/tests/test_canonical_duckdb_read_model.py`
  - passed
- Real dry-run CLI:
  - `tools/refresh_canonical_duckdb.py --tables canonical_task_status --push-hf-canonical-task-status`
  - `remote_push_fired=false`, `private=true`, source rows 77
- Real private upload CLI:
  - `tools/refresh_canonical_duckdb.py --tables canonical_task_status --push-hf-canonical-task-status --operator-approved --fire-hf-push`
  - `status=pushed`, `remote_push_fired=true`, `private=true`
- Remote HF verification script confirmed privacy, file list, README, and
  sanitized manifest.

## Residual Risk

No score, rank, promotion, dispatch, or contest authority is created by this
landing. The dataset is private read-model infrastructure for operator
observability only.
