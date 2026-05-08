# Cathedral Autopilot Lossy-Coarsening Catalog Closure - Worker E1 - 2026-05-08

## Scope

Closed the Worker B catalog/evidence unknown gap for
`lossy_coarsening_analytical` without touching monolithic packet files or
Lightning harvester files.

Inputs reviewed:

- `AGENTS.md`
- `.omx/research/autopilot_evidence_semantics_review_20260508_worker_b.md`
- `tools/cathedral_autopilot.py`
- `src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `reports/lossy_coarsening_exact_cuda_evidence_row_20260508.json`
- `.omx/research/lossy_coarsening_exact_cuda_result_review_20260508_codex.json`

## Patch

- Added `lossy_coarsening_analytical` to the architecture catalog with an
  explicit model spec for per-tensor K-step coarsening of PR101 quantized
  renderer symbols.
- Extended `TechniqueEvidence` parsing to preserve
  `falsification_scope` and `reactivation_criteria` from exact result review
  rows.
- Exact-negative catalog updates now emit:
  - `measured_config_retired_only`
  - `exact_negative_classification`
  - `exact_negative_falsification_scopes`
  - `measured_config_statuses`
  - `reactivation_criteria`
  - `supporting_non_promotable_evidence_n` and sources
- Proxy/MPS/CPU and ambiguous non-promotable evidence remain active-ranking
  blocked. Exact-negative evidence retires only the measured config unless
  `family_falsified` or `method_family_retired` is explicitly true.

## Live Evidence Check

Command:

```bash
.venv/bin/python tools/cathedral_autopilot.py evidence-update \
  --prior-evidence reports/cathedral_autopilot_evidence.jsonl \
  --output /tmp/cathedral_autopilot_evidence_update_e1.json
```

Observed summary:

- `unknown_exact_negative_row_count`: `0`
- `cataloged_exact_negative_techniques`: `["lossy_coarsening_analytical"]`
- `lossy_coarsening_analytical` is present in
  `active_ranking_blocked_techniques`
- Catalog row classification:
  - `measured_config_retired=true`
  - `measured_config_retired_only=true`
  - `exact_negative_classification="measured_config_retired_only"`
  - `family_falsified=false`
  - `method_family_retired=false`

The live feed currently contains duplicate reviewed A-negative rows for the
same measured config, so the updated catalog reports `exact_negative_evidence_n`
as `2`. This is duplicate evidence for the same measured config, not a broader
method or family retirement.

## Verification

- `.venv/bin/python -m py_compile tools/cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `.venv/bin/python -m ruff check tools/cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_cathedral_autopilot.py`
- `.venv/bin/python tools/cathedral_autopilot.py evidence-update --prior-evidence reports/cathedral_autopilot_evidence.jsonl --output /tmp/cathedral_autopilot_evidence_update_e1.json`
