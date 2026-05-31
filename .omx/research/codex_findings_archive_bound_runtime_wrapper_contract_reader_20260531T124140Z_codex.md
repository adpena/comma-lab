# Codex Findings: Archive-Bound Runtime Wrapper Contract Reader

Date: 2026-05-31T12:41:40Z
Agent: Codex
Scope: adversarial review of live SegNet boundary repair materializer custody, exact-gating inputs, and shared archive-bound contract consumption.

## Finding

The live boundary repair materializer correctly emitted receiver-proven archive-bound candidates, but the shared contract reader only treated direct `archive_bound_candidate_contract` / `archive_bound_candidate_contract_surface` / adapter-package schemas as first-class custody surfaces. Runtime wrappers and materializer manifests carry the shared package under `archive_bound_candidate_adapter_package`, so consumers that call `selected_archive_bound_candidate_contract_from_payload(...)` on the whole wrapper or manifest could fail to see the canonical contract and fall back to duplicate readiness/archive fields.

That was a no-signal-loss risk for exact-ready bridge, stack search, and acquisition routing: the audit could recurse and pass, while direct consumers still missed the wrapper-level custody surface.

## Patch

Patched `src/tac/optimization/archive_bound_candidate_contract.py` so:

- `has_archive_bound_candidate_contract_payload(...)` recognizes `archive_bound_candidate_adapter_package`.
- `archive_bound_candidate_contracts_from_payload(...)` unwraps nested adapter packages.
- wrapper or manifest duplicate archive/readiness fields are compared against the selected nested contract and rejected as stale if they disagree.

## Evidence

Focused checks passed:

- `.venv/bin/python -m ruff check src/tac/optimization/archive_bound_candidate_contract.py src/tac/tests/test_archive_bound_candidate_adapter_spine.py src/tac/tests/test_boundary_repair_runtime_materializer.py`
- `.venv/bin/python -m pytest src/tac/tests/test_archive_bound_candidate_adapter_spine.py src/tac/tests/test_boundary_repair_runtime_materializer.py -q` -> 12 passed
- `tools/audit_archive_bound_candidate_contracts.py` over the two live boundary materializer manifests -> 2/2 valid, 0 blocking, 0 migration-required, 0 advisory

Live manifest contract extraction now returns exact-handoff-ready archive custody for:

- `source_pixel_patch`: archive sha `f2f75d548d7d9a36e144d9ba61edf73547c67a6acf907d9386581bc2119379b7`, 267855 bytes, receiver proof `.omx/research/live_best_bridge_b7106_20260531Tlocal/materialized_boundary_repair_k512/receiver_proof/boundary_repair_receiver_proof.json`
- `masked_local_median`: archive sha `26093bed66f546871677d3b75d6c76ae49ea7a4552e484af7ee79b9b5defc4b3`, 378597 bytes, receiver proof `.omx/research/live_best_bridge_b7106_20260531Tlocal/materialized_boundary_postfilter_k2048/receiver_proof/boundary_repair_receiver_proof.json`

## Remaining Blocker

These candidates are still not score authority. The next gate is exact CPU/CUDA preclaim/dispatch with axis-specific result harvest. MLX and receiver proof can route acquisition and exact-handoff planning only; promotion remains blocked until exact-axis evidence lands.
