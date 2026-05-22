# Codex Findings: MLX Source Identity Refresh

Date: 2026-05-22T09:44:34Z
Lane: lane_mlx_source_identity_refresh_20260522
Author: Codex

## Verdict

LANDED. Added a reusable source-identity refresh tool for derived
scorer-response datasets. It rehydrates MLX scorer-response row custody fields
from each row's `source_path`, verifies the source payload remains
non-authoritative, and fails closed on mismatched existing identity.

This is not a score claim, rank claim, promotion claim, or dispatch authority.
It is a custody repair/diagnostic step for local MLX spend-triage gating.

## Why This Was Needed

The live 600-row FEC6/decoder-q MLX response dataset was blocked at the
effective MLX spend-triage gate because every row was classified as
production-contract identity-missing. Direct inspection showed the original
window response payloads still contained the missing identity fields, so the
derived dataset had stale/missing row identity rather than a missing source
artifact.

## Code Changes

- `src/tac/optimization/scorer_response_dataset.py`
  - Added `refresh_mlx_scorer_response_source_identity(...)`.
  - The helper fills missing row identity fields from source MLX response JSON:
    archive/raw hashes, inflated-output aggregate hash, window metadata, cache
    array hashes, and component hashes.
  - Existing non-null mismatches become blockers, not silent overwrites.
- `tools/refresh_scorer_response_source_identity.py`
  - CLI wrapper that writes refreshed JSON plus an optional markdown report.
- `src/tac/tests/test_scorer_response_dataset.py`
  - Added restore and mismatch-blocking coverage.

## Live Check

Refresh artifact:
`experiments/results/mlx_source_identity_refresh_20260522T0945Z/mlx_fec6_decoderq_same_axis_600row_baseline_structural_oof_predicted_dataset_identity_refreshed.json`

Refresh result:

- `passed`: `true`
- `mlx_row_count`: `600`
- `refreshed_row_count`: `600`
- `updated_row_count`: `600`
- `changed_field_count`: `1802`
- `blocker_count`: `0`

Rerunning the LL planner against the refreshed dataset changed the production
contract blocker from identity-missing to uncovered rows:

- `effective_status`: `blocked`
- `effective_blockers`: `["mlx_production_contract_gate_not_strict_pass"]`
- `production_contract_status`: `blocked`
- `production_contract_row_count`: `600`
- `production_contract_matched_row_count`: `0`
- `production_contract_unmatched_row_count`: `600`
- sampled blocker class:
  `mlx_production_contract_bundle_row_unmatched:<row_id>`

Interpretation: source identity is now recoverable and no longer the immediate
blocker. The remaining blocker is actual strict production-contract coverage
for the exact FEC6/decoder-q singleton windows, not missing row custody fields.

## Verification

- `.venv/bin/ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py tools/refresh_scorer_response_source_identity.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py`
- `tools/refresh_scorer_response_source_identity.py` on the live 600-row
  dataset, with zero blockers.
- `tools/plan_ll_scorer_response_next.py --require-effective-mlx-spend-triage`
  on the refreshed dataset; expected exit code `2` because strict production
  contracts still do not cover the rows.

## Next Action

Build strict production-contract bundle coverage for the exact refreshed
singleton-window dataset, or run the production-contract gate generation during
future MLX window harvests so row identity and strict contracts are born
together.
