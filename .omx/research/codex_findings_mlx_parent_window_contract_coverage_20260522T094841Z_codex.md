# Codex Findings: MLX Parent-Window Contract Coverage

Date: 2026-05-22T09:48:41Z
Lane: lane_mlx_parent_window_contract_coverage_20260522
Author: Codex

## Verdict

LANDED. The MLX production-contract bundle gate can now cover child singleton
scorer-response rows with a strict parent-window contract, provided the row is
inside the parent pair window and the scorer-input cache identity matches.

This remains non-authoritative. It allows only local MLX exact-eval spend
triage after all MLX gates pass; it does not create score, rank, promotion, or
dispatch authority.

## Why This Was Needed

The current MLX response harvest is organized as many singleton-window rows.
Requiring one strict production contract per singleton row would create
unnecessary ceremony when all rows were harvested from the same full parent
cache and strict validation can be attached to that parent cache. The safe
contract is not "same archive name"; it is:

- same archive SHA-256
- same inflated-output aggregate SHA-256
- same batch shape
- same candidate scorer-input cache array hashes
- same reference scorer-input cache array hashes
- child row pair window contained inside the parent contract pair window

The implementation intentionally does not compare full parent response
component hashes against child singleton response component hashes, because
those are hashes of different-length distortion vectors.

## Code Changes

- `src/tac/optimization/scorer_response_dataset.py`
  - Added parent-window containment matching for strict production-contract
    bundles.
  - Added `parent_window_matched_row_count` and sampled row IDs to the bundle
    summary.
  - Exposed parent-window matched count through the effective MLX spend-triage
    gate and markdown report.
- `src/tac/tests/test_scorer_response_dataset.py`
  - Added positive coverage for parent-window contract covering a child row.
  - Added negative coverage proving cache identity mismatch still blocks.

## Verification

- `.venv/bin/ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`

## Next Action

Build two strict parent-window production contracts for the refreshed 600-row
dataset: one for the FEC6 parent cache and one for the decoder-q parent cache.
If the cache-auth, Torch parity, profile stability, and score-calibration gates
all pass for those parent caches, the effective MLX spend-triage gate should be
able to cover the singleton rows without generating 600 separate contracts.
