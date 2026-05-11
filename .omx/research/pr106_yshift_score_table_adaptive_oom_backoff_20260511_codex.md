# PR106 y-shift score table adaptive CUDA-OOM backoff

Generated: 2026-05-11T09:23:50Z
Owner: codex

## Summary

The PR106 y-shift score-table producer now retries CUDA OOMs by reducing only
execution tiling, not the scorer objective. The scored rows remain the same
pair-marginal objective:

`100 * seg_dist + sqrt(10 * pose_dist)`

The adaptive path halves `candidate_batch_size` first, then `pair_chunk_size`,
and fails closed if a single pair and single candidate still OOMs.

## Motivation

Kaggle v5 reached the official scorer on a Tesla P100 after the PyTorch P100
fallback, then failed with CUDA OOM at `batch_pairs=8` and
`candidate_batch_size=32`. Kaggle v6 is running with smaller static settings,
but future score-table jobs should not rely on manual relaunches to discover a
safe tile size.

## Implementation

- `experiments/build_pr106_yshift_score_table.py`
  - added `score_pair_batch_candidate_table_adaptive`;
  - added CUDA-OOM detection and retry cache cleanup;
  - records adaptive retry telemetry in `score_table_manifest.json`;
  - clears CUDA cache after each scored DALI batch.
- `src/tac/tests/test_pr106_yshift_score_table.py`
  - compares adaptive output against the non-adaptive oracle;
  - forces candidate-tile backoff;
  - forces pair-tile backoff after candidate-tile reaches one;
  - checks that non-OOM errors are not swallowed.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_pr106_yshift_score_table.py -q`
  - 13 passed
- `.venv/bin/python -m pytest src/tac/tests/test_deploy_claims_active_row.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_pr106_yshift_deploy_contract.py src/tac/tests/test_pr106_yshift_score_table.py src/tac/tests/test_pr106_latent_score_table.py -q`
  - 39 passed
- `.venv/bin/ruff check --select F821 experiments/build_pr106_yshift_score_table.py src/tac/tests/test_pr106_yshift_score_table.py`
  - passed
- `/usr/bin/time -p .venv/bin/python tools/all_lanes_preflight.py`
  - all 29 checks passed
  - `real 2.45`

## Claim discipline

This patch does not create a score claim. It only hardens the compression-time
CUDA scorer-table profiler. Promotion still requires:

1. a completed `score_table.npy` with manifest and lane claim;
2. a charged archive built from the table;
3. exact CUDA auth eval on the built archive;
4. component recomputation and custody review.

## Active dispatch state

As of this ledger, Kaggle v6 is still running under active lane claim
`lane_pr106_yshift_score_table` / `kaggle_pr106_yshift_score_table_v6`. If v6
fails with CUDA OOM, the next relaunch should use this adaptive code rather than
another static batch-size-only patch.
