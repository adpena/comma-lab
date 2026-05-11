# PR106 latent score table adaptive CUDA-OOM backoff

Generated: 2026-05-11T09:39:00Z
Owner: codex

## Summary

The PR106 latent-sidecar score-table producer now has the same deterministic
CUDA-OOM backoff as the y-shift score-table producer. This protects the next
score-lowering branch from the Kaggle P100 memory failure class without changing
the scored objective.

## Implementation

- `src/tac/sidechannel_score_table.py`
  - centralizes CUDA-OOM detection and CUDA retry cleanup.
- `experiments/build_pr106_yshift_score_table.py`
  - now imports the shared helpers rather than carrying script-local copies.
- `experiments/build_pr106_latent_score_table.py`
  - adds adaptive scoring over pair chunks and candidate chunks;
  - halves candidate tile first, then pair tile;
  - records `adaptive_batching` telemetry in `score_table_manifest.json`;
  - clears CUDA allocator state after scored batches.
- `src/tac/tests/test_pr106_latent_score_table.py`
  - verifies adaptive output equals the non-adaptive oracle;
  - forces candidate-tile and pair-tile fallback with a fake decoder.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_pr106_latent_score_table.py src/tac/tests/test_pr106_yshift_score_table.py -q`
  - 20 passed
- `.venv/bin/python -m pytest src/tac/tests/test_deploy_claims_active_row.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_pr106_yshift_deploy_contract.py src/tac/tests/test_pr106_yshift_score_table.py src/tac/tests/test_pr106_latent_score_table.py -q`
  - 41 passed
- `.venv/bin/ruff check --select F821 experiments/build_pr106_latent_score_table.py experiments/build_pr106_yshift_score_table.py src/tac/sidechannel_score_table.py src/tac/tests/test_pr106_latent_score_table.py src/tac/tests/test_pr106_yshift_score_table.py`
  - passed
- `/usr/bin/time -p .venv/bin/python tools/all_lanes_preflight.py`
  - all 29 checks passed
  - `real 2.46`

## Claim discipline

This patch does not create a score claim and does not emit charged score bytes.
It only hardens the CUDA compression-time table builder. The latent sidecar
still needs:

1. claimed CUDA table generation;
2. charged sidecar archive build from the measured table;
3. exact CUDA auth eval of the built archive;
4. exact component recomputation and result review.

## Next dispatch implication

Once current active claims clear, the PR106 latent sidecar lane can run with
provider-constrained memory without manual batch-size relaunches. On Kaggle P100
or other constrained GPUs, start with the existing conservative defaults
(`batch_pairs=2`, `candidate_batch_size=8`) and let the adaptive path shrink
tiles if needed.
