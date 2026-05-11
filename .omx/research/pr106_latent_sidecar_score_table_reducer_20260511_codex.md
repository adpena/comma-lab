# PR106 latent sidecar score-table reducer (2026-05-11)

## Summary

`experiments/build_pr106_latent_sidecar.py` now supports a scorer-free
`--search-mode score_table` reducer. The reducer consumes a precomputed CUDA
score table over canonical `(dim_idx, delta_q)` candidates, emits charged
per-pair latent sidecar bytes, and stays fail-closed with `score_claim=false`
and `ready_for_exact_eval_dispatch=false` until an exact CUDA auth eval scores
the built archive.

This converts the lane from heuristic-only smoke scaffolding into a real
compress-time selection surface without loading scorers at inflate time.

The matching producer is now canonicalized as
`experiments/build_pr106_latent_score_table.py`. It mirrors the yshift
score-table pattern: active lane claim before real CUDA work, checkpoint/resume
artifacts, official DistortionNet objective, and dry-run plan mode for $0
operator validation.

Shared score-table plumbing is centralized in `src/tac/sidechannel_score_table.py`
so latent and yshift producers use the same atomic writes, active-claim parser,
finite-prefix checkpoint validation, objective function, and official scorer /
ground-truth loader wiring.

## Scientific guardrails

- Candidate row `0` is the single no-op baseline: `[255, 0]`.
- Non-no-op rows enumerate each latent dimension with nonzero integer deltas.
- The reducer selects a candidate only when its measured table score strictly
  improves over the no-op row for the same pair.
- `--top-k` caps selected corrections by measured improvement magnitude.
- The PR100-vs-PR105 `-0.00218` sidecar gain remains a planning target, not a
  prediction for heuristic or score-table output.
- A score-table manifest can be validated, but even validated score-table output
  remains non-promotable until exact CUDA auth eval.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_latent_sidecar.py`
  passed: 16 tests.
- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_latent_sidecar.py
  src/tac/tests/test_pr106_latent_score_table.py
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py` passed: 29 tests.
- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_latent_score_table.py
  src/tac/tests/test_pr106_yshift_score_table.py
  src/tac/tests/test_pr106_latent_sidecar.py
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py` passed: 39 tests
  after shared-helper canonicalization.
- `.venv/bin/python tools/dispatch_dryrun_pr106_sidechannels.py` passed and now
  checks the latent reducer and latent score-table producer surfaces.

## Next score-lowering step

Run a claimed CUDA latent score-table job, build a charged sidecar from the
table, then exact-CUDA score the emitted archive. The table itself is not score
evidence; it is only the compress-time selector for the charged packet.
