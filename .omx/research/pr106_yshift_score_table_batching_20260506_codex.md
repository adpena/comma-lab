# PR106 Yshift Score-Table Batching

Date: 2026-05-06
Agent: Codex
Evidence grade: empirical

## Finding

`experiments/build_pr106_yshift_score_table.py` already implements the real
CUDA score-table path, but the scoring loop evaluated one frame-pair at a time
inside each DALI batch. That leaves GPU throughput on the table and slows the
highest-EV PR106 yshift route from CUDA profile artifact to charged archive.

## Change

Added `score_pair_batch_candidate_table()`, which scores all pairs in the
current scoring batch for each candidate chunk while preserving the exact
existing objective:

```text
100 * seg_dist + sqrt(10 * pose_dist)
```

The emitted table schema, manifest fields, score-claim flags, checkpoint
semantics, and exact-eval blockers are unchanged.

## Guard

`src/tac/tests/test_pr106_yshift_score_table.py::test_batched_candidate_table_matches_pairwise_reference`
compares the batched helper against the existing pairwise helper using a fake
`compute_distortion()` implementation and asserts exact table-shape/score
equivalence.

## Dispatch Status

No exact eval or remote GPU dispatch was attempted. This is throughput work for
the CUDA profile generator only; score claims still require a charged yshift
archive and canonical CUDA auth eval on the archive bytes.
