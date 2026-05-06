# PR106 Y-Shift Score-Table Manifest Gate - 2026-05-06

This tranche hardens the PR106 y-shift sidechannel path before real CUDA score
work. The builder already supported `--search-mode score_table`, but a provided
CUDA score-table manifest was only recorded by SHA. It is now validated before
the table can be reduced into charged SC01 bytes.

## Guard

`experiments/build_pr106_yshift_sidechannel.py` now validates:

- score-table manifest is JSON object
- `score_claim=false`
- `ready_for_builder=true`
- `ready_for_exact_eval_dispatch` is not true
- `dispatch_attempted` and `remote_jobs_dispatched` are not true
- `source_archive_sha256` matches `--pr106-archive`
- `score_table_npy_sha256` matches `--score-table-npy`
- `candidate_radius`, `candidate_count`, `n_frames`, `score_table_shape`, and
  `score_step` match the builder invocation

If any field drifts, the builder raises before writing a score-table-derived
candidate archive.

## Evidence

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pr106_yshift_sidechannel.py \
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py
```

Result: `39 passed`.

```bash
.venv/bin/python tools/all_lanes_preflight.py --timings
```

Result: `ALL 23 PREFLIGHT CHECKS PASSED`.

## Status

- `score_claim=false`
- `dispatch_attempted=false`
- `ready_for_exact_eval_dispatch=false`
- no CUDA work dispatched

This does not lower score by itself. It removes a high-risk custody gap on the
current PR106 sidechannel score path so the next real CUDA table can be reduced
deterministically and audited before exact auth eval.
