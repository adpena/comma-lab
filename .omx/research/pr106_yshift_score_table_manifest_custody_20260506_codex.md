# PR106 Yshift Score-Table Manifest Custody - 2026-05-06

## Context

`experiments/build_pr106_yshift_sidechannel.py --search-mode score_table`
reduces a precomputed CUDA scorer table into charged PR106 yshift sidechannel
bytes. The reducer is scorer-free and non-promotable, but the table provenance
must be exact before its rows influence archive bytes.

## Finding

The reducer validated source archive SHA, table SHA, shape, radius, frame count,
and score step. It did not require the canonical score-table manifest schema,
producer identity, or candidate-grid `.npy` SHA-256. A stale or incompatible
manifest could therefore pass some custody checks while describing a different
candidate-grid contract.

Evidence grade: `empirical` score-table custody hardening, not score evidence.

## Change

- Add deterministic `yshift_candidate_grid_npy_sha256(...)` using the same
  `.npy` serialization contract as the score-table builder.
- Require score-table manifests to declare:
  - `manifest_schema == pr106_yshift_score_table_manifest_v1`
  - `producer == experiments/build_pr106_yshift_score_table.py`
  - `candidate_grid_sha256` matching the canonical grid for the selected radius.
- Add regression tests for schema drift and candidate-grid drift.

## Verification

Focused:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_pr106_yshift_sidechannel.py \
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py -q
```

Full preflight:

```text
.venv/bin/python tools/all_lanes_preflight.py --timings
```

## Promotion Status

This does not dispatch CUDA scoring and does not claim a score. It hardens the
compress-time table-to-sidechannel path before future exact CUDA auth eval.
