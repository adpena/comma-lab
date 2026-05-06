# PR106 Yshift Score-Table NPY Pickle Hardening - 2026-05-06

## Context

The PR106 yshift score-table producer writes numeric `.npy` artifacts that are
later consumed by a scorer-free sidechannel reducer. These artifacts are not
score evidence by themselves, but they directly influence charged archive
bytes.

## Finding

Several score-table write/load paths used NumPy defaults for `.npy` pickle
handling. The arrays are numeric and do not need pickle support. Leaving the
default in place weakens cross-platform, deterministic, and fail-closed artifact
handling.

Evidence grade: `empirical` artifact-hardening, not score evidence.

## Change

- Save score tables and candidate grids with `allow_pickle=False`.
- Load completed/checkpoint score tables with `allow_pickle=False`, including
  mmap reuse.
- Update tests to use the same no-pickle contract for fixture artifacts.

## Verification

Focused:

```text
.venv/bin/python -m pytest src/tac/tests/test_pr106_yshift_score_table.py -q
```

Full preflight:

```text
.venv/bin/python tools/all_lanes_preflight.py --timings
```

## Promotion Status

This does not dispatch CUDA work and does not claim a score. It hardens the
local and remote score-table artifact contract before future exact eval.
