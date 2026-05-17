# L5 v2 TT5L Materialized Work Unit Num-Pairs Guard - 2026-05-17

## Purpose

Prevent a non-contest TT5L advisory archive from becoming a materialized paired
CPU/CUDA dispatch work unit merely because it has nonzero side-info bytes.

## Finding

`_tt5l_materialized_paired_work_unit_status()` already checked archive
existence, SHA/byte custody, runtime paths, lane ids, paired Modal command
shape, and TT5L side-info liveness. It recorded `tt5l_sideinfo_stats.num_pairs`
but did not require the materialized archive to carry the full contest pair
count.

That left a false-authority path: a tiny advisory archive such as the current
two-pair nonzero side-info smoke could satisfy the nonzero-side-info predicate
and look dispatch-reviewable even though it cannot represent the full
`upstream/videos/0.mkv` contest workload.

## Change

Added a materialized-work-unit blocker when parsed TT5L `num_pairs` is not the
full contest count:

`l5_v2_tt5l_materialized_paired_work_unit_tt5l_num_pairs_not_full_contest`

The positive materialized-work-unit fixture now writes a 600-pair TT5L archive.
The new negative regression writes a two-pair archive with nonzero side-info and
proves it still fails closed.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_probe_action_advances_after_work_unit_materialized \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_all_zero_sideinfo \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_noncontest_pair_count \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_weak_axis_commands \
  -p no:cacheprovider
```

Result: `4 passed`.

## Current Frontier Impact

No score claim and no dispatch. This is a dispatch-custody hardening step for
the L5 v2 staircase: advisory nonzero side-info is useful signal, but only
full-contest, byte-closed TT5L archives can become paired CPU/CUDA work units.
