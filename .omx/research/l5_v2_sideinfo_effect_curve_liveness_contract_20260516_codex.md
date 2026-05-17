# L5 v2 side-info effect-curve liveness contract

Date: 2026-05-16

Scope: TT5L / L5-v2 effect-curve gating.

This is a hardening ledger, not a score claim.

## Change

The L5-v2 TT5L side-info effect curve now requires per-cell side-info
liveness evidence. Every observed cell must carry a checked liveness block with
positive `total_values` and an explicit `nonzero_values` count. The active
side-channel variants (`random_lsb`, `shuffled`, `trained`) must also prove
`nonzero_values > 0`; zero and ablated controls may be all-zero but still must
report the liveness evidence.

The builder accepts liveness evidence from the cell, the evidence row, or nested
provenance fields such as `provenance.per_pair_side_info_liveness`. The
measurement scheduler independently rejects missing, unchecked, empty, or
all-zero active-variant liveness before it can advance to the paired-anchor
packet.

## Rationale

The 25ep paired TT5L result showed that side-info consumption alone is not a
usefulness proof, and the exported archive had an all-zero side-info channel.
The effect curve must therefore be byte-closed to the side-channel reality of
each measured cell. A trained-vs-control curve without liveness evidence is a
score-only artifact detached from the mechanism under test.

## Files

- `src/tac/optimization/l5_v2_sideinfo_effect_curve.py`
- `src/tac/optimization/l5_v2_measurement_schedule.py`
- `src/tac/substrates/time_traveler_l5_autonomy/archive.py`
- `experiments/train_substrate_time_traveler_l5_autonomy.py`
- `src/tac/optimization/l5_staircase_v2.py`
- `src/tac/tests/test_l5_v2_sideinfo_effect_curve.py`
- `src/tac/tests/test_l5_v2_measurement_schedule.py`
- `src/tac/substrates/time_traveler_l5_autonomy/tests/test_time_traveler_archive.py`
- `src/tac/tests/test_l5_staircase_v2.py`

## Verification

Focused tests should include:

- `src/tac/tests/test_l5_v2_sideinfo_effect_curve.py`
- `src/tac/tests/test_l5_v2_measurement_schedule.py`
- `src/tac/substrates/time_traveler_l5_autonomy/tests/test_time_traveler_archive.py`
- `src/tac/tests/test_l5_staircase_v2.py`
- `src/tac/tests/test_train_time_traveler_full_cpu_mode.py`

Expected behavior:

- Missing liveness blocks fail closed.
- Unchecked or empty liveness blocks fail closed.
- `random_lsb`, `shuffled`, and `trained` all-zero liveness fail closed.
- `zero` and `ablated` controls can carry all-zero side-info only when the
  liveness evidence is explicit and checked.

## Next Action

Materialize the paired CPU/CUDA TT5L side-info effect curve with these liveness
blocks embedded in every cell. Do not promote L5-v2 TT5L to a paired anchor from
the existing 25ep packet; that packet remains a negative anchor for the
`training_export_zero_sideinfo_mismatch` class.
