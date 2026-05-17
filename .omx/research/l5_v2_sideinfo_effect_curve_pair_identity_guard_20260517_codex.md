# L5 v2 TT5L Side-Info Effect-Curve Pair Identity Guard

Date: 2026-05-17
Author: Codex
Scope: TT5L L5 v2 side-info effect-curve custody hardening

## Finding

The probe-axis intake already binds paired CPU/CUDA observations to
`pair_group_id` and `run_id`, but the downstream TT5L side-info effect-curve
validator still paired each variant only by archive SHA and runtime content tree.
That left a false-authority path where cells from different paired runs could be
mixed into one variant row set if their archive/runtime hashes matched.

## Fix

- `build_l5_v2_sideinfo_effect_curve()` now carries `pair_group_id` and `run_id`
  through normalized observed cells.
- `validate_l5_v2_sideinfo_effect_curve()` now fails closed when a variant's
  CPU/CUDA cells are missing or mismatching `pair_group_id` or `run_id`.
- Focused tests cover valid paired identity, missing identity, and mismatched
  identity.

## Operational consequence

The 10-cell TT5L side-info effect-curve run must harvest every CPU/CUDA variant
cell with the same per-variant `pair_group_id` and `run_id`. This blocks stale
or cross-run cell mixing before any L5 v2 architecture-lock authority can flip.

No score claim is made here. This is a custody guard before paid/provider
dispatch.
