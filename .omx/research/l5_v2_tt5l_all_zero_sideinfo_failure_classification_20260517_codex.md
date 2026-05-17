# L5 v2 TT5L all-zero side-info failure classification - 2026-05-17

## Scope

- Lane: `lane_time_traveler_l5_autonomy_substrate_20260513`
- Candidate archive:
  `experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/archive.zip`
- Archive SHA-256:
  `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`
- Axis: packet-custody / side-info liveness only; no score claim.

## Finding

The recovered TT5L 25ep packet has an all-zero trained side-info stream:

- shape: `[600, 45]`
- total values: `27000`
- nonzero values: `0`
- nonzero pairs: `0`
- all-zero pairs: `600`

This is a measured archive-level failure classification for that packet, not a
paradigm falsification. The packet cannot close the L5-v2 TT5L side-info gate,
cannot support architecture lock, and cannot be used as trained side-info
evidence in the side-info effect curve.

## Current trainer contract

Current `experiments/train_substrate_time_traveler_l5_autonomy.py` already
contains the required structural fix:

- `_initialize_per_pair_side_info_float(...)` initializes the trainable
  side-info tensor at roughly the int8 LSB scale instead of exact zero.
- `_require_live_quantized_side_info_for_export(...)` refuses all-zero or empty
  quantized side-info before archive export.

Focused regression proof:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_train_time_traveler_full_cpu_mode.py::test_full_trainer_side_info_initializes_at_quantized_lsb \
  src/tac/tests/test_train_time_traveler_full_cpu_mode.py::test_export_side_info_liveness_stats_reject_all_zero_int8 \
  src/tac/tests/test_train_time_traveler_full_cpu_mode.py::test_export_side_info_liveness_stats_record_full_section_coverage \
  src/tac/tests/test_train_time_traveler_full_cpu_mode.py::test_export_side_info_liveness_rejects_empty_int8
```

Result: `4 passed in 0.48s`.

## Packet controls now materialized

`tools/build_tt5l_sideinfo_variant_packets.py` materializes byte-closed TT5L
side-info effect-curve controls from the old 25ep packet:

- `zero`
- `random_lsb`
- `shuffled`
- `trained`
- `ablated`

Durable custody artifact:

- `.omx/research/l5_v2_tt5l_sideinfo_variant_packets_20260517_codex.json`
- `.omx/research/l5_v2_tt5l_sideinfo_variant_packets_20260517_codex.md`

The control manifest correctly fails closed with:

- `tt5l_source_trained_sideinfo_all_zero`
- `requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve`
- `requires_dispatch_lane_claim_before_auth_eval`
- `score_claim_forbidden_until_effect_curve_artifact_passes`

## Classification

`measured-config-regression / stale-artifact-sideinfo-dead`

The old recovered 25ep TT5L archive is unsuitable for side-info usefulness
evidence. The current trainer/export code has a regression guard against
emitting the same all-zero side-info class, but a new current-code timing smoke
or short TT5L training run is still required to produce a nonzero trained
archive before the L5-v2 side-info effect curve can be meaningful.

## Next action

Run a current-code TT5L timing smoke or short training run that emits a
nonzero-side-info archive, then rebuild:

1. `l5_v2_tt5l_sideinfo_variant_packets_*.json`
2. `l5_v2_tt5l_sideinfo_effect_curve_*.json`
3. `l5_v2_lattice_measurement_schedule_*.json`
4. `l5_v2_paired_measurement_dispatch_plan_*.json`

Only after paired CPU/CUDA exact-eval cells exist for all five variants may
the side-info effect curve influence L5-v2 architecture lock.
