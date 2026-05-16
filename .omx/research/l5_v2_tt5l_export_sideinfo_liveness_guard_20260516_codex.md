# L5 v2 TT5L Export Side-Info Liveness Guard - 2026-05-16

## Scope

This note records the producer-side fix that follows the paired TT5L exact-eval
negative preserved in
`.omx/research/l5_v2_tt5l_paired_dispatch_result_review_20260516_codex.md`.

The measured packet was byte-closed and paired across `[contest-CPU]` and
`[contest-CUDA]`, but the parsed TT5L archive carried a `600 x 45` per-pair
side-info section with `0 / 27000` nonzero int8 values. That made the excellent
rate term non-causal: the archive did not contain an active temporal correction
signal.

## Code Guard

Changed file:

- `experiments/train_substrate_time_traveler_l5_autonomy.py`

New guard:

- `_quantized_side_info_liveness_stats(...)` records dtype, shape, total values,
  nonzero values, nonzero fraction, min, and max for the quantized TT5L side
  channel.
- `_require_live_quantized_side_info_for_export(...)` raises before archive
  packing if the export side-info tensor is empty or all-zero.
- The full trainer now runs this assertion immediately after quantizing
  `best_ckpt["per_pair_side_info_float"]` and before `pack_archive(...)`.
- Successful exports record `per_pair_side_info_liveness` in `provenance.json`
  and append a stage marker
  `side_info_liveness_nonzero_<nonzero>_of_<total>`.

## Regression Coverage

Changed test:

- `src/tac/tests/test_train_time_traveler_full_cpu_mode.py`

New coverage:

- all-zero int8 side-info raises `tt5l_side_info_all_zero_export`;
- empty int8 side-info raises `tt5l_side_info_empty_export`;
- live int8 side-info returns deterministic liveness statistics.

## Evidence Boundary

This is a producer-side hardening fix, not a score claim and not a promotion.
It prevents recurrence of the measured TT5L 25ep zero-side-info export failure.
The L5-v2 campaign still requires a rebuilt current TT5L packet with nonzero
side-info, followed by paired `[contest-CPU]` and `[contest-CUDA]` exact eval
and the existing side-info effect-curve gate.

