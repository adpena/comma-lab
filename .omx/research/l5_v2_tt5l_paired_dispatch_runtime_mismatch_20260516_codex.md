# L5 v2 TT5L paired dispatch runtime mismatch

- schema: `l5_v2_tt5l_paired_dispatch_runtime_mismatch_v1`
- created_at_utc: `2026-05-16T20:55:05Z`
- technique: `time_traveler_l5_autonomy_tt5l_25ep`
- archive: `experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/archive.zip`
- archive_sha256: `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`
- archive_bytes: `34603`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

The recovered TT5L `contest_cuda` anchor is a reviewed single-axis result, but it is not reusable as the CUDA half of the current L5 v2 paired CPU/CUDA measurement.

The recovered CUDA anchor used runtime tree `ed41e941b624b00412c680c56dc5b9b23db32c70ce1008d03f5bca939917b6cd` and runtime content tree `105fc0834cfb8a54b8f46edb81a030d076369c3062f3066c1800602f9d6035f5`. The current local runtime at `experiments/results/time_traveler_recovered_exact_eval_20260514_codex/runtime` projects to Modal runtime tree `2b0dcb5a148ddef7bf56c833bd46fa5830bdde88929b9fd417b4985bea678a28` on `contest_cuda` and `2b2b9dfdb0f3e59af3511e4502a3a4c0cbe9c1f52405b98eb4dec331db248584` on `contest_cpu`, with shared runtime content tree `630970e9dc78c6e2f8dc2ed8d1e22503ea7d0cab17b4da5615a8a1c5b83ac718`.

`tools/dispatch_modal_paired_auth_eval.py` was run in plan-only mode with `--skip-axis-if-promotable-anchor-exists`. It skipped neither axis because no matching promotable anchor exists for the current archive SHA plus expected runtime tree.

## Decision

Treat this as `existing_cuda_anchor_runtime_mismatch_for_paired_measurement`. Do not splice the old CUDA score into a current-runtime paired L5 v2 packet.

Next valid choices:

1. Reconstruct the exact recovered CUDA runtime from source commit `f6df2418171c359d493a434f7c2b5d6492470209`, then dispatch only the missing `contest_cpu` half if the reconstructed runtime tree/content matches.
2. Rerun both `contest_cpu` and `contest_cuda` with the current committed runtime via the paired dispatcher.
3. Generate TT5L sideinfo control variants only after selecting the runtime policy, so all sideinfo cells share one runtime contract.

## Plan-Only Command

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/archive.zip --submission-dir experiments/results/time_traveler_recovered_exact_eval_20260514_codex/runtime --inflate-sh inflate.sh --label l5_v2_time_traveler_l5_autonomy --expected-archive-sha256 2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a --run-id l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement --pair-group-id pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda --lane-id-base lane_l5_v2_measure_tt5l_autonomy_paired_exact --output-root experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_paired_measurement_dispatch --claim-notes l5_v2_paired_measurement:pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists
```

Append `--execute` only after choosing the runtime policy. The paired dispatcher and Modal wrappers own the lane-claim lifecycle.
