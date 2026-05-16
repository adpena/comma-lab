# L5 v2 TT5L Paired Dispatch Result Review - 2026-05-16

## Scope

This ledger preserves the durable signal from the L5-v2 materialized TT5L paired
exact-eval dispatch. Raw Modal logs and provider work directories remain in
ignored experiment-result locations; this tracked ledger is the no-signal-loss
summary for future operators and agents.

Parent materialized work unit:
`.omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.md`

Commit with materialized paired work unit:
`a31d2e3b2e94593bb80c84ac6e01943bcbd2641c`

## Candidate Custody

| field | value |
|---|---|
| archive path | `experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/archive.zip` |
| archive sha256 | `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a` |
| archive bytes | `34603` |
| runtime path | `experiments/results/time_traveler_recovered_exact_eval_20260514_codex/runtime` |
| runtime content tree sha256 | `630970e9dc78c6e2f8dc2ed8d1e22503ea7d0cab17b4da5615a8a1c5b83ac718` |
| projected runtime tree sha256, CPU | `2b2b9dfdb0f3e59af3511e4502a3a4c0cbe9c1f52405b98eb4dec331db248584` |
| projected runtime tree sha256, CUDA | `2b0dcb5a148ddef7bf56c833bd46fa5830bdde88929b9fd417b4985bea678a28` |
| pair group | `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda` |

## Dispatch Custody

Executed through the canonical paired Modal auth-eval dispatcher:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/archive.zip \
  --submission-dir experiments/results/time_traveler_recovered_exact_eval_20260514_codex/runtime \
  --inflate-sh inflate.sh \
  --label l5_v2_time_traveler_l5_autonomy \
  --expected-archive-sha256 2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a \
  --run-id l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement \
  --pair-group-id pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda \
  --lane-id-base lane_l5_v2_measure_tt5l_autonomy_paired_exact \
  --output-root experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact \
  --modal-bin .venv/bin/modal \
  --gpu T4 \
  --claim-agent codex:l5_v2_paired_measurement_dispatch \
  --claim-notes l5_v2_paired_measurement:pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda \
  --expected-runtime-tree-sha256 auto \
  --skip-axis-if-promotable-anchor-exists \
  --execute \
  --json-out experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/paired_dispatch_execute_plan_20260516T2324Z.json
```

Raw local dispatcher log:
`experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/paired_dispatch_execute_20260516T2324Z.log`

The raw dispatcher JSON/log files are intentionally not tracked because they
contain machine-local and provider-local paths. This ledger records the durable
relative paths, hashes, lane ids, call ids, and recovery commands.

## Axis Results

### CUDA axis

| field | value |
|---|---|
| lane id | `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cuda` |
| Modal call id | `fc-01KRSJ60VVR2PJHYBF47ABFPC0` |
| output dir | `experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cuda` |
| recovery status | `recovered` |
| evidence grade | `[contest-CUDA]` |
| hardware substrate | `linux_x86_64_t4` |
| passed | `true` |
| archive bytes | `34603` |
| avg_segnet_dist | `0.02515214` |
| avg_posenet_dist | `0.18563657` |
| score_rate_contribution | `0.023040717354886494` |
| score_seg_contribution | `2.515214` |
| score_pose_contribution | `1.362485119184793` |
| recomputed score | `3.9007398365396795` |
| promotion_eligible | `false` |

CUDA recovery command:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cuda
```

Result JSON:
`experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cuda/modal_cuda_auth_eval_result.json`

### CPU axis

| field | value |
|---|---|
| lane id | `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cpu` |
| Modal call id | `fc-01KRSJ6J0R0X21BWR2PSFMHP0P` |
| output dir | `experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval_cpu/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu` |
| recovery status at ledger time | `pending` |
| evidence grade | `[pending-contest-CPU]` |
| score_claim | `false` |
| promotion_eligible | `false` |

CPU recovery command:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval_cpu/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu
```

## Classification

This is a legitimate recovered `[contest-CUDA]` result for the exact archive and
runtime above, but it is a negative result for the measured configuration:

- Not a promotion result.
- Not submission-ready.
- Not a reason to kill L5 or L5-v2 as a campaign.
- Classification: `component_collapse_or_recovered_runtime_config_regression`
  until xray review proves whether the failure is model-quality, archive
  grammar, inflate-runtime, scorer-device sensitivity, or a training/export
  mismatch.

The small archive byte term succeeded mechanically, but SegNet and PoseNet
distortions dominate the score. The measured failure therefore lives in the
distortion contract, not in rate accounting.

## Required Follow-Up

1. Recover the CPU axis through the command above and append a paired result
   addendum with `[contest-CPU]` fields.
2. Run an xray-style output review on the CUDA inflated frames before changing
   lane status. Minimum checks: frame count, raw output manifest, first/last
   frame sanity, per-component collapse localization, and comparison against the
   original 25ep smoke artifacts.
3. Check whether the recovered runtime tree matches the intended TT5L smoke
   runtime semantics, not just the projected tree hash.
4. If the failure is export/runtime mismatch, repair and re-pair the same archive
   class. If the failure is trained model quality, treat this exact 25ep archive
   as a negative anchor and continue L5-v2 with a stronger byte-closed
   predictive-receiver/export design.

## No-Signal-Loss Notes

- The untracked live claim ledger records terminal CUDA recovery and active CPU
  pending rows, including the same call ids above.
- Raw Modal artifacts are retained under `experiments/results/l5_v2_probe/` and
  intentionally remain ignored.
- This tracked ledger is the durable source for future analysis, paper
  provenance, and operator handoff.
