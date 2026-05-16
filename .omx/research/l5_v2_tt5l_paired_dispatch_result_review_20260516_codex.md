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
| recovery status | `recovered` |
| evidence grade | `[contest-CPU]` |
| hardware substrate | `linux_x86_64_cpu` |
| passed | `true` |
| archive bytes | `34603` |
| avg_segnet_dist | `0.02515302` |
| avg_posenet_dist | `0.18508005` |
| score_rate_contribution | `0.023040717354886494` |
| score_seg_contribution | `2.515302` |
| score_pose_contribution | `1.360441288700104` |
| recomputed score | `3.8987840060549908` |
| promotion_eligible | `false` |

CPU recovery command:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval_cpu/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu
```

Result JSON:
`experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval_cpu/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu/modal_cpu_auth_eval_result.json`

Note: the first local CPU recovery attempt at `2026-05-16T23:38:09Z` failed
with `ConnectionError: Could not connect to the Modal server` and recorded a
non-score terminal row. A second recovery with network access succeeded at
`2026-05-16T23:40:48Z`; the later terminal row is authoritative.

## Paired Classification

Both axes are now harvested for the same archive SHA
`2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a` and the
same runtime content tree
`630970e9dc78c6e2f8dc2ed8d1e22503ea7d0cab17b4da5615a8a1c5b83ac718`.

| axis | score | seg | pose | rate term | runtime tree |
|---|---:|---:|---:|---:|---|
| `[contest-CPU]` | `3.8987840060549908` | `0.02515302` | `0.18508005` | `0.023040717354886494` | `2b2b9dfdb0f3e59af3511e4502a3a4c0cbe9c1f52405b98eb4dec331db248584` |
| `[contest-CUDA]` | `3.9007398365396795` | `0.02515214` | `0.18563657` | `0.023040717354886494` | `2b0dcb5a148ddef7bf56c833bd46fa5830bdde88929b9fd417b4985bea678a28` |

The CPU/CUDA gap is only about `0.001956`, so this is not primarily a
hardware-axis surprise. It is a paired, component-collapsed measured
configuration: the byte term is excellent, but the frame distortion contract is
not remotely score-bearing.

## Side-Info Liveness Finding

The materialized TT5L archive now fails the tightened work-unit status check
because its per-pair temporal side-info stream is all zero:

| field | value |
|---|---|
| archive member | `0.bin` |
| num_pairs | `600` |
| per_pair_bytes | `45` |
| total_values | `27000` |
| nonzero_values | `0` |
| nonzero_fraction | `0.0` |

This explains why the tiny archive can pass rate accounting while collapsing
SegNet/PoseNet: the measured 25ep packet did not carry an active per-pair
temporal correction signal. This is not a proof that TT5L or the L5 staircase
is dead; it is a precise measured-config failure for the all-zero-side-info
packet.

Hardening added in the live code path:

- `src/tac/optimization/l5_staircase_v2.py` inspects TT5L `archive.zip`
  side-info during materialized paired work-unit validation.
- All-zero side-info now blocks future TT5L paired work-unit dispatch readiness
  with
  `l5_v2_tt5l_materialized_paired_work_unit_tt5l_sideinfo_all_zero`.
- Focused tests cover both nonzero-side-info pass-through and all-zero-side-info
  rejection.

Follow-up hardening added after the paired review:

- `experiments/train_substrate_time_traveler_l5_autonomy.py` now refuses to
  export a TT5L archive when quantized per-pair side-info is empty or all zero,
  so the exact 25ep failure class cannot silently recur at trainer export time.
- Export provenance now records pair coverage and section coverage, not just a
  global nonzero count: `nonzero_pair_count`, `all_zero_pair_count`,
  `min/max/mean_nonzero_values_per_pair`, `section_liveness`, and
  `liveness_warnings`.
- The guard deliberately does not impose an arbitrary nonzero-fraction
  threshold. Sparse side-info may be valid for hard-pair targeting, but the
  provenance now exposes nearly-dead packets such as one-nonzero-byte exports
  for adversarial review and effect-curve gating.
- Focused tests cover empty rejection, all-zero rejection, sparse warning
  provenance, and full section-coverage provenance.

## Probe Intake Update

The returned paired results were converted into the L5 v2 probe observation
intake and gate artifacts:

- `.omx/research/l5_v2_probe_observation_intake_20260516_codex.json`
- `.omx/research/l5_v2_probe_observation_intake_20260516_codex.md`
- `.omx/research/l5_v2_probe_gate_artifact_20260516_codex.json`

The updated intake now records TT5L exact axes as
`['contest_cpu', 'contest_cuda']`, but the gate remains fail-closed:

- C1 and Z5 paired exact observations are still missing.
- TT5L `sideinfo_consumed=false`.
- TT5L score deltas are still missing because no axis-matched non-TT5L baseline
  row has been bound into this probe observation.
- `architecture_lock_allowed=false`.

One hardening bug was fixed in `src/tac/optimization/l5_v2_probe_intake.py`:
Modal CPU evidence records Linux/x86_64 custody through
`provenance.platform_system` + `provenance.platform_machine`, and the CPU run
can carry an `auto` inflate policy while the actual provenance device is CPU.
The intake now normalizes that to hardware `Linux x86_64` and inflate device
`cpu` for CPU-axis validation. It also ignores the expected Modal CPU
`gpu_model=<error:FileNotFoundError...>` probe artifact instead of treating that
as hardware authority.

## Classification

This is a legitimate recovered `[contest-CUDA]` result for the exact archive and
runtime above, and now a legitimate recovered `[contest-CPU]` result for the
same paired packet. It is a negative result for the measured configuration:

- Not a promotion result.
- Not submission-ready.
- Not a reason to kill L5 or L5-v2 as a campaign.
- Classification: `training_export_zero_sideinfo_mismatch`.

The small archive byte term succeeded mechanically, but SegNet and PoseNet
distortions dominate the score. The measured failure therefore lives in the
distortion contract, not in rate accounting. The proximate engineering cause is
a dead per-pair side-info channel in the exported archive/checkpoint, not a
CPU/CUDA axis surprise.

## Required Follow-Up

1. Run an xray-style output review on the CPU and CUDA inflated frames before changing
   lane status. Minimum checks: frame count, raw output manifest, first/last
   frame sanity, per-component collapse localization, and comparison against the
   original 25ep smoke artifacts.
2. Check whether the recovered runtime tree matches the intended TT5L smoke
   runtime semantics, not just the projected tree hash.
3. Materialize the TT5L side-info effect curve (`zero`, `random_lsb`,
   `shuffled`, `trained`, `ablated`) on both axes before treating side-info as
   causally useful.
4. Repair TT5L training/export so `trained` side-info is nonzero and consumed,
   then rerun paired exact only after the liveness guard passes and the
   recorded pair/section coverage is plausible for the intended score-lowering
   mechanism.
5. Materialize C1 and Z5 paired exact work units so the C1/Z5/TT5L probe gate
   can arbitrate the staircase from comparable evidence.
6. Build the Z6 L1 scaffold as the next non-local-minimum score-lowering
   action; do not wait for TT5L 25ep debugging to finish before starting Z6.
7. If the failure is export/runtime mismatch, repair and re-pair the same archive
   class. If the failure is trained model quality, treat this exact 25ep archive
   as a negative anchor and continue L5-v2 with a stronger byte-closed
   predictive-receiver/export design.

## No-Signal-Loss Notes

- The untracked live claim ledger records terminal CUDA and CPU recovery rows,
  including the same call ids above.
- Raw Modal artifacts are retained under `experiments/results/l5_v2_probe/` and
  intentionally remain ignored.
- This tracked ledger is the durable source for future analysis, paper
  provenance, and operator handoff.
