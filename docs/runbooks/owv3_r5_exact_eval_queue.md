# OWV3 R5/R6 Exact-Eval Forensic Runbook

Updated: 2026-05-01T00:12Z

This runbook is historical plus current forensic guidance. It is not a score
claim. OWV3 R5 and R6 both have exact CUDA/T4 evidence and are
non-promotable. R6 completed at `2026-04-30T23:47:45Z`; R7 scalar-threshold
selection currently returns zero candidates on the existing grid. No OWV3 job
is active from this runbook.

## Exact R5 Outcome

Paired calibration on the same Lightning/T4 path:

```text
pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv
score_recomputed_from_components = 1.037045485927815
bytes = 686635
archive_sha256 = 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
avg_posenet_dist = 0.00316404
avg_segnet_dist = 0.00401966
promotion_eligible = false
lane_status = COMPONENT_GATE_REVIEW_REQUIRED
```

R5 rank-1 exact eval:

```text
owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv
candidate = owv3_0047_bbr0p67_protect0p00135_aggr1em05
score_recomputed_from_components = 1.0373951773937642
bytes = 686468
archive_sha256 = 16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518
avg_posenet_dist = 0.0031739
avg_segnet_dist = 0.0040215
promotion_eligible = false
lane_status = COMPONENT_GATE_REVIEW_REQUIRED
```

Paired result: R5 is `+0.00034969146594909795` worse than paired PFP16 despite
`167` fewer bytes. Do not promote R5. Do not relax the component gate
retroactively.

## R6 Exact Result

R6 candidate selected by the strict post-R5 rule and evaluated on exact
CUDA/T4:

```text
candidate = owv3_0076_bbr0p65_protect0p0013_aggr1em05
bytes = 686531
archive_sha256 = 9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91
frontier_delta_bytes = -104
owv2_low_bit_channels = 58
keep_asym_channels = 695
```

Selection rationale: R5 failed the SegNet component gate with 62 OWV2-low-bit
channels. R6 requires a strictly lower low-bit channel count before spending
another exact CUDA eval. The current top candidate reduces that count to 58
while preserving a small byte win versus PFP16.

Lightning job:

```text
job_name = owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1
SDK job = owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1
status at 2026-04-30T23:47:45Z = Completed
machine = g4dn.2xlarge (T4)
remote_output_dir = /teamspace/studios/this_studio/pact/experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1
local_artifact_dir = experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1
```

Exact result:

```text
score_recomputed_from_components = 1.0393166493980681
final_score = 1.04
archive_bytes = 686531
archive_sha256 = 9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91
device = cuda
gpu_model = Tesla T4
n_samples = 600
avg_posenet_dist = 0.00323147
avg_segnet_dist = 0.00402421
```

Paired PFP16 gates:

```text
baseline_score = 1.037045485927815
baseline_archive_bytes = 686635
baseline_posenet_dist = 0.00316404
baseline_segnet_dist = 0.00401966
max_posenet_relative = 1.002
max_segnet_relative = 1.002
allow_component_gate_forensic_success = true
```

Verdict:

- R6 regressed versus paired PFP16 by `+0.0022711634702530237` score while
  saving `104` bytes.
- Strict final-deploy adjudication returned exit code `2`.
- PoseNet component gate failed: `0.00323147 / 0.00316404 =
  1.0213113614240024`, above the `1.002` limit.
- SegNet component gate passed: `0.00402421 / 0.00401966 =
  1.0011319365319455`, below the `1.002` limit.
- Classification: A++ exact CUDA/T4 scoped forensic negative for this R6
  implementation/config. It is not promotable and is not an OWV3 family KILL.

## Artifact Custody

Canonical artifacts were harvested with `harvest-ssh` from the SDK artifact
mirror and locally validated with:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts \
  --artifact-dir experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1 \
  --expected-archive-sha256 9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91 \
  --expected-archive-size-bytes 686531 \
  --require-adjudication
```

Next OWV3 work must address PoseNet drift. Do not rerun this same R6
candidate unless the eval harness or archive identity changes; treat it as a
closed scoped negative.
