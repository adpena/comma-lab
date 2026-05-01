# OWV3 R6 SegNet-Conservative Readiness - Worker A - 2026-04-30

Scope: OWV3 R6 design/implementation readiness only. No builder/runtime code was
edited. Evidence here is diagnostic/planning unless it cites exact CUDA auth
eval JSON on exact archive bytes.

## Inputs Read

- `AGENTS.md`
- `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`
- `experiments/build_lane_g_v3_owv3_stack.py`
- `experiments/sweep_owv3_byte_plan.py`
- `src/tac/owv3_sensitivity_weighted.py`
- Harvested exact R5 artifact:
  `experiments/results/lightning_batch/owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv/`
- Paired calibration artifact:
  `experiments/results/lightning_batch/pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv/`
- R5 byte-plan sweep:
  `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/`

## Exact CUDA Metrics

Final deploy PFP16 A++ reference from the claim matrix:

- Score: `1.043987524793892`
- Bytes: `686635`
- SHA-256: `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
- PoseNet: `0.00346442`
- SegNet: `0.00400656`

Clean paired PFP16 calibration on isolated Lightning T4:

- Path:
  `experiments/results/lightning_batch/pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv/contest_auth_eval.json`
- Score recomputed: `1.037045485927815`
- Final rounded score: `1.04`
- Bytes: `686635`
- SHA-256: `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
- PoseNet: `0.00316404`
- SegNet: `0.00401966`
- Samples/device/hardware: `600`, `cuda`, `Tesla T4`, `gpu_t4_match=true`
- Contributions: pose `0.17787748592781494`, SegNet `0.40196600000000005`,
  rate `0.45720206227704213`
- Note: this paired PFP16 run itself fails the final-deploy SegNet relative cap
  (`0.00401966 / 0.00400656 = 1.0032696377940178 > 1.002`). Do not use that
  mismatch to promote OWV3; use it to keep paired-run forensic deltas separate
  from strict final-deploy promotion gates.

R4 exact OWV3 byte-feasible reference:

- Path:
  `experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4/contest_auth_eval.json`
- Candidate: `owv3_0018_bbr0p69_protect0p0014_aggr1em05`
- Score recomputed: `1.0378905176070103`
- Bytes: `686557` (`-78` versus PFP16)
- SHA-256: `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`
- PoseNet: `0.00319052`
- SegNet: `0.00402120`
- Samples/device/hardware: `600`, `cuda`, `Tesla T4`, `gpu_t4_match=true`
- Failed final-deploy SegNet gate:
  `0.00402120 / 0.00400656 = 1.0036540074278184 > 1.002`

R5 exact OWV3 rank-1:

- Path:
  `experiments/results/lightning_batch/owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv/contest_auth_eval.json`
- Candidate: `owv3_0047_bbr0p67_protect0p00135_aggr1em05`
- Score recomputed: `1.0373951773937642`
- Final rounded score: `1.04`
- Bytes: `686468` (`-167` versus PFP16)
- SHA-256: `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`
- PoseNet: `0.00317390`
- SegNet: `0.00402150`
- Samples/device/hardware: `600`, `cuda`, `Tesla T4`, `gpu_t4_match=true`
- Contributions: pose `0.1781544273937642`, SegNet `0.40215`,
  rate `0.4570908638318707`
- Failed final-deploy SegNet gate:
  `0.00402150 / 0.00400656 = 1.003728884629208 > 1.002`

R5 deltas versus paired PFP16:

- Score delta: `+0.00034969146594909795` worse
- PoseNet dist delta: `+0.00000986`
- Pose contribution delta: `+0.0002769414659492542`
- SegNet dist delta: `+0.00000184`
- SegNet contribution delta: `+0.0001839999999999481`
- Byte delta: `-167`
- Rate contribution delta: `-0.00011119844517140262`
- PoseNet ratio: `1.0031162690737159`
- SegNet ratio: `1.0004577501579734`

R5 deltas versus R4:

- Score delta: `-0.0004953402132461537` better
- PoseNet dist delta: `-0.00001662`
- Pose contribution delta: `-0.0004658402132461936`
- SegNet dist delta: `+0.00000030`
- SegNet contribution delta: `+0.000029999999999995308`
- Byte delta: `-89`
- Rate contribution delta: `-0.00005926144682787325`

Interpretation: R5 improved the total versus R4 mostly through PoseNet and a
small byte win, but it did not reduce SegNet drift. Versus clean paired PFP16,
R5's byte win is too small to offset PoseNet plus SegNet drift.

## Codec/Selector Facts

Current OWV3 mechanics:

- `protect_threshold` controls channel routing:
  sensitivity `> protect_threshold` stays protected, sensitivity
  `<= protect_threshold` goes through OWV2 low-bit coding.
- Lowering `protect_threshold` is SegNet-conservative in channel routing because
  it keeps more channels in ASYM.
- Lowering `bit_budget_ratio` is not conservative for the channels that remain
  in OWV2, but may be needed to keep archive bytes below the PFP16 frontier.
- Promotion-compatible candidates must use `fallback_action=keep_asym`,
  `diagnostic_fp16_layers=0`, `fp16_protect_channels=0`, and
  `promotion_eligible=true` in the OWV3 byte plan.

R4/R5 exact-evaluated headers:

| Candidate | bbr | protect | bytes | delta vs PFP16 | OWV2 low-bit channels | keep ASYM channels |
|---|---:|---:|---:|---:|---:|---:|
| R4 `owv3_0018...` | `0.69` | `0.00140` | `686557` | `-78` | `65` | `688` |
| R5 `owv3_0047...` | `0.67` | `0.00135` | `686468` | `-167` | `62` | `691` |

Layer-level R5 versus R4 routing:

- `renderer.down_res.conv1`: `3` quantized / `57` protected in both.
- `renderer.down_res.conv2`: R4 `11/49`, R5 `10/50`.
- `renderer.bottleneck.conv2`: R4 `31/29`, R5 `30/30`.
- `renderer.up_res.conv2`: R4 `20/16`, R5 `19/17`.

## R6 Candidate Order

Primary R6 exact-eval candidate: smallest controlled additional protection
already materialized in the R5 sweep.

- Candidate id: `owv3_0076_bbr0p65_protect0p0013_aggr1em05`
- Archive:
  `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/archives/owv3_0076_bbr0p65_protect0p0013_aggr1em05.zip`
- Bytes: `686531`
- SHA-256: `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`
- Byte delta versus PFP16: `-104`
- Byte delta versus R5: `+63` (spends back 63 of R5's 167-byte saving)
- `bit_budget_ratio=0.65`
- `protect_threshold=0.00130`
- `aggressive_threshold=0.00001`
- `fallback_action=keep_asym`
- OWV2 low-bit channels: `58`
- keep ASYM channels: `695`
- Layer routing:
  `renderer.down_res.conv1 3/57`,
  `renderer.down_res.conv2 8/52`,
  `renderer.bottleneck.conv2 29/31`,
  `renderer.up_res.conv2 18/18`.

Why this is the shortest correct R6 spend: it is the nearest already-built
candidate that lowers OWV2 low-bit channels below R5 (`62 -> 58`) while keeping
archive bytes below the PFP16 frontier. It directly tests whether protecting
four more channels is enough to reduce/avoid SegNet drift without mixing in a
larger sweep or code changes.

Second candidate only if the primary has acceptable components but insufficient
rate/total score:

- Candidate id: `owv3_0091_bbr0p64_protect0p0013_aggr1em05`
- Archive:
  `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/archives/owv3_0091_bbr0p64_protect0p0013_aggr1em05.zip`
- Bytes: `686228`
- SHA-256: `31f9f1527a9733c24bc48d28aff6bbf881540873f6423b33b79fafac0493ea13`
- Byte delta versus PFP16: `-407`
- `bit_budget_ratio=0.64`
- `protect_threshold=0.00130`
- OWV2 low-bit channels: `58`
- keep ASYM channels: `695`
- This is not more SegNet-conservative than the primary; it is a rate-margin
  variant of the same protected-channel set.

Temporary byte-only R6 probe outside the repo found finer-grid alternatives.
These are not score evidence and are not materialized under `experiments/`.
They are useful only if primary R6 still shows SegNet drift:

| Probe candidate | bbr | protect | bytes | delta vs PFP16 | OWV2 low-bit | keep ASYM | SHA-256 if rebuilt deterministically |
|---|---:|---:|---:|---:|---:|---:|---|
| `r6_55a` | `0.63` | `0.00120` | `686458` | `-177` | `55` | `698` | `20103c3716ff8a4eaf3068475eb9bd358c0f6f0055b6b99207164d166fea6b6b` |
| `r6_55b` | `0.63` | `0.00125` | `686459` | `-176` | `55` | `698` | `450d81bd8a42e27cc8918cffe7929f2b831476ace0c3dfeee94430b2e18bf42e` |
| `r6_52` | `0.61` | `0.00110` | `686419` | `-216` | `52` | `701` | `6236f98a90a126d354346a1750a43f4d4b62b4b4d5f70b674d1854dfb3f7accf` |
| `r6_50` | `0.60` | `0.00105` | `686572` | `-63` | `50` | `703` | `50dcfa41074cfce6b9ff9ef50af5f3b4a4b08229d4618bc5432045d14f4e9aee` |
| `r6_47` | `0.57` | `0.00100` | `686430` | `-205` | `47` | `706` | `219db690235db865f3e8a8cf0c1e2d95a777f882f2bfb6539c30bb888b09284c` |

The `r6_55a/r6_55b` pair is the best fallback direction if the primary R6
keeps SegNet high: it protects seven more channels than R5 while preserving
essentially the same byte savings. The risk is the lower `bit_budget_ratio`
on the remaining OWV2 channels.

## R6 Gates

Predeclare both forensic paired-run gates and strict final-deploy gates:

- Archive bytes must be `<= 686635`.
- Archive SHA-256 and bytes must match the queued expectation.
- Device must be `cuda`, samples must be `600`, and `gpu_t4_match` must be
  `true`.
- Forensic paired-run component caps:
  - paired baseline score `1.037045485927815`
  - paired baseline PoseNet `0.00316404`
  - paired baseline SegNet `0.00401966`
  - max PoseNet relative `1.002`
  - max SegNet relative `1.002`
- Strict final-deploy promotion caps remain:
  - final-deploy PoseNet reference `0.00346442`
  - final-deploy SegNet reference `0.00400656`
  - max SegNet relative `1.002`, threshold `0.00401457312`
- If paired-run gates pass but strict final-deploy SegNet gate fails, record
  diagnostic evidence only. Do not promote by retroactively relaxing the gate.
- Score must beat paired PFP16 before any "current frontier" discussion:
  `score_recomputed_from_components < 1.037045485927815`.

Useful score math for the primary candidate:

- Primary R6 byte saving versus paired PFP16 is `104` bytes, worth
  `-0.00006924933112470582` score.
- If R6 primary keeps R5's PoseNet contribution, SegNet would need to be
  `<= 0.004017582455881335` to tie paired PFP16.
- If R6 primary keeps paired PFP16 SegNet exactly, PoseNet can rise only to
  `0.003166501842532642` to tie paired PFP16.

## Next Commands

Local byte/custody sanity for the primary candidate:

```bash
stat -f '%z %N' \
  experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/archives/owv3_0076_bbr0p65_protect0p0013_aggr1em05.zip
shasum -a 256 \
  experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/archives/owv3_0076_bbr0p65_protect0p0013_aggr1em05.zip
unzip -lv \
  experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/archives/owv3_0076_bbr0p65_protect0p0013_aggr1em05.zip
```

Lightning exact eval for primary R6, using the same isolated-UV wrapper pattern:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
  --job-name owv3_r6_segnet_conservative_0076_exact_cuda_20260430_worker_a \
  --archive /teamspace/studios/this_studio/pact/experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/archives/owv3_0076_bbr0p65_protect0p0013_aggr1em05.zip \
  --repo-dir /teamspace/studios/this_studio/pact \
  --upstream-dir /teamspace/studios/this_studio/upstream \
  --machine T4 \
  --studio lossy-compression-challenge \
  --teamspace comma-lab \
  --python-bin /teamspace/studios/this_studio/pact_pfp16_exact_20260430T1625Z/.venv/bin/python \
  --expected-archive-sha256 9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91 \
  --expected-archive-size-bytes 686531 \
  --adjudicate \
  --baseline-score 1.037045485927815 \
  --predicted-band 1.02 1.037045485927815 \
  --regression-threshold 1.037045485927815 \
  --baseline-archive-bytes 686635 \
  --baseline-posenet-dist 0.00316404 \
  --baseline-segnet-dist 0.00401966 \
  --max-posenet-relative 1.002 \
  --max-segnet-relative 1.002 \
  --component-reference-label pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv \
  --queue-metadata lane=owv3_r6_segnet_conservative \
  --queue-metadata candidate_id=owv3_0076_bbr0p65_protect0p0013_aggr1em05 \
  --queue-metadata strict_final_deploy_segnet_reference=0.00400656 \
  --queue-metadata strict_final_deploy_segnet_relative_cap=1.002 \
  --queue-metadata isolated_uv_project_environment=true
```

After harvest, run a separate strict final-deploy component check before any
promotion discussion. This may fail even if the paired forensic gate passes:

```bash
.venv/bin/python scripts/adjudicate_contest_auth_eval.py \
  --contest-json experiments/results/lightning_batch/owv3_r6_segnet_conservative_0076_exact_cuda_20260430_worker_a/contest_auth_eval.json \
  --provenance experiments/results/lightning_batch/owv3_r6_segnet_conservative_0076_exact_cuda_20260430_worker_a/adjudication_final_deploy_provenance.json \
  --archive experiments/results/lightning_batch/owv3_r6_segnet_conservative_0076_exact_cuda_20260430_worker_a/archive.zip \
  --result-copy experiments/results/lightning_batch/owv3_r6_segnet_conservative_0076_exact_cuda_20260430_worker_a/contest_auth_eval.final_deploy_gated.json \
  --baseline-score 1.043987524793892 \
  --predicted-band 1.02 1.043987524793892 \
  --regression-threshold 1.043987524793892 \
  --baseline-archive-bytes 686635 \
  --baseline-posenet-dist 0.00346442 \
  --baseline-segnet-dist 0.00400656 \
  --max-posenet-relative 1.002 \
  --max-segnet-relative 1.002 \
  --component-reference-label pfp16_final_deploy_bundle_20260430 \
  --required-device cuda \
  --required-samples 600
```

If the primary R6 still shows SegNet drift and the team wants the finer-grid
fallback, materialize `r6_55b` deterministically before exact eval:

```bash
.venv/bin/python experiments/build_lane_g_v3_owv3_stack.py \
  --sensitivity-map experiments/results/lane_g_v3_owv3_fisher_lightning_20260430_codex_r2/owv3_sensitivity_map.pt \
  --output experiments/results/lane_g_v3_owv3_r6_segnet_conservative_20260430_worker_a/r6_55b/archive_lane_g_v3_owv3.zip \
  --provenance-json experiments/results/lane_g_v3_owv3_r6_segnet_conservative_20260430_worker_a/r6_55b/build_provenance.json \
  --bit-budget-ratio 0.63 \
  --protect-threshold 0.00125 \
  --aggressive-threshold 0.00001 \
  --fallback-action keep_asym
```

Expected rebuilt `r6_55b` archive identity from byte-only probe:

- Bytes: `686459`
- SHA-256: `450d81bd8a42e27cc8918cffe7929f2b831476ace0c3dfeee94430b2e18bf42e`

## Proposed Code Changes Only If Automation Is Needed

No code change is required for the shortest R6 eval because the primary archive
already exists. If this selector should be made reusable, add a narrow
`r6-segnet-conservative` mode to `experiments/sweep_owv3_byte_plan.py`:

- Reference the exact failed R5 candidate instead of the R4 candidate.
- Require `byte_feasible_vs_frontier=true`, `fallback_action=keep_asym`,
  no diagnostic FP16, and `promotion_eligible=true`.
- Require `owv2_low_bit_channels < 62`.
- Sort by the smallest additional protection step first:
  `(62 - owv2_low_bit_channels)`, then highest `bit_budget_ratio`, then
  smallest frontier byte margin, then candidate id.
- Emit both paired-run forensic adjudication placeholders and strict
  final-deploy gate metadata so the queue cannot accidentally adjudicate R6
  against the wrong baseline.

Suggested focused tests if implemented:

- Extend `src/tac/tests/test_sweep_owv3_byte_plan.py` with a fixture proving
  the R6 selector picks the `58`-low-bit nearest neighbor before lower-bit-rate
  or much more aggressive candidates.
- Add a queue-template assertion that R6 exact-eval metadata contains both
  paired PFP16 baseline metrics and strict final-deploy SegNet reference.
