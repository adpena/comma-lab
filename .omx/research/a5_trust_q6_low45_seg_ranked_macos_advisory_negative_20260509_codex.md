# A5 Trust-Region q6-low45 SegNet-Ranked macOS CPU Advisory - 2026-05-09

## Verdict

Measured advisory result: **0.21066071006845696** on macOS CPU.

This improves the scalar blended q6-low45 advisory (`0.21129939214393487`) but
is still not competitive with the A1/PR101 CPU anchor band. It is a scoped A5
component-ranking improvement, not a score claim and not promotion-eligible.

## Candidate

- Technique: `a5_score_marginal_trust_region_q6_low45_seg_ranked`
- Candidate id: `a5_trust_q6_low0p45_seg_20260509_codex`
- Archive:
  `experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/packet/archive.zip`
- Archive bytes: `178138`
- Archive SHA-256:
  `1ffb328240ccb9a067b1203258274c3edfba9f0a98212f1ce1c0a40b3e016501`
- Runtime-tree SHA-256 from compliance packet:
  `9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5`
- Runtime-tree SHA-256 from auth-eval workdir:
  `8f19e65d7b6b97ddb4626b4928efa0901f968b9d7469305ea384534cdc4ccb60`
  (includes packet `report.txt`)
- q-bit schedule: `q6` for the lowest `270 / 600` SegNet-marginal pairs,
  `q8` otherwise
- q-bit mean: `7.1`
- q-bit SHA-256:
  `0915571611d8260e901d7812c38abd8b10cddcdbd54451a191e57eaf223ab886`
- q-bit side-info SHA-256:
  `ed5b98b8ec48d6cf64bdcff7517153c7b36d2c92aa9de6a1a9a37fa4a9e9790e`

## Artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `reports/a5_score_marginal_trust_region_q6_low0p45_seg_20260509.json` | `8214` | `167035cf7dab0d81bddc4356f749228c10cfae2515522adf5f7444b74ca993d2` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/candidate_archive_manifest.json` | `31058` | `972389ddeb7f66c2dc3aba4922e2d781206e8b6839cbcb5ec7b60b7e569f4453` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/runtime_consumption_proof.json` | `4613` | `70443111cde59d228e25559ddf00ca774c6b50945cb0805415440507544b2047` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/pre_submission_compliance.no_auth.json` | `11666` | `1c1f224edf63b03be31ffeb1d931a6a1187fe6a3f531e45af6340635af6454e5` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/readiness.with_score_marginal_packet.json` | `8788` | `a47d8548c9c0eb9d41ca6f6e8230dda647dcc98256f980c743eee1dbc6393d40` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/contest_auth_eval.macos_cpu_advisory.json` | `7792` | `bf7fd2cda0437bdb1c718d12b01766d08797df11468b2886c75f7d745c38f447` |

## Commands

```bash
.venv/bin/python tools/build_a5_score_marginal_qbits_schedule.py \
  --score-marginal-manifest experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
  --json-out reports/a5_score_marginal_trust_region_q6_low0p45_seg_20260509.json \
  --candidate-id a5_trust_q6_low0p45_seg_20260509_codex \
  --base-q-bits 8 \
  --low-q-bits 6 \
  --low-fraction 0.45 \
  --marginal-source seg \
  --latent-dim 28

.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --q-bits-json reports/a5_score_marginal_trust_region_q6_low0p45_seg_20260509.json \
  --recompute-wire-contract-for-q-bits \
  --output-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex \
  --candidate-id pr101_a5_trust_q6_low0p45_seg_20260509_codex \
  --force

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/packet \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/packet/archive.zip \
  --archive-manifest-json experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/candidate_archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 1ffb328240ccb9a067b1203258274c3edfba9f0a98212f1ce1c0a40b3e016501 \
  --expected-archive-size-bytes 178138 \
  --expected-runtime-tree-sha256 9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5 \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/pre_submission_compliance.no_auth.json \
  --strict

.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id a5_trust_q6_low0p45_seg_macos_cpu_advisory \
  --platform local_macos_cpu \
  --instance-job-id local:a5-trust-q6-low0p45-seg-macos-cpu-20260509T0359Z \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-09T04:10Z \
  --status active_eval \
  --notes "Local macOS CPU advisory eval for SegNet-ranked A5 q6-low45 packet; non-promotable diagnostic; archive_sha=1ffb328240ccb9a067b1203258274c3edfba9f0a98212f1ce1c0a40b3e016501"

PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/macos_cpu_advisory_work \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low0p45_seg_20260509_codex/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Terminal claim row:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id a5_trust_q6_low0p45_seg_macos_cpu_advisory \
  --platform local_macos_cpu \
  --instance-job-id local:a5-trust-q6-low0p45-seg-macos-cpu-20260509T0359Z \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-09T04:10Z \
  --status completed_macos_cpu_advisory_negative \
  --notes "Completed local macOS CPU advisory eval: score=0.210660710068 pose=0.00004059 seg=0.00071899 bytes=178138 archive_sha=1ffb328240ccb9a067b1203258274c3edfba9f0a98212f1ce1c0a40b3e016501; non-promotable" \
  --force
```

## Advisory Eval Result

- evidence grade: `[macOS-CPU advisory]`
- canonical score: `0.21066071006845696`
- pose distance: `0.00004059`
- seg distance: `0.00071899`
- rate contribution: `0.11861475`
- samples: `600`
- inflate elapsed: `35.03284608304966 s`
- evaluate elapsed: `411.76618012494873 s`

## Interpretation

SegNet-ranked q6-low45 improves over the scalar blended q6-low45 configuration:

- scalar q6-low45: `0.21129939214393487`, SegNet `0.00072565`
- SegNet-ranked q6-low45: `0.21066071006845696`, SegNet `0.00071899`

The improvement is real but too small. At the same byte count, component
ranking recovers only `0.00000666` SegNet distortion, worth about `0.000666`
score points. The candidate still loses about `0.0178` score points versus
the A1 Linux CPU anchor (`0.192847577437`).

Classification: measured-config regression. This retires scalar/one-component
ranked A5 q6-low45 schedules at the current q-bit geometry. It does not kill
A5 as a family; it says per-pair scalar component marginals are too coarse.

## Generated Custody Disposition

The packet and auth-eval work directories under `experiments/results/` remain
raw rebuildable custody. They were not promoted as tracked source. The durable
tracked signal is this ledger plus
`reports/a5_score_marginal_trust_region_q6_low0p45_seg_20260509.json`.

`experiments/results/` runtime-source baseline in
`.omx/research/untracked_source_dispositions_20260505_codex.json` was
rebaselined after this packet/eval:

- count: `10583`
- SHA-256: `7628917a3bc32dd5fd1a199d01f40f71e7bab905ccc26533e630235e4d05c267`

## Reactivation Criteria

- Use a richer SegNet-boundary feature than per-pair scalar average; e.g.
  boundary mass, class-change mass, or localized component response.
- Try nonuniform q-level ladders (`q7`/`q8`, or mixed `q6/q7/q8`) that reduce
  distortion before paying another exact eval.
- Require local advisory score near the A1/PR101 band before paired exact
  `[contest-CPU]` and `[contest-CUDA]` dispatch.
