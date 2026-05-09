# A5 q7/q8 SegNet-Protected Scalar Negative - 2026-05-09

## Verdict

Measured advisory result: **0.20111041630821824** on macOS CPU.

This is the best scalar A5 q-bit schedule tested so far, and it proves the
SegNet marginal ranking has a real but insufficient signal. Preserving the
top `90 / 600` SegNet-marginal pairs at `q8` improves q7-all by about
`0.00153` score points, but it remains about `0.00826` worse than the A1 Linux
CPU anchor while saving only `19 B`.

This retires post-hoc scalar A5 q-bit schedules as exact-eval candidates until
the allocator becomes local to SegNet boundary structure or the quantization
noise is present during training.

## Candidate

- Technique: `a5_score_marginal_trust_region_q7_low0p85_seg`
- Candidate id: `a5_trust_q7_low0p85_seg_20260509_codex`
- Archive:
  `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/packet/archive.zip`
- Archive bytes: `178243`
- Archive SHA-256:
  `28f4e0ee3ede86f323b484e36711fd01204ad101517f40ef32000417d2e4a896`
- Runtime-tree SHA-256 from compliance packet:
  `9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5`
- Runtime-tree SHA-256 from auth-eval workdir:
  `708ef03ec2b912a742b47cd99b2bcaaf0c66735daa6c82306ffc294a3550cb6b`
  (includes packet `report.txt`)
- q-bit schedule: `q7` for the lowest `510 / 600` SegNet-marginal pairs,
  `q8` otherwise
- q-bit mean: `7.15`
- q-bit SHA-256:
  `05651cda24b53a3a9214371ccc57c2fd259ea8c4c52cf98483fce9b658e676aa`
- q-bit side-info SHA-256:
  `26e0cbad5253bcf6851e7675c64c3142e4ad69d62b798f59c6c468f93c09ca3e`

## Artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `reports/a5_score_marginal_trust_region_q7_low0p85_seg_20260509.json` | `8210` | `b033f36365fbeab533d426b6591834162710844078cd0c1f5fe2979f31be846f` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/candidate_archive_manifest.json` | `31059` | `9b6386b35869276180e8c91f65ffad265adb58ce0f11eb1d6c9fec53e61e9028` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/runtime_consumption_proof.json` | `4613` | `646fce7c9c137a282263727c3cbc11a41de0cdd00c839bd0015e878a64d6d475` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/pre_submission_compliance.no_auth.json` | `11666` | `76e2b26e6433e9140524cccef34b0c4b077c7622b671b91081ba58dcf15905c8` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/readiness.with_score_marginal_packet.json` | `9564` | `4acb16c3a553bb565a00a9107fa2a5030f5404c53cb38238e288f70acb519e04` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/contest_auth_eval.macos_cpu_advisory.json` | `7778` | `2181374990f796b53a68b43ac5030b7e011e4adcad2d5b61e031e951f65c00b0` |

## Commands

```bash
.venv/bin/python tools/build_a5_score_marginal_qbits_schedule.py \
  --score-marginal-manifest experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
  --json-out reports/a5_score_marginal_trust_region_q7_low0p85_seg_20260509.json \
  --candidate-id a5_trust_q7_low0p85_seg_20260509_codex \
  --base-q-bits 8 \
  --low-q-bits 7 \
  --low-fraction 0.85 \
  --marginal-source seg \
  --latent-dim 28

.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --q-bits-json reports/a5_score_marginal_trust_region_q7_low0p85_seg_20260509.json \
  --recompute-wire-contract-for-q-bits \
  --output-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex \
  --candidate-id pr101_a5_trust_q7_low0p85_seg_20260509_codex \
  --force

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/packet \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/packet/archive.zip \
  --archive-manifest-json experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/candidate_archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 28f4e0ee3ede86f323b484e36711fd01204ad101517f40ef32000417d2e4a896 \
  --expected-archive-size-bytes 178243 \
  --expected-runtime-tree-sha256 9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5 \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/pre_submission_compliance.no_auth.json \
  --strict

.venv/bin/python tools/build_pr101_frame_conditional_packet_readiness.py \
  --a5-manifest experiments/results/pr101_frame_conditional_bit_codex_20260508T_wire_contract_smoke/build_manifest.json \
  --candidate-archive-manifest experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/candidate_archive_manifest.json \
  --packet-runtime-patch-manifest experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/packet_runtime_patch_manifest.json \
  --runtime-consumption-proof experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/runtime_consumption_proof.json \
  --per-pair-score-marginal-manifest reports/a5_score_marginal_trust_region_q7_low0p85_seg_20260509.json \
  --strict-pre-submission-compliance-json experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/pre_submission_compliance.no_auth.json \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/readiness.with_score_marginal_packet.json

.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id a5_trust_q7_low0p85_seg_macos_cpu_advisory \
  --platform local_macos_cpu \
  --instance-job-id local:a5-trust-q7-low0p85-seg-macos-cpu-20260509T0431Z \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-09T04:43Z \
  --status active_eval \
  --notes "Local macOS CPU advisory eval for A5 SegNet-ranked q7/q8 low0p85 packet; non-promotable diagnostic; archive_sha=28f4e0ee3ede86f323b484e36711fd01204ad101517f40ef32000417d2e4a896"

PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/macos_cpu_advisory_work \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_low0p85_seg_20260509_codex/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Terminal claim row:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id a5_trust_q7_low0p85_seg_macos_cpu_advisory \
  --platform local_macos_cpu \
  --instance-job-id local:a5-trust-q7-low0p85-seg-macos-cpu-20260509T0431Z \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-09T04:43Z \
  --status completed_macos_cpu_advisory_negative \
  --notes "Completed local macOS CPU advisory eval: score=0.201110416308 pose=0.00003517 seg=0.00063672 bytes=178243 archive_sha=28f4e0ee3ede86f323b484e36711fd01204ad101517f40ef32000417d2e4a896; non-promotable scalar A5 negative" \
  --force
```

## Advisory Eval Result

- evidence grade: `[macOS-CPU advisory]`
- canonical score: `0.20111041630821824`
- pose distance: `0.00003517`
- seg distance: `0.00063672`
- rate contribution: `0.11868475`
- samples: `600`
- inflate elapsed: `34.94052504200954 s`
- evaluate elapsed: `408.9117468749173 s`

## Interpretation

The scalar ranking signal is real:

- q7-all: `0.2026389105740624`, SegNet `0.00065288`, bytes `177928`
- SegNet-protected q7/q8 low0p85: `0.20111041630821824`, SegNet
  `0.00063672`, bytes `178243`

The improvement is not large enough:

- A1 Linux CPU anchor: `0.192847577437`, bytes `178262`
- q7/q8 low0p85 saves only `19 B` versus A1, worth about `0.00001265`
  score points.
- It remains about `0.00826` score points worse than A1.

Classification: measured-config regression. This is the strongest evidence so
far that global scalar per-pair q-bit schedules are the wrong A5 abstraction.
The ranking has signal, but the sidechannel needs local SegNet-boundary
resolution or training-time adaptation to make the signal matter.

## Generated Custody Disposition

The packet and auth-eval work directories under `experiments/results/` remain
raw rebuildable custody. They were not promoted as tracked source. The durable
tracked signal is this ledger plus
`reports/a5_score_marginal_trust_region_q7_low0p85_seg_20260509.json`.

`experiments/results/` runtime-source baseline in
`.omx/research/untracked_source_dispositions_20260505_codex.json` was
rebaselined after this packet/eval:

- count: `10587`
- SHA-256: `d25e6037c98b49988b8b2c27af5cd8f4d455d10d41eff6dafd3c287af0a85ae0`

## Reactivation Criteria

- Replace scalar per-pair q values with a local boundary-aware allocation
  within each pair or latent channel.
- Train with q-bit noise in the loop so the renderer can move error away from
  SegNet boundaries.
- Reopen exact-eval spend only if local macOS CPU advisory reaches within
  `0.001` of A1/PR101 or if a new archive-size reduction is large enough to
  pay the measured SegNet distortion.
