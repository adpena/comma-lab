# A5 Trust-Region q7-All macOS CPU Advisory - 2026-05-09

## Verdict

Measured advisory result: **0.2026389105740624** on macOS CPU.

This is the best A5 trust-region packet tested so far, but it is still not
competitive with the A1/PR101 CPU anchor band. It is a scoped geometry
negative: `q7` for all frame pairs preserves much more SegNet/PoseNet signal
than the `q6` schedules, but the remaining SegNet distortion dominates the
small `334 B` rate win versus the A1 anchor.

## Candidate

- Technique: `a5_score_marginal_trust_region_q7_all`
- Candidate id: `a5_trust_q7_all_20260509_codex`
- Archive:
  `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/packet/archive.zip`
- Archive bytes: `177928`
- Archive SHA-256:
  `39dbfd05d4861c6c5ea12e7bfc8fba17e8249dcc761e9c44943eeba8d56c6ade`
- Runtime-tree SHA-256 from compliance packet:
  `9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5`
- Runtime-tree SHA-256 from auth-eval workdir:
  `9b2bc39c58fc02e3cbda3f3830da487b10ccac1cc658f7a0ecb9277ddd912700`
  (includes packet `report.txt`)
- q-bit schedule: `q7` for all `600 / 600` pairs
- q-bit mean: `7.0`
- q-bit SHA-256:
  `f3729137ab245824f035f3a855878a52ffed858b8b01482a1d6bae6d0d9f7028`
- q-bit side-info SHA-256:
  `550c83172b44065a3e85c65e789ed90adbb6f658e66bb7a9023b2f8772df8909`

## Artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `reports/a5_score_marginal_trust_region_q7_all_20260509.json` | `8091` | `c7a842c6e1814ec5ccfca38a096e0fedb39a5442f509eb3e1a7a0c0d97a586e7` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/candidate_archive_manifest.json` | `30946` | `b18732ae08957f6c899b48d8e2f4bb341fe683963554d4a1b2843de5344ab853` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/runtime_consumption_proof.json` | `4589` | `cb6dcbe109d985630e8ab18976053071c7012deaf6fcce957084e248529ad7f8` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/pre_submission_compliance.no_auth.json` | `11482` | `7ec4941bf8fe6e0c261a50bd9d53e51b9ccaf98a9a3c0ffe8413342be57a5f2c` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/readiness.with_score_marginal_packet.json` | `8652` | `cdce96e338f7f38db55a5ad3994bfcd21d76ff4cd20dd30362751709564877a1` |
| `experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/contest_auth_eval.macos_cpu_advisory.json` | `7653` | `9a92aff5ace9aadabcd42de1b559d3d3b55f4676e37d079d0c12718a0b408db3` |

## Commands

```bash
.venv/bin/python tools/build_a5_score_marginal_qbits_schedule.py \
  --score-marginal-manifest experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
  --json-out reports/a5_score_marginal_trust_region_q7_all_20260509.json \
  --candidate-id a5_trust_q7_all_20260509_codex \
  --base-q-bits 8 \
  --low-q-bits 7 \
  --low-fraction 1.0 \
  --marginal-source score \
  --latent-dim 28

.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --q-bits-json reports/a5_score_marginal_trust_region_q7_all_20260509.json \
  --recompute-wire-contract-for-q-bits \
  --output-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex \
  --candidate-id pr101_a5_trust_q7_all_20260509_codex \
  --force

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/packet \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/packet/archive.zip \
  --archive-manifest-json experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/candidate_archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 39dbfd05d4861c6c5ea12e7bfc8fba17e8249dcc761e9c44943eeba8d56c6ade \
  --expected-archive-size-bytes 177928 \
  --expected-runtime-tree-sha256 9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5 \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/pre_submission_compliance.no_auth.json \
  --strict

.venv/bin/python tools/build_pr101_frame_conditional_packet_readiness.py \
  --a5-manifest experiments/results/pr101_frame_conditional_bit_codex_20260508T_wire_contract_smoke/build_manifest.json \
  --candidate-archive-manifest experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/candidate_archive_manifest.json \
  --packet-runtime-patch-manifest experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/packet_runtime_patch_manifest.json \
  --runtime-consumption-proof experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/runtime_consumption_proof.json \
  --per-pair-score-marginal-manifest reports/a5_score_marginal_trust_region_q7_all_20260509.json \
  --strict-pre-submission-compliance-json experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/pre_submission_compliance.no_auth.json \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/readiness.with_score_marginal_packet.json

.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id a5_trust_q7_all_macos_cpu_advisory \
  --platform local_macos_cpu \
  --instance-job-id local:a5-trust-q7-all-macos-cpu-20260509T0418Z \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-09T04:30Z \
  --status active_eval \
  --notes "Local macOS CPU advisory eval for A5 q7-all packet; non-promotable diagnostic; archive_sha=39dbfd05d4861c6c5ea12e7bfc8fba17e8249dcc761e9c44943eeba8d56c6ade"

PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/macos_cpu_advisory_work \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q7_all_20260509_codex/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Terminal claim row:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id a5_trust_q7_all_macos_cpu_advisory \
  --platform local_macos_cpu \
  --instance-job-id local:a5-trust-q7-all-macos-cpu-20260509T0418Z \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-09T04:30Z \
  --status completed_macos_cpu_advisory_negative \
  --notes "Completed local macOS CPU advisory eval: score=0.202638910574 pose=0.00003563 seg=0.00065288 bytes=177928 archive_sha=39dbfd05d4861c6c5ea12e7bfc8fba17e8249dcc761e9c44943eeba8d56c6ade; non-promotable" \
  --force
```

## Advisory Eval Result

- evidence grade: `[macOS-CPU advisory]`
- canonical score: `0.2026389105740624`
- pose distance: `0.00003563`
- seg distance: `0.00065288`
- rate contribution: `0.118475`
- samples: `600`
- inflate elapsed: `34.79741450003348 s`
- evaluate elapsed: `409.44104937498923 s`

## Interpretation

`q7` for all pairs is the first A5 trust-region geometry that meaningfully
recovers the q6 collapse while still saving bytes:

- scalar q6-low45: `0.21129939214393487`, SegNet `0.00072565`, bytes `178138`
- SegNet-ranked q6-low45: `0.21066071006845696`, SegNet `0.00071899`, bytes
  `178138`
- q7-all: `0.2026389105740624`, SegNet `0.00065288`, bytes `177928`

The rate win is too small to overcome the remaining SegNet loss. Compared with
the A1 Linux CPU anchor (`0.192847577437`, `178262 B`), q7-all saves `334 B`
but loses about `0.00979` score points. The rate gain is only about `0.000222`,
so the distortion miss is roughly `44x` larger than the byte benefit.

Classification: measured-config regression. This retires uniform A5 q7-all as
an exact-eval candidate. It does not kill A5 as a family; it says the current
frame-conditional q-bit sidechannel needs local SegNet-boundary allocation or
must be folded into training, not applied as a post-hoc uniform quantizer.

## Generated Custody Disposition

The packet and auth-eval work directories under `experiments/results/` remain
raw rebuildable custody. They were not promoted as tracked source. The durable
tracked signal is this ledger plus
`reports/a5_score_marginal_trust_region_q7_all_20260509.json`.

`experiments/results/` runtime-source baseline in
`.omx/research/untracked_source_dispositions_20260505_codex.json` was
rebaselined after this packet/eval:

- count: `10585`
- SHA-256: `032778d1eabb5b8960566393fa78dde762cb17391a7dde62c89f82851b2b2d20`

## Reactivation Criteria

- Build a true SegNet-boundary feature map, not per-pair scalar score ranking.
- Make q-bit allocation local within a pair or latent channel, rather than one
  scalar q value per pair.
- Retrain with frame-conditional q-bit/dropout in the loop so the renderer
  learns to place entropy where A5 can remove it.
- Only run paired exact `[contest-CPU]` and `[contest-CUDA]` after local
  advisory is within `0.001` of the A1/PR101 CPU band or a new byte saving is
  large enough to pay for the measured SegNet increase.
