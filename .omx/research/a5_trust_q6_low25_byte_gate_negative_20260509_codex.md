# A5 Trust Q6-Low25 Byte-Gate Negative — 2026-05-09

## Classification

Measured-config byte-gate negative. No scorer eval was run.

The `q6` on lowest-25% / `q8` otherwise score-marginal trust schedule is a
tighter follow-up to the q6-low50 macOS CPU advisory negative. It preserves the
typed A5 runtime wire path, but it produces an archive larger than the PR101
brotli source packet, so it is not worth exact CUDA or contest-CPU spend.

## Schedule

Commands:

```bash
.venv/bin/python tools/build_a5_score_marginal_qbits_schedule.py \
  --score-marginal-manifest experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
  --json-out reports/a5_score_marginal_trust_region_q6_low25_20260509.json \
  --candidate-id a5_trust_q6_low25_20260509_codex \
  --base-q-bits 8 \
  --low-q-bits 6 \
  --low-fraction 0.25

.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --q-bits-json reports/a5_score_marginal_trust_region_q6_low25_20260509.json \
  --recompute-wire-contract-for-q-bits \
  --candidate-id pr101_a5_trust_q6_low25_20260509_codex \
  --output-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex \
  --force

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/packet \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/packet/archive.zip \
  --archive-manifest-json experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/candidate_archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 3056e6762a2bd7b5e62158d2d1cb2db7115789c35bbd839811d2cf3e79e9f3a7 \
  --expected-archive-size-bytes 178978 \
  --expected-runtime-tree-sha256 9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5 \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/pre_submission_compliance.no_auth.json \
  --strict

.venv/bin/python tools/build_pr101_frame_conditional_packet_readiness.py \
  --a5-manifest experiments/results/pr101_frame_conditional_bit_codex_20260508T_wire_contract_smoke/build_manifest.json \
  --candidate-archive-manifest experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/candidate_archive_manifest.json \
  --packet-runtime-patch-manifest experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/packet_runtime_patch_manifest.json \
  --runtime-consumption-proof experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/runtime_consumption_proof.json \
  --per-pair-score-marginal-manifest reports/a5_score_marginal_trust_region_q6_low25_20260509.json \
  --strict-pre-submission-compliance-json experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/pre_submission_compliance.no_auth.json \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex/readiness.with_score_marginal_packet.json
```

- Schedule: `reports/a5_score_marginal_trust_region_q6_low25_20260509.json`
- Schedule SHA-256: `84cea8cf2286f7b8ae393ab7f8479cb6c7cf20aa7c8f99deb26fbb8e5415a9ed`
- Schema: `pr101_a5_score_marginal_qbits_schedule.v1`
- Source score-marginal manifest:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json`
- Q-bit distribution: `q6=150`, `q8=450`, mean `7.5`
- Q-bit vector SHA-256: `058ebde683ce1ac23f8b16928b836b11e5a8af8a5b22abd8f50cb6a044b63949`
- Q-bits vs score marginal Pearson: `0.5791130749748051`

## Packet Custody

- Packet output directory:
  `experiments/results/pr101_frame_conditional_runtime_packet_trust_q6_low25_20260509_codex`
- Archive bytes: `178,978`
- Archive SHA-256: `3056e6762a2bd7b5e62158d2d1cb2db7115789c35bbd839811d2cf3e79e9f3a7`
- Single member `x` SHA-256:
  `40e7d219058079d566f8404b79c2c2726010714e1e33013ef5e9d15cea27dba3`
- Runtime tree SHA-256:
  `9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5`
- Q-bit side-info SHA-256:
  `c5b8fa1aa8b5d5e98df34b10af43a97a34017d8352445ccdbe4bae4ab535112e`
- Latent wire payload SHA-256:
  `11986d920a560e67158dc5962b208a23bca64ed10131280bc8e138c9bdde75df`

## Readiness Review

- Candidate manifest SHA-256:
  `908fbf70f3e28294f569dca6c708b899971d6517b615b977916c66c9e308542d`
- Runtime consumption proof SHA-256:
  `98aeff22ca4870dd6c85e2f0a51a946d81aeaf99632acba8c891321232bd435d`
- Packet runtime patch manifest SHA-256:
  `ce59153418612a634cde4642c3a25e2e6eff2361efcc87905f0e88f23e6cc187`
- Strict compliance no-auth JSON SHA-256:
  `0e749f00af8987770b2b261bece9c5b6184a71d71317c620b8f4c3acc900d7b3`
- Readiness JSON SHA-256:
  `4dc7822aabf3964e7b6fda1d3c544142f07d24648513f8e4d328a67549a64998`

Readiness is fail-closed:

```text
ready_for_exact_eval_after_lane_claim = false
readiness_blockers = [
  missing_candidate_archive_manifest,
  candidate_archive_manifest:candidate_archive_bytes_not_below_source,
]
```

All runtime-consumption and score-marginal schedule artifacts validate after
the readiness parser was hardened to accept
`pr101_a5_score_marginal_qbits_schedule.v1` as a derived score-marginal
evidence manifest. The remaining blocker is strictly rate-side: `178,978 B` is
larger than the PR101 brotli target `178,144 B`.

## Disposition

Retire this measured q6-low25 schedule. It is useful because it brackets the A5
trust region:

- q6-low50: byte-smaller (`177,928 B`) but macOS CPU advisory score `0.213365`
  due to SegNet rise.
- q6-low25: likely lower distortion but byte-larger (`178,978 B`) and therefore
  fails before eval.

Reactivation criteria:

1. Component-aware schedule that gets below `178,144 B` while keeping SegNet
   near the PR101/PR107 CPU band.
2. Boundary-aware or score-gradient marginal maps instead of scalar
   complexity/marginal schedules.
3. Runtime-consumed packet that clears local byte gate before any exact CUDA or
   contest-CPU spend.
