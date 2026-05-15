# PR106 Fixed-Meta Rank-Elided PacketIR Closure - 2026-05-14

## Scope

This is a byte-grammar cleanup artifact, not a frontier score claim. It closes
the interrupted PR106/R2 PR101-sidecar format `0x05` work so follow-on work can
move to divergence lanes without leaving a half-supported archive grammar in
the tree.

## Artifact Custody

- Source archive:
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`
- Source archive bytes: `186780`
- Source archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- Candidate archive:
  `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/archive.zip`
- Candidate archive bytes: `186771`
- Candidate archive SHA-256:
  `c9d117fd6e9b5ef3fe7c1165d724c572560bfbc7f3b4cb8cc485b9bb4c612af5`
- Rate-only byte delta versus format `0x02`: `-9` bytes

Ignored artifact manifests preserved locally:

- `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/packetir_manifest.json`
  SHA-256 `c1cf2f65a9c54f34afb211b026515132785716fcf780801bee69282349645ddd`
- `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/runtime_consumption.json`
  SHA-256 `0a9a61eb0dd0aa03f81475a1149469224ce18ef94b08aed08e7114262ec03e39`
- `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/prefix_parity_cpu_1pair.json`
  SHA-256 `e54f6ceee07e6125606a096c17a6d1b2a2207c4c4e359c39457845cf45f4fdbb`

## Evidence Axis

- `[packet-ir-parser-local-no-score]`: parse/emit identity and consumed-byte
  accounting for format `0x05`.
- `[runtime-sidecar-decode-local-no-score]`: runtime parser/decoder consumes
  format `0x05` and the semantic mutation changes corrected latents.
- `[macOS-CPU prefix parity no-score]`: one-pair same-runtime raw-output prefix
  parity versus the source `0x02` archive.
- `[local-cpu-streaming-runtime full-frame parity no-score]`: same-runtime
  full-frame streaming hash over all 600 pairs / 1200 frames matches the source
  `0x02` archive exactly:
  `30bc014709737aa0a17aaef525183b13fefa6531ec3410ab84d91dc1199c387b`.
- `[contest-CUDA]`: recovered Modal T4 auth eval for the `0x05` candidate.

No `[contest-CPU]` score is claimed from this artifact. The exact-CUDA result
below is a valid raw auth-eval score claim, but it is not promotion-ready because
pre-submission compliance/adjudication gates are not recorded.

## Exact CUDA Recovery

Full-frame parity proof:

- JSON:
  `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/full_frame_parity_cpu.json`
- source format: `0x02`
- candidate format: `0x05`
- source/candidate full-frame digest: `true`
- pairs hashed: `600`
- frames hashed: `1200`
- total bytes hashed per side: `3662409600`
- streaming raw SHA-256:
  `30bc014709737aa0a17aaef525183b13fefa6531ec3410ab84d91dc1199c387b`
- `full_frame_inflate_output_parity_claim`: `true`

Exact CUDA dispatch:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval.py \
  --archive experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/archive.zip \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --inflate-sh inflate.sh \
  --gpu T4 \
  --output-dir experiments/results/modal_auth_eval/pr106_fixed_meta_rank_elided_exact_cuda_20260514T2359Z \
  --expected-runtime-tree-sha256 0ec752724dde95867928cdf30c50db5eccc2d120054ec22529d0754a38a8ce5d \
  --detach \
  --provider-detach-ack \
  --lane-id pr106_fixed_meta_rank_elided_exact_cuda_20260514 \
  --instance-job-id modal_pr106_fixed_meta_rank_elided_cuda_20260514T2359Z
```

Recovered artifact:

- output dir:
  `experiments/results/modal_auth_eval/pr106_fixed_meta_rank_elided_exact_cuda_20260514T2359Z`
- Modal call id: `fc-01KRMEZF31Y2Z1XXYFD5EE73C0`
- terminal lane-claim status:
  `completed_contest_cuda_modal_auth_eval_recovered`
- result JSON:
  `experiments/results/modal_auth_eval/pr106_fixed_meta_rank_elided_exact_cuda_20260514T2359Z/modal_cuda_auth_eval_result.json`
- contest JSON:
  `experiments/results/modal_auth_eval/pr106_fixed_meta_rank_elided_exact_cuda_20260514T2359Z/contest_auth_eval.json`

Exact CUDA result:

- evidence grade: `[contest-CUDA]`
- archive bytes: `186771`
- archive SHA-256:
  `c9d117fd6e9b5ef3fe7c1165d724c572560bfbc7f3b4cb8cc485b9bb4c612af5`
- runtime tree SHA-256:
  `0ec752724dde95867928cdf30c50db5eccc2d120054ec22529d0754a38a8ce5d`
- samples: `600`
- avg SegNet distance: `0.0006426`
- avg PoseNet distance: `0.00003236`
- canonical score: `0.20661202799099615`
- score claim: `true`
- score claim valid: `true`
- promotion eligible: `false`
- rank/kill eligible: `false`

Comparison:

- Source `0x02` PR106/R2 exact-CUDA score:
  `0.20661813545741509` at `186780` bytes.
- Previous `0x04` rank-elided exact-CUDA score:
  `0.20661535728576175` at `186776` bytes.
- New `0x05` exact-CUDA score:
  `0.20661202799099615` at `186771` bytes.

Classification: legitimate rate-only PacketIR cleanup win inside the PR106/R2
basin. This does not address the PR101/FEC3 `0.19209788683213053`
`[contest-CPU]` near-miss and does not change the need for CUDA-safe selector
or representation-divergence work.

## Test Evidence

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py -q
# 40 passed

PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_build_pr106_sidecar_rank_elided_candidate.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py -q
# 12 passed
```

## Classification

Legitimate byte-grammar cleanup, not a dispatch priority. The main score-lowering
path remains divergence work: Z3 v2 latent replacement, C1/Z5 predictive receiver
surfaces, C6 executable campaign repair, Tier-C real-scorer MDL, and exact
CUDA/CPU axis-labelled candidate evaluation.
