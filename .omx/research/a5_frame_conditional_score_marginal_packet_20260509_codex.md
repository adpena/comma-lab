# A5 Frame-Conditional Score-Marginal Packet Custody

Date: 2026-05-09

Scope: local-only A5 packet hardening. No eval dispatch, no GPU run, no score
promotion.

## What Changed

`tools/build_pr101_frame_conditional_runtime_packet.py` now accepts a strict
`--q-bits-json` input and records q-bit schedule custody in the candidate
archive manifest. The packet builder only clears the
`per_pair_score_marginal_manifest_missing` blocker when the consumed JSON is a
`pr101_a5_per_pair_score_marginals.v1` manifest with `per_pair_q_bits`.

This closes the previous plumbing gap: A5 can now rebuild a byte-closed runtime
packet from a scored/advisory per-pair schedule artifact instead of recomputing
the video-complexity schedule implicitly.

## Commands

```bash
.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --q-bits-json experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
  --output-dir experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex \
  --candidate-id pr101_a5_frame_conditional_runtime_packet_score_marginal \
  --force

.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/packet \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/packet/archive.zip \
  --archive-manifest-json experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/candidate_archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 cde5a1e0ad49ec56856a8b1f0e4d7c329193955211d0f9fe314366d992c4c737 \
  --expected-archive-size-bytes 172615 \
  --expected-runtime-tree-sha256 9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5 \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/pre_submission_compliance.no_auth.json \
  --strict

.venv/bin/python tools/build_pr101_frame_conditional_packet_readiness.py \
  --a5-manifest experiments/results/pr101_frame_conditional_bit_codex_20260508T_wire_contract_smoke/build_manifest.json \
  --candidate-archive-manifest experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/candidate_archive_manifest.json \
  --packet-runtime-patch-manifest experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/packet_runtime_patch_manifest.json \
  --runtime-consumption-proof experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/runtime_consumption_proof.json \
  --per-pair-score-marginal-manifest experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
  --strict-pre-submission-compliance-json experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/pre_submission_compliance.no_auth.json \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/readiness.with_score_marginal_packet.json
```

## Artifacts

The artifacts live under ignored `experiments/results/`; this ledger is the
tracked control-plane pointer.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/candidate_archive_manifest.json` | 30,455 | `3313866bc55147e68a746c6a7f331711080b25d55ee8269b3cbd503227bdb38f` |
| `experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/runtime_consumption_proof.json` | 4,611 | `d19ba39b6fe19f566516a23ed9ac067d6d5b238105010f0d8c6b5bcc2ee07567` |
| `experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/pre_submission_compliance.no_auth.json` | 11,528 | `b627dd0f01ceafc06e8474212ea9975c46397795a4f485d28b0874cab171d670` |
| `experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/readiness.with_score_marginal_packet.json` | 8,564 | `f963ab2c74b1dda85cba23bc96810d555a792a1175950b3f92dcd56ad5f347cc` |

Candidate archive:

- Path: `experiments/results/pr101_frame_conditional_runtime_packet_score_marginal_20260509_codex/packet/archive.zip`
- Bytes: `172,615`
- SHA-256: `cde5a1e0ad49ec56856a8b1f0e4d7c329193955211d0f9fe314366d992c4c737`
- Member `x` SHA-256:
  `cea4e6a10613e6920873cf1bd48a987f05a9bbeb3bb7b6de17f324b81af30569`
- Runtime tree SHA-256:
  `9084d00f236755d00974aa025aee2bbdbac6bfdc5df84094c83eaf28e8afe9d5`

Q-bit schedule custody:

- Source:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json`
- Source SHA-256:
  `6c564c0f75e1fc471f12a34595a8f8a4ae5acf7c437b781056dee7601510ac13`
- Source schema: `pr101_a5_per_pair_score_marginals.v1`
- Source key: `per_pair_q_bits`
- q-bit SHA-256:
  `f84ac40a2ba30b7dc8fc5d765f36dab49918f9439e5106a5ff9553ae3ec60423`
- q-bit side-info SHA-256:
  `2685786664c499e41e768259823c91e8b4164cd95845e5765aa658620fee05d4`
- Distribution: `2:283`, `3:40`, `4:24`, `5:24`, `6:21`, `7:10`, `8:198`

## Readiness Result

- `ready_for_exact_eval_after_lane_claim=true`
- `readiness_blockers=[]`
- Remaining dispatch blockers:
  - `requires_level2_dispatch_claim_before_exact_eval`
  - `requires_exact_cuda_auth_eval_before_score_promotion`
- Strict pre-submission compliance passed in non-final/no-auth mode. It retains
  the expected warning that no auth-eval JSON exists.

## Interpretation

This is a plumbing/custody greenup, not a new score win. The candidate archive
SHA matches the previous A5 runtime packet because the consumed score-marginal
manifest describes the same q-bit schedule. That schedule already has a macOS
CPU advisory collapse (`score=1.937884`, `pose=0.078646`, `seg=0.009361`), so
it should not be dispatched as a frontier candidate unless we deliberately want
a formal exact negative.

The value is that future A5 variants can now emit a scored or scorer-proxy
`per_pair_q_bits` artifact and build a runtime-consumed packet with exact q-bit
custody, no implicit recomputation, and fail-closed readiness checks.

## Tests

```bash
.venv/bin/python -m py_compile tools/build_pr101_frame_conditional_runtime_packet.py
.venv/bin/python -m pytest tests/test_pr101_frame_conditional_runtime_packet_qbits_override.py -q
.venv/bin/python -m pytest src/tac/tests/test_pr101_frame_conditional_runtime_packet.py tests/test_pr101_frame_conditional_runtime_packet_qbits_override.py -q
.venv/bin/python -m pytest src/tac/tests/test_a5_per_pair_score_marginal_manifest.py -q
git diff --check
```

Results:

- `tests/test_pr101_frame_conditional_runtime_packet_qbits_override.py`: `2 passed`
- Runtime packet suite plus override suite: `4 passed`
- Score-marginal manifest suite: `2 passed`

## Reactivation Criteria

Reactivate A5 for exact eval only when one of these is true:

1. A new score-domain q-bit schedule changes the archive SHA and preserves
   packet/runtime consumption proof.
2. A conservative trust-region q-bit schedule reduces byte cost without the
   prior SegNet/PoseNet collapse on advisory CPU.
3. The existing `cde5...` archive is dispatched intentionally as an exact
   negative/control, with lane claim and no promotion semantics.
