# A5 Frame-Conditional Runtime Packet Scaffold

Date: 2026-05-08

Scope: local-only A5 runtime-consumption scaffold for PR101. No remote dispatch,
no GPU eval, no upstream scorer edits, and no score promotion.

## Command

```bash
.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --a5-manifest experiments/results/pr101_frame_conditional_bit_codex_20260508T_wire_contract_smoke/build_manifest.json \
  --source-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --source-runtime-dir experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec \
  --output-dir experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex \
  --candidate-id pr101_a5_frame_conditional_runtime_packet \
  --force
```

## Artifacts

- Candidate manifest:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/candidate_archive_manifest.json`
  - Bytes: `28920`
  - SHA-256: `d7b18c305adef37c68fd72199e21c7d76e76033078678733f58f40c011d44bbc`
- Candidate archive:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/packet/archive.zip`
  - Bytes: `172615`
  - SHA-256: `cde5a1e0ad49ec56856a8b1f0e4d7c329193955211d0f9fe314366d992c4c737`
  - Member bytes: `172515`
  - Member SHA-256: `cea4e6a10613e6920873cf1bd48a987f05a9bbeb3bb7b6de17f324b81af30569`
- Runtime patch manifest:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/packet_runtime_patch_manifest.json`
  - Bytes: `1879`
  - SHA-256: `759ddfd44939ceebf8dc87d86cd9ea27ae353f92b24901980c09c67f8ea235ac`
- Runtime-consumption proof:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/runtime_consumption_proof.json`
  - Bytes: `4566`
  - SHA-256: `1f8cafce22b3ce35207fd92554ef2591840271415511f60bc7b5e1e4d307b393`
- Readiness with runtime packet artifacts:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/readiness.with_runtime_packet.json`
  - Bytes: `7898`
  - SHA-256: `a11f437840a567719379143caf8043a14e5585999329f1312892ce57e3ce087d`

## Status

- Packet grammar: `A5FC`
- Wire schema consumed: `tac_frame_conditional_latent_wire.v1`
- Runtime patch: packet-local only; public PR101 source tree untouched.
- Runtime proof status: `ready_for_exact_eval_runtime=true`
- Runtime proof blockers: `[]`
- Consumed q-bit side-info SHA-256:
  `2685786664c499e41e768259823c91e8b4164cd95845e5765aa658620fee05d4`
- Consumed variable-width latent payload SHA-256:
  `7b222b242523d7f86f4e4e420da6467f1a7916d91577bf6fa0912a469a3c0f13`
- Negative controls:
  - zeroed side-info rejected:
    `A5 latent bitstream length 9387 != expected 2100`
  - latent-wire bit flip changed decoded latents: `true`

## Readiness

The fail-closed planner accepted these three local artifacts:

1. `candidate_archive_manifest`
2. `packet_local_runtime_patch_manifest`
3. `frame_conditional_runtime_consumption_proof`

Remaining blockers:

1. `missing_per_pair_score_marginal_manifest`
2. `missing_strict_pre_submission_compliance_json`

`score_claim=false`, `dispatch_attempted=false`, and
`ready_for_exact_eval_dispatch=false` throughout.

## Focused Tests

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_frame_conditional_bit_budget.py \
  src/tac/tests/test_pr101_frame_conditional_packet_readiness.py \
  src/tac/tests/test_pr101_frame_conditional_runtime_packet.py
```

Result: `18 passed`.
