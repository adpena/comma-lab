# PR106 HDM11/HLM3 Magicless PacketIR Format 0x0B

Date: 2026-05-15
Agent: codex
Axis: no-score local proof packet; exact CUDA pending

## Summary

Implemented PR106 sidecar `format_id=0x0B`, a runtime-supported PacketIR recode
that starts from the exact-CUDA-reviewed `format_id=0x0A` HDM9/HLM3 packet and
elides the fixed `HDM9` decoder magic plus fixed `HLM3` latent magic. The
submission runtime reconstructs those constants from committed grammar and fixed
section lengths.

This is a rate-only transform. It does not claim score movement until exact
CUDA auth eval returns.

## Candidate

- Archive:
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip`
- Archive SHA-256:
  `5c9ef623a089893d6f2dd13c0417149b86a6bdfe52b1f472988e73bd2cfddc4d`
- Archive bytes: `186341`
- Source exact-CUDA reference: format `0x0A`, archive SHA
  `186a3d59f2038be61bfda7aa97cdc7abcf970ce4f2d20cd84d42386e894d2ce7`,
  bytes `186349`, score `[contest-CUDA] 0.2063310355127786`
- Byte delta vs `0x0A`: `-8`
- Expected pure-rate score delta if components match:
  `-0.000005326871624977371`
- Expected score if exact CUDA components match `0x0A`:
  `0.20632570864115363`

## Local Proofs

- Runtime consumption proof:
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/runtime_consumption_hdm11_magicless.json`
- Runtime consumption blockers: `[]`
- Runtime source tree SHA-256:
  `8ddce462a0d300f29ff9dddca8683cbe08bf97fc7959c296e06647ef12d4249b`
- Uploaded Modal runtime tree SHA-256 expectation:
  `d82027132eeb2801bfab72d94ac003a291e6716b907d5d0cdc7e89e43de720b8`
- Uploaded Modal runtime content tree SHA-256 expectation:
  `a5d2a8a5abd9844d277533f6321b6ad30e6c24bc7a61e3d124cb0e3f07a73191`

- Same-runtime full-frame parity proof:
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/same_runtime_full_frame_parity_hdm10_vs_hdm11.json`
- Parity scope: full local CPU streaming runtime, all 600 pairs
- Source format: `0x0A`
- Candidate format: `0x0B`
- Shared streaming raw SHA-256:
  `e6d9170b92997db1e6211eeb665187a3ac6a23c743dd3659c46633e509af329f`
- Total frames: `1200`
- Total bytes hashed: `3662409600`

Profile row after proofs:

```json
{
  "sidecar_format_id": "0x0B",
  "emitted_candidate_archive_bytes": 186341,
  "emitted_candidate_archive_sha256": "5c9ef623a089893d6f2dd13c0417149b86a6bdfe52b1f472988e73bd2cfddc4d",
  "runtime_consumption_claim": true,
  "full_frame_inflate_output_parity_claim": true,
  "candidate_exact_eval_blockers": [
    "exact_cuda_auth_eval_missing",
    "contest_auth_eval_adjudication_missing"
  ],
  "score_claim": false,
  "ready_for_exact_eval_dispatch": false
}
```

## Commands

Profile and emit archive:

```bash
.venv/bin/python tools/profile_pr106_latent_sidecar_recode.py \
  --sidecar-archive experiments/results/pr106_hdm9_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x09.archive.zip \
  --json-out experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/profile.with_proofs.json \
  --md-out experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/profile.with_proofs.md \
  --emit-runtime-candidates-dir experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates \
  --runtime-consumption-proof experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/runtime_consumption_hdm11_magicless.json \
  --same-runtime-full-frame-parity experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/same_runtime_full_frame_parity_hdm10_vs_hdm11.json
```

Runtime consumption:

```bash
.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --output-json experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/runtime_consumption_hdm11_magicless.json
```

Full-frame parity:

```bash
.venv/bin/python tools/prove_pr106_same_runtime_frame_parity.py \
  --source-archive experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_inner_headerless_fixed_meta_noop_rank_elided_sidecar_format_0x0a.archive.zip \
  --candidate-archive experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --device cpu \
  --batch-pairs 16 \
  --output-json experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/same_runtime_full_frame_parity_hdm10_vs_hdm11.json
```

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -p no:cacheprovider -q \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py \
  src/tac/tests/test_packetir_exact_closure.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Result: `61 passed in 1.91s`

After adversarial review, the parser was hardened against same-length random
payload false-positives, bogus `0xff` headerless packets, and off-by-one
magicless payloads. Full-frame parity proof now records the full runtime source
manifest, and exact closure requires parity/runtime-consumption/CUDA runtime
files to match.

Re-run result: `64 passed in 1.89s`

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packet_compiler/__init__.py \
  src/tac/packetir_exact_closure.py \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py \
  src/tac/tests/test_packetir_exact_closure.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Result: `All checks passed!`

```bash
git diff --check
```

Result: clean.

## Exact-Eval Gate

Next action is claimed Modal T4 exact CUDA eval for:

- Lane: `lane_pr106_hdm11_hlm3_magicless_packetir_format0b_20260515`
- Job id: `pr106_hdm11_hlm3_fmt0b_t4_20260515T082000Z` or newer timestamp
- Archive SHA-256:
  `5c9ef623a089893d6f2dd13c0417149b86a6bdfe52b1f472988e73bd2cfddc4d`
- Expected uploaded runtime tree SHA-256:
  `d82027132eeb2801bfab72d94ac003a291e6716b907d5d0cdc7e89e43de720b8`

No promotion, leaderboard, or score-lowering claim should be made until the
exact CUDA result review exists and binds archive, runtime tree, component
distances, and dispatch claim.
