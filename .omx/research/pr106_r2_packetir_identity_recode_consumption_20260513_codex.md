# PR106/R2 PacketIR Identity, Runtime Consumption, And Sidecar Recode Closure

Date: 2026-05-13
Author: Codex
Evidence grade: byte-closed planning proof, no score claim

## Scope

This pass refreshes the PR106/R2 PacketIR evidence on current `main` for the
short-term objective: parse/re-emit identity, consumed-byte proof, runtime
sidecar decode/apply proof, and lossless sidecar recompression candidates.

No provider job was launched. No auth eval was run. No score, promotion, or
dispatch-readiness claim is made from this pass.

## Artifacts

Root:
`experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/`

Original PR106/R2 sidecar packet:

- source archive:
  `submissions/pr106_latent_sidecar_r2/archive.zip`
- archive bytes: `186822`
- archive sha256:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- member: `0.bin`, stored, payload bytes `186714`
- PacketIR identity proof:
  `.../original_r2/packetir_identity.json`
- runtime sidecar consumption proof:
  `.../original_r2/runtime_consumption.json`
- recode profile:
  `.../original_r2/recode_profile.json`
- emitted runtime-supported candidate manifests:
  `.../original_r2/runtime_candidates/*.manifest.json`

PR106/R2 PR101-grammar sidecar packet:

- source archive:
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`
- archive bytes: `186780`
- archive sha256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- member: `0.bin`, stored, payload bytes `186672`
- PacketIR identity proof:
  `.../pr101_grammar/packetir_identity.json`
- runtime sidecar consumption proof:
  `.../pr101_grammar/runtime_consumption.json`
- recode profile:
  `.../pr101_grammar/recode_profile.json`
- emitted runtime-supported candidate manifests:
  `.../pr101_grammar/runtime_candidates/*.manifest.json`

## Findings

### Parse/Re-Emit Identity

Both source archives pass PacketIR identity:

- `packet_ir_identity_passed=true`
- emitted payload is byte-identical to the source member
- emitted archive is byte-identical to the source archive
- parser consumed-byte proof accounts for every payload byte
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

This proves the wrapper grammar is typed and deterministic, but it is not a
runtime/full-frame/eval proof by itself.

### Runtime Sidecar Consumption

Both source archives pass runtime sidecar decode/apply consumption:

- original R2 runtime `inflate.py` sha256:
  `093d20785ff29759ac835f01efc3caa2210d31b11b5b3874256be774bc22a6db`
- PR101-grammar runtime `inflate.py` sha256:
  `5fa58d34dbf195e960d9a3db6370bf238c1e4e459de6cf3f11487b0a1f4b272f`
- `runtime_sidecar_decode_consumption_claim=true`
- `runtime_sidecar_apply_consumption_claim=true`
- `runtime_corrected_latents_digest_changed=true`
- mutated PacketIR consumed-byte proof still has
  `all_payload_bytes_accounted=true`

This proves valid sidecar mutations are consumed by the submission runtime's
sidecar parser/decoder and change corrected latents. It does not claim
full-frame parity or exact score movement.

### Lossless Sidecar Recode

For the original PR106/R2 brotli sidecar:

| candidate | runtime decoder | charged sidecar bytes | delta vs current | rate score delta if consumed |
|---|---|---:|---:|---:|
| `pr101_ranked_no_op_sidecar_format_0x02` | true | `533` | `-42` | `-0.000027966076031131196` |
| `vocab_bitpack_dim_delta_raw` | false | `539` | `-36` | `-0.00002397092231239817` |
| `vocab_bitpack_dim_delta_brotli_q11` | false | `547` | `-28` | `-0.000018644050687420796` |
| `current_pr100_dim_delta_brotli_q11` | true | `575` | `0` | `0.0` |

The runtime-supported best lossless recode is the PR101-ranked sidecar format,
which emits the canonical PR101-grammar archive:

- emitted archive bytes: `186780`
- emitted archive sha256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`

For the PR101-grammar source packet, lossless sidecar grammar is saturated:

| candidate | runtime decoder | charged sidecar bytes | delta vs current |
|---|---|---:|---:|
| `pr101_ranked_no_op_sidecar_format_0x02` | true | `533` | `0` |
| `vocab_bitpack_dim_delta_raw` | false | `539` | `+6` |
| `vocab_bitpack_dim_delta_brotli_q11` | false | `547` | `+14` |
| `current_pr100_dim_delta_brotli_q11` | true | `575` | `+42` |

Conclusion: pure lossless sidecar grammar work has no further byte win over the
PR101-ranked format for this semantic correction table. Future PR106/R2
score-lowering should move to semantic sidecar search, inner PR106 payload /
decoder recode, or composition atoms rather than more generic sidecar
compression.

## Tool Hardening

`tools/profile_pr106_latent_sidecar_recode.py` previously kept
`no_candidate_archive_emitted` in `dispatch_blockers` even when
`--emit-runtime-candidates-dir` emitted byte-closed candidate archives and
manifests. This was misleading for operator routing.

This pass changes the blocker to be conditional:

- without emitted archives: `no_candidate_archive_emitted` remains present
- with emitted runtime candidates: the blocker is removed, while real blockers
  remain:
  - `candidate_runtime_decoder_missing_for_noncurrent_rows`
  - `missing_no_op_runtime_consumption_proof_for_new_grammar`
  - `missing_exact_contest_eval_for_any_candidate`

Tests now cover both modes.

## Commands

```bash
.venv/bin/python tools/prove_pr106_packetir_identity.py \
  --archive submissions/pr106_latent_sidecar_r2/archive.zip \
  --expected-archive-sha256 7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f \
  --output-json experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/original_r2/packetir_identity.json

.venv/bin/python tools/prove_pr106_packetir_identity.py \
  --archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip \
  --expected-archive-sha256 c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383 \
  --output-json experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/pr101_grammar/packetir_identity.json

.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive submissions/pr106_latent_sidecar_r2/archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2 \
  --output-json experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/original_r2/runtime_consumption.json

.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --output-json experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/pr101_grammar/runtime_consumption.json

.venv/bin/python tools/profile_pr106_latent_sidecar_recode.py \
  --sidecar-archive submissions/pr106_latent_sidecar_r2/archive.zip \
  --json-out experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/original_r2/recode_profile.json \
  --md-out experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/original_r2/recode_profile.md \
  --emit-runtime-candidates-dir experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/original_r2/runtime_candidates

.venv/bin/python tools/profile_pr106_latent_sidecar_recode.py \
  --sidecar-archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip \
  --json-out experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/pr101_grammar/recode_profile.json \
  --md-out experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/pr101_grammar/recode_profile.md \
  --emit-runtime-candidates-dir experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/pr101_grammar/runtime_candidates
```

Focused verification:

```bash
.venv/bin/ruff check \
  tools/profile_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py

PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -q \
  src/tac/tests/test_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py
```

Result: `36 passed`.

## Next Work

1. Do not spend more time on lossless PR106/R2 sidecar grammar compression
   unless the semantic correction table changes. The PR101-ranked grammar is
   already the best runtime-supported lossless sidecar grammar here.
2. Prioritize PR106/R2 inner payload / decoder structural recode and semantic
   sidecar search, with the same PacketIR consumed-byte and runtime-consumption
   gates.
3. If a new semantic sidecar candidate is produced, rerun this exact bundle
   before any exact eval claim.
4. Keep any future exact eval axis-labelled; this memo is neither
   `[contest-CUDA]` nor `[contest-CPU]`.

