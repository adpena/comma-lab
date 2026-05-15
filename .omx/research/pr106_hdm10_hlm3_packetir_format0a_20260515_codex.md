# PR106 HDM9/HLM3 PacketIR format0A byte-closed recode

Date: 2026-05-15

score_claim: false
ready_for_exact_eval_dispatch: false
dispatch_attempted: false

## Summary

Landed a PR106/R2 PacketIR continuation after HDM9. Format `0x0A` keeps the
HDM9 decoder stream, recodes the fixed HLM2 latent payload as HLM3 by deriving
the low-byte Brotli length from the Brotli stream boundary, and elides the
fixed zero no-op rank byte from the PR101 sidecar grammar.

This is a byte-closed parser/runtime improvement, not a contest score claim.

## Artifacts

- source format09 archive:
  `experiments/results/pr106_hdm9_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x09.archive.zip`
- source bytes: `186352`
- source SHA-256:
  `09bcd867c2778d38d5ac04b648d44cb3bdfcfd3e3db402beb8886826cced50e9`
- candidate format0A archive:
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_inner_headerless_fixed_meta_noop_rank_elided_sidecar_format_0x0a.archive.zip`
- candidate bytes: `186349`
- candidate SHA-256:
  `186a3d59f2038be61bfda7aa97cdc7abcf970ce4f2d20cd84d42386e894d2ce7`
- archive delta vs format09: `-3` bytes
- pure-rate delta vs format09 if exact CUDA components match:
  `-0.000001997576859366514`
- inferred rate-only score from HDM9 exact CUDA if exact CUDA components match:
  `0.2063310355127786`

The inferred score is not promoted because no exact CUDA auth eval was
dispatched for the format0A archive.

## Proofs

- profile:
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/profile.with_proofs.json`
- runtime consumption:
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/runtime_consumption_hdm10_hlm3.json`
- same-runtime prefix parity:
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/same_runtime_prefix_parity_hdm9_vs_hdm10.json`
- same-runtime full-frame parity:
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/same_runtime_full_frame_parity_hdm9_vs_hdm10.json`

Runtime consumption result:

- `format_id=0x0A`
- `blockers=[]`
- `runtime_sidecar_decode_consumption_claim=true`
- `runtime_sidecar_apply_consumption_claim=true`
- `runtime_all_score_affecting_sections_consumed=true`
- `score_claim=false`
- `promotion_eligible=false`

Same-runtime full-frame parity result:

- device axis: `local-cpu-streaming-runtime`
- source format: `0x09`
- candidate format: `0x0A`
- pairs hashed: `600`
- frames hashed: `1200`
- bytes hashed: `3662409600`
- streaming SHA-256:
  `e6d9170b92997db1e6211eeb665187a3ac6a23c743dd3659c46633e509af329f`
- `streaming_output_sha256_equal=true`
- `streaming_output_total_bytes_equal=true`
- `full_frame_inflate_output_parity_claim=true`
- `score_claim=false`

## Code surfaces

- `src/tac/packet_compiler/pr106_sidecar_packet.py`
  - adds format `0x0A` parser/emitter/proof/manifest support;
  - adds HLM3 encode/decode proof helpers;
  - adds no-op-rank-elided PR101 sidecar re-expansion;
  - adds HDM8/HDM9 HLM2 to HDM9/HLM3 recode.
- `submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py`
  - adds runtime HLM3 latent decode.
- `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py`
  - adds runtime format `0x0A` parse and sidecar decode.
- `src/tac/packet_compiler/pr106_runtime_consumption.py`
  - adds runtime consumption proof support for format `0x0A`.
- `src/tac/packetir_exact_closure.py`
  - adds format `0x0A` score-affecting section alias.
- `tools/profile_pr106_latent_sidecar_recode.py`
  - includes format `0x0A` in emitted runtime candidate profiling.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py \
  submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packetir_exact_closure.py \
  tools/profile_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/packet_compiler/__init__.py
```

Observed: `All checks passed`.

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_pr106_hdm9_decoder_recode.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packetir_exact_closure.py -q
```

Observed: `45 passed`.

Additional regression slice:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_submission_pr101_decoder_adapter.py \
  src/tac/tests/test_pr106_fixed_latent_recode.py -q
```

Observed: `69 passed`.

## Dispatch decision

No CUDA dispatch was launched for format0A in this turn. The archive is
full-frame identical to HDM9 under the same runtime and saves only three bytes,
so a paid exact CUDA run is low EV unless it is bundled with another PacketIR
candidate batch. Promotion still requires exact contest auth eval with explicit
`[contest-CUDA]` or `[contest-CPU]` axis labels.
