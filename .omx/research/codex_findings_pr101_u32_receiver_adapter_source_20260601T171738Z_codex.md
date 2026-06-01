# Codex Findings: PR101 U32 Receiver Adapter Source

UTC: 2026-06-01T17:17:38Z
Author: Codex
Axis: `[receiver-adapter-source-only]`

## Landing

The PR101 grouped grammar lane now emits generated parser adapter source for `u32_decoder_len_adapter` archives. The generated source parses:

```text
uint32_le decoder_section_total_bytes
decoder_blob[4:decoder_section_total_bytes]
latent_blob[decoder_section_total_bytes:decoder_section_total_bytes + 15387]
sidecar_blob[decoder_section_total_bytes + 15387:]
```

It embeds the grouped report's byte-map, storage-order, stream-end, and conv4-perm constants as deterministic JSON and calls the runtime-supplied `decode_decoder_compact`, `decode_latents_compact`, and `apply_latent_sidecar` functions. This avoids duplicating codec code while making the receiver-adapter contract executable and testable.

## Local Proof

Focused test:

```text
uv run pytest src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q
15 passed in 2.23s
```

Lint:

```text
uv run ruff check src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py tools/pr101_per_tensor_grammar_solver.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py
All checks passed!
```

CLI smoke used `/Volumes/VertigoDataTier/pact/pr101_adapter_cli_smoke_20260601T171738Z`, emitted report/grouped report/adapter source/proof, byte-compiled the generated source, printed the proof summary, and deleted the scratch tree.

Smoke result:

```text
grouped selected bytes=198539; current grouped bytes=198941; saved=402; runtime=tac_decode_decoder_compact_with_overrides_required
adapter_bytes 2956
adapter_sha 2898cf98373fd656ef70f43dc930747926848f280fbeff4e189f6968333f8eef
status u32_decoder_len_adapter_source_emitted
blockers inflate_sh_integration_missing,full_frame_inflate_parity_missing,contest_cpu_cuda_exact_eval_not_executed
```

## Verdict

The receiver-adapter source blocker is narrowed to submission-runtime integration. The remaining exact-readiness sequence is: vendor this adapter into an `inflate.py` tree, run `inflate.sh` full-frame replay, then exact CPU/CUDA only if local replay wins.
