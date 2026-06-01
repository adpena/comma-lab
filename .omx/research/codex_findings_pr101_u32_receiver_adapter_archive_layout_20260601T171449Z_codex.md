# Codex Findings: PR101 U32 Decoder-Length Archive Layout

UTC: 2026-06-01T17:14:49Z
Author: Codex
Axis: `[archive-zip-materialization-only]`

## Landing

The PR101 grouped archive materializer now supports two explicit archive layouts:

- `fixed_pr101`: preserves PR101's original fixed decoder offset contract.
- `u32_decoder_len_adapter`: emits `uint32_le decoder_section_total` followed by the grouped decoder blob, then the preserved PR101 latent and sidecar blobs.

This makes variable-length grouped decoder blobs parser-addressable without corrupting the latent offset. The materializer still blocks exact readiness because the contest runtime source that consumes this layout and any non-stock codec constants has not been emitted.

## Local Proof

Focused test:

```text
uv run pytest src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q
14 passed in 2.12s
```

Lint:

```text
uv run ruff check src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py tools/pr101_per_tensor_grammar_solver.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py
All checks passed!
```

CLI smoke used `/Volumes/VertigoDataTier/pact/pr101_u32_archive_cli_smoke_20260601T171449Z`, emitted report/grouped report/archive ZIP/proof, printed the proof summary, and deleted the scratch tree.

Smoke result:

```text
grouped selected bytes=201178; current grouped bytes=201511; saved=333; runtime=tac_decode_decoder_compact_with_overrides_required
layout u32_decoder_len_adapter
archive_bytes 217276
decoder_bytes 201178
section_total 201182
u32_parse_safe True
blockers full_frame_inflate_parity_missing,contest_cpu_cuda_exact_eval_not_executed,receiver_runtime_source_not_emitted,receiver_codec_constants_override_source_not_emitted
```

## Verdict

This closes the parser-offset corruption blocker for variable-length grouped PR101 decoder blobs. Remaining blocker is now narrower and actionable: emit the actual receiver runtime source that reads the `u32_decoder_len_adapter` layout and applies emitted codec constants.
