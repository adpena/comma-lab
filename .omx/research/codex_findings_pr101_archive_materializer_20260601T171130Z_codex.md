# Codex Findings: PR101 Grouped Archive Materializer

UTC: 2026-06-01T17:11:30Z
Author: Codex
Axis: `[archive-zip-materialization-only]`

## Landing

`tac.packet_compiler.pr101_per_tensor_grammar_solver` now has a receiver-fail-closed archive materializer for grouped PR101 decoder-blob reports:

- builds a deterministic single-member stored ZIP archive with member `x`;
- preserves the source PR101 latent blob and sidecar blob byte-for-byte;
- records source/output ZIP, inner member, decoder, latent, and sidecar SHA-256s;
- proves ZIP roundtrip, stored method, empty extras/comments, deterministic timestamp, and decoder materialization proof inheritance;
- blocks exact readiness unless the decoder blob is stock-runtime fixed-offset safe.

This turns the previous section-only decoder blob proof into a byte-closed archive artifact while preserving the correct refusal boundary: variable-length grouped decoder blobs are not stock PR101 runtime consumable until a receiver adapter exists.

## Local Proof

Focused test:

```text
uv run pytest src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q
13 passed in 1.77s
```

Lint:

```text
uv run ruff check src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py tools/pr101_per_tensor_grammar_solver.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py
All checks passed!
```

CLI smoke used `/Volumes/VertigoDataTier/pact/pr101_archive_cli_smoke_20260601T171130Z`, emitted report/grouped report/decoder blob/archive ZIP/proofs, printed the proof summary, and deleted the scratch tree after verification.

Smoke result:

```text
grouped selected bytes=202300; current grouped bytes=202753; saved=453; runtime=tac_decode_decoder_compact_with_overrides_required
archive_bytes 218394
decoder_bytes 202300
delta_bytes 40136
fixed_offset_safe False
blockers full_frame_inflate_parity_missing,contest_cpu_cuda_exact_eval_not_executed,stock_runtime_fixed_offset_decoder_blob_length_mismatch,receiver_adapter_not_emitted,runtime_consumption_proof_missing
zip_member x 0 218294 218294 (1980, 1, 1, 0, 0, 0)
```

## Verdict

The materializer is useful for queue-owned byte closure and future receiver-adapter work, but the tested grouped win is still not exact-ready because it changes decoder length and requires adapter/runtime consumption proof. This is the correct fail-closed state.

## Next Step

Add the receiver-adapter path for variable-length decoder sections or constrain the grouped solver to stock fixed-length substitutions only. Only the former can preserve the measured grouped-byte win; the latter is mostly a runtime-safety gate.
