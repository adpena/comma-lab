# Codex Findings - PR101 Decoder Blob Materializer

UTC: 2026-06-01T17:03:28Z

## Landing

The PR101 grouped grammar lane now has a byte-closed decoder-section
materializer:

- consumes `pr101_grouped_brotli_packet_grammar.v1` reports;
- rebuilds the grouped decoder blob from report adapter params;
- records decoder blob bytes and SHA-256;
- proves Brotli stream roundtrip;
- proves `decode_decoder_compact(...)` parser roundtrip;
- proves decoded tensors match the quantized state dict exactly;
- keeps the result blocked until archive ZIP splicing, receiver-runtime proof,
  full-frame inflate parity, and exact CPU/CUDA eval exist.

## Authority

This is decoder-section custody, not contest authority. It still reports:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Non-stock grouped layouts additionally require a receiver adapter before
runtime consumption can be claimed.

## Verification

- `uv run ruff check src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py tools/pr101_per_tensor_grammar_solver.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py`
- `uv run pytest src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q`

## Next Step

The next promotion gate is archive-level materialization: splice the decoder
blob into a legal single-member `archive.zip`/`0.bin` packet, prove the runtime
actually consumes the adapter params, and keep all score claims blocked until
full-frame replay and exact auth evidence exist.
