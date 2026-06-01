# Codex Findings - PR101 Grouped Packet Grammar Solver

UTC: 2026-06-01T16:57:31Z

## Landing

The PR101 grammar solver now prices the grouped split-Brotli packet layer, not just isolated tensor payloads:

- builds transformed tensor payloads once, then concatenates in storage order;
- solves Brotli stream split points by dynamic programming over real compressed byte counts;
- measures grouped bytes against current stock PR101 grouped bytes;
- proves concatenated stream roundtrip and, for full-schema runs, parser roundtrip through `decode_decoder_compact` with explicit adapter params;
- emits a grouped planning-only optimizer queue with candidate saved bytes only when grouped savings survive.

## Authority

This is still planning-only packet intelligence. It blocks on byte-closed archive materialization, receiver adapter proof for non-stock layouts, full-frame inflate parity, and exact CPU/CUDA eval before any score or promotion claim.

## Verification

- `.venv/bin/ruff check src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py tools/pr101_per_tensor_grammar_solver.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q`

## Score-Lowering Relevance

This closes the first false optimism gap in per-tensor codec work: isolated tensor savings can disappear once Brotli context and stream boundaries are priced. Future PR101/HNeRV/RNeRV packet changes should spend runtime-adapter effort only on grouped-positive rows, not on isolated wins.
