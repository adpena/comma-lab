# Codex Findings - PR101 Per-Tensor Grammar Solver

UTC: 2026-06-01T16:49:25Z

## Landing

The new PR101/HNeRV per-tensor grammar solver turns manual decoder-weight packet choices into reusable byte intelligence:

- measures byte-map strategies, 4D storage permutations, Brotli, raw LZMA1, canonical Huffman, and optional PR103 range/AC;
- proves every selected codec branch by exact local encode/decode roundtrip;
- reports empirical Shannon-floor saturation diagnostics per tensor;
- emits planning-only optimizer candidate rows consumable by existing byte-shaving signal surfaces;
- blocks promotion until grouped packet compilation, receiver adapter consumption, full-frame inflate parity, and byte-closed replay exist.

## Authority

This is a codec profiler and planning surface, not a contest candidate. Every queue row stays false-authority:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The intended next materializer step is a grouped PR101 packet compiler only for selected operations whose isolated byte savings survive grouped/window-context measurement.

## Verification

- `.venv/bin/ruff check src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py tools/pr101_per_tensor_grammar_solver.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q`

## Score-Lowering Relevance

The frontier HNeRV grammar is already close to saturated in the current substrate, so this tool is not expected to buy large score movement by itself. Its value is keeping future compact-base and codec substrates from repeating manual packet folklore: every decoder/tensor grammar choice becomes measured, replay-blocked, and planner-consumable before any runtime adapter work is funded.
