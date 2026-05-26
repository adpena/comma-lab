# Codex Findings — Receiver-Closed Rate Packet Context Propagation

**UTC:** 2026-05-26T20:17:27Z
**Lane:** `lane_fec8_rate_packet_budget_bridge_20260526`
**Scope:** FEC8/FEC6 rate-packet repair-budget bridge, targeted component correction materializer chain
**Authority:** false-authority only; no score claim, no promotion, no dispatch authority

## What changed

The receiver-closed FEC8-vs-FEC6 rate-packet context now survives the full local
repair planning chain instead of terminating at the targeted acquisition row.
The propagated context includes:

- `rate_packet_manifest_path`
- `parent_rate_packet_manifest_path`
- `candidate_compact_selector_codec`
- `parent_compact_selector_codec`
- selector wire/code-bit deltas
- palette metadata
- entropy-position label `at_entropy_coder_integer_codeword_boundary`
- receiver-closed saved-byte budget

The context is carried through:

1. targeted component correction acquisition rows;
2. component-correction work orders;
3. local CPU/MLX response harvest rows;
4. grouped materialization requests;
5. operation-chain compiler work orders;
6. targeted chain materializer handoff `operation_params`.

## Why this matters

The repair allocator can now distinguish "10 bytes saved by FEC8 static
second-order Markov K16 versus FEC6 fixed Huffman K16 at the entropy-coder
integer-codeword boundary" from a generic freed-byte budget. That preserves the
information-theoretic position of the transformation for downstream
waterfilling: before-entropy, at-entropy, and after-entropy opportunities should
not be pooled as interchangeable bytes.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
  - `44 passed in 49.64s`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `OK — 1421 lane(s) validated cleanly.`
- `.venv/bin/python tools/review_tracker.py policy-check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py`
  - `0 violations`

## Remaining blocker

This is still local planning and materializer-context propagation. Budget spend,
promotion, rank/kill, and score authority remain blocked until receiver-consumed
materialization, full-frame inflate parity, SegNet/PoseNet component replay, and
contest-axis auth eval are present.
