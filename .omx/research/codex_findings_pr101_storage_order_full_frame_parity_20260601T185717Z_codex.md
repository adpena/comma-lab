# PR101 optimal grammar storage-order search + full-frame parity

- **Date:** 2026-06-01T18:57:17Z
- **Axis:** `[planning-only byte-profile]` plus shell-inflate full-sample parity; **no score claim**
- **Commit owner:** codex

## Landing

`tac.packet_compiler.pr101_per_tensor_grammar_solver` now treats grouped
decoder section order as a deterministic grammar search axis instead of a
manual PR101 constant.

Implemented order candidates:

- `pr101`: existing hand order.
- `identity`: schema order.
- `size_desc`: largest transformed tensor payloads first.
- `histogram_greedy`: byte-distribution nearest-neighbor order.
- `best-of-builtins`: measures all of the above and selects the smallest real
  grouped Brotli packet.

The queue and campaign consumer preserve the selected order label/candidates,
so this signal is planner-consumable instead of report-only. Campaign summaries
can now consume `shell_inflate_parity_proof_v2` and escalate only blocker-free,
contest-full-sample, byte-identical archive wins to the exact CPU auth gate.

## Real PR101 artifact

Root:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_storage_order_20260601T185043Z`

Key artifacts:

- `grouped_report.json`
  - SHA-256: `8f1f546bba6a3a502c0a8f5d6555da6f4fca22bfdba7e12e4a0b24145f4f11b2`
- `archive.zip`
  - SHA-256: `47217085a97d19f3b783106f3f4ba5390658ef5f7558d93769116b6d78ab1801`
- `campaign_summary_with_shell_parity.json`
- `campaign_consumer_result_with_shell_parity.json`
- `shell_inflate_parity_full600/shell_inflate_parity.json`

Measured result:

- Selected storage order: `size_desc`
- Grouped decoder bytes: `162152`
- Current stock PR101 grouped bytes: `162164`
- Grouped delta: `-12` bytes
- Legal len24 archive bytes: `178249`
- Source PR101 archive bytes: `178258`
- Archive delta: `-9` bytes
- Rate score delta if components unchanged: `-0.0000059927305780995425`

Full-shell parity:

- Scope: `contest_full_sample`
- File list: `0.mkv`
- Raw output bytes per side: `3662409600`
- Raw output SHA-256 both sides:
  `e63942793f963fa1e0f1ab195f9819519d8f63c067f9959cb5efc7879a4ef386`
- `cmp_equal=true`
- `full_frame_inflate_output_parity_claim=true`
- Scratch retained: `false`

Planner result:

- Campaign verdict: `grouped_positive_full_frame_parity_passed_exact_auth_gate`
- Consumer action: `queue_exact_cpu_auth_eval_after_lane_claim`
- Remaining blocker: `contest_cpu_cuda_exact_eval_not_executed`

## Interpretation

This does not overturn the earlier PR101 grammar saturation finding: the win is
9 archive bytes, not a frontier-scale lever. It does close the objective's
section-order gap and proves the automated grammar solver can find a legal,
receiver-consumed, full-frame-identical archive improvement without manual
section-order hand tuning. The bigger payoff remains future unsaturated
substrates where grammar is not already near the floor.

## Verification

```bash
uv run ruff check \
  src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py \
  tools/pr101_per_tensor_grammar_solver.py \
  src/tac/cathedral_consumers/pr101_optimal_grammar_campaign_consumer/__init__.py \
  src/tac/tests/test_pr101_per_tensor_grammar_solver.py \
  src/tac/tests/test_pr101_optimal_grammar_campaign_consumer.py

uv run pytest \
  src/tac/tests/test_pr101_per_tensor_grammar_solver.py \
  src/tac/tests/test_pr101_optimal_grammar_campaign_consumer.py -q
```

Result: `31 passed`; ruff clean.
