# PR101 Seeded Selector Adapter Probe - 2026-05-20T23:20Z

## Verdict

Built a byte-closed seeded-selector feasibility probe for the selector-only
null-span follow-up. The result is a clean negative for the simple seed
adapter family.

The selector order signal is real: zero-model pair-index context floors go as
low as 193 bytes versus the current 249-byte FEC6 selector payload. But once
the context model is charged inside archive bytes, the byte-closed floors are
304 bytes or worse. The best actual seeded/residual adapter found is 326 bytes,
77 bytes larger than FEC6.

## Landed artifacts

- `src/tac/packet_compiler/pr101_seeded_selector_adapter.py`
- `tools/probe_pr101_seeded_selector_adapter.py`
- `src/tac/tests/test_pr101_seeded_selector_adapter.py`
- `.omx/research/pr101_seeded_selector_adapter_profile_20260520T232000Z_codex.json`
- `.omx/research/pr101_seeded_selector_adapter_profile_20260520T232000Z_codex.md`

## Live fec6 profile

Source archive:
`experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`

| surface | bytes |
| --- | ---: |
| current FEC6 selector payload | 249 |
| global entropy floor | 241 |
| first-order transition floor, zero-model | 227 |
| first-order transition floor with u16 transition counts charged | 739 |
| best pairmod zero-model floor | 193 |
| best pairmod byte-closed floor with u16 counts charged | 304 |
| best actual seeded/residual adapter | 326 |

Search:

- generator kinds spot-checked: `xorshift`, `lcg`, `pcg64`
- seed lengths: `1,2,4,8,16,32`
- deterministic seeds per length in committed profile: `1024`
- local scratch confirmation with `4096` seeds per length across all 3
  generator kinds: same best candidate, `constant code0`, 326 bytes.

## Interpretation

The operator concern about order relative to entropy is correct. The selector
sequence has order structure that a free model could exploit. The blocker is
deliverability: charging a context/transition model or residual override bytes
erases the gain for this simple adapter class.

This falsifies the naive selector-only seeded adapter as the next
materialization target. It does not kill selector work globally: reactivation
requires a predictor whose charged model/seed plus residual stream is below
249 bytes before runtime adapter work is worth doing.

## Tests

`5 passed`:

```text
.venv/bin/python -m pytest src/tac/tests/test_pr101_seeded_selector_adapter.py -q
```

## Next action

Do not spend on selector-seed runtime materialization yet. The next higher-EV
path is either:

1. source-payload null-span decomposition into smaller parser-safe subspans,
   or
2. an order model whose parameters are derived from an already-charged runtime
   prior rather than stored as a per-video selector table.
