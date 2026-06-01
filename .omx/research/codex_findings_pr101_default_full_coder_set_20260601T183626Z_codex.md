# Codex findings - PR101 default full coder set

- **Date:** 2026-06-01T18:36:26Z
- **Axis:** `[planning-only byte-profile]`
- **Score authority:** false
- **Promotion authority:** false

## Landing

The per-tensor grammar solver already had a
`range_ac_empirical_hist_u16` measurement branch, but it was not part of the
default solver or CLI coder set. That meant the stated optimal-grammar search
universe was only available by hand.

This landing promotes the full coder universe into the default path:

- Brotli
- raw LZMA1
- canonical Huffman
- empirical-histogram range/AC

The report now also emits aggregate `candidate_status_counts` and
`coder_status_counts`, so optional entropy-codec failures or wins are visible
planner signal instead of silent omissions.

## Real PR101 run

Artifact root:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_default_full_coder_set_20260601T183626Z`

Report:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_default_full_coder_set_20260601T183626Z/per_tensor_report.json`

Report SHA-256:

`9283e78e61d8e3731530f5d5d0be2370487fb51635ad96b18a95718843e4dde6`

Measured result:

- coders:
  `brotli,lzma_raw,canonical_huffman,range_ac_empirical_hist_u16`
- candidate status counts: `{"ok": 656}`
- range/AC status count: `{"range_ac_empirical_hist_u16:ok": 164}`
- selected isolated tensor bytes: `162226`
- empirical Shannon floor: `159877.34863949975`
- saturation status: `entropy_saturated`

## Interpretation

The current PR101/fec6 decoder grammar still saturates under the full default
coder universe. The value of this landing is therefore not a current score
claim; it is that future unsaturated substrates automatically price range/AC
without a hand-added CLI flag, and their reports expose whether the entropy
backend was measured, unavailable, or selected.

## Verification

```bash
uv run pytest \
  src/tac/tests/test_pr101_per_tensor_grammar_solver.py \
  src/tac/tests/test_pr101_optimal_grammar_campaign_consumer.py -q

uv run ruff check \
  src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py \
  tools/pr101_per_tensor_grammar_solver.py \
  src/tac/tests/test_pr101_per_tensor_grammar_solver.py
```

Result:

- `25 passed`
- ruff clean
