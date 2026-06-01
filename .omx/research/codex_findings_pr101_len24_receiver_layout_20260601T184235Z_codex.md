# Codex findings - PR101 len24 receiver-compatible layout

- **Date:** 2026-06-01T18:42:35Z
- **Axis:** `[planning-only byte-profile]`
- **Score authority:** false
- **Promotion authority:** false

## Landing

Added a minimal receiver-compatible decoder-length layout:

- `len24_decoder_len_adapter`
- `build_len24_receiver_adapter_source_from_report(...)`
- `build_len24_receiver_runtime_tree_from_report(...)`
- CLI support via `--grouped-archive-layout len24_decoder_len_adapter`
- campaign summary compatibility for len24 archive/runtime proofs

The prior self-describing layout used a u32 decoder-section length. PR101
decoder blobs are far below 16 MiB, so a 24-bit little-endian section length is
the smallest fixed-length legal field for this adapter family.

## Real PR101 run

Artifact root:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_campaign_summary_len24_20260601T184235Z`

Campaign summary:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_campaign_summary_len24_20260601T184235Z/campaign_summary.json`

Campaign summary SHA-256:

`bf698ff96b77d1021aa95a5b170592397e5219edfc6a4da9320c5776c97cd292`

Consumer result:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_campaign_summary_len24_20260601T184235Z/campaign_consumer_result.json`

Consumer result SHA-256:

`4079eafcdcc6317d61bad9dc62bcafe99fdb52a6a74a6bf760db48f8a4f301da`

Measured result:

- grouped decoder saved bytes: `1`
- len24 decoder-length header bytes: `3`
- legal archive delta: `+2`
- verdict: `grouped_positive_consumed_by_archive_overhead`
- consumer planner action:
  `record_negative_rate_posterior_and_demote_format_churn`

## Interpretation

This improves the receiver-compatible header overhead by one byte versus the
previous u32 layout (`+3` archive delta -> `+2`), but it does not change the
current PR101/fec6 conclusion. The current substrate remains grammar-saturated;
the value is that future unsaturated substrates now inherit the minimal
fixed-length receiver-compatible layout instead of paying a u32 tax by default.

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

- `28 passed`
- ruff clean
