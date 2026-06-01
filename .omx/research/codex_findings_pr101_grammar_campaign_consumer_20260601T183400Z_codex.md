# Codex findings - PR101 grammar campaign consumer

- **Date:** 2026-06-01T18:34:00Z
- **Axis:** `[planning-only byte-profile]`
- **Score authority:** false
- **Promotion authority:** false

## Landing

Added the cathedral/autopilot consumer for
`pr101_optimal_grammar_campaign_summary.v1`:

- `tac.cathedral_consumers.pr101_optimal_grammar_campaign_consumer`
- consumer contract validation
- replay-routing coverage for archive-positive runtime-compatible summaries
- demotion coverage for archive-overhead / saturated summaries
- false-authority overclaim and schema-mismatch blockers

This closes the loop left by the campaign summary landing: the deterministic
packet compiler now emits a campaign verdict that a normal cathedral consumer
can ingest as planning signal.

## Real artifact consumption

Source summary:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_campaign_summary_u32_20260601T182700Z/campaign_summary.json`

Consumer result:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_campaign_summary_u32_20260601T182700Z/campaign_consumer_result.json`

Consumer result SHA-256:

`b722aab040bab8c26c741a9b6e838adccc1f1f467823275fdae18e310e8f83c3`

Verdict:

- planner action:
  `record_negative_rate_posterior_and_demote_format_churn`
- demotion recommended: `true`
- local replay recommended: `false`
- archive delta: `+3` bytes

## Interpretation

The current PR101/fec6-style grammar path is now machine-demotable rather than
chat-demotable. Future unsaturated substrates can reuse the same summary and
consumer contract: if archive bytes remain positive after legal receiver
layout, the consumer routes to local replay; if grouped savings vanish or are
eaten by header/runtime overhead, it records a negative rate posterior and
keeps exact auth off.

## Verification

```bash
uv run pytest \
  src/tac/tests/test_pr101_optimal_grammar_campaign_consumer.py \
  src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q

uv run ruff check \
  src/tac/cathedral_consumers/pr101_optimal_grammar_campaign_consumer/__init__.py \
  src/tac/tests/test_pr101_optimal_grammar_campaign_consumer.py \
  src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py \
  src/tac/tests/test_pr101_per_tensor_grammar_solver.py
```

Result:

- `24 passed`
- ruff clean
