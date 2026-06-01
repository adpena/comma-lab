# PR101 Grouped Optimal Grammar Operator-Briefing Wire-In

- **UTC:** 2026-06-01T19:51:21Z
- **Author:** Codex
- **Axis:** `[planning-only byte-profile]`
- **Score claim:** `false`
- **Promotion eligible:** `false`
- **Ready for exact eval dispatch:** `false`

## Finding

The isolated generic tensor grammar reports correctly demote current PR101-like
tensor format churn, but they are not the whole optimal-grammar objective. The
remaining PR101 grammar axis is grouped split-Brotli interaction: transform
payloads, storage order, and stream partitioning can produce byte deltas that
do not appear in isolated per-tensor accounting.

I regenerated the grouped PR101 campaign through the canonical CLI:

```bash
uv run python tools/pr101_per_tensor_grammar_solver.py \
  --state-dict-path /Volumes/VertigoDataTier/pact/pr101_real_grouped_runtime_campaign_20260601T172357Z/source_decoder_state_dict.pt \
  --output /Volumes/VertigoDataTier/pact/pr101_grouped_grammar_operator_surface_20260601T1948Z/per_tensor_report.json \
  --queue-output /Volumes/VertigoDataTier/pact/pr101_grouped_grammar_operator_surface_20260601T1948Z/per_tensor_queue.json \
  --grouped-output /Volumes/VertigoDataTier/pact/pr101_grouped_grammar_operator_surface_20260601T1948Z/grouped_report.json \
  --grouped-queue-output /Volumes/VertigoDataTier/pact/pr101_grouped_grammar_operator_surface_20260601T1948Z/grouped_queue.json \
  --campaign-summary-output /Volumes/VertigoDataTier/pact/pr101_grouped_grammar_operator_surface_20260601T1948Z/campaign_summary.json \
  --grouped-storage-order-mode best-of-builtins \
  --grouped-exact-stream-count 7 \
  --grouped-max-streams 7
```

Real result:

- isolated selected bytes: `162226`
- current isolated bytes: `162260`
- isolated saved bytes: `34`
- grouped selected bytes: `162152`
- current grouped bytes: `162164`
- grouped saved bytes: `12`
- grouped runtime: `tac_decode_decoder_compact_with_overrides_required`
- campaign verdict: `grouped_positive_build_receiver_adapter`
- exact eval readiness: `false`

Interpretation: the grouped axis is not zero, but current payoff is tiny and
requires adapter/archive/runtime binding before it can even reach local replay.
This is exactly the substrate-conditional grammar lesson: PR101/fec6 is near
saturated; the solver remains valuable because future unsaturated substrates
inherit the grouped-order/stream-partition search automatically.

## Landing

Code:

- `tools/operator_briefing.py`
- `src/tac/tests/test_operator_briefing.py`

New operator surface:

- JSON key: `optimal_grammar_campaign`
- readiness key: `phase_6c_optimal_grammar_campaign`
- text section: `Phase 6c.2 — PR101 grouped optimal-grammar campaign`

The scanner is schema-gated on:

- `pr101_optimal_grammar_campaign_summary.v1`

Live briefing helper summary after the landing:

```json
{
  "campaign_count": 7,
  "exact_auth_work_justified_count": 0,
  "receiver_adapter_work_justified_count": 3,
  "ready_for_exact_eval_dispatch": false,
  "score_claim": false,
  "status": "NEEDS_RECEIVER_BINDING",
  "total_grouped_saved_bytes": 29
}
```

## Verification

```bash
uv run ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py
uv run pytest -q src/tac/tests/test_operator_briefing.py -k 'tensor_payload or optimal_grammar'
```

Results:

- `ruff`: pass
- focused tests: `3 passed, 51 deselected`

## Next

Do not exact-dispatch the current grouped PR101 grammar signal. The total
visible grouped savings are only tens of bytes and remain blocked on receiver
binding and local replay. The correct campaign behavior is:

- preserve the grouped-positive rows as planner signal;
- bind receiver/archive only if a substrate shows a much larger grouped gap;
- keep PR101/fec6 format churn demoted unless grouped savings exceed adapter
  overhead and local full-frame replay wins.

