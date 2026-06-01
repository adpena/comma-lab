# Codex findings - PR101 optimal grammar campaign summary

- **Date:** 2026-06-01T18:22:22Z
- **Axis:** `[planning-only byte-profile]`
- **Score authority:** false
- **Promotion authority:** false

## Landing

Added the campaign-level planner verdict for the PR101/HNeRV optimal-grammar
lane:

- `build_pr101_optimal_grammar_campaign_summary(...)`
- CLI flag `--campaign-summary-output`
- regression coverage for saturated grouped-zero, grouped-positive replay-ready,
  archive/runtime layout incompatibility, and adapter-overhead demotion.

This is the missing bridge between lower-level packet measurements and the
queue/autopilot decision surface. It prevents isolated or grouped byte wins from
being treated as useful unless they survive the legal archive/runtime layout.

## Real PR101 run

Artifact root:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_campaign_summary_u32_20260601T182700Z`

Campaign summary:

`/Volumes/VertigoDataTier/pact/pr101_optimal_grammar_campaign_summary_u32_20260601T182700Z/campaign_summary.json`

Summary SHA-256:

`929a16a58744e9744910d34b7578575ee8e8c1b64451322249f1f4a098910e72`

Measured result:

- selected isolated tensor bytes: `162226`
- current isolated tensor bytes: `162260`
- empirical Shannon floor: `159877.3`
- saturation status: `entropy_saturated`
- grouped selected bytes: `162163`
- current grouped bytes: `162164`
- grouped saved bytes: `1`
- u32 receiver-compatible archive delta: `+3`
- verdict: `grouped_positive_consumed_by_archive_overhead`
- remaining blockers: `full_frame_inflate_parity_missing`,
  `contest_cpu_cuda_exact_eval_not_executed`
- next action:
  `demote_current_pr101_adapter_branch; require_fixed_runtime_or_larger_grouped_savings`

## Interpretation

The current PR101/fec6 decoder grammar remains effectively saturated. A
compress-time exhaustive run found one grouped decoder byte, but the smallest
receiver-compatible self-describing archive layout costs four bytes, so the
legal archive is three bytes larger. The solver is therefore useful
infrastructure for future unsaturated substrates, not a score-lowering branch
for the current PR101 decoder.

The important system behavior is now automatic: future grammar campaigns emit a
single false-authority campaign summary that can demote format churn, justify
receiver-adapter work only when archive-level bytes remain positive, or route a
byte-closed candidate to local replay gates.

## Verification

```bash
uv run pytest src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q
uv run ruff check \
  src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py \
  tools/pr101_per_tensor_grammar_solver.py \
  src/tac/tests/test_pr101_per_tensor_grammar_solver.py
```

Result:

- `20 passed`
- ruff clean
