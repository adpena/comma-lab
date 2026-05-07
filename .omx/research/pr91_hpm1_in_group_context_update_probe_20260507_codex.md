# PR91 HPM1 In-Group Context Update Probe - 2026-05-07

Scope: PR91/HPM1 semantic entropy grammar only. Local CPU probe; no remote,
GPU, lane claim, exact eval, or score claim.

Artifact:

- `experiments/results/pr91_hpm1_in_group_context_update_probe_20260507_codex/in_group_context_update_probe.json`

Finding:

- Source-contract replay still fails at frame `0`, group `10`, symbol `191`,
  after `5951` decoded symbols.
- Same-group serial-prefix context was tested by assigning the `191` prior
  decoded symbols in failing group `10` into the current-frame context before
  recomputing only the failing probability row.
- The recomputed row changed slightly but did not clear the range assertion:
  `max_abs_probability_delta=9.013e-8`, source and serial argmax are both
  symbol `2`, and serial-prefix decode still raises `AssertionError`.

Conclusion:

- The current PR91/HPM1 blocker is not explained by the bounded same-group
  serial-prefix context hypothesis at the first source failure row.
- Remaining blockers are encoder-side probability numeric grammar, range-coder
  construction/finalization grammar, context drift before group `10`, or
  submitted-token semantics not represented by the public decode source.
