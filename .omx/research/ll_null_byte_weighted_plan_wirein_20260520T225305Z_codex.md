# LL Null-Byte Weighted Planner Wire-In 2026-05-20

## Scope

Codex integration sweep for the operator-routed handoff:

- `tools/probe_null_byte_master_gradient_matrix.py` output is now an optional
  side-input to `tools/plan_ll_scorer_response_next.py`.
- `procedural_codebook_savings_consumer` now delegates per-frame ordering to
  `per_frame_sensitivity_consumer` when a candidate carries a per-frame
  decomposition payload, with explicit affected-frame scope required for
  allocation.
- `CandidateRow.consumer_payload` now passes bounded consumer-only schemas
  through the cathedral autopilot invocation path, so direct consumer tests and
  live autopilot consumption use the same payload surface.

## Generated artifact

Command:

```bash
.venv/bin/python tools/plan_ll_scorer_response_next.py \
  --dataset experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_response_dataset_20260520_codex.json \
  --null-byte-matrix .omx/research/null_byte_probe_matrix_20260520T223927Z_codex/null_byte_matrix.json \
  --null-byte-seed-budget-k 16 \
  --json-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_weighted_20260520_codex.json \
  --md-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_weighted_20260520_codex.md
```

Top null-byte LL priority rows by absolute predicted score delta at `K=16`:

| rank | substrate | null bytes | priority weight |
| ---: | --- | ---: | ---: |
| 1 | `pr106_format0d` | 16909 | 0.01124835529509284 |
| 2 | `pr101_fec6_frontier` | 16292 | 0.010837520321016461 |
| 3 | `a1_finetuned` | 16037 | 0.010667726287970308 |
| 4 | `pr101_lc_v2` | 16033 | 0.010665062852157818 |
| 5 | `pr107_apogee` | 15987 | 0.010634433340314199 |

## Authority

All outputs remain fail-closed:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- LL priority weights are `[predicted]` routing priors only.

The new first LL probe is
`ll_null_byte_procedural_codebook_candidates`, with acceptance gated on typed
`CandidateModificationSpec`, byte-consumption/no-op proof, and exact contest
eval before any score claim.

## Validation

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  src/tac/tests/test_probe_null_byte_master_gradient_matrix.py \
  src/tac/cathedral_consumers/procedural_codebook_savings_consumer/tests
```

Result: `50 passed in 0.81s`.

Adjacent autopilot/consumer regression:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_check_336_337_cathedral_main_discovery_invoker.py \
  src/tac/tests/test_hf_jobs_dispatcher_consumer.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py
```

Result: `207 passed in 2.59s`.
