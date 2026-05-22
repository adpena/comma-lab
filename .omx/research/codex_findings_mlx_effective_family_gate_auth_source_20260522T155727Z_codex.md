# Codex Findings - MLX Effective Family Gate And Auth-Source Custody

Date: 2026-05-22T15:57:27Z

## Summary

Follow-up hardening converted the family-conditional OOF gate from an emitted
signal into an enforced downstream contract.

Changes:

- `build_next_probe_plan(..., required_spend_triage_families=...)` can now pass
  the required family list through the CLI via `--required-spend-triage-family`.
- `mlx_effective_spend_triage_selection` refuses selected families that are not
  present in the effective gate's `spend_triage_allowed_families`.
- Exact contest-axis dynamic sweep observations now require a local auth-eval
  JSON source artifact before preserving `[contest-CPU]` or `[contest-CUDA]`
  labels.

## Authority

This remains candidate-generation-only machinery:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`

The observation validator checks `score_axis`, `evidence_grade`,
`score_claim_valid=true`, archive SHA-256, inflated aggregate SHA-256, and
canonical score agreement against the local auth-eval JSON source artifact.

The DQS1 prefix K28 observation points at:

`experiments/results/modal_auth_eval_cpu/dqs1_prefix_k028_gap_uleb_selective_decoderq_20260522T153210Z_cpu/contest_auth_eval.json`

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py -q
.venv/bin/python -m ruff check src/tac/optimization/mlx_dynamic_sweep_observations.py src/tac/optimization/mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py tools/plan_ll_scorer_response_next.py
```

Results:

- `121 passed`
- Ruff passed

## Next

Use `--required-spend-triage-family mlx_decoder_q` when selecting real DQS1
follow-up candidates. Exact-eval dispatch remains blocked until materialization,
custody, lane claim, and auth-axis gates pass.
