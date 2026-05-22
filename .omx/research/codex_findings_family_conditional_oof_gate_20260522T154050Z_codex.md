# Codex Findings - Family-Conditional OOF Gate

Date: 2026-05-22T15:40:50Z

## Summary

The scorer-response validation gate now emits explicit per-family spend-triage
gates. Aggregate held-out correlation can still pass for local surrogate
training, but candidate selection can now fail closed by family. This prevents
a strong control family from masking decoder-q family failure.

## Contract

Each family gate has schema `scorer_response_family_spend_triage_gate.v1` and
keeps:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`

The new top-level fields are:

- `family_spend_triage_gates`
- `spend_triage_usable_families`
- `spend_triage_blocked_families`

Downstream MLX/dynamic-sweep candidate selectors should consult the target
family gate before using OOF predictions for spend triage. A blocked
`mlx_decoder_q` or `decoder_q` family must route through observed local rows,
new features, or exact-axis calibration instead of inheriting aggregate OOF
confidence.

## Prefix K028 Exact CPU Calibration

The pending DQS1 `prefix_k028` Modal CPU exact eval recovered while this gate
was landing:

- Archive SHA-256:
  `9da13be0a0eac60d0aa22219c325f38e56d2534fa0045050f365615ded3f9c5a`
- Archive bytes: `178556`
- Score: `0.19202928538027739 [contest-CPU]`
- Result path:
  `experiments/results/modal_auth_eval_cpu/dqs1_prefix_k028_gap_uleb_selective_decoderq_20260522T153210Z_cpu/contest_auth_eval.json`

This is a useful selector-curve calibration point, not a frontier move. It is
slightly worse than the current DQS1 top32 compact gap-ULEB CPU anchor
`0.19202894881608987 [contest-CPU]`.

## Pair-set acquisition and dynamic observations

Two follow-on local-only surfaces landed with the gate:

- `tac.optimization.decoder_q_pairset_acquisition` builds DQS1 pair-set
  acquisition rows from the selector Pareto output: prefix, drop-one,
  bounded drop-two, swap-in, and diversity-spaced probes.
- `tac.optimization.mlx_dynamic_sweep_observations` appends fail-closed MLX
  dynamic-sweep observations with archive/runtime/cache SHA-256 identity and
  component deltas.

Both surfaces explicitly keep `score_claim=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false`, and `dispatch_attempted=false`. Their
allowed use is local candidate generation and replanning only; exact-eval
dispatch still requires materialization, archive custody, and the normal lane
claim / auth-eval gate.

The current DQS1 pair-set acquisition artifact is ignored rebuildable state:

- JSON:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.json`
- Markdown:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.md`
- Candidate count: `175`
- Recommended local observation acquisition: `pairset_diversity_k002`

That recommendation is not an exact score claim. It is a first local
observation probe chosen by a planning score that mixes inherited
non-authoritative selector score, descriptor bytes, and diversity.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_decoder_q_pairset_acquisition.py src/tac/tests/test_mlx_dynamic_sweep_observations.py -q
ruff check src/tac/local_acceleration/mlx_score_calibration.py src/tac/optimization/scorer_response_dataset.py src/tac/optimization/decoder_q_pairset_acquisition.py src/tac/optimization/mlx_dynamic_sweep_observations.py src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_decoder_q_pairset_acquisition.py src/tac/tests/test_mlx_dynamic_sweep_observations.py tools/validate_scorer_response_dataset.py tools/plan_decoder_q_pairset_acquisition.py tools/append_mlx_dynamic_sweep_observation.py
.venv/bin/python tools/scan_best_anchor_per_axis.py --format json
```

Results:

- `127 passed`
- Ruff passed
- Frontier scanner reported no drift

## Next

Use the pair-set acquisition output to select the next local MLX observation
and append it through the dynamic-sweep observation ledger before allowing the
family OOF gate to steer exact-eval spend.
