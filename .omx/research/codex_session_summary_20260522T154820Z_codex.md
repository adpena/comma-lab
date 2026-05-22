# Codex Session Summary - MLX Continual-Learning Swarm

Date: 2026-05-22T15:48:20Z

## Summary

Converted the fresh-eyes recommendations into concrete fail-closed machinery for
local MLX/advisory/exact feedback loops.

Landed surfaces:

- Family-conditional OOF hard gates in
  `tac.optimization.scorer_response_dataset`.
- Dual-axis CPU/CUDA calibration policy in
  `tac.local_acceleration.mlx_score_calibration`.
- Append-only dynamic sweep observation ledger in
  `tac.optimization.mlx_dynamic_sweep_observations`.
- Planning-only DQS1 pair-set acquisition planner in
  `tac.optimization.decoder_q_pairset_acquisition`.

All new/changed planning and observation surfaces remain false-authority:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`

Adversarial follow-up after review:

- The effective MLX spend-triage path now enforces family-level OOF gates via
  `required_spend_triage_families` and refuses selected families that are not
  present in `spend_triage_allowed_families`.
- Exact contest-axis dynamic sweep observations now validate against a local
  auth-eval JSON payload before preserving `[contest-CPU]` / `[contest-CUDA]`
  labels.

## Exact CPU Observation

Recovered DQS1 `prefix_k028` exact CPU calibration from Modal:

- Archive SHA-256:
  `9da13be0a0eac60d0aa22219c325f38e56d2534fa0045050f365615ded3f9c5a`
- Archive bytes: `178556`
- Score: `0.19202928538027739 [contest-CPU]`
- Runtime tree SHA-256:
  `58b6785f65b4c8a533877bee6e60caf242b68ef58033c4523b693736c9425f5e`
- Inflated output aggregate SHA-256:
  `e141577524a423037ce9f563dee553f72143581b688ad5301770b2e9807cb64e`

This is not a frontier move; it is slightly worse than current compact DQS1
top32 at `0.19202894881608987 [contest-CPU]`. It is useful as selector-curve
calibration and was appended to the ignored dynamic sweep observation JSONL.

## Generated Local Artifacts

Ignored by `.gitignore` under `experiments/results/*`:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.json`
- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.md`
- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl`

Pair-set planner emitted `175` local-only candidates. The top local acquisition
row is intentionally planning-only and requires materialization/local controls
before any exact-eval queueing.

The dynamic observation row points at
`experiments/results/modal_auth_eval_cpu/dqs1_prefix_k028_gap_uleb_selective_decoderq_20260522T153210Z_cpu/contest_auth_eval.json`
as its validated auth-eval source artifact.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_decoder_q_pairset_acquisition.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py -q
ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py tools/validate_scorer_response_dataset.py src/tac/local_acceleration/mlx_score_calibration.py src/tac/tests/test_mlx_score_calibration.py src/tac/optimization/mlx_dynamic_sweep_observations.py tools/append_mlx_dynamic_sweep_observation.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/optimization/decoder_q_pairset_acquisition.py tools/plan_decoder_q_pairset_acquisition.py src/tac/tests/test_decoder_q_pairset_acquisition.py src/tac/optimization/mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py tools/plan_ll_scorer_response_next.py
.venv/bin/python tools/scan_best_anchor_per_axis.py --format json
git check-ignore -v experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.md experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl
```

Results:

- `144 passed`
- Ruff passed
- Frontier scanner reported no drift
- `.gitignore` covers generated artifacts

## Next

1. Feed observation JSONL summaries back into `mlx_dynamic_learned_sweep` so
   known exact negatives suppress repeat rows.
2. Feed required-family gates into downstream MLX selection CLIs by default
   where a target family is known.
3. Materialize only the top pair-set acquisition rows that pass local controls,
   then use exact CPU/CUDA spend only after family OOF, calibration, custody,
   and lane-claim gates pass.
