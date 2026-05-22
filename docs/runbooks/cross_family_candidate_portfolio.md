# Cross-Family Candidate Portfolio

This runbook wires the learned-sweep and cross-family planner into an operator
flow. The output is planning-only. It never claims a score, promotes a row,
rank/kills a lane, launches a GPU job, or skips exact CPU/CUDA auth eval.

## Inputs

- MLX spend-triage selections from `mlx_effective_spend_triage_candidate_selection.v1`.
- DQS1 pairset acquisitions from `decoder_q_pairset_acquisition.v1`.
- Byte-closed outside-family manifests, such as
  `hfv1_to_hfv2_sparse_sidecar_candidate_v1`.
- Optional exact-axis observations from
  `mlx_dynamic_sweep_observation.v1` JSONL.

Every source row must preserve false-authority fields:
`score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`, and
`promotable=false`.

## Command

```bash
.venv/bin/python tools/plan_cross_family_candidate_portfolio.py \
  --incumbent-score 0.2053300290 \
  --incumbent-score-by-axis contest_cpu=0.19202894881608987 \
  --pairset-acquisition experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.json \
  --observation-jsonl experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl \
  --json-out experiments/results/cross_family_candidate_portfolio/latest/portfolio.json \
  --md-out experiments/results/cross_family_candidate_portfolio/latest/portfolio.md \
  --summary-json-out experiments/results/cross_family_candidate_portfolio/latest/action_summary.json \
  --top-actions 8 \
  --require-active-pairset-observation-model
```

Use `--require-active-pairset-observation-model` when the operator expects the
DQS1 pairset response curve to be evidence-informed. It fails closed unless the
observation JSONL contains at least two exact-axis pairset observations at
distinct `selected_pair_count` values for the same selector kind. Pairset
observations should include `selected_pair_indices`; the model uses them to
verify candidate identity before fitting. Omit the guard only for exploratory
portfolio composition before exact pairset feedback exists.

## Outputs

- `portfolio.json`: full planner payload, Bayesian rows, source custody hashes,
  observation feedback, and dispatch blockers.
- `portfolio.md`: human-readable queue with an appended CLI action summary.
- `action_summary.json`: compact operator handoff with the top N next actions,
  pairset observation-response status, and dispatch blockers.
- stdout: same compact handoff, suitable for terminal monitoring.

The top action is the next materialization/control step, not a dispatch
authorization. Before exact eval spend, claim the lane, materialize or refresh
the archive/runtime packet, run locality and no-op controls, then dispatch
through the canonical auth-eval path and harvest the result.

## Authority Boundary

MLX, macOS CPU, MPS, Bayesian, and portfolio rows are local/proxy/planning
signals. They may choose local follow-up candidates or exact-eval spend
candidates only after the required custody and calibration gates pass. They
cannot claim score, promote, rank/kill, or replace contest CPU/CUDA auth eval.

The only score-bearing rows are harvested contest-auth artifacts with explicit
axis labels such as `[contest-CPU Linux x86_64]` or `[contest-CUDA T4]`, archive
SHA-256, runtime-tree SHA-256, inflated-output aggregate SHA-256, and the
canonical auth-eval schema.
