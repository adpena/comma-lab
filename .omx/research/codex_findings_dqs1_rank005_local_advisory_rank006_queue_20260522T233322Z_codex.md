# Codex Findings: DQS1 Rank005 Local Advisory And Rank006 Queue Reroute

UTC: 2026-05-22T23:33:22Z
Agent: Codex
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank005_pair0467`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,479,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank005_pair0467.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank005_pair0467/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank005_pair0467/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank005_pair0467/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank005_pair0467_20260522T233322Z.json`

Custody hashes:

- Archive SHA-256:
  `6b6c232af008a1e256184774cce11fe87bbc6d64aa674b47cc10fecfc61aa576`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `00aa2d450e8311f691b5f72e5f50adb5e7a235b1d1f2b811ca76f330db3268fa`
- Runtime content tree SHA-256:
  `819b08d96d8b41f3556a8537b896d08988c11d80a89bfde7c3170e87fe159ce0`
- Inflated output manifest SHA-256:
  `3878815e7e4c5b949784d1ac6fc9d02b4cd2537f614013e50c1057ff9f35fd91`
- Inflated output aggregate SHA-256:
  `b0f374a107a5e67fe9b06fb7ec6c1b4f6a0904155fb45a248979e32247b0f777`

Local advisory metrics:

- Axis: `[macOS-CPU advisory]`
- `canonical_score`: `0.19204028295713674`
- `avg_segnet_dist`: `0.0005599`
- `avg_posenet_dist`: `0.00002943`
- `score_rate_contribution`: `0.11889510881054179`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `rank_or_kill_eligible`: `false`

Drift/eureka interpretation:

- Calibration:
  `.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`
- Projected contest CPU score: `0.19203028295713673`
- Conservative projected contest CPU score: `0.19203328295713673`
- Current auth frontier score: `0.19202828295713675`
- Eureka trigger: `false`
- Recommended action: `observe_only`

Interpretation: rank005 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank005's eureka record was present:

- Next candidate: `pairset_drop_one_rank006_pair0544`
- Lane: `lane_dqs1_pairset_drop_one_rank006_pair0544_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,555,588`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank005_pair0467 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank005_pair0467/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank005_pair0467_20260522T233322Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`

## Next Action

Do not promote rank005. Continue with rank006 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
