# Codex Findings: DQS1 Rank003 Local Advisory And Rank004 Queue Reroute

UTC: 2026-05-22T23:06:37Z
Agent: Codex
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank003_pair0479`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank003_pair0479.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank003_pair0479/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank003_pair0479/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank003_pair0479/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank003_pair0479_20260522T230637Z.json`

Custody hashes:

- Archive SHA-256:
  `52e3cf3a681fc3c4fbb9eec2a327ceb463249e0a730206a315ebe9ad7f4338ff`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `fe9d9ac6a8b8ca207eeb063b8927de1ca7cc41294161615dfb04b8474dcd5610`
- Runtime content tree SHA-256:
  `e170955f222af1fe1f3a91ccc954b17103119c10935e7ab11b3c7368d499bd1b`
- Inflated output manifest SHA-256:
  `046028c47aa1e464b1f8934416b5f7793461f47f8f863bd60a7f0382f1414d5f`
- Inflated output aggregate SHA-256:
  `07ac95718f5e5e8643a153d2ef2ec2c110ca739cf3c800f2a15678662fc809f1`

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

Interpretation: rank003 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank003's eureka record was present:

- Next candidate: `pairset_drop_one_rank004_pair0098`
- Lane: `lane_dqs1_pairset_drop_one_rank004_pair0098_local_first_20260522`
- Selected pair indices:
  `26,59,68,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank003_pair0479 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank003_pair0479/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank003_pair0479_20260522T230637Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`

## Next Action

Do not promote rank003. Continue with rank004 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
