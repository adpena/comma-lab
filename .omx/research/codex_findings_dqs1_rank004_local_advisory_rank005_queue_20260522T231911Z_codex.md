# Codex Findings: DQS1 Rank004 Local Advisory And Rank005 Queue Reroute

UTC: 2026-05-22T23:19:11Z
Agent: Codex
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank004_pair0098`

Selected pair indices:

`26,59,68,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank004_pair0098.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank004_pair0098/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank004_pair0098/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank004_pair0098/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank004_pair0098_20260522T231911Z.json`

Custody hashes:

- Archive SHA-256:
  `5650a8715b85530b7a48b5be9d02e9425484e7be2d341e96533f15067677b1c6`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `8867503413a6b06fb3836ffb1ccd82aa6bacb84682edfc069ac763d329c00fcb`
- Runtime content tree SHA-256:
  `b6353f44fa28504758d7eb47bcf9e94d28c790a8023617c03ab373f08751d4c4`
- Inflated output manifest SHA-256:
  `812ad278e8b48373b31d931fc397712b0c763c10b8c223100e67c05b357d96e0`
- Inflated output aggregate SHA-256:
  `9053b430f25d389fba3e35d8c2779f13fa34a9e38e9565d77d4f3611c3e149d5`

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

Interpretation: rank004 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank004's eureka record was present:

- Next candidate: `pairset_drop_one_rank005_pair0467`
- Lane: `lane_dqs1_pairset_drop_one_rank005_pair0467_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,479,492,496,501,520,544,555,588`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank004_pair0098 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank004_pair0098/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank004_pair0098_20260522T231911Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`

## Next Action

Do not promote rank004. Continue with rank005 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
