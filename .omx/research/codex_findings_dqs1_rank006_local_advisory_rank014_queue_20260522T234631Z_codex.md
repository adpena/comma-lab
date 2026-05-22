# Codex Findings: DQS1 Rank006 Local Advisory And Rank014 Queue Reroute

UTC: 2026-05-22T23:46:31Z
Agent: Codex
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank006_pair0544`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank006_pair0544.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank006_pair0544/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank006_pair0544/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank006_pair0544/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank006_pair0544_20260522T234631Z.json`

Custody hashes:

- Archive SHA-256:
  `ece073e1e072933d35d8eadc0cf1dc400ab9f60630621dc1753313fd68bc9867`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `ebe00a96037445b6a0c43a045d46a34434be1e7ef05fbd4ac34667ea569cfd29`
- Runtime content tree SHA-256:
  `95abea1622de09e90de2aa0e9d546510a3dcc98cbc8840ba614403a373d250ab`
- Inflated output manifest SHA-256:
  `10118cdaec0d3e422b510fe9591df5d41b6d58cb4c9e124ef76bf7dc8d0cdcd7`
- Inflated output aggregate SHA-256:
  `23a3d576a1a341859bd290227430d8c7abf00e4c771bd453d9e5bcb20947dcc4`

Local advisory metrics:

- Axis: `[macOS-CPU advisory]`
- `canonical_score`: `0.19204128295713674`
- `avg_segnet_dist`: `0.00055991`
- `avg_posenet_dist`: `0.00002943`
- `score_rate_contribution`: `0.11889510881054179`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `rank_or_kill_eligible`: `false`

Drift/eureka interpretation:

- Calibration:
  `.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`
- Projected contest CPU score: `0.19203128295713673`
- Conservative projected contest CPU score: `0.19203428295713673`
- Current auth frontier score: `0.19202828295713675`
- Eureka trigger: `false`
- Recommended action: `observe_only`

Interpretation: rank006 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank006's eureka record was present:

- Next candidate: `pairset_drop_one_rank014_pair0492`
- Lane: `lane_dqs1_pairset_drop_one_rank014_pair0492_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,496,501,520,544,555,588`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank006_pair0544 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank006_pair0544/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank006_pair0544_20260522T234631Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`

## Next Action

Do not promote rank006. Continue with rank014 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
