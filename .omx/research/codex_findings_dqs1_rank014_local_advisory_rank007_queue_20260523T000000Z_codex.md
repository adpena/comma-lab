# Codex Findings: DQS1 Rank014 Local Advisory And Rank007 Queue Reroute

UTC: 2026-05-23T00:00:00Z
Agent: Codex
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank014_pair0492`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank014_pair0492.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank014_pair0492/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank014_pair0492/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank014_pair0492/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank014_pair0492_20260523T000000Z.json`

Custody hashes:

- Archive SHA-256:
  `5f95a8d6c7a68a9f6829d0b5f5e94447183a8553584535dba3b62de620c3ec74`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `6290bda0cf01735ef5759ff1cc5956d7b7d6c382dc9e2586b11899cecc823d7d`
- Runtime content tree SHA-256:
  `dfdac0364d07d7f8e40edfe5d9baec103cea1d2d689835ceeb662de7d2ad7610`
- Inflated output manifest SHA-256:
  `614c1b555ce8e69f8e13f258bb22e10e2ca3dd43b03e83020a41a99bf2f9c3c5`
- Inflated output aggregate SHA-256:
  `228a41a83db1c50cdb16c5c448ce7782d2a85fb5bcef92d42c16b74481ac423e`

Local advisory metrics:

- Axis: `[macOS-CPU advisory]`
- `canonical_score`: `0.19203928295713674`
- `avg_segnet_dist`: `0.00055989`
- `avg_posenet_dist`: `0.00002943`
- `score_rate_contribution`: `0.11889510881054179`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `rank_or_kill_eligible`: `false`

Drift/eureka interpretation:

- Calibration:
  `.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`
- Projected contest CPU score: `0.19202928295713673`
- Conservative projected contest CPU score: `0.19203228295713673`
- Current auth frontier score: `0.19202828295713675`
- Eureka trigger: `false`
- Recommended action: `observe_only`

Interpretation: rank014 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank014's eureka record was present:

- Next candidate: `pairset_drop_one_rank007_pair0059`
- Lane: `lane_dqs1_pairset_drop_one_rank007_pair0059_local_first_20260522`
- Selected pair indices:
  `26,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank014_pair0492 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank014_pair0492/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank014_pair0492_20260523T000000Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`

## Next Action

Do not promote rank014. Continue with rank007 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
