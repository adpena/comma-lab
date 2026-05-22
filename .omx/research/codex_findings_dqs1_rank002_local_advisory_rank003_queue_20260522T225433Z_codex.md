# Codex Findings: DQS1 Rank002 Local Advisory And Rank003 Queue Reroute

UTC: 2026-05-22T22:54:33Z
Agent: Codex
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank002_pair0109`

Selected pair indices:

`26,59,68,98,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank002_pair0109.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank002_pair0109/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank002_pair0109/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank002_pair0109/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank002_pair0109_20260522T225433Z.json`

Custody hashes:

- Archive SHA-256:
  `cfd6328ea72774b4075b67a5a85ff0db9e82684e3bb7c82648df9a4efe14e67d`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `35f6579789dc9a8dfa1cd29cc5bc30dea688a273476e75f3641650570b0c1917`
- Runtime content tree SHA-256:
  `5fda8426c42c8e8538054ba084608cd61be251543f9c3a9c5c9a0fceea3b960d`
- Inflated output manifest SHA-256:
  `93e22f5c278d5df9284faa453fc072656dd24cac90cd5eb2ec57a49c0c1a4d73`
- Inflated output aggregate SHA-256:
  `74b943aeacfabc3e33a321a783215debd243abdf8077ebb604a4912b7d2d1587`

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

Interpretation: rank002 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

`tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`
selected the next unobserved top-32 DQS1-safe action after rank002's eureka
record was present:

- Next candidate: `pairset_drop_one_rank003_pair0479`
- Lane: `lane_dqs1_pairset_drop_one_rank003_pair0479_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,492,496,501,520,544,555,588`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank002_pair0109 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank002_pair0109/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank002_pair0109_20260522T225433Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`

## Next Action

Do not promote rank002. Continue with rank003 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
