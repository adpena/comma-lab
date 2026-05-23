# Codex Findings: DQS1 Rank008 Local Advisory And Rank028 Queue Reroute

UTC: 2026-05-23T00:49:15Z

## Evidence Axis

This memo records `[macOS-CPU advisory]` local evidence and
`[contest-CPU drift-projected; false authority]` spend-triage projection. It is
not a score claim, not promotion evidence, and not rank/kill evidence.

## Candidate

Candidate: `pairset_drop_one_rank008_pair0496`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,501,520,544,555,588`

Local-first queue steps completed:

- `plan_packet`
- `materialize`
- `locality_controls`
- `local_cpu_advisory`

Durable/ignored source artifacts:

- plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank008_pair0496.json`
- materialization:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank008_pair0496/materialization_manifest.json`
- locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank008_pair0496/locality_controls.json`
- local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank008_pair0496/local_cpu_advisory.json`
- eureka:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank008_pair0496_20260523T004915Z.json`

## Local Advisory Result

- Local score: `0.19204028295713674` `[macOS-CPU advisory]`
- Archive bytes: `178559`
- PoseNet distortion: `0.00002943`
- SegNet distortion: `0.00055990`
- Rate: `0.0047558043524216715`
- Archive SHA-256:
  `0012eafab390d4586e524f207a6c4c71b40e07df2833a91f7c5267aa6adc0411`
- Runtime tree SHA-256:
  `8a43b61f6e03c0be921196b0dd1cb9b5c90a1337fa67a62cfda89a986551b458`
- Runtime content tree SHA-256:
  `a49f5a03a6dfaeeb99fe4e14b749e6687c48cdc2a5b9015042df37282598a86c`
- Inflated output manifest SHA-256:
  `518fe23b9243bb6b5a06b0d217fc1c7287621b641435a929e15f9b558054946e`
- Inflated output aggregate SHA-256:
  `c5a5c2ad0b9fa6e5d389f692d8109ee9bf18f01b14004382f0bdd6499315b6ff`
- Local advisory JSON SHA-256:
  `7638db7de492dd1f0cce61581ec8bc368c6d91d5865f57607bb3f89e75745e52`
- Locality controls SHA-256:
  `8d26dcb260e382582698d49e9fba4d1eae7a049f9f0176a64a99ea220bd8ccdc`
- Materialization manifest SHA-256:
  `55d86abd61ef3aae5fee255bb31530b5b9646188c8ee191abf5a8787133d204e`

## Drift/Eureka Projection

Calibration:
`.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`

- Bias local minus contest: `0.000010500000000010501`
- Guard band: `0.000003`
- Calibration confidence: `stable_core`
- Anchor count: `4`
- Projected contest score:
  `0.19202978295713674`
  `[contest-CPU drift-projected; false authority]`
- Conservative projected contest score:
  `0.19203278295713674`
  `[contest-CPU drift-projected; false authority]`
- Current exact CPU frontier anchor:
  `0.19202828295713675`
- Eureka margin: `-0.000004499999999990623`
- Candidate blocker:
  `candidate_local_authority_field_not_false:ready_for_exact_eval_dispatch`
- Eureka trigger: `false`
- Recommended action: `observe_only`

Interpretation: rank008 is worse than the current exact CPU frontier after the
stricter DQS1/FEC6 same-archive drift correction. The missing explicit
`ready_for_exact_eval_dispatch=false` field in the local advisory payload is
fail-closed and does not change the no-dispatch decision.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank008's eureka record was present:

- Next candidate: `pairset_drop_one_rank028_pair0257`
- Lane: `lane_dqs1_pairset_drop_one_rank028_pair0257_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`
- Queue file:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank008_pair0496 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank008_pair0496/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank008_pair0496_20260523T004915Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`

## Next Action

Continue the local-first queue with `pairset_drop_one_rank028_pair0257` after
the current scheduler/drift hardening WIP is either committed or intentionally
set aside. Do not dispatch exact eval for rank008.
