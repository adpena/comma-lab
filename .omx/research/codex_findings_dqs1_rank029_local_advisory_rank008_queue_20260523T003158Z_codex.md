# Codex Findings: DQS1 Rank029 Local Advisory And Rank008 Queue Reroute

UTC: 2026-05-23T00:31:58Z

## Evidence Axis

This memo records `[macOS-CPU advisory]` local evidence and
`[contest-CPU drift-projected; false authority]` spend-triage projection. It is
not a score claim, not promotion evidence, and not rank/kill evidence.

## Candidate

Candidate: `pairset_drop_one_rank029_pair0259`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue steps completed:

- `plan_packet`
- `materialize`
- `locality_controls`
- `local_cpu_advisory`

Durable/ignored source artifacts:

- plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank029_pair0259.json`
- materialization:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank029_pair0259/materialization_manifest.json`
- locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank029_pair0259/locality_controls.json`
- local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank029_pair0259/local_cpu_advisory.json`
- eureka:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank029_pair0259_20260523T003158Z.json`

## Local Advisory Result

- Local score: `0.19203928295713674` `[macOS-CPU advisory]`
- Archive bytes: `178559`
- PoseNet distortion: `0.00002943`
- SegNet distortion: `0.00055989`
- Rate: `0.0047558043524216715`
- Archive SHA-256:
  `4d0572240e3776bbeca2ad84753016e8e846ccf91645563f2cd372b459bb37b3`
- Runtime tree SHA-256:
  `0066f0f94934f9ca02bb9e221e042ddc621b7feab4293e60c194a29d312cfb9b`
- Runtime content tree SHA-256:
  `a48086d79a46cafb68dbe9cc04084ad0bf28e7c23a5f243129585cc64e7c70a7`
- Inflated output manifest SHA-256:
  `1d1d5a4604731975952a6f89e71bf4cc481503840b25e9ecf2a2fb81f2b4dff3`
- Inflated output aggregate SHA-256:
  `b7803e43bcd6644cd3f1ddc29771a2508a14745be2c164e71e5c11e8ae86c815`
- Local advisory JSON SHA-256:
  `74fdb23eb915a11521cbe71963d12a15013596460f12c4e08a5789bcbf1e9e8e`
- Locality controls SHA-256:
  `0536e97685233c49b29349fab26ec7593a96c7811ef3e14edd75938b61de683b`
- Materialization manifest SHA-256:
  `f461a7b8a6cc3079c45aca46c153410f44c368eb9548755e9b4933314e37ba17`

## Drift/Eureka Projection

Calibration:
`.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`

- Bias local minus contest: `0.000010000000000010001`
- Guard band: `0.000003`
- Calibration confidence: `stable_core`
- Anchor count: `5`
- Projected contest score:
  `0.19202928295713673`
  `[contest-CPU drift-projected; false authority]`
- Conservative projected contest score:
  `0.19203228295713673`
  `[contest-CPU drift-projected; false authority]`
- Current exact CPU frontier anchor:
  `0.19202828295713675`
- Eureka margin: `-0.000003999999999976245`
- Eureka trigger: `false`
- Recommended action: `observe_only`

Interpretation: rank029 is worse than the current exact CPU frontier after
stable-core DQS1/FEC6 same-archive drift correction. Do not dispatch exact eval
for rank029 from this signal.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank029's eureka record was present:

- Next candidate: `pairset_drop_one_rank008_pair0496`
- Lane: `lane_dqs1_pairset_drop_one_rank008_pair0496_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,501,520,544,555,588`
- Queue file:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank029_pair0259 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank029_pair0259/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank029_pair0259_20260523T003158Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_local_cpu_contest_drift.py`

## Next Action

Continue the local-first queue with `pairset_drop_one_rank008_pair0496`.
If its eureka signal is positive, treat that as an exact-auth-anchor dispatch
trigger only after the normal lane-claim and exact CPU/CUDA custody gates; it
still will not be score authority until exact auth eval lands.
