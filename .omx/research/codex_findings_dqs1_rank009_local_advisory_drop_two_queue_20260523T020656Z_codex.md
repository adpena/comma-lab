# Codex Findings: DQS1 Rank009 Local Advisory And Drop-Two Queue Reroute

UTC: 2026-05-23T02:06:56Z

## Evidence Axis

This memo records `[macOS-CPU advisory]` local evidence and
`[contest-CPU drift-projected; false authority]` spend-triage projection. It is
not a score claim, not promotion evidence, and not rank/kill evidence.

## Candidate

Candidate: `pairset_drop_one_rank009_pair0459`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,467,479,492,496,501,520,544,555,588`

Local-first queue steps completed:

- `plan_packet`
- `materialize`
- `locality_controls`
- `local_cpu_advisory`
- `local_cpu_contest_drift_eureka`

Durable/ignored source artifacts:

- plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank009_pair0459.json`
- materialization:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank009_pair0459/materialization_manifest.json`
- locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank009_pair0459/locality_controls.json`
- local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank009_pair0459/local_cpu_advisory.json`
- eureka:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank009_pair0459_20260523T015233Z.json`
- harvest:
  `.omx/research/dqs1_local_first_harvest_pairset_drop_one_rank009_pair0459_20260523T020656Z.json`

## Local Advisory Result

- Local score: `0.19203928295713674` `[macOS-CPU advisory]`
- Archive bytes: `178559`
- PoseNet distortion: `0.00002943`
- SegNet distortion: `0.00055989`
- Rate: `0.0047558043524216715`
- Archive SHA-256:
  `c2f03946b92a26557d51b8386772c04702faa8b36956d11bd9a8d1db7ccb72f0`
- Runtime tree SHA-256:
  `9ddef8fd1e20f6b374cbe2c9946bb4f97c588c0f8f79a1bd9dfe26ff32d36e16`
- Runtime content tree SHA-256:
  `10629a6373b396c06cb61a24c72050474a408a462be570bbccac1e80a399bde5`
- Inflated output manifest SHA-256:
  `dfb6dd5161541f45a45e0016b9f3b58579c3d19bf3aca2d6271fe8110537e118`
- Inflated output aggregate SHA-256:
  `2291d47fd7342a4eb4a024c9a4773436d6c4034904fd5021240c6e52bed5de2b`
- Local advisory JSON SHA-256:
  `485ca040a0e6c9a1e1290dae70855e64b9edcd4987b748c771714b735ced2f9e`
- Locality controls SHA-256:
  `c14ce15e61c39087b7635c1d2a984556568d09c712d802746e796374164d18f6`
- Materialization manifest SHA-256:
  `bb86e9bd10a41f2788c681f0d8cb39e1a48ddf4c564997269ead23b6a98675a9`

## Drift/Eureka Projection

Calibration:
`.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`

- Bias local minus contest: `0.000010500000000010501`
- Guard band: `0.000003`
- Calibration confidence: `stable_core`
- Anchor count: `4`
- Projected contest score:
  `0.1920287829571367`
  `[contest-CPU drift-projected; false authority]`
- Conservative projected contest score:
  `0.19203178295713672`
  `[contest-CPU drift-projected; false authority]`
- Current exact CPU frontier anchor:
  `0.19202828295713675`
- Eureka margin: `-0.000003499999999961867`
- Eureka trigger: `false`
- Recommended action: `observe_only`

Interpretation: rank009 is still worse than the current exact CPU frontier after
the stricter DQS1/FEC6 same-archive drift correction. Do not dispatch exact eval
for rank009 from this signal.

## Queue Reroute

The current validated local-first queue is now routed to:

- Next candidate: `pairset_drop_two_r029_010_p0259_0376`
- Lane: `lane_dqs1_pairset_drop_two_r029_010_p0259_0376_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,296,320,327,371,378,412,430,440,459,467,479,492,496,501,520,544,555,588`
- Queue file:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 5 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/harvest_dqs1_local_first_result.py --queue .omx/tmp_rank009_queue_for_harvest_20260523T020656Z.json --timestamp 20260523T020656Z --harvest-out .omx/research/dqs1_local_first_harvest_pairset_drop_one_rank009_pair0459_20260523T020656Z.json`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`

## Next Action

Continue the local-first queue with `pairset_drop_two_r029_010_p0259_0376`.
If its eureka signal is positive, treat that as an exact-auth-anchor dispatch
trigger only after the normal lane-claim and exact CPU/CUDA custody gates; it
still will not be score authority until exact auth eval lands.
