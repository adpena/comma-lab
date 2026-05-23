# Codex Findings: DQS1 Rank030 Local Advisory And Rank012 Queue Reroute

UTC: 2026-05-23T01:23:22Z

## Evidence Axis

This memo records `[macOS-CPU advisory]` local evidence and
`[contest-CPU drift-projected; false authority]` spend-triage projection. It is
not a score claim, not promotion evidence, and not rank/kill evidence.

## Candidate

Candidate: `pairset_drop_one_rank030_pair0412`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue steps completed:

- `plan_packet`
- `materialize`
- `locality_controls`
- `local_cpu_advisory`
- `local_cpu_contest_drift_eureka`

Durable/ignored source artifacts:

- plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank030_pair0412.json`
- materialization:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank030_pair0412/materialization_manifest.json`
- locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank030_pair0412/locality_controls.json`
- local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank030_pair0412/local_cpu_advisory.json`
- eureka:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank030_pair0412_20260523T010732Z.json`

## Local Advisory Result

- Local score: `0.19203928295713674` `[macOS-CPU advisory]`
- Archive bytes: `178559`
- PoseNet distortion: `0.00002943`
- SegNet distortion: `0.00055989`
- Rate: `0.0047558043524216715`
- Archive SHA-256:
  `78c3d31eac2537420b2cba504c8519cc2faf363d720e72f32dda2e31b532cf53`
- Runtime tree SHA-256:
  `6568b8d2cd0e1267c883892bf1b309a1224b66f4c5002d49ad84752c9ea7acb3`
- Runtime content tree SHA-256:
  `f145940a75806c5cc5e20d74c432d26c8646899d17d4442f5970f297ca9fae76`
- Inflated output manifest SHA-256:
  `3580c266636280c75c32904354be56d8281a8553d6e9eec40d8b77186b366d91`
- Inflated output aggregate SHA-256:
  `a92c2ac574d241f94a7ee3b4a24efc71b8be4cf47d15400db79a6ac2651424cc`
- Local advisory JSON SHA-256:
  `c7d3e3f3e95d697bf3d29f84dfa89fbfcb2d3d5d8ce5a358e08611c6d30b895f`
- Locality controls SHA-256:
  `4d241741f483069e775218c4488fa5f89b0832770bc9584f65efc2c61f1e7f1b`
- Materialization manifest SHA-256:
  `e816708d7d91141611f66bc3a107ccd411820aeebb32f54ca44e1e4d2b4c4ca9`

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

Interpretation: rank030 is worse than the current exact CPU frontier after the
stricter DQS1/FEC6 same-archive drift correction. Do not dispatch exact eval
for rank030 from this signal.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank030's eureka record was present:

- Next candidate: `pairset_drop_one_rank012_pair0134`
- Lane: `lane_dqs1_pairset_drop_one_rank012_pair0134_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`
- Queue file:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 5 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`

## Next Action

Continue the local-first queue with `pairset_drop_one_rank012_pair0134`.
If its eureka signal is positive, treat that as an exact-auth-anchor dispatch
trigger only after the normal lane-claim and exact CPU/CUDA custody gates; it
still will not be score authority until exact auth eval lands.
