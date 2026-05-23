# Codex Findings: DQS1 Rank011 Local Advisory And Rank009 Queue Reroute

UTC: 2026-05-23T01:55:00Z

## Evidence Axis

This memo records `[macOS-CPU advisory]` local evidence and
`[contest-CPU drift-projected; false authority]` spend-triage projection. It is
not a score claim, not promotion evidence, and not rank/kill evidence.

## Candidate

Candidate: `pairset_drop_one_rank011_pair0555`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,588`

Local-first queue steps completed:

- `plan_packet`
- `materialize`
- `locality_controls`
- `local_cpu_advisory`
- `local_cpu_contest_drift_eureka`

Durable/ignored source artifacts:

- plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank011_pair0555.json`
- materialization:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank011_pair0555/materialization_manifest.json`
- locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank011_pair0555/locality_controls.json`
- local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank011_pair0555/local_cpu_advisory.json`
- eureka:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank011_pair0555_20260523T013925Z.json`
- harvest:
  `.omx/research/dqs1_local_first_harvest_pairset_drop_one_rank011_pair0555_20260523T015500Z.json`

## Local Advisory Result

- Local score: `0.19204028295713674` `[macOS-CPU advisory]`
- Archive bytes: `178559`
- PoseNet distortion: `0.00002943`
- SegNet distortion: `0.00055990`
- Rate: `0.0047558043524216715`
- Archive SHA-256:
  `8c36698203213172c7442faaf0a430629dd8af45c09f71ef8e4f419bfe2c7afa`
- Runtime tree SHA-256:
  `c090b19748766a551b930c7b34b40fbef837e6e10a4d52d41d08cbd4d6ddbcdd`
- Runtime content tree SHA-256:
  `6c6ae7fd109d5356ca16618bd8ff660d67864ab38d4878106160fc528b49ff7f`
- Inflated output manifest SHA-256:
  `b935b7c18899652d5290bc4ca87ff5f783d7709542546790e61603404b962162`
- Inflated output aggregate SHA-256:
  `baf5611c7f5034c8d633f7bec5cfc9809b17d7f6da534c0aa844047094cf93f4`
- Local advisory JSON SHA-256:
  `b22bd94d7b9d0f038b23f132a7b66161e398ded205e3897fdfcefd13798e10fd`
- Locality controls SHA-256:
  `c479c74b8869ef72dcb0eb0779b67203c7786a561c75b4323c311a5dde07e60b`
- Materialization manifest SHA-256:
  `86dc74a493ece95373cdb57763dd8647b2786f88e8309166a96c13e0041f89a4`

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
- Eureka trigger: `false`
- Recommended action: `observe_only`

Interpretation: rank011 ties rank012 locally and remains worse than the
current exact CPU frontier after the stricter DQS1/FEC6 same-archive drift
correction. Do not dispatch exact eval for rank011 from this signal.

## Queue Reroute

The current validated local-first queue is now routed to:

- Next candidate: `pairset_drop_one_rank009_pair0459`
- Lane: `lane_dqs1_pairset_drop_one_rank009_pair0459_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,467,479,492,496,501,520,544,555,588`
- Queue file:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml status`
- `.venv/bin/python tools/harvest_dqs1_local_first_result.py --queue .omx/tmp_rank011_queue_for_harvest_20260523T015500Z.json --timestamp 20260523T015500Z --harvest-out .omx/research/dqs1_local_first_harvest_pairset_drop_one_rank011_pair0555_20260523T015500Z.json`
- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_local_cpu_contest_drift.py src/tac/tests/test_experiment_queue.py`

## Next Action

Continue the local-first queue with `pairset_drop_one_rank009_pair0459`.
If its eureka signal is positive, treat that as an exact-auth-anchor dispatch
trigger only after the normal lane-claim and exact CPU/CUDA custody gates; it
still will not be score authority until exact auth eval lands.
