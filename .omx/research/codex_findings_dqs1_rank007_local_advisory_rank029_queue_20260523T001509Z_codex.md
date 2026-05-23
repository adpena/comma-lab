# Codex Findings: DQS1 Rank007 Local Advisory And Rank029 Queue Reroute

UTC: 2026-05-23T00:15:09Z

## Evidence Axis

This memo records `[macOS-CPU advisory]` local evidence and
`[contest-CPU drift-projected; false authority]` spend-triage projection. It is
not a score claim, not promotion evidence, and not rank/kill evidence.

## Candidate

Candidate: `pairset_drop_one_rank007_pair0059`

Local-first queue steps completed:

- `plan_packet`
- `materialize`
- `locality_controls`
- `local_cpu_advisory`

Durable/ignored source artifacts:

- plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank007_pair0059.json`
- materialization:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank007_pair0059/materialization_manifest.json`
- locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank007_pair0059/locality_controls.json`
- local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank007_pair0059/local_cpu_advisory.json`
- eureka:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank007_pair0059_20260523T001509Z.json`

## Local Advisory Result

- Local score: `0.19204028295713674` `[macOS-CPU advisory]`
- Archive bytes: `178559`
- PoseNet distortion: `0.00002943`
- SegNet distortion: `0.00055990`
- Rate: `0.0047558043524216715`
- Archive SHA-256:
  `c027f055f854ef65bf012d2b767d5fd165e92cf60099ca2bd7f964cef635670f`
- Inflated output manifest SHA-256:
  `addde714aff40d5aa0e0101c3c9143c2fd5b11065f8abc5422ea3ab29a828611`
- Local advisory JSON SHA-256:
  `e3e727de1b3a80bd3fe6d00d448e877b5775d0a515790923d500a65a9d25a2ac`
- Locality controls SHA-256:
  `8bfbd9a8d65a39830a9bd44ba1822c954b3456249c02cdea89556686b5b9ba1e`
- Materialization manifest SHA-256:
  `d268f795722882327ab72d1b46f79ee3f368d48cdadb65a7e8254ad3e3a7de29`

## Drift/Eureka Projection

Calibration:
`.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`

- Bias local minus contest: `0.000010000000000010001`
- Guard band: `0.000003`
- Calibration confidence: `stable_core`
- Anchor count: `5`
- Projected contest score:
  `0.19203028295713673`
  `[contest-CPU drift-projected; false authority]`
- Conservative projected contest score:
  `0.19203328295713673`
  `[contest-CPU drift-projected; false authority]`
- Current exact CPU frontier anchor:
  `0.19202828295713675`
- Eureka margin: `-0.000004999999999977245`
- Eureka trigger: `false`
- Recommended action: `observe_only`

Interpretation: rank007 is worse than the current exact CPU frontier after
stable-core DQS1/FEC6 same-archive drift correction. Do not dispatch exact eval
for rank007 from this signal.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank007's eureka record was present:

- Next candidate: `pairset_drop_one_rank029_pair0259`
- Lane: `lane_dqs1_pairset_drop_one_rank029_pair0259_local_first_20260522`
- Queue file:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_local_cpu_contest_drift.py`

## Next Action

Continue the local-first queue with `pairset_drop_one_rank029_pair0259`.
If its eureka signal is positive, treat that as an exact-auth-anchor dispatch
trigger only after the normal lane-claim and exact CPU/CUDA custody gates; it
still will not be score authority until exact auth eval lands.
