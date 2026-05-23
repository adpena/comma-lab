# Codex Findings: DQS1 Rank012 Local Advisory And Rank011 Queue Reroute

UTC: 2026-05-23T01:38:13Z

## Evidence Axis

This memo records `[macOS-CPU advisory]` local evidence and
`[contest-CPU drift-projected; false authority]` spend-triage projection. It is
not a score claim, not promotion evidence, and not rank/kill evidence.

## Candidate

Candidate: `pairset_drop_one_rank012_pair0134`

Selected pair indices:

`26,59,68,98,109,112,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue steps completed:

- `plan_packet`
- `materialize`
- `locality_controls`
- `local_cpu_advisory`
- `local_cpu_contest_drift_eureka`

Durable/ignored source artifacts:

- plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank012_pair0134.json`
- materialization:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank012_pair0134/materialization_manifest.json`
- locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank012_pair0134/locality_controls.json`
- local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank012_pair0134/local_cpu_advisory.json`
- eureka:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank012_pair0134_20260523T012437Z.json`

## Local Advisory Result

- Local score: `0.19204028295713674` `[macOS-CPU advisory]`
- Archive bytes: `178559`
- PoseNet distortion: `0.00002943`
- SegNet distortion: `0.00055990`
- Rate: `0.0047558043524216715`
- Archive SHA-256:
  `efc6cec8fbe0327c43fe7aecbf96e2ce05abc9bc0cd12eb8339e28e6a3828d71`
- Runtime tree SHA-256:
  `b4bed9ea5814093c55fff9fe8e177987b330a47206ce56f59a39fec43ed72139`
- Runtime content tree SHA-256:
  `e7429623793043962edce303f06541b044ef0913bdb3984ae2dbda39bb81b819`
- Inflated output manifest SHA-256:
  `da33b629e8c229b57e43c8b942441a6f151527a9e326fb1ba463e094fd6f3480`
- Inflated output aggregate SHA-256:
  `3b03d20847f633d8c8e5700206f90e77685b2001752066992ad08ed1bd9aa365`
- Local advisory JSON SHA-256:
  `abada366acdae5353e0c6bbcedbfc05dcd666d7856494c9cb6241c8d50abeaa4`
- Locality controls SHA-256:
  `caefffa1cc7d08c39ae34139ed6e4b47e854c0219adbbf20fe3b15567365ffd6`
- Materialization manifest SHA-256:
  `2acc8cc69b3d50f3b75a76172dc6c2c436718cb698b315bc302cf4dbcea08007`

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

Interpretation: rank012 is worse than the current exact CPU frontier after the
stricter DQS1/FEC6 same-archive drift correction. Do not dispatch exact eval
for rank012 from this signal.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank012's eureka record was present:

- Next candidate: `pairset_drop_one_rank011_pair0555`
- Lane: `lane_dqs1_pairset_drop_one_rank011_pair0555_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,588`
- Queue file:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 5 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`

## Next Action

Continue the local-first queue with `pairset_drop_one_rank011_pair0555`.
If its eureka signal is positive, treat that as an exact-auth-anchor dispatch
trigger only after the normal lane-claim and exact CPU/CUDA custody gates; it
still will not be score authority until exact auth eval lands.
