# Codex Findings: DQS1 Rank028 Local Advisory And Rank030 Queue Reroute

UTC: 2026-05-23T01:07:02Z

## Evidence Axis

This memo records `[macOS-CPU advisory]` local evidence and
`[contest-CPU drift-projected; false authority]` spend-triage projection. It is
not a score claim, not promotion evidence, and not rank/kill evidence.

## Candidate

Candidate: `pairset_drop_one_rank028_pair0257`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue steps completed:

- `plan_packet`
- `materialize`
- `locality_controls`
- `local_cpu_advisory`
- `local_cpu_contest_drift_eureka`

Durable/ignored source artifacts:

- plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank028_pair0257.json`
- materialization:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank028_pair0257/materialization_manifest.json`
- locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank028_pair0257/locality_controls.json`
- local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank028_pair0257/local_cpu_advisory.json`
- eureka:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank028_pair0257_20260523T004936Z.json`

## Local Advisory Result

- Local score: `0.19204028295713674` `[macOS-CPU advisory]`
- Archive bytes: `178559`
- PoseNet distortion: `0.00002943`
- SegNet distortion: `0.00055990`
- Rate: `0.0047558043524216715`
- Archive SHA-256:
  `9fb1fc6e8eef29b5fa266cb2d817a06361cf906f2224901e778ddab222b812e9`
- Runtime tree SHA-256:
  `262485bae823cf6f10f298617340b6963cfa271f400d983e1cf57c6797caac9b`
- Runtime content tree SHA-256:
  `cd12046f3f04dc1a60132c22eddff7d80c4058731949ca6a1bca0b71c700e034`
- Inflated output manifest SHA-256:
  `a44afa7da5f057c0bfc85133a41fb9f9b0083ec96fac41fa70290979b9fe6967`
- Inflated output aggregate SHA-256:
  `dab2d26d738876e2e92108b6615b32f570da59cdf7a553c8b2bb06c961ce6bdf`
- Local advisory JSON SHA-256:
  `c10cb4afcb031be0b6fa563c5caeabbac75e2e2bebcafaf019667a3da272f048`
- Locality controls SHA-256:
  `0940a08b35491fb646422dccc4a6782685b1b43a00307f2a33e7f26cc0447e22`
- Materialization manifest SHA-256:
  `1666248b3cb6f56853706e30f09e72a07e464778c527b66656529efdd04ece83`

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

Interpretation: rank028 is worse than the current exact CPU frontier after the
stricter DQS1/FEC6 same-archive drift correction. Do not dispatch exact eval
for rank028 from this signal.

## Queue Reroute

The top-32 action summary selected the next unobserved DQS1-safe action after
rank028's eureka record was present:

- Next candidate: `pairset_drop_one_rank030_pair0412`
- Lane: `lane_dqs1_pairset_drop_one_rank030_pair0412_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,430,440,459,467,479,492,496,501,520,544,555,588`
- Queue file:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 5 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260522T224218Z_pairset_component_rank001_exhausted_top32/action_summary.json --write`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`

## Next Action

Continue the local-first queue with `pairset_drop_one_rank030_pair0412`.
If its eureka signal is positive, treat that as an exact-auth-anchor dispatch
trigger only after the normal lane-claim and exact CPU/CUDA custody gates; it
still will not be score authority until exact auth eval lands.
