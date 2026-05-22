# Codex Findings: DQS1 Rank025 Local Advisory And Rank015 Queue Reroute

UTC: 2026-05-22T22:06:21Z
Agent: Codex
Lane: `lane_dqs1_pairset_drop_one_rank025_pair0026_local_first_20260522`
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank025_pair0026`

Selected pair indices:

`59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank025_pair0026.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank025_pair0026/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank025_pair0026/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank025_pair0026/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank025_pair0026_20260522T220621Z.json`

Custody hashes:

- Archive SHA-256:
  `9beb0dfc02062fa30d02c8ec6ec8747d82ba343c994bf8d2d13eabebdb789f8a`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `d9194c34a83563416fac876ad4cf75a4baa1837461c166e0f18b7eabc8be5bb5`
- Runtime content tree SHA-256:
  `15140fe680c880260207d80b913231fdd85fa303c54330254552f302d78c5a73`
- Inflated output manifest SHA-256:
  `c45110fdc62566d5245b4282016a32e10be230d00ec164c143c51286cdfcfc10`
- Inflated output aggregate SHA-256:
  `73e174a73bc32e98aaffabff13287d45fb5474b290ac841e056268ed80ad2c0d`

Local advisory metrics:

- Axis: `[macOS-CPU advisory]`
- `canonical_score`: `0.19203928295713674`
- `avg_segnet_dist`: `0.00055989`
- `avg_posenet_dist`: `0.00002943`
- `rate_unscaled`: `0.0047558043524216715`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `rank_or_kill_eligible`: `false`

Drift/eureka interpretation:

- Calibration: `.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`
- Trust region: `dqs1_fec6_like_same_archive_segnet_rounding`
- Calibration bias local-minus-contest: `0.000010000000000010001`
- Guard band: `0.000003`
- Projected contest CPU score: `0.19202928295713673`
- Conservative projected contest CPU score:
  `0.19203228295713673`
- Current auth frontier score:
  `0.19202828295713675`
- Eureka trigger: `false`
- Eureka margin: `-0.000003999999999976245`
- Recommended action: `observe_only`

Interpretation: rank025 remains above the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

`tools/build_dqs1_local_first_queue.py --action-summary latest --write`
selected the next unobserved DQS1-safe action:

- Next candidate: `pairset_drop_one_rank015_pair0068`
- Lane: `lane_dqs1_pairset_drop_one_rank015_pair0068_local_first_20260522`
- Selected pair indices:
  `26,59,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`
- Skipped as already locally observed:
  `pairset_drop_one_rank023_pair0440`,
  `pairset_drop_one_rank024_pair0112`,
  `pairset_drop_one_rank018_pair0588`,
  `pairset_drop_one_rank017_pair0242`,
  `pairset_drop_one_rank016_pair0229`,
  `pairset_drop_one_rank025_pair0026`.

Historical succeeded/skipped rows for older candidates remain in the SQLite
queue state as provenance; the active checked-in queue has one ready
`plan_packet` step for rank015.

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 3 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 1 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `jq '{canonical_score, final_score, archive_size_bytes, avg_posenet_dist, avg_segnet_dist, score_rate_contribution, score_axis, score_claim, promotion_eligible, rank_or_kill_eligible, inflate_elapsed_seconds, evaluate_elapsed_seconds, contest_auth_eval_elapsed_seconds}' experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank025_pair0026/local_cpu_advisory.json`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank025_pair0026 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank025_pair0026/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank025_pair0026_20260522T220621Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --write`

## Next Action

Do not promote rank025. Continue with rank015 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
