# Codex Findings: DQS1 Rank015 Local Advisory And Rank001 Queue Reroute

UTC: 2026-05-22T22:20:11Z
Agent: Codex
Lane: `lane_dqs1_pairset_drop_one_rank015_pair0068_local_first_20260522`
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank015_pair0068`

Selected pair indices:

`26,59,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank015_pair0068.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank015_pair0068/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank015_pair0068/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank015_pair0068/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank015_pair0068_20260522T222011Z.json`

Custody hashes:

- Archive SHA-256:
  `2776804063ab15e0a760a0d8c525d3b0828113bf1a8774e310a96baf62152f4e`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `10afc8b0d1ccb3dbb6c4553d1c96292d5b4c961db2267becd95a18dc7b4e212b`
- Runtime content tree SHA-256:
  `d1b490028c28bbee027fad3098dbcc94d507ab566e214327afc44dc1e20a9181`
- Inflated output manifest SHA-256:
  `00f0c16a333c3885a78bd116df11ea981eaa408f1c8f9216e8ac849ee857d55d`
- Inflated output aggregate SHA-256:
  `0d3d9565fcd74cc0435a08f21c3e3003a7ca1c51056fef756efb0e68c277c881`

Local advisory metrics:

- Axis: `[macOS-CPU advisory]`
- `canonical_score`: `0.19204028295713674`
- `avg_segnet_dist`: `0.0005599`
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
- Projected contest CPU score: `0.19203028295713673`
- Conservative projected contest CPU score:
  `0.19203328295713673`
- Current auth frontier score:
  `0.19202828295713675`
- Eureka trigger: `false`
- Eureka margin: `-0.000004999999999977245`
- Recommended action: `observe_only`

Interpretation: rank015 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

`tools/build_dqs1_local_first_queue.py --action-summary latest --write`
selected the next unobserved DQS1-safe action after rank015's eureka record was
present:

- Next candidate: `pairset_drop_one_rank001_pair0501`
- Lane: `lane_dqs1_pairset_drop_one_rank001_pair0501_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,520,544,555,588`

Historical succeeded/skipped rows for older candidates remain in the SQLite
queue state as provenance; the active checked-in queue has one ready
`plan_packet` step for rank001.

## Verification

- Existing rank015 worker: `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 4 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `jq '{canonical_score, final_score, archive_size_bytes, avg_posenet_dist, avg_segnet_dist, score_rate_contribution, score_axis, score_claim, promotion_eligible, rank_or_kill_eligible, inflate_elapsed_seconds, evaluate_elapsed_seconds, contest_auth_eval_elapsed_seconds}' experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank015_pair0068/local_cpu_advisory.json`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank015_pair0068 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank015_pair0068/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank015_pair0068_20260522T222011Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --write`

## Next Action

Do not promote rank015. Continue with rank001 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
