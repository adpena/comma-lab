# Codex Findings: DQS1 Rank018 Local Advisory And Rank017 Queue Reroute

UTC: 2026-05-22T21:26:31Z
Agent: Codex
Lane: `lane_dqs1_pairset_drop_one_rank018_pair0588_local_first_20260522`
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank018_pair0588`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank018_pair0588.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank018_pair0588/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank018_pair0588/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank018_pair0588/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank018_pair0588_20260522T212631Z.json`

Custody hashes:

- Archive SHA-256:
  `7c2fbd0037d5f98ee4b52196f70fad54469bb8fda542a2b6b1b925933a38d1fa`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `20921cb811b196e7a2afea99baf679eeeebf65e91754e1703781e62af8c25de9`
- Runtime content tree SHA-256:
  `a8aae492bb1757bf25d83892055e4bcd272b5eb561f36ad3647b1c6b2ed2e32e`
- Inflated output manifest SHA-256:
  `24be01710ca8b3343eb8b79be7820b91afdd3fb2b3ce38f1336d832af0d43248`
- Inflated output aggregate SHA-256:
  `49f5fc4c7caf136c73c4dac567b5ddf072e23a9ceaea04c88926d7c9c995b65d`

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

Interpretation: rank018 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

`tools/build_dqs1_local_first_queue.py --action-summary latest --write`
selected the next unobserved DQS1-safe action:

- Next candidate: `pairset_drop_one_rank017_pair0242`
- Lane: `lane_dqs1_pairset_drop_one_rank017_pair0242_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,229,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`
- Skipped as already locally observed:
  `pairset_drop_one_rank023_pair0440`,
  `pairset_drop_one_rank024_pair0112`,
  `pairset_drop_one_rank018_pair0588`.

Historical succeeded/skipped rows for older candidates remain in the SQLite
queue state as provenance; the active checked-in queue has one ready
`plan_packet` step for rank017.

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 1 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `jq '{canonical_score, final_score, archive_size_bytes, avg_posenet_dist, avg_segnet_dist, score_rate_contribution, score_axis, score_claim, promotion_eligible, rank_or_kill_eligible, inflate_elapsed_seconds, evaluate_elapsed_seconds, contest_auth_eval_elapsed_seconds}' experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank018_pair0588/local_cpu_advisory.json`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank018_pair0588 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank018_pair0588/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank018_pair0588_20260522T212631Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --write`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml status`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml retire-orphans --reason rerouted_to_rank017_after_rank018_observe_only`

## Next Action

Do not promote rank018. Continue with rank017 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
