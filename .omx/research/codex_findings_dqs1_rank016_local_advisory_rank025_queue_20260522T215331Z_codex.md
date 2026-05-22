# Codex Findings: DQS1 Rank016 Local Advisory And Rank025 Queue Reroute

UTC: 2026-05-22T21:53:31Z
Agent: Codex
Lane: `lane_dqs1_pairset_drop_one_rank016_pair0229_local_first_20260522`
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank016_pair0229`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank016_pair0229.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank016_pair0229/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank016_pair0229/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank016_pair0229/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank016_pair0229_20260522T215331Z.json`

Custody hashes:

- Archive SHA-256:
  `b49ea7e29c22dd67b18a8cb59c4672adc6f9748b6681ce1327e16d258858dbf6`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `55eca45c3ef464ef617544c5072548ed3ad8c9fcedad8bca8bb0b92f29d7a37b`
- Runtime content tree SHA-256:
  `92ee617fbd63a47c887d68f106a4b47f370568bcdda7df67af605990ec85b2b1`
- Inflated output manifest SHA-256:
  `5936f4bae5bfc68c1ef228bdaf0690aaee885adca805941b2fd2b19bfba0726c`
- Inflated output aggregate SHA-256:
  `22aeb9b5652297597ad801ae26f1483189d991d5c5c622ae2222412c75b0df61`

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

Interpretation: rank016 is worse than the current exact CPU frontier after
local CPU drift correction and after the conservative guard band. It is not an
exact CPU/CUDA spend trigger.

## Queue Reroute

`tools/build_dqs1_local_first_queue.py --action-summary latest --write`
selected the next unobserved DQS1-safe action:

- Next candidate: `pairset_drop_one_rank025_pair0026`
- Lane: `lane_dqs1_pairset_drop_one_rank025_pair0026_local_first_20260522`
- Selected pair indices:
  `59,68,98,109,112,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`
- Skipped as already locally observed:
  `pairset_drop_one_rank023_pair0440`,
  `pairset_drop_one_rank024_pair0112`,
  `pairset_drop_one_rank018_pair0588`,
  `pairset_drop_one_rank017_pair0242`,
  `pairset_drop_one_rank016_pair0229`.

Historical succeeded/skipped rows for older candidates remain in the SQLite
queue state as provenance; the active checked-in queue has one ready
`plan_packet` step for rank025.

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 3 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 1 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `jq '{canonical_score, final_score, archive_size_bytes, avg_posenet_dist, avg_segnet_dist, score_rate_contribution, score_axis, score_claim, promotion_eligible, rank_or_kill_eligible, inflate_elapsed_seconds, evaluate_elapsed_seconds, contest_auth_eval_elapsed_seconds}' experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank016_pair0229/local_cpu_advisory.json`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank016_pair0229 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank016_pair0229/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank016_pair0229_20260522T215331Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --write`

## Next Action

Do not promote rank016. Continue with rank025 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
