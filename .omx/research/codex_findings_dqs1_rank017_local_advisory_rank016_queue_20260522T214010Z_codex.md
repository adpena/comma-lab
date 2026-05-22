# Codex Findings: DQS1 Rank017 Local Advisory And Rank016 Queue Reroute

UTC: 2026-05-22T21:40:10Z
Agent: Codex
Lane: `lane_dqs1_pairset_drop_one_rank017_pair0242_local_first_20260522`
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank017_pair0242`

Selected pair indices:

`26,59,68,98,109,112,134,151,167,229,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank017_pair0242.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank017_pair0242/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank017_pair0242/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank017_pair0242/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank017_pair0242_20260522T214010Z.json`

Custody hashes:

- Archive SHA-256:
  `f1de4e213b6e02f81d1e1a5c764f8942bf0946fb9d6544c0b268fd76b00e8d76`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `de95f2319ef584b47555acfecbcc94033898f349287ca214db034a9576b3e87a`
- Runtime content tree SHA-256:
  `cbc55d544a0fc16dd698b2b9360de96954df98fc32033e778fff0fb863310381`
- Inflated output manifest SHA-256:
  `0755c20ae1b8714c6a750753580a84c50e54062bd0b9799028f1e000a0000dfd`
- Inflated output aggregate SHA-256:
  `2c2de94fde2ec1427058357767c87bdf2ba714f34ec4e0d2d2c511561dec3830`

Local advisory metrics:

- Axis: `[macOS-CPU advisory]`
- `canonical_score`: `0.19203828295713676`
- `avg_segnet_dist`: `0.00055988`
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
- Projected contest CPU score: `0.19202828295713675`
- Conservative projected contest CPU score:
  `0.19203128295713676`
- Current auth frontier score:
  `0.19202828295713675`
- Eureka trigger: `false`
- Eureka margin: `-0.0000030000000000030003`
- Recommended action: `observe_only`

Interpretation: rank017 only ties the current exact CPU frontier after the
local CPU drift point correction, and loses after the conservative guard band.
It is not an exact CPU/CUDA spend trigger.

## Queue Reroute

`tools/build_dqs1_local_first_queue.py --action-summary latest --write`
selected the next unobserved DQS1-safe action:

- Next candidate: `pairset_drop_one_rank016_pair0229`
- Lane: `lane_dqs1_pairset_drop_one_rank016_pair0229_local_first_20260522`
- Selected pair indices:
  `26,59,68,98,109,112,134,151,167,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`
- Skipped as already locally observed:
  `pairset_drop_one_rank023_pair0440`,
  `pairset_drop_one_rank024_pair0112`,
  `pairset_drop_one_rank018_pair0588`,
  `pairset_drop_one_rank017_pair0242`.

Historical succeeded/skipped rows for older candidates remain in the SQLite
queue state as provenance; the active checked-in queue has one ready
`plan_packet` step for rank016.

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 3 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 1 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `jq '{canonical_score, final_score, archive_size_bytes, avg_posenet_dist, avg_segnet_dist, score_rate_contribution, score_axis, score_claim, promotion_eligible, rank_or_kill_eligible, inflate_elapsed_seconds, evaluate_elapsed_seconds, contest_auth_eval_elapsed_seconds}' experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank017_pair0242/local_cpu_advisory.json`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank017_pair0242 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank017_pair0242/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank017_pair0242_20260522T214010Z.json --min-margin 0.0`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --write`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml status`

## Next Action

Do not promote rank017. Continue with rank016 through the canonical local-first
queue, then apply the same drift/eureka gate before any exact-eval spend.
