# Codex Findings: DQS1 Rank023 Local Advisory And Rank024 Queue Reroute

Date: 2026-05-22T20:37:17Z

## Verdict

`pairset_drop_one_rank023_pair0440` completed the local-first queue through
plan, materialization, locality controls, and macOS CPU advisory scoring. It is
not an exact-eval spend trigger under the current DQS1/FEC6 local CPU drift
calibration. The checked-in queue now routes the next local-first candidate,
`pairset_drop_one_rank024_pair0112`.

This memo is append-only continuation state after
`codex_findings_dqs1_rank019_queue_worker_drift_hardening_20260522T201951Z_codex.md`.

## Rank023 Local Results

- Candidate: `pairset_drop_one_rank023_pair0440`
- Evidence axis: `[macOS-CPU advisory]`
- Local advisory score: `0.19203928295713674`
- Archive bytes: `178559`
- Archive SHA-256:
  `fab595af500cbf5cc4383c8a4224da98f9caec05a18c61c2fcfa66490fe53aa6`
- Inflated raw aggregate SHA-256:
  `fbf4ad51d9ba440691360041275efcacd4c2b67d1d35df35e19204631ea5ab97`
- Runtime content-tree SHA-256:
  `65456af05324e70aebc41069d98c26459d5d90a290ec38e141c5da8470da2583`
- Locality controls: passed, with zero selected-frame, unselected-frame,
  raw-size, or missing-raw mismatches.
- Local CPU advisory JSON:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/local_cpu_advisory.json`

## Drift/Eureka Result

- Eureka artifact:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank023_pair0440_20260522T203717Z.json`
- Calibration: five-anchor stable-core DQS1/FEC6 same-archive SegNet-rounding
  region.
- Point projected contest CPU score: `0.19202928295713673`
- Conservative projected contest CPU score: `0.19203228295713673`
- Current auth frontier score: `0.19202828295713675`
- Eureka margin: `-0.000004`
- Recommendation: `observe_only`
- Authority: false-authority exact-eval spend trigger only, not a score claim,
  not rank/kill evidence, and not promotion-eligible.

## Queue Reroute

- New current queue lane:
  `lane_dqs1_pairset_drop_one_rank024_pair0112_local_first_20260522`
- Queue definition:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`
- Rank024 selected pairs:
  `[26, 59, 68, 98, 109, 134, 151, 167, 229, 242, 257, 259, 296, 320, 327, 371, 376, 378, 412, 430, 440, 459, 467, 479, 492, 496, 501, 520, 544, 555, 588]`

## Commands

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml --state /tmp/pact_dqs1_pairset_rank023_execute.sqlite run-worker --execute --max-steps 3 --idle-sleep-seconds 0 --max-idle-cycles 0 --log-root /tmp/pact_dqs1_pairset_rank023_logs`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml --state /tmp/pact_dqs1_pairset_rank023_execute.sqlite run-worker --execute --max-steps 1 --idle-sleep-seconds 0 --max-idle-cycles 0 --log-root /tmp/pact_dqs1_pairset_rank023_logs`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank023_pair0440_20260522T203717Z.json --candidate-id pairset_drop_one_rank023_pair0440 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --min-margin 0.0`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml --state /tmp/pact_dqs1_pairset_rank024_plan.sqlite run-worker --max-steps 2 --idle-sleep-seconds 0 --max-idle-cycles 0`
- `.venv/bin/python tools/lane_maturity.py validate`
- `git diff --check`

## Next Actions

1. Execute rank024 through plan, materialization, and locality controls.
2. If locality passes, run local macOS CPU advisory and eureka calibration.
3. Only if a calibrated eureka signal is positive, route an exact contest CPU or
   CUDA anchor through the existing lane-claim and recovery gates.
