# Codex Findings: DQS1 Rank023 Local Controls And Rank024 Queue Reroute

Date: 2026-05-22T20:41:09Z

## Verdict

Rank023/pair0440 completed the local-first queue through plan, materialization,
raw-output locality controls, and macOS CPU advisory scoring. It remains a
false-authority local advisory signal only. The calibrated drift/eureka gate did
not trigger exact contest eval spend, so the checked-in local-first queue is
rerouted to rank024/pair0112.

No score or promotion claim is made from this memo.

## Rank023 Materialization

- Candidate: `pairset_drop_one_rank023_pair0440`
- Archive path:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/submission_dir/archive.zip`
- ZIP SHA-256:
  `fab595af500cbf5cc4383c8a4224da98f9caec05a18c61c2fcfa66490fe53aa6`
- ZIP bytes: `178559`
- ZIP member: `x`
- Member SHA-256:
  `969a11705f5f07cece54caf6ed6a17fb1aeee56f59c14631227c81b3514d938c`
- Member bytes: `178459`
- DQS1 payload bytes: `42`
- DQS1 payload SHA-256:
  `b91511ff7a4b5e1181606d7e9ad41babc4ff2fdc1e094b55ab40192d00b6f198`
- Selected pairs:
  `[26, 59, 68, 98, 109, 112, 134, 151, 167, 229, 242, 257, 259, 296, 320, 327, 371, 376, 378, 412, 430, 459, 467, 479, 492, 496, 501, 520, 544, 555, 588]`
- Runtime contract: copies FEC6 runtime, derives mutated decoder from base
  decoder plus DQS1 patch, stores no second full decoder, applies selector
  after selective frame stitching.

## Locality Controls

- Artifact:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/locality_controls.json`
- `locality_controls_passed`: `true`
- Selected-frame mismatch count: `0`
- Unselected-frame mismatch count: `0`
- Raw-size mismatch count: `0`
- Missing-raw count: `0`
- Selective selected-frame SHA-256:
  `38b4eac1cffeb178d0adf80928be60af5225b4ef793d22ff7cd0bc84236b2ccd`
- Global-mutated selected-frame SHA-256:
  `38b4eac1cffeb178d0adf80928be60af5225b4ef793d22ff7cd0bc84236b2ccd`
- Selective unselected-frame SHA-256:
  `5d2947431d6929196928c21b1f2b65b9609b17d279f6cc4e685b18797d4b2c66`
- Parent unselected-frame SHA-256:
  `5d2947431d6929196928c21b1f2b65b9609b17d279f6cc4e685b18797d4b2c66`
- Selective raw SHA-256:
  `117e0afacdab883203ed4ce086a5db86f775e9c36dcfa208cd27262e13b782ba`

## Local Advisory

- Axis: `[macOS-CPU advisory]`
- Artifact:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/local_cpu_advisory.json`
- Canonical score: `0.19203928295713674`
- Archive bytes: `178559`
- Avg PoseNet distortion: `0.00002943`
- Avg SegNet distortion: `0.00055989`
- Rate contribution: `0.11889510881054179`
- Runtime content-tree SHA-256:
  `65456af05324e70aebc41069d98c26459d5d90a290ec38e141c5da8470da2583`
- Inflated raw aggregate SHA-256:
  `fbf4ad51d9ba440691360041275efcacd4c2b67d1d35df35e19204631ea5ab97`
- Inflate elapsed seconds: `61.087051333044656`
- Evaluate elapsed seconds: `446.17849470902`
- Total elapsed seconds: `508.72778658301104`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `rank_or_kill_eligible`: `false`

## Drift And Eureka Gate

- Artifact:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank023_pair0440_20260522T203717Z.json`
- Calibration artifact:
  `.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`
- Target axis: `[contest-CPU]`
- Auth frontier score used for spend gate: `0.19202828295713675`
- Projected contest CPU score: `0.19202928295713673`
- Conservative projected contest CPU score: `0.19203228295713673`
- Eureka margin: `-0.000003999999999976245`
- Recommended action: `observe_only`
- Dispatch attempted: `false`
- Ready for exact-eval dispatch: `false`
- Blocking reasons include planning-only queue semantics, exact-eval readiness
  gate, lane-dispatch-claim requirement, false-authority eureka status, and
  exact contest CPU requirement before any frontier claim.

## Queue Reroute

- Previous local-first queue lane:
  `lane_dqs1_pairset_drop_one_rank023_pair0440_local_first_20260522`
- New local-first queue lane:
  `lane_dqs1_pairset_drop_one_rank024_pair0112_local_first_20260522`
- Queue definition:
  `configs/experiment_queues/dqs1_pairset_local_first.yaml`
- Rank024 selected pairs:
  `[26, 59, 68, 98, 109, 134, 151, 167, 229, 242, 257, 259, 296, 320, 327, 371, 376, 378, 412, 430, 440, 459, 467, 479, 492, 496, 501, 520, 544, 555, 588]`
- The rank024 lane is registered at L1 with `impl_complete` evidence pointing
  at the queue file and `strict_preflight` evidence pointing at queue tests.

## Commands

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 3 --idle-sleep-seconds 0 --max-idle-cycles 1`
- `.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/submission_dir/archive.zip --inflate-sh experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/submission_dir/inflate.sh --upstream-dir upstream --video-names-file upstream/public_test_video_names.txt --device cpu --work-dir experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/local_cpu_advisory_work --json-out experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/local_cpu_advisory.json --inflate-timeout 600 --evaluate-timeout 900 --keep-work-dir`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank023_pair0440 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank023_pair0440/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank023_pair0440_20260522T203717Z.json --min-margin 0.0`

## Next Actions

1. Run rank024/pair0112 through local plan, materialization, and locality
   controls.
2. If locality controls pass, run macOS CPU advisory and the calibrated
   eureka gate.
3. Do not dispatch exact contest eval unless the eureka gate flips from
   `observe_only` to a dispatch-worthy result and the lane dispatch claim path
   is opened.
