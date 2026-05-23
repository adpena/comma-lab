# Codex Findings: DQS1 Locality Timeout Queue Closure

UTC: 2026-05-23T18:57:43Z
Lane: `lane_codex_dqs1_locality_control_timeout_hardening_20260523`
Authority axis: local queue control / false authority

## Finding

The DQS1 locality timeout hardening lane is empirically closed for the current
rank023/rank024 active queue definition. The active `dqs1_pairset_local_first`
definition has 16/16 current steps succeeded and no ready current-definition
work. The 188 orphaned rows visible in queue status are historical/nonblocking:
`reconcile-state` found `blocking_orphan_count_before=0`,
`retired_step_count=0`, and `blocking_orphan_count_after=0`.

## Evidence

- Queue performance summary:
  `.omx/research/dqs1_locality_timeout_hardening_queue_performance_20260523T1855Z.json`
- Orphan reconciliation audit:
  `.omx/research/dqs1_local_first_orphan_reconcile_20260523T1856Z.json`
- rank024 locality controls:
  `/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first/materialized/drop_rank024_pair0112/locality_controls.json`
- rank024 locality progress:
  `/Volumes/VertigoDataTier/pact/experiments/results/dqs1_local_first/materialized/drop_rank024_pair0112/locality_work/locality_controls_progress.jsonl`
- rank024 eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank024_pair0112_20260523T143352Z.json`

## Measured Closure

- `rank024` locality controls passed with only the expected false-authority
  blockers: `promotion_requires_exact_contest_auth_eval` and
  `score_claim_false_locality_control_only`.
- `rank024` reused manifest-verified parent/selective/global inflates under
  `max_parallelism=3`.
- `rank024` locality phase timings: `inflate_targets=0.34118987497640774s`;
  `compare_inflated_outputs=6.936644040979445s`.
- Queue performance for `local_io_heavy`: `run_count=5`,
  `success_count=5`, `failure_count=0`,
  `elapsed_seconds_mean=145.88102978320094`,
  `elapsed_seconds_max=411.09687350003514`.
- `rank024` eureka remained observe-only:
  `eureka_trigger=false`,
  `eureka_margin=-0.0000024999999999886224`,
  `recommended_action=observe_only`.

## Lane Update

Marked `real_archive_empirical` for
`lane_codex_dqs1_locality_control_timeout_hardening_20260523` using the
rank024 locality-controls artifact. The lane is now L2.

## Queue Hygiene

Do not run `.omx/state/experiment_queue_dqs1_pairset_local_first_storage.sqlite`
for this lane. It is a stale storage-queue state for
`dqs1_pairset_local_first_storage`, not the current
`dqs1_pairset_local_first` active definition. Advancing beyond rank023/rank024
requires rebuilding/selecting the next queue target first.

## Authority

No score is claimed from this tranche. No exact-auth dispatch was attempted.
Local CPU advisory, MLX response, locality-control, and queue-performance rows
remain false-authority planning/control evidence only.
