# DQS1 Local-First Harvest Observations

- schema: `dqs1_local_first_harvest_observations.v1`
- row_count: `2`
- observed_axis: `macos_cpu_advisory`
- evidence: `[macOS-CPU advisory only]`
- allowed_use: `macos_cpu_advisory_replanning_signal_only`
- score_claim: `False`
- promotion_eligible: `False`
- ready_for_exact_eval_dispatch: `False`
- component_delta_baseline_policy: `local_top32_advisory_scorer_components_with_gap_uleb_archive_size_override`

## Best Local Advisory

- candidate_id: `pairset_drop_one_rank007_pair0059`
- observed_score: `0.19204028295713674`
- score_delta_vs_baseline: `1.3341410468603598e-06`
- source_artifact_path: `/Volumes/VertigoDataTier/pact_experiments/frontier_rate_attack_feedback_cycle_20260525T221326Z/materialized/drop_rank007_pair0059/local_cpu_advisory.json`

This artifact is planning signal only. Exact contest CPU/CUDA auth eval is still required for any score, rank, promotion, or submission claim.
