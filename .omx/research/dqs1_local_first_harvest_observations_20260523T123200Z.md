# DQS1 Local-First Harvest Observations

- schema: `dqs1_local_first_harvest_observations.v1`
- row_count: `13`
- observed_axis: `macos_cpu_advisory`
- evidence: `[macOS-CPU advisory only]`
- allowed_use: `macos_cpu_advisory_replanning_signal_only`
- score_claim: `False`
- promotion_eligible: `False`
- ready_for_exact_eval_dispatch: `False`
- component_delta_baseline_policy: `local_top32_advisory_scorer_components_with_gap_uleb_archive_size_override`

## Best Local Advisory

- candidate_id: `pairset_drop_two_r029_017_p0259_0242`
- observed_score: `0.19203861709818362`
- score_delta_vs_baseline: `-3.31717906254525e-07`
- source_artifact_path: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_two_r029_017_p0259_0242/local_cpu_advisory.json`

This artifact is planning signal only. Exact contest CPU/CUDA auth eval is still required for any score, rank, promotion, or submission claim.
