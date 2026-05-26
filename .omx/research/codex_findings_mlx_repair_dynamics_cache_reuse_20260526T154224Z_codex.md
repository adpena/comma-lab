# Codex Findings: MLX Repair Dynamics Cache Reuse

## Context

The operator pointed out that the canonical K=16 palette signal is heavily frame-0 concentrated and asked to continue all work under MLX while preserving repair dynamics signal without contest-authority leakage.

## What landed

- The repair-dynamics queue integration from `ec4c0584c` now has a concrete MLX cache-reuse run artifact at `.omx/research/frontier_mlx_repair_dynamics_cache_reuse_20260526T153810Z/`.
- The run imported a prior false-authority local CPU advisory cache instead of rerunning local CPU eval.
- The run reused an existing false-authority MLX scorer-input cache and emitted a new `[macOS-MLX research-signal]` scorer response for `packet_member_merge_cbe7d79124ba`.
- The queue carried the repair operator portfolio with 8 repair families and 4 stack candidates, including palette-region waterfill, motion-palette interaction, inverse-basis palette expansion, and full-video residual waterfill stacks.
- The queue worker completed all 7 steps and wrote the work order, repair palette-probe matrix, imported advisory/hash artifacts, MLX scorer response, component response harvest, and worker result summary.

## Evidence

- Queue summary: `.omx/research/frontier_mlx_repair_dynamics_cache_reuse_20260526T153810Z/cache_reuse_queue_summary.json`
- Worker summary: `.omx/research/frontier_mlx_repair_dynamics_cache_reuse_20260526T153810Z/worker_result_summary.json`
- MLX response: `experiments/results/frontier_mlx_repair_dynamics_cache_reuse_20260526T153810Z/frontier_targeted_component_correction/shared_candidate_component_response/candidate_component_response_experiments_results_frontier_final_rate_attack_f_69eeac5a5371ec64/mlx_scorer_response.json`
- Harvest: `experiments/results/frontier_mlx_repair_dynamics_cache_reuse_20260526T153810Z/frontier_targeted_component_correction/frontier_mlx_repair_dynamics_cache_reuse_20260526t153810z/packet_member_merge_cbe7d79124ba/targeted_component_correction_packet_member_merge_cbe7d79124ba_repair_dynamics_frame0_palette_interaction_waterfill/component_correction_response_harvest.json`

## Authority

This is local acquisition signal only. It is not a score claim, promotion claim, rank/kill claim, paid dispatch authority, or exact-eval substitute. The harvest blocked budget spend because paired reference component deltas and receiver materialization gates are still missing.

## Next integration pressure

The frame-0 palette asymmetry should become a measured stack search, not a single repair family. The next executor should generate paired candidate/reference deltas for the repair portfolio stacks under MLX first, then graduate only negative same-axis stack deltas into byte-closed encoder-side materialization proposals.
