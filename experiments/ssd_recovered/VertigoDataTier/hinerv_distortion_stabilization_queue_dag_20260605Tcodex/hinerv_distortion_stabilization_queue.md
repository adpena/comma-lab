# HiNeRV Distortion Stabilization Queue

- schema: `hinerv_distortion_stabilization_queue.v1`
- lane: `lane_hinerv_distortion_stabilization_20260605`
- archive bytes: `181295`
- receiver proof ready: `True`
- dynamic-range/scorer-input stable: `False`
- blocked DAG nodes: `6`

## DAG Nodes

### `dynamic_range_scorer_input_stabilization`
- status: `blocked_until_prerequisites`
- blocked: `True`
- depends on: ``
- blockers:
  - `hi_nerv_score_aware_training_direct_live_segnet_candidate_argmax_collapsed`
  - `candidate_segnet_last_rgb_dynamic_range_too_low`
  - `candidate_segnet_argmax_disagreement_too_high`
  - `hi_nerv_post_export_receiver_cache_quality_gate_failed`
  - `scorer_input_segnet_last_rgb_std_ratio_lt_0_25`
  - `mlx_renderer_prefilter_scorer_input_out_of_distribution`
  - `hinerv_checkpoint_fit_scale_gate_failed`
  - `candidate_segnet_last_rgb_far_from_reference_fit_gate`
  - `hinerv_receiver_cache_quality_gate_failed`

### `byte_closed_archive_export`
- status: `ready_no_authority`
- blocked: `False`
- depends on: `dynamic_range_scorer_input_stabilization`
- blockers:

### `receiver_archive_replay_proof`
- status: `ready_no_authority`
- blocked: `False`
- depends on: `byte_closed_archive_export`
- blockers:

### `ema_archive_in_loop_selection`
- status: `blocked_until_prerequisites`
- blocked: `True`
- depends on: `receiver_archive_replay_proof`
- blockers:
  - `hinerv_candidate_curriculum_recon_pixel_weight_missing`
  - `hi_nerv_pr95_staged_curriculum_missing`
  - `hi_nerv_archive_in_loop_byte_oracle_missing`

### `decoder_weight_waterfill_recon_pixel_proof`
- status: `blocked_until_prerequisites`
- blocked: `True`
- depends on: `ema_archive_in_loop_selection`
- blockers:
  - `contest_cpu_cuda_exact_eval_not_executed`
  - `decoder_weight_saliency_replay_required_for_authority`
  - `decoder_weight_saliency_replay_has_blockers`
  - `score_loss_proxy_outside_allocator_linearization_basin`
  - `receiver_replay_is_scorer_loss_saliency_not_archive_score`
  - `decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin`
  - `hinerv_recon_pixel_weight_proof_missing`

### `full_video_mlx_prefilter_gate`
- status: `blocked_until_prerequisites`
- blocked: `True`
- depends on: `dynamic_range_scorer_input_stabilization, decoder_weight_waterfill_recon_pixel_proof`
- blockers:
  - `full_video_mlx_scorer_replay_not_attached`
  - `sampled_mlx_prefilter_requires_full_video_rerun`
  - `mlx_profile_not_full_video_executed`
  - `mlx_profile_pair_count_below_full_video`
  - `scorer_input_segnet_last_rgb_std_ratio_lt_0_25`
  - `mlx_renderer_prefilter_scorer_input_out_of_distribution`
  - `mlx_cache_quality_gate_is_false_authority`
  - `candidate_segnet_last_rgb_far_from_reference_fit_gate`

### `local_cpu_replay_gate`
- status: `blocked_until_prerequisites`
- blocked: `True`
- depends on: `full_video_mlx_prefilter_gate`
- blockers:
  - `hi_nerv_full_video_local_prefilter_missing`
  - `hi_nerv_local_cpu_replay_gate_missing`
  - `hi_nerv_local_cpu_replay_not_passed`

### `exact_cpu_cuda_dispatch_gate`
- status: `blocked_until_prerequisites`
- blocked: `True`
- depends on: `local_cpu_replay_gate`
- blockers:
  - `contest_cpu_cuda_exact_eval_blocked_until_local_replay_wins`
