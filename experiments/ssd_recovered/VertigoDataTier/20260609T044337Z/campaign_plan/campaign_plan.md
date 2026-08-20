# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `6`
Launchable local rows: `0`
Blocked rows: `6`
Score claim: `False`
Ready for exact dispatch: `False`

## Authority Bindings

- upstream_evaluate: `upstream/evaluate.py`
  baseline_to_beat: `full_pr95_fidelity_or_better_on_exact_upstream_evaluate_axes`
  canonical_rate_denominator_bytes: `37545489`
  pose_marginal_formula: `5/sqrt(10*d_pose)`
- tilde_oss_policy: `nerv_tilde_oss_leverage_policy.v1`
  parallax_official_tilde_surface: `False`
  parallax_direct_runtime_import_allowed: `False`
  wall_attention_direct_kernel_import_allowed: `False`
- pr95_baseline_identity_attached: `False`
  baseline_id: `None`
  selected_archive_sha256: `None`
  selected_archive_bytes: `None`
  local_cpu_mlx_ready: `None`
  local_cpu_axis: `None`
  mlx_axis: `None`
  modal_dispatch_allowed: `None`
  paired_exact_eval_ready: `None`
  exact_axis_blockers: `pr95_baseline_identity_missing`

## Rows

- `hi_nerv::hinerv_np600_ld16_ed32_dc16_mi1fi4_hfg_cnx_lg2c4_cx2k7_int8_mixed_ceil216000_tgtmp0p229::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `17`
- `hi_nerv::hinerv_np600_ld16_ed32_dc16_mi1fi4_hfg_cnx_lg2c4_cx2k7_portfolio_auto_ceil216000_tgtmp0p229::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `17`
- `hi_nerv::hinerv_np600_ld4_ed24_dc24_mi1fi4_hfg_cnx_lg2c4_cx2k7_int7_mixed_ceil216000_tgtmp0p229::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `17`
- `snerv::snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t0_adbase_int8_symmetric_ceil216000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `28`
  snerv_runtime_authority: `receiver_bound_training_waits_on_required_primitive_rows`
  snerv_receiver_training_evidence: `False`
  snerv_full_source_forward_authority: `False`
- `snerv::snerv_np600_haar_lv5_lfb2_stepb1_fc9e0_p1_mfu1-2-4_hfr0_t0_adbase_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `28`
  snerv_runtime_authority: `receiver_bound_training_waits_on_required_primitive_rows`
  snerv_receiver_training_evidence: `False`
  snerv_full_source_forward_authority: `False`
- `snerv::snerv_np600_haar_lv5_lfb2p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t0_adbase_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `28`
  snerv_runtime_authority: `receiver_bound_training_waits_on_required_primitive_rows`
  snerv_receiver_training_evidence: `False`
  snerv_full_source_forward_authority: `False`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
