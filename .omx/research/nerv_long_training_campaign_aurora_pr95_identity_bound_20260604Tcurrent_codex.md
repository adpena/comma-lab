# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `3`
Launchable local rows: `1`
Blocked rows: `3`
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
- pr95_baseline_identity_attached: `True`
  baseline_id: `pr95_public_hnerv_muon_control_arm`
  selected_archive_sha256: `61fae2691fc674e11307307e1a87e8f0aef75ebcad4f34fd780980ed68e87f74`
  selected_archive_bytes: `178363`
  local_cpu_mlx_ready: `True`
  local_cpu_axis: `[macOS-CPU advisory]`
  mlx_axis: `[macOS-MLX research-signal]`
  modal_dispatch_allowed: `False`
  paired_exact_eval_ready: `False`
  exact_axis_blockers: `pr95_contest_cpu_exact_eval_missing, pr95_contest_cuda_exact_eval_missing`

## Rows

- `hi_nerv::hinerv_np600_ld4_ed8_dc4_mi1fi4_hfg_cnx_lg2c4_cx2k7_int8_mixed_ceil178000_tgtmp0p036::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld4_ed8_dc4_mi1fi4_hfg_cnx_lg2c4_cx2k7_int8_mixed_ceil178000_tgtmp0p036::aurora_like`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `aurora_like`
  blockers: `11`
- `snerv::snerv_np600_haar_lv5_lfb2p5_stepb0p5_fc36e0_p1_mfu1-2-4_hfr0_t1_adbase_oms0p285_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `pact_muon_adamw`
  blockers: `14`
  snerv_runtime_authority: `receiver_bound_training_waits_on_required_primitive_rows`
  snerv_receiver_training_evidence: `False`
  snerv_full_source_forward_authority: `False`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
