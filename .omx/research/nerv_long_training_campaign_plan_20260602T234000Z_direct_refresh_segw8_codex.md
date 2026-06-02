# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `6`
Launchable local rows: `6`
Blocked rows: `6`
Score claim: `False`
Ready for exact dispatch: `False`

## Rows

- `hi_nerv::hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld4_ed16_dc8_hfg_cnx_int2_mixed_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld4_ed24_dc8_hfg_cnx_int2_mixed_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `10`
- `snerv::snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc9e0_int8_symmetric_ceil216000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `19`
- `snerv::snerv_np600_haar_lv5_lfb2_stepb1_fc9e0_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `19`
- `snerv::snerv_np600_haar_lv5_lfb2p5_stepb0p5_fc9e0_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `19`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
- `snerv_native_rate_pressure_in_loop_not_yet_training_authority`
