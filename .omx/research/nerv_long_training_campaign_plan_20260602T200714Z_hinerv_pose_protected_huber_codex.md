# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `8`
Launchable local rows: `8`
Blocked rows: `8`
Score claim: `False`
Ready for exact dispatch: `False`

## Rows

- `hi_nerv::hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld4_ed32_dc16_int2_mixed_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `10`
- `snerv::snerv_np600_lv2_lfb1p5_stepb0p5_int2_symmetric_ceil36000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `20`
- `snerv::snerv_np600_lv3_lfb1p5_stepb0p5_int2_symmetric_ceil36000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `20`
- `snerv::snerv_np600_lv4_lfb1p5_stepb0p5_int2_symmetric_ceil36000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `20`
- `snerv::snerv_np600_lv5_lfb1p5_stepb0p5_int2_symmetric_ceil36000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `20`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
- `snerv_native_rate_pressure_in_loop_not_yet_training_authority`
- `snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes`
