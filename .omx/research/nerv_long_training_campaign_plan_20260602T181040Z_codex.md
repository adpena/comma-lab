# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `15`
Launchable local rows: `12`
Blocked rows: `15`
Score claim: `False`
Ready for exact dispatch: `False`

## Rows

- `hi_nerv::hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000::lion`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `lion`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000::lion`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `lion`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000::lion`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `lion`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000::adafactor`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adafactor`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000::rmsprop`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `rmsprop`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000::adafactor`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adafactor`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000::rmsprop`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `rmsprop`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000::adafactor`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adafactor`
  blockers: `3`
- `hi_nerv::hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000::rmsprop`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `rmsprop`
  blockers: `3`
- `snerv::snerv_np600_lv2_lfb1p5_stepb0p5_int2_symmetric_ceil36000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `None`
  blockers: `22`
- `snerv::snerv_np600_lv3_lfb1p5_stepb0p5_int2_symmetric_ceil36000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `None`
  blockers: `22`
- `snerv::snerv_np600_lv4_lfb1p5_stepb0p5_int2_symmetric_ceil36000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `None`
  blockers: `22`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
- `snerv_shared_mlx_scoreaware_long_training_harness_not_bound`
- `snerv_native_rate_pressure_in_loop_not_yet_training_authority`
- `snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes`
