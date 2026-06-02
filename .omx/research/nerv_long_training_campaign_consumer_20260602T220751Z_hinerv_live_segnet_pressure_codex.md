# NeRV Long-Training Campaign Consumer Verdict

Schema: `nerv_long_training_campaign_consumer_result.v1`
Source schema: `nerv_long_training_campaign_plan.v1`
Planner action: `route_launchable_local_mlx_campaign_rows_without_exact_dispatch`
Local MLX ready rows: `10`
Exact auth recommended: `False`
Score claim: `False`

## Selected Local MLX Rows

- `hi_nerv_hinerv_np600_ld4_ed12_dc12_int4_mixed_ceil36000_adamw` family=`hi_nerv` priority=`10`
- `hi_nerv_hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000_adamw` family=`hi_nerv` priority=`10`
- `hi_nerv_hinerv_np600_ld4_ed12_dc8_portfolio_auto_ceil36000_adamw` family=`hi_nerv` priority=`10`
- `hi_nerv_hinerv_np600_ld4_ed24_dc16_int2_mixed_ceil36000_adamw` family=`hi_nerv` priority=`10`
- `hi_nerv_hinerv_np600_ld4_ed32_dc16_int2_mixed_ceil36000_adamw` family=`hi_nerv` priority=`10`
- `snerv_snerv_np600_lv2_lfb1p5_stepb0p5_int2_symmetric_ceil36000_native_rate_aware_training` family=`snerv` priority=`12`
- `snerv_snerv_np600_lv3_lfb1p5_stepb0p5_int2_symmetric_ceil36000_native_rate_aware_training` family=`snerv` priority=`12`
- `snerv_snerv_np600_lv4_lfb1p5_stepb0p5_int2_symmetric_ceil36000_native_rate_aware_training` family=`snerv` priority=`12`
- `snerv_snerv_np600_lv5_lfb1p5_stepb0p5_int2_symmetric_ceil36000_native_rate_aware_training` family=`snerv` priority=`12`
- `snerv_snerv_np600_lv5_lfb2_stepb0p5_int2_symmetric_ceil36000_native_rate_aware_training` family=`snerv` priority=`12`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
- `snerv_native_rate_pressure_in_loop_not_yet_training_authority`
- `snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes`
