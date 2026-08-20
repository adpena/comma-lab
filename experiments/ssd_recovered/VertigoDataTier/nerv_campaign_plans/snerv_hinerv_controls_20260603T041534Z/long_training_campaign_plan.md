# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `12`
Launchable local rows: `0`
Blocked rows: `12`
Score claim: `False`
Ready for exact dispatch: `False`

## Rows

- `hi_nerv::hinerv_np600_ld16_ed24_dc6_int2_mixed_ceil178000_tgtmp0p05::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld16_ed24_dc6_int4_mixed_ceil178000_tgtmp0p05::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld16_ed24_dc6_int8_mixed_ceil178000_tgtmp0p05::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld16_ed24_dc6_portfolio_auto_ceil178000_tgtmp0p05::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld16_ed24_dc6_int2_mixed_ceil178000_tgtmp0p05::adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld16_ed24_dc6_int4_mixed_ceil178000_tgtmp0p05::adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld16_ed24_dc6_int8_mixed_ceil178000_tgtmp0p05::adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `adamw`
  blockers: `10`
- `hi_nerv::hinerv_np600_ld16_ed24_dc6_portfolio_auto_ceil178000_tgtmp0p05::adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `adamw`
  blockers: `10`
- `snerv::snerv_np600_haar_lv3_lfb1p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t2_adbase_int2_symmetric_ceil178000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `None`
  blockers: `26`
- `snerv::snerv_np600_haar_lv4_lfb1p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t2_adbase_int2_symmetric_ceil178000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `None`
  blockers: `26`
- `snerv::snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t2_adbase_int2_symmetric_ceil178000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `25`
- `snerv::snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t2_tmhaar1_adbase_int2_symmetric_ceil178000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `25`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
- `snerv_native_rate_pressure_in_loop_not_yet_training_authority`
- `snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes`
