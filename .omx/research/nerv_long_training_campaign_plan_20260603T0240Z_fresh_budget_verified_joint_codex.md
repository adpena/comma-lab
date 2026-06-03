# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `9`
Launchable local rows: `9`
Blocked rows: `9`
Score claim: `False`
Ready for exact dispatch: `False`

## Rows

- `hi_nerv::hinerv_np600_ld4_ed24_dc24_hfg_int8_mixed_ceil216000::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `pact_muon_adamw`
  blockers: `9`
- `hi_nerv::hinerv_np600_ld4_ed24_dc24_hfg_portfolio_auto_ceil216000::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `pact_muon_adamw`
  blockers: `9`
- `hi_nerv::hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil36000_tgtmp0p02::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `pact_muon_adamw`
  blockers: `9`
- `hi_nerv::hinerv_np600_ld4_ed24_dc24_hfg_int8_mixed_ceil216000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `9`
- `hi_nerv::hinerv_np600_ld4_ed24_dc24_hfg_portfolio_auto_ceil216000::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `9`
- `hi_nerv::hinerv_np600_ld4_ed32_dc4_cnx_int8_mixed_ceil36000_tgtmp0p02::adamw`
  family: `hi_nerv`
  launchable_mlx: `True`
  optimizer: `adamw`
  blockers: `9`
- `snerv::snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc19e0_p1_mfu1-2-4_hfr0_t0_adbase_oms0p1_int8_symmetric_ceil216000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `20`
- `snerv::snerv_np600_haar_lv5_lfb2_stepb1_fc19e0_p1_mfu1-2-4_hfr0_t0_adbase_oms0p1_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `20`
- `snerv::snerv_np600_haar_lv5_lfb2p5_stepb0p5_fc19e0_p1_mfu1-2-4_hfr0_t0_adbase_oms0p1_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `None`
  blockers: `20`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
- `snerv_native_rate_pressure_in_loop_not_yet_training_authority`
