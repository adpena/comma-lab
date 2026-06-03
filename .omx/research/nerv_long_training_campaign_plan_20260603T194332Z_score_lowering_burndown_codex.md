# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `8`
Launchable local rows: `4`
Blocked rows: `8`
Score claim: `False`
Ready for exact dispatch: `False`

## Rows

- `hi_nerv::hinerv_np600_ld4_ed8_dc4_mi1fi4_hfg_cnx_lg2c4_cx2k7_int6_mixed_ceil178000_tgtmp0p036::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `9`
- `hi_nerv::hinerv_np600_ld4_ed8_dc4_mi1fi4_hfg_cnx_lg2c4_cx2k7_int7_mixed_ceil178000_tgtmp0p036::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `9`
- `hi_nerv::hinerv_np600_ld4_ed8_dc4_mi1fi4_hfg_cnx_lg2c4_cx2k7_int8_mixed_ceil178000_tgtmp0p036::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `9`
- `hi_nerv::hinerv_np600_ld4_ed8_dc4_mi1fi4_hfg_cnx_lg2c4_cx2k7_portfolio_auto_ceil178000_tgtmp0p036::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `9`
- `snerv::snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc36e0_p1_mfu1-2-4_hfr0_t1_adbase_oms0p285_int8_symmetric_ceil216000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `pact_muon_adamw`
  blockers: `14`
- `snerv::snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc36e0_p1_mfu1-2-4_hfr0_t1_tmhaar1_adbase_oms0p285_int8_symmetric_ceil216000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `pact_muon_adamw`
  blockers: `14`
- `snerv::snerv_np600_haar_lv5_lfb2p5_stepb0p5_fc36e0_p1_mfu1-2-4_hfr0_t1_adbase_oms0p285_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `pact_muon_adamw`
  blockers: `14`
- `snerv::snerv_np600_haar_lv5_lfb2p5_stepb0p5_fc36e0_p1_mfu1-2-4_hfr0_t1_tmhaar1_adbase_oms0p285_int8_symmetric_ceil285000::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `True`
  optimizer: `pact_muon_adamw`
  blockers: `14`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
