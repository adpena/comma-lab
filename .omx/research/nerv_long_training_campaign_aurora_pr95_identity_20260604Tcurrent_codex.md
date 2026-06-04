# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `3`
Launchable local rows: `1`
Blocked rows: `3`
Score claim: `False`
Ready for exact dispatch: `False`

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
