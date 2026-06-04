# NeRV Long-Training Campaign Plan

Schema: `nerv_long_training_campaign_plan.v1`
Rows: `2`
Launchable local rows: `0`
Blocked rows: `2`
Score claim: `False`
Ready for exact dispatch: `False`

## Rows

- `hi_nerv::auto_bytecap::pact_muon_adamw`
  family: `hi_nerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `11`
- `snerv::auto_bytecap::native_rate_aware_training`
  family: `snerv`
  launchable_mlx: `False`
  optimizer: `pact_muon_adamw`
  blockers: `16`

## Blockers

- `campaign_plan_is_not_execution`
- `exact_cpu_cuda_not_launched_by_campaign_plan`
- `snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes`
