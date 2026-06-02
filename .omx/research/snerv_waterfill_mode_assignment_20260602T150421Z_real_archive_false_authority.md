# SNeRV waterfill mode assignment

Schema: `snerv_waterfill_mode_assignment.v1`
Authority: `false_authority_snerv_waterfill_mode_assignment_no_score_claim`

| row | modes | local probe | receiver export | blocker count |
|---|---|---:|---:|---:|
| snerv_trained_ladder_row_archive | `fp16,fp16,fp16` | true | false | 13 |

## Blockers

- `mode_assignment_is_false_authority_until_receiver_replay_and_exact_eval`
- `contest_cpu_cuda_exact_eval_not_executed`
- `decoder_weight_saliency_replay_required_for_authority`
- `sample_pair_count_below_full600`
- `required_emission_field_missing:official_controls.--modelsize`
- `required_emission_field_missing:official_controls.mfu_enabled`
- `required_emission_field_missing:official_controls.hfr_enabled`
- `required_emission_field_missing:official_controls.snerv_t_enabled`
- `required_emission_field_missing:qat_bits`
- `decoder_weight_saliency_missing_for_some_groups`
- `full_video_coverage_missing`
- `receiver_proof_not_satisfied`
- `mixed_decoder_modes_do_not_support_fp32`
- `fp32_protect_downgraded_to_fp16_requires_receiver_replay`
