# SNeRV trained ladder decoder waterfill

Schema: `snerv_trained_ladder_waterfill.v1`
Authority: `false_authority_snerv_trained_ladder_decoder_waterfill_no_score_claim`

| row | groups | byte delta | source codec | blocker count |
|---|---:|---:|---|---:|
| snerv_trained_ladder_row_archive | 3 | 0 | snerv_decoder_payload.v3 | 10 |

## Blockers

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
