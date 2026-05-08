# PR106 Entropy Floor Probe

- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`
- source_mode: `archive`
- source_path: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- archive_bytes: `186239`
- payload_bytes: `186131`
- decoder_section_bytes: `170278`
- latents_section_bytes: `15849`

## Group Floors

| group | current bytes | best transform | best markov2 floor | delta vs current |
|---|---:|---|---:|---:|
| `decoder_q_zz_plus_f32_scales` | 170278 | `delta_mod` | 45660 | -124618 |
| `fixed_latents_delta_zz_plus_fp16_meta` | 15849 | `delta_mod` | 849 | -15000 |
| `decoded_payload_sections_without_ff_header` | 186127 | `delta_mod` | 46508 | -139619 |

## Adversarial Claim Check

- claim: `encoder_side_bounded_at_about_178kb_without_ml`
- verdict: `pr101_only_not_transferable_to_pr106_without_pr106_specific_codec_and_exact_eval`
- pr101_reference_archive_bytes: `178258`
- active_floor_archive_bytes: `185578`
- best_decoded_payload_markov2_floor_bytes_before_overhead: `46508`

The floor rows are oracle model-class bounds, not charged archive candidates. Transfer from PR101 to PR106 requires a PR106-specific bitstream/runtime that pays model tables, metadata, packet overhead, and exact CUDA replay on the resulting archive.
