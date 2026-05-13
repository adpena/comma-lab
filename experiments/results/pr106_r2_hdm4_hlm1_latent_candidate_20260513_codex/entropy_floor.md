# PR106 Entropy Floor Probe

- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`
- source_mode: `archive`
- source_path: `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip`
- archive_bytes: `186423`
- payload_bytes: `185774`
- decoder_section_bytes: `169990`
- latents_section_bytes: `15780`

## Group Floors

| group | current bytes | best transform | best markov2 floor | delta vs current |
|---|---:|---|---:|---:|
| `decoder_q_zz_plus_f32_scales` | 169990 | `delta_mod` | 45660 | -124330 |
| `fixed_latents_delta_zz_plus_fp16_meta` | 15780 | `delta_mod` | 849 | -14931 |
| `decoded_payload_sections_without_ff_header` | 185770 | `delta_mod` | 46508 | -139262 |

## Adversarial Claim Check

- claim: `encoder_side_bounded_at_about_178kb_without_ml`
- verdict: `insufficient_cross_archive_reference`
- pr101_reference_archive_bytes: `None`
- active_floor_archive_bytes: `186423`
- best_decoded_payload_markov2_floor_bytes_before_overhead: `46508`

The floor rows are oracle model-class bounds, not charged archive candidates. Transfer from PR101 to PR106 requires a PR106-specific bitstream/runtime that pays model tables, metadata, packet overhead, and exact CUDA replay on the resulting archive.
