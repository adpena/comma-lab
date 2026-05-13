# HNeRV Frontier Entropy Gap Ranking

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- frontier_mode: `score_lowering`

## Selected Frontier

- label: `PR106-R2-lowlevel-HDM3`
- score: `0.2065081539943091`
- archive_bytes: `186615`
- archive_sha256: `8cc7e3b21a5f77604331abb727c105e21351e8c199456db741eecb1fc7714093`
- eval_artifact: `experiments/results/modal_auth_eval/pr106_r2_lowlevel_hdm3_candidate_pr101_runtime_cuda_20260513_codex/contest_auth_eval.json`

## Next Rate-Only Action

- action_id: `build_byte_different_recode_for_largest_current_frontier_section`
- target_label: `PR106-R2-lowlevel-HDM3`
- target_section: `inner_decoder_packed_brotli`
- required_next_artifact: `old_new_section_sha256_and_charged_byte_diff_manifest`
- dispatch_allowed: `false`
- rationale: largest current exact-frontier section is the deterministic rate target

## Current Frontier Byte Mass

| rank | section | role | bytes | entropy b/B | rate mass |
|---:|---|---|---:|---:|---:|
| 1 | `inner_decoder_packed_brotli` | `decoder_weight_stream` | 170113 | 7.99804 | 0.113271264092 |
| 2 | `inner_latents_and_sidecar_brotli` | `latent_stream` | 15849 | 7.985386 | 0.010553198548 |
| 3 | `sidecar_payload_pr101_ranked_no_op` | `sidecar_or_correction_stream` | 527 | 7.429261 | 0.000350907668 |
| 4 | `sidecar_len_u16` | `sidecar_or_correction_stream` | 2 | 1.0 | 1.331718e-06 |
| 5 | `pr106_sidecar_header_fe_fmt_len_u32` | `control_or_metadata` | 6 | 2.251629 | 3.995154e-06 |
| 6 | `sidecar_framing_meta_pr101` | `control_or_metadata` | 6 | 1.459148 | 3.995154e-06 |
| 7 | `inner_packed_header_ff_len24` | `control_or_metadata` | 4 | 2.0 | 2.663436e-06 |

Interpretation: this manifest ranks rate-only work. It is not a new
score claim, not a candidate archive manifest, and not a dispatch
authorization.
