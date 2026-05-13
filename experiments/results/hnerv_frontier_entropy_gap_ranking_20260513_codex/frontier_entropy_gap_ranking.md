# HNeRV Frontier Entropy Gap Ranking

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- frontier_mode: `score_lowering`

## Selected Frontier

- label: `PR106-R2-lowlevel-HDM4`
- score: `0.20642625334307507`
- archive_bytes: `186492`
- archive_sha256: `218ae16f3f13b722e9752d698667ed8770151e40d44b5756c0ebbccb7682825f`
- eval_artifact: `experiments/results/modal_auth_eval/pr106_r2_lowlevel_hdm4_candidate_pr101_runtime_cuda_20260513_codex/contest_auth_eval.json`

## Next Rate-Only Action

- action_id: `review_current_exact_lossless_brotli_control_before_promotion`
- target_label: `PR106-R2-lowlevel-HDM4`
- target_section: `decoder_packed_brotli`
- required_next_artifact: `operator_promotion_review_note_for_existing_exact_archive_sha`
- dispatch_allowed: `false`
- rationale: existing exact CUDA custody already covers the byte-different lossless repack; review/promotion is the next rate-only step, not another dispatch

## Exact Lossless Controls

| label | source bytes | candidate bytes | byte delta | review status |
|---|---:|---:|---:|---|
| PR106-R2-lowlevel-HDM4 | 186629 | 186492 | -137 | `ready_for_promotion_review_existing_exact_custody` |

## Current Frontier Byte Mass

| rank | section | role | bytes | entropy b/B | rate mass |
|---:|---|---|---:|---:|---:|
| 1 | `inner_decoder_packed_brotli` | `decoder_weight_stream` | 169990 | 7.998504 | 0.113189363441 |
| 2 | `inner_latents_and_sidecar_brotli` | `latent_stream` | 15849 | 7.985386 | 0.010553198548 |
| 3 | `sidecar_payload_pr101_ranked_no_op` | `sidecar_or_correction_stream` | 527 | 7.429261 | 0.000350907668 |
| 4 | `sidecar_len_u16` | `sidecar_or_correction_stream` | 2 | 1.0 | 1.331718e-06 |
| 5 | `pr106_sidecar_header_fe_fmt_len_u32` | `control_or_metadata` | 6 | 2.251629 | 3.995154e-06 |
| 6 | `sidecar_framing_meta_pr101` | `control_or_metadata` | 6 | 1.459148 | 3.995154e-06 |
| 7 | `inner_packed_header_ff_len24` | `control_or_metadata` | 4 | 2.0 | 2.663436e-06 |

Interpretation: this manifest ranks rate-only work. It is not a new
score claim, not a candidate archive manifest, and not a dispatch
authorization.
