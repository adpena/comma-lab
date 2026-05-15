# HNeRV Frontier Entropy Gap Ranking

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- frontier_mode: `score_lowering`

## Selected Frontier

- label: `PR106-R2-HDM12-HLM3-MAGICLESS-FMT0C`
- score: `0.2063163866158099`
- archive_bytes: `186327`
- archive_sha256: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- eval_artifact: `experiments/results/modal_auth_eval/pr106_format0c_exact_radix_paired_20260515T0918Z_cuda/contest_auth_eval.json`

## Next Rate-Only Action

- action_id: `build_byte_different_recode_for_largest_current_frontier_section`
- target_label: `PR106-R2-HDM12-HLM3-MAGICLESS-FMT0C`
- target_section: `decoder_compact_brotli_streams`
- required_next_artifact: `old_new_section_sha256_and_charged_byte_diff_manifest`
- dispatch_allowed: `false`
- rationale: largest current exact-frontier section is the deterministic rate target

## Exact Lossless Controls

| label | source bytes | candidate bytes | byte delta | review status |
|---|---:|---:|---:|---|
| PR106-R2-HDM12-HLM3-MAGICLESS-FMT0C | None | 186327 | None | `blocked_pending_candidate_diff_review` |

## Current Frontier Byte Mass

| rank | section | role | bytes | entropy b/B | rate mass |
|---:|---|---|---:|---:|---:|
| 1 | `decoder_compact_brotli_streams` | `decoder_weight_stream` | 162164 | 7.998487 | 0.107978351274 |
| 2 | `latents_raw_lzma_delta_u8` | `latent_stream` | 15387 | 7.985609 | 0.010245571712 |
| 3 | `sidecar_dim_delta_huffman_enum` | `sidecar_or_correction_stream` | 8676 | 7.977717 | 0.005776992277 |

Interpretation: this manifest ranks rate-only work. It is not a new
score claim, not a candidate archive manifest, and not a dispatch
authorization.
