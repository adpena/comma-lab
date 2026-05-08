# HNeRV Frontier Entropy Gap Ranking

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`

## Current Frontier

- label: `PR106x-lowlevel-brotli`
- score: `0.20935073680571203`
- archive_bytes: `186080`
- archive_sha256: `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- eval_artifact: `experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json`

## Next Rate-Only Action

- action_id: `build_byte_different_recode_for_largest_current_frontier_section`
- target_label: `PR106x-lowlevel-brotli`
- target_section: `decoder_packed_brotli`
- required_next_artifact: `old_new_section_sha256_and_charged_byte_diff_manifest`
- dispatch_allowed: `false`
- rationale: largest current exact-frontier section is the deterministic rate target

## Combined Entropy Gap Groups

| target | section | HDC2 bytes | current section bytes | known target bytes | net now | net after known targets | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts | `None` | 221381 | None | 129638 | None | None | `unmatched_to_current_frontier_section` |

## Entropy Target Ranking

| rank | target | kind | target bytes | section | net now | net after target | required artifact |
|---:|---|---|---:|---|---:|---:|---|
| 1 | public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts | `known_model_overhead` | 40840 | `None` | None | None | `byte_accounted_model_overhead_reduction_manifest` |
| 2 | public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts | `known_model_overhead` | 40840 | `None` | None | None | `byte_accounted_model_overhead_reduction_manifest` |
| 3 | public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts | `known_payload_entropy_gap` | 23979 | `None` | None | None | `roundtrip_payload_recode_manifest` |
| 4 | public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts | `known_payload_entropy_gap` | 23979 | `None` | None | None | `roundtrip_payload_recode_manifest` |

## Current Frontier Byte Mass

| rank | section | role | bytes | entropy b/B | rate mass |
|---:|---|---|---:|---:|---:|
| 1 | `decoder_packed_brotli` | `decoder_weight_stream` | 170127 | 7.998224 | 0.113280586118 |
| 2 | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | 7.985386 | 0.010553198548 |
| 3 | `packed_header_ff_len24` | `control_or_metadata` | 4 | 2.0 | 2.663436e-06 |

Interpretation: this manifest ranks rate-only work. It is not a new
score claim, not a candidate archive manifest, and not a dispatch
authorization.
