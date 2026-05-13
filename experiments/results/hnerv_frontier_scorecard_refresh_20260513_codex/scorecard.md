# HNeRV Frontier Scorecard

| label | grade | canonical | scope | score | bytes | seg | pose | rate | largest section | archive sha |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| PR106-R2-lowlevel | A++ | no | `exact_local_cuda_custody` | 0.206517476020 | 186629 | 0.064260000 | 0.017988885 | 0.124268591 | `decoder_compact_brotli_streams:162164` | `287e6edc612803a9` |
| PR103-ac-repack | A++ | yes | `exact_local_cuda_custody` | 0.208981052780 | 185578 | 0.067082000 | 0.018330303 | 0.123568750 | `merged_range_coded_weights_and_hi_latents:153856` | `ec0890c2d2317dca` |
| PR106x-lowlevel | A++ | yes | `exact_local_cuda_custody` | 0.209350736806 | 186080 | 0.067142000 | 0.018305737 | 0.123903000 | `decoder_packed_brotli:170127` | `b0a12549a39e34a0` |
| PR106x | A++ | yes | `exact_local_cuda_custody` | 0.209451236806 | 186231 | 0.067142000 | 0.018305737 | 0.124003500 | `decoder_packed_brotli:170278` | `d25bca80057e8b53` |
| PR106 | A++ | yes | `exact_local_cuda_custody` | 0.209456736806 | 186239 | 0.067142000 | 0.018305737 | 0.124009000 | `decoder_packed_brotli:170278` | `3fefbe5dfdd73817` |
| PR102 | A++ | yes | `exact_local_cuda_custody` | 0.228393729891 | 178981 | 0.067568000 | 0.041649730 | 0.119176000 | `merged_range_coded_weights_and_hi_latents:153856` | `afd53348f50303bf` |
| PR105x | A++ | yes | `exact_local_cuda_custody` | 0.230431829870 | 177849 | 0.070456000 | 0.041553580 | 0.118422250 | `decoder_packed_brotli:161891` | `692a46931f66416a` |
| PR105 | A++ | yes | `exact_local_cuda_custody` | 0.230437329870 | 177857 | 0.070456000 | 0.041553580 | 0.118427750 | `decoder_packed_brotli:161891` | `597ba0732810eba0` |
| PR104 | A++ | yes | `exact_local_cuda_custody` | 0.231134466204 | 178637 | 0.070670000 | 0.041517466 | 0.118947000 | `merged_range_coded_weights_and_hi_latents:153856` | `6564c32a9edeeaf0` |

## Byte-Identical Payload Groups

| labels | archive byte span | same seg | same pose | payload sha |
|---|---:|---:|---:|---|
| PR105, PR105x | 8 | true | true | `5f88fa8ed26816cf` |
| PR106, PR106x | 8 | true | true | `7f2cc905b7611ae8` |

## Payload Follow-Up Targets

| label | section | bytes | entropy b/B | required next gate | suggested action |
|---|---|---:|---:|---|---|
| PR106 | `decoder_packed_brotli` | 170278 | 7.998152 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR106x | `decoder_packed_brotli` | 170278 | 7.998152 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR106x-lowlevel | `decoder_packed_brotli` | 170127 | 7.998224 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR106-R2-lowlevel | `decoder_compact_brotli_streams` | 162164 | 7.998197 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR105 | `decoder_packed_brotli` | 161891 | 7.998095 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR105x | `decoder_packed_brotli` | 161891 | 7.998095 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR102 | `merged_range_coded_weights_and_hi_latents` | 153856 | 7.996868 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR103-ac-repack | `merged_range_coded_weights_and_hi_latents` | 153856 | 7.998668 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR104 | `merged_range_coded_weights_and_hi_latents` | 153856 | 7.997740 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR105 | `latents_and_sidecar_brotli` | 15854 | 7.985312 | build byte-different archive, then exact CUDA replay | latent/sidecar arithmetic-coding parity fixture |
| PR105x | `latents_and_sidecar_brotli` | 15854 | 7.985312 | build byte-different archive, then exact CUDA replay | latent/sidecar arithmetic-coding parity fixture |
| PR106 | `latents_and_sidecar_brotli` | 15849 | 7.985386 | build byte-different archive, then exact CUDA replay | latent/sidecar arithmetic-coding parity fixture |

## Next Exact-Evaluable Target

| frontier | target label | section | role | bytes | score gap | required next gate |
|---|---|---|---|---:|---:|---|
| PR103-ac-repack | PR103-ac-repack | `merged_range_coded_weights_and_hi_latents` | `decoder_weight_stream` | 153856 | 0.000000000000 | build byte-different archive with old/new section SHA-256 and charged-byte proof, then exact CUDA auth eval after lane claim |

This target is ranked by exact-frontier proximity first, then charged
payload byte mass. It is a routing target only; it is not a new score
claim.

## Hidden-Gem Byte-Mass Ranking

| rank | label | section | role | bytes | score gap | priority |
|---:|---|---|---|---:|---:|---|
| 1 | PR103-ac-repack | `merged_range_coded_weights_and_hi_latents` | `decoder_weight_stream` | 153856 | 0.000000000000 | `current_frontier_primary` |
| 2 | PR103-ac-repack | `latent_low_bytes_brotli` | `latent_stream` | 15537 | 0.000000000000 | `current_frontier_primary` |
| 3 | PR103-ac-repack | `sidecar_corrections_brotli` | `sidecar_or_correction_stream` | 7902 | 0.000000000000 | `current_frontier_primary` |
| 4 | PR103-ac-repack | `non_ac_weights_brotli` | `decoder_weight_stream` | 7097 | 0.000000000000 | `current_frontier_primary` |
| 5 | PR103-ac-repack | `ac_histograms_brotli` | `entropy_model_or_range_stream` | 895 | 0.000000000000 | `current_frontier_primary` |
| 6 | PR103-ac-repack | `latent_hi_histogram_brotli` | `latent_stream` | 15 | 0.000000000000 | `current_frontier_primary` |
| 7 | PR103-ac-repack | `latent_min_scale_fp16` | `control_or_metadata` | 112 | 0.000000000000 | `low` |
| 8 | PR103-ac-repack | `scales_fp16` | `control_or_metadata` | 56 | 0.000000000000 | `low` |
| 9 | PR106x-lowlevel | `decoder_packed_brotli` | `decoder_weight_stream` | 170127 | 0.000369684026 | `near_frontier_secondary` |
| 10 | PR106x-lowlevel | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | 0.000369684026 | `near_frontier_secondary` |
| 11 | PR106x-lowlevel | `packed_header_ff_len24` | `control_or_metadata` | 4 | 0.000369684026 | `low` |
| 12 | PR106x | `decoder_packed_brotli` | `decoder_weight_stream` | 170278 | 0.000470184026 | `near_frontier_secondary` |

## Internal Score-Lowering Frontier

| label | score | bytes | canonical eligible | blockers | archive sha |
|---|---:|---:|---:|---|---|
| PR106-R2-lowlevel | 0.206517476020 | 186629 | false | promotion_ineligible | `287e6edc612803a9` |

This is an internal exact-CUDA score-lowering route, not promotion
authority. It can point byte-closed optimizer work at a lower exact
score even when public/canonical adjudication blockers remain.

## Next Internal Score-Lowering Target

| frontier | target label | section | role | bytes | score gap | required next gate |
|---|---|---|---|---:|---:|---|
| PR106-R2-lowlevel | PR106-R2-lowlevel | `decoder_compact_brotli_streams` | `decoder_weight_stream` | 162164 | 0.000000000000 | build byte-different archive with old/new section SHA-256 and charged-byte proof, then exact CUDA auth eval after lane claim |

## Internal Score-Lowering Byte-Mass Ranking

| rank | label | section | role | bytes | score gap | priority |
|---:|---|---|---|---:|---:|---|
| 1 | PR106-R2-lowlevel | `decoder_compact_brotli_streams` | `decoder_weight_stream` | 162164 | 0.000000000000 | `internal_score_lowering_frontier_primary` |
| 2 | PR106-R2-lowlevel | `latents_raw_lzma_delta_u8` | `latent_stream` | 15387 | 0.000000000000 | `internal_score_lowering_frontier_primary` |
| 3 | PR106-R2-lowlevel | `sidecar_dim_delta_huffman_enum` | `sidecar_or_correction_stream` | 8970 | 0.000000000000 | `internal_score_lowering_frontier_primary` |
| 4 | PR103-ac-repack | `merged_range_coded_weights_and_hi_latents` | `decoder_weight_stream` | 153856 | 0.002463576760 | `near_frontier_secondary` |
| 5 | PR103-ac-repack | `latent_low_bytes_brotli` | `latent_stream` | 15537 | 0.002463576760 | `near_frontier_secondary` |
| 6 | PR103-ac-repack | `sidecar_corrections_brotli` | `sidecar_or_correction_stream` | 7902 | 0.002463576760 | `near_frontier_secondary` |
| 7 | PR103-ac-repack | `non_ac_weights_brotli` | `decoder_weight_stream` | 7097 | 0.002463576760 | `near_frontier_secondary` |
| 8 | PR103-ac-repack | `ac_histograms_brotli` | `entropy_model_or_range_stream` | 895 | 0.002463576760 | `near_frontier_secondary` |
| 9 | PR103-ac-repack | `latent_min_scale_fp16` | `control_or_metadata` | 112 | 0.002463576760 | `low` |
| 10 | PR103-ac-repack | `scales_fp16` | `control_or_metadata` | 56 | 0.002463576760 | `low` |
| 11 | PR103-ac-repack | `latent_hi_histogram_brotli` | `latent_stream` | 15 | 0.002463576760 | `near_frontier_secondary` |
| 12 | PR106x-lowlevel | `decoder_packed_brotli` | `decoder_weight_stream` | 170127 | 0.002833260786 | `near_frontier_secondary` |

## Payload Section Manifests

| label | section | role | bytes | sha256 | required proof |
|---|---|---|---:|---|---|
| PR102 | `scales_fp16` | `control_or_metadata` | 56 | `58df9e14c55704cd` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR102 | `non_ac_weights_brotli` | `decoder_weight_stream` | 7097 | `143d2aa8c76fb2ec` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR102 | `ac_histograms_brotli` | `entropy_model_or_range_stream` | 895 | `00b7baafb5ff987b` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR102 | `merged_range_coded_weights_and_hi_latents` | `decoder_weight_stream` | 153856 | `3fef8b33482a9a28` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR102 | `latent_min_scale_fp16` | `control_or_metadata` | 112 | `bcef60a788b3027d` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR102 | `latent_low_bytes_brotli` | `latent_stream` | 15537 | `da9718e2ee05ab20` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR103-ac-repack | `scales_fp16` | `control_or_metadata` | 56 | `0aacfa8d29a41b3d` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR103-ac-repack | `non_ac_weights_brotli` | `decoder_weight_stream` | 7097 | `4e56ba8016027cb0` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR103-ac-repack | `ac_histograms_brotli` | `entropy_model_or_range_stream` | 895 | `78ee3d1c22ad5070` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR103-ac-repack | `merged_range_coded_weights_and_hi_latents` | `decoder_weight_stream` | 153856 | `11e9614692d36a07` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR103-ac-repack | `latent_min_scale_fp16` | `control_or_metadata` | 112 | `5ad90dded558282a` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR103-ac-repack | `latent_low_bytes_brotli` | `latent_stream` | 15537 | `f7d0226b39530e71` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR104 | `scales_fp16` | `control_or_metadata` | 56 | `71ac3e619b74d4c1` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR104 | `non_ac_weights_brotli` | `decoder_weight_stream` | 7097 | `d92a351476e89962` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR104 | `ac_histograms_brotli` | `entropy_model_or_range_stream` | 895 | `89b15a26e383a80b` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR104 | `merged_range_coded_weights_and_hi_latents` | `decoder_weight_stream` | 153856 | `baea0cc3626228dc` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR104 | `latent_min_scale_fp16` | `control_or_metadata` | 112 | `83b3a98546668117` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR104 | `latent_low_bytes_brotli` | `latent_stream` | 15537 | `38d57ee74cac0afc` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105 | `packed_header_ff_len24` | `control_or_metadata` | 4 | `837b877d82478a79` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105 | `decoder_packed_brotli` | `decoder_weight_stream` | 161891 | `e241db1c2cbfb53a` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105 | `latents_and_sidecar_brotli` | `latent_stream` | 15854 | `aeea25a3116b4457` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105x | `packed_header_ff_len24` | `control_or_metadata` | 4 | `837b877d82478a79` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105x | `decoder_packed_brotli` | `decoder_weight_stream` | 161891 | `e241db1c2cbfb53a` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105x | `latents_and_sidecar_brotli` | `latent_stream` | 15854 | `aeea25a3116b4457` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106 | `packed_header_ff_len24` | `control_or_metadata` | 4 | `7939f08db7d18dd4` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106 | `decoder_packed_brotli` | `decoder_weight_stream` | 170278 | `654999f81f0552fb` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106 | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | `94257b33cf3083c5` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106-R2-lowlevel | `decoder_compact_brotli_streams` | `decoder_weight_stream` | 162164 | `b91cdfb888580def` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106-R2-lowlevel | `latents_raw_lzma_delta_u8` | `latent_stream` | 15387 | `76e632e1749f4695` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106-R2-lowlevel | `sidecar_dim_delta_huffman_enum` | `sidecar_or_correction_stream` | 8970 | `e7ec8a186e284b4f` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x | `packed_header_ff_len24` | `control_or_metadata` | 4 | `7939f08db7d18dd4` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x | `decoder_packed_brotli` | `decoder_weight_stream` | 170278 | `654999f81f0552fb` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | `94257b33cf3083c5` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x-lowlevel | `packed_header_ff_len24` | `control_or_metadata` | 4 | `c670d1bee2140039` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x-lowlevel | `decoder_packed_brotli` | `decoder_weight_stream` | 170127 | `07725c39ff436195` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x-lowlevel | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | `94257b33cf3083c5` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |

Interpretation: score truth remains the exact CUDA replay JSON. Payload
sections are forensic signals for the next compression action; they do
not imply score deltas without a new exact archive eval.

Guardrail: a lossless Brotli repack row is a local exact-custody
byte-control. It does not supersede categorical/range-coded HNeRV
families unless that exact candidate archive has a lower CUDA score
under the same custody standard.
