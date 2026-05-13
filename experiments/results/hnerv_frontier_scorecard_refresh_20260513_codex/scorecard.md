# HNeRV Frontier Scorecard

| label | grade | canonical | scope | score | bytes | seg | pose | rate | largest section | archive sha |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| PR106-R2-lowlevel-HDM3 | A++ | no | `exact_local_cuda_custody` | 0.206508153994 | 186615 | 0.064260000 | 0.017988885 | 0.124259269 | `inner_decoder_packed_brotli:170113` | `8cc7e3b21a5f7760` |
| PR106-R2-lowlevel | A++ | no | `exact_local_cuda_custody` | 0.206517476020 | 186629 | 0.064260000 | 0.017988885 | 0.124268591 | `inner_decoder_packed_brotli:170127` | `287e6edc612803a9` |
| PR103-ac-repack | A++ | yes | `exact_local_cuda_custody` | 0.208981052780 | 185578 | 0.067082000 | 0.018330303 | 0.123568750 | `merged_range_coded_weights_and_hi_latents:153856` | `ec0890c2d2317dca` |
| PR106x-lowlevel | A++ | yes | `exact_local_cuda_custody` | 0.209350736806 | 186080 | 0.067142000 | 0.018305737 | 0.123903000 | `decoder_packed_brotli:170127` | `b0a12549a39e34a0` |
| PR106x | A++ | yes | `exact_local_cuda_custody` | 0.209451236806 | 186231 | 0.067142000 | 0.018305737 | 0.124003500 | `decoder_packed_brotli:170278` | `d25bca80057e8b53` |
| PR106 | A++ | yes | `exact_local_cuda_custody` | 0.209456736806 | 186239 | 0.067142000 | 0.018305737 | 0.124009000 | `decoder_packed_brotli:170278` | `3fefbe5dfdd73817` |
| PR102 | A++ | yes | `exact_local_cuda_custody` | 0.228393729891 | 178981 | 0.067568000 | 0.041649730 | 0.119176000 | `merged_range_coded_weights_and_hi_latents:153856` | `afd53348f50303bf` |
| PR105x | A++ | yes | `exact_local_cuda_custody` | 0.230431829870 | 177849 | 0.070456000 | 0.041553580 | 0.118422250 | `decoder_packed_brotli:161891` | `692a46931f66416a` |
| PR105 | A++ | yes | `exact_local_cuda_custody` | 0.230437329870 | 177857 | 0.070456000 | 0.041553580 | 0.118427750 | `decoder_packed_brotli:161891` | `597ba0732810eba0` |
| PR104 | A++ | yes | `exact_local_cuda_custody` | 0.231134466204 | 178637 | 0.070670000 | 0.041517466 | 0.118947000 | `merged_range_coded_weights_and_hi_latents:153856` | `6564c32a9edeeaf0` |

## Internal Score-Lowering Frontier

| label | score | bytes | canonical eligible | blockers | archive sha |
|---|---:|---:|---:|---|---|
| PR106-R2-lowlevel-HDM3 | 0.206508153994 | 186615 | false | promotion_ineligible | `8cc7e3b21a5f7760` |

## Next Internal Score-Lowering Target

| frontier | target label | section | role | bytes | score gap | required next gate |
|---|---|---|---|---:|---:|---|
| PR106-R2-lowlevel-HDM3 | PR106-R2-lowlevel-HDM3 | `inner_decoder_packed_brotli` | `decoder_weight_stream` | 170113 | 0.000000000000 | build byte-different archive with old/new section SHA-256 and charged-byte proof, then exact CUDA auth eval after lane claim |

## Internal Score-Lowering Byte-Mass Ranking

| rank | label | section | role | bytes | score gap | priority |
|---:|---|---|---|---:|---:|---|
| 1 | PR106-R2-lowlevel-HDM3 | `inner_decoder_packed_brotli` | `decoder_weight_stream` | 170113 | 0.000000000000 | `internal_score_lowering_frontier_primary` |
| 2 | PR106-R2-lowlevel-HDM3 | `inner_latents_and_sidecar_brotli` | `latent_stream` | 15849 | 0.000000000000 | `internal_score_lowering_frontier_primary` |
| 3 | PR106-R2-lowlevel-HDM3 | `sidecar_payload_pr101_ranked_no_op` | `sidecar_or_correction_stream` | 527 | 0.000000000000 | `internal_score_lowering_frontier_primary` |
| 4 | PR106-R2-lowlevel-HDM3 | `pr106_sidecar_header_fe_fmt_len_u32` | `control_or_metadata` | 6 | 0.000000000000 | `low` |
| 5 | PR106-R2-lowlevel-HDM3 | `sidecar_framing_meta_pr101` | `control_or_metadata` | 6 | 0.000000000000 | `low` |
| 6 | PR106-R2-lowlevel-HDM3 | `inner_packed_header_ff_len24` | `control_or_metadata` | 4 | 0.000000000000 | `low` |
| 7 | PR106-R2-lowlevel-HDM3 | `sidecar_len_u16` | `sidecar_or_correction_stream` | 2 | 0.000000000000 | `internal_score_lowering_frontier_primary` |
| 8 | PR106-R2-lowlevel | `inner_decoder_packed_brotli` | `decoder_weight_stream` | 170127 | 0.000009322025 | `near_frontier_secondary` |
| 9 | PR106-R2-lowlevel | `inner_latents_and_sidecar_brotli` | `latent_stream` | 15849 | 0.000009322025 | `near_frontier_secondary` |
| 10 | PR106-R2-lowlevel | `sidecar_payload_pr101_ranked_no_op` | `sidecar_or_correction_stream` | 527 | 0.000009322025 | `near_frontier_secondary` |
| 11 | PR106-R2-lowlevel | `pr106_sidecar_header_fe_fmt_len_u32` | `control_or_metadata` | 6 | 0.000009322025 | `near_frontier_secondary` |
| 12 | PR106-R2-lowlevel | `sidecar_framing_meta_pr101` | `control_or_metadata` | 6 | 0.000009322025 | `near_frontier_secondary` |

Score truth remains exact CUDA replay JSON. The score-lowering frontier is internal routing authority only; promotion still requires adjudication policy review.
