# HNeRV Frontier Scorecard

| label | grade | canonical | scope | score | bytes | seg | pose | rate | largest section | archive sha |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| PR106x-lowlevel-brotli | A++ | yes | `exact_local_cuda_custody_lossless_repack_control` | 0.209350736806 | 186080 | 0.067142000 | 0.018305737 | 0.123903000 | `decoder_packed_brotli:170127` | `b0a12549a39e34a0` |
| PR106x | A++ | yes | `exact_local_cuda_custody` | 0.209451236806 | 186231 | 0.067142000 | 0.018305737 | 0.124003500 | `decoder_packed_brotli:170278` | `d25bca80057e8b53` |
| PR106 | A++ | yes | `exact_local_cuda_custody` | 0.209456736806 | 186239 | 0.067142000 | 0.018305737 | 0.124009000 | `decoder_packed_brotli:170278` | `3fefbe5dfdd73817` |
| PR105x | A++ | yes | `exact_local_cuda_custody` | 0.230431829870 | 177849 | 0.070456000 | 0.041553580 | 0.118422250 | `decoder_packed_brotli:161891` | `692a46931f66416a` |
| PR105 | A++ | yes | `exact_local_cuda_custody` | 0.230437329870 | 177857 | 0.070456000 | 0.041553580 | 0.118427750 | `decoder_packed_brotli:161891` | `597ba0732810eba0` |

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
| PR106x-lowlevel-brotli | `decoder_packed_brotli` | 170127 | 7.998224 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR105 | `decoder_packed_brotli` | 161891 | 7.998095 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR105x | `decoder_packed_brotli` | 161891 | 7.998095 | build byte-different archive, then exact CUDA replay | decoder self-compression or weight-stream recoding fixture |
| PR105 | `latents_and_sidecar_brotli` | 15854 | 7.985312 | build byte-different archive, then exact CUDA replay | latent/sidecar arithmetic-coding parity fixture |
| PR105x | `latents_and_sidecar_brotli` | 15854 | 7.985312 | build byte-different archive, then exact CUDA replay | latent/sidecar arithmetic-coding parity fixture |
| PR106 | `latents_and_sidecar_brotli` | 15849 | 7.985386 | build byte-different archive, then exact CUDA replay | latent/sidecar arithmetic-coding parity fixture |
| PR106x | `latents_and_sidecar_brotli` | 15849 | 7.985386 | build byte-different archive, then exact CUDA replay | latent/sidecar arithmetic-coding parity fixture |
| PR106x-lowlevel-brotli | `latents_and_sidecar_brotli` | 15849 | 7.985386 | build byte-different archive, then exact CUDA replay | latent/sidecar arithmetic-coding parity fixture |
| PR105 | `packed_header_ff_len24` | 4 | 2.000000 | build byte-different archive, then exact CUDA replay | payload grammar audit |
| PR105x | `packed_header_ff_len24` | 4 | 2.000000 | build byte-different archive, then exact CUDA replay | payload grammar audit |

## Next Exact-Evaluable Target

| frontier | target label | section | role | bytes | score gap | required next gate |
|---|---|---|---|---:|---:|---|
| PR106x-lowlevel-brotli | PR106x-lowlevel-brotli | `decoder_packed_brotli` | `decoder_weight_stream` | 170127 | 0.000000000000 | build byte-different archive with old/new section SHA-256 and charged-byte proof, then exact CUDA auth eval after lane claim |

This target is ranked by exact-frontier proximity first, then charged
payload byte mass. It is a routing target only; it is not a new score
claim.

## Hidden-Gem Byte-Mass Ranking

| rank | label | section | role | bytes | score gap | priority |
|---:|---|---|---|---:|---:|---|
| 1 | PR106x-lowlevel-brotli | `decoder_packed_brotli` | `decoder_weight_stream` | 170127 | 0.000000000000 | `current_frontier_primary` |
| 2 | PR106x-lowlevel-brotli | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | 0.000000000000 | `current_frontier_primary` |
| 3 | PR106x-lowlevel-brotli | `packed_header_ff_len24` | `control_or_metadata` | 4 | 0.000000000000 | `low` |
| 4 | PR106x | `decoder_packed_brotli` | `decoder_weight_stream` | 170278 | 0.000100500000 | `near_frontier_secondary` |
| 5 | PR106x | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | 0.000100500000 | `near_frontier_secondary` |
| 6 | PR106x | `packed_header_ff_len24` | `control_or_metadata` | 4 | 0.000100500000 | `low` |
| 7 | PR106 | `decoder_packed_brotli` | `decoder_weight_stream` | 170278 | 0.000106000000 | `near_frontier_secondary` |
| 8 | PR106 | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | 0.000106000000 | `near_frontier_secondary` |
| 9 | PR106 | `packed_header_ff_len24` | `control_or_metadata` | 4 | 0.000106000000 | `low` |
| 10 | PR105x | `decoder_packed_brotli` | `decoder_weight_stream` | 161891 | 0.021081093064 | `near_frontier_secondary` |
| 11 | PR105x | `latents_and_sidecar_brotli` | `latent_stream` | 15854 | 0.021081093064 | `near_frontier_secondary` |
| 12 | PR105x | `packed_header_ff_len24` | `control_or_metadata` | 4 | 0.021081093064 | `low` |

## Payload Section Manifests

| label | section | role | bytes | sha256 | required proof |
|---|---|---|---:|---|---|
| PR105 | `packed_header_ff_len24` | `control_or_metadata` | 4 | `837b877d82478a79` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105 | `decoder_packed_brotli` | `decoder_weight_stream` | 161891 | `e241db1c2cbfb53a` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105 | `latents_and_sidecar_brotli` | `latent_stream` | 15854 | `aeea25a3116b4457` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105x | `packed_header_ff_len24` | `control_or_metadata` | 4 | `837b877d82478a79` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105x | `decoder_packed_brotli` | `decoder_weight_stream` | 161891 | `e241db1c2cbfb53a` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR105x | `latents_and_sidecar_brotli` | `latent_stream` | 15854 | `aeea25a3116b4457` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106 | `packed_header_ff_len24` | `control_or_metadata` | 4 | `7939f08db7d18dd4` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106 | `decoder_packed_brotli` | `decoder_weight_stream` | 170278 | `654999f81f0552fb` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106 | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | `94257b33cf3083c5` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x | `packed_header_ff_len24` | `control_or_metadata` | 4 | `7939f08db7d18dd4` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x | `decoder_packed_brotli` | `decoder_weight_stream` | 170278 | `654999f81f0552fb` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | `94257b33cf3083c5` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x-lowlevel-brotli | `packed_header_ff_len24` | `control_or_metadata` | 4 | `c670d1bee2140039` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x-lowlevel-brotli | `decoder_packed_brotli` | `decoder_weight_stream` | 170127 | `07725c39ff436195` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |
| PR106x-lowlevel-brotli | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | `94257b33cf3083c5` | future candidate must record old/new section SHA-256 and old/new charged bytes before exact-eval dispatch |

Interpretation: score truth remains the exact CUDA replay JSON. Payload
sections are forensic signals for the next compression action; they do
not imply score deltas without a new exact archive eval.

Guardrail: a lossless Brotli repack row is a local exact-custody
byte-control. It does not supersede categorical/range-coded HNeRV
families unless that exact candidate archive has a lower CUDA score
under the same custody standard.
