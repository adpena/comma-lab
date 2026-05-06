# HNeRV Section Repack Plan

- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`
- selected_labels: `PR106x`
- total_section_bytes: `186131`

| label | section | role | bytes | 1% rate gain | 5% rate gain | required proof | action |
|---|---|---|---:|---:|---:|---|---|
| PR106x | `decoder_packed_brotli` | `decoder_weight_stream` | 170278 | 0.001133292 | 0.005668457 | candidate manifest must include source section sha256, candidate section sha256, source bytes, candidate bytes, and exact archive sha256 | build decoder self-compression or weight-stream recoding fixture |
| PR106x | `latents_and_sidecar_brotli` | `latent_stream` | 15849 | 0.000105206 | 0.000527360 | candidate manifest must include source section sha256, candidate section sha256, source bytes, candidate bytes, and exact archive sha256 | build latent arithmetic-coding parity fixture |
| PR106x | `packed_header_ff_len24` | `control_or_metadata` | 4 | 0.000000666 | 0.000000666 | candidate manifest must include source section sha256, candidate section sha256, source bytes, candidate bytes, and exact archive sha256 | audit fixed header and metadata compaction |
