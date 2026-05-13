# HNeRV Frontier Payload Profile: experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/pr106_r2_pr101_grammar_hnerv_brotli_repack_candidate.zip

- archive_bytes: `186629`
- archive_sha256: `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- zip_member: `0.bin`
- member_bytes: `186521`
- member_sha256: `01bd18ebf200533eb9ee81a3ca545ed5df07b95afb3608bbb195b00b4c37fb71`
- inferred_kind: `pr106_sidecar_wrapper`
- zip_overhead_bytes: `108`

| section | start | end | bytes | entropy b/B | sha256 |
|---|---:|---:|---:|---:|---|
| pr106_sidecar_header_fe_fmt_len_u32 | 0 | 6 | 6 | 2.251629 | `0dfe359c42f7430dd4ca7e743182a3cddedd56b7c433e51f1ba795a857383f1d` |
| inner_packed_header_ff_len24 | 6 | 10 | 4 | 2.000000 | `c670d1bee2140039bc6fc23599dd4bdc2ec00a659e0815945a7d60cfe6daccc7` |
| inner_decoder_packed_brotli | 10 | 170137 | 170127 | 7.998224 | `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c` |
| inner_latents_and_sidecar_brotli | 170137 | 185986 | 15849 | 7.985386 | `94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32` |
| sidecar_len_u16 | 185986 | 185988 | 2 | 1.000000 | `d26b715881de3752c4e857a13e1e3d00d65ff80c57cdfcb81bc2753cfc8f7a41` |
| sidecar_payload_pr101_ranked_no_op | 185988 | 186515 | 527 | 7.429261 | `b017563f85cd8e6b44bed10782451894f18eb75b8bcf54945ba6a288b9516dd6` |
| sidecar_framing_meta_pr101 | 186515 | 186521 | 6 | 1.459148 | `62fefebec3cdc23b944f9a1d02f68c8bcc6861fa45beea02d92d2c28be03de09` |
