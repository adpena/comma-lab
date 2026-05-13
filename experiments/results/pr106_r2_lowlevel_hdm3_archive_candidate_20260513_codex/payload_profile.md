# HNeRV Frontier Payload Profile: experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/pr106_r2_lowlevel_hdm3_archive_candidate.zip

- archive_bytes: `186615`
- archive_sha256: `8cc7e3b21a5f77604331abb727c105e21351e8c199456db741eecb1fc7714093`
- zip_member: `0.bin`
- member_bytes: `186507`
- member_sha256: `c22a9507869a720bb57ccdfae39f7a9afd2981ff5dc7d1ee36922af5a5dafcef`
- inferred_kind: `pr106_sidecar_wrapper`
- zip_overhead_bytes: `108`

| section | start | end | bytes | entropy b/B | sha256 |
|---|---:|---:|---:|---:|---|
| pr106_sidecar_header_fe_fmt_len_u32 | 0 | 6 | 6 | 2.251629 | `c0d40b5b0147c8695ab900e290d9e450239441fa29bbd5621aeabdb747d20807` |
| inner_packed_header_ff_len24 | 6 | 10 | 4 | 2.000000 | `5196cd3e7b079990527695553be2683aff54fd65f2d4b72de34c1d32c2d4ec14` |
| inner_decoder_packed_brotli | 10 | 170123 | 170113 | 7.998040 | `149a41ecf4d1614757c9369838e3a7cb9f03a648fe9b61a2317e9b7f2996b256` |
| inner_latents_and_sidecar_brotli | 170123 | 185972 | 15849 | 7.985386 | `94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32` |
| sidecar_len_u16 | 185972 | 185974 | 2 | 1.000000 | `d26b715881de3752c4e857a13e1e3d00d65ff80c57cdfcb81bc2753cfc8f7a41` |
| sidecar_payload_pr101_ranked_no_op | 185974 | 186501 | 527 | 7.429261 | `b017563f85cd8e6b44bed10782451894f18eb75b8bcf54945ba6a288b9516dd6` |
| sidecar_framing_meta_pr101 | 186501 | 186507 | 6 | 1.459148 | `62fefebec3cdc23b944f9a1d02f68c8bcc6861fa45beea02d92d2c28be03de09` |
