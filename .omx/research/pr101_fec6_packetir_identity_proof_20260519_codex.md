# PR101/FEC6 PacketIR Identity Proof

- Schema: `pr101_fec6_packetir_identity_proof_v1`
- Archive: `/Users/adpena/Projects/pact/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`
- Archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Member: `x`
- Member bytes: `178417`
- Member SHA-256: `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`
- Re-emit identity: `True`
- PacketIR identity passed: `True`
- Score claim: `False`
- Promotion eligible: `False`
- Ready for exact eval dispatch: `False`

## Blockers

- None

## Sections

| name | offset | end | length | sha256 |
|---|---:|---:|---:|---|
| fp11_magic | 0 | 4 | 4 | `349d73e1d171f3a88dabf70abb9a5a9ef7c24194a1ff68347479eb6d7d2d3115` |
| source_len_u32le | 4 | 8 | 4 | `fbf50690bdbd147096140de72c452ac52e1e154e160ba2d823bf48a1f136d845` |
| source_pr101_payload | 8 | 178166 | 178158 | `5f1948f9572e65f71c614d2ff15764ee416522e25cb1b06c8b1299c1306e8aaf` |
| selector_len_u16le | 178166 | 178168 | 2 | `29fb468aa10598ab84c07ba09235b22ba69986fb964ac16f536a100ce00c54ea` |
| selector_fec6_payload | 178168 | 178417 | 249 | `fc5c431b5d793c33e2f320076fe6f0dd76c2d91e3826ae4b05abfb4f86f453ca` |
| selector_fec6_magic | 178168 | 178172 | 4 | `5efca915edc72a2bfb69666763994407fe322834a4276604cf4ae32b6e9cfa3d` |
| selector_fec6_n_pairs_u16le | 178172 | 178174 | 2 | `8269f5fe5232319b06beee2ebf5111e87e4755866dbdf4d28d2aa1b0c10fa2df` |
| selector_fec6_fixed_huffman_bitstream | 178174 | 178417 | 243 | `d4f30e9900fa2902f11edd16c5aa3d63c71a076aa0cccca442f7f1c200b33c37` |
| packet_member_payload | 0 | 178417 | 178417 | `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd` |
