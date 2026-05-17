# Rule #6 A1 Bolt-On Byte Profile Refresh

Date: 2026-05-17
Author: Codex
Scope: byte-closed Rule #6-on-A1 frontier preparation, no dispatch, no score claim.

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- Evidence axis: `[macOS-CPU advisory]` byte/information profile only.
- A1 archive: `submissions/a1/archive.zip`
- Archive bytes: `178262`
- Archive SHA-256: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- Single member: `x`, `178162` bytes, SHA-256 `8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243`

## Parser Correction

The live A1 packet is not the public-PR101 fixed-offset layout starting at byte
0. It is the prefixed no-dead-K layout consumed by `submissions/a1/inflate.py`:

| section | byte range | bytes | SHA-256 |
|---|---:|---:|---|
| `decoder_section_total_u32le` | `[0, 4)` | 4 | `e06964de19aafb9522af14a6619f90c05058093a39cfc3ffec555a8f95417e8a` |
| `decoder_blob` | `[4, 162168)` | 162164 | `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6` |
| `latent_blob` | `[162168, 177555)` | 15387 | `de8a0da594f073efc43849334573ba06438bb37d53f9343ee6367659c0106bbe` |
| `sidecar_blob` | `[177555, 178162)` | 607 | `6c2946e323bbbc6f8d906ef6c68989e8acbd8d60332c87da8fe8147f1ea7b12f` |

This matters for Rule #6: a fixed-offset PR101 surgery tool would slice the A1
decoder four bytes early and corrupt the latent/sidecar boundary. The parser
surface now recognizes `a1_prefixed_hnerv_microcodec`, and the PR101 surgery
tool now refuses A1-prefixed bytes.

## Entropy Profile

Generated artifact:
`.omx/research/rule6_a1_bolton_byte_profile_20260517_codex.json`

Command:

```bash
.venv/bin/python tools/payload_entropy_density_map_local.py \
  --archive submissions/a1/archive.zip \
  --output-json .omx/research/rule6_a1_bolton_byte_profile_20260517_codex.json
```

Section summary:

| section | bytes | H1 bits/byte | 1st-order savings | H(X\|1) bits/byte | implied order-1 headroom |
|---|---:|---:|---:|---:|---:|
| `decoder_blob` | 162164 | 7.9984 | 31 B | 7.6706 | ~6645 B |
| `latent_blob` | 15387 | 7.9883 | 22 B | 5.6948 | ~4411 B |
| `sidecar_blob` | 607 | 7.7105 | 21 B | 1.5218 | ~470 B |

Interpretation:

1. ZIP-level or first-order entropy repacking is saturated. It cannot credibly
   explain a frontier-breaking gain.
2. The sidecar is small but highly structured. It is worth a surgical Huffman /
   enumerative-code pass because it is isolated and easy to prove consumed.
3. The latent blob is the highest-EV immediate Rule #6 target: it has much more
   higher-order conditional structure than first-order entropy suggests, and it
   is the section Z3/Ballé/VQ-style bolt-ons are supposed to replace.
4. Decoder-byte work should only proceed as a parser-faithful compiler pass or
   generated-schema codec; byte-only generic recompression is not a serious
   next move.

## Next Build Decision

Immediate next artifact should be an A1-specific Rule #6 latent replacement
prototype that:

1. Uses the prefixed A1 parser above, not PR101 fixed-offset slicing.
2. Emits a single-member `x` packet with the same decoder header/decoder bytes
   unless the decoder itself is the explicit target.
3. Replaces `latent_blob` with a byte-counted section and keeps A1 sidecar
   semantics explicit.
4. Ships a byte-mutation/no-op proof at the latent-section boundary before any
   CPU/CUDA exact eval claim.
5. Keeps `score_claim=false` until paired exact CPU/CUDA artifacts exist.

