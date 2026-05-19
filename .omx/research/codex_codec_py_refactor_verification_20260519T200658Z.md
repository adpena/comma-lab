# Codex Verification — codec.py sidecar refactor byte identity

**UTC:** 2026-05-19T20:06:58Z  
**Directive:** `.omx/research/codex_routing_directive_codec_py_refactor_with_byte_identity_verification_20260519T211500Z.md`  
**Target:** `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py`

## Operator directive

> "we should delegate this to our pal codex with a design memo"

The routed task was to refactor `codec.py` while preserving byte identity for
the frozen FEC6 packet anchored by archive SHA-256:

```text
6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
```

## Protocol corrections

The directive's intent was correct, but the literal shell sketch needed three
mechanical corrections before execution:

- `inflate.sh` expects an extracted member `x`, not `archive.zip` directly.
  I extracted `archive.zip` member `x` and used that extracted directory as
  the inflate data directory.
- Output hashes were computed from inside each output directory, producing
  relative paths like `./0.raw`; this avoids false diffs from temp-directory
  prefixes.
- The target file lives under an ignored experiment artifact tree. The refactor
  commit therefore force-added this one explicit source file before invoking
  the canonical commit serializer with `--no-stage`.

No score or promotion claim is made by this report.

## Pre-refactor inventory

- LOC: 480
- Pre-refactor source SHA-256: `637fa5e4b47bfb2595358903dfaafbbc7a7ca48d5f7ccd81c06d1397e060bb72`
- Public API consumed by `inflate.py`: `parse_archive(archive_bytes) -> (decoder_sd, latents, meta)`
- Function inventory:

```text
unpack_nibbles
unpack_3bit_lengths
decode_canonical_huffman
decode_canonical_huffman_all
huff_length_vector_count
decode_huff_length_rank
decode_combination_colex
zigzag_decode_u8
decode_mapped_u8
decompress_brotli_streams
decode_decoder_compact
decode_latents_compact
apply_latent_sidecar
parse_archive
```

## Post-refactor inventory

- LOC: 508
- Post-refactor source SHA-256: `124623c07e066e82f0f1ad5e2aaec6c0cb31b195266c406eca76604e8a934deb`
- Refactor commit: `32c4e87d4`
- Public API preserved: `parse_archive(archive_bytes)` still returns
  `(decoder_sd, latents, meta)`.
- Refactor shape: same-file private helper extraction inside the latent
  sidecar decoder. Decoder tensor reconstruction, latent cumsum/wraparound,
  `parse_archive`, imports, constants, and final Torch application order were
  left unchanged.
- New helper inventory:

```text
_packed_dims
_vectors_from_valid
_vectors_from_choices
_decode_enum_rank_sidecar
_decode_comb_rank_sidecar
_decode_split_sidecar
_decode_packed_choice_sidecar
_decode_latent_sidecar_vectors
```

## Baseline proof

Archive member `x` SHA-256:

```text
f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd
```

Baseline run 1:

```text
d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c  ./0.raw
```

Baseline run 2:

```text
d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c  ./0.raw
```

Baseline determinism diff: empty (`0` bytes).

## Post-refactor proof

Post-refactor run:

```text
d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c  ./0.raw
```

Post-refactor diff against baseline run 1: empty (`0` bytes).

## Verdict

PASS. The refactor preserves inflated output byte identity for the full
1200-frame public-test output under the local CPU inflate path. The frozen
archive bytes were not changed.

Residual caution: the tracked commit now includes this source file from an
otherwise ignored experiment artifact tree. That was intentional for no-signal
loss and because the operator-routed design memo targets this exact file, but
future packet-maintenance work should either promote the full submission packet
into a first-class tracked surface or keep such changes as explicit patch
artifacts.
