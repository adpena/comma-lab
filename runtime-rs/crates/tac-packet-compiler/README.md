# `tac-packet-compiler`

Rust native port for [`tac.packet_compiler`](../../../src/tac/packet_compiler/),
the reusable byte-grammar and entropy-coder primitives inside the `tac`
Task-Aware Compression library, extracted from the public PR101
(`hnerv_ft_microcodec`) and PR103 (`hnerv_lc_ac`) submissions.

> **Status — `v0.2.0-rc1` LOCAL release tag (2026-05-11). The 19 committed golden-vector primitives are byte-for-byte parity GREEN; this is not complete crate parity because selector/search/decode meta surfaces remain explicit scaffolds.**
>
> Per council Q5 verdict (b) (9/10 Shannon/Dykstra/Yousfi/Fridrich/Quantizr/
> Hotz/Selfcomp/MacKay/Ballé): tag the GitHub release locally first as a
> reversible artifact; **crates.io publish is OPERATOR-GATED** and
> IRREVERSIBLE per crates.io no-unpublish policy. The `-rc1` suffix +
> `publish = false` are intentional; they stay until the operator approves
> crates.io publish (drop suffix; flip to `publish = true`).
>
> Per operator directive 2026-05-11 ("don't submit the PR yet"): the
> v0.2.0-rc1 tag is **LOCAL ONLY** — no `git push --tags` until operator
> explicitly authorizes.
>
> Implemented + parity-verified against committed Python golden vectors:
>
> 1. **`encode_centered_delta_uint8` / `decode_centered_delta_uint8`** —
>    raw LZMA1 (FILTER_LZMA1, dict=4 KiB, lc=3, lp=0, pb=0) over fp16
>    mins/scales + column-major centered-delta uint8 latents. SHA-256
>    `939c66…4af9`.
> 2. **`split_brotli_self_delimiting` / `parse_split_brotli_self_delimiting`** —
>    PR101's concatenated-Brotli wire format (no length prefixes; the
>    decoder uses Brotli frame structure to detect each stream's end).
>    SHA-256 `8184f2…4915`.
> 3. **`encode_latent_hi_arithmetic` / `decode_latent_hi_arithmetic`** —
>    PR103's single-tensor categorical arithmetic coder (constriction
>    `DefaultRangeEncoder` + `DefaultContiguousCategoricalEntropyModel`,
>    serialised as **big-endian uint32**). SHA-256 `6381f1…befc`.
> 4. **`encode_ranked_no_op_sidecar`** (PR101 "huff_enum" variant) —
>    bounded-length package-merge Huffman + canonical-Huffman MSB-first
>    bit-packing + co-lex combination rank over no-op positions +
>    mixed-radix dim packing. SHA-256
>    `565a7b…786ba`.
> 5. **`encode_merged_range_stream` / `decode_merged_range_stream`** —
>    PR103's multi-tensor constriction range coder with per-tensor
>    categoricals re-used across every symbol of that tensor; symbol-count
>    boundaries serialised out-of-band. SHA-256 `ab00a8…b1a1`.
> 6. **`encode_delta_varint_pose` / `decode_delta_varint_pose`** —
>    PR93's QZPDV1 pose codec: 8-byte magic + LE shape header + fp32
>    lo/scale + uint8/uint16 absolute first row + zigzag-LEB128 deltas.
>    SHA-256 `e10f29…d988`.
> 7. **`encode_categorical_stream` / `decode_categorical_stream`** —
>    PR91 HPACMini universal AC wrapper: one
>    `DefaultContiguousCategoricalEntropyModel` per symbol position;
>    big-endian uint32 serialisation. SHA-256 `f208c4…c88f`.
> 8. **`encode_adaptive_context_stream` / `decode_adaptive_context_stream`** —
>    PR84 adaptive context-routed range coder: per-context categoricals
>    routed by parallel `context_ids` array. SHA-256 `cd0f0d…a28e`.
> 9. **`FP4Codebook::quantize` + `pack_nibbles`** — PR81 asymmetric 8-level
>    FP4 codebook (`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`) + sign bit; 2 nibbles per
>    byte (hi-first). SHA-256 `12b69f…4e40`.
> 10. **`encode_router_actions` / `decode_router_actions`** — PR81 LSB-first
>     bit-packing for small-integer per-frame action streams (`bits=3,
>     count=600 → 225 bytes`); generalises to `1 <= bits <= 8`. SHA-256
>     `3360cb…9878`.
> 11. **`emit_qmqh_header` + `pack_hi_lo_split` / `parse_qmqh_header` +
>     `unpack_hi_lo_split`** — PR91 QM0/QH0 magic-prefix header + hi-lo
>     nibble split permutation. SHA-256 `6262b0…25ac`.
> 12. **`pack_rmc1_composite` + `pack_rsa1_side` (+ `pack_rsb1_side`)** —
>     PR92 RMC1/RSA1/RSB1 joint-stream meta-codec: composite framing for
>     correlated mask + side-action streams; RSB1 brotli-fallback for
>     peaked action distributions. SHA-256 `683827…b2f5`.
> 13. **`pack_qzmb1_block` / `unpack_qzmb1_block`** — PR93 QZMB1 compact-
>     model block grammar: 8-byte magic + `<HH` header (block_size,
>     arch_len) + arch JSON + opaque body. SHA-256 `d1c85e…8725`.
> 14. **`serialize_lowpass_luma_residual` / `deserialize_lowpass_luma_residual`** —
>     PR93 low-pass luma residual: `[u8 n_coeffs][u16 LE height][u16 LE width][n_coeffs * fp32 LE]`.
>     SHA-256 `72705f…ceaa`.
> 15. **`encode_length_prefixed_sections` / `decode_length_prefixed_sections`** —
>     PR97 H3 multi-section grammar: concatenated `[u32 LE len][bytes]` pairs;
>     no global section count prefix. SHA-256 `2be96e…d517`.
> 16. **`encode_tile_band_streams` / `decode_tile_band_streams`** — PR97 H3
>     tile-band wire format: `[u32 LE n_chunks][u32 LE size_i][bytes_i]...`.
>     SHA-256 `cb3ca2…0029`.
> 17. **`encode_rle_of_zeros` / `decode_rle_of_zeros` /
>     `serialize_rle_of_zeros` / `deserialize_rle_of_zeros`** — sparse PacketIR
>     RLE-of-zeros: `b"SRL1"` magic + LE u32 lengths + LE i8/i16/i32 nonzero
>     values. Compresses 1024 dense int8 to 333 bytes at 93.75% sparsity.
>     SHA-256 `fcff82…fd57`.
> 18. **`encode_arithmetic_coefficients` / `decode_arithmetic_coefficients` /
>     `serialize_arithmetic_coefficients` / `deserialize_arithmetic_coefficients`** —
>     sparse PacketIR arithmetic coder: `b"SAC1"` magic + per-stream f32
>     histogram + constriction big-endian uint32 word stream; +1.0 Laplace
>     smoothing matches Python oracle's `np.bincount + 1.0`. SHA-256
>     `267221…29d8`.
> 19. **`encode_temporal_subsampled` / `decode_temporal_subsampled` /
>     `serialize_temporal_subsampled` / `deserialize_temporal_subsampled`** —
>     sparse PacketIR temporal subsampling: `b"STS1"` magic + LE u32 N/K/per_frame_bytes
>     + LSB-first indicator bitmap + densely-packed K residuals. SHA-256
>     `f4b2e6…38c0`.
>
> Remaining non-primitive/meta scaffolds are explicit: `magic_codec_v1` is a
> per-stream auto-selector OVER primitives 1-19 and remains `try_load_only`;
> `adaptive_brotli_param_search` has no byte-level golden vector by design;
> `decode_ranked_no_op_sidecar` still needs the inverse length-rank + Huffman
> bit-unpacker. The Python oracle remains authoritative for those surfaces.

## Why this exists

The 2026-05-11 score-lowering handoff names `tac.packet_compiler` as the
reusable layer all sub-0.20 PRs rediscover. The same handoff's
[Bottom-line next tranche, item #7](../../../../../Downloads/pact_score_lowering_handoff_2026-05-11.md)
calls for "the first native PacketIR proof over the committed golden
vectors, not over an unpinned research script." This crate is that native
proof for the implemented golden-vector primitives.

Per [CLAUDE.md "Deterministic packet compiler" non-negotiable](../../../CLAUDE.md):

> Low-level native/codegen work should converge into a separate deterministic
> submission-packet compiler. It must ingest a contest-compliant packet,
> deconstruct archive/runtime/payload bytes into a typed manifest and golden
> vectors, then emit either byte-identical output or an intentionally
> byte-different packet with exact old/new SHA-256 and charged-byte proof.

Per [CLAUDE.md "Beauty, simplicity, and developer experience"](../../../CLAUDE.md):

> Rust/Zig is a speed layer, not a license to change semantics. Python
> remains the oracle until native parity is proven on the golden vectors.

## Contract

| Property | Honoured by |
|---|---|
| **Oracle** | [`src/tac/packet_compiler/`](../../../src/tac/packet_compiler/) (Python). |
| **Speed layer** | This crate. |
| **Promotion path** | Python ➜ golden vector ➜ Rust impl ➜ parity test ➜ deterministic rebuild. |
| **Identity rule** | `decode(encode(x)) == x` byte-for-byte across both languages. |
| **No scorer load** | No PoseNet / SegNet / FastViT / EfficientNet types anywhere in this crate. |
| **No `/tmp`** | Golden vectors live under [`src/tac/packet_compiler/golden_vectors/`](../../../src/tac/packet_compiler/golden_vectors/). |
| **OSS license** | MIT OR Apache-2.0 — matches the Python `tac` crate. |
| **No `unsafe`** | `#![forbid(unsafe_code)]` at the crate root. |

## Layout

```
runtime-rs/crates/tac-packet-compiler/
├── Cargo.toml
├── README.md                            (this file)
├── src/
│   ├── lib.rs                            public API + error type
│   ├── pr101_sidecar_grammar/
│   │   ├── mod.rs                        re-exports
│   │   ├── stubs.rs                      delegates to impls; remaining unimplemented!() stubs
│   │   ├── centered_delta_uint8.rs       IMPL: LZMA1 raw centered-delta uint8 (parity GREEN)
│   │   ├── ranked_no_op_sidecar.rs       IMPL: PR101 ranked Huffman/no-op sidecar (parity GREEN)
│   │   └── split_brotli.rs               IMPL: self-delimiting concat-Brotli (parity GREEN)
│   ├── pr103_arithmetic_coding/
│   │   ├── mod.rs                        re-exports
│   │   ├── stubs.rs                      delegates to impls; remaining unimplemented!() stubs
│   │   ├── latent_hi.rs                  IMPL: single-tensor categorical AC (parity GREEN)
│   │   └── merged_range_stream.rs        IMPL: multi-tensor constriction range coder (parity GREEN)
│   ├── pr93_pose_codec/
│   │   ├── mod.rs                        re-exports
│   │   └── delta_varint.rs               IMPL: PR93 QZPDV1 zigzag-LEB128 pose codec (parity GREEN)
│   ├── pr91_hpac_grammar/
│   │   ├── mod.rs                        re-exports
│   │   └── arithmetic_coder_constriction.rs  IMPL: per-symbol categorical AC (parity GREEN)
│   ├── pr84_adaptive_mask/
│   │   ├── mod.rs                        re-exports
│   │   └── adaptive_mask_context.rs      IMPL: per-context adaptive-context coder (parity GREEN)
│   ├── pr81_quantizr/
│   │   ├── mod.rs                        re-exports
│   │   ├── fp4_codebook.rs               IMPL: asymmetric 8-level FP4 codebook + nibble packing (parity GREEN)
│   │   └── router_action.rs              IMPL: LSB-first bit-stream pack/unpack (parity GREEN)
│   ├── pr92_joint_stream/
│   │   ├── mod.rs                        re-exports
│   │   └── rmc.rs                        IMPL: RMC1/RSA1/RSB1 joint-stream meta-codec (parity GREEN)
│   ├── pr91_hpac_grammar/qmqh_grammar.rs    IMPL: QM0/QH0 magic + hi-lo split (parity GREEN; sibling of arithmetic_coder_constriction)
│   ├── pr93_pose_codec/qzmb1.rs             IMPL: QZMB1 compact-model block (parity GREEN; sibling of delta_varint)
│   ├── sparse_packet_ir/                 IMPL: sparse RLE/AC/temporal primitives (parity GREEN)
│   └── conformance/
│       └── mod.rs                        golden-vector loader + sha256 helpers
├── tests/
│   └── golden_vector_parity.rs           parity gate against committed vectors
└── benches/
    └── golden_vector_parity.rs           criterion scaffold (no-op sentinel)
```

The binary input fixtures consumed by the parity tests live alongside the
JSON manifests under
[`src/tac/packet_compiler/golden_vectors/*_v1_*.bin`](../../../src/tac/packet_compiler/golden_vectors/).
Regenerate them via
[`tools/regenerate_packet_compiler_rust_parity_fixtures.py`](../../../tools/regenerate_packet_compiler_rust_parity_fixtures.py)
whenever the Python recipe changes.

## Dependency choices

| Dep | Version | Why |
|---|---|---|
| [`constriction`](https://crates.io/crates/constriction) | `0.4` | Same upstream (Bamler Lab) as the Python `constriction` package; license MIT OR Apache-2.0 OR BSL-1.0; byte-for-byte uint32 stream parity is the design target. |
| [`brotli`](https://crates.io/crates/brotli) | `8` | Pure Rust impl; wire-format identical to the Python `brotli` package; matches PR101 `lgwin=22, quality=11` and PR103 adaptive search. |
| [`liblzma`](https://crates.io/crates/liblzma) | `0.4` | Rust bindings to the **same** liblzma C library that Python's stdlib `lzma` wraps; raw `FILTER_LZMA1 dict=4096 lc=3 lp=0 pb=0` parity is the design target. |
| [`serde` + `serde_json`](https://crates.io/crates/serde) | `1` | Golden-vector manifest parsing. |
| [`sha2`](https://crates.io/crates/sha2) | `0.10` | Byte-level parity verification against the committed digests. |
| [`hex`](https://crates.io/crates/hex) | `0.4` | Human-readable SHA-256 diff diagnostics. |
| [`ndarray`](https://crates.io/crates/ndarray) | `0.16` | numpy-equivalent shape semantics for the merged range stream + centered-delta uint8 reshape. |
| `proptest` (dev) | `1` | Round-trip property tests once impl lands. |
| `criterion` (dev) | `0.6` | Microbenchmark scaffold. |

## How to verify the native port

```
$ cargo test -p tac-packet-compiler
```

Should pass green. Implemented primitives assert byte-for-byte SHA parity
against the committed Python golden vectors; remaining scaffold-only surfaces
are explicit `try_load_only` or `NotImplemented` tests so they cannot masquerade
as native implementations. The coverage gate
(`every_golden_vector_has_paired_parity_test`) confirms every committed vector
has a matching Rust-side test entry. If you add a new vector on the Python side,
this test fails until you wire a paired Rust parity or explicit load-only test.

## How to promote a remaining scaffold

When a remaining Rust function lands, flip the corresponding test in
`tests/golden_vector_parity.rs`:

```rust
// BEFORE (scaffold):
let result = encode_ranked_no_op_sidecar(&dims, &delta_indices, &schema);
assert_scaffold_refuses(result, "encode_ranked_no_op_sidecar");

// AFTER (impl):
let manifest = try_load("ranked_no_op_sidecar_v1").expect("vector must exist");
let payload = encode_ranked_no_op_sidecar(&dims, &delta_indices, &schema)
    .expect("encode must succeed");
tac_packet_compiler::conformance::assert_sha256_parity(&payload, &manifest)
    .expect("byte-for-byte parity against committed vector");
```

Any mismatch surfaces as a structured
`PacketCompilerError::SidecarShaMismatch { schema, produced, expected }` so
the CI diagnostic is human-readable.

## Roadmap

The first 3 primitives (cheapest per N D4 council verdict) landed
2026-05-11 with byte-for-byte parity GREEN; the next 5 (PR101 ranked
sidecar + PR103 merged range stream + PR93 pose + PR91 per-symbol AC +
PR84 adaptive context) landed in the same session per operator directive
"compiler and insanely low level" + "keep building outside the \$5 window";
the next 5 (PR81 FP4 codebook + PR81 ROUTER_ACTION + PR91 QMQH grammar +
PR92 RMC joint stream + PR93 QZMB1 grammar) landed in the same session
under the same directive; the FINAL 6 (PR93 lowpass-luma + PR97 H3
length-prefixed sections + PR97 H3 tile-band streams + sparse RLE-of-zeros
+ sparse arithmetic coefficients + sparse temporal-subsampled) landed in
the same session, bringing the committed golden-vector primitive set to
**19 GREEN**. That is a primitive-set parity claim, not a blanket complete
crate-parity claim; the remaining meta/decode surfaces stay scaffolded below.

| # | Function | Status | Notes |
|---|---|---|---|
| 1 | `encode_centered_delta_uint8` | **GREEN 2026-05-11** | LZMA1 raw stream; +half crate for fp16. |
| 2 | `split_brotli_self_delimiting` | **GREEN 2026-05-11** | brotli 8 + brotli-decompressor 5; pure-Rust brotli IS byte-for-byte compatible with Python C brotli for `(GENERIC, q=11, lgwin=22)`. |
| 3 | `encode_latent_hi_arithmetic` | **GREEN 2026-05-11** | constriction 0.4.2 `DefaultRangeEncoder` + 24-bit categorical; big-endian uint32 serialisation. |
| 4 | `encode_ranked_no_op_sidecar` | **GREEN 2026-05-11** | Most algorithmic of all primitives: bounded-length package-merge Huffman + canonical-Huffman MSB-first bit-packer + co-lex combination rank + mixed-radix dim packing. ~480 LOC. |
| 5 | `encode_merged_range_stream` | **GREEN 2026-05-11** | Multi-tensor constriction encode; one Categorical per tensor; symbol-count boundaries serialised out-of-band. |
| 6 | `encode_delta_varint_pose` (PR93 QZPDV1) | **GREEN 2026-05-11** | Pure-stdlib: 8-byte magic + LE shape header + fp32 lo/scale + uint8/uint16 first row + zigzag-LEB128 row-major deltas. |
| 7 | `encode_categorical_stream` (PR91) | **GREEN 2026-05-11** | Per-position constriction Categorical (one model PER symbol; matrix layout `(n_symbols, alphabet)`). |
| 8 | `encode_adaptive_context_stream` (PR84) | **GREEN 2026-05-11** | Per-context categorical lookup (matrix layout `(n_contexts, alphabet)`); symbols routed by parallel `context_ids` array. |
| 9 | `FP4Codebook::quantize` + `pack_nibbles` (PR81) | **GREEN 2026-05-11** | Asymmetric 8-level positive codebook + sign bit; 2 nibbles per byte (hi << 4 \| lo). Numpy `argmin` tie-break preserved. |
| 10 | `encode_router_actions` (PR81) | **GREEN 2026-05-11** | LSB-first bit packing for `1 <= bits <= 8`; pure stdlib `u64` accumulator. |
| 11 | `emit_qmqh_header` + `pack_hi_lo_split` (PR91) | **GREEN 2026-05-11** | 3-byte magic + per-byte hi-/lo-nibble permutation (run-length-friendly under Brotli). |
| 12 | `pack_rmc1_composite` + `pack_rsa1_side` + `pack_rsb1_side` (PR92) | **GREEN 2026-05-11** | RMC1 outer frame + RSA1 range-coded inner frame + RSB1 brotli-fallback. Brotli compress/decompress mirrors PR101 split-Brotli helpers. |
| 13 | `pack_qzmb1_block` (PR93) | **GREEN 2026-05-11** | 8-byte magic + `<HH` (block_size, arch_len) + opaque arch JSON + opaque tensor body. |
| 14 | `serialize_lowpass_luma_residual` (PR93) | **GREEN 2026-05-11** | Pure stdlib: 5-byte header (`<BHH`) + 3 or 6 fp32 LE coefficients. SHA-256 `72705f…ceaa`. |
| 15 | `encode_length_prefixed_sections` (PR97 H3) | **GREEN 2026-05-11** | Pure stdlib: concatenated `[u32 LE len][bytes]` pairs; no global section count. SHA-256 `2be96e…d517`. |
| 16 | `encode_tile_band_streams` (PR97 H3) | **GREEN 2026-05-11** | Pure stdlib: leading `u32 LE n_chunks` + `[u32 LE size][bytes]` pairs. SHA-256 `cb3ca2…0029`. |
| 17 | `encode_rle_of_zeros` (sparse PacketIR) | **GREEN 2026-05-11** | `b"SRL1"` magic + LE u32 lengths + auto-dtype int8/int16/int32 nonzero values; strictly-increasing index invariant enforced. SHA-256 `fcff82…fd57`. |
| 18 | `encode_arithmetic_coefficients` (sparse PacketIR) | **GREEN 2026-05-11** | `b"SAC1"` magic + LE per-stream f32 histogram + constriction big-endian uint32 stream; +1.0 Laplace smoothing matches Python `np.bincount + 1.0` exactly. SHA-256 `267221…29d8`. |
| 19 | `encode_temporal_subsampled` (sparse PacketIR) | **GREEN 2026-05-11** | `b"STS1"` magic + LE N/K/per_frame_bytes + LSB-first indicator bitmap + densely-packed K residuals. SHA-256 `f4b2e6…38c0`. |
| -  | `adaptive_brotli_param_search` | scaffold | Contract test (Pareto-frontier membership) — no byte-level golden vector by design. |
| -  | `decode_ranked_no_op_sidecar` | scaffold | Inverse of impl #4; needs decode-side length-rank inversion + huff bit-unpacker. |
| -  | `magic_codec_v1` | `try_load_only` | Per-stream auto-selector OVER primitives 1-19; trivial to port now that all underlying primitives are GREEN. |

| Question | Resolution |
|---|---|
| Who implements? | Subagent dispatch under operator directive 2026-05-11 ("compiler and insanely low level" + "keep building outside the \$5 window" + "recursively adversarially review and greenup"); 19 done across 4 sibling-subagent tranches (Y: 3, CC: 5, EE: 5, FF [this landing]: 6). |
| Effort delivered | 19 impls + 1 regen helper + 21 binary fixtures + parity-test wire-in; current unit/integration counts are owned by `cargo test -p tac-packet-compiler`, not this static README. |
| Cost | \$0 GPU. |
| Risk audit | Brotli pure-Rust ↔ Python C library byte-parity was the largest unknown going in; verified GREEN on the committed 3-stream fixture AND on the PR92 RSB1 fallback path. constriction (same upstream as Python package) and liblzma (same C library) were lower-risk and also GREEN. The bounded-length package-merge in PR101 ranked sidecar was the largest algorithmic risk — also GREEN on first attempt. PR81 FP4 codebook required numpy `argmin` tie-break semantics (smallest-index wins) — reproduced exactly in Rust via `<` strict-less comparison. The sparse-AC primitive's Laplace-smoothed empirical histogram was the largest f32-byte-parity risk for the final tranche — verified GREEN on a 500-symbol int32 stream by replicating Python's exact operation order (`np.bincount → astype f32 → += 1.0`). |
| Publish eligibility | **Not publish-ready as a complete native crate** while selector/search/decode meta surfaces remain scaffolded. `publish = false` stays in force until the public API contract and remaining scaffolds are promoted or intentionally excluded by operator decision. |

See [`.omx/research/staged_rust_packet_compiler_native_port_readiness_*.md`](
../../../.omx/research/) for the operator-decision packet.

## CLAUDE.md compliance summary

| Non-negotiable | How this crate complies |
|---|---|
| **Deterministic packet compiler** | Scaffold + golden vectors + parity test harness. Byte-for-byte SHA-256 gate via [`conformance::assert_sha256_parity`](src/conformance/mod.rs). |
| **Beauty, simplicity, DX** | Narrow public API; typed errors; docstrings cite Python oracle line-by-line; `#![forbid(unsafe_code)]`. |
| **Native acceleration as conformance-backed PacketIR port** | Crate is named, wired to committed golden vectors, and implemented primitive surfaces are parity-gated before promotion. |
| **Rust/Zig is a speed layer** | `lib.rs` doc, README, and every stub doc reiterate that Python is the oracle. |
| **No MPS authoritative / no scorer load** | No torch / scorer / inflate-side imports anywhere. |
| **No `/tmp` paths** | Golden-vector resolution uses `CARGO_MANIFEST_DIR`, not env. |
| **Production-hardened OSS direction** | MIT/Apache-2.0 dual license; narrow public surface; `repository` + `description` populated; `publish = false` until contract stabilises. |

## Cross-references

- Python oracle: [`src/tac/packet_compiler/README.md`](../../../src/tac/packet_compiler/README.md).
- Golden vectors: [`src/tac/packet_compiler/golden_vectors/`](../../../src/tac/packet_compiler/golden_vectors/).
- Handoff: `~/Downloads/pact_score_lowering_handoff_2026-05-11.md` (Bottom-line item #7, Native-first candidates 1-2, P4 deterministic packet compiler).
- Operator directive: 2026-05-11, "production hardened OSS direction" + "all of the stuff that would cost more than $100 ready to go in parallel for as soon as we secure funding".
