# `tac-packet-compiler`

Rust native port for [`tac.packet_compiler`](../../../src/tac/packet_compiler/),
the reusable byte-grammar and entropy-coder primitives extracted from the
public PR101 (`hnerv_ft_microcodec`) and PR103 (`hnerv_lc_ac`) submissions.

> **Status — 8 of 19 primitives byte-for-byte-parity GREEN (2026-05-11).**
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
>
> Still scaffold-only (return `NotImplemented` so they cannot silently lie):
> `adaptive_brotli_param_search`, `decode_ranked_no_op_sidecar`, the
> PR81/PR92/PR97 / sparse PacketIR codecs / PR93 lowpass-luma / magic-codec
> / PR91 QMQH / PR93 QZMB1 grammars (each still has its `try_load_only`
> stub harness in `tests/golden_vector_parity.rs`).
>
> See "Roadmap" below for the remaining implementation order.

## Why this exists

The 2026-05-11 score-lowering handoff names `tac.packet_compiler` as the
reusable layer all sub-0.20 PRs rediscover. The same handoff's
[Bottom-line next tranche, item #7](../../../../../Downloads/pact_score_lowering_handoff_2026-05-11.md)
calls for "the first native PacketIR proof over the committed golden
vectors, not over an unpinned research script." This crate is that first
native proof — once it is implemented.

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
│   ├── sparse_packet_ir/                 (all scaffold-only)
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

## How to use the scaffold today

```
$ cargo test -p tac-packet-compiler
```

Should pass green: the parity tests assert that every stub returns
`NotImplemented`. The coverage gate
(`every_golden_vector_has_paired_parity_test`) confirms every committed
vector has a matching Rust-side test entry. If you add a new vector on the
Python side, this test will fail until you wire a paired Rust parity test.

## How to use the scaffold after impl

Once a Rust function lands, flip the corresponding test in
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
"compiler and insanely low level" + "keep building outside the \$5 window".
Remaining order:

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
| 9 | `adaptive_brotli_param_search` | scaffold | Contract test (Pareto-frontier membership) — no byte-level golden vector. |
| 10 | `decode_ranked_no_op_sidecar` | scaffold | Inverse of impl #4; needs decode-side length-rank inversion + huff bit-unpacker. |
| 11-19 | PR81/PR92/PR97 / sparse PacketIR codecs / PR93 lowpass-luma / PR91 QMQH / PR93 QZMB1 magic grammars / magic-codec | scaffold | Each has a paired Python golden vector + paired Rust test stub already wired. |

| Question | Resolution |
|---|---|
| Who implements? | Subagent dispatch under operator directive 2026-05-11 ("compiler and insanely low level" + "keep building outside the \$5 window" + "recursively adversarially review and greenup"); 8 done. |
| Effort delivered | 8 impls + 1 regen helper + 13 binary fixtures + parity-test wire-in — ~1700 LOC of Rust + ~200 LOC of Python + 70 tests passing (49 unit + 21 integration). |
| Cost | \$0 GPU. |
| Risk audit | Brotli pure-Rust ↔ Python C library byte-parity was the largest unknown going in; verified GREEN on the committed 3-stream fixture. constriction (same upstream as Python package) and liblzma (same C library) were lower-risk and also GREEN. The bounded-length package-merge in PR101 ranked sidecar was the largest algorithmic risk — also GREEN on first attempt (matched the Python's iteration count + truncation + Kraft fallback exactly). |

See [`.omx/research/staged_rust_packet_compiler_native_port_readiness_*.md`](
../../../.omx/research/) for the operator-decision packet.

## CLAUDE.md compliance summary

| Non-negotiable | How this crate complies |
|---|---|
| **Deterministic packet compiler** | Scaffold + golden vectors + parity test harness. Byte-for-byte SHA-256 gate via [`conformance::assert_sha256_parity`](src/conformance/mod.rs). |
| **Beauty, simplicity, DX** | Narrow public API; typed errors; docstrings cite Python oracle line-by-line; `#![forbid(unsafe_code)]`. |
| **Native acceleration as conformance-backed PacketIR port** | Crate is named, scaffolded, and wired to the committed golden vectors before any impl is written. |
| **Rust/Zig is a speed layer** | `lib.rs` doc, README, and every stub doc reiterate that Python is the oracle. |
| **No MPS authoritative / no scorer load** | No torch / scorer / inflate-side imports anywhere. |
| **No `/tmp` paths** | Golden-vector resolution uses `CARGO_MANIFEST_DIR`, not env. |
| **Production-hardened OSS direction** | MIT/Apache-2.0 dual license; narrow public surface; `repository` + `description` populated; `publish = false` until contract stabilises. |

## Cross-references

- Python oracle: [`src/tac/packet_compiler/README.md`](../../../src/tac/packet_compiler/README.md).
- Golden vectors: [`src/tac/packet_compiler/golden_vectors/`](../../../src/tac/packet_compiler/golden_vectors/).
- Handoff: `~/Downloads/pact_score_lowering_handoff_2026-05-11.md` (Bottom-line item #7, Native-first candidates 1-2, P4 deterministic packet compiler).
- Operator directive: 2026-05-11, "production hardened OSS direction" + "all of the stuff that would cost more than $100 ready to go in parallel for as soon as we secure funding".
