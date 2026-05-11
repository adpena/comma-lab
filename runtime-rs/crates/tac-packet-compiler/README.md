# `tac-packet-compiler`

Rust native port **scaffold** for [`tac.packet_compiler`](../../../src/tac/packet_compiler/),
the reusable byte-grammar and entropy-coder primitives extracted from the
public PR101 (`hnerv_ft_microcodec`) and PR103 (`hnerv_lc_ac`) submissions.

> **Status — SCAFFOLD ONLY (no Rust implementation yet).** Every public
> function in this crate is `unimplemented!()` and returns
> `PacketCompilerError::NotImplemented`. The crate exists so that:
>
> 1. The byte-for-byte parity contract is documented and reviewable.
> 2. The golden-vector parity test harness is wired and runnable.
> 3. The dependency choices (`constriction`, `brotli`, `liblzma`, `ndarray`)
>    are pinned before an implementer starts.
>
> See "Roadmap" below for the implementation gate.

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
│   │   └── stubs.rs                      unimplemented!() stubs
│   ├── pr103_arithmetic_coding/
│   │   ├── mod.rs                        re-exports
│   │   └── stubs.rs                      unimplemented!() stubs
│   └── conformance/
│       └── mod.rs                        golden-vector loader + sha256 helpers
├── tests/
│   └── golden_vector_parity.rs           parity gate against committed vectors
└── benches/
    └── golden_vector_parity.rs           criterion scaffold (no-op sentinel)
```

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

The implementation work itself is **deferred pending operator approval** on
ownership + effort. Surfaced for decision:

| Question | Options |
|---|---|
| Who implements? | (a) claude-direct in a follow-up subagent; (b) dispatch to a Rust-specialist subagent; (c) defer pending external contributor / contest-window timing. |
| Effort estimate | ~900–1400 LOC of Rust + ~400 LOC of tests; 1–2 wall-clock days of focused work. Constriction-sensitive parity is the highest-risk axis (5/5 golden vectors require it). |
| Cost estimate | $0 GPU (the parity tests are pure CPU). |
| Order of attack | 1) `encode_centered_delta_uint8` (no constriction, exercise liblzma binding) → 2) `split_brotli_self_delimiting` (exercise brotli binding) → 3) `encode_latent_hi_arithmetic` (single-tensor constriction smoke) → 4) `encode_merged_range_stream` (multi-tensor constriction) → 5) `encode_ranked_no_op_sidecar` (most algorithmic — package-merge + co-lex combination rank). |

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
