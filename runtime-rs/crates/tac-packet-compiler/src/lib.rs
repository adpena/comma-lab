//! `tac-packet-compiler` — Rust native port for `tac.packet_compiler`.
//!
//! # Status — MIXED NATIVE IMPLEMENTATION
//!
//! This crate is the speed layer for the Python packet-compiler oracle. Many
//! hot-path primitives are implemented and must pass byte-for-byte SHA parity
//! against committed Python golden vectors. A small number of lower-priority
//! surfaces remain scaffold-only and return
//! [`PacketCompilerError::NotImplemented`].
//!
//! The crate exists to:
//!
//! 1. Document the **byte-for-byte parity contract** against the Python oracle.
//! 2. Run the **golden-vector parity test harness** for implemented native
//!    functions and explicit load-only coverage for remaining scaffolds.
//! 3. Pin the **dependency choices** (constriction, brotli, liblzma, ndarray)
//!    so an implementer does not relitigate them.
//!
//! # Oracle vs speed layer
//!
//! Per CLAUDE.md "Rust/Zig is a speed layer, not a license to change semantics":
//!
//! - **Oracle:** `src/tac/packet_compiler/` (Python, in this repository).
//! - **Speed layer:** this crate.
//!
//! The promotion path for any function in this crate is:
//!
//! 1. The Python implementation lands and is covered by focused conformance
//!    tests.
//! 2. A golden vector is regenerated and committed under
//!    `src/tac/packet_compiler/golden_vectors/<name>_v<N>.json`.
//! 3. The Rust function is implemented to match byte-for-byte.
//! 4. `cargo test -p tac-packet-compiler` is green on the golden-vector parity
//!    test (`tests/golden_vector_parity.rs`).
//! 5. A deterministic rebuild on a different host reproduces identical
//!    binaries.
//!
//! # Contract reproduced from Python oracle
//!
//! Every primitive must satisfy (lifted verbatim from
//! `src/tac/packet_compiler/README.md`):
//!
//! - **No scorer load** — no PoseNet / SegNet / FastViT / EfficientNet types.
//! - **No torch / MPS dependency** — pure Rust + native deps (brotli,
//!   liblzma, constriction).
//! - **No `/tmp` paths** — golden vectors live under
//!   `src/tac/packet_compiler/golden_vectors/` and are bytes-identical across
//!   Python and Rust.
//! - **Deterministic + byte-stable** — every `encode → decode` is a bit-exact
//!   identity. Native ports must match the SHA-256 of the golden-vector
//!   payload before promotion.
//! - **OSS-friendly** — public surface is narrow; everything else is private.
//!
//! # Module layout
//!
//! - [`pr101_sidecar_grammar`] — ranked Huffman/no-op sidecar +
//!   centered-delta uint8 + self-delimiting split Brotli.
//! - [`pr103_arithmetic_coding`] — merged range stream over multiple weight
//!   tensors + latent-hi arithmetic + adaptive Brotli param search.
//! - [`pr93_pose_codec`] — delta-varint pose codec (QZPDV1) + QZMB1 magic
//!   grammar (2026-05-11).
//! - [`pr91_hpac_grammar`] — universal per-symbol constriction AC wrapper
//!   + QM0/QH0 magic grammar (2026-05-11).
//! - [`pr84_adaptive_mask`] — per-context adaptive-context range coder
//!   (2026-05-11).
//! - [`pr81_quantizr`] — asymmetric 8-level FP4 codebook (+ sign bit) +
//!   ROUTER_ACTION small-integer LSB-first bit-packer (2026-05-11).
//! - [`pr92_joint_stream`] — RMC1 / RSA1 / RSB1 joint-stream meta-codec
//!   for correlated mask + action byte streams (2026-05-11).
//! - [`sparse_packet_ir`] — RLE-of-zeros + arithmetic-coded coefficient
//!   stream + temporal-subsampling indicator vector. Closes O's L2
//!   wire-format ceiling (2026-05-11).
//! - [`simd`] — NEON (aarch64) + AVX2 (x86_64) hot-path kernels for
//!   pre-encoding transforms (hi-byte extraction, RLE-of-zeros nonzero
//!   scan, centered-delta column-major emit) with portable Rust fallback
//!   and byte-for-byte parity proptests (2026-05-11).
//! - [`custom_binary_container`] — RESEARCH-ONLY non-ZIP container format
//!   exploration; archive bytes do NOT enter the contest packet (2026-05-11).
//! - [`conformance`] — golden-vector loader + byte-for-byte parity helpers.
//!
//! # Source references
//!
//! - Handoff `~/Downloads/pact_score_lowering_handoff_2026-05-11.md`
//!   "Bottom-line next tranche" item #7 + "Native-first candidates" 1-2.
//! - Python oracle: `src/tac/packet_compiler/pr101_sidecar_grammar.py`,
//!   `src/tac/packet_compiler/pr103_arithmetic_coding.py`.
//! - Compliance: CLAUDE.md "Deterministic packet compiler" non-negotiable.

#![warn(missing_docs)]
// `unsafe_code` is FORBIDDEN at the crate's public surface. Architecture-
// specific SIMD intrinsics (NEON / AVX2) require `unsafe` to call the
// `std::arch::*` family; those usages live exclusively in the [`simd`]
// submodule and carry per-function `#[allow(unsafe_code)]` + a `# Safety`
// doc comment. Every other module remains `unsafe_code`-free.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unsafe_code)]

pub mod conformance;
pub mod custom_binary_container;
pub mod pr101_sidecar_grammar;
pub mod pr103_arithmetic_coding;
pub mod pr81_quantizr;
pub mod pr84_adaptive_mask;
pub mod pr91_hpac_grammar;
pub mod pr92_joint_stream;
pub mod pr93_pose_codec;
pub mod pr97_h3_grammar;
pub mod simd;
pub mod sparse_packet_ir;

/// Crate-level error type.
#[derive(Debug)]
pub enum PacketCompilerError {
    /// Function is still scaffold-only; no Rust implementation has landed yet.
    NotImplemented(&'static str),
    /// Generic parity-vector load failure (corrupt JSON / missing file).
    GoldenVectorIo(String),
    /// SHA-256 mismatch against a committed golden vector.
    SidecarShaMismatch {
        /// Vector schema label (e.g. `"ranked_no_op_sidecar.v1"`).
        schema: String,
        /// SHA-256 hex digest the encoder produced.
        produced: String,
        /// SHA-256 hex digest pinned by the committed golden vector.
        expected: String,
    },
}

impl std::fmt::Display for PacketCompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented(name) => {
                write!(f, "tac-packet-compiler: {name} is scaffold-only (no Rust impl yet); the Python oracle is the canonical reference")
            }
            Self::GoldenVectorIo(msg) => write!(f, "golden-vector i/o error: {msg}"),
            Self::SidecarShaMismatch {
                schema,
                produced,
                expected,
            } => write!(
                f,
                "byte-for-byte parity mismatch on {schema}: produced sha256={produced}; expected sha256={expected}"
            ),
        }
    }
}

impl std::error::Error for PacketCompilerError {}

/// Convenience alias for the crate's `Result`.
pub type Result<T> = std::result::Result<T, PacketCompilerError>;
