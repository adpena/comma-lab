// SPDX-License-Identifier: MIT OR Apache-2.0
//! `tac-levelset-inflate` — native DECODE-side port of the level-set inflate
//! primitives whose Python oracle is bit-exact-PORTABLE.
//!
//! # Discipline (CLAUDE.md "Native eval-time runtime discipline")
//!
//! - The Python oracle (`tac.lossless.range_coder`, `tac.boundary_math.xi_pose_coder`)
//!   stays canonical. Each primitive has a bit-identical equivalence test against the
//!   oracle on the SAME bytes (`tests/golden_vector_parity.rs` →
//!   [`conformance::assert_sha256_parity`]). A single-bit divergence fails the SHA gate.
//! - NO learned / video-derived constant is embedded in this binary. The decoder is
//!   FIXED, rate-free code; the ξ payload lives in `archive.zip`.
//!
//! # Why only these two primitives (the measured feasibility boundary)
//!
//! The level-set inflate's wall-clock hot path is the coord-INR neural forward
//! (matmuls + hosc `tanh(β·sin)` `_act`). That forward is **NOT bit-exact-portable**
//! against the numpy-fp64 oracle:
//!
//! - numpy fp64 `@` dispatches to Apple Accelerate `dgemm` (blocked + FMA reduction
//!   order). MEASURED: a naive fma-off sequential Rust-equivalent diverges on
//!   4497/6144 elements at ~1e-14. A portable Rust matmul cannot match it bit-for-bit.
//! - numpy fp64 `tanh` differs from the scalar system libm `tanh`. MEASURED:
//!   844/4096 elements at 1 ULP. So even a Rust `f64::tanh` (system libm) misses it.
//!   (numpy `sin`/`cos`/`exp` DO match the scalar libm bit-for-bit — those are not
//!   the blocker.)
//!
//! The ONE genuinely bit-exact-by-CONSTRUCTION piece is the #257 store-nothing
//! POSE-CARRIER ξ decode: a pure-INTEGER arithmetic range-decoder + integer column
//! reconstruction. Pure integer ⇒ no FMA / rounding / reduction-order / transcendental
//! dependence ⇒ byte-for-byte parity is provable by construction. That is what lives here.
//!
//! | fn | Python oracle | parity gate |
//! |----|---------------|-------------|
//! | [`range_decode::decode_static_symbols`] | `range_coder.decode_static_symbols` | `levelset_xi_range_decode_v1` |
//! | [`xi_column::decode_xi_column`] | `xi_pose_coder._channel_delta_decode` | `levelset_xi_column_delta_v1` |

pub mod conformance;
pub mod range_decode;
pub mod xi_column;

use std::fmt;

/// Crate-wide error type. Mirrors the fail-loud convention of the sibling
/// `tac-boundary-decode` / `tac-packet-compiler` crates (no silent-skip; every
/// failure carries a human-readable diagnostic).
#[derive(Debug, Clone)]
pub enum LevelSetInflateError {
    /// A frequency table entry was non-positive (the coder requires every
    /// symbol frequency ≥ 1) or the table summed to a non-positive total.
    InvalidFrequencies(String),
    /// The range-coded stream is inconsistent with the frequency table
    /// (decoded target fell outside `[0, total)`), mirroring the Python
    /// oracle's `ValueError`.
    InconsistentStream(String),
    /// Golden-vector manifest i/o or JSON parse failure.
    GoldenVectorIo(String),
    /// Byte-for-byte parity mismatch vs the committed Python-oracle digest.
    ShaMismatch {
        schema: String,
        produced: String,
        expected: String,
    },
}

impl fmt::Display for LevelSetInflateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LevelSetInflateError::InvalidFrequencies(m) => write!(f, "invalid frequency table: {m}"),
            LevelSetInflateError::InconsistentStream(m) => {
                write!(f, "encoded range stream inconsistent with frequency table: {m}")
            }
            LevelSetInflateError::GoldenVectorIo(m) => write!(f, "golden vector io: {m}"),
            LevelSetInflateError::ShaMismatch {
                schema,
                produced,
                expected,
            } => write!(
                f,
                "SHA-256 parity mismatch for {schema}: produced {produced} != expected {expected}"
            ),
        }
    }
}

impl std::error::Error for LevelSetInflateError {}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, LevelSetInflateError>;
