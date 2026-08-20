// SPDX-License-Identifier: MIT OR Apache-2.0
//! `tac-boundary-decode` — native DECODE-side port of the settled score-native
//! carrier primitives in `src/tac/boundary_math/`.
//!
//! # Discipline (CLAUDE.md "Native eval-time runtime discipline")
//!
//! - The Python oracle in `src/tac/boundary_math/` is canonical.
//! - Each primitive has a bit-identical equivalence test against the oracle on
//!   the same archive bytes (`tests/golden_vector_parity.rs` →
//!   [`conformance::assert_sha256_parity`]). The decoded raw-output SHA-256 must
//!   match Python's byte-for-byte.
//! - NO learned / video-derived constant is embedded in this binary. The decoder
//!   is FIXED, rate-free code; the weights/contours/labels live in `archive.zip`.
//!   Embedding the answer in the binary is a payload-cleanliness violation.
//!
//! # Primitives
//!
//! | fn | Python oracle | parity gate |
//! |----|---------------|-------------|
//! | [`contour::decode_partition_raw`] | `dense_raster_lzma_baseline.decode_partition` | `contour_decode_{full,small}_v1` |
//! | [`dseg::flip_count`] / [`dseg::d_seg`] | `bitmask_dseg.flip_count` / `d_seg_reference` | `dseg_popcount_v1` |
//! | [`components::connected_components`] | `partition.connected_components` | `connected_components_v1` |

pub mod components;
pub mod conformance;
pub mod contour;
pub mod dseg;

use std::fmt;

/// Crate-wide error type. Mirrors the fail-loud convention of the sibling
/// `tac-packet-compiler` crate (no silent-skip; every failure carries a
/// human-readable diagnostic).
#[derive(Debug, Clone)]
pub enum BoundaryDecodeError {
    /// RAW-LZMA2 decode failure (corrupt payload, wrong filter spec, etc.).
    LzmaDecode(String),
    /// Decoded byte length did not match the declared `(H, W)` shape.
    ShapeMismatch { expected: usize, got: usize },
    /// Two label arrays compared for d_seg had mismatched lengths.
    LengthMismatch { a: usize, b: usize },
    /// A class id fell outside `[0, n_classes)`.
    ClassOutOfRange { value: i64, n_classes: usize },
    /// Golden-vector manifest i/o or JSON parse failure.
    GoldenVectorIo(String),
    /// Byte-for-byte parity mismatch vs the committed Python-oracle digest.
    ShaMismatch {
        schema: String,
        produced: String,
        expected: String,
    },
}

impl fmt::Display for BoundaryDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundaryDecodeError::LzmaDecode(m) => write!(f, "RAW-LZMA2 decode failed: {m}"),
            BoundaryDecodeError::ShapeMismatch { expected, got } => {
                write!(f, "decoded len {got} != H*W {expected}")
            }
            BoundaryDecodeError::LengthMismatch { a, b } => {
                write!(f, "label array length mismatch: {a} vs {b}")
            }
            BoundaryDecodeError::ClassOutOfRange { value, n_classes } => {
                write!(f, "class id {value} out of range [0,{n_classes})")
            }
            BoundaryDecodeError::GoldenVectorIo(m) => write!(f, "golden vector io: {m}"),
            BoundaryDecodeError::ShaMismatch {
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

impl std::error::Error for BoundaryDecodeError {}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, BoundaryDecodeError>;
