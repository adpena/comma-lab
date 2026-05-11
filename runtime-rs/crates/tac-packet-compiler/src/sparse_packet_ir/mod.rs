//! Sparse PacketIR codec — Rust port scaffold.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.sparse_packet_ir`](
//! file:../../../../../src/tac/packet_compiler/sparse_packet_ir.py).
//!
//! Three orthogonal-composable sparse residual primitives are exposed (none
//! implemented yet):
//!
//! 1. **RLE-of-zeros** — sparse representation via `(index, value)` pairs;
//!    best for natural-image residuals where most quantised coefficients
//!    are zero.
//! 2. **Arithmetic-coded coefficient stream** — biased coefficient
//!    distributions (Laplacian, peaked-at-zero) compressed via constriction
//!    range coding with explicit per-stream histogram.
//! 3. **Temporal-subsampling indicator vector** — K-of-N indicator bitmap +
//!    densely-packed K residuals; best for per-frame residuals where only a
//!    subset of frames carries signal.
//!
//! Each primitive has a paired golden vector under
//! `src/tac/packet_compiler/golden_vectors/`:
//!
//! - `sparse_rle_of_zeros_v1.json`
//! - `sparse_arithmetic_coefficients_v1.json`
//! - `sparse_temporal_subsampled_v1.json`
//!
//! # Wire-format note
//!
//! The Python oracle uses **little-endian** uint32 length prefixes (matches
//! the wider `tac.packet_compiler` framing convention used by PR106 sidecar
//! packing). The arithmetic-coded payload itself uses constriction's
//! **big-endian** uint32 word serialisation (matches PR103). The Rust impl
//! MUST honour both endian conventions exactly.
//!
//! # Closes O's L2 wire-format ceiling
//!
//! Per memory `feedback_l2_score_aware_encoders_wavelet_c3_cool_chic_landed_20260511.md`,
//! the L2 score-aware encoders REFUSE to emit dense residual blobs because
//! the L1 inflate format charges zero-padded skipped frames. Sparse PacketIR
//! is the substrate-engineering lane that closes that ceiling.

pub mod stubs;

pub use stubs::{
    decode_arithmetic_coefficients, decode_rle_of_zeros, decode_temporal_subsampled,
    deserialize_arithmetic_coefficients, deserialize_rle_of_zeros, deserialize_temporal_subsampled,
    encode_arithmetic_coefficients, encode_rle_of_zeros, encode_temporal_subsampled,
    serialize_arithmetic_coefficients, serialize_rle_of_zeros, serialize_temporal_subsampled,
    ArithmeticCodedCoefficientStream, RleOfZerosStream, TemporalSubsampledResidualStream,
};
