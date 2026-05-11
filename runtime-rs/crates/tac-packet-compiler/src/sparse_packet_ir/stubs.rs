//! Stub function signatures for the sparse PacketIR codec primitives.
//!
//! Every function is `unimplemented!()` and returns
//! [`PacketCompilerError::NotImplemented`](crate::PacketCompilerError::NotImplemented).
//! Signatures mirror the Python oracle one-to-one.

use crate::{PacketCompilerError, Result};

// ── Public dataclasses (mirror Python @dataclass(frozen=True)) ──────────────

/// Sparse representation of a dense array via non-zero (index, value) pairs.
///
/// Mirrors `tac.packet_compiler.RleOfZerosStream`. The Python oracle's wire
/// format is `magic(4)=b"SRL1" || total_length(LE u32) || n_nonzero(LE u32)
/// || dtype_code(u8) || indices(n*LE u32) || values(n*signed itemsize)`.
#[derive(Debug, Clone)]
pub struct RleOfZerosStream {
    /// `uint32` indices into the dense layout (C-order flat).
    pub nonzero_indices: Vec<u32>,
    /// Non-zero values as little-endian signed bytes (dtype-tagged).
    pub nonzero_values: Vec<u8>,
    /// Width of one value in bytes (1 = int8, 2 = int16, 4 = int32).
    pub nonzero_value_itemsize: u8,
    /// Length of the dense array the stream represents.
    pub total_length: u32,
}

/// AC-coded coefficient stream with explicit per-stream histogram.
///
/// Mirrors `tac.packet_compiler.ArithmeticCodedCoefficientStream`. The
/// constriction range-coded payload uses **big-endian** uint32 words (matches
/// PR103 wire format).
#[derive(Debug, Clone)]
pub struct ArithmeticCodedCoefficientStream {
    /// Range-coded uint32 stream (constriction big-endian bytes).
    pub encoded_bytes: Vec<u8>,
    /// Per-symbol categorical distribution, length `alphabet_size`.
    pub histogram: Vec<f32>,
    /// Number of symbols encoded.
    pub n_symbols: u32,
    /// Symbol alphabet cardinality.
    pub alphabet_size: u32,
    /// Integer offset added when decoding to recover signed values.
    pub symbol_offset: i32,
}

/// K-of-N temporal subsampling stream.
///
/// Mirrors `tac.packet_compiler.TemporalSubsampledResidualStream`. The
/// indicator bitmap is LSB-first within each byte; the residuals are
/// concatenated raw bytes of the K signal-carrying frames in original order.
///
/// The `N` and `K` fields are intentionally uppercase to mirror the Python
/// oracle's wire-format documentation (K-of-N indicator); the
/// `#[allow(non_snake_case)]` opt-out scopes that exemption to this struct.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct TemporalSubsampledResidualStream {
    /// Packed `ceil(N/8)` bytes; bit `i` indicates whether frame `i` carries
    /// signal.
    pub indicator_bitmap: Vec<u8>,
    /// Concatenated bytes of the K signal-carrying frames.
    pub residuals_packed: Vec<u8>,
    /// Total frame count.
    pub N: u32,
    /// Number of signal-carrying frames (must equal popcount of bitmap).
    pub K: u32,
    /// Byte count of each signal-carrying frame's residual.
    pub per_frame_bytes: u32,
}

// ── Stubbed encode/decode contracts: RLE-of-zeros ───────────────────────────

/// Build an [`RleOfZerosStream`] from a dense int8 array.
///
/// **SCAFFOLD-ONLY** — returns
/// `Err(PacketCompilerError::NotImplemented("encode_rle_of_zeros"))`.
///
/// Target SHA-256: see
/// `src/tac/packet_compiler/golden_vectors/sparse_rle_of_zeros_v1.json`.
pub fn encode_rle_of_zeros(_dense: &[i8]) -> Result<RleOfZerosStream> {
    Err(PacketCompilerError::NotImplemented("encode_rle_of_zeros"))
}

/// Inverse of [`encode_rle_of_zeros`].
///
/// **SCAFFOLD-ONLY**. Returns the dense int8 representation. Wider integer
/// dtypes will require a generic adapter once the impl lands.
pub fn decode_rle_of_zeros(_stream: &RleOfZerosStream) -> Result<Vec<i8>> {
    Err(PacketCompilerError::NotImplemented("decode_rle_of_zeros"))
}

/// Serialise an [`RleOfZerosStream`] to self-delimiting bytes.
///
/// **SCAFFOLD-ONLY**. Mirrors `tac.packet_compiler.serialize_rle_of_zeros`.
pub fn serialize_rle_of_zeros(_stream: &RleOfZerosStream) -> Result<Vec<u8>> {
    Err(PacketCompilerError::NotImplemented("serialize_rle_of_zeros"))
}

/// Inverse of [`serialize_rle_of_zeros`].
///
/// **SCAFFOLD-ONLY**.
pub fn deserialize_rle_of_zeros(_blob: &[u8]) -> Result<RleOfZerosStream> {
    Err(PacketCompilerError::NotImplemented(
        "deserialize_rle_of_zeros",
    ))
}

// ── Stubbed encode/decode contracts: AC coefficient stream ──────────────────

/// Range-code a 1D integer array with a per-stream histogram.
///
/// **SCAFFOLD-ONLY**. Returns
/// `Err(PacketCompilerError::NotImplemented("encode_arithmetic_coefficients"))`.
///
/// Target SHA-256: see
/// `src/tac/packet_compiler/golden_vectors/sparse_arithmetic_coefficients_v1.json`.
pub fn encode_arithmetic_coefficients(
    _values: &[i32],
    _histogram: Option<&[f32]>,
    _symbol_offset: Option<i32>,
    _alphabet_size: Option<u32>,
) -> Result<ArithmeticCodedCoefficientStream> {
    Err(PacketCompilerError::NotImplemented(
        "encode_arithmetic_coefficients",
    ))
}

/// Inverse of [`encode_arithmetic_coefficients`].
///
/// **SCAFFOLD-ONLY**.
pub fn decode_arithmetic_coefficients(
    _stream: &ArithmeticCodedCoefficientStream,
) -> Result<Vec<i32>> {
    Err(PacketCompilerError::NotImplemented(
        "decode_arithmetic_coefficients",
    ))
}

/// Serialise an [`ArithmeticCodedCoefficientStream`] to self-delimiting bytes.
///
/// **SCAFFOLD-ONLY**.
pub fn serialize_arithmetic_coefficients(
    _stream: &ArithmeticCodedCoefficientStream,
) -> Result<Vec<u8>> {
    Err(PacketCompilerError::NotImplemented(
        "serialize_arithmetic_coefficients",
    ))
}

/// Inverse of [`serialize_arithmetic_coefficients`].
///
/// **SCAFFOLD-ONLY**.
pub fn deserialize_arithmetic_coefficients(
    _blob: &[u8],
) -> Result<ArithmeticCodedCoefficientStream> {
    Err(PacketCompilerError::NotImplemented(
        "deserialize_arithmetic_coefficients",
    ))
}

// ── Stubbed encode/decode contracts: temporal-subsampled stream ─────────────

/// Build a [`TemporalSubsampledResidualStream`] from per-frame residuals.
///
/// **SCAFFOLD-ONLY**. The Python oracle accepts a list of optional uniform-
/// size numpy arrays. The Rust API uses a parallel `(indicator, frame_bytes)`
/// signature so the `Option<&[u8]>` element shape is explicit.
pub fn encode_temporal_subsampled(
    _per_frame_residuals: &[Option<&[u8]>],
) -> Result<TemporalSubsampledResidualStream> {
    Err(PacketCompilerError::NotImplemented(
        "encode_temporal_subsampled",
    ))
}

/// Inverse of [`encode_temporal_subsampled`].
///
/// **SCAFFOLD-ONLY**. Returns a list of length `N` with `None` for skipped
/// frames and `Some(bytes)` for signal-carrying frames.
pub fn decode_temporal_subsampled(
    _stream: &TemporalSubsampledResidualStream,
) -> Result<Vec<Option<Vec<u8>>>> {
    Err(PacketCompilerError::NotImplemented(
        "decode_temporal_subsampled",
    ))
}

/// Serialise a [`TemporalSubsampledResidualStream`] to self-delimiting bytes.
///
/// **SCAFFOLD-ONLY**.
///
/// Target SHA-256: see
/// `src/tac/packet_compiler/golden_vectors/sparse_temporal_subsampled_v1.json`.
pub fn serialize_temporal_subsampled(
    _stream: &TemporalSubsampledResidualStream,
) -> Result<Vec<u8>> {
    Err(PacketCompilerError::NotImplemented(
        "serialize_temporal_subsampled",
    ))
}

/// Inverse of [`serialize_temporal_subsampled`].
///
/// **SCAFFOLD-ONLY**.
pub fn deserialize_temporal_subsampled(_blob: &[u8]) -> Result<TemporalSubsampledResidualStream> {
    Err(PacketCompilerError::NotImplemented(
        "deserialize_temporal_subsampled",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All scaffold stubs report `NotImplemented` so they cannot silently
    /// pretend to be authoritative.
    #[test]
    fn sparse_packet_ir_scaffold_refuses_with_not_implemented_error() {
        let err = encode_rle_of_zeros(&[]).expect_err("scaffold must refuse");
        assert!(matches!(
            err,
            PacketCompilerError::NotImplemented("encode_rle_of_zeros")
        ));
    }
}
