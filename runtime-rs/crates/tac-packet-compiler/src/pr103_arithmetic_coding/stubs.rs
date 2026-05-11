//! Stub function signatures for the PR103 arithmetic-coding primitives.
//!
//! Every function is `unimplemented!()` and returns
//! [`PacketCompilerError::NotImplemented`](crate::PacketCompilerError::NotImplemented).
//! Signatures mirror the Python oracle one-to-one.

use crate::{PacketCompilerError, Result};

// ── Public dataclasses (mirror Python @dataclass(frozen=True)) ──────────────

/// Per-tensor specification for the merged range stream.
///
/// Mirrors `tac.packet_compiler.WeightTensorACSpec`. The histogram is shipped
/// out-of-band (PR103 packs the eight 256-byte histograms in a separate
/// Brotli stream).
#[derive(Debug, Clone)]
pub struct WeightTensorACSpec {
    /// Short identifier for diagnostics (never encoded into the bytes).
    pub name: String,
    /// Expected tensor shape; the encoder flattens to a 1D symbol stream;
    /// the decoder reshapes back.
    pub shape: Vec<usize>,
    /// Categorical distribution of length `alphabet_size` (typically 256).
    /// Float-valued; the coder normalises internally.
    pub histogram: Vec<f64>,
    /// Number of symbols. Default 256 (PR103 int8 offset +128 convention).
    pub alphabet_size: u32,
}

/// Range-coded byte payload over a sequence of weight tensors.
///
/// Mirrors `tac.packet_compiler.MergedRangeStream`.
#[derive(Debug, Clone)]
pub struct MergedRangeStream {
    /// Range-coded bytes (constriction's uint32 stream serialised as
    /// **big-endian** bytes).
    pub payload: Vec<u8>,
    /// Per-tensor symbol counts in encode order. Required by the decoder
    /// because the range-coded stream itself does not carry boundaries.
    pub tensor_symbol_counts: Vec<usize>,
    /// Length of the underlying uint32 array. `payload.len() == word_count*4`.
    pub word_count: usize,
}

/// Best-found Brotli parameter set + payload.
///
/// Mirrors `tac.packet_compiler.AdaptiveBrotliResult`.
#[derive(Debug, Clone)]
pub struct AdaptiveBrotliResult {
    /// Brotli-compressed bytes corresponding to `(lgwin, quality)`.
    pub payload: Vec<u8>,
    /// Chosen `lgwin` (10..24).
    pub lgwin: u8,
    /// Chosen `quality` (0..11).
    pub quality: u8,
    /// Number of `(lgwin, quality)` tuples actually evaluated.
    pub tested_count: u32,
    /// Wall-clock spent in the search (seconds).
    pub elapsed_seconds: f64,
    /// Tested `(lgwin, quality, output_size)` triples, in evaluation order.
    pub explored: Vec<(u8, u8, usize)>,
}

// ── Stubbed encode/decode contracts ─────────────────────────────────────────

/// Encode multiple weight tensors into a single range-coded byte string.
///
/// **SCAFFOLD-ONLY** — returns
/// `Err(PacketCompilerError::NotImplemented("encode_merged_range_stream"))`.
///
/// Mirrors `tac.packet_compiler.encode_merged_range_stream`. Tensors are
/// encoded in order; each symbol is coded against its corresponding
/// [`WeightTensorACSpec::histogram`].
///
/// `tensors_flat_int32` is the concatenated flat int32 symbol stream for all
/// tensors in `specs` order; `specs` carries the per-tensor shape so the
/// implementer can slice on encode. The Python oracle's wire format uses
/// `constriction::stream::queue::RangeEncoder` + big-endian uint32
/// serialisation; the Rust impl MUST match byte-for-byte.
///
/// Target SHA-256: see
/// `src/tac/packet_compiler/golden_vectors/merged_range_stream_v1.json`.
pub fn encode_merged_range_stream(
    _tensors_flat_int32: &[i32],
    _specs: &[WeightTensorACSpec],
) -> Result<MergedRangeStream> {
    Err(PacketCompilerError::NotImplemented(
        "encode_merged_range_stream",
    ))
}

/// Inverse of [`encode_merged_range_stream`].
///
/// **SCAFFOLD-ONLY**. Returns a flat int32 buffer; the caller reshapes per
/// `specs[i].shape`.
pub fn decode_merged_range_stream(
    _stream: &MergedRangeStream,
    _specs: &[WeightTensorACSpec],
) -> Result<Vec<i32>> {
    Err(PacketCompilerError::NotImplemented(
        "decode_merged_range_stream",
    ))
}

/// Encode the high byte of a uint16 zigzag-delta latent stream.
///
/// **SCAFFOLD-ONLY**.
///
/// Input is a flat `&[u16]`; the high byte is `(x >> 8) & 0xFF`. Output is
/// constriction's uint32 stream serialised as big-endian bytes.
///
/// Target SHA-256: see
/// `src/tac/packet_compiler/golden_vectors/latent_hi_arithmetic_v1.json`.
pub fn encode_latent_hi_arithmetic(_latents: &[u16], _histogram: &[f64]) -> Result<Vec<u8>> {
    Err(PacketCompilerError::NotImplemented(
        "encode_latent_hi_arithmetic",
    ))
}

/// Decode a high-byte stream produced by [`encode_latent_hi_arithmetic`].
///
/// **SCAFFOLD-ONLY**.
pub fn decode_latent_hi_arithmetic(
    _payload: &[u8],
    _histogram: &[f64],
    _n_symbols: usize,
) -> Result<Vec<u8>> {
    Err(PacketCompilerError::NotImplemented(
        "decode_latent_hi_arithmetic",
    ))
}

/// Sweep Brotli `(lgwin, quality)` and return the smallest output.
///
/// **SCAFFOLD-ONLY**.
///
/// Per-host CPU timing makes byte-level golden-vector validation
/// inappropriate here; the implementer instead pins a contract test:
/// "given a fixed `payload`, the result's `(lgwin, quality)` must be in the
/// Pareto-frontier set listed by the Python oracle on the same input".
///
/// `lgwin_range` / `quality_range` are inclusive. `max_evaluations = 0`
/// means "no hard cap; honour `time_budget_seconds`".
pub fn adaptive_brotli_param_search(
    _payload: &[u8],
    _time_budget_seconds: f64,
    _lgwin_range: (u8, u8),
    _quality_range: (u8, u8),
    _max_evaluations: u32,
) -> Result<AdaptiveBrotliResult> {
    Err(PacketCompilerError::NotImplemented(
        "adaptive_brotli_param_search",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All three stubs report `NotImplemented` so they cannot silently
    /// pretend to be authoritative.
    #[test]
    fn pr103_scaffold_refuses_with_not_implemented_error() {
        let err = encode_latent_hi_arithmetic(&[], &[]).expect_err("scaffold must refuse");
        assert!(matches!(
            err,
            PacketCompilerError::NotImplemented("encode_latent_hi_arithmetic")
        ));
    }
}
