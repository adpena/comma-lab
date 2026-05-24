//! Stable function signatures for the PR103 arithmetic-coding primitives.
//!
//! Some functions are scaffold-only and return
//! [`PacketCompilerError::NotImplemented`].
//! Implemented primitives delegate to concrete modules. Signatures mirror the
//! Python oracle one-to-one.

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
/// Mirrors `tac.packet_compiler.encode_merged_range_stream`. Tensors are
/// encoded in order; each symbol is coded against its corresponding
/// [`WeightTensorACSpec::histogram`].
///
/// `tensors_flat_int32` is the concatenated flat int32 symbol stream for all
/// tensors in `specs` order; `specs` carries the per-tensor shape so the
/// implementer can slice on encode. The Python oracle's wire format uses
/// `constriction::stream::queue::RangeEncoder` + big-endian uint32
/// serialisation; the Rust impl matches byte-for-byte.
///
/// Implementation lives in [`super::merged_range_stream::encode_merged_range_stream`].
/// Byte-for-byte parity target:
/// `src/tac/packet_compiler/golden_vectors/merged_range_stream_v1.json`.
pub fn encode_merged_range_stream(
    tensors_flat_int32: &[i32],
    specs: &[WeightTensorACSpec],
) -> Result<MergedRangeStream> {
    super::merged_range_stream::encode_merged_range_stream(tensors_flat_int32, specs)
}

/// Inverse of [`encode_merged_range_stream`]. Returns a flat int32 buffer;
/// the caller reshapes per `specs[i].shape`.
pub fn decode_merged_range_stream(
    stream: &MergedRangeStream,
    specs: &[WeightTensorACSpec],
) -> Result<Vec<i32>> {
    super::merged_range_stream::decode_merged_range_stream(stream, specs)
}

/// Encode the high byte of a uint16 zigzag-delta latent stream.
///
/// Implementation lives in [`super::latent_hi::encode_latent_hi_arithmetic`].
/// Output is constriction's uint32 stream serialised as **big-endian bytes**
/// to match the Python wire format.
pub fn encode_latent_hi_arithmetic(latents: &[u16], histogram: &[f64]) -> Result<Vec<u8>> {
    super::latent_hi::encode_latent_hi_arithmetic(latents, histogram)
}

/// Decode a high-byte stream produced by [`encode_latent_hi_arithmetic`].
///
/// Implementation lives in [`super::latent_hi::decode_latent_hi_arithmetic`].
pub fn decode_latent_hi_arithmetic(
    payload: &[u8],
    histogram: &[f64],
    n_symbols: usize,
) -> Result<Vec<u8>> {
    super::latent_hi::decode_latent_hi_arithmetic(payload, histogram, n_symbols)
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

    /// `encode_merged_range_stream` is implemented (see merged_range_stream.rs);
    /// callers cannot silently get a fake stream. The remaining scaffold
    /// surface is [`adaptive_brotli_param_search`]; we keep this regression
    /// pinned so a future refactor cannot accidentally re-stub the impl.
    #[test]
    fn pr103_merged_range_stream_impl_landed() {
        // Empty inputs surface a `GoldenVectorIo` error, NOT `NotImplemented`;
        // the impl is now wired through.
        let err = encode_merged_range_stream(&[], &[]).expect_err("must reject empty input");
        assert!(
            !matches!(err, PacketCompilerError::NotImplemented(_)),
            "encode_merged_range_stream must NOT return NotImplemented; got {err:?}"
        );
    }

    /// `adaptive_brotli_param_search` is still scaffold-only.
    #[test]
    fn pr103_adaptive_brotli_param_search_still_scaffold() {
        let err = adaptive_brotli_param_search(b"x", 1.0, (10, 24), (0, 11), 0)
            .expect_err("scaffold must refuse");
        assert!(matches!(
            err,
            PacketCompilerError::NotImplemented("adaptive_brotli_param_search")
        ));
    }
}
