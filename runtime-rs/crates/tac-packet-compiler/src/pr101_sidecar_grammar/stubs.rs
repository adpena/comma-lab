//! Stub function signatures for the PR101 sidecar grammar primitives.
//!
//! Every function is `unimplemented!()` and returns
//! [`PacketCompilerError::NotImplemented`](crate::PacketCompilerError::NotImplemented)
//! when called via the explicit-Result variant. The signatures mirror the
//! Python oracle one-to-one so an implementer can build inside-out without
//! relitigating the API surface.

use crate::{PacketCompilerError, Result};

// ── Public dataclasses (mirror Python @dataclass(frozen=True)) ──────────────

/// Schema for a ranked Huffman/no-op sidecar over per-pair corrections.
///
/// Mirrors `tac.packet_compiler.RankedSidecarSchema`. All fields are
/// `pub(crate)` until the impl lands so we can add validators here without an
/// API break.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedSidecarSchema {
    /// Total number of per-pair slots (e.g. 600 for the contest's 600
    /// pair-aligned latent positions).
    pub n_pairs: usize,
    /// Number of dimensions a correction may target (e.g. PR101 uses 28).
    pub n_dims: usize,
    /// Allowed signed delta codes (PR101 uses
    /// `[-10,-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,8,10]` interpreted at 1/100
    /// scale). Must be strictly ascending and `len >= 2`.
    pub deltas: Vec<i32>,
    /// Lower bound on canonical Huffman code lengths. PR101 uses 2.
    pub huff_min_len: u8,
    /// Upper bound on canonical Huffman code lengths. PR101 uses 8.
    pub huff_max_len: u8,
    /// Value in the decoded `dims` array that marks a no-op slot. PR101
    /// reserves 255. Must satisfy `no_op_sentinel >= n_dims`.
    pub no_op_sentinel: i64,
}

/// Per-column centered-delta uint8 latents wrapped in raw-LZMA1 bytes.
///
/// Mirrors `tac.packet_compiler.CenteredDeltaUint8Stream`.
#[derive(Debug, Clone)]
pub struct CenteredDeltaUint8Stream {
    /// fp16-precision per-column `min` (length `n_dims`). Stored as bytes
    /// here (fp16 is not a stable Rust primitive type without `half` crate);
    /// the implementer chooses whether to lift to `half::f16` or hold raw
    /// little-endian bytes — golden vector mins/scales arrays will pin the
    /// answer.
    pub mins: Vec<u8>,
    /// fp16-precision per-column `scale` (length `n_dims`). Storage parity
    /// with `mins`.
    pub scales: Vec<u8>,
    /// First-row absolute quantised base values (length `n_dims`).
    pub base: Vec<u8>,
    /// Centered temporal deltas, shape `(n_pairs - 1, n_dims)` row-major.
    pub deltas: Vec<u8>,
    /// Number of rows the original input had (the implementer reshapes
    /// `deltas` to `(n_pairs - 1) × n_dims` at decode).
    pub n_pairs: usize,
    /// Number of columns the original input had.
    pub n_dims: usize,
    /// Raw-LZMA1-compressed concatenation of the three sections that ships
    /// in the archive. Header parity with Python's
    /// `lzma.FILTER_LZMA1 dict=4096 lc=3 lp=0 pb=0` is non-negotiable.
    pub lzma_bytes: Vec<u8>,
}

/// Output of [`split_brotli_self_delimiting`].
///
/// Mirrors `tac.packet_compiler.SplitBrotliStream`.
#[derive(Debug, Clone)]
pub struct SplitBrotliStream {
    /// Concatenation of N self-delimiting Brotli streams.
    pub payload: Vec<u8>,
    /// Number of independently-compressed sub-streams encoded.
    pub n_streams: usize,
    /// Byte offsets *after* each stream within `payload`. Last entry equals
    /// `payload.len()`.
    pub stream_byte_offsets: Vec<usize>,
}

// ── Stubbed encode/decode contracts ─────────────────────────────────────────

/// Encode a per-pair sparse correction sidecar.
///
/// **SCAFFOLD-ONLY** — returns
/// `Err(PacketCompilerError::NotImplemented("encode_ranked_no_op_sidecar"))`.
///
/// Mirrors `tac.packet_compiler.encode_ranked_no_op_sidecar`. Output bytes
/// are the layout `[dim_packed_le | length_rank_le | huffman_bits | noop_rank_le]`
/// (PR101 "huff_enum" variant).
///
/// Target SHA-256: see
/// `src/tac/packet_compiler/golden_vectors/ranked_no_op_sidecar_v1.json`.
pub fn encode_ranked_no_op_sidecar(
    _dims: &[i64],
    _delta_indices: &[i64],
    _schema: &RankedSidecarSchema,
) -> Result<Vec<u8>> {
    Err(PacketCompilerError::NotImplemented(
        "encode_ranked_no_op_sidecar",
    ))
}

/// Decode a ranked Huffman/no-op sidecar produced by
/// [`encode_ranked_no_op_sidecar`].
///
/// **SCAFFOLD-ONLY**. The implementer must keep the parameter set identical
/// to the Python oracle so callers can serialise the metadata header out-of-band
/// once.
pub fn decode_ranked_no_op_sidecar(
    _payload: &[u8],
    _schema: &RankedSidecarSchema,
    _dim_bytes: usize,
    _rank_bytes: usize,
    _noop_rank_bytes: usize,
    _noop_count: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    Err(PacketCompilerError::NotImplemented(
        "decode_ranked_no_op_sidecar",
    ))
}

/// Encode a per-column quantised stream as centered-delta uint8 under raw-LZMA.
///
/// **SCAFFOLD-ONLY**.
///
/// `values` is expected as row-major `(n_pairs, n_dims)` `f32`. The
/// implementer derives per-column `mins` / `scales` (fp16) when the caller
/// passes `None`. Header parity with Python's `lzma.FILTER_LZMA1 dict=4096
/// lc=3 lp=0 pb=0` is non-negotiable; see liblzma's
/// `LzmaOptions::dict_size(4096).literal_context_bits(3)` style API.
pub fn encode_centered_delta_uint8(
    _values: &[f32],
    _n_pairs: usize,
    _n_dims: usize,
    _mins: Option<&[u8]>,
    _scales: Option<&[u8]>,
) -> Result<CenteredDeltaUint8Stream> {
    Err(PacketCompilerError::NotImplemented(
        "encode_centered_delta_uint8",
    ))
}

/// Decode a centered-delta uint8 stream back to row-major `(n_pairs, n_dims)`
/// `f32`.
///
/// **SCAFFOLD-ONLY**. The implementer accepts either the typed stream
/// (preferred) or raw `lzma_bytes` with explicit `n_pairs` / `n_dims`.
pub fn decode_centered_delta_uint8(
    _lzma_bytes: &[u8],
    _n_pairs: usize,
    _n_dims: usize,
) -> Result<Vec<f32>> {
    Err(PacketCompilerError::NotImplemented(
        "decode_centered_delta_uint8",
    ))
}

/// Concatenate N independently-Brotli-compressed byte streams.
///
/// **SCAFFOLD-ONLY**.
///
/// PR101 hardcodes `lgwin=22, quality=11`; the implementer must accept those
/// as inputs so other callers (PR103 adaptive search) can sweep the space.
/// Each sub-stream is compressed with the same parameters and the Brotli
/// payloads are concatenated; the reader uses Brotli's frame structure to
/// know where each stream ends — there is no length prefix.
pub fn split_brotli_self_delimiting(
    _streams: &[&[u8]],
    _lgwin: u8,
    _quality: u8,
) -> Result<SplitBrotliStream> {
    Err(PacketCompilerError::NotImplemented(
        "split_brotli_self_delimiting",
    ))
}

/// Inverse of [`split_brotli_self_delimiting`].
///
/// **SCAFFOLD-ONLY**. The reader walks the concatenated Brotli payload
/// byte-by-byte until each `n_streams` decoder reports end-of-stream. PR101
/// does exactly this in `decompress_brotli_streams`.
pub fn parse_split_brotli_self_delimiting(
    _payload: &[u8],
    _n_streams: usize,
) -> Result<Vec<Vec<u8>>> {
    Err(PacketCompilerError::NotImplemented(
        "parse_split_brotli_self_delimiting",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scaffold MUST refuse to be called silently — explicit
    /// `NotImplemented` error matches the contract documented in
    /// `crate::PacketCompilerError`.
    #[test]
    fn scaffold_refuses_with_not_implemented_error() {
        let schema = RankedSidecarSchema {
            n_pairs: 24,
            n_dims: 8,
            deltas: vec![-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
            huff_min_len: 2,
            huff_max_len: 8,
            no_op_sentinel: 255,
        };
        let dims = vec![255i64; 24];
        let delta_indices = vec![0i64; 24];
        let err = encode_ranked_no_op_sidecar(&dims, &delta_indices, &schema)
            .expect_err("scaffold must refuse");
        assert!(matches!(
            err,
            PacketCompilerError::NotImplemented("encode_ranked_no_op_sidecar")
        ));
    }
}
