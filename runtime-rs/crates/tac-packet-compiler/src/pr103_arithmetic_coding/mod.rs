//! PR103 arithmetic-coding — Rust native port with one remaining search scaffold.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.pr103_arithmetic_coding`](
//! file:../../../../../src/tac/packet_compiler/pr103_arithmetic_coding.py).
//!
//! Three primitives are exposed:
//!
//! 1. **Merged range stream over multiple weight tensors** — implemented. One
//!    range-coded byte string spanning N tensors with per-tensor histograms.
//!    Uses `constriction` range coding with byte-for-byte Python parity.
//! 2. **Latent-hi arithmetic** — implemented. Encode the high byte of a uint16 zigzag
//!    delta with a peaked histogram; beats LZMA/Brotli by ≈ 8 KB on PR103's
//!    600×28 latent stream.
//! 3. **Adaptive Brotli parameter search** — scaffold-only. Sweep `(lgwin, quality)` under
//!    a time/eval budget and keep the smallest output.
//!
//! Each primitive has a paired golden vector under
//! `src/tac/packet_compiler/golden_vectors/`:
//!
//! - `merged_range_stream_v1.json`
//! - `latent_hi_arithmetic_v1.json`
//!
//! Adaptive Brotli search has no golden vector by design (the result depends
//! on per-host CPU timing); its parity is contract-level, not byte-level.
//!
//! # Constriction wire-format note
//!
//! The Python oracle serialises `constriction`'s uint32 word array to
//! **big-endian** bytes so the wire format is portable across architectures.
//! The Rust impl MUST honour the same big-endian serialisation when calling
//! the underlying `get_compressed()` API. See the Python helpers
//! `_words_to_uint32_bytes` / `_uint32_bytes_to_words`.

pub mod latent_hi;
pub mod latent_hi_hand_optimized;
pub mod merged_range_stream;
pub mod stubs;

pub use latent_hi_hand_optimized::{
    encode_latent_hi_arithmetic_hand_optimized, PreparedCategorical,
};

pub use stubs::{
    adaptive_brotli_param_search, decode_latent_hi_arithmetic, decode_merged_range_stream,
    encode_latent_hi_arithmetic, encode_merged_range_stream, AdaptiveBrotliResult,
    MergedRangeStream, WeightTensorACSpec,
};
