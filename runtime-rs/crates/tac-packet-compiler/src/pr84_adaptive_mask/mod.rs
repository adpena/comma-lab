//! PR84 ``adaptive_range_mask`` adaptive-context primitive — Rust port.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.pr84_adaptive_mask`](
//! file:../../../../../src/tac/packet_compiler/pr84_adaptive_mask.py).
//!
//! One primitive lands here:
//!
//! * **Adaptive-context categorical stream** ([`encode_adaptive_context_stream`]
//!   / [`decode_adaptive_context_stream`]) — implemented; byte-for-byte
//!   parity against `pr84_adaptive_mask_context_v1.json`.

pub mod adaptive_mask_context;

pub use adaptive_mask_context::{
    decode_adaptive_context_stream, encode_adaptive_context_stream, AdaptiveContextSpec,
};
