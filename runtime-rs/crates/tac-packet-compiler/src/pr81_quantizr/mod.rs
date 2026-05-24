//! PR81 ``qzs3`` Quantizr primitives — Rust port.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.pr81_quantizr`](
//! file:../../../../../src/tac/packet_compiler/pr81_quantizr.py).
//!
//! Two primitives are exposed:
//!
//! 1. **PR81 FP4 codebook** ([`FP4Codebook`] +
//!    [`quantize_to_nibbles`] +
//!    [`dequantize_from_nibbles`] +
//!    [`pack_nibbles`]) — implemented;
//!    byte-for-byte parity against `pr81_fp4_codebook_v1.json`.
//!
//!    The codebook is an asymmetric 8-level positive table
//!    ``[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`` plus an explicit sign
//!    bit. Each value is encoded as a 4-bit nibble ``sssM`` where ``s`` is
//!    the sign bit and ``MMM`` is the 3-bit magnitude index. Two nibbles
//!    per byte (``hi << 4 | lo``). Decoded against per-block fp32 scale.
//!
//! 2. **PR81 ROUTER_ACTION 3-bit packing**
//!    ([`encode_router_actions`] +
//!    [`decode_router_actions`]) —
//!    implemented; byte-for-byte parity against
//!    `pr81_router_action_v1.json`.
//!
//!    LSB-first bit-stream pack/unpack for small-integer per-frame action
//!    streams. PR81 uses ``bits=3`` (8-class router) and ``count=600`` →
//!    225 bytes. Generalises to any ``1 <= bits <= 8``.

pub mod fp4_codebook;
pub mod router_action;

pub use fp4_codebook::{
    dequantize_from_nibbles, pack_nibbles, quantize_to_nibbles, unpack_nibbles, FP4Codebook,
    PR81_POS_LEVELS,
};
pub use router_action::{decode_router_actions, encode_router_actions};
