//! PR92 ``qzs3_range_joint_r258`` joint-stream primitives — Rust port.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.pr92_joint_stream`](
//! file:../../../../../src/tac/packet_compiler/pr92_joint_stream.py).
//!
//! PR92's `submissions/qzs3_range_joint_r258/inflate.py` upgraded PR81 with
//! JOINT range coding of correlated mask + action streams. The novel
//! insight is the META-CODEC pattern: when two byte streams are
//! correlated, encoding their JOIN as one composite payload is strictly
//! smaller than encoding each independently.
//!
//! Three magic-prefix grammars land here:
//!
//! 1. **RMC1 — Range Mask Composite**
//!    ([`pack_rmc1_composite`] +
//!    [`unpack_rmc1_composite`]).
//!    Frame: ``b"RMC1" || <u32 seg_len LE> || <u32 side_len LE> ||
//!    seg_bytes || side_bytes``.
//!
//! 2. **RSA1 — Side Action (range-coded)**
//!    ([`pack_rsa1_side`] +
//!    [`unpack_rsa1_side`]).
//!    Frame: ``b"RSA1" || <u16 count LE> || <u8 action_bits> ||
//!    <u8 table_id> || packed_bits``. The packed-bits layout matches PR81's
//!    [`encode_router_actions`](crate::pr81_quantizr::encode_router_actions).
//!
//! 3. **RSB1 — Side Action (brotli-fallback)**
//!    ([`pack_rsb1_side`] +
//!    [`unpack_rsb1_side`]).
//!    Frame: ``b"RSB1" || <u16 count LE> || <u8 table_id> || <u8 0> ||
//!    brotli(body_bytes)``.

pub mod rmc;

pub use rmc::{
    pack_rmc1_composite, pack_rsa1_side, pack_rsb1_side, unpack_rmc1_composite, unpack_rsa1_side,
    unpack_rsb1_side, RMC1Composite, RSA1Side, RSB1Side, MAGIC_RMC1, MAGIC_RSA1, MAGIC_RSB1,
};
