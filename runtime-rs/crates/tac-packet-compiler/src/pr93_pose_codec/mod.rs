//! PR93 ``flatpup`` pose-codec primitives — Rust port.
//!
//! Mirrors the public API of
//! [`tac.packet_compiler.pr93_pose_codec`](
//! file:../../../../../src/tac/packet_compiler/pr93_pose_codec.py).
//!
//! Two primitives are exposed:
//!
//! 1. **Delta-varint pose codec** ([`encode_delta_varint_pose`] /
//!    [`decode_delta_varint_pose`]) — implemented; byte-for-byte parity
//!    against `pr93_delta_varint_pose_v1.json`.
//!
//! 2. **QZMB1 compact-model block grammar** ([`pack_qzmb1_block`] /
//!    [`unpack_qzmb1_block`]) — implemented; byte-for-byte parity against
//!    `pr93_qzmb1_v1.json`. The QZPDV1 magic is consumed inside
//!    [`encode_delta_varint_pose`].
//!
//! # Byte layout (matches the Python oracle exactly)
//!
//! ```text
//! MAGIC_POSE_DV         (8 bytes — b"QZPDV1\0\0")
//! n_rows u32 LE
//! n_dims u32 LE
//! bits   u32 LE         (8 or 16)
//! lo      fp32 LE * n_dims
//! scale   fp32 LE * n_dims
//! first   {u8|u16} LE * n_dims     (width = bits/8)
//! deltas  signed-varint stream * (n_rows-1)*n_dims      (zigzag + LEB128)
//! ```
//!
//! Sibling binary fixtures live at:
//!
//! - `src/tac/packet_compiler/golden_vectors/pr93_delta_varint_pose_v1_poses.bin`
//! - `src/tac/packet_compiler/golden_vectors/pr93_delta_varint_pose_v1_lo.bin`
//! - `src/tac/packet_compiler/golden_vectors/pr93_delta_varint_pose_v1_scale.bin`

pub mod delta_varint;
pub mod qzmb1;

pub use delta_varint::{
    decode_delta_varint_pose, encode_delta_varint_pose, DeltaVarintPoseStream, MAGIC_MODEL_COMPACT,
    MAGIC_POSE_DV,
};
pub use qzmb1::{pack_qzmb1_block, unpack_qzmb1_block, QZMB1Block};
