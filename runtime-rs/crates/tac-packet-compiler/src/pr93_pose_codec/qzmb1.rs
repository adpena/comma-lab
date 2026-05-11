//! Rust port of PR93's QZMB1 compact-model block grammar.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`src/tac/packet_compiler/pr93_pose_codec.py::{pack_qzmb1_block,
//! unpack_qzmb1_block}`) frames a model-config + tensor-record body under
//! the 8-byte [`MAGIC_MODEL_COMPACT`] magic with a fixed `<HH` header.
//! This module reproduces that framing identically.
//!
//! # Layout
//!
//! ```text
//! MAGIC_MODEL_COMPACT     (8 bytes — b"QZMB1\0\0\0")
//! block_size       u16 LE     (PR93 uses 32)
//! arch_config_len  u16 LE
//! arch_config_json *arch_config_len bytes (UTF-8 JSON, opaque to this primitive)
//! body            *N bytes    (opaque tensor records)
//! ```

use crate::{PacketCompilerError, Result};

use super::delta_varint::MAGIC_MODEL_COMPACT;

/// Parsed QZMB1 compact-model block.
///
/// Mirrors `tac.packet_compiler.QZMB1Block`. The full magic-prefixed
/// payload + the header fields + the opaque body bytes.
#[derive(Debug, Clone)]
pub struct QZMB1Block {
    /// Concatenation of magic + header + arch-config + body.
    pub payload: Vec<u8>,
    /// FP4 codebook block size (PR93 uses 32).
    pub block_size: u16,
    /// JSON-encoded architecture configuration bytes.
    pub arch_config_json: Vec<u8>,
    /// Opaque post-arch-config body bytes (tensor records).
    pub body: Vec<u8>,
}

/// Frame a QZMB1 compact-model block from its header + body bytes.
///
/// Mirrors `tac.packet_compiler.pack_qzmb1_block`.
///
/// - `block_size` must satisfy `0 < block_size < 65536`.
/// - `arch_config_json.len()` must satisfy `< 65536`.
/// - `body` may be empty.
pub fn pack_qzmb1_block(
    block_size: u16,
    arch_config_json: &[u8],
    body: &[u8],
) -> Result<QZMB1Block> {
    if block_size == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "block_size must satisfy 0 < block_size < 65536; got {block_size}"
        )));
    }
    if arch_config_json.len() >= 65536 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "arch_config_json too long ({} bytes); max 65535",
            arch_config_json.len()
        )));
    }
    let mut payload =
        Vec::with_capacity(MAGIC_MODEL_COMPACT.len() + 4 + arch_config_json.len() + body.len());
    payload.extend_from_slice(&MAGIC_MODEL_COMPACT);
    payload.extend_from_slice(&block_size.to_le_bytes());
    payload.extend_from_slice(&(arch_config_json.len() as u16).to_le_bytes());
    payload.extend_from_slice(arch_config_json);
    payload.extend_from_slice(body);
    Ok(QZMB1Block {
        payload,
        block_size,
        arch_config_json: arch_config_json.to_vec(),
        body: body.to_vec(),
    })
}

/// Inverse of [`pack_qzmb1_block`].
///
/// Mirrors `tac.packet_compiler.unpack_qzmb1_block`. Returns the parsed
/// fields; raises `GoldenVectorIo` on missing magic or truncated header.
pub fn unpack_qzmb1_block(payload: &[u8]) -> Result<QZMB1Block> {
    let magic_len = MAGIC_MODEL_COMPACT.len();
    if payload.len() < magic_len || payload[..magic_len] != MAGIC_MODEL_COMPACT {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "missing QZMB1 magic; got prefix {:?}",
            &payload[..payload.len().min(magic_len)]
        )));
    }
    if payload.len() < magic_len + 4 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "truncated QZMB1 header".into(),
        ));
    }
    let block_size = u16::from_le_bytes([payload[magic_len], payload[magic_len + 1]]);
    let arch_len =
        u16::from_le_bytes([payload[magic_len + 2], payload[magic_len + 3]]) as usize;
    let arch_start = magic_len + 4;
    let arch_end = arch_start
        .checked_add(arch_len)
        .ok_or_else(|| PacketCompilerError::GoldenVectorIo("arch_len overflow".into()))?;
    if arch_end > payload.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "arch_config truncated: header says {} bytes but only {} available",
            arch_len,
            payload.len() - arch_start
        )));
    }
    let arch_config_json = payload[arch_start..arch_end].to_vec();
    let body = payload[arch_end..].to_vec();
    Ok(QZMB1Block {
        payload: payload.to_vec(),
        block_size,
        arch_config_json,
        body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_pack_layout() {
        let arch_json = b"{\"hidden\": 64}";
        let body = vec![0xABu8, 0xCD, 0xEF];
        let block = pack_qzmb1_block(32, arch_json, &body).unwrap();
        // 8 (magic) + 2 (block_size) + 2 (arch_len) + 14 (arch_json) + 3 = 29
        assert_eq!(block.payload.len(), 8 + 2 + 2 + arch_json.len() + body.len());
        assert_eq!(&block.payload[..8], &MAGIC_MODEL_COMPACT);
        assert_eq!(u16::from_le_bytes([block.payload[8], block.payload[9]]), 32);
        assert_eq!(
            u16::from_le_bytes([block.payload[10], block.payload[11]]),
            arch_json.len() as u16
        );
    }

    #[test]
    fn roundtrip_small() {
        let arch_json = b"{\"hidden\": 64, \"blocks\": 3, \"input_dim\": 5}";
        let body: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let block = pack_qzmb1_block(32, arch_json, &body).unwrap();
        let parsed = unpack_qzmb1_block(&block.payload).unwrap();
        assert_eq!(parsed.block_size, 32);
        assert_eq!(parsed.arch_config_json, arch_json);
        assert_eq!(parsed.body, body);
    }

    #[test]
    fn empty_body_supported() {
        let block = pack_qzmb1_block(32, b"{}", &[]).unwrap();
        let parsed = unpack_qzmb1_block(&block.payload).unwrap();
        assert!(parsed.body.is_empty());
        assert_eq!(parsed.arch_config_json, b"{}");
    }

    #[test]
    fn rejects_zero_block_size() {
        assert!(pack_qzmb1_block(0, b"{}", &[]).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut payload = vec![0u8; 12];
        payload[..8].copy_from_slice(b"BADMAGIC");
        assert!(unpack_qzmb1_block(&payload).is_err());
    }

    #[test]
    fn rejects_truncated_header() {
        // 8 bytes magic + 0 trailing bytes → can't read u16+u16 header.
        let payload = MAGIC_MODEL_COMPACT.to_vec();
        assert!(unpack_qzmb1_block(&payload).is_err());
    }

    #[test]
    fn rejects_truncated_arch_config() {
        // arch_len says 100 but only ~12 bytes remain.
        let mut payload = Vec::new();
        payload.extend_from_slice(&MAGIC_MODEL_COMPACT);
        payload.extend_from_slice(&32u16.to_le_bytes());
        payload.extend_from_slice(&100u16.to_le_bytes());
        payload.extend_from_slice(b"short");
        assert!(unpack_qzmb1_block(&payload).is_err());
    }
}
