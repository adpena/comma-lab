//! Rust port of PR92's RMC1 / RSA1 / RSB1 joint-stream meta-codec.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`src/tac/packet_compiler/pr92_joint_stream.py::{pack_rmc1_composite,
//! pack_rsa1_side, pack_rsb1_side, unpack_*}`) produces magic-prefixed
//! payloads whose wire format this module reproduces exactly.
//!
//! # Layout
//!
//! ```text
//! RMC1: b"RMC1" || <u32 seg_len LE> || <u32 side_len LE> ||
//!       seg_bytes || side_bytes
//!
//! RSA1: b"RSA1" || <u16 count LE> || <u8 action_bits> ||
//!       <u8 table_id> || packed_bits
//!
//! RSB1: b"RSB1" || <u16 count LE> || <u8 table_id> || <u8 0> ||
//!       brotli(body_bytes)
//! ```

use std::io::Write;

use brotli::enc::backward_references::BrotliEncoderParams;
use brotli::enc::writer::CompressorWriter;
use brotli_decompressor::{BrotliDecompressStream, BrotliResult, BrotliState, StandardAlloc};

use crate::{PacketCompilerError, Result};

/// 4-byte magic identifying a RMC1 composite mask+action payload.
pub const MAGIC_RMC1: [u8; 4] = *b"RMC1";

/// 4-byte magic identifying a RSA1 range-coded side-action payload.
pub const MAGIC_RSA1: [u8; 4] = *b"RSA1";

/// 4-byte magic identifying a RSB1 brotli-compressed side-action payload.
pub const MAGIC_RSB1: [u8; 4] = *b"RSB1";

/// Parsed RMC1 composite mask + side-action payload.
#[derive(Debug, Clone)]
pub struct RMC1Composite {
    /// The full magic-prefixed payload bytes.
    pub payload: Vec<u8>,
    /// Mask byte stream (first component).
    pub seg_bytes: Vec<u8>,
    /// Side-action byte stream (second component; typically an RSA1 or RSB1
    /// framed payload).
    pub side_bytes: Vec<u8>,
}

/// Parsed RSA1 side-action payload (range-coded).
#[derive(Debug, Clone)]
pub struct RSA1Side {
    /// The full magic-prefixed payload bytes.
    pub payload: Vec<u8>,
    /// Action count.
    pub count: u16,
    /// Bit-width per action (1..=8).
    pub action_bits: u8,
    /// Caller-defined table identifier.
    pub table_id: u8,
    /// Packed-bits body (LSB-first; same layout as
    /// [`encode_router_actions`](crate::pr81_quantizr::encode_router_actions)).
    pub body: Vec<u8>,
}

/// Parsed RSB1 side-action payload (brotli-compressed).
#[derive(Debug, Clone)]
pub struct RSB1Side {
    /// The full magic-prefixed payload bytes.
    pub payload: Vec<u8>,
    /// Action count.
    pub count: u16,
    /// Caller-defined table identifier.
    pub table_id: u8,
    /// Decompressed body bytes (length `count`).
    pub body_bytes: Vec<u8>,
}

// ── RMC1 composite ──────────────────────────────────────────────────────────

/// Pack two correlated byte streams under the RMC1 magic frame.
///
/// Both inputs may be empty. Each length is stored as a little-endian
/// `u32` immediately after the magic so the parser can split the
/// concatenation back without external metadata.
pub fn pack_rmc1_composite(seg_bytes: &[u8], side_bytes: &[u8]) -> Result<RMC1Composite> {
    if seg_bytes.len() > u32::MAX as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RMC1 seg component too large ({} bytes); must fit in u32",
            seg_bytes.len()
        )));
    }
    if side_bytes.len() > u32::MAX as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RMC1 side component too large ({} bytes); must fit in u32",
            side_bytes.len()
        )));
    }
    let mut payload = Vec::with_capacity(12 + seg_bytes.len() + side_bytes.len());
    payload.extend_from_slice(&MAGIC_RMC1);
    payload.extend_from_slice(&(seg_bytes.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(side_bytes.len() as u32).to_le_bytes());
    payload.extend_from_slice(seg_bytes);
    payload.extend_from_slice(side_bytes);
    Ok(RMC1Composite {
        payload,
        seg_bytes: seg_bytes.to_vec(),
        side_bytes: side_bytes.to_vec(),
    })
}

/// Inverse of [`pack_rmc1_composite`].
pub fn unpack_rmc1_composite(payload: &[u8]) -> Result<RMC1Composite> {
    if payload.len() < 4 || payload[..4] != MAGIC_RMC1 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "missing RMC1 magic; got prefix {:?}",
            &payload[..payload.len().min(4)]
        )));
    }
    if payload.len() < 12 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "truncated RMC1 header".into(),
        ));
    }
    let seg_len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    let side_len =
        u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize;
    let off: usize = 12;
    let end = off
        .checked_add(seg_len)
        .and_then(|v: usize| v.checked_add(side_len))
        .ok_or_else(|| {
            PacketCompilerError::GoldenVectorIo(format!(
                "RMC1 lengths overflow: seg={seg_len} side={side_len}"
            ))
        })?;
    if end != payload.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RMC1 length mismatch: header={end} actual={}",
            payload.len()
        )));
    }
    let seg_bytes = payload[off..off + seg_len].to_vec();
    let side_bytes = payload[off + seg_len..end].to_vec();
    Ok(RMC1Composite {
        payload: payload.to_vec(),
        seg_bytes,
        side_bytes,
    })
}

// ── RSA1 side-action (range-coded) ──────────────────────────────────────────

/// Frame an RSA1 side-action payload.
///
/// `body` must carry at least `ceil(count * action_bits / 8)` bytes of
/// LSB-first packed actions. `action_bits` must satisfy `1 <= bits <= 8`.
pub fn pack_rsa1_side(
    count: u16,
    action_bits: u8,
    table_id: u8,
    body: &[u8],
) -> Result<RSA1Side> {
    if !(1..=8).contains(&action_bits) {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "action_bits must satisfy 1 <= action_bits <= 8; got {action_bits}"
        )));
    }
    let required = ((count as usize) * (action_bits as usize)).div_ceil(8);
    if body.len() < required {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "body too small for count={count} action_bits={action_bits}; need >= {required} bytes, got {}",
            body.len()
        )));
    }
    let mut payload = Vec::with_capacity(8 + body.len());
    payload.extend_from_slice(&MAGIC_RSA1);
    payload.extend_from_slice(&count.to_le_bytes());
    payload.push(action_bits);
    payload.push(table_id);
    payload.extend_from_slice(body);
    Ok(RSA1Side {
        payload,
        count,
        action_bits,
        table_id,
        body: body.to_vec(),
    })
}

/// Inverse of [`pack_rsa1_side`].
pub fn unpack_rsa1_side(payload: &[u8]) -> Result<RSA1Side> {
    if payload.len() < 4 || payload[..4] != MAGIC_RSA1 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "missing RSA1 magic; got prefix {:?}",
            &payload[..payload.len().min(4)]
        )));
    }
    if payload.len() < 8 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "truncated RSA1 side-action payload".into(),
        ));
    }
    let count = u16::from_le_bytes([payload[4], payload[5]]);
    let action_bits = payload[6];
    let table_id = payload[7];
    if !(1..=8).contains(&action_bits) {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "unsupported RSA1 action_bits {action_bits}; expected 1..8"
        )));
    }
    let body = payload[8..].to_vec();
    let required = ((count as usize) * (action_bits as usize)).div_ceil(8);
    if body.len() < required {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "RSA1 body too small for count={count} action_bits={action_bits}; need >= {required} bytes, got {}",
            body.len()
        )));
    }
    Ok(RSA1Side {
        payload: payload.to_vec(),
        count,
        action_bits,
        table_id,
        body,
    })
}

// ── RSB1 side-action (brotli) ───────────────────────────────────────────────

/// Frame an RSB1 brotli-compressed side-action payload.
///
/// `actions` is a uint8 byte stream that is brotli-compressed with
/// `(quality, lgwin)`. The body bytes after the 8-byte header are the
/// compressed result.
///
/// PR92's default is `quality=11, lgwin=22` (matching the rest of the
/// PR101/103 family). Callers may override.
pub fn pack_rsb1_side(
    actions: &[u8],
    table_id: u8,
    brotli_quality: u8,
    brotli_lgwin: u8,
) -> Result<RSB1Side> {
    let count = actions.len();
    if count > u16::MAX as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "count must satisfy 0 <= count < 65536; got {count}"
        )));
    }
    if !(10..=24).contains(&brotli_lgwin) {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "brotli_lgwin must be in [10, 24]; got {brotli_lgwin}"
        )));
    }
    if brotli_quality > 11 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "brotli_quality must be in [0, 11]; got {brotli_quality}"
        )));
    }
    let compressed = brotli_compress(actions, brotli_quality, brotli_lgwin)?;
    let mut payload = Vec::with_capacity(8 + compressed.len());
    payload.extend_from_slice(&MAGIC_RSB1);
    payload.extend_from_slice(&(count as u16).to_le_bytes());
    payload.push(table_id);
    payload.push(0u8); // reserved
    payload.extend_from_slice(&compressed);
    Ok(RSB1Side {
        payload,
        count: count as u16,
        table_id,
        body_bytes: actions.to_vec(),
    })
}

/// Inverse of [`pack_rsb1_side`].
pub fn unpack_rsb1_side(payload: &[u8]) -> Result<RSB1Side> {
    if payload.len() < 4 || payload[..4] != MAGIC_RSB1 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "missing RSB1 magic; got prefix {:?}",
            &payload[..payload.len().min(4)]
        )));
    }
    if payload.len() < 8 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "truncated RSB1 side-action payload".into(),
        ));
    }
    let count = u16::from_le_bytes([payload[4], payload[5]]);
    let table_id = payload[6];
    let reserved = payload[7];
    if reserved != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "unsupported RSB1 reserved byte {reserved}; expected 0"
        )));
    }
    let compressed = &payload[8..];
    let body_bytes = brotli_decompress(compressed)?;
    if body_bytes.len() != count as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "decoded {} RSB1 actions, expected {count}",
            body_bytes.len()
        )));
    }
    Ok(RSB1Side {
        payload: payload.to_vec(),
        count,
        table_id,
        body_bytes,
    })
}

// ── Brotli helpers ──────────────────────────────────────────────────────────

fn brotli_params(quality: u8, lgwin: u8) -> BrotliEncoderParams {
    BrotliEncoderParams {
        quality: quality as i32,
        lgwin: lgwin as i32,
        ..Default::default()
    }
}

fn brotli_compress(raw: &[u8], quality: u8, lgwin: u8) -> Result<Vec<u8>> {
    let params = brotli_params(quality, lgwin);
    let mut out = Vec::with_capacity(raw.len() / 2 + 64);
    {
        let mut writer = CompressorWriter::with_params(&mut out, 4096, &params);
        writer.write_all(raw).map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!("brotli compress write failed: {e:?}"))
        })?;
    }
    Ok(out)
}

fn brotli_decompress(input: &[u8]) -> Result<Vec<u8>> {
    let mut state: BrotliState<StandardAlloc, StandardAlloc, StandardAlloc> = BrotliState::new(
        StandardAlloc::default(),
        StandardAlloc::default(),
        StandardAlloc::default(),
    );
    let mut output = Vec::new();
    let mut input_offset = 0usize;
    let mut output_buf = vec![0u8; 65536];
    loop {
        let mut available_in = input.len() - input_offset;
        let mut available_out = output_buf.len();
        let mut out_offset = 0usize;
        let mut total_out: usize = 0;
        let result = BrotliDecompressStream(
            &mut available_in,
            &mut input_offset,
            input,
            &mut available_out,
            &mut out_offset,
            &mut output_buf,
            &mut total_out,
            &mut state,
        );
        if out_offset > 0 {
            output.extend_from_slice(&output_buf[..out_offset]);
        }
        match result {
            BrotliResult::ResultSuccess => return Ok(output),
            BrotliResult::NeedsMoreInput => {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "truncated brotli payload (needs more input at offset {input_offset})"
                )));
            }
            BrotliResult::NeedsMoreOutput => continue,
            BrotliResult::ResultFailure => {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "brotli decode failed at input offset {input_offset}"
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmc1_roundtrip() {
        let seg = [1u8, 2, 3, 4, 5];
        let side = [10u8, 20, 30];
        let composite = pack_rmc1_composite(&seg, &side).unwrap();
        // 4 (magic) + 4 (u32 seg_len) + 4 (u32 side_len) + 5 + 3 = 20
        assert_eq!(composite.payload.len(), 20);
        assert_eq!(&composite.payload[..4], &MAGIC_RMC1);
        let parsed = unpack_rmc1_composite(&composite.payload).unwrap();
        assert_eq!(parsed.seg_bytes, seg);
        assert_eq!(parsed.side_bytes, side);
    }

    #[test]
    fn rmc1_empty_components() {
        let composite = pack_rmc1_composite(&[], &[]).unwrap();
        assert_eq!(composite.payload.len(), 12);
        let parsed = unpack_rmc1_composite(&composite.payload).unwrap();
        assert!(parsed.seg_bytes.is_empty());
        assert!(parsed.side_bytes.is_empty());
    }

    #[test]
    fn rmc1_rejects_bad_magic() {
        let mut payload = vec![b'X', b'Y', b'Z', b'Q'];
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        assert!(unpack_rmc1_composite(&payload).is_err());
    }

    #[test]
    fn rsa1_roundtrip() {
        let body = [0x01u8, 0x02, 0x03];
        let side = pack_rsa1_side(8, 3, 7, &body).unwrap();
        // header layout check
        assert_eq!(&side.payload[..4], &MAGIC_RSA1);
        assert_eq!(u16::from_le_bytes([side.payload[4], side.payload[5]]), 8);
        assert_eq!(side.payload[6], 3);
        assert_eq!(side.payload[7], 7);
        assert_eq!(&side.payload[8..], &body);
        let parsed = unpack_rsa1_side(&side.payload).unwrap();
        assert_eq!(parsed.count, 8);
        assert_eq!(parsed.action_bits, 3);
        assert_eq!(parsed.table_id, 7);
        assert_eq!(parsed.body, body);
    }

    #[test]
    fn rsa1_rejects_short_body() {
        // 8 actions × 3 bits = 24 bits = 3 bytes, but we only provide 2.
        let body = [0u8; 2];
        assert!(pack_rsa1_side(8, 3, 0, &body).is_err());
    }

    #[test]
    fn rsa1_rejects_invalid_action_bits() {
        let body = vec![0u8; 8];
        assert!(pack_rsa1_side(8, 0, 0, &body).is_err());
        assert!(pack_rsa1_side(8, 9, 0, &body).is_err());
    }

    #[test]
    fn rsb1_roundtrip() {
        let actions: Vec<u8> = (0..200).map(|i| (i % 4) as u8).collect();
        let side = pack_rsb1_side(&actions, 3, 11, 22).unwrap();
        assert_eq!(&side.payload[..4], &MAGIC_RSB1);
        assert_eq!(side.count, 200);
        assert_eq!(side.table_id, 3);
        let parsed = unpack_rsb1_side(&side.payload).unwrap();
        assert_eq!(parsed.count, 200);
        assert_eq!(parsed.table_id, 3);
        assert_eq!(parsed.body_bytes, actions);
    }

    #[test]
    fn nested_rmc1_around_rsa1() {
        // Compose RSA1 inside RMC1 — exactly the PR92 golden vector shape.
        let body = vec![0x12u8, 0x34, 0x56];
        let rsa = pack_rsa1_side(8, 3, 2, &body).unwrap();
        let seg = vec![0xABu8; 16];
        let composite = pack_rmc1_composite(&seg, &rsa.payload).unwrap();
        let parsed_outer = unpack_rmc1_composite(&composite.payload).unwrap();
        assert_eq!(parsed_outer.seg_bytes, seg);
        let parsed_inner = unpack_rsa1_side(&parsed_outer.side_bytes).unwrap();
        assert_eq!(parsed_inner.count, 8);
        assert_eq!(parsed_inner.table_id, 2);
        assert_eq!(parsed_inner.body, body);
    }
}
