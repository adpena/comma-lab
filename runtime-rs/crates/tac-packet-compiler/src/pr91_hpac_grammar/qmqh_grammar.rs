//! Rust port of PR91's QM0 / QH0 magic-prefix header grammar and hi-lo
//! nibble split permutation.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`src/tac/packet_compiler/pr91_hpac_grammar.py::{emit_qmqh_header,
//! parse_qmqh_header, pack_hi_lo_split, unpack_hi_lo_split}`) produces
//! magic-prefixed payloads + a deterministic byte permutation. This module
//! reproduces both layers identically.
//!
//! # Layout
//!
//! ```text
//! emit_qmqh_header(hilo_split=true)  → b"QH0"     (3 bytes)
//! emit_qmqh_header(hilo_split=false) → b"QM0"     (3 bytes)
//!
//! pack_hi_lo_split(packed_nibbles): every byte carries two nibbles
//!   `hi << 4 | lo`. Split into the two byte streams of hi-nibbles
//!   re-packed (hi-of-pair × 2 → one byte) and lo-nibbles re-packed.
//!   Output = concat(hi_packed, lo_packed). Even byte count required.
//! ```

use crate::{PacketCompilerError, Result};

/// 3-byte magic for plain QM0 (single-byte FP4 packed) layout.
pub const MAGIC_QM0: [u8; 3] = *b"QM0";

/// 3-byte magic for QH0 (hi-lo split FP4 packed) layout.
pub const MAGIC_QH0: [u8; 3] = *b"QH0";

/// Parsed QM0 / QH0 magic-prefix header.
///
/// Mirrors `tac.packet_compiler.QMQHHeader`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QMQHHeader {
    /// Either [`MAGIC_QM0`] or [`MAGIC_QH0`].
    pub magic: [u8; 3],
    /// `true` iff `magic == MAGIC_QH0`.
    pub hilo_split: bool,
    /// Byte offset where the per-tensor body begins (always 3).
    pub body_offset: usize,
}

/// Emit a 3-byte QMQH header (the magic alone — no body).
///
/// Mirrors `tac.packet_compiler.emit_qmqh_header`.
pub fn emit_qmqh_header(hilo_split: bool) -> [u8; 3] {
    if hilo_split {
        MAGIC_QH0
    } else {
        MAGIC_QM0
    }
}

/// Parse the leading 3-byte QM0 / QH0 magic into a typed header.
///
/// Mirrors `tac.packet_compiler.parse_qmqh_header`. Returns an error if
/// the payload is too short or carries an unknown magic prefix.
pub fn parse_qmqh_header(payload: &[u8]) -> Result<QMQHHeader> {
    if payload.len() < 3 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "QMQH payload too short ({} bytes); need at least 3",
            payload.len()
        )));
    }
    let magic: [u8; 3] = [payload[0], payload[1], payload[2]];
    if magic == MAGIC_QM0 {
        Ok(QMQHHeader {
            magic: MAGIC_QM0,
            hilo_split: false,
            body_offset: 3,
        })
    } else if magic == MAGIC_QH0 {
        Ok(QMQHHeader {
            magic: MAGIC_QH0,
            hilo_split: true,
            body_offset: 3,
        })
    } else {
        Err(PacketCompilerError::GoldenVectorIo(format!(
            "unknown QMQH magic {:?}; expected {:?} or {:?}",
            magic, MAGIC_QM0, MAGIC_QH0
        )))
    }
}

/// Split a hi-nibble-low-nibble packed byte stream into two byte streams.
///
/// Mirrors `tac.packet_compiler.pack_hi_lo_split`. PR91's QH0 layout
/// stores the hi-nibbles of every byte first (run-length-friendly under
/// Brotli), then the lo-nibbles. This is a pure byte permutation.
///
/// Output layout:
///
/// ```text
/// hi_packed = [(hi[0]<<4)|hi[1], (hi[2]<<4)|hi[3], ...]   (input.len()/2 bytes)
/// lo_packed = [(lo[0]<<4)|lo[1], (lo[2]<<4)|lo[3], ...]   (input.len()/2 bytes)
/// output    = hi_packed || lo_packed                       (input.len() bytes)
/// ```
///
/// Requires `packed_nibbles.len()` to be even (the per-side packing pairs
/// up the hi-nibbles in adjacent input bytes).
pub fn pack_hi_lo_split(packed_nibbles: &[u8]) -> Result<Vec<u8>> {
    let n = packed_nibbles.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n & 1 != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "hi-lo split requires even byte count; got {n}"
        )));
    }
    let half = n / 2;
    let mut out: Vec<u8> = Vec::with_capacity(n);
    // First half: hi_packed = pack-2 of hi-nibbles of input bytes.
    // Hi-nibbles array (length n): [b0>>4, b1>>4, ..., b_{n-1}>>4]
    // Pack 2 at a time: out[i] = (hi[2i] << 4) | hi[2i+1]
    for i in 0..half {
        let h0 = (packed_nibbles[2 * i] >> 4) & 0xF;
        let h1 = (packed_nibbles[2 * i + 1] >> 4) & 0xF;
        out.push((h0 << 4) | h1);
    }
    // Second half: lo_packed = pack-2 of lo-nibbles of input bytes.
    for i in 0..half {
        let l0 = packed_nibbles[2 * i] & 0xF;
        let l1 = packed_nibbles[2 * i + 1] & 0xF;
        out.push((l0 << 4) | l1);
    }
    Ok(out)
}

/// Inverse of [`pack_hi_lo_split`].
///
/// Mirrors `tac.packet_compiler.unpack_hi_lo_split`. Splits the input in
/// half, unpacks each half back to per-nibble arrays of length `n`, then
/// re-interleaves `(hi << 4) | lo` to reproduce the original packed bytes.
pub fn unpack_hi_lo_split(split_payload: &[u8]) -> Result<Vec<u8>> {
    let n = split_payload.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n & 1 != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "split payload must have even length; got {n}"
        )));
    }
    let half = n / 2;
    let (hi_packed, lo_packed) = split_payload.split_at(half);
    // Unpack hi_packed into per-byte hi nibbles (length n).
    let mut hi_nibbles = Vec::with_capacity(n);
    for &b in hi_packed {
        hi_nibbles.push((b >> 4) & 0xF);
        hi_nibbles.push(b & 0xF);
    }
    // Unpack lo_packed similarly.
    let mut lo_nibbles = Vec::with_capacity(n);
    for &b in lo_packed {
        lo_nibbles.push((b >> 4) & 0xF);
        lo_nibbles.push(b & 0xF);
    }
    debug_assert_eq!(hi_nibbles.len(), n);
    debug_assert_eq!(lo_nibbles.len(), n);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((hi_nibbles[i] << 4) | lo_nibbles[i]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_parse_qm0() {
        let h = emit_qmqh_header(false);
        assert_eq!(h, MAGIC_QM0);
        let parsed = parse_qmqh_header(&h).unwrap();
        assert_eq!(parsed.magic, MAGIC_QM0);
        assert!(!parsed.hilo_split);
        assert_eq!(parsed.body_offset, 3);
    }

    #[test]
    fn emit_parse_qh0() {
        let h = emit_qmqh_header(true);
        assert_eq!(h, MAGIC_QH0);
        let parsed = parse_qmqh_header(&h).unwrap();
        assert_eq!(parsed.magic, MAGIC_QH0);
        assert!(parsed.hilo_split);
        assert_eq!(parsed.body_offset, 3);
    }

    #[test]
    fn parse_rejects_too_short() {
        assert!(parse_qmqh_header(&[]).is_err());
        assert!(parse_qmqh_header(b"QM").is_err());
    }

    #[test]
    fn parse_rejects_unknown_magic() {
        assert!(parse_qmqh_header(b"XYZ").is_err());
    }

    #[test]
    fn hilo_split_roundtrip_4_bytes() {
        // packed_nibbles = 4 bytes = 8 nibbles total
        // hi-nibbles: [0, 2, 4, 6]; lo-nibbles: [1, 3, 5, 7]
        let input = vec![0x01u8, 0x23, 0x45, 0x67];
        let split = pack_hi_lo_split(&input).unwrap();
        // hi_packed = [(0<<4)|2, (4<<4)|6] = [0x02, 0x46]
        // lo_packed = [(1<<4)|3, (5<<4)|7] = [0x13, 0x57]
        assert_eq!(split, vec![0x02, 0x46, 0x13, 0x57]);
        let recovered = unpack_hi_lo_split(&split).unwrap();
        assert_eq!(recovered, input);
    }

    #[test]
    fn hilo_split_empty() {
        assert!(pack_hi_lo_split(&[]).unwrap().is_empty());
        assert!(unpack_hi_lo_split(&[]).unwrap().is_empty());
    }

    #[test]
    fn hilo_split_rejects_odd() {
        let input = vec![0x01u8, 0x23, 0x45];
        assert!(pack_hi_lo_split(&input).is_err());
    }

    #[test]
    fn hilo_split_roundtrip_64_bytes() {
        let input: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let split = pack_hi_lo_split(&input).unwrap();
        assert_eq!(split.len(), 64);
        let recovered = unpack_hi_lo_split(&split).unwrap();
        assert_eq!(recovered, input);
    }
}
