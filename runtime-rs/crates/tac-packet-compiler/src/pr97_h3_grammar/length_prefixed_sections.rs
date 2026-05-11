//! PR97 length-prefixed multi-section payload grammar.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`tac.packet_compiler.pr97_h3_grammar::encode_length_prefixed_sections`)
//! emits `[u32 LE len_i][bytes_i]` pairs concatenated with NO global
//! section count prefix. The consumer declares `min_sections` and an
//! optional `max_sections` cap.
//!
//! Wire format:
//!
//! ```text
//! section_0_len  u32 LE
//! section_0_body bytes
//! section_1_len  u32 LE
//! section_1_body bytes
//! …
//! ```

use crate::{PacketCompilerError, Result};

/// Parsed length-prefixed multi-section payload.
///
/// Mirrors `tac.packet_compiler.LengthPrefixedSectionPayload`.
#[derive(Debug, Clone)]
pub struct LengthPrefixedSectionPayload {
    /// The decoded section bodies (in input order).
    pub sections: Vec<Vec<u8>>,
    /// Length of the consumed blob (informational; equals the encoded
    /// payload length when the decoder consumed the slice cleanly).
    pub total_bytes: usize,
}

/// Pack N byte sections into a single blob via `[u32 len_i][bytes_i]` pairs.
///
/// Mirrors `tac.packet_compiler.encode_length_prefixed_sections`.
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if any section is longer than
///   `u32::MAX` bytes (the wire length prefix is a `u32`).
pub fn encode_length_prefixed_sections(sections: &[&[u8]]) -> Result<Vec<u8>> {
    let total: usize = sections
        .iter()
        .map(|s| 4usize.saturating_add(s.len()))
        .sum();
    let mut out = Vec::with_capacity(total);
    for (i, s) in sections.iter().enumerate() {
        if s.len() > u32::MAX as usize {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "length-prefixed section {i} too large ({} bytes); must fit in u32",
                s.len()
            )));
        }
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s);
    }
    Ok(out)
}

/// Decode a length-prefixed multi-section payload.
///
/// Mirrors `tac.packet_compiler.decode_length_prefixed_sections`. The
/// consumer declares `min_sections` (required) and `max_sections`
/// (optional cap). Any sections beyond `min_sections` and up to
/// `max_sections` (or until the blob is exhausted) are returned. Trailing
/// bytes after the last consumed section trigger a hard error so silent
/// corruption is impossible.
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if `max_sections < min_sections`,
///   a length prefix is truncated, a body is truncated, fewer than
///   `min_sections` sections are decoded, or trailing bytes remain.
pub fn decode_length_prefixed_sections(
    blob: &[u8],
    min_sections: usize,
    max_sections: Option<usize>,
) -> Result<LengthPrefixedSectionPayload> {
    if let Some(cap) = max_sections {
        if cap < min_sections {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "max_sections {cap} cannot be less than min_sections {min_sections}"
            )));
        }
    }
    let mut sections: Vec<Vec<u8>> = Vec::new();
    let mut o = 0usize;
    while o < blob.len() {
        if let Some(cap) = max_sections {
            if sections.len() >= cap {
                break;
            }
        }
        if o + 4 > blob.len() {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "length-prefixed truncated length prefix at offset {o}: need 4 bytes have {}",
                blob.len() - o
            )));
        }
        let n = u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]) as usize;
        o += 4;
        if o + n > blob.len() {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "length-prefixed truncated section {} at offset {o}: need {n} bytes have {}",
                sections.len(),
                blob.len() - o
            )));
        }
        sections.push(blob[o..o + n].to_vec());
        o += n;
    }
    if sections.len() < min_sections {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "length-prefixed only decoded {} sections, required min {min_sections}",
            sections.len()
        )));
    }
    if o != blob.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "length-prefixed payload trailing bytes: {o} consumed vs {} total",
            blob.len()
        )));
    }
    Ok(LengthPrefixedSectionPayload {
        sections,
        total_bytes: blob.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let sections: Vec<&[u8]> = vec![b"abc", b"defgh", b""];
        let blob = encode_length_prefixed_sections(&sections).unwrap();
        // 4 + 3 + 4 + 5 + 4 + 0 = 20 bytes total.
        assert_eq!(blob.len(), 20);
        let payload = decode_length_prefixed_sections(&blob, 3, None).unwrap();
        assert_eq!(payload.sections.len(), 3);
        assert_eq!(payload.sections[0], b"abc");
        assert_eq!(payload.sections[1], b"defgh");
        assert_eq!(payload.sections[2], b"");
        assert_eq!(payload.total_bytes, 20);
    }

    #[test]
    fn encode_empty_section_list() {
        let blob = encode_length_prefixed_sections(&[]).unwrap();
        assert!(blob.is_empty());
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let sections: Vec<&[u8]> = vec![b"abc"];
        let mut blob = encode_length_prefixed_sections(&sections).unwrap();
        blob.push(0xFF);
        let r = decode_length_prefixed_sections(&blob, 1, None);
        assert!(r.is_err());
    }

    #[test]
    fn decode_rejects_truncated_length_prefix() {
        // 3 bytes — not enough for a u32 length prefix.
        let blob = [1u8, 0, 0];
        assert!(decode_length_prefixed_sections(&blob, 1, None).is_err());
    }

    #[test]
    fn decode_rejects_truncated_body() {
        // Length prefix claims 100 bytes but only 4 follow.
        let blob = [100u8, 0, 0, 0, 0, 0, 0, 0];
        assert!(decode_length_prefixed_sections(&blob, 1, None).is_err());
    }

    #[test]
    fn decode_rejects_too_few_sections() {
        let sections: Vec<&[u8]> = vec![b"a"];
        let blob = encode_length_prefixed_sections(&sections).unwrap();
        assert!(decode_length_prefixed_sections(&blob, 3, None).is_err());
    }

    #[test]
    fn decode_respects_max_sections_cap() {
        // Three sections in the blob, but cap = 2 → decoded == 2 + trailing
        // bytes from the third → trailing-bytes error.
        let sections: Vec<&[u8]> = vec![b"a", b"b", b"c"];
        let blob = encode_length_prefixed_sections(&sections).unwrap();
        let r = decode_length_prefixed_sections(&blob, 2, Some(2));
        // Cap stops decode at 2; the unconsumed bytes for section 3 raise.
        assert!(r.is_err());
    }

    #[test]
    fn decode_rejects_max_lt_min() {
        let blob = vec![];
        assert!(decode_length_prefixed_sections(&blob, 5, Some(2)).is_err());
    }

    #[test]
    fn golden_layout_4_sections() {
        // Recipe from src/tac/tests/test_packet_compiler_pr97_h3_grammar.py:
        //   sections = [b"\x00" * 16, b"\x01" * 32, b"\x02" * 8, b"\x03" * 4];
        // Wire-format-byte spot-check (without SHA — that's the parity test).
        let s0 = vec![0u8; 16];
        let s1 = vec![1u8; 32];
        let s2 = vec![2u8; 8];
        let s3 = vec![3u8; 4];
        let sections: Vec<&[u8]> = vec![&s0, &s1, &s2, &s3];
        let blob = encode_length_prefixed_sections(&sections).unwrap();
        // Expected layout:
        // [16 LE u32][16 0x00] [32 LE u32][32 0x01] [8 LE u32][8 0x02] [4 LE u32][4 0x03]
        // = 4+16 + 4+32 + 4+8 + 4+4 = 76 bytes
        assert_eq!(blob.len(), 76);
        assert_eq!(&blob[0..4], &[16, 0, 0, 0]);
        assert_eq!(blob[4], 0);
        assert_eq!(&blob[20..24], &[32, 0, 0, 0]);
        assert_eq!(blob[24], 1);
        assert_eq!(&blob[56..60], &[8, 0, 0, 0]);
        assert_eq!(blob[60], 2);
        assert_eq!(&blob[68..72], &[4, 0, 0, 0]);
        assert_eq!(blob[72], 3);
    }
}
