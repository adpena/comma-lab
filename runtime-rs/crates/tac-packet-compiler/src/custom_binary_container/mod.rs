//! Custom binary container format — **RESEARCH-ONLY**.
//!
//! # Status
//!
//! `score_claim=false`, `promotion_eligible=false`,
//! `ready_for_exact_eval_dispatch=false`. This module designs and prototypes a
//! non-ZIP container format optimised for the contest's specific archive
//! pattern (typically ONE member, no per-member CRC redundancy, no central
//! directory). It is **NOT** wired into the contest runtime; the contest
//! `inflate.sh` continues to expect a standard `archive.zip`. See
//! [`section_savings_vs_zip`] for the byte-savings analysis.
//!
//! # Why this is research-only
//!
//! Per CLAUDE.md "Contest vs production target modes — non-negotiable":
//!
//! > Self-compression, neural compression, on-device learning, generated
//! > decoders, Rust/Zig/C kernels, and assembly kernels are first-class
//! > optimization directions. They are contest-admissible only when charged
//! > bits changed and exact archive custody exists.
//!
//! A custom container format changes the wire envelope the contest runtime
//! observes. Promoting it requires:
//!
//! 1. Operator approval to ship a custom `inflate.sh` that parses this
//!    format alongside (or instead of) ZIP.
//! 2. Council deliberation on the contest-compliance interpretation.
//! 3. Exact CUDA + CPU auth eval on a real submission archive.
//!
//! Until all three land, this module is a **research scaffold** that
//! demonstrates the byte-savings math and provides a clean encode/decode
//! API for future evaluation.
//!
//! # Wire format
//!
//! The container is a single-stream record-oriented format:
//!
//! ```text
//! magic              : 4 bytes  = b"TACP"        (TAC Packet)
//! version            : u8       = 0x01
//! flags              : u8       = bit 0: 0 = uncompressed body, 1 = body is a sub-payload
//! n_records          : u16 LE
//! ─ per record ─
//!   name_len         : u16 LE   (record-name byte length; <= 255)
//!   name             : name_len bytes (UTF-8; e.g. "renderer.bin")
//!   body_len         : u32 LE   (record body byte length)
//!   body             : body_len bytes (RAW; the encoder MAY compress
//!                                       further outside this layer)
//! ─ trailer ─
//! container_sha256   : 32 bytes (SHA-256 over `magic || version || flags ||
//!                                n_records || records[…]`)
//! ```
//!
//! No per-record CRC (the trailer SHA-256 covers the whole stream),
//! no end-of-central-directory, no per-record local file header. For a
//! single-member archive (the contest's pattern) this saves:
//!
//! - 30-byte local file header
//! - 46-byte central directory record
//! - 22-byte end-of-central-directory record
//! - 4-byte CRC32 (we use SHA-256 over the whole stream instead, but the
//!   stream SHA-256 lands ONCE for N records vs N×CRC32 in ZIP)
//!
//! Net savings for a single-record archive: ~98 bytes ZIP overhead replaced
//! by ~40 bytes TACP overhead (magic + version + flags + n_records +
//! one (name_len + name + body_len) header + 32-byte trailer) → **~58 bytes
//! saved per archive at typical name lengths**.
//!
//! For a multi-record archive the savings grow with N (each record skips a
//! 46-byte CDR + the per-record local file header trims from 30 bytes to
//! `2 + name_len + 4` ≈ 18 bytes for a 12-byte name → ~58 bytes saved per
//! extra record).
//!
//! # Byte-parity contract
//!
//! Both the Python encoder (committed as a sibling under
//! `src/tac/packet_compiler/custom_binary_container.py`) and the Rust
//! encoder produce identical bytes for the same input. A golden-vector pair
//! is committed and exercised by the parity harness.

use sha2::{Digest, Sha256};

use crate::{PacketCompilerError, Result};

/// Magic bytes identifying the TACP container.
pub const TACP_MAGIC: [u8; 4] = *b"TACP";
/// Wire-format version. Bump on any breaking change.
pub const TACP_VERSION: u8 = 0x01;

/// One named record inside a TACP container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TacpRecord {
    /// UTF-8 record name. Length must fit in `u16` (we further cap at 255
    /// bytes via [`encode_container`] to mirror ZIP-friendly archive names).
    pub name: String,
    /// Raw record body bytes.
    pub body: Vec<u8>,
}

/// Encode a list of records into a TACP container.
///
/// The output is `magic + version + flags + n_records + per-record(name_len,
/// name, body_len, body) + sha256_trailer`.
///
/// Returns [`PacketCompilerError::GoldenVectorIo`] on impossible records
/// (name too long, body too long, record count overflow).
pub fn encode_container(records: &[TacpRecord]) -> Result<Vec<u8>> {
    if records.len() > u16::MAX as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "TACP supports at most {} records; got {}",
            u16::MAX,
            records.len()
        )));
    }
    let mut out: Vec<u8> = Vec::with_capacity(48 + records.iter().map(|r| 6 + r.name.len() + r.body.len()).sum::<usize>());
    out.extend_from_slice(&TACP_MAGIC);
    out.push(TACP_VERSION);
    out.push(0u8); // flags = 0 (no sub-payload)
    out.extend_from_slice(&(records.len() as u16).to_le_bytes());
    for r in records {
        let name_bytes = r.name.as_bytes();
        if name_bytes.len() > 255 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "record name length {} exceeds the 255-byte ZIP-friendly cap (name={:?})",
                name_bytes.len(),
                r.name
            )));
        }
        if r.body.len() > u32::MAX as usize {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "record body length {} exceeds u32::MAX",
                r.body.len()
            )));
        }
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&(r.body.len() as u32).to_le_bytes());
        out.extend_from_slice(&r.body);
    }
    // Trailer SHA-256 over the body-so-far.
    let mut hasher = Sha256::new();
    hasher.update(&out);
    let digest = hasher.finalize();
    out.extend_from_slice(&digest);
    Ok(out)
}

/// Decode a TACP container into the original record list.
///
/// Returns [`PacketCompilerError::GoldenVectorIo`] on any corruption
/// (bad magic, bad version, truncated record, trailer SHA-256 mismatch).
pub fn decode_container(blob: &[u8]) -> Result<Vec<TacpRecord>> {
    if blob.len() < 4 + 1 + 1 + 2 + 32 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "TACP blob too short: {} bytes (minimum 40)",
            blob.len()
        )));
    }
    if &blob[..4] != TACP_MAGIC {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "bad TACP magic: {:02x?}",
            &blob[..4]
        )));
    }
    if blob[4] != TACP_VERSION {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "unsupported TACP version {} (we only support {})",
            blob[4], TACP_VERSION
        )));
    }
    let flags = blob[5];
    if flags != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "unsupported TACP flags {flags:#04x} (sub-payload mode is reserved)"
        )));
    }
    // Verify trailer SHA-256 first so we don't dereference into corrupt offsets.
    let trailer_start = blob.len() - 32;
    let body = &blob[..trailer_start];
    let expected = &blob[trailer_start..];
    let mut hasher = Sha256::new();
    hasher.update(body);
    let actual = hasher.finalize();
    if actual.as_slice() != expected {
        return Err(PacketCompilerError::GoldenVectorIo(
            "TACP trailer SHA-256 mismatch (container corrupt)".into(),
        ));
    }
    let n_records = u16::from_le_bytes([blob[6], blob[7]]) as usize;
    let mut records = Vec::with_capacity(n_records);
    let mut pos = 8usize;
    for r_idx in 0..n_records {
        if pos + 2 > trailer_start {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "TACP truncated reading name_len at record {r_idx}"
            )));
        }
        let name_len = u16::from_le_bytes([blob[pos], blob[pos + 1]]) as usize;
        pos += 2;
        if pos + name_len > trailer_start {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "TACP truncated reading name at record {r_idx}"
            )));
        }
        let name = std::str::from_utf8(&blob[pos..pos + name_len])
            .map_err(|e| {
                PacketCompilerError::GoldenVectorIo(format!(
                    "TACP record {r_idx} name is not valid UTF-8: {e}"
                ))
            })?
            .to_string();
        pos += name_len;
        if pos + 4 > trailer_start {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "TACP truncated reading body_len at record {r_idx}"
            )));
        }
        let body_len = u32::from_le_bytes([
            blob[pos],
            blob[pos + 1],
            blob[pos + 2],
            blob[pos + 3],
        ]) as usize;
        pos += 4;
        if pos + body_len > trailer_start {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "TACP truncated reading body at record {r_idx}"
            )));
        }
        let body = blob[pos..pos + body_len].to_vec();
        pos += body_len;
        records.push(TacpRecord { name, body });
    }
    if pos != trailer_start {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "TACP has {} bytes of slack after records (corrupt)",
            trailer_start - pos
        )));
    }
    Ok(records)
}

/// Byte savings of TACP vs ZIP for an n-record archive with the given name
/// and body byte budgets.
///
/// Returns `(zip_overhead, tacp_overhead, savings)` in bytes. Overhead is
/// the container framing overhead **not counting record bodies** (those are
/// identical between ZIP and TACP at our compression layer).
///
/// # ZIP overhead model
///
/// Per `PKWARE APPNOTE.TXT` 4.5:
/// - 30 bytes local file header + name bytes + 0 bytes extra field
/// - 46 bytes central directory file header + name bytes + 0 bytes extra field
/// - 22 bytes end-of-central-directory (one per archive)
///
/// Each record contributes `30 + 46 + 2*name_len = 76 + 2*name_len`.
/// EOCD adds `22` once.
///
/// # TACP overhead model
///
/// Per the wire format above:
/// - Header: 4 + 1 + 1 + 2 = 8 bytes
/// - Per record: 2 + name_len + 4 = 6 + name_len
/// - Trailer: 32 bytes
///
/// Total = 8 + sum(6 + name_len) + 32 = 40 + 6*n + sum(name_len).
pub fn section_savings_vs_zip(record_names: &[&str]) -> (u64, u64, i64) {
    let n = record_names.len() as u64;
    let total_name_len: u64 = record_names.iter().map(|s| s.len() as u64).sum();
    let zip_overhead: u64 = 22 + 76 * n + 2 * total_name_len;
    let tacp_overhead: u64 = 40 + 6 * n + total_name_len;
    let savings: i64 = zip_overhead as i64 - tacp_overhead as i64;
    (zip_overhead, tacp_overhead, savings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_container_roundtrips() {
        let blob = encode_container(&[]).expect("encode empty");
        let recs = decode_container(&blob).expect("decode empty");
        assert!(recs.is_empty());
    }

    #[test]
    fn single_record_roundtrips() {
        let recs = vec![TacpRecord {
            name: "renderer.bin".to_string(),
            body: vec![1u8, 2, 3, 4, 5],
        }];
        let blob = encode_container(&recs).expect("encode");
        let decoded = decode_container(&blob).expect("decode");
        assert_eq!(decoded, recs);
    }

    #[test]
    fn multi_record_roundtrips() {
        let recs = vec![
            TacpRecord {
                name: "renderer.bin".to_string(),
                body: vec![0xAAu8; 100],
            },
            TacpRecord {
                name: "masks.mkv".to_string(),
                body: vec![0xBBu8; 200],
            },
            TacpRecord {
                name: "poses.pt".to_string(),
                body: vec![0xCCu8; 50],
            },
        ];
        let blob = encode_container(&recs).expect("encode");
        let decoded = decode_container(&blob).expect("decode");
        assert_eq!(decoded, recs);
    }

    #[test]
    fn corruption_in_body_fails_loud() {
        let recs = vec![TacpRecord {
            name: "renderer.bin".to_string(),
            body: vec![1u8, 2, 3, 4, 5],
        }];
        let mut blob = encode_container(&recs).expect("encode");
        // Flip a byte in the body region.
        let mid = blob.len() / 2;
        blob[mid] ^= 0xFF;
        let err = decode_container(&blob).expect_err("should fail trailer SHA");
        if let PacketCompilerError::GoldenVectorIo(msg) = err {
            assert!(msg.contains("SHA-256"), "wrong error msg: {msg}");
        } else {
            panic!("wrong error variant: {err:?}");
        }
    }

    #[test]
    fn bad_magic_fails_loud() {
        let mut blob = encode_container(&[]).expect("encode");
        blob[0] = b'X';
        let err = decode_container(&blob).expect_err("should fail magic");
        if let PacketCompilerError::GoldenVectorIo(msg) = err {
            assert!(msg.contains("magic"), "wrong error msg: {msg}");
        } else {
            panic!("wrong error variant: {err:?}");
        }
    }

    #[test]
    fn savings_math_known_values() {
        // Contest pattern: ONE record named "archive.zip" / "renderer.bin"
        // (~12 chars). ZIP overhead = 22 + 76 + 24 = 122 bytes; TACP
        // overhead = 40 + 6 + 12 = 58 bytes. Savings = 64 bytes.
        let (zip, tacp, savings) = section_savings_vs_zip(&["renderer.bin"]);
        assert_eq!(zip, 122);
        assert_eq!(tacp, 58);
        assert_eq!(savings, 64);
    }

    #[test]
    fn savings_grow_with_record_count() {
        // 3 records, each 12 chars: ZIP = 22 + 3*(76 + 24) = 322; TACP =
        // 40 + 3*(6 + 12) = 94. Savings = 228 bytes.
        let (zip, tacp, savings) =
            section_savings_vs_zip(&["renderer.bin", "masks.mkv___", "poses.pt____"]);
        assert_eq!(zip, 322);
        assert_eq!(tacp, 94);
        assert_eq!(savings, 228);
    }
}
