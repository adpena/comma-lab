//! Temporal-subsampled K-of-N indicator vector — Rust port.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`tac.packet_compiler.sparse_packet_ir::encode_temporal_subsampled`,
//! `serialize_temporal_subsampled`) emits the wire format below.
//!
//! # Wire format
//!
//! ```text
//! magic(4)         = b"STS1"
//! N                u32 LE
//! K                u32 LE
//! per_frame_bytes  u32 LE
//! indicator_bitmap ceil(N/8) bytes (LSB-first within each byte)
//! residuals_packed K * per_frame_bytes bytes
//! ```
//!
//! A bit set in the indicator bitmap at position `i` (LSB-first within
//! each byte) means frame `i` carries a residual; all signal-carrying
//! frames must have identical byte size (the caller zero-pads if needed).

use crate::{PacketCompilerError, Result};

/// 4-byte magic prefix for STS1 temporal-subsampled payloads.
pub const SPARSE_TEMPORAL_MAGIC: [u8; 4] = *b"STS1";

/// K-of-N temporal subsampling stream.
///
/// Mirrors `tac.packet_compiler.TemporalSubsampledResidualStream`. The
/// `N` and `K` fields are intentionally uppercase to mirror the Python
/// oracle's wire-format documentation; the `#[allow(non_snake_case)]`
/// opt-out scopes that exemption to this struct.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct TemporalSubsampledResidualStream {
    /// Packed `ceil(N/8)` bytes; bit `i` (LSB-first within each byte)
    /// indicates whether frame `i` carries signal.
    pub indicator_bitmap: Vec<u8>,
    /// Concatenated bytes of the K signal-carrying frames in original order.
    pub residuals_packed: Vec<u8>,
    /// Total frame count.
    pub N: u32,
    /// Number of signal-carrying frames (must equal popcount of bitmap).
    pub K: u32,
    /// Byte count of each signal-carrying frame's residual (uniform).
    pub per_frame_bytes: u32,
}

impl TemporalSubsampledResidualStream {
    fn validate(&self) -> Result<()> {
        if self.K > self.N {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "temporal stream K {} > N {}",
                self.K, self.N
            )));
        }
        let expected_bitmap = (self.N as usize).div_ceil(8);
        if self.indicator_bitmap.len() != expected_bitmap {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "temporal indicator_bitmap len {} != ceil(N/8) = {expected_bitmap}",
                self.indicator_bitmap.len()
            )));
        }
        // Padding bits beyond N must be zero.
        if !self.indicator_bitmap.is_empty() && self.N % 8 != 0 {
            let used_bits = self.N as usize % 8;
            let padding_mask = !((1u8 << used_bits) - 1);
            if (*self.indicator_bitmap.last().expect("non-empty checked") & padding_mask) != 0 {
                return Err(PacketCompilerError::GoldenVectorIo(
                    "temporal indicator_bitmap has non-zero padding bits beyond N".into(),
                ));
            }
        }
        let expected_packed = (self.K as u64) * (self.per_frame_bytes as u64);
        if (self.residuals_packed.len() as u64) != expected_packed {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "temporal residuals_packed len {} != K * per_frame_bytes = {expected_packed}",
                self.residuals_packed.len()
            )));
        }
        // popcount(bitmap) == K
        let popcount: u32 = self
            .indicator_bitmap
            .iter()
            .map(|&b| b.count_ones())
            .sum();
        if popcount != self.K {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "temporal indicator_bitmap popcount {popcount} != K {}",
                self.K
            )));
        }
        Ok(())
    }
}

fn pack_indicator_bitmap(indicators: &[bool]) -> Vec<u8> {
    let n = indicators.len();
    let mut out = vec![0u8; n.div_ceil(8)];
    for (i, &flag) in indicators.iter().enumerate() {
        if flag {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn unpack_indicator_bitmap(bitmap: &[u8], n: u32) -> Vec<bool> {
    let mut out = vec![false; n as usize];
    for i in 0..n as usize {
        if bitmap[i / 8] & (1 << (i % 8)) != 0 {
            out[i] = true;
        }
    }
    out
}

/// Build a [`TemporalSubsampledResidualStream`] from per-frame residuals.
///
/// Mirrors `tac.packet_compiler.encode_temporal_subsampled`. The input is
/// a slice of `Option<&[u8]>`: `Some(bytes)` for signal-carrying frames,
/// `None` for skipped frames. All `Some` entries must share the same byte
/// length (the contract is uniform-size frames; the caller zero-pads to
/// a common size).
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if frame count exceeds
///   `u32::MAX` or signal-carrying frames have non-uniform sizes.
pub fn encode_temporal_subsampled(
    per_frame_residuals: &[Option<&[u8]>],
) -> Result<TemporalSubsampledResidualStream> {
    if per_frame_residuals.len() > u32::MAX as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "temporal frame count {} exceeds u32::MAX",
            per_frame_residuals.len()
        )));
    }
    let n = per_frame_residuals.len() as u32;
    let indicators: Vec<bool> = per_frame_residuals.iter().map(|r| r.is_some()).collect();
    let bitmap = pack_indicator_bitmap(&indicators);
    let signal: Vec<&[u8]> = per_frame_residuals
        .iter()
        .filter_map(|r| r.as_ref().copied())
        .collect();
    let k = signal.len() as u32;
    let per_frame_bytes = if signal.is_empty() {
        0u32
    } else {
        let first_len = signal[0].len();
        for (i, s) in signal.iter().enumerate() {
            if s.len() != first_len {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "temporal non-uniform per_frame_bytes: frame {i} has {} bytes != first {first_len}",
                    s.len()
                )));
            }
        }
        if first_len > u32::MAX as usize {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "temporal per_frame_bytes {} exceeds u32::MAX",
                first_len
            )));
        }
        first_len as u32
    };
    let packed: Vec<u8> = signal.iter().flat_map(|s| s.iter().copied()).collect();
    let stream = TemporalSubsampledResidualStream {
        indicator_bitmap: bitmap,
        residuals_packed: packed,
        N: n,
        K: k,
        per_frame_bytes,
    };
    stream.validate()?;
    Ok(stream)
}

/// Inverse of [`encode_temporal_subsampled`].
///
/// Returns a `Vec<Option<Vec<u8>>>` of length `N`: `None` for skipped
/// frames, `Some(bytes)` for signal-carrying frames.
pub fn decode_temporal_subsampled(
    stream: &TemporalSubsampledResidualStream,
) -> Result<Vec<Option<Vec<u8>>>> {
    stream.validate()?;
    let indicators = unpack_indicator_bitmap(&stream.indicator_bitmap, stream.N);
    let mut out: Vec<Option<Vec<u8>>> = vec![None; stream.N as usize];
    if stream.K == 0 {
        return Ok(out);
    }
    let pf = stream.per_frame_bytes as usize;
    let mut k_index = 0usize;
    for (i, &flag) in indicators.iter().enumerate() {
        if !flag {
            continue;
        }
        let start = k_index * pf;
        let end = start + pf;
        out[i] = Some(stream.residuals_packed[start..end].to_vec());
        k_index += 1;
    }
    Ok(out)
}

/// Serialise a [`TemporalSubsampledResidualStream`] to wire bytes.
pub fn serialize_temporal_subsampled(
    stream: &TemporalSubsampledResidualStream,
) -> Result<Vec<u8>> {
    stream.validate()?;
    let mut out = Vec::with_capacity(
        4 + 4 + 4 + 4 + stream.indicator_bitmap.len() + stream.residuals_packed.len(),
    );
    out.extend_from_slice(&SPARSE_TEMPORAL_MAGIC);
    out.extend_from_slice(&stream.N.to_le_bytes());
    out.extend_from_slice(&stream.K.to_le_bytes());
    out.extend_from_slice(&stream.per_frame_bytes.to_le_bytes());
    out.extend_from_slice(&stream.indicator_bitmap);
    out.extend_from_slice(&stream.residuals_packed);
    Ok(out)
}

/// Inverse of [`serialize_temporal_subsampled`]. Hard errors on truncation,
/// magic mismatch, or trailing bytes.
pub fn deserialize_temporal_subsampled(
    blob: &[u8],
) -> Result<TemporalSubsampledResidualStream> {
    if blob.len() < 16 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "temporal blob too short for header: {} < 16",
            blob.len()
        )));
    }
    if blob[0..4] != SPARSE_TEMPORAL_MAGIC {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "temporal magic mismatch: got {:?} expected {:?}",
            &blob[0..4],
            SPARSE_TEMPORAL_MAGIC
        )));
    }
    let n = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
    let k = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
    let per_frame_bytes = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
    let bitmap_len = (n as usize).div_ceil(8);
    let mut pos = 16usize;
    if pos + bitmap_len > blob.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "temporal blob truncated reading bitmap: need {} bytes, have {}",
            pos + bitmap_len,
            blob.len()
        )));
    }
    let bitmap = blob[pos..pos + bitmap_len].to_vec();
    pos += bitmap_len;
    let packed_len = (k as usize) * (per_frame_bytes as usize);
    if pos + packed_len != blob.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "temporal blob length mismatch: pos+packed={} total={}",
            pos + packed_len,
            blob.len()
        )));
    }
    let packed = blob[pos..pos + packed_len].to_vec();
    let stream = TemporalSubsampledResidualStream {
        indicator_bitmap: bitmap,
        residuals_packed: packed,
        N: n,
        K: k,
        per_frame_bytes,
    };
    stream.validate()?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream_roundtrip() {
        let frames: Vec<Option<&[u8]>> = vec![None, None, None];
        let s = encode_temporal_subsampled(&frames).unwrap();
        assert_eq!(s.N, 3);
        assert_eq!(s.K, 0);
        assert_eq!(s.per_frame_bytes, 0);
        let blob = serialize_temporal_subsampled(&s).unwrap();
        let s2 = deserialize_temporal_subsampled(&blob).unwrap();
        let back = decode_temporal_subsampled(&s2).unwrap();
        assert_eq!(back.len(), 3);
        assert!(back.iter().all(|r| r.is_none()));
    }

    #[test]
    fn smoke_roundtrip_5_frames() {
        let f0: [u8; 4] = [1, 2, 3, 4];
        let f3: [u8; 4] = [5, 6, 7, 8];
        let frames: Vec<Option<&[u8]>> = vec![Some(&f0), None, None, Some(&f3), None];
        let s = encode_temporal_subsampled(&frames).unwrap();
        assert_eq!(s.N, 5);
        assert_eq!(s.K, 2);
        assert_eq!(s.per_frame_bytes, 4);
        // Bitmap layout: bits 0 and 3 set → 0b00001001 = 0x09.
        assert_eq!(s.indicator_bitmap, vec![0x09]);
        let blob = serialize_temporal_subsampled(&s).unwrap();
        let s2 = deserialize_temporal_subsampled(&blob).unwrap();
        let back = decode_temporal_subsampled(&s2).unwrap();
        assert_eq!(back[0].as_deref(), Some(&f0[..]));
        assert_eq!(back[1], None);
        assert_eq!(back[2], None);
        assert_eq!(back[3].as_deref(), Some(&f3[..]));
        assert_eq!(back[4], None);
    }

    #[test]
    fn rejects_non_uniform_frame_sizes() {
        let f0: [u8; 4] = [1, 2, 3, 4];
        let f1: [u8; 5] = [1, 2, 3, 4, 5];
        let frames: Vec<Option<&[u8]>> = vec![Some(&f0), Some(&f1)];
        assert!(encode_temporal_subsampled(&frames).is_err());
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let bad = vec![b'X', b'X', b'X', b'X', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(deserialize_temporal_subsampled(&bad).is_err());
    }

    #[test]
    fn deserialize_rejects_trailing_bytes() {
        let f0: [u8; 2] = [1, 2];
        let frames: Vec<Option<&[u8]>> = vec![Some(&f0)];
        let s = encode_temporal_subsampled(&frames).unwrap();
        let mut blob = serialize_temporal_subsampled(&s).unwrap();
        blob.push(0xFF);
        assert!(deserialize_temporal_subsampled(&blob).is_err());
    }

    #[test]
    fn validate_rejects_padding_bits_beyond_n() {
        let s = TemporalSubsampledResidualStream {
            indicator_bitmap: vec![0xFF], // bit 7 (beyond N=5) is set
            residuals_packed: vec![],
            N: 5,
            K: 5,
            per_frame_bytes: 0,
        };
        assert!(serialize_temporal_subsampled(&s).is_err());
    }

    #[test]
    fn validate_rejects_popcount_mismatch() {
        let s = TemporalSubsampledResidualStream {
            indicator_bitmap: vec![0x03], // popcount = 2
            residuals_packed: vec![],
            N: 5,
            K: 0,
            per_frame_bytes: 0,
        };
        assert!(serialize_temporal_subsampled(&s).is_err());
    }
}
