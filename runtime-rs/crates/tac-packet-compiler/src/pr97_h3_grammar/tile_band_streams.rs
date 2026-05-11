//! PR97 tile-band multi-stream wire-format grammar.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`tac.packet_compiler.pr97_h3_grammar::encode_tile_band_streams`) emits
//! a `u32 LE n_chunks` global prefix followed by `[u32 LE size_i][bytes_i]`
//! pairs.
//!
//! PR97 uses this for 4 horizontal bands × per-band W-splits = 22 streams,
//! one per (band, W-split) tile, where each stream is range-coded mask data
//! for that tile. The tile-reassembly logic is PR97-specific and stays out
//! of this primitive — this module only handles the wire format.
//!
//! Wire format:
//!
//! ```text
//! n_chunks       u32 LE
//! stream_0_len   u32 LE
//! stream_0_body  bytes
//! stream_1_len   u32 LE
//! stream_1_body  bytes
//! …
//! ```

use crate::{PacketCompilerError, Result};

/// Parsed tile-band multi-stream wire-format payload.
///
/// Mirrors `tac.packet_compiler.TileBandStreamPayload`.
#[derive(Debug, Clone)]
pub struct TileBandStreamPayload {
    /// The decoded per-tile streams (in input order).
    pub streams: Vec<Vec<u8>>,
    /// Chunk count parsed from the leading `u32 LE` prefix.
    pub n_chunks: u32,
    /// Total bytes consumed from the input blob.
    pub total_bytes: usize,
}

/// Pack N per-tile streams into a tile-band wire-format blob.
///
/// Mirrors `tac.packet_compiler.encode_tile_band_streams`.
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if the stream count exceeds
///   `u32::MAX` or any per-stream body exceeds `u32::MAX` bytes (the wire
///   length prefixes are `u32`).
pub fn encode_tile_band_streams(streams: &[&[u8]]) -> Result<Vec<u8>> {
    if streams.len() > u32::MAX as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "tile-band stream count {} too large; must fit in u32",
            streams.len()
        )));
    }
    let total: usize = 4 + streams
        .iter()
        .map(|s| 4usize.saturating_add(s.len()))
        .sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(streams.len() as u32).to_le_bytes());
    for (i, s) in streams.iter().enumerate() {
        if s.len() > u32::MAX as usize {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "tile-band stream {i} too large ({} bytes); must fit in u32",
                s.len()
            )));
        }
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s);
    }
    Ok(out)
}

/// Decode a tile-band multi-stream wire-format payload.
///
/// Mirrors `tac.packet_compiler.decode_tile_band_streams`. If
/// `expected_n_chunks` is given, the decoded chunk count must match or a
/// hard error is returned. Trailing bytes also raise.
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if the header is truncated,
///   the chunk count mismatches `expected_n_chunks`, any per-stream length
///   prefix is truncated, any per-stream body is truncated, or trailing
///   bytes remain after the last consumed stream.
pub fn decode_tile_band_streams(
    blob: &[u8],
    expected_n_chunks: Option<u32>,
) -> Result<TileBandStreamPayload> {
    if blob.len() < 4 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "tile-band truncated header: need 4 bytes have {}",
            blob.len()
        )));
    }
    let n_chunks = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
    if let Some(expected) = expected_n_chunks {
        if n_chunks != expected {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "tile-band chunk count mismatch: {n_chunks} != expected {expected}"
            )));
        }
    }
    let mut o = 4usize;
    let mut streams: Vec<Vec<u8>> = Vec::with_capacity(n_chunks as usize);
    for i in 0..n_chunks {
        if o + 4 > blob.len() {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "tile-band truncated stream-{i} length prefix at offset {o}"
            )));
        }
        let sz =
            u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]) as usize;
        o += 4;
        if o + sz > blob.len() {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "tile-band truncated stream-{i} body at offset {o}: need {sz} bytes have {}",
                blob.len() - o
            )));
        }
        streams.push(blob[o..o + sz].to_vec());
        o += sz;
    }
    if o != blob.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "tile-band payload trailing bytes: {o} consumed vs {} total",
            blob.len()
        )));
    }
    Ok(TileBandStreamPayload {
        streams,
        n_chunks,
        total_bytes: blob.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let streams: Vec<&[u8]> = vec![b"a", b"bb", b"ccc"];
        let blob = encode_tile_band_streams(&streams).unwrap();
        // 4 (n_chunks) + 4+1 + 4+2 + 4+3 = 22 bytes total.
        assert_eq!(blob.len(), 22);
        let payload = decode_tile_band_streams(&blob, Some(3)).unwrap();
        assert_eq!(payload.n_chunks, 3);
        assert_eq!(payload.streams[0], b"a");
        assert_eq!(payload.streams[1], b"bb");
        assert_eq!(payload.streams[2], b"ccc");
    }

    #[test]
    fn encode_empty_stream_list_emits_zero_count() {
        let blob = encode_tile_band_streams(&[]).unwrap();
        assert_eq!(blob, vec![0u8, 0, 0, 0]);
        let payload = decode_tile_band_streams(&blob, Some(0)).unwrap();
        assert_eq!(payload.n_chunks, 0);
        assert!(payload.streams.is_empty());
    }

    #[test]
    fn decode_rejects_truncated_header() {
        let bad = [0u8, 0, 0]; // only 3 bytes
        assert!(decode_tile_band_streams(&bad, None).is_err());
    }

    #[test]
    fn decode_rejects_chunk_count_mismatch() {
        let blob = encode_tile_band_streams(&[b"x" as &[u8]]).unwrap();
        assert!(decode_tile_band_streams(&blob, Some(5)).is_err());
    }

    #[test]
    fn decode_rejects_truncated_stream() {
        // 1 chunk claimed, length=10, but only 2 body bytes given.
        let blob = [1u8, 0, 0, 0, 10, 0, 0, 0, b'a', b'b'];
        assert!(decode_tile_band_streams(&blob, None).is_err());
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let streams: Vec<&[u8]> = vec![b"x"];
        let mut blob = encode_tile_band_streams(&streams).unwrap();
        blob.push(0xFF);
        assert!(decode_tile_band_streams(&blob, None).is_err());
    }

    #[test]
    fn golden_layout_22_chunks_growing_size() {
        // Recipe from src/tac/tests/test_packet_compiler_pr97_h3_grammar.py:
        //   streams = [bytes([i]) * (i + 1) for i in range(22)]
        let bufs: Vec<Vec<u8>> = (0..22u8).map(|i| vec![i; (i + 1) as usize]).collect();
        let streams: Vec<&[u8]> = bufs.iter().map(|v| v.as_slice()).collect();
        let blob = encode_tile_band_streams(&streams).unwrap();
        // n_chunks (4) + 22 length prefixes (4 each) + sum(1..=22) bodies
        // = 4 + 88 + 253 = 345
        assert_eq!(blob.len(), 345);
        // Check header.
        assert_eq!(&blob[0..4], &[22, 0, 0, 0]);
        // Check first stream length prefix + body.
        assert_eq!(&blob[4..8], &[1, 0, 0, 0]);
        assert_eq!(blob[8], 0);
        // Round-trip.
        let payload = decode_tile_band_streams(&blob, Some(22)).unwrap();
        for (i, s) in payload.streams.iter().enumerate() {
            assert_eq!(s.len(), i + 1);
            assert!(s.iter().all(|&b| b == i as u8));
        }
    }
}
