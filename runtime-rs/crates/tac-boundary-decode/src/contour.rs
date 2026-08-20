// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dense-raster LZMA DECODE — native port of
//! `tac.boundary_math.dense_raster_lzma_baseline.decode_partition`.
//!
//! The Python oracle stores the SegNet argmax partition `L*` as a RAW-LZMA2
//! stream of the per-pixel `uint8` class labels (raster order). Interior pixels
//! are often constant-label runs, so LZMA2 compresses them well. Decode is
//! `lzma.decompress(payload, FORMAT_RAW, [LZMA2 preset 9|EXTREME, lc=0,lp=0,pb=0])`
//! → reshape `(H, W)`.
//!
//! This Rust port feeds the same payload bytes to the SAME liblzma C library
//! (via the `liblzma` crate) with the matching RAW-LZMA2 filter spec and yields
//! the bit-identical raw label-byte stream. The decoded-raw SHA-256 must match
//! the Python oracle (`contour_decode_{full,small}_v1` golden vectors).
//!
//! PAYLOAD CLEANLINESS: this is FIXED decode code. The label map (the partition
//! the SegNet derives masks from) is carried in `archive.zip` as the LZMA
//! payload, NOT embedded here. The only constants are the LZMA filter params
//! (structural codec config, not learned data).

use liblzma::stream::{Action, Filters, LzmaOptions, Status, Stream, PRESET_EXTREME};

use crate::{BoundaryDecodeError, Result};

/// LZMA preset level the Python oracle uses: `9 | PRESET_EXTREME`.
///
/// For RAW decode, the decoder must use the same `dict_size` (set by the preset)
/// and the explicit `lc`/`lp`/`pb` overrides as the encoder. These are the codec
/// config from `dense_raster_lzma_baseline._LZMA_FILTERS`, NOT learned/video data.
const LZMA_PRESET_LEVEL_9: u32 = 9;
const LZMA_LC: u32 = 0;
const LZMA_LP: u32 = 0;
const LZMA_PB: u32 = 0;

/// Build the RAW-LZMA2 filter chain that mirrors `dense_raster_lzma_baseline._LZMA_FILTERS`.
fn contour_filters() -> Result<Filters> {
    let mut opts = LzmaOptions::new_preset(LZMA_PRESET_LEVEL_9 | PRESET_EXTREME)
        .map_err(|e| BoundaryDecodeError::LzmaDecode(format!("preset init: {e:?}")))?;
    opts.literal_context_bits(LZMA_LC)
        .literal_position_bits(LZMA_LP)
        .position_bits(LZMA_PB);
    let mut filters = Filters::new();
    filters.lzma2(&opts);
    Ok(filters)
}

/// Decode a RAW-LZMA2 dense-label payload into the raw `uint8` label bytes.
///
/// `expected_len = height * width` is the decoded byte count the `(H, W)` shape
/// implies; the decode fails closed if the produced length differs (shape lie).
/// Returns the raster-order `uint8` class labels — byte-for-byte identical to
/// `np.asarray(decode_partition(code)).astype(np.uint8).tobytes()`.
pub fn decode_partition_raw(payload: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let filters = contour_filters()?;
    let mut stream = Stream::new_raw_decoder(&filters)
        .map_err(|e| BoundaryDecodeError::LzmaDecode(format!("raw decoder init: {e:?}")))?;

    // Allocate the exact output capacity the shape implies, plus a small slack so
    // `process_vec` never has to grow mid-stream (RAW decode has no length frame,
    // so we drive it to StreamEnd / input exhaustion explicitly).
    let mut out: Vec<u8> = Vec::with_capacity(expected_len + 64);
    let status = stream
        .process_vec(payload, &mut out, Action::Finish)
        .map_err(|e| BoundaryDecodeError::LzmaDecode(format!("process: {e:?}")))?;

    // RAW LZMA2 has no end-of-stream marker in the same sense as .xz; the LZMA2
    // chunk framing terminates the stream, so a single Finish pass over the full
    // payload yields StreamEnd. Accept StreamEnd or a clean Ok that consumed all
    // input and produced the expected length.
    match status {
        Status::StreamEnd => {}
        Status::Ok | Status::MemNeeded | Status::GetCheck => {
            // Drive one more Finish pass with no new input to flush any tail.
            let cap = expected_len + 64 - out.len();
            if cap > 0 {
                let st2 = stream
                    .process_vec(&[], &mut out, Action::Finish)
                    .map_err(|e| BoundaryDecodeError::LzmaDecode(format!("flush: {e:?}")))?;
                if !matches!(st2, Status::StreamEnd) && out.len() != expected_len {
                    return Err(BoundaryDecodeError::LzmaDecode(format!(
                        "stream did not terminate cleanly (status {st2:?}, got {} bytes)",
                        out.len()
                    )));
                }
            }
        }
    }

    if out.len() != expected_len {
        return Err(BoundaryDecodeError::ShapeMismatch {
            expected: expected_len,
            got: out.len(),
        });
    }
    Ok(out)
}

/// Decode + reshape to `(H, W)` `u8` rows (convenience for region rasterize/fill).
///
/// The label map IS the rasterized partition: each pixel's class label, raster
/// order. This is the inflate-time "fill" — region interiors are reconstructed
/// directly from the decoded constant-label runs. Returns rows of length `width`.
pub fn decode_partition_hw(payload: &[u8], height: usize, width: usize) -> Result<Vec<Vec<u8>>> {
    let flat = decode_partition_raw(payload, height * width)?;
    Ok(flat.chunks_exact(width).map(|r| r.to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_fails_closed_on_wrong_len() {
        // A 2-byte RAW-LZMA2 payload cannot possibly decode to 1_000_000 bytes;
        // either decode errors or the length guard fires. Either way: not Ok(()).
        let bogus = vec![0u8, 1u8, 2u8];
        let res = decode_partition_raw(&bogus, 1_000_000);
        assert!(res.is_err(), "wrong-length decode must fail closed");
    }
}
