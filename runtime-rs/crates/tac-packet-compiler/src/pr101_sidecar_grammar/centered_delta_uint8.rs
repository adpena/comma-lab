//! Rust port of `encode_centered_delta_uint8` / `decode_centered_delta_uint8`.
//!
//! # Byte-for-byte parity contract
//!
//! The Python oracle (`src/tac/packet_compiler/pr101_sidecar_grammar.py`)
//! serialises the pre-LZMA payload as:
//!
//! ```text
//! [mins fp16 LE | scales fp16 LE | column_major uint8]
//! ```
//!
//! and compresses with raw LZMA1 (`format=FORMAT_RAW`,
//! `filters=[{id=FILTER_LZMA1, dict_size=4096, lc=3, lp=0, pb=0}]`). CPython's
//! `_lzmamodule.c::parse_filter_spec_lzma` seeds the options struct via
//! `lzma_lzma_preset(opts, LZMA_PRESET_DEFAULT=6)` before applying the named
//! fields; the Rust port mirrors that via
//! `LzmaOptions::new_preset(6).dict_size(4096).literal_context_bits(3).literal_position_bits(0).position_bits(0)`.
//!
//! # Endianness
//!
//! Per Selfcomp's gotcha (recorded in the N D4 verdict) and the Python
//! oracle's `np.float16` / `np.uint8` `tobytes()` semantics: fp16 mins/scales
//! are little-endian; the column-major uint8 block has no multi-byte words.
//! The LZMA bit-stream itself is opaque (the parity SHA-256 is the
//! authority).

use std::io::Write;

use half::f16;
use liblzma::stream::{Action, Filters, LzmaOptions, Status, Stream};

use crate::{PacketCompilerError, Result};

use super::stubs::CenteredDeltaUint8Stream;

/// PR101's raw-LZMA1 filter chain: dict 4 KiB, lc=3, lp=0, pb=0, preset(6).
fn build_filters() -> Result<Filters> {
    let mut opts = LzmaOptions::new_preset(6).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("lzma preset(6) init failed: {e:?}"))
    })?;
    opts.dict_size(4096)
        .literal_context_bits(3)
        .literal_position_bits(0)
        .position_bits(0);
    let mut filters = Filters::new();
    filters.lzma1(&opts);
    Ok(filters)
}

/// Raw-LZMA1 compress (Stream::new_raw_encoder; no XZ container).
fn lzma1_raw_compress(input: &[u8]) -> Result<Vec<u8>> {
    let filters = build_filters()?;
    let mut stream = Stream::new_raw_encoder(&filters).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("lzma_raw_encoder init failed: {e:?}"))
    })?;
    let mut output = Vec::with_capacity(input.len() + 64);
    let mut input_offset = 0;
    let mut tmp = [0u8; 16384];
    loop {
        let action = if input_offset == input.len() {
            Action::Finish
        } else {
            Action::Run
        };
        let before_in = stream.total_in();
        let before_out = stream.total_out();
        let status = stream
            .process(&input[input_offset..], &mut tmp[..], action)
            .map_err(|e| {
                PacketCompilerError::GoldenVectorIo(format!("lzma encode step failed: {e:?}"))
            })?;
        let consumed = (stream.total_in() - before_in) as usize;
        let produced = (stream.total_out() - before_out) as usize;
        input_offset += consumed;
        output.extend_from_slice(&tmp[..produced]);
        if matches!(status, Status::StreamEnd) {
            break;
        }
        if consumed == 0 && produced == 0 && input_offset == input.len() {
            // Avoid an infinite loop if the encoder neither consumes nor
            // produces and we have nothing left to feed. Defensive guard
            // (this should not happen in practice once Finish is signalled).
            return Err(PacketCompilerError::GoldenVectorIo(
                "lzma encode stalled with no input consumed and no output produced".into(),
            ));
        }
    }
    Ok(output)
}

/// Raw-LZMA1 decompress (Stream::new_raw_decoder).
fn lzma1_raw_decompress(input: &[u8], hint_capacity: usize) -> Result<Vec<u8>> {
    let filters = build_filters()?;
    let mut stream = Stream::new_raw_decoder(&filters).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("lzma_raw_decoder init failed: {e:?}"))
    })?;
    let mut output = Vec::with_capacity(hint_capacity.max(input.len()));
    let mut input_offset = 0;
    let mut tmp = [0u8; 16384];
    loop {
        let action = if input_offset == input.len() {
            Action::Finish
        } else {
            Action::Run
        };
        let before_in = stream.total_in();
        let before_out = stream.total_out();
        let status = stream
            .process(&input[input_offset..], &mut tmp[..], action)
            .map_err(|e| {
                PacketCompilerError::GoldenVectorIo(format!("lzma decode step failed: {e:?}"))
            })?;
        let consumed = (stream.total_in() - before_in) as usize;
        let produced = (stream.total_out() - before_out) as usize;
        input_offset += consumed;
        output.extend_from_slice(&tmp[..produced]);
        if matches!(status, Status::StreamEnd) {
            break;
        }
        if consumed == 0 && produced == 0 {
            // Same defensive guard as the encoder. Raw LZMA1 has no explicit
            // EOS marker, so the encoder side controls when StreamEnd fires.
            break;
        }
    }
    Ok(output)
}

/// Encode a per-column quantised stream as centered-delta uint8 under raw LZMA.
///
/// Mirrors `tac.packet_compiler.encode_centered_delta_uint8`. `values` is
/// **row-major** `(n_pairs, n_dims)` `f32`. When `mins` / `scales` are
/// `None`, per-column fp16 calibration is derived from the data exactly as
/// Python does it.
///
/// Byte-for-byte parity target: see
/// `src/tac/packet_compiler/golden_vectors/centered_delta_uint8_v1.json`.
pub fn encode_centered_delta_uint8(
    values: &[f32],
    n_pairs: usize,
    n_dims: usize,
    mins: Option<&[u8]>,
    scales: Option<&[u8]>,
) -> Result<CenteredDeltaUint8Stream> {
    if n_pairs < 1 || n_dims < 1 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "values must have positive shape; got ({n_pairs}, {n_dims})"
        )));
    }
    if values.len() != n_pairs * n_dims {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "values len {} != n_pairs*n_dims={}",
            values.len(),
            n_pairs * n_dims
        )));
    }

    // mins / scales as fp16 little-endian bytes (12 bytes for n_dims=6).
    let mins_bytes = match mins {
        Some(m) => {
            if m.len() != n_dims * 2 {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "mins len {} != n_dims*2={}",
                    m.len(),
                    n_dims * 2
                )));
            }
            m.to_vec()
        }
        None => {
            // Per-column min as fp16.
            let mut out = Vec::with_capacity(n_dims * 2);
            for col in 0..n_dims {
                let mut col_min = f32::INFINITY;
                for row in 0..n_pairs {
                    let v = values[row * n_dims + col];
                    if v < col_min {
                        col_min = v;
                    }
                }
                let h = f16::from_f32(col_min);
                out.extend_from_slice(&h.to_le_bytes());
            }
            out
        }
    };
    let scales_bytes = match scales {
        Some(s) => {
            if s.len() != n_dims * 2 {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "scales len {} != n_dims*2={}",
                    s.len(),
                    n_dims * 2
                )));
            }
            s.to_vec()
        }
        None => {
            // Per-column scale = (max - min) / 255 as fp16; guard against
            // zero-width columns.
            let mut out = Vec::with_capacity(n_dims * 2);
            for col in 0..n_dims {
                let mut col_min = f32::INFINITY;
                let mut col_max = f32::NEG_INFINITY;
                for row in 0..n_pairs {
                    let v = values[row * n_dims + col];
                    if v < col_min {
                        col_min = v;
                    }
                    if v > col_max {
                        col_max = v;
                    }
                }
                // Python casts both endpoints to fp16 *first* before computing
                // the diff (col_max.astype(np.float16) etc.) and then back to
                // fp32 for the divide. We mirror that exactly.
                let col_min_fp16 = f16::from_f32(col_min).to_f32();
                let col_max_fp16 = f16::from_f32(col_max).to_f32();
                let diff = col_max_fp16 - col_min_fp16;
                let diff = if diff <= 0.0 { 1.0 } else { diff };
                let scale = f16::from_f32(diff / 255.0);
                out.extend_from_slice(&scale.to_le_bytes());
            }
            out
        }
    };

    // Recover fp32 mins / scales from the byte buffers so quantisation uses
    // the same fp16-rounded values Python uses.
    let mins_f32 = decode_fp16_le(&mins_bytes);
    let scales_f32 = decode_fp16_le(&scales_bytes);
    for (col, s) in scales_f32.iter().enumerate() {
        if *s == 0.0 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "scales must be non-zero per column; column {col} has scale 0"
            )));
        }
    }

    // Quantise + clamp to uint8. Row-major.
    let mut q = vec![0u8; n_pairs * n_dims];
    for row in 0..n_pairs {
        for col in 0..n_dims {
            let v = values[row * n_dims + col];
            let raw = ((v - mins_f32[col]) / scales_f32[col]).round() as i64;
            let clamped = raw.clamp(0, 255) as u8;
            q[row * n_dims + col] = clamped;
        }
    }

    // Base row + centered temporal deltas (mod 256 - 128 ⇒ `(diff + 128) & 0xFF`).
    let mut base = vec![0u8; n_dims];
    base.copy_from_slice(&q[..n_dims]);
    // Deltas: shape (n_pairs-1, n_dims) row-major.
    let mut deltas_row_major = vec![0u8; (n_pairs - 1) * n_dims];
    for row in 1..n_pairs {
        for col in 0..n_dims {
            let prev = q[(row - 1) * n_dims + col] as i32;
            let cur = q[row * n_dims + col] as i32;
            let diff = cur - prev;
            // Python: ((deltas_int + 128) & 0xFF); centered at 128.
            let centered = ((diff + 128) & 0xFF) as u8;
            deltas_row_major[(row - 1) * n_dims + col] = centered;
        }
    }

    // Column-major block: for each col, emit [base[col], delta_row_0[col], …,
    // delta_row_{n_pairs-2}[col]]. Size = n_pairs * n_dims.
    let mut column_major = vec![0u8; n_pairs * n_dims];
    for col in 0..n_dims {
        column_major[col * n_pairs] = base[col];
        for row in 0..(n_pairs - 1) {
            column_major[col * n_pairs + 1 + row] = deltas_row_major[row * n_dims + col];
        }
    }

    // Compose the pre-LZMA buffer: mins | scales | column_major.
    let mut raw = Vec::with_capacity(n_dims * 4 + n_pairs * n_dims);
    raw.write_all(&mins_bytes).expect("write mins");
    raw.write_all(&scales_bytes).expect("write scales");
    raw.write_all(&column_major).expect("write column_major");

    let lzma_bytes = lzma1_raw_compress(&raw)?;

    Ok(CenteredDeltaUint8Stream {
        mins: mins_bytes,
        scales: scales_bytes,
        base,
        deltas: deltas_row_major,
        n_pairs,
        n_dims,
        lzma_bytes,
    })
}

/// Inverse of [`encode_centered_delta_uint8`]; reconstructs the row-major
/// `(n_pairs, n_dims)` float32 buffer from the raw LZMA bytes.
pub fn decode_centered_delta_uint8(
    lzma_bytes: &[u8],
    n_pairs: usize,
    n_dims: usize,
) -> Result<Vec<f32>> {
    if n_pairs < 1 || n_dims < 1 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "decode shape must be positive; got ({n_pairs}, {n_dims})"
        )));
    }
    let expected = n_dims * 2 + n_dims * 2 + n_pairs * n_dims;
    let raw = lzma1_raw_decompress(lzma_bytes, expected)?;
    if raw.len() != expected {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "decoded raw length {} != expected {}",
            raw.len(),
            expected
        )));
    }
    let mins_bytes = &raw[0..n_dims * 2];
    let scales_bytes = &raw[n_dims * 2..2 * n_dims * 2];
    let cm = &raw[2 * n_dims * 2..];

    let mins_f32 = decode_fp16_le(mins_bytes);
    let scales_f32 = decode_fp16_le(scales_bytes);

    // Reconstruct q (row-major n_pairs x n_dims) from column-major centered deltas.
    let mut q = vec![0u8; n_pairs * n_dims];
    for col in 0..n_dims {
        let col_base_idx = col * n_pairs;
        let mut acc = cm[col_base_idx];
        q[col] = acc;
        for row in 1..n_pairs {
            let step = cm[col_base_idx + row] as i32 - 128;
            acc = ((acc as i32 + step) & 0xFF) as u8;
            q[row * n_dims + col] = acc;
        }
    }

    let mut out = vec![0.0f32; n_pairs * n_dims];
    for row in 0..n_pairs {
        for col in 0..n_dims {
            out[row * n_dims + col] =
                q[row * n_dims + col] as f32 * scales_f32[col] + mins_f32[col];
        }
    }
    Ok(out)
}

fn decode_fp16_le(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0);
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let h = f16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(h.to_f32());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_roundtrip_matches_python_linspace() {
        // np.linspace(-1.0, -0.5, 6, dtype=np.float16):
        // [-1.0, -0.89990234375, -0.7998046875, -0.7001953125, -0.60009765625, -0.5]
        let mins: Vec<f32> = (0..6)
            .map(|i| f16::from_f32(-1.0 + (i as f32) * 0.5 / 5.0).to_f32())
            .collect();
        let expected = [
            -1.0_f32,
            -0.899_902_34,
            -0.799_804_7,
            -0.700_195_3,
            -0.600_097_66,
            -0.5,
        ];
        for (a, b) in mins.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_smoke() {
        // 4×3 deterministic float32 values; encode→decode→assert close.
        let values: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05).collect();
        let stream = encode_centered_delta_uint8(&values, 4, 3, None, None)
            .expect("encode must succeed");
        assert!(!stream.lzma_bytes.is_empty());
        let recovered = decode_centered_delta_uint8(&stream.lzma_bytes, 4, 3)
            .expect("decode must succeed");
        assert_eq!(recovered.len(), values.len());
        // Quantisation tolerance: at most ~2 quanta where scale = (max-min)/255.
        // Range per column is small (~0.05*3 = 0.15), so quanta ≈ 0.0006.
        for (a, b) in recovered.iter().zip(values.iter()) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }
}
