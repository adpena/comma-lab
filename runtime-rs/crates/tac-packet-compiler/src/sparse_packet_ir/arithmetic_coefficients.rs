//! Arithmetic-coded coefficient stream — Rust port.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`tac.packet_compiler.sparse_packet_ir::encode_arithmetic_coefficients`,
//! `serialize_arithmetic_coefficients`) emits the wire format below. The
//! arithmetic coder is `constriction.stream.queue.RangeEncoder` with a
//! single `constriction.stream.model.Categorical(perfect=False)` shared by
//! every symbol.
//!
//! # Wire format
//!
//! ```text
//! magic(4)          = b"SAC1"
//! n_symbols         u32 LE
//! alphabet_size     u32 LE
//! symbol_offset     i32 LE
//! histogram         alphabet_size * fp32 LE
//! word_count        u32 LE
//! encoded_bytes     word_count * u32 BE  (constriction big-endian convention)
//! ```
//!
//! # Histogram contract
//!
//! When the caller does not provide an explicit histogram, the encoder
//! builds one from the empirical symbol distribution with **+1.0 Laplace
//! smoothing** (matching the Python oracle's
//! `np.bincount(symbols, minlength=alphabet_size).astype(np.float32) + 1.0`).
//! Storage dtype is `f32`; the categorical model runs in `f64` after a
//! `max(p, 1e-10)` floor + per-row renormalisation — same shape as the
//! existing PR91 / PR103 wrappers.

use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::queue::{DefaultRangeDecoder, DefaultRangeEncoder};
use constriction::stream::{Decode, Encode};

use crate::{PacketCompilerError, Result};

/// 4-byte magic prefix for SAC1 arithmetic-coded coefficient payloads.
pub const SPARSE_AC_MAGIC: [u8; 4] = *b"SAC1";

/// AC-coded coefficient stream with explicit per-stream histogram.
///
/// Mirrors `tac.packet_compiler.ArithmeticCodedCoefficientStream`. The
/// constriction range-coded payload uses **big-endian** uint32 words
/// (matches PR103 wire format).
#[derive(Debug, Clone)]
pub struct ArithmeticCodedCoefficientStream {
    /// Range-coded uint32 stream (constriction big-endian bytes).
    pub encoded_bytes: Vec<u8>,
    /// Per-symbol categorical distribution, length `alphabet_size`. Stored
    /// as `f32` (matching the Python `<f4` wire dtype); the categorical
    /// model itself runs in `f64`.
    pub histogram: Vec<f32>,
    /// Number of symbols encoded (required by the decoder).
    pub n_symbols: u32,
    /// Symbol alphabet cardinality.
    pub alphabet_size: u32,
    /// Integer offset added when decoding to recover signed values.
    pub symbol_offset: i32,
}

/// Floor + per-row renormalisation matching Python's `_make_categorical`:
/// `p = max(p, 1e-10); p /= p.sum()`. Operates in f64.
fn floor_and_renormalise(histogram_f32: &[f32]) -> Result<Vec<f64>> {
    if histogram_f32.is_empty() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "AC histogram must be non-empty".into(),
        ));
    }
    let mut p: Vec<f64> = histogram_f32
        .iter()
        .map(|&v| (v as f64).max(1e-10))
        .collect();
    let sum: f64 = p.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC histogram sum {sum} is non-finite or non-positive"
        )));
    }
    for x in &mut p {
        *x /= sum;
    }
    Ok(p)
}

fn build_categorical(histogram_f32: &[f32]) -> Result<DefaultContiguousCategoricalEntropyModel> {
    let p = floor_and_renormalise(histogram_f32)?;
    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(&p, None)
        .map_err(|_| {
            PacketCompilerError::GoldenVectorIo(
                "constriction Categorical fixed-point quantisation failed".into(),
            )
        })
}

/// Range-code a 1D integer array with a per-stream histogram.
///
/// Mirrors `tac.packet_compiler.encode_arithmetic_coefficients`. When
/// `histogram` is `None`, the encoder builds a Laplace-smoothed empirical
/// distribution. When `symbol_offset` is `None`, `-min(values)` is used.
/// When `alphabet_size` is `None`, `max - min + 1` is used.
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if any input invariant is
///   violated (alphabet too small, mapped symbols out of range, etc.).
pub fn encode_arithmetic_coefficients(
    values: &[i32],
    histogram: Option<&[f32]>,
    symbol_offset: Option<i32>,
    alphabet_size: Option<u32>,
) -> Result<ArithmeticCodedCoefficientStream> {
    if values.is_empty() {
        // Empty stream short-circuit (constriction refuses zero symbols).
        return Ok(ArithmeticCodedCoefficientStream {
            encoded_bytes: Vec::new(),
            histogram: vec![1.0_f32, 1.0_f32],
            n_symbols: 0,
            alphabet_size: 2,
            symbol_offset: 0,
        });
    }

    let v_min = values.iter().copied().min().expect("non-empty");
    let v_max = values.iter().copied().max().expect("non-empty");
    let symbol_offset = symbol_offset.unwrap_or(-v_min);
    let auto_alphabet = (v_max - v_min).checked_add(1).ok_or_else(|| {
        PacketCompilerError::GoldenVectorIo(format!(
            "AC value range overflow: v_min={v_min} v_max={v_max}"
        ))
    })? as i64;
    let mut alphabet = match alphabet_size {
        Some(a) => a as i64,
        None => auto_alphabet,
    };
    if alphabet < auto_alphabet {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC alphabet_size {alphabet} too small for value range [{v_min}, {v_max}] (need >= {auto_alphabet})"
        )));
    }
    if alphabet < 2 {
        // constriction Categorical requires at least 2 symbols.
        alphabet = 2;
    }
    if alphabet > u32::MAX as i64 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC alphabet_size {alphabet} exceeds u32::MAX"
        )));
    }
    let alphabet_u = alphabet as u32;

    // Map values → unsigned symbols [0, alphabet).
    let mut symbols: Vec<usize> = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        let mapped = (v as i64) + (symbol_offset as i64);
        if mapped < 0 || mapped >= alphabet {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "AC mapped symbol {mapped} at index {i} out of [0, {alphabet})"
            )));
        }
        symbols.push(mapped as usize);
    }

    // Build histogram (empirical + Laplace, or caller-supplied).
    let hist_f32: Vec<f32> = match histogram {
        Some(h) => {
            if h.len() as i64 != alphabet {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "AC histogram shape {} != alphabet_size {alphabet}",
                    h.len()
                )));
            }
            h.to_vec()
        }
        None => {
            // np.bincount(symbols, minlength=alphabet) + 1.0
            // Note: Python's bincount returns int64; cast to f32 then add 1.0.
            // We must replicate the EXACT operation order so the resulting
            // f32 bytes are byte-identical.
            let mut bins = vec![0.0_f32; alphabet_u as usize];
            for &s in &symbols {
                bins[s] += 1.0;
            }
            // Wait — Python does `np.bincount(...).astype(np.float32)` THEN
            // `+= 1.0`. The bincount result is an INT64 array; converting
            // to float32 happens before the +1.0 add. For integer counts
            // up to ~2^23 the integer→f32 cast is exact, and the +1.0 add
            // is also exact for those magnitudes. So the byte-exact
            // equivalent is to count as integers, cast, then add 1.0.
            let mut int_bins = vec![0u64; alphabet_u as usize];
            for &s in &symbols {
                int_bins[s] += 1;
            }
            for (i, &c) in int_bins.iter().enumerate() {
                bins[i] = (c as f32) + 1.0;
            }
            bins
        }
    };

    let cat = build_categorical(&hist_f32)?;
    let mut encoder = DefaultRangeEncoder::new();
    encoder
        .encode_iid_symbols(symbols.iter().copied(), cat.as_view())
        .map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!(
                "constriction encode_iid_symbols failed: {e:?}"
            ))
        })?;
    let compressed: Vec<u32> = encoder.into_compressed().map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("constriction into_compressed failed: {e:?}"))
    })?;
    let mut encoded_bytes = Vec::with_capacity(compressed.len() * 4);
    for w in &compressed {
        encoded_bytes.extend_from_slice(&w.to_be_bytes());
    }

    Ok(ArithmeticCodedCoefficientStream {
        encoded_bytes,
        histogram: hist_f32,
        n_symbols: values.len() as u32,
        alphabet_size: alphabet_u,
        symbol_offset,
    })
}

/// Inverse of [`encode_arithmetic_coefficients`].
///
/// Returns the recovered i32 values (signed; `symbol_offset` already
/// subtracted).
pub fn decode_arithmetic_coefficients(
    stream: &ArithmeticCodedCoefficientStream,
) -> Result<Vec<i32>> {
    if stream.n_symbols == 0 {
        return Ok(Vec::new());
    }
    if stream.encoded_bytes.len() % 4 != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC encoded_bytes len {} not multiple of 4",
            stream.encoded_bytes.len()
        )));
    }
    let mut words = Vec::with_capacity(stream.encoded_bytes.len() / 4);
    for chunk in stream.encoded_bytes.chunks_exact(4) {
        words.push(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let mut decoder = DefaultRangeDecoder::from_compressed(words).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!(
            "AC RangeDecoder::from_compressed failed: {e:?}"
        ))
    })?;
    let cat = build_categorical(&stream.histogram)?;
    let mut out = Vec::with_capacity(stream.n_symbols as usize);
    for i in 0..stream.n_symbols {
        let sym = Decode::decode_symbol(&mut decoder, cat.as_view()).map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!("AC decode_symbol failed at i={i}: {e:?}"))
        })?;
        let signed = (sym as i64) - (stream.symbol_offset as i64);
        if !(i32::MIN as i64..=i32::MAX as i64).contains(&signed) {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "AC decoded value {signed} overflows i32"
            )));
        }
        out.push(signed as i32);
    }
    Ok(out)
}

/// Serialise an [`ArithmeticCodedCoefficientStream`] to wire bytes.
pub fn serialize_arithmetic_coefficients(
    stream: &ArithmeticCodedCoefficientStream,
) -> Result<Vec<u8>> {
    if stream.encoded_bytes.len() % 4 != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC encoded_bytes len {} not multiple of 4",
            stream.encoded_bytes.len()
        )));
    }
    if stream.histogram.len() != stream.alphabet_size as usize {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC histogram len {} != alphabet_size {}",
            stream.histogram.len(),
            stream.alphabet_size
        )));
    }
    let word_count = (stream.encoded_bytes.len() / 4) as u32;
    let header_len = 4 + 4 + 4 + 4;
    let total = header_len + (stream.alphabet_size as usize) * 4 + 4 + stream.encoded_bytes.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&SPARSE_AC_MAGIC);
    out.extend_from_slice(&stream.n_symbols.to_le_bytes());
    out.extend_from_slice(&stream.alphabet_size.to_le_bytes());
    out.extend_from_slice(&stream.symbol_offset.to_le_bytes());
    for &h in &stream.histogram {
        out.extend_from_slice(&h.to_le_bytes());
    }
    out.extend_from_slice(&word_count.to_le_bytes());
    out.extend_from_slice(&stream.encoded_bytes);
    Ok(out)
}

/// Inverse of [`serialize_arithmetic_coefficients`]. Hard errors on any
/// truncation / magic mismatch / trailing bytes.
pub fn deserialize_arithmetic_coefficients(
    blob: &[u8],
) -> Result<ArithmeticCodedCoefficientStream> {
    if blob.len() < 16 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC blob too short for header: {} < 16",
            blob.len()
        )));
    }
    if blob[0..4] != SPARSE_AC_MAGIC {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC magic mismatch: got {:?} expected {:?}",
            &blob[0..4],
            SPARSE_AC_MAGIC
        )));
    }
    let n_symbols = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
    let alphabet_size = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
    let symbol_offset = i32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
    let mut pos = 16usize;
    let hist_bytes = (alphabet_size as usize) * 4;
    if pos + hist_bytes + 4 > blob.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC blob truncated reading histogram: need {} bytes, have {}",
            pos + hist_bytes + 4,
            blob.len()
        )));
    }
    let mut histogram = Vec::with_capacity(alphabet_size as usize);
    for _ in 0..alphabet_size {
        histogram.push(f32::from_le_bytes([
            blob[pos],
            blob[pos + 1],
            blob[pos + 2],
            blob[pos + 3],
        ]));
        pos += 4;
    }
    let word_count = u32::from_le_bytes([blob[pos], blob[pos + 1], blob[pos + 2], blob[pos + 3]]);
    pos += 4;
    let encoded_len = (word_count as usize) * 4;
    if pos + encoded_len != blob.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "AC blob length mismatch: pos+encoded={} total={}",
            pos + encoded_len,
            blob.len()
        )));
    }
    let encoded_bytes = blob[pos..pos + encoded_len].to_vec();
    Ok(ArithmeticCodedCoefficientStream {
        encoded_bytes,
        histogram,
        n_symbols,
        alphabet_size,
        symbol_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty_stream() {
        let s = encode_arithmetic_coefficients(&[], None, None, None).unwrap();
        assert_eq!(s.n_symbols, 0);
        assert_eq!(s.alphabet_size, 2);
        let blob = serialize_arithmetic_coefficients(&s).unwrap();
        let s2 = deserialize_arithmetic_coefficients(&blob).unwrap();
        assert_eq!(s2.n_symbols, 0);
    }

    #[test]
    fn small_roundtrip_recovers_values() {
        let values = vec![-2i32, -1, 0, 1, 2, 0, 0, 1];
        let s = encode_arithmetic_coefficients(&values, None, None, None).unwrap();
        let blob = serialize_arithmetic_coefficients(&s).unwrap();
        let s2 = deserialize_arithmetic_coefficients(&blob).unwrap();
        let back = decode_arithmetic_coefficients(&s2).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn explicit_symbol_offset_and_alphabet() {
        let values = vec![-3i32, -2, -1, 0, 1, 2, 3];
        let s = encode_arithmetic_coefficients(&values, None, Some(3), Some(7)).unwrap();
        assert_eq!(s.symbol_offset, 3);
        assert_eq!(s.alphabet_size, 7);
        let blob = serialize_arithmetic_coefficients(&s).unwrap();
        let s2 = deserialize_arithmetic_coefficients(&blob).unwrap();
        let back = decode_arithmetic_coefficients(&s2).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn rejects_alphabet_too_small() {
        let values = vec![-3i32, 5];
        // Range needs 9 symbols ([-3, 5]) but caller asks for 5.
        assert!(encode_arithmetic_coefficients(&values, None, None, Some(5)).is_err());
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let bad = vec![b'X', b'X', b'X', b'X', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(deserialize_arithmetic_coefficients(&bad).is_err());
    }

    #[test]
    fn deserialize_rejects_trailing_bytes() {
        let s = encode_arithmetic_coefficients(&[1i32, 2, 3], None, None, None).unwrap();
        let mut blob = serialize_arithmetic_coefficients(&s).unwrap();
        blob.push(0xFF);
        assert!(deserialize_arithmetic_coefficients(&blob).is_err());
    }
}
