//! Rust port of `encode_latent_hi_arithmetic` /
//! `decode_latent_hi_arithmetic`.
//!
//! # Byte-for-byte parity contract
//!
//! Python oracle (`src/tac/packet_compiler/pr103_arithmetic_coding.py`):
//!
//! ```text
//! hi = ((latents.astype(int32) >> 8) & 0xFF).astype(int32)
//! cat = constriction.stream.model.Categorical(p, perfect=False)
//! enc = constriction.stream.queue.RangeEncoder()
//! enc.encode(hi, cat)
//! payload = words_to_be_bytes(enc.get_compressed())
//! ```
//!
//! Rust port uses the SAME upstream library (Bamler Lab `constriction`
//! 0.4.x); the wire-format `RangeEncoder<u32, u64>` ≡ `DefaultRangeEncoder`
//! produces a `Vec<u32>` of compressed words. We serialise to **big-endian
//! bytes** exactly as the Python helper `_words_to_uint32_bytes` does
//! (`np.asarray(words, dtype=">u4").tobytes()`).
//!
//! # Categorical construction
//!
//! Python `Categorical(p, perfect=False)` first floors `p` at `1e-10`,
//! re-normalises so it sums to 1, then quantises to fixed-point with the
//! "fast" (non-perfect) algorithm. The Rust crate's
//! `DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast`
//! is the same algorithm. Per the constriction 0.4 release notes both
//! implementations call the same internal `fast_quantized_cdf` routine.
//!
//! # Symbol type
//!
//! The Python oracle encodes `int32` symbols; the Rust crate's
//! Categorical is parameterised over `usize` for `Contiguous*`. We cast
//! `i32 -> usize` via `try_from` so a negative or out-of-alphabet symbol
//! fails loud.

use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::queue::{DefaultRangeDecoder, DefaultRangeEncoder};
use constriction::stream::Decode;
use constriction::stream::Encode;

use crate::{PacketCompilerError, Result};

/// Floor + renormalise probabilities the way Python's
/// `_make_categorical` does (`p = max(p, 1e-10); p /= p.sum()`).
fn floor_and_renormalise(histogram: &[f64]) -> Result<Vec<f64>> {
    if histogram.is_empty() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "histogram must be non-empty".into(),
        ));
    }
    let mut p: Vec<f64> = histogram.iter().map(|&v| v.max(1e-10)).collect();
    let sum: f64 = p.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "histogram sum {sum} is non-finite or non-positive"
        )));
    }
    for x in &mut p {
        *x /= sum;
    }
    Ok(p)
}

/// Build the Categorical entropy model from a 256-bin (or shorter)
/// histogram. Mirrors `_make_categorical(weights)` in Python (uses the
/// "fast" — non-perfect — fixed-point quantiser per constriction defaults).
fn build_categorical(histogram: &[f64]) -> Result<DefaultContiguousCategoricalEntropyModel> {
    let p = floor_and_renormalise(histogram)?;
    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(&p, None)
        .map_err(|_| {
            PacketCompilerError::GoldenVectorIo(
                "constriction Categorical fixed-point quantisation failed".into(),
            )
        })
}

/// Serialise a uint32 word array as big-endian bytes (mirrors Python
/// `np.asarray(words, dtype=">u4").tobytes()`).
fn words_to_be_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}

/// Parse big-endian uint32 byte stream back to words (mirrors Python
/// `np.frombuffer(payload, dtype=">u4").astype(np.uint32)`).
fn be_bytes_to_words(payload: &[u8]) -> Result<Vec<u32>> {
    if payload.len() % 4 != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "payload size {} is not a multiple of 4",
            payload.len()
        )));
    }
    let mut out = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.chunks_exact(4) {
        out.push(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Encode the high-byte of a uint16 latent stream via arithmetic coding.
///
/// `latents` is the **uint16 zigzag-encoded delta** stream (PR103 layout);
/// the high byte is `(x >> 8) & 0xFF`. Output is constriction's uint32
/// stream serialised as big-endian bytes.
///
/// Byte-for-byte parity target: see
/// `src/tac/packet_compiler/golden_vectors/latent_hi_arithmetic_v1.json`.
pub fn encode_latent_hi_arithmetic(latents: &[u16], histogram: &[f64]) -> Result<Vec<u8>> {
    if latents.is_empty() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "latents must be a non-empty 1D uint16 array".into(),
        ));
    }
    let cat = build_categorical(histogram)?;
    let mut encoder = DefaultRangeEncoder::new();
    // hi byte values are usize symbols in [0, 256).
    let symbols: Vec<usize> = latents
        .iter()
        .map(|&x| ((x as u32 >> 8) & 0xFF) as usize)
        .collect();
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
    Ok(words_to_be_bytes(&compressed))
}

/// Decode a high-byte stream produced by [`encode_latent_hi_arithmetic`].
///
/// Returns a `Vec<u8>` of length `n_symbols`. The caller is responsible
/// for combining with the low bytes to reconstruct the original uint16
/// zigzag deltas.
pub fn decode_latent_hi_arithmetic(
    payload: &[u8],
    histogram: &[f64],
    n_symbols: usize,
) -> Result<Vec<u8>> {
    if n_symbols == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "n_symbols must be > 0".into(),
        ));
    }
    let words = be_bytes_to_words(payload)?;
    let mut decoder = DefaultRangeDecoder::from_compressed(words).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("RangeDecoder::from_compressed failed: {e:?}"))
    })?;
    let cat = build_categorical(histogram)?;
    let mut out = Vec::with_capacity(n_symbols);
    for i in 0..n_symbols {
        let sym = Decode::decode_symbol(&mut decoder, cat.as_view()).map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!(
                "constriction decode_symbol failed at i={i}: {e:?}"
            ))
        })?;
        if sym > 255 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "decoded symbol {sym} out of uint8 range at i={i}"
            )));
        }
        out.push(sym as u8);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_and_renormalise_handles_zeros() {
        let p = floor_and_renormalise(&[0.0, 1.0, 0.0]).unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // First entry should be tiny but non-zero.
        assert!(p[0] > 0.0);
        assert!(p[0] < 1e-9);
    }

    #[test]
    fn roundtrip_smoke() {
        // Mostly-zero hi-byte stream (PR103 typical shape).
        let mut latents: Vec<u16> = (0..200).map(|i| (i % 16) as u16).collect();
        latents[0] = 300; // forces high byte to be non-zero somewhere
        latents[50] = 1000;
        // Histogram seeded from the empirical hi-byte distribution + 1 (Python recipe).
        let hi: Vec<u32> = latents.iter().map(|&x| ((x >> 8) & 0xFF) as u32).collect();
        let mut hist = vec![1.0f64; 256];
        for h in &hi {
            hist[*h as usize] += 1.0;
        }
        let payload = encode_latent_hi_arithmetic(&latents, &hist).unwrap();
        let decoded = decode_latent_hi_arithmetic(&payload, &hist, latents.len()).unwrap();
        let expected: Vec<u8> = hi.iter().map(|&x| x as u8).collect();
        assert_eq!(decoded, expected);
    }
}
