//! Rust port of `encode_categorical_stream` / `decode_categorical_stream`.
//!
//! Byte-for-byte parity contract: the Python oracle uses one
//! `constriction.stream.model.Categorical` PER SYMBOL (the symbol stream has
//! a `(n_symbols, alphabet)` probability matrix; each position selects its
//! own categorical). The Rust port walks the same loop and emits the same
//! big-endian uint32 word array.

use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::queue::{DefaultRangeDecoder, DefaultRangeEncoder};
use constriction::stream::{Decode, Encode};

use crate::{PacketCompilerError, Result};

// MAGIC_QM0 / MAGIC_QH0 live in `super::qmqh_grammar` now (re-exported from
// `super::*` for backward compatibility).

/// Floor + per-row renormalisation matching Python's `_normalise_probs`:
/// each row floored at `1e-10`, then divided by its row sum.
fn normalise_probs(probs: &[f64], n_symbols: usize, alphabet: usize) -> Result<Vec<f64>> {
    if probs.len() != n_symbols * alphabet {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "probs length {} != n_symbols*alphabet = {}",
            probs.len(),
            n_symbols * alphabet
        )));
    }
    let mut out = Vec::with_capacity(probs.len());
    for i in 0..n_symbols {
        let mut row: Vec<f64> = probs[i * alphabet..(i + 1) * alphabet]
            .iter()
            .map(|&v| v.max(1e-10))
            .collect();
        let sum: f64 = row.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "row {i} probability sum {sum} is non-finite or non-positive"
            )));
        }
        for v in &mut row {
            *v /= sum;
        }
        out.extend(row);
    }
    Ok(out)
}

/// Serialise a uint32 word array as big-endian bytes (matches Python
/// `np.asarray(words, dtype=">u4").tobytes()`).
fn words_to_be_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}

/// Parse big-endian uint32 byte stream back to words.
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

/// Range-encode a symbol stream against a per-symbol categorical model.
///
/// `symbols.len()` must equal `n_symbols`; `probs` is row-major
/// `(n_symbols, alphabet)`.
pub fn encode_categorical_stream(
    symbols: &[i32],
    probs: &[f64],
    n_symbols: usize,
    alphabet: usize,
) -> Result<Vec<u8>> {
    if n_symbols == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "must encode at least one symbol".into(),
        ));
    }
    if symbols.len() != n_symbols {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "symbol count {} != n_symbols {}",
            symbols.len(),
            n_symbols
        )));
    }
    let p = normalise_probs(probs, n_symbols, alphabet)?;
    let alphabet_i32 = alphabet as i32;
    for (i, &s) in symbols.iter().enumerate() {
        if s < 0 || s >= alphabet_i32 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "symbol {s} at index {i} out of range [0, {alphabet})"
            )));
        }
    }
    let mut encoder = DefaultRangeEncoder::new();
    // One Categorical per symbol — exactly mirrors the Python loop.
    for i in 0..n_symbols {
        let row = &p[i * alphabet..(i + 1) * alphabet];
        let cat = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
            row, None,
        )
        .map_err(|_| {
            PacketCompilerError::GoldenVectorIo(format!(
                "constriction Categorical quantisation failed at i={i}"
            ))
        })?;
        let sym_usize = symbols[i] as usize;
        Encode::encode_symbol(&mut encoder, sym_usize, cat.as_view()).map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!(
                "constriction encode_symbol failed at i={i}: {e:?}"
            ))
        })?;
    }
    let compressed: Vec<u32> = encoder.into_compressed().map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("constriction into_compressed failed: {e:?}"))
    })?;
    Ok(words_to_be_bytes(&compressed))
}

/// Inverse of [`encode_categorical_stream`]. Returns the recovered i32 symbols.
pub fn decode_categorical_stream(
    payload: &[u8],
    probs: &[f64],
    n_symbols: usize,
    alphabet: usize,
) -> Result<Vec<i32>> {
    if n_symbols == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(
            "must declare at least one symbol".into(),
        ));
    }
    let p = normalise_probs(probs, n_symbols, alphabet)?;
    let words = be_bytes_to_words(payload)?;
    let mut decoder = DefaultRangeDecoder::from_compressed(words).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("RangeDecoder::from_compressed failed: {e:?}"))
    })?;
    let mut out: Vec<i32> = Vec::with_capacity(n_symbols);
    for i in 0..n_symbols {
        let row = &p[i * alphabet..(i + 1) * alphabet];
        let cat = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
            row, None,
        )
        .map_err(|_| {
            PacketCompilerError::GoldenVectorIo(format!(
                "constriction Categorical quantisation failed at i={i}"
            ))
        })?;
        let sym = Decode::decode_symbol(&mut decoder, cat.as_view()).map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!(
                "constriction decode_symbol failed at i={i}: {e:?}"
            ))
        })?;
        if sym >= alphabet {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "decoded symbol {sym} >= alphabet {alphabet} at i={i}"
            )));
        }
        out.push(sym as i32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_smoke() {
        let n_symbols = 50;
        let alphabet = 4;
        // Probability matrix peaked on (i % 4).
        let mut probs = vec![0.05_f64; n_symbols * alphabet];
        for i in 0..n_symbols {
            probs[i * alphabet + (i % alphabet)] = 0.85;
        }
        // Symbols sampled by argmax for determinism.
        let symbols: Vec<i32> = (0..n_symbols).map(|i| (i % alphabet) as i32).collect();
        let payload = encode_categorical_stream(&symbols, &probs, n_symbols, alphabet).unwrap();
        let decoded = decode_categorical_stream(&payload, &probs, n_symbols, alphabet).unwrap();
        assert_eq!(decoded, symbols);
    }
}
