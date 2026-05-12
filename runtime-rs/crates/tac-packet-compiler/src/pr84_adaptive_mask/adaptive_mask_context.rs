//! Rust port of `encode_adaptive_context_stream` / `decode_adaptive_context_stream`.
//!
//! Byte-for-byte parity contract: the Python oracle builds one
//! `constriction.stream.model.Categorical` per CONTEXT (n_contexts << n_symbols),
//! then routes each symbol through the categorical chosen by its parallel
//! context_id. The Rust port walks the same loop and emits the same big-endian
//! uint32 word array.

use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::queue::{DefaultRangeDecoder, DefaultRangeEncoder};
use constriction::stream::{Decode, Encode};

use crate::{PacketCompilerError, Result};

/// Per-context categorical model for an adaptive-context range coder.
///
/// Mirrors `tac.packet_compiler.AdaptiveContextSpec`.
#[derive(Debug, Clone)]
pub struct AdaptiveContextSpec {
    /// Number of distinct symbol values (each symbol in `[0, alphabet_size)`).
    pub alphabet_size: u32,
    /// Categorical frequency table of length `n_contexts * alphabet_size`,
    /// row-major; each row is one context's cdf.
    pub cdf_table: Vec<f64>,
    /// Number of context rows (= `cdf_table.len() / alphabet_size`).
    pub n_contexts: u32,
}

impl AdaptiveContextSpec {
    /// Construct a spec from a flat row-major `(n_contexts, alphabet_size)` table.
    pub fn new(alphabet_size: u32, n_contexts: u32, cdf_table: Vec<f64>) -> Result<Self> {
        if alphabet_size < 2 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "alphabet_size must be >= 2; got {alphabet_size}"
            )));
        }
        if n_contexts < 1 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "n_contexts must be >= 1; got {n_contexts}"
            )));
        }
        let expected = (alphabet_size as usize) * (n_contexts as usize);
        if cdf_table.len() != expected {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "cdf_table length {} != n_contexts*alphabet = {}",
                cdf_table.len(),
                expected
            )));
        }
        Ok(Self {
            alphabet_size,
            cdf_table,
            n_contexts,
        })
    }
}

/// Floor + renormalise a single context's row to sum to 1, matching Python's
/// `_normalise_row` (floor at `1e-10`, then divide by sum).
fn normalise_row(row: &[f64]) -> Result<Vec<f64>> {
    let mut p: Vec<f64> = row.iter().map(|&v| v.max(1e-10)).collect();
    let sum: f64 = p.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "context row sum {sum} non-finite or non-positive"
        )));
    }
    for v in &mut p {
        *v /= sum;
    }
    Ok(p)
}

/// Build one Categorical per context id (mirrors Python's `_build_categoricals`).
fn build_categoricals(
    spec: &AdaptiveContextSpec,
) -> Result<Vec<DefaultContiguousCategoricalEntropyModel>> {
    let mut cats = Vec::with_capacity(spec.n_contexts as usize);
    let alphabet = spec.alphabet_size as usize;
    for ctx in 0..spec.n_contexts as usize {
        let row = &spec.cdf_table[ctx * alphabet..(ctx + 1) * alphabet];
        let normalised = normalise_row(row)?;
        let cat = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
            &normalised,
            None,
        )
        .map_err(|_| {
            PacketCompilerError::GoldenVectorIo(format!(
                "constriction Categorical quantisation failed at ctx={ctx}"
            ))
        })?;
        cats.push(cat);
    }
    Ok(cats)
}

/// Serialise constriction's uint32 word array as big-endian bytes.
fn words_to_be_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}

/// Parse big-endian uint32 byte stream.
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

/// Range-encode `symbols` against the context-routed categorical model.
pub fn encode_adaptive_context_stream(
    symbols: &[i32],
    context_ids: &[i32],
    spec: &AdaptiveContextSpec,
) -> Result<Vec<u8>> {
    if symbols.len() != context_ids.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "symbols ({}) and context_ids ({}) must have the same length",
            symbols.len(),
            context_ids.len()
        )));
    }
    if symbols.is_empty() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "must encode at least one symbol".into(),
        ));
    }
    let alphabet = spec.alphabet_size as i32;
    let n_contexts = spec.n_contexts as i32;
    for (i, (&s, &c)) in symbols.iter().zip(context_ids.iter()).enumerate() {
        if s < 0 || s >= alphabet {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "symbol {s} at index {i} out of range [0, {alphabet})"
            )));
        }
        if c < 0 || c >= n_contexts {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "context_id {c} at index {i} out of range [0, {n_contexts})"
            )));
        }
    }
    let cats = build_categoricals(spec)?;
    let mut encoder = DefaultRangeEncoder::new();
    for (i, (&s, &c)) in symbols.iter().zip(context_ids.iter()).enumerate() {
        let cat = &cats[c as usize];
        let sym_usize = s as usize;
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

/// Inverse of [`encode_adaptive_context_stream`].
pub fn decode_adaptive_context_stream(
    payload: &[u8],
    context_ids: &[i32],
    spec: &AdaptiveContextSpec,
) -> Result<Vec<i32>> {
    if context_ids.is_empty() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "context_ids must declare at least one symbol".into(),
        ));
    }
    let n_contexts = spec.n_contexts as i32;
    for (i, &c) in context_ids.iter().enumerate() {
        if c < 0 || c >= n_contexts {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "context_id {c} at index {i} out of range [0, {n_contexts})"
            )));
        }
    }
    let cats = build_categoricals(spec)?;
    let words = be_bytes_to_words(payload)?;
    let mut decoder = DefaultRangeDecoder::from_compressed(words).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("RangeDecoder::from_compressed failed: {e:?}"))
    })?;
    let mut out: Vec<i32> = Vec::with_capacity(context_ids.len());
    let alphabet = spec.alphabet_size as usize;
    for (i, &c) in context_ids.iter().enumerate() {
        let cat = &cats[c as usize];
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
        // 3 contexts × 4 alphabet; each context peaks on a distinct symbol.
        let n_contexts = 3u32;
        let alphabet = 4u32;
        let mut cdf = vec![0.05_f64; (n_contexts * alphabet) as usize];
        for ctx in 0..n_contexts as usize {
            cdf[ctx * alphabet as usize + ctx] = 0.85;
        }
        let spec = AdaptiveContextSpec::new(alphabet, n_contexts, cdf).unwrap();
        let n_symbols = 30;
        let context_ids: Vec<i32> = (0..n_symbols).map(|i| i % n_contexts as i32).collect();
        let symbols: Vec<i32> = (0..n_symbols).map(|i| i % alphabet as i32).collect();
        let payload = encode_adaptive_context_stream(&symbols, &context_ids, &spec).unwrap();
        let decoded = decode_adaptive_context_stream(&payload, &context_ids, &spec).unwrap();
        assert_eq!(decoded, symbols);
    }
}
