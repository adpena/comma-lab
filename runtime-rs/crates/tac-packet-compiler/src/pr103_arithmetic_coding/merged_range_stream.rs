//! Rust port of `encode_merged_range_stream` / `decode_merged_range_stream`.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`src/tac/packet_compiler/pr103_arithmetic_coding.py`) encodes N tensors
//! into a SINGLE range-coded byte stream with one `Categorical` per tensor
//! (the same categorical applies to every symbol of that tensor). The
//! per-tensor encode order is encode-time + decode-time critical; the
//! decoder cannot recover boundaries from the byte stream alone, hence
//! `tensor_symbol_counts` ships out-of-band.
//!
//! Sibling binary fixtures live at:
//!
//! - `src/tac/packet_compiler/golden_vectors/merged_range_stream_v1_flat.bin`
//! - `src/tac/packet_compiler/golden_vectors/merged_range_stream_v1_hist{0,1,2}.bin`

use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::queue::{DefaultRangeDecoder, DefaultRangeEncoder};
use constriction::stream::{Decode, Encode};

use crate::{PacketCompilerError, Result};

use super::stubs::{MergedRangeStream, WeightTensorACSpec};

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
            "histogram sum {sum} non-finite or non-positive"
        )));
    }
    for v in &mut p {
        *v /= sum;
    }
    Ok(p)
}

fn build_categorical(
    histogram: &[f64],
) -> Result<DefaultContiguousCategoricalEntropyModel> {
    let p = floor_and_renormalise(histogram)?;
    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(&p, None)
        .map_err(|_| {
            PacketCompilerError::GoldenVectorIo(
                "constriction Categorical quantisation failed".into(),
            )
        })
}

fn words_to_be_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}

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

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Encode multiple weight tensors into a single range-coded byte string.
///
/// `tensors_flat_int32` is the concatenated flat int32 symbol stream for all
/// tensors in `specs` order. Each tensor is encoded against its corresponding
/// [`WeightTensorACSpec::histogram`] (one categorical re-used for every
/// symbol of that tensor — this is the merged-tensor encoding pattern that
/// saves per-tensor framing overhead).
pub fn encode_merged_range_stream(
    tensors_flat_int32: &[i32],
    specs: &[WeightTensorACSpec],
) -> Result<MergedRangeStream> {
    if specs.is_empty() {
        return Err(PacketCompilerError::GoldenVectorIo(
            "must encode at least one tensor".into(),
        ));
    }
    // Validate shapes and compute symbol-count budget.
    let mut counts: Vec<usize> = Vec::with_capacity(specs.len());
    let mut expected_total = 0usize;
    for s in specs {
        let n = product(&s.shape);
        if n == 0 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "tensor {} has zero-size shape {:?}",
                s.name, s.shape
            )));
        }
        counts.push(n);
        expected_total += n;
    }
    if tensors_flat_int32.len() != expected_total {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "tensors_flat length {} != sum of spec.shape products = {}",
            tensors_flat_int32.len(),
            expected_total
        )));
    }
    let mut encoder = DefaultRangeEncoder::new();
    let mut offset = 0usize;
    for (s, &n) in specs.iter().zip(counts.iter()) {
        let alphabet_i32 = s.alphabet_size as i32;
        let chunk = &tensors_flat_int32[offset..offset + n];
        for (i, &v) in chunk.iter().enumerate() {
            if v < 0 || v >= alphabet_i32 {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "tensor {} symbol {v} at i={i} out of range [0, {})",
                    s.name, s.alphabet_size
                )));
            }
        }
        let cat = build_categorical(&s.histogram)?;
        // Encode every symbol of this tensor against the same categorical;
        // the Python oracle does this in one batched call:
        // `encoder.encode(flat, cat)`. The Rust API exposes
        // `encode_iid_symbols`; both produce the same bit-stream.
        let cat_view = cat.as_view();
        let symbols_iter = chunk.iter().map(|&v| v as usize);
        encoder
            .encode_iid_symbols(symbols_iter, cat_view)
            .map_err(|e| {
                PacketCompilerError::GoldenVectorIo(format!(
                    "constriction encode_iid_symbols failed for tensor {}: {e:?}",
                    s.name
                ))
            })?;
        offset += n;
    }
    let compressed: Vec<u32> = encoder.into_compressed().map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("constriction into_compressed failed: {e:?}"))
    })?;
    let word_count = compressed.len();
    let payload = words_to_be_bytes(&compressed);
    Ok(MergedRangeStream {
        payload,
        tensor_symbol_counts: counts,
        word_count,
    })
}

/// Inverse of [`encode_merged_range_stream`].
///
/// Returns a flat int32 buffer; the caller reshapes per `specs[i].shape`.
pub fn decode_merged_range_stream(
    stream: &MergedRangeStream,
    specs: &[WeightTensorACSpec],
) -> Result<Vec<i32>> {
    if stream.tensor_symbol_counts.len() != specs.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "stream has {} tensor counts; specs has {}",
            stream.tensor_symbol_counts.len(),
            specs.len()
        )));
    }
    let words = be_bytes_to_words(&stream.payload)?;
    if words.len() != stream.word_count {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "payload word count mismatch: bytes imply {}; stream declared {}",
            words.len(),
            stream.word_count
        )));
    }
    let mut decoder = DefaultRangeDecoder::from_compressed(words).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!(
            "RangeDecoder::from_compressed failed: {e:?}"
        ))
    })?;
    let mut out: Vec<i32> = Vec::with_capacity(stream.tensor_symbol_counts.iter().sum());
    for (s, &count) in specs.iter().zip(stream.tensor_symbol_counts.iter()) {
        let cat = build_categorical(&s.histogram)?;
        let cat_view = cat.as_view();
        for i in 0..count {
            let sym = Decode::decode_symbol(&mut decoder, cat_view).map_err(|e| {
                PacketCompilerError::GoldenVectorIo(format!(
                    "constriction decode_symbol failed for tensor {} at i={i}: {e:?}",
                    s.name
                ))
            })?;
            if sym >= s.alphabet_size as usize {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "decoded symbol {sym} >= alphabet {} for tensor {}",
                    s.alphabet_size, s.name
                )));
            }
            out.push(sym as i32);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec(name: &str, shape: Vec<usize>, alphabet: u32) -> WeightTensorACSpec {
        WeightTensorACSpec {
            name: name.to_string(),
            shape,
            histogram: vec![1.0; alphabet as usize],
            alphabet_size: alphabet,
        }
    }

    #[test]
    fn roundtrip_small_two_tensor() {
        let specs = vec![
            make_spec("t0", vec![6], 8),
            make_spec("t1", vec![2, 3], 8),
        ];
        let flat: Vec<i32> = vec![3, 1, 0, 5, 2, 7, 4, 6, 1, 0, 2, 7];
        let stream = encode_merged_range_stream(&flat, &specs).unwrap();
        let decoded = decode_merged_range_stream(&stream, &specs).unwrap();
        assert_eq!(decoded, flat);
        assert_eq!(stream.tensor_symbol_counts, vec![6, 6]);
    }
}
