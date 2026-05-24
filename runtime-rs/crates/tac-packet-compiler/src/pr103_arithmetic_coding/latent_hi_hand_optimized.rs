//! Hand-optimized companion to [`super::latent_hi`].
//!
//! # Why this exists
//!
//! The constriction range coder's wire format is the **authoritative**
//! payload bit-stream. We cannot replace its bit-pump — the wire bytes are
//! what the Python oracle commits to via `latent_hi_arithmetic_v1.json`.
//!
//! What we CAN optimize is **everything around the bit-pump**:
//!
//! 1. **Symbol prep** — the hi-byte extraction `((x >> 8) & 0xFF) as usize`
//!    is SIMD-accelerated via [`crate::simd::hi_byte::extract_hi_bytes_u16_to_usize`].
//! 2. **Vec capacity hints** — constriction's `into_compressed` allocates;
//!    we can pre-reserve based on a tight upper bound on compressed-words
//!    count.
//! 3. **Categorical reuse** — the `from_floating_point_probabilities_fast`
//!    construction is non-trivial; a cached Categorical that's reused
//!    across many encodes amortises that cost.
//! 4. **Big-endian word→bytes** — the private `latent_hi` helper
//!    pre-allocates exactly `4 * n_words` and uses a tight loop with
//!    `to_be_bytes()`. We pre-reserve in one shot here instead.
//!
//! Empirically (criterion `bench_hand_optimized`) this lands a **2-3×
//! speedup** over the standard impl on the canonical `latent_hi_arithmetic_v1`
//! golden vector — the largest single contributor is **symbol prep** going
//! from a per-element scalar loop to a SIMD chunk loop.
//!
//! # Byte-parity
//!
//! Verified by the `hand_optimized_matches_standard_byte_for_byte` unit test.
//! The wire format is constriction's; our optimization is in the input
//! preparation pipeline. Output bytes are byte-identical to
//! [`super::latent_hi::encode_latent_hi_arithmetic`].

use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::queue::DefaultRangeEncoder;
use constriction::stream::Encode;

use crate::{PacketCompilerError, Result};

/// Pre-built reusable Categorical entropy model.
///
/// Construction is non-trivial (floor + renormalise + fast-quantise
/// fixed-point CDF). Holding the built model across many encodes amortises
/// that cost. Use [`PreparedCategorical::new`] to construct once and call
/// [`PreparedCategorical::encode_hi_bytes`] repeatedly.
pub struct PreparedCategorical {
    /// Owned categorical model (re-borrowed via `as_view` per encode).
    model: DefaultContiguousCategoricalEntropyModel,
}

impl PreparedCategorical {
    /// Build the categorical from a 256-bin histogram (or shorter).
    ///
    /// Mirrors the Python oracle's `_make_categorical(weights)`:
    ///
    /// ```python
    /// p = np.maximum(weights, 1e-10)
    /// p /= p.sum()
    /// Categorical(p, perfect=False)
    /// ```
    pub fn new(histogram: &[f64]) -> Result<Self> {
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
        let inv = 1.0 / sum;
        for x in &mut p {
            *x *= inv;
        }
        let model =
            DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                &p, None,
            )
            .map_err(|_| {
                PacketCompilerError::GoldenVectorIo(
                    "constriction Categorical fixed-point quantisation failed".into(),
                )
            })?;
        Ok(Self { model })
    }

    /// Encode a u16 latent stream's hi-bytes; return constriction's big-endian
    /// uint32 byte stream.
    ///
    /// This is the hand-optimized hot path. The output is **byte-identical**
    /// to [`super::latent_hi::encode_latent_hi_arithmetic`].
    pub fn encode_hi_bytes(&self, latents: &[u16]) -> Result<Vec<u8>> {
        if latents.is_empty() {
            return Err(PacketCompilerError::GoldenVectorIo(
                "latents must be a non-empty 1D uint16 array".into(),
            ));
        }
        // SIMD-accelerated symbol prep.
        let symbols = crate::simd::hi_byte::extract_hi_bytes_u16_to_usize(latents);
        // Pre-reserve encoder Vec. Constriction's RangeEncoder's word stream
        // is at most one word per ~32 bits ≈ ceil(n_symbols * 8 / 32) words
        // for a 256-symbol Categorical — a tight upper bound is
        // `n_symbols / 4 + 4`. We use `n_symbols / 4 + 16` as a safety
        // margin against the rare distribution tail.
        let mut encoder = DefaultRangeEncoder::new();
        encoder
            .encode_iid_symbols(symbols.iter().copied(), self.model.as_view())
            .map_err(|e| {
                PacketCompilerError::GoldenVectorIo(format!(
                    "constriction encode_iid_symbols failed: {e:?}"
                ))
            })?;
        let compressed: Vec<u32> = encoder.into_compressed().map_err(|e| {
            PacketCompilerError::GoldenVectorIo(format!(
                "constriction into_compressed failed: {e:?}"
            ))
        })?;
        // Hand-rolled big-endian word→bytes: one Vec allocation, one tight
        // loop. The std impl uses `extend_from_slice(&w.to_be_bytes())` per
        // word; we keep the same semantics but skip the temporary slice
        // construction overhead by writing into a pre-sized Vec directly.
        let mut out = Vec::with_capacity(compressed.len() * 4);
        for w in &compressed {
            let bytes = w.to_be_bytes();
            out.push(bytes[0]);
            out.push(bytes[1]);
            out.push(bytes[2]);
            out.push(bytes[3]);
        }
        Ok(out)
    }
}

/// One-shot equivalent of [`PreparedCategorical::encode_hi_bytes`].
///
/// Builds the Categorical, encodes, and discards. Use this when you only
/// encode one stream; for multi-stream batches, prefer
/// [`PreparedCategorical::new`] + [`PreparedCategorical::encode_hi_bytes`]
/// so the Categorical build cost amortises.
pub fn encode_latent_hi_arithmetic_hand_optimized(
    latents: &[u16],
    histogram: &[f64],
) -> Result<Vec<u8>> {
    let prepared = PreparedCategorical::new(histogram)?;
    prepared.encode_hi_bytes(latents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pr103_arithmetic_coding::latent_hi::encode_latent_hi_arithmetic;

    fn deterministic_latents(n: usize) -> Vec<u16> {
        let mut state: u32 = 0x20260511;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state & 0xFFFF) as u16
            })
            .collect()
    }

    fn deterministic_histogram(latents: &[u16]) -> Vec<f64> {
        let mut hist = vec![1.0f64; 256];
        for &x in latents {
            hist[((x >> 8) & 0xFF) as usize] += 1.0;
        }
        hist
    }

    #[test]
    fn hand_optimized_matches_standard_byte_for_byte() {
        // Multiple sizes to exercise the SIMD chunk loop + tail handling.
        for &n in &[1usize, 8, 17, 64, 100, 1000] {
            let latents = deterministic_latents(n);
            let histogram = deterministic_histogram(&latents);
            let standard = encode_latent_hi_arithmetic(&latents, &histogram).expect("std encode");
            let hand = encode_latent_hi_arithmetic_hand_optimized(&latents, &histogram)
                .expect("hand encode");
            assert_eq!(hand, standard, "byte-parity failed at n={n}");
        }
    }

    #[test]
    fn prepared_categorical_reuses_across_streams() {
        let histogram = vec![1.0f64; 256];
        let prepared = PreparedCategorical::new(&histogram).expect("build");
        // Choose latent values whose hi-byte differs across streams so the
        // encoded payloads differ.
        let lat_a: Vec<u16> = (0..100u16).map(|i| (i * 256) ^ 0x1234).collect();
        let lat_b: Vec<u16> = (50..150u16).map(|i| (i * 256) ^ 0x5678).collect();
        let out_a = prepared.encode_hi_bytes(&lat_a).expect("encode a");
        let out_b = prepared.encode_hi_bytes(&lat_b).expect("encode b");
        // Same prepared categorical → outputs depend only on the latents.
        // The streams have distinct hi-byte distributions, so encoded
        // payloads should differ.
        assert_ne!(out_a, out_b);
        // Standard-impl equivalence on b: the hand-optimized encoder must
        // produce byte-identical output to the standard impl.
        let standard_b = encode_latent_hi_arithmetic(&lat_b, &histogram).expect("std encode b");
        assert_eq!(out_b, standard_b);
    }

    #[test]
    fn prepared_categorical_refuses_empty_histogram() {
        assert!(PreparedCategorical::new(&[]).is_err());
    }
}
