//! Rust port of PR81's asymmetric 8-level FP4 codebook (+ sign bit) and
//! 2-nibbles-per-byte packing.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`src/tac/packet_compiler/pr81_quantizr.py::FP4Codebook.quantize` +
//! `pack_nibbles`) produces a packed byte stream from a fp32 value array +
//! per-block fp32 scales + block size. This module reproduces that
//! quantisation + packing identically.
//!
//! # Algorithm (matches Python oracle)
//!
//! 1. Reshape `values` into blocks of `block_size`. Pad with zeros if the
//!    final block is short.
//! 2. For each block, divide by the scalar `scales[block]` (fp32).
//! 3. For each scaled value: sign bit = `scaled < 0`; magnitude index =
//!    argmin over `|abs(scaled) - level|` for the 8 entries of
//!    [`PR81_POS_LEVELS`] (`argmin` ties go to the smallest index — numpy
//!    default).
//! 4. Pack the nibble `(sign << 3) | mag_idx`.
//! 5. After all nibbles are produced (length `n_blocks * block_size`,
//!    including the padding), pack two nibbles per byte (`hi << 4 | lo`).
//!
//! `quantize_to_nibbles` returns the nibble stream pre-packing (length =
//! `n_blocks * block_size`); `pack_nibbles` converts it to bytes.

use crate::{PacketCompilerError, Result};

/// Canonical PR81/Quantizr asymmetric positive-level table. Index 0 → 0.0
/// through index 7 → 6.0; the sign bit doubles the alphabet to 16 codes.
pub const PR81_POS_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// PR81-compatible asymmetric 8-level FP4 codebook with sign bit.
///
/// Mirrors `tac.packet_compiler.FP4Codebook`. The default constructor
/// returns the canonical PR81 codebook; a custom 8-entry positive table can
/// be supplied via [`FP4Codebook::with_levels`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FP4Codebook {
    /// 8-entry non-negative magnitude table. Default: [`PR81_POS_LEVELS`].
    pub pos_levels: [f32; 8],
}

impl Default for FP4Codebook {
    fn default() -> Self {
        Self {
            pos_levels: PR81_POS_LEVELS,
        }
    }
}

impl FP4Codebook {
    /// Construct an FP4 codebook with a custom 8-entry positive level table.
    ///
    /// Validates that all levels are non-negative and non-decreasing,
    /// matching the Python `__post_init__` guard.
    pub fn with_levels(pos_levels: [f32; 8]) -> Result<Self> {
        let mut prev = -1.0_f32;
        for &level in &pos_levels {
            if level < 0.0 {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "pos_levels must be non-negative; got {level}"
                )));
            }
            // Use a positive comparison instead of `!(level >= prev)`: NaN
            // is rejected via the `is_finite` guard so partial_cmp is well-
            // defined; the check is "current level must be >= prev".
            if !level.is_finite() || level < prev {
                return Err(PacketCompilerError::GoldenVectorIo(format!(
                    "pos_levels must be non-decreasing and finite; got {pos_levels:?}"
                )));
            }
            prev = level;
        }
        Ok(Self { pos_levels })
    }

    /// Quantise float values to PR81 4-bit nibbles.
    ///
    /// Mirrors `tac.packet_compiler.FP4Codebook.quantize`. Returns a
    /// `Vec<u8>` of nibble codes (`(sign << 3) | mag_idx` each in
    /// `[0, 15]`); length is `n_blocks_needed * block_size` (post-padding).
    ///
    /// # Tie-breaking
    ///
    /// Equal distances to two levels resolve to the smaller index (numpy
    /// `np.argmin` default behavior). On the PR81 codebook this only
    /// matters at the midpoints (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
    /// which are unlikely to be hit exactly by the float input but the
    /// behavior is pinned for reproducibility.
    pub fn quantize(&self, values: &[f32], scales: &[f32], block_size: usize) -> Result<Vec<u8>> {
        quantize_to_nibbles(values, scales, block_size, &self.pos_levels)
    }

    /// Dequantise PR81 nibble codes back to a float32 array.
    ///
    /// Mirrors `tac.packet_compiler.FP4Codebook.dequantize_from_nibbles`.
    /// Returns at most `nibbles.len()` floats; if `n_values` is supplied
    /// the result is truncated to that length (trailing padding stripped).
    pub fn dequantize(
        &self,
        nibbles: &[u8],
        scales: &[f32],
        block_size: usize,
        n_values: Option<usize>,
    ) -> Result<Vec<f32>> {
        dequantize_from_nibbles(nibbles, scales, block_size, &self.pos_levels, n_values)
    }
}

/// Free-function form of [`FP4Codebook::quantize`] taking the level table
/// explicitly. Lets callers operate without instantiating the dataclass.
pub fn quantize_to_nibbles(
    values: &[f32],
    scales: &[f32],
    block_size: usize,
    pos_levels: &[f32; 8],
) -> Result<Vec<u8>> {
    if block_size == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "block_size must be > 0; got {block_size}"
        )));
    }
    let n_values = values.len();
    let n_blocks_needed = n_values.div_ceil(block_size);
    if scales.len() != n_blocks_needed {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "scales has {} entries; need {} for {} values @ block_size={}",
            scales.len(),
            n_blocks_needed,
            n_values,
            block_size
        )));
    }
    for (i, &s) in scales.iter().enumerate() {
        // Reject NaN, zero, and negative values via finite check + positivity.
        if !s.is_finite() || s <= 0.0 {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "scales must be finite and strictly positive; got scales[{i}] = {s}"
            )));
        }
    }
    let total_nibbles = n_blocks_needed * block_size;
    let mut nibbles: Vec<u8> = Vec::with_capacity(total_nibbles);
    for (b, &scale) in scales.iter().enumerate().take(n_blocks_needed) {
        let block_start = b * block_size;
        let block_end = block_start + block_size;
        // Inner loop uses a numeric range because we may step PAST `values.len()`
        // (the Python oracle right-pads with zeros so a fixed
        // `n_blocks * block_size` length emits a clean reshape). Iterating
        // `values` directly would silently truncate the final block.
        #[allow(clippy::needless_range_loop)]
        for j in block_start..block_end {
            // Pad missing trailing values with 0.0 (matches Python pad).
            let v = if j < n_values { values[j] } else { 0.0_f32 };
            let scaled = v / scale;
            let sign: u8 = if scaled < 0.0 { 1 } else { 0 };
            let mag = scaled.abs();
            // argmin |mag - level| with smallest-index tie-break.
            let mut best_idx: u8 = 0;
            let mut best_dist = (mag - pos_levels[0]).abs();
            for (k, &level) in pos_levels.iter().enumerate().skip(1) {
                let d = (mag - level).abs();
                if d < best_dist {
                    best_dist = d;
                    best_idx = k as u8;
                }
            }
            let nibble = ((sign & 0x1) << 3) | (best_idx & 0x7);
            nibbles.push(nibble);
        }
    }
    Ok(nibbles)
}

/// Free-function form of [`FP4Codebook::dequantize`].
pub fn dequantize_from_nibbles(
    nibbles: &[u8],
    scales: &[f32],
    block_size: usize,
    pos_levels: &[f32; 8],
    n_values: Option<usize>,
) -> Result<Vec<f32>> {
    if block_size == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "block_size must be > 0; got {block_size}"
        )));
    }
    for (i, &n) in nibbles.iter().enumerate() {
        if n > 0xF {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "nibble values must fit in 4 bits ([0, 15]); got nibbles[{i}] = {n}"
            )));
        }
    }
    let n = nibbles.len();
    let n_blocks_needed = n.div_ceil(block_size);
    if scales.len() != n_blocks_needed {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "scales has {} entries; need {} for {} nibbles @ block_size={}",
            scales.len(),
            n_blocks_needed,
            n,
            block_size
        )));
    }
    let total = n_blocks_needed * block_size;
    let mut out: Vec<f32> = Vec::with_capacity(total);
    for (b, &scale) in scales.iter().enumerate().take(n_blocks_needed) {
        for j in 0..block_size {
            let pos = b * block_size + j;
            let nibble = if pos < n { nibbles[pos] } else { 0 };
            let sign_bit = (nibble >> 3) & 0x1;
            let mag_idx = (nibble & 0x7) as usize;
            let q = pos_levels[mag_idx];
            let v = if sign_bit == 1 { -q } else { q };
            out.push(v * scale);
        }
    }
    let cap = n_values.unwrap_or(total).min(out.len());
    out.truncate(cap);
    Ok(out)
}

/// Pack a `uint8` nibble array (values in `[0, 15]`) to bytes.
///
/// Two nibbles per byte: `hi << 4 | lo`. Length must be even.
/// Mirrors `tac.packet_compiler.pack_nibbles`.
pub fn pack_nibbles(nibbles: &[u8]) -> Result<Vec<u8>> {
    for (i, &n) in nibbles.iter().enumerate() {
        if n > 0xF {
            return Err(PacketCompilerError::GoldenVectorIo(format!(
                "nibble values must fit in 4 bits ([0, 15]); got nibbles[{i}] = {n}"
            )));
        }
    }
    if nibbles.len() & 1 != 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "nibble count must be even for clean packing; got {}",
            nibbles.len()
        )));
    }
    let mut out: Vec<u8> = Vec::with_capacity(nibbles.len() / 2);
    for chunk in nibbles.chunks_exact(2) {
        let hi = chunk[0] & 0xF;
        let lo = chunk[1] & 0xF;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

/// Inverse of [`pack_nibbles`]. Mirrors `tac.packet_compiler.unpack_nibbles`.
pub fn unpack_nibbles(packed: &[u8], count: usize) -> Result<Vec<u8>> {
    if count > 2 * packed.len() {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "count {} exceeds available nibbles {}",
            count,
            2 * packed.len()
        )));
    }
    let mut out: Vec<u8> = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        out.push((byte >> 4) & 0xF);
        out.push(byte & 0xF);
    }
    out.truncate(count);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_zero_block() {
        let cb = FP4Codebook::default();
        let values = vec![0.0_f32; 32];
        let scales = vec![1.0_f32; 1];
        let nibbles = cb.quantize(&values, &scales, 32).unwrap();
        // All zero → nibble 0 (sign 0, mag_idx 0).
        assert!(nibbles.iter().all(|&n| n == 0));
        let recovered = cb.dequantize(&nibbles, &scales, 32, Some(values.len())).unwrap();
        for (i, &v) in recovered.iter().enumerate() {
            assert!(v.abs() < 1e-6, "recovered[{i}] = {v}");
        }
    }

    #[test]
    fn signed_quantization() {
        let cb = FP4Codebook::default();
        // Single block of 32 values: +1.0, -1.0, +2.0, -3.0, +6.0, -6.0, 0, 0, ...
        let mut values = vec![0.0_f32; 32];
        values[0] = 1.0;
        values[1] = -1.0;
        values[2] = 2.0;
        values[3] = -3.0;
        values[4] = 6.0;
        values[5] = -6.0;
        let scales = vec![1.0_f32; 1];
        let nibbles = cb.quantize(&values, &scales, 32).unwrap();
        // Expected: (sign << 3) | mag_idx where mag_idx = index of nearest level
        assert_eq!(nibbles[0], 0b0_010); // +1.0 → sign=0, mag=index 2 (1.0)
        assert_eq!(nibbles[1], 0b1_010); // -1.0 → sign=1, mag=index 2 (1.0)
        assert_eq!(nibbles[2], 0b0_100); // +2.0 → sign=0, mag=index 4 (2.0)
        assert_eq!(nibbles[3], 0b1_101); // -3.0 → sign=1, mag=index 5 (3.0)
        assert_eq!(nibbles[4], 0b0_111); // +6.0 → sign=0, mag=index 7 (6.0)
        assert_eq!(nibbles[5], 0b1_111); // -6.0 → sign=1, mag=index 7 (6.0)
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let nibbles: Vec<u8> = (0..16).map(|n| (n & 0xF) as u8).collect();
        let packed = pack_nibbles(&nibbles).unwrap();
        assert_eq!(packed.len(), 8);
        // hi=0, lo=1 → 0x01; hi=2, lo=3 → 0x23; ...
        assert_eq!(packed[0], 0x01);
        assert_eq!(packed[1], 0x23);
        assert_eq!(packed[7], 0xEF);
        let unpacked = unpack_nibbles(&packed, nibbles.len()).unwrap();
        assert_eq!(unpacked, nibbles);
    }

    #[test]
    fn pack_rejects_odd_count() {
        let nibbles = vec![0u8; 3];
        let r = pack_nibbles(&nibbles);
        assert!(matches!(r, Err(PacketCompilerError::GoldenVectorIo(_))));
    }

    #[test]
    fn pack_rejects_out_of_range_nibble() {
        let nibbles = vec![0u8, 16];
        let r = pack_nibbles(&nibbles);
        assert!(matches!(r, Err(PacketCompilerError::GoldenVectorIo(_))));
    }

    #[test]
    fn with_levels_rejects_negative() {
        let mut bad = PR81_POS_LEVELS;
        bad[3] = -1.0;
        assert!(FP4Codebook::with_levels(bad).is_err());
    }

    #[test]
    fn with_levels_rejects_non_monotonic() {
        let bad = [0.0_f32, 0.5, 1.0, 1.5, 1.0, 3.0, 4.0, 6.0];
        assert!(FP4Codebook::with_levels(bad).is_err());
    }
}
