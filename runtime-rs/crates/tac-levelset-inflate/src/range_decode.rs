// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pure-integer static-frequency arithmetic (CACM range) DECODER.
//!
//! Bit-for-bit mirror of `tac.lossless.range_coder.decode_static_symbols`. This is
//! the ONE level-set inflate primitive that is bit-exact-by-CONSTRUCTION: it is pure
//! INTEGER arithmetic (32-bit range state, `u64` intermediates) with zero dependence
//! on FMA, floating-point rounding, matmul reduction order, or a `libm` transcendental,
//! so a portable Rust port reproduces the Python oracle byte-for-byte.
//!
//! # Magnitudes (why `u64` is exact, not merely close)
//!
//! The range state `low`/`high`/`code` stay in `[0, 2^32)`. The largest intermediate is
//! `(code - low + 1) * total` where `code - low + 1 ≤ 2^32` and `total = Σ freq`. For the
//! real #257 store-nothing ξ payload the max `total` is 7706 ⇒ the product ≤ `2^32 · 7706`
//! ≈ `2^44.9 < 2^64`. `u64` holds every intermediate exactly; the floor divisions match
//! Python's `//` for non-negative operands.

use crate::{LevelSetInflateError, Result};

const STATE_BITS: u32 = 32;
const FULL_RANGE: u64 = 1 << STATE_BITS;
const HALF: u64 = FULL_RANGE >> 1;
const QUARTER: u64 = HALF >> 1;
const THREE_QUARTERS: u64 = QUARTER * 3;

/// MSB-first bit reader over a byte slice; returns 0 once the stream is exhausted
/// (mirror of `range_coder._BitReader.read`).
struct BitReader<'a> {
    data: &'a [u8],
    byte_index: usize,
    bit_index: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, byte_index: 0, bit_index: 0 }
    }

    #[inline]
    fn read(&mut self) -> u64 {
        if self.byte_index >= self.data.len() {
            return 0;
        }
        let byte = self.data[self.byte_index];
        let bit = (byte >> (7 - self.bit_index)) & 1;
        self.bit_index += 1;
        if self.bit_index == 8 {
            self.bit_index = 0;
            self.byte_index += 1;
        }
        bit as u64
    }
}

/// Build the exclusive cumulative-frequency table and total, validating that
/// every frequency is ≥ 1 (mirror of `range_coder._cumulative`). Returns
/// `(cumulative[0..=n], total)` where `cumulative[0] == 0` and `cumulative[n] == total`.
fn cumulative(frequencies: &[u32]) -> Result<(Vec<u64>, u64)> {
    if frequencies.is_empty() {
        return Err(LevelSetInflateError::InvalidFrequencies(
            "frequency table is empty".to_string(),
        ));
    }
    let mut cum = Vec::with_capacity(frequencies.len() + 1);
    cum.push(0u64);
    let mut running: u64 = 0;
    for (i, &f) in frequencies.iter().enumerate() {
        if f == 0 {
            return Err(LevelSetInflateError::InvalidFrequencies(format!(
                "frequency[{i}] must be positive"
            )));
        }
        running += f as u64;
        cum.push(running);
    }
    if running == 0 {
        return Err(LevelSetInflateError::InvalidFrequencies(
            "frequencies must sum to a positive total".to_string(),
        ));
    }
    Ok((cum, running))
}

/// Decode `count` symbols from a static-frequency range-coded `encoded` stream.
///
/// Byte-for-byte mirror of `tac.lossless.range_coder.decode_static_symbols`:
/// initialise the 32-bit code register, then per symbol scale the code into the
/// frequency space, `bisect_right`-locate the symbol, narrow `[low, high)`, and
/// renormalise (E1/E2/E3 CACM cases) feeding fresh code bits.
///
/// Returns the decoded symbol indices as `i64` (the oracle returns Python `int`s;
/// symbols are non-negative alphabet indices in `[0, frequencies.len())`).
pub fn decode_static_symbols(encoded: &[u8], count: usize, frequencies: &[u32]) -> Result<Vec<i64>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if encoded.is_empty() {
        return Err(LevelSetInflateError::InconsistentStream(
            "encoded range stream is empty".to_string(),
        ));
    }
    let (cum, total) = cumulative(frequencies)?;

    let mut reader = BitReader::new(encoded);
    let mut low: u64 = 0;
    let mut high: u64 = FULL_RANGE - 1;
    let mut code: u64 = 0;
    for _ in 0..STATE_BITS {
        code = (code << 1) | reader.read();
    }

    let mut restored = Vec::with_capacity(count);
    for _ in 0..count {
        let current_range = high - low + 1;
        let code_minus_low = code.checked_sub(low).ok_or_else(|| {
            LevelSetInflateError::InconsistentStream("code < low (invariant broken)".to_string())
        })?;
        let scaled = ((code_minus_low + 1) * total - 1) / current_range;
        if scaled >= total {
            return Err(LevelSetInflateError::InconsistentStream(format!(
                "scaled target {scaled} >= total {total}"
            )));
        }
        // bisect_right(cum, scaled) - 1: the largest index i with cum[i] <= scaled.
        // `partition_point(|x| x <= scaled)` returns the count of leading elements
        // satisfying the predicate == bisect_right index (cum is strictly increasing).
        let symbol = cum.partition_point(|&x| x <= scaled) - 1;
        if symbol + 1 >= cum.len() {
            return Err(LevelSetInflateError::InconsistentStream(
                "decoded symbol outside frequency table".to_string(),
            ));
        }
        restored.push(symbol as i64);

        high = low + (current_range * cum[symbol + 1] / total) - 1;
        low += current_range * cum[symbol] / total;

        loop {
            if high < HALF {
                // E1: pass — fall through to the shift.
            } else if low >= HALF {
                low -= HALF;
                high -= HALF;
                code -= HALF;
            } else if low >= QUARTER && high < THREE_QUARTERS {
                low -= QUARTER;
                high -= QUARTER;
                code -= QUARTER;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
            code = (code << 1) | reader.read();
        }
    }
    Ok(restored)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_count_returns_empty() {
        assert_eq!(decode_static_symbols(&[0xFF], 0, &[1, 1]).unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn zero_frequency_rejected() {
        let err = decode_static_symbols(&[0x00], 1, &[1, 0, 1]).unwrap_err();
        matches!(err, LevelSetInflateError::InvalidFrequencies(_));
    }

    #[test]
    fn empty_stream_with_count_rejected() {
        let err = decode_static_symbols(&[], 1, &[1, 1]).unwrap_err();
        matches!(err, LevelSetInflateError::InconsistentStream(_));
    }
}
