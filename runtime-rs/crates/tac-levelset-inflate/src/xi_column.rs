// SPDX-License-Identifier: MIT OR Apache-2.0
//! Store-nothing ξ pose-carrier COLUMN decode (the on-path #257 unit).
//!
//! Bit-for-bit mirror of `tac.boundary_math.xi_pose_coder._channel_delta_decode`,
//! MINUS the brotli-of-counts step (a separately-covered primitive — the caller
//! supplies the already-decompressed `frequencies`). One ξ channel is stored as a
//! seed + a temporal-delta stream range-coded against a per-channel histogram; this
//! reconstructs the integer column `q[:, k]` the inflate dequantizes to ξ.
//!
//! Pure INTEGER: arithmetic decode → `+lo` offset → prefix-sum → `seed`. Bit-exact by
//! construction (no float, no transcendental, no reduction-order dependence).

use crate::range_decode::decode_static_symbols;
use crate::Result;

/// Reconstruct one ξ channel's integer column of length `p_count`.
///
/// `seed` is `col[0]`; the remaining `p_count - 1` entries are `seed + cumsum(δ)` where
/// `δ = decode_static_symbols(stream, p_count-1, frequencies) + lo` (the `lo`-offset
/// reverses the encoder's `bincount((deltas - lo))` shift). Mirrors
/// `_channel_delta_decode`'s `col[1:] = seed + np.cumsum(d)`.
pub fn decode_xi_column(
    seed: i64,
    lo: i64,
    frequencies: &[u32],
    stream: &[u8],
    p_count: usize,
) -> Result<Vec<i64>> {
    let mut col = vec![0i64; p_count];
    if p_count == 0 {
        return Ok(col);
    }
    col[0] = seed;
    if p_count > 1 {
        let syms = decode_static_symbols(stream, p_count - 1, frequencies)?;
        let mut acc = seed;
        for (i, s) in syms.iter().enumerate() {
            acc += s + lo; // prefix-sum of (symbol + lo), added onto the running seed
            col[i + 1] = acc;
        }
    }
    Ok(col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_count_one_returns_seed_only() {
        assert_eq!(decode_xi_column(4001, -899, &[1, 1], &[], 1).unwrap(), vec![4001]);
    }

    #[test]
    fn p_count_zero_returns_empty() {
        assert_eq!(decode_xi_column(0, 0, &[1, 1], &[], 0).unwrap(), Vec::<i64>::new());
    }
}
