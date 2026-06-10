// SPDX-License-Identifier: MIT OR Apache-2.0
//! d_seg popcount kernel — native port of `tac.boundary_math.bitmask_dseg`.
//!
//! The exact seg distortion `evaluate.py` scores is the per-pixel argmax
//! disagreement rate: `d_seg = (argmax(out1) != argmax(out2)).mean()`. On two
//! integer label arrays this is a popcount(XOR) functional:
//!
//!   - [`flip_count`] = number of pixels where the labels differ (the seg debt).
//!   - [`d_seg`] = `flip_count / n_pixels` (the rate).
//!
//! Per the Python `d_seg_bitmask` proof: a single flip from class a→b toggles
//! exactly two class one-hot planes, so `sum_c popcount(XOR(mask_c, gt_c)) == 2 *
//! flip_count`. We compute `flip_count` directly (the planes form is equivalent
//! and double-counts by 2) — both Python forms agree bit-for-bit, and so does
//! this kernel (`dseg_popcount_v1` golden vector pins `flip_count || d_seg`).
//!
//! PAYLOAD CLEANLINESS: this kernel reads two label arrays from the caller; no
//! constants beyond `n_pixels` arithmetic. No learned/video data embedded.

use crate::{BoundaryDecodeError, Result};

/// Number of pixels where the candidate and gt labels differ.
///
/// Byte-for-byte equal to Python `flip_count(cand, gt)` =
/// `int(np.count_nonzero(cand != gt))`.
pub fn flip_count(candidate: &[u8], gt: &[u8]) -> Result<u64> {
    if candidate.len() != gt.len() {
        return Err(BoundaryDecodeError::LengthMismatch {
            a: candidate.len(),
            b: gt.len(),
        });
    }
    // The popcount(XOR) form: a label-byte XOR is nonzero iff the labels differ.
    // Counting nonzero-XOR pixels == counting disagreements == the flip count.
    let mut flips: u64 = 0;
    for (c, g) in candidate.iter().zip(gt.iter()) {
        flips += ((c ^ g) != 0) as u64;
    }
    Ok(flips)
}

/// The seg distortion rate `flip_count / n_pixels` (f64, matches Python).
///
/// Byte-for-byte equal to Python `d_seg_reference(cand, gt)` =
/// `float(np.mean(cand != gt))`. `np.mean` over a boolean array divides the
/// True-count by the element count in f64 — identical to `flips / n` here.
pub fn d_seg(candidate: &[u8], gt: &[u8]) -> Result<f64> {
    let flips = flip_count(candidate, gt)?;
    let n = candidate.len();
    if n == 0 {
        // np.mean of empty is nan; mirror that rather than divide-by-zero panic.
        return Ok(f64::NAN);
    }
    Ok(flips as f64 / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_arrays_zero_flips() {
        let a = vec![0u8, 1, 2, 3, 4, 0, 1];
        assert_eq!(flip_count(&a, &a).unwrap(), 0);
        assert_eq!(d_seg(&a, &a).unwrap(), 0.0);
    }

    #[test]
    fn counts_only_disagreements() {
        let a = vec![0u8, 1, 2, 3];
        let b = vec![0u8, 9, 2, 7];
        assert_eq!(flip_count(&a, &b).unwrap(), 2);
        assert_eq!(d_seg(&a, &b).unwrap(), 0.5);
    }

    #[test]
    fn length_mismatch_fails_closed() {
        assert!(flip_count(&[0, 1], &[0]).is_err());
    }

    #[test]
    fn popcount_xor_equals_argmax_compare() {
        // The popcount(XOR) form must equal the direct disagreement count even
        // when labels differ by more than 1 (XOR of e.g. 3^4 = 7 is nonzero).
        let cand = vec![3u8, 0, 4, 4, 1];
        let gt = vec![4u8, 0, 4, 1, 1];
        assert_eq!(flip_count(&cand, &gt).unwrap(), 2);
    }
}
