//! SIMD-accelerated centered-delta transform for PR101 raw-LZMA payload.
//!
//! The Python oracle `tac.packet_compiler.encode_centered_delta_uint8` does:
//!
//! ```python
//! q = ((values - mins) / scales).round().clip(0, 255).astype(np.uint8)
//! base = q[0]
//! deltas = (np.diff(q.astype(np.int32), axis=0) + 128) & 0xFF
//! column_major = np.column_stack([base[None, :], deltas]).T.reshape(-1)
//! # then LZMA1-encode [mins_bytes || scales_bytes || column_major]
//! ```
//!
//! The quantise + delta + centered-mod-256 steps are scalar in pure Python
//! and well-suited to SIMD: 8 u8 lanes can compute the centered delta
//! per cycle on NEON or AVX2. Importantly the LZMA1-encode step is the
//! wire-format authority (we cannot replace it), but the **pre-LZMA buffer
//! bytes** must be byte-identical or the LZMA output will differ. Our SIMD
//! kernel produces the pre-LZMA buffer; the existing scalar
//! `lzma1_raw_compress` consumes it unchanged.
//!
//! # API
//!
//! Lower-level than the existing `encode_centered_delta_uint8`: it takes the
//! already-quantised `q` row-major matrix and emits the column-major
//! centered-delta block. This lets us SIMD-accelerate the bottleneck without
//! reworking the per-column fp16 calibration path (which is already
//! sub-microsecond for n_dims=6 typical).

use super::SimdBackend;

/// Compute the centered-delta column-major block from a row-major q matrix.
///
/// `q` is `(n_pairs, n_dims)` row-major. Output is `n_pairs * n_dims` bytes
/// laid out as: `[col0_base, col0_delta_0, …, col0_delta_{n_pairs-2}, col1_base, …]`.
#[allow(unsafe_code)] // dispatcher invokes #[target_feature]-gated SIMD entry points
pub fn centered_delta_uint8_column_major(q: &[u8], n_pairs: usize, n_dims: usize) -> Vec<u8> {
    assert_eq!(q.len(), n_pairs * n_dims, "q shape mismatch");
    match super::select_backend() {
        SimdBackend::Neon => {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: select_backend() returned Neon, ISA available on aarch64.
            unsafe {
                return centered_delta_neon(q, n_pairs, n_dims);
            }
            #[allow(unreachable_code)]
            {
                centered_delta_portable(q, n_pairs, n_dims)
            }
        }
        SimdBackend::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: AVX2 feature-detected.
            unsafe {
                return centered_delta_avx2(q, n_pairs, n_dims);
            }
            #[allow(unreachable_code)]
            {
                centered_delta_portable(q, n_pairs, n_dims)
            }
        }
        SimdBackend::Portable => centered_delta_portable(q, n_pairs, n_dims),
    }
}

/// Portable Rust baseline.
pub fn centered_delta_portable(q: &[u8], n_pairs: usize, n_dims: usize) -> Vec<u8> {
    let mut out = vec![0u8; n_pairs * n_dims];
    for col in 0..n_dims {
        out[col * n_pairs] = q[col]; // base = q[0, col]
        for row in 1..n_pairs {
            let prev = q[(row - 1) * n_dims + col] as i32;
            let cur = q[row * n_dims + col] as i32;
            let diff = cur - prev;
            let centered = ((diff + 128) & 0xFF) as u8;
            out[col * n_pairs + row] = centered;
        }
    }
    out
}

/// NEON-accelerated centered-delta (aarch64).
///
/// Strategy: process `n_dims` columns in parallel within each row-pair.
/// When `n_dims >= 16` we use a single `vsubq_u8` + `vaddq_u8(128)` per
/// row-pair. For typical `n_dims=6` we still use scalar (16-byte SIMD wastes
/// 10 lanes); the heuristic falls back to the portable path when n_dims < 16.
///
/// # Safety
///
/// Requires `target_arch = "aarch64"`.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[target_feature(enable = "neon")]
pub unsafe fn centered_delta_neon(q: &[u8], n_pairs: usize, n_dims: usize) -> Vec<u8> {
    // Heuristic: small n_dims doesn't benefit from 16-wide NEON.
    if n_dims < 16 {
        return centered_delta_portable(q, n_pairs, n_dims);
    }
    use std::arch::aarch64::{vaddq_u8, vdupq_n_u8, vld1q_u8, vst1q_u8, vsubq_u8};

    let mut out = vec![0u8; n_pairs * n_dims];
    // Base row.
    for col in 0..n_dims {
        out[col * n_pairs] = q[col];
    }
    let mut row_deltas = vec![0u8; n_dims];
    // vdupq_n_u8 is register-only (no memory access).
    let bias = vdupq_n_u8(128);
    for row in 1..n_pairs {
        let prev_row = (row - 1) * n_dims;
        let cur_row = row * n_dims;
        let mut col = 0usize;
        while col + 16 <= n_dims {
            // SAFETY: in-bounds 16-byte loads/stores: `prev_row + col + 15
            // < (row - 1) * n_dims + n_dims = row * n_dims <= n_pairs * n_dims`.
            // Similarly `cur_row + col + 15 < (row + 1) * n_dims`. Store
            // target `row_deltas` is of length n_dims; col + 15 < n_dims by
            // the while condition.
            unsafe {
                let p = vld1q_u8(q.as_ptr().add(prev_row + col));
                let c = vld1q_u8(q.as_ptr().add(cur_row + col));
                // (c - p) wrapping mod 256 == ((cur - prev) & 0xFF) in the
                // Python signed model (numpy uses wrapping uint arithmetic);
                // adding 128 then masking gives the centered delta.
                let diff = vsubq_u8(c, p);
                let centered = vaddq_u8(diff, bias);
                vst1q_u8(row_deltas.as_mut_ptr().add(col), centered);
            }
            col += 16;
        }
        // Tail (n_dims not a multiple of 16).
        while col < n_dims {
            let prev = q[prev_row + col] as i32;
            let cur = q[cur_row + col] as i32;
            let diff = cur - prev;
            row_deltas[col] = ((diff + 128) & 0xFF) as u8;
            col += 1;
        }
        // Scatter row_deltas to column-major output. Sequential within a
        // column; iterate columns so destination indices are also sequential
        // per column-group.
        for c in 0..n_dims {
            out[c * n_pairs + row] = row_deltas[c];
        }
    }
    out
}

/// AVX2-accelerated centered-delta (x86_64).
///
/// # Safety
///
/// Requires `is_x86_feature_detected!("avx2") == true`.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
pub unsafe fn centered_delta_avx2(q: &[u8], n_pairs: usize, n_dims: usize) -> Vec<u8> {
    if n_dims < 32 {
        return centered_delta_portable(q, n_pairs, n_dims);
    }
    use std::arch::x86_64::{
        _mm256_add_epi8, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_storeu_si256, _mm256_sub_epi8,
    };

    let mut out = vec![0u8; n_pairs * n_dims];
    for col in 0..n_dims {
        out[col * n_pairs] = q[col];
    }
    let mut row_deltas = vec![0u8; n_dims];
    // set1 is register-only.
    let bias = _mm256_set1_epi8(128u8 as i8);
    for row in 1..n_pairs {
        let prev_row = (row - 1) * n_dims;
        let cur_row = row * n_dims;
        let mut col = 0usize;
        while col + 32 <= n_dims {
            // SAFETY: in-bounds 32-byte loads/stores; same argument as the
            // NEON path above scaled to 32 lanes.
            unsafe {
                let p = _mm256_loadu_si256(q.as_ptr().add(prev_row + col) as *const _);
                let c = _mm256_loadu_si256(q.as_ptr().add(cur_row + col) as *const _);
                let diff = _mm256_sub_epi8(c, p);
                let centered = _mm256_add_epi8(diff, bias);
                _mm256_storeu_si256(row_deltas.as_mut_ptr().add(col) as *mut _, centered);
            }
            col += 32;
        }
        while col < n_dims {
            let prev = q[prev_row + col] as i32;
            let cur = q[cur_row + col] as i32;
            let diff = cur - prev;
            row_deltas[col] = ((diff + 128) & 0xFF) as u8;
            col += 1;
        }
        for c in 0..n_dims {
            out[c * n_pairs + row] = row_deltas[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parity_input(n_pairs: usize, n_dims: usize) -> Vec<u8> {
        let mut state: u32 = 0x20260511;
        (0..n_pairs * n_dims)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                state as u8
            })
            .collect()
    }

    #[test]
    fn portable_known_value() {
        // 3 pairs × 2 dims; q = [[10,20],[15,18],[14,30]]
        let q = vec![10u8, 20, 15, 18, 14, 30];
        // Column 0: base=10, delta_1=(15-10)+128=133, delta_2=(14-15)+128=127
        // Column 1: base=20, delta_1=(18-20)+128=126, delta_2=(30-18)+128=140
        let expected = vec![10u8, 133, 127, 20, 126, 140];
        let out = centered_delta_portable(&q, 3, 2);
        assert_eq!(out, expected);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_portable_byte_for_byte_small() {
        // n_dims < 16 → both paths take portable; verify dispatcher.
        let q = parity_input(40, 6);
        let portable = centered_delta_portable(&q, 40, 6);
        let neon = unsafe { centered_delta_neon(&q, 40, 6) };
        assert_eq!(neon, portable);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_portable_byte_for_byte_wide() {
        // n_dims >= 16 → NEON inner loop active.
        let q = parity_input(64, 32);
        let portable = centered_delta_portable(&q, 64, 32);
        let neon = unsafe { centered_delta_neon(&q, 64, 32) };
        assert_eq!(neon, portable);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_handles_tail_width() {
        let q = parity_input(8, 17); // 17 = 16 + 1 (tail)
        let portable = centered_delta_portable(&q, 8, 17);
        let neon = unsafe { centered_delta_neon(&q, 8, 17) };
        assert_eq!(neon, portable);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_portable_byte_for_byte_wide() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let q = parity_input(64, 64);
        let portable = centered_delta_portable(&q, 64, 64);
        let avx2 = unsafe { centered_delta_avx2(&q, 64, 64) };
        assert_eq!(avx2, portable);
    }

    #[test]
    fn dispatcher_matches_portable_typical() {
        // PR101 typical: n_pairs=40, n_dims=6 (pose 6D delta).
        let q = parity_input(40, 6);
        let portable = centered_delta_portable(&q, 40, 6);
        let dispatched = centered_delta_uint8_column_major(&q, 40, 6);
        assert_eq!(dispatched, portable);
    }
}
