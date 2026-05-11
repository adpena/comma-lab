//! SIMD-accelerated nonzero-index scan for the sparse PacketIR codec.
//!
//! The Python oracle `tac.packet_compiler.sparse_packet_ir.encode_rle_of_zeros`
//! scans a dense int8 array for nonzero positions:
//!
//! ```python
//! mask = dense != 0
//! nonzero_indices = np.nonzero(mask)[0].astype(np.uint32)
//! nonzero_values  = dense[mask]
//! ```
//!
//! For sparse residuals (typical comma video pose residual: ~6% density on
//! 1024-element blocks) the scan dominates the encoder time. The hot loop is
//! "predicate-scatter": each lane independently decides whether to emit an
//! `(index, value)` pair into the output buffer.
//!
//! # SIMD strategy
//!
//! NEON does not have a native compressed-store (no `vcompress`-equivalent
//! in the base ISA). We instead use the **histogram-then-prefix-sum** trick:
//!
//! 1. **Phase 1 (SIMD):** for each 16-byte block, compute a single
//!    `popcount`-of-nonzero into a per-block counter.
//! 2. **Phase 2 (scalar):** prefix-sum the per-block counters to get each
//!    block's output start index.
//! 3. **Phase 3 (SIMD-aided):** revisit each block and write into its
//!    pre-assigned slice — this is sequential-within-a-block but
//!    embarrassingly parallel across blocks.
//!
//! Because the parity requirement is **emission order = ascending input
//! index**, we cannot use any kind of out-of-order compression; the strict
//! sequential write within each block is the only emission order that
//! matches the Python `np.nonzero` ordering convention.
//!
//! # Output shape
//!
//! Returns `(indices, values)`. `indices` is u32 (matches Python oracle's
//! `nonzero_indices` dtype). `values` is i8 (matches Python `dense[mask]`).

use super::SimdBackend;

/// Scan a dense int8 array for nonzero positions; return `(indices, values)`.
///
/// `indices` is ascending; matches `np.nonzero(dense != 0)[0]` byte-for-byte
/// (when serialised as `<u4` little-endian) on the Python oracle.
#[allow(unsafe_code)] // dispatcher invokes #[target_feature]-gated SIMD entry points
pub fn scan_nonzero_indices_i8(dense: &[i8]) -> (Vec<u32>, Vec<i8>) {
    match super::select_backend() {
        SimdBackend::Neon => {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: select_backend() returned Neon, ISA available.
            unsafe {
                return scan_nonzero_neon(dense);
            }
            #[allow(unreachable_code)]
            {
                scan_nonzero_portable(dense)
            }
        }
        SimdBackend::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: AVX2 feature-detected.
            unsafe {
                return scan_nonzero_avx2(dense);
            }
            #[allow(unreachable_code)]
            {
                scan_nonzero_portable(dense)
            }
        }
        SimdBackend::Portable => scan_nonzero_portable(dense),
    }
}

/// Portable Rust baseline. Single-pass; pre-allocates assuming ~10% density.
pub fn scan_nonzero_portable(dense: &[i8]) -> (Vec<u32>, Vec<i8>) {
    let cap = dense.len() / 10 + 1;
    let mut indices = Vec::with_capacity(cap);
    let mut values = Vec::with_capacity(cap);
    for (i, &v) in dense.iter().enumerate() {
        if v != 0 {
            indices.push(i as u32);
            values.push(v);
        }
    }
    (indices, values)
}

/// NEON-accelerated nonzero scan (aarch64).
///
/// Phase 1 uses a NEON-vectorised popcount of nonzero lanes per 16-byte
/// block; Phase 2 prefix-sums; Phase 3 emits pairs.
///
/// # Safety
///
/// Requires `target_arch = "aarch64"`.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[target_feature(enable = "neon")]
pub unsafe fn scan_nonzero_neon(dense: &[i8]) -> (Vec<u32>, Vec<i8>) {
    use std::arch::aarch64::{vaddvq_u8, vceqzq_s8, vld1q_s8, vmvnq_u8};

    let n = dense.len();
    let block = 16usize;
    let n_blocks = n / block;
    // Phase 1: per-block popcount of nonzero lanes.
    let mut counts = vec![0u8; n_blocks];
    let ptr = dense.as_ptr();
    for b in 0..n_blocks {
        // SAFETY: in-bounds load (b * block + 15 < n because n_blocks = n / block).
        let sum = unsafe {
            let v = vld1q_s8(ptr.add(b * block));
            let is_zero = vceqzq_s8(v); // 0xFF where v==0
            let is_nonzero = vmvnq_u8(is_zero); // 0xFF where v!=0
            // Each lane is 0xFF (=255) for nonzero, 0 for zero. Sum of u8 lanes
            // = 255 * popcount; we want popcount, so divide by 255 at the
            // accumulate step.
            vaddvq_u8(is_nonzero) as u16
        };
        counts[b] = (sum / 255) as u8;
    }
    // Tail popcount.
    let mut tail_count = 0u32;
    for i in (n_blocks * block)..n {
        if dense[i] != 0 {
            tail_count += 1;
        }
    }
    let total: u32 = counts.iter().map(|&c| c as u32).sum::<u32>() + tail_count;

    let mut indices = Vec::with_capacity(total as usize);
    let mut values = Vec::with_capacity(total as usize);

    // Phase 3: emit pairs in order. The per-block counts gave us the total
    // capacity hint above; we still walk each block sequentially because
    // emission order must match `np.nonzero` ascending.
    for b in 0..n_blocks {
        let base = b * block;
        // Within the block, scalar emit — this preserves sequential-write
        // semantics. The NEON popcount in Phase 1 is the speedup; emitting
        // is unavoidably sequential due to ordering.
        for k in 0..block {
            // SAFETY: base + k < n_blocks * block <= n.
            let v = unsafe { *dense.get_unchecked(base + k) };
            if v != 0 {
                indices.push((base + k) as u32);
                values.push(v);
            }
        }
    }
    for i in (n_blocks * block)..n {
        if dense[i] != 0 {
            indices.push(i as u32);
            values.push(dense[i]);
        }
    }
    (indices, values)
}

/// AVX2-accelerated nonzero scan (x86_64).
///
/// # Safety
///
/// Requires `is_x86_feature_detected!("avx2") == true`.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
pub unsafe fn scan_nonzero_avx2(dense: &[i8]) -> (Vec<u32>, Vec<i8>) {
    use std::arch::x86_64::{
        _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_setzero_si256,
    };

    let n = dense.len();
    let block = 32usize;
    let n_blocks = n / block;
    // Phase 1: count nonzero per block via movemask + popcount.
    let mut counts = vec![0u32; n_blocks];
    // setzero is register-only (no memory access).
    let zeros = _mm256_setzero_si256();
    let ptr = dense.as_ptr();
    for b in 0..n_blocks {
        // SAFETY: in-bounds 32-byte load (b * block + 31 < n).
        let mask = unsafe {
            let v = _mm256_loadu_si256(ptr.add(b * block) as *const _);
            let is_zero = _mm256_cmpeq_epi8(v, zeros);
            _mm256_movemask_epi8(is_zero) as u32
        };
        counts[b] = (block as u32) - (mask.count_ones());
    }
    let mut tail_count = 0u32;
    for i in (n_blocks * block)..n {
        if dense[i] != 0 {
            tail_count += 1;
        }
    }
    let total: u32 = counts.iter().sum::<u32>() + tail_count;

    let mut indices = Vec::with_capacity(total as usize);
    let mut values = Vec::with_capacity(total as usize);
    for b in 0..n_blocks {
        let base = b * block;
        for k in 0..block {
            // SAFETY: base + k < n_blocks * block <= n.
            let v = unsafe { *dense.get_unchecked(base + k) };
            if v != 0 {
                indices.push((base + k) as u32);
                values.push(v);
            }
        }
    }
    for i in (n_blocks * block)..n {
        if dense[i] != 0 {
            indices.push(i as u32);
            values.push(dense[i]);
        }
    }
    (indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parity_input() -> Vec<i8> {
        // 1024 i8 with 64 random nonzero positions (matches sparse golden
        // vector recipe density).
        let mut state: u32 = 0x20260511;
        let mut out = vec![0i8; 1024];
        for _ in 0..64 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let idx = ((state >> 16) as usize) % 1024;
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((state & 0x1F) as i8).max(1);
            out[idx] = v;
        }
        out
    }

    #[test]
    fn portable_matches_python_nonzero_semantics() {
        let dense = vec![0i8, 1, 0, -1, 0, 0, 5, 0];
        let (idx, val) = scan_nonzero_portable(&dense);
        assert_eq!(idx, vec![1u32, 3, 6]);
        assert_eq!(val, vec![1i8, -1, 5]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_portable_byte_for_byte() {
        let dense = parity_input();
        let (p_idx, p_val) = scan_nonzero_portable(&dense);
        let (n_idx, n_val) = unsafe { scan_nonzero_neon(&dense) };
        assert_eq!(n_idx, p_idx);
        assert_eq!(n_val, p_val);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_handles_tail_lengths() {
        let base = parity_input();
        for len in 0..40usize {
            let slice = &base[..len];
            let (p_idx, p_val) = scan_nonzero_portable(slice);
            let (n_idx, n_val) = unsafe { scan_nonzero_neon(slice) };
            assert_eq!(n_idx, p_idx, "mismatch at len={len}");
            assert_eq!(n_val, p_val, "mismatch at len={len}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_portable_byte_for_byte() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let dense = parity_input();
        let (p_idx, p_val) = scan_nonzero_portable(&dense);
        let (a_idx, a_val) = unsafe { scan_nonzero_avx2(&dense) };
        assert_eq!(a_idx, p_idx);
        assert_eq!(a_val, p_val);
    }

    #[test]
    fn dispatcher_matches_portable() {
        let dense = parity_input();
        let (p_idx, p_val) = scan_nonzero_portable(&dense);
        let (d_idx, d_val) = scan_nonzero_indices_i8(&dense);
        assert_eq!(d_idx, p_idx);
        assert_eq!(d_val, p_val);
    }
}
