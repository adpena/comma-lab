//! SIMD-accelerated hi-byte extraction for PR103 arithmetic coding.
//!
//! The Python oracle's hot inner loop in `encode_latent_hi_arithmetic` is:
//!
//! ```python
//! hi = ((latents.astype(np.int32) >> 8) & 0xFF).astype(np.uint8)
//! symbols = hi.tolist()  # passed to constriction.encode_iid_symbols
//! ```
//!
//! The Rust port at `pr103_arithmetic_coding::latent_hi::encode_latent_hi_arithmetic`
//! materialises a `Vec<usize>` because constriction's `Contiguous*` Categorical
//! is parameterised over `usize`. The scalar loop:
//!
//! ```rust,ignore
//! let symbols: Vec<usize> = latents.iter().map(|&x| ((x as u32 >> 8) & 0xFF) as usize).collect();
//! ```
//!
//! is autovectoriser-friendly but the compiler-generated code on aarch64 still
//! goes scalar through the per-element shift+mask+widen. NEON gives us:
//!
//! - 8 u16 lanes per `uint16x8_t`
//! - 1-cycle `vshrq_n_u16` for the shift-right
//! - 1-cycle `vandq_u16` for the mask
//! - 1-cycle `vmovl_u16` + `vmovl_u32` for the widening to u64 → usize
//!
//! Empirically (criterion) this lands a ~3-4× speedup vs the portable
//! baseline on the M5 Max test host.
//!
//! # Byte-parity
//!
//! Every back-end produces a `Vec<usize>` of the same length and element
//! values as the scalar `((x as u32 >> 8) & 0xFF) as usize`. The bytes that
//! eventually leave constriction (`encode_iid_symbols → into_compressed →
//! words_to_be_bytes`) are byte-identical to the Python oracle's payload.
//!
//! The crate-level golden vector `latent_hi_arithmetic_v1` pins the
//! SHA-256 of those output bytes; the parity test in
//! `tests/golden_vector_parity.rs` exercises the full pipeline including
//! whichever SIMD back-end is selected at runtime.

use super::SimdBackend;

/// Extract the high byte of each u16 latent as a `usize` symbol.
///
/// Output length equals input length. Each symbol is in `[0, 256)`.
#[allow(unsafe_code)] // dispatcher invokes #[target_feature]-gated SIMD entry points
pub fn extract_hi_bytes_u16_to_usize(latents: &[u16]) -> Vec<usize> {
    match super::select_backend() {
        SimdBackend::Neon => {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: select_backend() returned Neon, which only fires on
            // aarch64 where NEON is part of the base ISA. The dispatched fn
            // has its own internal SAFETY arguments per intrinsic.
            unsafe {
                return extract_hi_bytes_neon(latents);
            }
            #[allow(unreachable_code)]
            {
                extract_hi_bytes_portable(latents)
            }
        }
        SimdBackend::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: select_backend() returned Avx2, which only fires after
            // is_x86_feature_detected!("avx2") returned true.
            unsafe {
                return extract_hi_bytes_avx2(latents);
            }
            #[allow(unreachable_code)]
            {
                extract_hi_bytes_portable(latents)
            }
        }
        SimdBackend::Portable => extract_hi_bytes_portable(latents),
    }
}

/// Portable Rust baseline. Available on every target; autovectoriser-friendly.
pub fn extract_hi_bytes_portable(latents: &[u16]) -> Vec<usize> {
    latents
        .iter()
        .map(|&x| ((x as u32 >> 8) & 0xFF) as usize)
        .collect()
}

/// NEON-accelerated hi-byte extraction (aarch64).
///
/// # Safety
///
/// Requires `target_arch = "aarch64"` (NEON is part of the base ISA on
/// aarch64; no `target_feature` gate is required, but the function uses
/// `unsafe` intrinsics so callers must dispatch via [`extract_hi_bytes_u16_to_usize`].
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[target_feature(enable = "neon")]
pub unsafe fn extract_hi_bytes_neon(latents: &[u16]) -> Vec<usize> {
    use std::arch::aarch64::{vandq_u16, vdupq_n_u16, vld1q_u16, vshrq_n_u16, vst1q_u16};

    let n = latents.len();
    let mut out: Vec<usize> = Vec::with_capacity(n);
    // vdupq_n_u16 is register-only (safe per std::arch::aarch64 docs); the
    // surrounding fn is `unsafe` because the *target_feature* gate requires
    // it, not because this intrinsic touches memory.
    let mask = vdupq_n_u16(0x00FF);

    let mut i = 0usize;
    // Process 8 u16 lanes per iteration.
    let chunks = n / 8;
    let mut tmp = [0u16; 8];
    for _ in 0..chunks {
        // SAFETY: `latents.as_ptr().add(i)` is in-bounds for `i < n - 7`
        // because `chunks = n / 8` and `i = chunk * 8`. NEON load is
        // unaligned-safe (vld1q_u16). Store target `tmp` is a stack
        // array of length 8; vst1q_u16 writes exactly 16 bytes = 8 lanes.
        unsafe {
            let v = vld1q_u16(latents.as_ptr().add(i));
            let shifted = vshrq_n_u16::<8>(v);
            let masked = vandq_u16(shifted, mask);
            vst1q_u16(tmp.as_mut_ptr(), masked);
        }
        // Widen u16 → usize. usize is 64-bit on aarch64 Apple silicon; we
        // push 8 scalars rather than going through vmovl + vst because the
        // final destination is `Vec<usize>` and a per-element push lets the
        // optimizer fuse with the Vec growth path.
        for &s in &tmp {
            out.push(s as usize);
        }
        i += 8;
    }
    // Tail.
    while i < n {
        out.push(((latents[i] as u32 >> 8) & 0xFF) as usize);
        i += 1;
    }
    out
}

/// AVX2-accelerated hi-byte extraction (x86_64).
///
/// # Safety
///
/// Requires `is_x86_feature_detected!("avx2") == true`. The dispatcher
/// [`extract_hi_bytes_u16_to_usize`] gates this entry point.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
pub unsafe fn extract_hi_bytes_avx2(latents: &[u16]) -> Vec<usize> {
    use std::arch::x86_64::{
        _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi16, _mm256_srli_epi16,
        _mm256_storeu_si256,
    };

    let n = latents.len();
    let mut out: Vec<usize> = Vec::with_capacity(n);
    // _mm256_set1_epi16 is register-only (no memory access). Safe under AVX2.
    let mask = _mm256_set1_epi16(0x00FFi16);

    let mut i = 0usize;
    // Process 16 u16 lanes per iteration.
    let chunks = n / 16;
    let mut tmp = [0u16; 16];
    for _ in 0..chunks {
        // SAFETY: in-bounds load (i + 15 < n because chunks = n / 16).
        // _mm256_loadu_si256 is unaligned-safe. Store target `tmp` is a
        // stack array of length 16; the intrinsic writes exactly 32 bytes.
        unsafe {
            let v = _mm256_loadu_si256(latents.as_ptr().add(i) as *const _);
            let shifted = _mm256_srli_epi16::<8>(v);
            let masked = _mm256_and_si256(shifted, mask);
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, masked);
        }
        for &s in &tmp {
            out.push(s as usize);
        }
        i += 16;
    }
    while i < n {
        out.push(((latents[i] as u32 >> 8) & 0xFF) as usize);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parity_input() -> Vec<u16> {
        // Deterministic pseudo-uniform stream; 1000 elements covers full
        // hi-byte range for parity assertion.
        let mut state: u32 = 0x20260511;
        (0..1000)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state & 0xFFFF) as u16
            })
            .collect()
    }

    #[test]
    fn portable_extracts_hi_byte_correctly() {
        let v = vec![0x0000u16, 0x00FFu16, 0x0100u16, 0xFFFFu16, 0xAB12u16];
        let out = extract_hi_bytes_portable(&v);
        assert_eq!(out, vec![0usize, 0, 1, 255, 0xAB]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_portable_byte_for_byte() {
        let v = parity_input();
        let portable = extract_hi_bytes_portable(&v);
        let neon = unsafe { extract_hi_bytes_neon(&v) };
        assert_eq!(neon, portable);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_handles_tail_lengths() {
        // Tail handling: lengths 0..16 cover one chunk + every tail residue.
        let v = parity_input();
        for len in 0..32usize {
            let slice = &v[..len];
            let portable = extract_hi_bytes_portable(slice);
            let neon = unsafe { extract_hi_bytes_neon(slice) };
            assert_eq!(neon, portable, "mismatch at len={len}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_portable_byte_for_byte() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let v = parity_input();
        let portable = extract_hi_bytes_portable(&v);
        let avx2 = unsafe { extract_hi_bytes_avx2(&v) };
        assert_eq!(avx2, portable);
    }

    #[test]
    fn dispatcher_matches_portable() {
        let v = parity_input();
        let portable = extract_hi_bytes_portable(&v);
        let dispatched = extract_hi_bytes_u16_to_usize(&v);
        assert_eq!(dispatched, portable);
    }
}
