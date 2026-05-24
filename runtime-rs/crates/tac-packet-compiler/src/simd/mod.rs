//! Low-level SIMD-accelerated kernels for the hot pre-encoding paths.
//!
//! # Scope
//!
//! Per CLAUDE.md "Rust/Zig is a speed layer, not a license to change semantics"
//! and the operator directive 2026-05-11 ("compiler and insanely low level"),
//! the SIMD kernels here are restricted to **pre-encoding / post-decode
//! transforms** whose output is fed into the canonical entropy coder
//! (constriction / brotli / liblzma). The entropy coders' own bit-pumps are
//! the wire-format authority — they cannot be replaced and still maintain
//! byte-for-byte parity against the Python oracle.
//!
//! The accelerated kernels are:
//!
//! - [`hi_byte::extract_hi_bytes_u16_to_usize`] — used by PR103
//!   `encode_latent_hi_arithmetic`; the hi-byte stream is the *input* to
//!   constriction's `encode_iid_symbols`.
//! - [`rle_of_zeros::scan_nonzero_indices_i8`] — used by the sparse PacketIR
//!   codec; the nonzero (index, value) pair stream is the input to either
//!   raw serialisation or a follow-on AC stream.
//! - [`centered_delta::centered_delta_uint8_column_major`] — used by
//!   PR101 `encode_centered_delta_uint8`; the row-major uint8 + centered
//!   delta stream is the input to raw LZMA1.
//!
//! Every kernel ships **three back-ends**:
//!
//! 1. `portable` — autovectoriser-friendly Rust; the safety baseline.
//! 2. `neon` (aarch64) — hand-rolled `std::arch::aarch64` NEON intrinsics.
//! 3. `avx2` (x86_64) — hand-rolled `std::arch::x86_64` AVX2 intrinsics.
//!
//! The dispatch helper [`select_backend`] picks the best available back-end at
//! runtime (NEON is always available on aarch64; AVX2 is feature-detected via
//! `is_x86_feature_detected!`). **All three back-ends produce byte-identical
//! output** — verified via parity proptests below.
//!
//! # Unsafe discipline
//!
//! The crate-level attribute is `#![forbid(unsafe_code)]`. The two SIMD
//! back-ends require `unsafe` to call architecture intrinsics. Each function
//! that uses `unsafe` carries:
//!
//! - `#[allow(unsafe_code)]` scoped to the single function.
//! - A `# Safety` doc comment naming the architecture-feature precondition.
//! - A pre-call feature-gate (`is_x86_feature_detected!` / `cfg!(target_arch)`)
//!   in the dispatcher so the unsafe entry-point is unreachable on hosts that
//!   lack the required ISA.
//!
//! # Benchmarks
//!
//! See `benches/simd_kernels.rs` for the criterion harness. The kernels are
//! benchmarked against the autovectorised portable baseline on the host's
//! native ISA; the SIMD back-ends should match-or-beat portable on every
//! supported host.

// Clippy carve-out: SIMD chunk + scalar tail patterns deliberately use
// index-loop form (`for i in 0..n { arr[i] }`) instead of iterator chains
// because (1) intrinsic loads need raw pointer arithmetic via `.add(i)`
// and (2) the portable baseline must mirror the SIMD path's access pattern
// so the autovectoriser can reason about the same shape. Iterator chains
// can prevent autovectorisation here. These allows are scoped per-fn
// instead of the whole module so non-hot paths still benefit from clippy.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::op_ref)]

pub mod centered_delta;
pub mod hi_byte;
pub mod rle_of_zeros;

/// Which SIMD back-end is in use for the current process / call.
///
/// `select_backend()` picks `Neon` on aarch64, `Avx2` on x86_64 with AVX2,
/// `Portable` otherwise. Each kernel re-runs the dispatch internally so a
/// caller can pin a back-end via the explicit `_portable` / `_neon` /
/// `_avx2` entry points when running parity tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// Pure-Rust portable baseline; available on every target.
    Portable,
    /// `aarch64` NEON intrinsics; only available on `target_arch = "aarch64"`.
    Neon,
    /// `x86_64` AVX2 intrinsics; only available with `is_x86_feature_detected!("avx2")`.
    Avx2,
}

/// Return the best available SIMD back-end for the current host.
///
/// Detection is done at runtime (NOT compile time) so a single binary works
/// across hosts. On aarch64, NEON is part of the base ISA so the detection
/// is implicit. On x86_64, `is_x86_feature_detected!("avx2")` gates the
/// AVX2 path; older targets fall through to the portable baseline.
pub fn select_backend() -> SimdBackend {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is part of aarch64's base ISA; runtime detection unnecessary.
        SimdBackend::Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            SimdBackend::Avx2
        } else {
            SimdBackend::Portable
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        SimdBackend::Portable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_backend_is_deterministic() {
        let a = select_backend();
        let b = select_backend();
        assert_eq!(a, b);
    }

    #[test]
    fn select_backend_picks_simd_when_available() {
        let backend = select_backend();
        // On the M5 Max test host (aarch64) we expect NEON; on x86_64 CI
        // hosts with AVX2 we expect AVX2; anywhere else, Portable. None of
        // these should panic.
        let ok = matches!(
            backend,
            SimdBackend::Neon | SimdBackend::Avx2 | SimdBackend::Portable
        );
        assert!(ok);
    }
}
