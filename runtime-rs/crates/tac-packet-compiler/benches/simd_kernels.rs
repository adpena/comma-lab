//! Criterion benchmarks for the SIMD-accelerated hot-path kernels.
//!
//! Run with:
//!
//! ```bash
//! cargo bench -p tac-packet-compiler --bench simd_kernels
//! ```
//!
//! Each benchmark compares the portable Rust baseline against the active
//! SIMD back-end ([`SimdBackend::Neon`] on aarch64; [`SimdBackend::Avx2`] on
//! x86_64 with AVX2 detected; [`SimdBackend::Portable`] otherwise).
//!
//! The portable baseline is autovectoriser-friendly; the SIMD back-ends are
//! expected to **match-or-beat** it on every supported host. The point of
//! the comparison is not "is SIMD faster" — it's "is our hand-rolled SIMD
//! at least as fast as the autovectoriser, and does it preserve byte
//! parity". The parity proptest lives in the SIMD module's unit tests.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tac_packet_compiler::simd::{centered_delta, hi_byte, rle_of_zeros, select_backend};

fn deterministic_u16_stream(n: usize) -> Vec<u16> {
    let mut state: u32 = 0x20260511;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state & 0xFFFF) as u16
        })
        .collect()
}

fn deterministic_sparse_i8(n: usize, n_nonzero: usize) -> Vec<i8> {
    let mut state: u32 = 0x20260511;
    let mut out = vec![0i8; n];
    for _ in 0..n_nonzero {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let idx = ((state >> 16) as usize) % n;
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = ((state & 0x1F) as i8).max(1);
        out[idx] = v;
    }
    out
}

fn deterministic_q_matrix(n_pairs: usize, n_dims: usize) -> Vec<u8> {
    let mut state: u32 = 0x20260511;
    (0..n_pairs * n_dims)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state as u8
        })
        .collect()
}

fn bench_hi_byte(c: &mut Criterion) {
    let backend = select_backend();
    let mut group = c.benchmark_group(format!("hi_byte_{backend:?}"));
    for &n in &[100usize, 1000, 10_000] {
        let latents = deterministic_u16_stream(n);
        group.bench_with_input(BenchmarkId::new("portable", n), &latents, |b, l| {
            b.iter(|| black_box(hi_byte::extract_hi_bytes_portable(l)));
        });
        group.bench_with_input(BenchmarkId::new("dispatched", n), &latents, |b, l| {
            b.iter(|| black_box(hi_byte::extract_hi_bytes_u16_to_usize(l)));
        });
    }
    group.finish();
}

fn bench_rle_of_zeros(c: &mut Criterion) {
    let backend = select_backend();
    let mut group = c.benchmark_group(format!("rle_of_zeros_{backend:?}"));
    for &(n, density_pct) in &[(1024usize, 6usize), (4096, 6), (16_384, 6)] {
        let dense = deterministic_sparse_i8(n, n * density_pct / 100);
        let label = format!("{n}/{density_pct}pct");
        group.bench_with_input(BenchmarkId::new("portable", &label), &dense, |b, d| {
            b.iter(|| black_box(rle_of_zeros::scan_nonzero_portable(d)));
        });
        group.bench_with_input(BenchmarkId::new("dispatched", &label), &dense, |b, d| {
            b.iter(|| black_box(rle_of_zeros::scan_nonzero_indices_i8(d)));
        });
    }
    group.finish();
}

fn bench_centered_delta(c: &mut Criterion) {
    let backend = select_backend();
    let mut group = c.benchmark_group(format!("centered_delta_{backend:?}"));
    // PR101 typical (n_pairs=40, n_dims=6) — SIMD heuristic falls back to
    // portable; benchmark documents the equivalence.
    // SIMD-active case (n_pairs=40, n_dims=32).
    for &(n_pairs, n_dims) in &[(40usize, 6usize), (40, 32), (200, 64)] {
        let q = deterministic_q_matrix(n_pairs, n_dims);
        let label = format!("{n_pairs}x{n_dims}");
        group.bench_with_input(BenchmarkId::new("portable", &label), &q, |b, q| {
            b.iter(|| black_box(centered_delta::centered_delta_portable(q, n_pairs, n_dims)));
        });
        group.bench_with_input(BenchmarkId::new("dispatched", &label), &q, |b, q| {
            b.iter(|| {
                black_box(centered_delta::centered_delta_uint8_column_major(
                    q, n_pairs, n_dims,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hi_byte, bench_rle_of_zeros, bench_centered_delta);
criterion_main!(benches);
