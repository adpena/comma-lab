//! Criterion benchmark scaffold.
//!
//! Benchmarks are SCAFFOLD-ONLY. Once a stub function is implemented, the
//! corresponding benchmark below should be flipped to measure throughput on
//! the golden-vector inputs.
//!
//! Until then, this binary exists so `cargo bench -p tac-packet-compiler`
//! is a valid no-op invocation that does not panic. It registers a single
//! "scaffold sentinel" benchmark that takes `O(1)` time.

use criterion::{criterion_group, criterion_main, Criterion};

fn scaffold_sentinel(c: &mut Criterion) {
    c.bench_function("scaffold_sentinel_noop", |b| {
        b.iter(|| {
            // Intentionally trivial: this confirms the bench target builds and
            // can be invoked; real benchmarks land alongside Rust impls.
            std::hint::black_box(0u64.wrapping_add(1))
        });
    });
}

criterion_group!(benches, scaffold_sentinel);
criterion_main!(benches);
