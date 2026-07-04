// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wall-clock micro-bench for the level-set inflate ξ range decode.
//!
//! Decodes the committed REAL golden-vector streams `iters` times and prints the
//! per-call wall-clock, so the Rust decode's contribution can be compared 1:1 to the
//! Python oracle (`python_reference_equivalence_test.py`-style timing). Run release:
//!
//! ```bash
//! cargo run --release -p tac-levelset-inflate --example bench_decode --offline
//! ```
//!
//! MEANS: this is decode wall-clock only. The ξ decode is a ONE-TIME per-inflate cost
//! (~3.6k symbols over 600 pairs), not a per-pixel hot path — the speedup is real but
//! its absolute share of the ~16-min contest inflate is negligible. The value of this
//! increment is the bit-exact-provable Rust beachhead, not a wall-clock win.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use tac_levelset_inflate::range_decode::decode_static_symbols;
use tac_levelset_inflate::xi_column::decode_xi_column;

fn gv_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden_vectors")
}

fn read(name: &str) -> Vec<u8> {
    fs::read(gv_dir().join(name)).expect("fixture")
}

fn read_u32(name: &str) -> Vec<u32> {
    read(name)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn manifest_usize(name: &str, key: &str) -> usize {
    let v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(gv_dir().join(name)).unwrap()).unwrap();
    v[key].as_u64().unwrap() as usize
}

fn manifest_i64(name: &str, key: &str) -> i64 {
    let v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(gv_dir().join(name)).unwrap()).unwrap();
    v[key].as_i64().unwrap()
}

fn main() {
    let iters: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);

    let stream_a = read("levelset_xi_range_decode_v1_stream.bin");
    let freqs_a = read_u32("levelset_xi_range_decode_v1_frequencies.bin");
    let count_a = manifest_usize("levelset_xi_range_decode_v1.json", "count");

    let stream_b = read("levelset_xi_column_delta_v1_stream.bin");
    let freqs_b = read_u32("levelset_xi_column_delta_v1_frequencies.bin");
    let p_b = manifest_usize("levelset_xi_column_delta_v1.json", "p_count");
    let seed_b = manifest_i64("levelset_xi_column_delta_v1.json", "seed");
    let lo_b = manifest_i64("levelset_xi_column_delta_v1.json", "lo");

    // Warm up.
    let _ = decode_static_symbols(&stream_a, count_a, &freqs_a).unwrap();
    let _ = decode_xi_column(seed_b, lo_b, &freqs_b, &stream_b, p_b).unwrap();

    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..iters {
        let s = decode_static_symbols(&stream_a, count_a, &freqs_a).unwrap();
        sink ^= s[s.len() - 1];
    }
    let dt_a = t0.elapsed().as_secs_f64() / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        let c = decode_xi_column(seed_b, lo_b, &freqs_b, &stream_b, p_b).unwrap();
        sink ^= c[c.len() - 1];
    }
    let dt_b = t1.elapsed().as_secs_f64() / iters as f64;

    println!("iters={iters} (sink={sink})");
    println!(
        "range_decode : {:.4} ms/call  ({} symbols, alphabet {})",
        dt_a * 1e3,
        count_a,
        freqs_a.len()
    );
    println!(
        "xi_column    : {:.4} ms/call  (P={}, alphabet {})",
        dt_b * 1e3,
        p_b,
        freqs_b.len()
    );
    println!(
        "6-column full ξ decode estimate: ~{:.4} ms/inflate (one-time)",
        dt_a * 1e3 * 6.0
    );
}
