// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wall-clock micro-bench: Rust contour decode + d_seg + connected-components
//! on the committed golden-vector fixtures. Reports ms/call over N iterations.
//! NOT a score claim — a speed measurement vs the Python oracle. [advisory]
use std::time::Instant;
use tac_boundary_decode::{
    components::connected_components, contour::decode_partition_raw, dseg::flip_count,
};

fn gv(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("golden_vectors")
        .join(name)
}

fn main() {
    let full = std::fs::read(gv("contour_decode_full_v1_payload.bin")).unwrap();
    let (h, w) = (384usize, 512usize);
    let n = 2000;
    let t = Instant::now();
    let mut sink = 0u64;
    for _ in 0..n {
        let d = decode_partition_raw(&full, h * w).unwrap();
        sink ^= d[0] as u64;
    }
    let ms = t.elapsed().as_secs_f64() / n as f64 * 1e3;
    println!("decode_partition_raw  384x512 : {ms:.5} ms/call  (sink={sink})");

    let cand = std::fs::read(gv("dseg_popcount_v1_cand.bin")).unwrap();
    let gt = std::fs::read(gv("dseg_popcount_v1_gt.bin")).unwrap();
    let t = Instant::now();
    let mut s2 = 0u64;
    for _ in 0..n {
        s2 ^= flip_count(&cand, &gt).unwrap();
    }
    let ms = t.elapsed().as_secs_f64() / n as f64 * 1e3;
    println!("flip_count            64x96   : {ms:.5} ms/call  (sink={s2})");

    let argmax = std::fs::read(gv("connected_components_v1_argmax.bin")).unwrap();
    let t = Instant::now();
    let mut s3 = 0usize;
    for _ in 0..n {
        s3 ^= connected_components(&argmax, 32, 48, 5).unwrap().n_regions;
    }
    let ms = t.elapsed().as_secs_f64() / n as f64 * 1e3;
    println!("connected_components  32x48   : {ms:.5} ms/call  (sink={s3})");
}
