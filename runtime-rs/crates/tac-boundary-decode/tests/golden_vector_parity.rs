//! Golden-vector parity gate — Python oracle vs Rust DECODE port.
//!
//! Each test loads a committed golden-vector manifest (produced by
//! `golden_vectors/generate_golden_vectors.py` calling the REAL
//! `tac.boundary_math` functions) plus its sibling input `.bin` fixtures, runs
//! the Rust decode primitive, and asserts the decoded-raw SHA-256 matches the
//! Python oracle byte-for-byte (`assert_sha256_parity`).
//!
//! NO FAKE: the manifests pin the oracle's actual decoded output; a Rust decode
//! that diverges by a single bit fails the SHA gate. The fixtures are the SAME
//! input bytes the oracle consumed (we never reimplement numpy's RNG in Rust).

use tac_boundary_decode::{
    components::connected_components,
    conformance::{
        assert_sha256_parity, golden_vectors_dir, load_golden_vector, GoldenVectorManifest,
    },
    contour::decode_partition_raw,
    dseg::{d_seg, flip_count},
};

fn try_load(name: &str) -> Option<GoldenVectorManifest> {
    let path = golden_vectors_dir().join(format!("{name}.json"));
    if !path.exists() {
        eprintln!(
            "golden vector {} not present at {}; skipping (regenerate via the Python recipe)",
            name,
            path.display()
        );
        return None;
    }
    Some(load_golden_vector(&path).expect("golden vector must parse"))
}

fn try_load_bin(name: &str) -> Option<Vec<u8>> {
    let path = golden_vectors_dir().join(name);
    if !path.exists() {
        eprintln!(
            "input fixture {} not present at {}; skipping",
            name,
            path.display()
        );
        return None;
    }
    Some(std::fs::read(&path).expect("input fixture must read"))
}

// ── Contour codec DECODE parity (RAW-LZMA2 → raw label bytes) ────────────────

fn contour_decode_case(name: &str, payload_fixture: &str) {
    let manifest = match try_load(name) {
        Some(m) => m,
        None => return,
    };
    let payload = match try_load_bin(payload_fixture) {
        Some(b) => b,
        None => return,
    };
    let h = manifest.usize_field("height").expect("height");
    let w = manifest.usize_field("width").expect("width");
    let decoded = decode_partition_raw(&payload, h * w).expect("RAW-LZMA2 decode must succeed");
    assert_sha256_parity(&decoded, &manifest)
        .expect("contour decoded-raw bytes must match Python oracle SHA-256");
}

#[test]
fn contour_decode_full_parity() {
    contour_decode_case(
        "contour_decode_full_v1",
        "contour_decode_full_v1_payload.bin",
    );
}

#[test]
fn contour_decode_small_parity() {
    contour_decode_case(
        "contour_decode_small_v1",
        "contour_decode_small_v1_payload.bin",
    );
}

// ── d_seg popcount parity (flip_count || d_seg) ──────────────────────────────

#[test]
fn dseg_popcount_parity() {
    let manifest = match try_load("dseg_popcount_v1") {
        Some(m) => m,
        None => return,
    };
    let cand = match try_load_bin("dseg_popcount_v1_cand.bin") {
        Some(b) => b,
        None => return,
    };
    let gt = match try_load_bin("dseg_popcount_v1_gt.bin") {
        Some(b) => b,
        None => return,
    };
    let fc = flip_count(&cand, &gt).expect("flip_count");
    let dseg = d_seg(&cand, &gt).expect("d_seg");
    // Output blob = flip_count(u64 LE) || d_seg(f64 LE), matching the oracle recipe.
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&fc.to_le_bytes());
    out.extend_from_slice(&dseg.to_le_bytes());
    assert_sha256_parity(&out, &manifest)
        .expect("d_seg popcount blob must match Python oracle SHA-256");
}

// ── Connected-components parity (region_of int32 LE raster) ──────────────────

#[test]
fn connected_components_parity() {
    let manifest = match try_load("connected_components_v1") {
        Some(m) => m,
        None => return,
    };
    let argmax = match try_load_bin("connected_components_v1_argmax.bin") {
        Some(b) => b,
        None => return,
    };
    let h = manifest.usize_field("height").expect("height");
    let w = manifest.usize_field("width").expect("width");
    let n_classes = manifest.usize_field("n_classes").expect("n_classes");
    let cc =
        connected_components(&argmax, h, w, n_classes).expect("connected_components must succeed");
    // Cross-check the region count too (structural agreement beyond the raster sha).
    let expected_regions = manifest.usize_field("n_regions").expect("n_regions");
    assert_eq!(
        cc.n_regions, expected_regions,
        "region count must match Python oracle"
    );
    assert_sha256_parity(&cc.region_of_i32_le_bytes(), &manifest)
        .expect("region_of int32 LE raster must match Python oracle SHA-256");
}

// ── Coverage gate — every golden vector must have a paired parity test ───────

#[test]
fn every_golden_vector_has_paired_parity_test() {
    let dir = golden_vectors_dir();
    if !dir.exists() {
        eprintln!(
            "golden vectors dir not present at {}; skipping coverage gate",
            dir.display()
        );
        return;
    }
    let known: std::collections::BTreeSet<&str> = [
        "contour_decode_full_v1",
        "contour_decode_small_v1",
        "dseg_popcount_v1",
        "connected_components_v1",
    ]
    .into_iter()
    .collect();
    let mut on_disk = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&dir).expect("read golden_vectors") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            on_disk.insert(
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string(),
            );
        }
    }
    let on_disk_str: std::collections::BTreeSet<&str> =
        on_disk.iter().map(|s| s.as_str()).collect();
    let missing: Vec<&&str> = on_disk_str.difference(&known).collect();
    assert!(
        missing.is_empty(),
        "new golden vectors without paired parity tests in Rust: {missing:?}"
    );
}
