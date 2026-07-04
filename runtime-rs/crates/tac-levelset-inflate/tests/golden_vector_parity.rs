//! Golden-vector parity gate — Python oracle vs Rust DECODE port.
//!
//! Each test loads a committed golden-vector manifest (produced by
//! `generate_golden_vectors.py` calling the REAL `tac.lossless.range_coder` /
//! `tac.boundary_math.xi_pose_coder` functions on REAL contest gt_poses ξ) plus its
//! sibling input `.bin` fixtures, runs the Rust decode primitive, and asserts the
//! decoded-output SHA-256 matches the Python oracle byte-for-byte
//! (`assert_sha256_parity`).
//!
//! NO FAKE: the manifests pin the oracle's actual decoded output; a Rust decode that
//! diverges by a single bit fails the SHA gate. The fixtures are the SAME input bytes
//! the oracle consumed — the range-coded stream + the uint32 frequency table.

use std::fs;

use tac_levelset_inflate::{
    conformance::{
        assert_sha256_parity, golden_vectors_dir, i64_slice_to_le_bytes, load_golden_vector,
        GoldenVectorManifest,
    },
    range_decode::decode_static_symbols,
    xi_column::decode_xi_column,
};

fn load_manifest(name: &str) -> Option<GoldenVectorManifest> {
    let path = golden_vectors_dir().join(format!("{name}.json"));
    if !path.exists() {
        eprintln!("golden vector {name}.json not committed — skipping (regenerate to enable)");
        return None;
    }
    Some(load_golden_vector(&path).expect("manifest parse"))
}

fn load_bin(name: &str) -> Vec<u8> {
    let path = golden_vectors_dir().join(name);
    fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {}", path.display(), e))
}

/// Parse a little-endian `<u4` fixture into `Vec<u32>` (the frequency table).
fn load_u32_le(name: &str) -> Vec<u32> {
    let bytes = load_bin(name);
    assert_eq!(bytes.len() % 4, 0, "frequency fixture {name} not u32-aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn range_decode_parity() {
    let Some(m) = load_manifest("levelset_xi_range_decode_v1") else { return };
    let stream = load_bin("levelset_xi_range_decode_v1_stream.bin");
    let freqs = load_u32_le("levelset_xi_range_decode_v1_frequencies.bin");
    let count = m.usize_field("count").expect("count");
    // sanity: manifest metadata agrees with the fixtures.
    assert_eq!(freqs.len(), m.usize_field("n_frequencies").expect("n_frequencies"));
    assert_eq!(stream.len(), m.usize_field("stream_bytes").expect("stream_bytes"));

    let symbols = decode_static_symbols(&stream, count, &freqs).expect("decode");
    assert_eq!(symbols.len(), count);
    let produced = i64_slice_to_le_bytes(&symbols);
    assert_sha256_parity(&produced, &m).expect("range-decode SHA parity vs Python oracle");
}

#[test]
fn xi_column_delta_parity() {
    let Some(m) = load_manifest("levelset_xi_column_delta_v1") else { return };
    let stream = load_bin("levelset_xi_column_delta_v1_stream.bin");
    let freqs = load_u32_le("levelset_xi_column_delta_v1_frequencies.bin");
    let p_count = m.usize_field("p_count").expect("p_count");
    let seed = m.i64_field("seed").expect("seed");
    let lo = m.i64_field("lo").expect("lo");

    let col = decode_xi_column(seed, lo, &freqs, &stream, p_count).expect("decode");
    assert_eq!(col.len(), p_count);
    assert_eq!(col[0], seed);
    let produced = i64_slice_to_le_bytes(&col);
    assert_sha256_parity(&produced, &m).expect("xi-column SHA parity vs Python oracle");
}

/// Guard: every committed golden-vector JSON must have a paired parity test above,
/// so a new vector can never land un-verified (mirror of the sibling crates' gate).
#[test]
fn every_golden_vector_has_paired_parity_test() {
    let covered = ["levelset_xi_range_decode_v1", "levelset_xi_column_delta_v1"];
    let dir = golden_vectors_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        eprintln!("golden_vectors dir absent — skipping coverage guard");
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(stem) = name.strip_suffix(".json") {
            assert!(
                covered.contains(&stem),
                "golden vector {stem}.json has no paired parity test in this file"
            );
        }
    }
}
