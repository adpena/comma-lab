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
        sha256_hex, GoldenVectorManifest,
    },
    lane_coverage::{coverage_stack_to_le_bytes, lane_coverage, parse_lane_band_lbnd1},
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

/// Decode the whole LBND1 blob's per-pair coverage at `(rh, rw)` and return the stacked
/// `<f4` C-order bytes (the digest input the manifest pins).
fn lane_coverage_stack_bytes(blob: &[u8], rh: usize, rw: usize) -> (usize, Vec<u8>) {
    let (pairs, geom) = parse_lane_band_lbnd1(blob).expect("LBND1 parse");
    let covs: Vec<Vec<f32>> = pairs
        .iter()
        .map(|lines| lane_coverage(lines, rh, rw, &geom))
        .collect();
    (pairs.len(), coverage_stack_to_le_bytes(&covs))
}

/// #283: the lane AA-SDF render-band rasterizer, bit-for-bit vs the shipped inflate
/// `_lane_coverage` on REAL lane coeffs (fitted from the frozen SegNet argmax `gt_n96`).
#[test]
fn lane_coverage_parity() {
    let Some(m) = load_manifest("levelset_lane_coverage_v1") else { return };
    let blob = load_bin("levelset_lane_coverage_v1_band.bin");
    let rh = m.usize_field("render_h").expect("render_h");
    let rw = m.usize_field("render_w").expect("render_w");
    let n_pairs = m.usize_field("n_pairs").expect("n_pairs");
    assert_eq!(blob.len(), m.usize_field("band_blob_bytes").expect("band_blob_bytes"));

    let (parsed_pairs, produced) = lane_coverage_stack_bytes(&blob, rh, rw);
    assert_eq!(parsed_pairs, n_pairs, "n_pairs mismatch vs manifest");
    assert_eq!(produced.len(), n_pairs * rh * rw * 4, "stacked <f4 byte length");
    assert_sha256_parity(&produced, &m).expect("lane-coverage SHA parity vs Python oracle");
}

/// #283 NEGATIVE CONTROL (proves the gate is non-vacuous): flip one bit of a lane
/// COEFFICIENT byte in the LBND1 payload -> the coverage raster changes -> the SHA no
/// longer matches the pinned oracle digest. (Mirror of the ξ `range_decode` negative
/// control.) The header stays intact so the blob still parses — only a coeff differs.
#[test]
fn lane_coverage_negative_control_bit_flip_breaks_parity() {
    let Some(m) = load_manifest("levelset_lane_coverage_v1") else { return };
    let blob = load_bin("levelset_lane_coverage_v1_band.bin");
    let rh = m.usize_field("render_h").expect("render_h");
    let rw = m.usize_field("render_w").expect("render_w");

    // Sanity: the pristine blob matches.
    let (_n, clean) = lane_coverage_stack_bytes(&blob, rh, rw);
    assert!(sha256_hex(&clean).eq_ignore_ascii_case(&m.sha256), "pristine must match");

    // Locate the f64 payload start (magic6 + u32 header_len + header_json) and flip a
    // high-order byte of the FIRST coeff f64 (a centerline coeff -> large lateral shift).
    let hlen = u32::from_le_bytes([blob[6], blob[7], blob[8], blob[9]]) as usize;
    let payload_start = 6 + 4 + hlen;
    let mut bad = blob.clone();
    bad[payload_start + 6] ^= 0xFF; // exponent-ish byte of the first payload f64

    let (_n2, mutated) = lane_coverage_stack_bytes(&bad, rh, rw);
    assert!(
        !sha256_hex(&mutated).eq_ignore_ascii_case(&m.sha256),
        "NEGATIVE CONTROL FAILED: a flipped coeff bit still matched the oracle digest \
         (the parity gate would be vacuous)"
    );
}

/// Guard: every committed golden-vector JSON must have a paired parity test above,
/// so a new vector can never land un-verified (mirror of the sibling crates' gate).
#[test]
fn every_golden_vector_has_paired_parity_test() {
    let covered = [
        "levelset_xi_range_decode_v1",
        "levelset_xi_column_delta_v1",
        "levelset_lane_coverage_v1",
    ];
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
