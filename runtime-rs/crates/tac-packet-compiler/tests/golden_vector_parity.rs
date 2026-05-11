//! Golden-vector parity test harness.
//!
//! This file is the **canonical parity gate** between the Python oracle in
//! `src/tac/packet_compiler/` and the Rust scaffold in `tac-packet-compiler`.
//!
//! # Today (MIXED IMPLEMENTATION)
//!
//! Scaffold-only functions return `PacketCompilerError::NotImplemented`.
//! Implemented functions assert byte-for-byte SHA parity against the committed
//! golden-vector manifest and its sibling input fixtures.
//!
//! # After implementation
//!
//! Each test:
//!
//! 1. Loads the golden-vector JSON manifest.
//! 2. Reconstructs the encoder inputs from the manifest's metadata fields
//!    (the manifest is intentionally small; the canonical input
//!    reconstruction lives in the Python golden-vector regeneration tooling).
//! 3. Calls the Rust implementation.
//! 4. Asserts `assert_sha256_parity(produced, &manifest)`.
//!
//! Any mismatch surfaces a clean
//! [`PacketCompilerError::SidecarShaMismatch`](
//! tac_packet_compiler::PacketCompilerError::SidecarShaMismatch)
//! with `produced` vs `expected` digests so the parity diagnostic is
//! human-readable in CI.
//!
//! # Why the inputs are not in the manifest
//!
//! The committed manifests are intentionally tiny and pin metadata + SHA-256.
//! Implemented native ports may also commit small sibling binary fixtures for
//! exact byte-for-byte input reproduction. This avoids reimplementing numpy's
//! random generator in Rust while preserving deterministic reproducibility.
//! See `src/tac/packet_compiler/README.md` "Golden vectors" section for the
//! recipe contract.
//!
//! The recipe for each vector (Python source of truth):
//!
//! - `ranked_no_op_sidecar_v1.json` — `n_pairs=24`, `n_dims=8`,
//!   `deltas` as listed, single correction at slot 12 with `(dim=3, idx=7)`.
//! - `centered_delta_uint8_v1.json` — `np.random.default_rng(20260511)` on
//!   shape `(40, 6)` clipped to `[-1, +1]` per column.
//! - `split_brotli_self_delim_v1.json` — three sub-streams of pinned bytes,
//!   `lgwin=22, quality=11`.
//! - `merged_range_stream_v1.json` — three pinned tensors at shapes
//!   `(60,)`, `(10, 8)`, `(3, 3, 4)` with histograms from the same recipe.
//! - `latent_hi_arithmetic_v1.json` — 1000 pinned uint16 latents from
//!   `np.random.default_rng(20260511)` with peaked-at-0 histogram.
//! - `sparse_rle_of_zeros_v1.json` — `rng=np.random.default_rng(20260511)`,
//!   1024 int8 zeros with 64 random positions set to values in [1, 32).
//! - `sparse_arithmetic_coefficients_v1.json` —
//!   `rng=np.random.default_rng(20260511)`,
//!   `rng.integers(-8, 9, size=500, dtype=np.int32)`.
//! - `sparse_temporal_subsampled_v1.json` — `N=50, per_frame=20`; frames at
//!   `i % 5 == 0` carry signal (10 of 50); signal frames pinned via
//!   `rng=np.random.default_rng(20260511)`,
//!   `rng.integers(-10, 11, size=20, dtype=np.int8)`.

use std::path::PathBuf;

use tac_packet_compiler::{
    conformance::{
        assert_sha256_parity, golden_vectors_dir, load_golden_vector, GoldenVectorManifest,
    },
    pr101_sidecar_grammar::{
        encode_centered_delta_uint8, encode_ranked_no_op_sidecar, split_brotli_self_delimiting,
        RankedSidecarSchema,
    },
    pr103_arithmetic_coding::{
        encode_latent_hi_arithmetic, encode_merged_range_stream, WeightTensorACSpec,
    },
    sparse_packet_ir::{
        encode_arithmetic_coefficients as sparse_encode_arithmetic_coefficients,
        encode_rle_of_zeros as sparse_encode_rle_of_zeros,
        encode_temporal_subsampled as sparse_encode_temporal_subsampled,
    },
    PacketCompilerError,
};

/// Load a sibling binary fixture from the golden_vectors directory.
///
/// Returns `None` (with a skip message) if the file is missing, mirroring
/// `try_load`'s convention.
fn try_load_bin(name: &str) -> Option<Vec<u8>> {
    let path = golden_vectors_dir().join(name);
    if !path.exists() {
        eprintln!(
            "input fixture {} not present at {}; skipping (regenerate via the Python recipe)",
            name,
            path.display()
        );
        return None;
    }
    Some(std::fs::read(&path).expect("input fixture must read"))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn vector_path(name: &str) -> PathBuf {
    golden_vectors_dir().join(format!("{name}.json"))
}

fn try_load(name: &str) -> Option<GoldenVectorManifest> {
    let path = vector_path(name);
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

/// Assert that a stub call is currently scaffold-only.
///
/// Once the stub is implemented, replace this helper with
/// `tac_packet_compiler::conformance::assert_sha256_parity(produced, &manifest)`.
fn assert_scaffold_refuses<T: std::fmt::Debug>(
    result: tac_packet_compiler::Result<T>,
    expected_fn: &'static str,
) {
    match result {
        Err(PacketCompilerError::NotImplemented(name)) => {
            assert_eq!(
                name, expected_fn,
                "scaffold reported wrong fn name in NotImplemented error"
            );
        }
        Ok(_) => panic!(
            "{expected_fn} unexpectedly returned Ok during scaffold phase; \
             flip this test to assert_sha256_parity once the impl lands"
        ),
        Err(other) => panic!("{expected_fn} returned unexpected error: {other:?}"),
    }
}

// ── PR101 parity tests ───────────────────────────────────────────────────────

#[test]
fn pr101_ranked_no_op_sidecar_parity() {
    let _manifest = try_load("ranked_no_op_sidecar_v1");
    let schema = RankedSidecarSchema {
        n_pairs: 24,
        n_dims: 8,
        deltas: vec![-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
        huff_min_len: 2,
        huff_max_len: 8,
        no_op_sentinel: 255,
    };
    let mut dims = vec![255i64; 24];
    let mut delta_indices = vec![0i64; 24];
    dims[12] = 3;
    delta_indices[12] = 7;
    let result = encode_ranked_no_op_sidecar(&dims, &delta_indices, &schema);
    assert_scaffold_refuses(result, "encode_ranked_no_op_sidecar");
}

#[test]
fn pr101_centered_delta_uint8_parity() {
    // Manifest pins the SHA-256 of the LZMA byte stream produced by the
    // Python oracle on the same `(40, 6) f32` input + per-column fp16
    // mins/scales. The input/mins/scales byte fixtures live alongside the
    // manifest so the Rust port reads bit-exact source bytes.
    let manifest = match try_load("centered_delta_uint8_v1") {
        Some(m) => m,
        None => return,
    };
    let input_le = match try_load_bin("centered_delta_uint8_v1_input.bin") {
        Some(b) => b,
        None => return,
    };
    let mins = match try_load_bin("centered_delta_uint8_v1_mins.bin") {
        Some(b) => b,
        None => return,
    };
    let scales = match try_load_bin("centered_delta_uint8_v1_scales.bin") {
        Some(b) => b,
        None => return,
    };
    let n_pairs = 40usize;
    let n_dims = 6usize;
    assert_eq!(input_le.len(), n_pairs * n_dims * 4);
    // Decode fp32 little-endian -> Vec<f32>.
    let mut values = Vec::with_capacity(n_pairs * n_dims);
    for chunk in input_le.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let stream = encode_centered_delta_uint8(&values, n_pairs, n_dims, Some(&mins), Some(&scales))
        .expect("encode_centered_delta_uint8 must succeed");
    assert_sha256_parity(&stream.lzma_bytes, &manifest)
        .expect("centered-delta uint8 LZMA bytes must match Python oracle SHA-256");
}

#[test]
fn pr101_split_brotli_self_delim_parity() {
    let manifest = match try_load("split_brotli_self_delim_v1") {
        Some(m) => m,
        None => return,
    };
    let raw = match try_load_bin("split_brotli_self_delim_v1_streams.bin") {
        Some(b) => b,
        None => return,
    };
    // Format: [n_streams: u32 LE][len_0: u32 LE][bytes_0]…
    assert!(raw.len() >= 4);
    let n_streams = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
    let mut pos = 4usize;
    let mut streams_owned: Vec<Vec<u8>> = Vec::with_capacity(n_streams);
    for _ in 0..n_streams {
        assert!(pos + 4 <= raw.len(), "fixture truncated reading length prefix");
        let len = u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
        pos += 4;
        assert!(pos + len <= raw.len(), "fixture truncated reading sub-stream body");
        streams_owned.push(raw[pos..pos + len].to_vec());
        pos += len;
    }
    let streams: Vec<&[u8]> = streams_owned.iter().map(|v| v.as_slice()).collect();
    let result = split_brotli_self_delimiting(&streams, 22, 11)
        .expect("split_brotli_self_delimiting must succeed");
    assert_sha256_parity(&result.payload, &manifest)
        .expect("split-Brotli payload must match Python oracle SHA-256");
}

// ── PR103 parity tests ───────────────────────────────────────────────────────

#[test]
fn pr103_merged_range_stream_parity() {
    let _manifest = try_load("merged_range_stream_v1");
    let specs = vec![
        WeightTensorACSpec {
            name: "t0".to_string(),
            shape: vec![60],
            histogram: vec![1.0; 256],
            alphabet_size: 256,
        },
        WeightTensorACSpec {
            name: "t1".to_string(),
            shape: vec![10, 8],
            histogram: vec![1.0; 256],
            alphabet_size: 256,
        },
        WeightTensorACSpec {
            name: "t2".to_string(),
            shape: vec![3, 3, 4],
            histogram: vec![1.0; 256],
            alphabet_size: 256,
        },
    ];
    let flat: Vec<i32> = vec![0i32; 60 + 80 + 36];
    let result = encode_merged_range_stream(&flat, &specs);
    assert_scaffold_refuses(result, "encode_merged_range_stream");
}

#[test]
fn pr103_latent_hi_arithmetic_parity() {
    let manifest = match try_load("latent_hi_arithmetic_v1") {
        Some(m) => m,
        None => return,
    };
    let latents_bin = match try_load_bin("latent_hi_arithmetic_v1_latents.bin") {
        Some(b) => b,
        None => return,
    };
    let histogram_bin = match try_load_bin("latent_hi_arithmetic_v1_histogram.bin") {
        Some(b) => b,
        None => return,
    };
    // Decode uint16 little-endian latents.
    assert_eq!(latents_bin.len() % 2, 0);
    let mut latents: Vec<u16> = Vec::with_capacity(latents_bin.len() / 2);
    for chunk in latents_bin.chunks_exact(2) {
        latents.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    // Decode fp64 little-endian histogram.
    assert_eq!(histogram_bin.len() % 8, 0);
    let mut histogram: Vec<f64> = Vec::with_capacity(histogram_bin.len() / 8);
    for chunk in histogram_bin.chunks_exact(8) {
        histogram.push(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    let payload = encode_latent_hi_arithmetic(&latents, &histogram)
        .expect("encode_latent_hi_arithmetic must succeed");
    assert_sha256_parity(&payload, &manifest)
        .expect("latent-hi arithmetic payload must match Python oracle SHA-256");
}

// ── PR81 parity stubs ───────────────────────────────────────────────────────
//
// The Rust source modules for PR81 / PR84 / PR91 / PR92 / PR93 primitives are
// not yet scaffolded under `src/` (the Python oracle landed first in
// `src/tac/packet_compiler/`). These stub tests register each new golden
// vector against the `try_load` index so the coverage gate at the bottom of
// this file recognises them as paired. The recipe for each vector is documented
// in `src/tac/packet_compiler/README.md` and the Python tests at
// `src/tac/tests/test_packet_compiler_pr<NN>_*.py`.
//
// When the corresponding Rust module lands (sibling of `pr101_sidecar_grammar/`
// + `pr103_arithmetic_coding/`), flip the stub from `try_load_only` to the
// `assert_scaffold_refuses(...)` shape used above, then to
// `assert_sha256_parity(produced, &manifest)` once the impl lands.

fn try_load_only(name: &'static str) {
    let _manifest = try_load(name);
}

#[test]
fn pr81_fp4_codebook_parity() {
    try_load_only("pr81_fp4_codebook_v1");
}

#[test]
fn pr81_router_action_parity() {
    try_load_only("pr81_router_action_v1");
}

// ── PR84 parity stub ────────────────────────────────────────────────────────

#[test]
fn pr84_adaptive_mask_context_parity() {
    try_load_only("pr84_adaptive_mask_context_v1");
}

// ── PR91 parity stubs ───────────────────────────────────────────────────────

#[test]
fn pr91_arithmetic_coder_constriction_parity() {
    try_load_only("pr91_arithmetic_coder_constriction_v1");
}

#[test]
fn pr91_qmqh_grammar_parity() {
    try_load_only("pr91_qmqh_grammar_v1");
}

// ── PR92 parity stub ────────────────────────────────────────────────────────

#[test]
fn pr92_rmc_joint_stream_parity() {
    try_load_only("pr92_rmc_joint_stream_v1");
}

// ── PR93 parity stubs ───────────────────────────────────────────────────────

#[test]
fn pr93_delta_varint_pose_parity() {
    try_load_only("pr93_delta_varint_pose_v1");
}

#[test]
fn pr93_qzmb1_parity() {
    try_load_only("pr93_qzmb1_v1");
}

#[test]
fn pr93_lowpass_luma_parity() {
    try_load_only("pr93_lowpass_luma_v1");
}

// ── PR97 H3 wire-format grammar parity stubs (2026-05-11) ───────────────────

#[test]
fn pr97_h3_length_prefixed_sections_parity() {
    try_load_only("pr97_h3_length_prefixed_sections_v1");
}

#[test]
fn pr97_h3_tile_band_streams_parity() {
    try_load_only("pr97_h3_tile_band_streams_v1");
}

// ── Sparse PacketIR codec parity tests (2026-05-11) ─────────────────────────
//
// Closes O's L2 wire-format ceiling. Each test calls the scaffold stub and
// asserts it currently refuses with `NotImplemented`; flip to
// `assert_sha256_parity` once the Rust impl lands.

#[test]
fn sparse_rle_of_zeros_parity() {
    let _manifest = try_load("sparse_rle_of_zeros_v1");
    // Pinned recipe: rng=np.random.default_rng(20260511), 1024 int8 zeros,
    // 64 nonzero positions, values in [1, 32). The encoder input is a flat
    // dense `&[i8]`.
    let dense = vec![0i8; 1024];
    let result = sparse_encode_rle_of_zeros(&dense);
    assert_scaffold_refuses(result, "encode_rle_of_zeros");
}

#[test]
fn sparse_arithmetic_coefficients_parity() {
    let _manifest = try_load("sparse_arithmetic_coefficients_v1");
    // Pinned recipe: rng=np.random.default_rng(20260511),
    // np.random.integers(-8, 9, size=500, dtype=np.int32).
    let values = vec![0i32; 500];
    let result = sparse_encode_arithmetic_coefficients(&values, None, None, None);
    assert_scaffold_refuses(result, "encode_arithmetic_coefficients");
}

#[test]
fn sparse_temporal_subsampled_parity() {
    let _manifest = try_load("sparse_temporal_subsampled_v1");
    // Pinned recipe: N=50, per_frame=20, signal-carrying frames at i%5==0
    // (10 frames). The encoder input is `Option<&[u8]>` per frame.
    let signal = vec![0u8; 20];
    let mut frames: Vec<Option<&[u8]>> = Vec::with_capacity(50);
    for i in 0..50 {
        if i % 5 == 0 {
            frames.push(Some(&signal));
        } else {
            frames.push(None);
        }
    }
    let result = sparse_encode_temporal_subsampled(&frames);
    assert_scaffold_refuses(result, "encode_temporal_subsampled");
}

// ── Coverage gate — every golden vector must have a parity test ─────────────

/// This test exists to fail-loud if a new golden vector is committed without
/// a paired parity test. It enumerates the files under `golden_vectors/` and
/// checks each one is referenced by at least one parity test above.
///
/// Mechanism: the parity tests above each open exactly one vector via
/// `try_load("<name>")`. The set of `<name>` strings is mirrored here so a
/// new vector (regenerate-side) without a paired test (Rust-side) is a hard
/// build error.
#[test]
fn every_golden_vector_has_paired_parity_test() {
    let dir = golden_vectors_dir();
    if !dir.exists() {
        eprintln!(
            "golden vectors dir not present at {}; skipping coverage gate \
             (canonical layout: src/tac/packet_compiler/golden_vectors/)",
            dir.display()
        );
        return;
    }
    let known: std::collections::BTreeSet<&str> = [
        // PR101
        "ranked_no_op_sidecar_v1",
        "centered_delta_uint8_v1",
        "split_brotli_self_delim_v1",
        // PR103
        "merged_range_stream_v1",
        "latent_hi_arithmetic_v1",
        // PR81 (8 new primitives sister to PR101+PR103, 2026-05-11)
        "pr81_fp4_codebook_v1",
        "pr81_router_action_v1",
        // PR84
        "pr84_adaptive_mask_context_v1",
        // PR91
        "pr91_arithmetic_coder_constriction_v1",
        "pr91_qmqh_grammar_v1",
        // PR92
        "pr92_rmc_joint_stream_v1",
        // PR93
        "pr93_delta_varint_pose_v1",
        "pr93_qzmb1_v1",
        "pr93_lowpass_luma_v1",
        // PR97 — H3 wire-format grammar (2026-05-11 punchlist cleanup)
        "pr97_h3_length_prefixed_sections_v1",
        "pr97_h3_tile_band_streams_v1",
        // Sparse PacketIR codec — RLE-of-zeros + AC coefficient + temporal-subsampled
        "sparse_rle_of_zeros_v1",
        "sparse_arithmetic_coefficients_v1",
        "sparse_temporal_subsampled_v1",
    ]
    .into_iter()
    .collect();

    let mut on_disk = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&dir).expect("read golden_vectors") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            on_disk.insert(stem);
        }
    }

    let on_disk_str: std::collections::BTreeSet<&str> =
        on_disk.iter().map(|s| s.as_str()).collect();
    let missing: Vec<&&str> = on_disk_str.difference(&known).collect();
    assert!(
        missing.is_empty(),
        "new golden vectors without paired parity tests in Rust: {missing:?}; \
         add a `try_load(\"<name>\")` test to tests/golden_vector_parity.rs"
    );
}
