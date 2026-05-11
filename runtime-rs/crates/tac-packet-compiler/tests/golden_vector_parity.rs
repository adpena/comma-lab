//! Golden-vector parity test harness.
//!
//! This file is the **canonical parity gate** between the Python oracle in
//! `src/tac/packet_compiler/` and the Rust scaffold in `tac-packet-compiler`.
//!
//! # Today (SCAFFOLD)
//!
//! Every stub function returns `PacketCompilerError::NotImplemented`. The
//! parity tests are written to **expect that error explicitly** so the
//! scaffold passes `cargo test -p tac-packet-compiler` without lying about
//! parity. The moment any stub is implemented, the corresponding test must
//! be flipped to assert byte-for-byte parity (the test is structured so the
//! `match` arm is the only thing that changes).
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
//! The committed manifests are intentionally tiny (a few hundred bytes each).
//! They pin metadata + SHA-256 only; the inputs are reconstructed
//! deterministically from a documented seed/recipe. This avoids inflating the
//! repo with binary fixtures while preserving byte-level reproducibility.
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

use std::path::PathBuf;

use tac_packet_compiler::{
    conformance::{golden_vectors_dir, load_golden_vector, GoldenVectorManifest},
    pr101_sidecar_grammar::{
        encode_centered_delta_uint8, encode_ranked_no_op_sidecar, split_brotli_self_delimiting,
        RankedSidecarSchema,
    },
    pr103_arithmetic_coding::{
        encode_latent_hi_arithmetic, encode_merged_range_stream, WeightTensorACSpec,
    },
    PacketCompilerError,
};

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
    let _manifest = try_load("centered_delta_uint8_v1");
    let n_pairs = 40usize;
    let n_dims = 6usize;
    let values = vec![0.0f32; n_pairs * n_dims];
    let result = encode_centered_delta_uint8(&values, n_pairs, n_dims, None, None);
    assert_scaffold_refuses(result, "encode_centered_delta_uint8");
}

#[test]
fn pr101_split_brotli_self_delim_parity() {
    let _manifest = try_load("split_brotli_self_delim_v1");
    let s0: &[u8] = b"stream-zero placeholder bytes";
    let s1: &[u8] = b"stream-one placeholder bytes";
    let s2: &[u8] = b"stream-two placeholder bytes";
    let streams = [s0, s1, s2];
    let result = split_brotli_self_delimiting(&streams, 22, 11);
    assert_scaffold_refuses(result, "split_brotli_self_delimiting");
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
    let _manifest = try_load("latent_hi_arithmetic_v1");
    let latents = vec![0u16; 1000];
    let histogram = vec![1.0; 256];
    let result = encode_latent_hi_arithmetic(&latents, &histogram);
    assert_scaffold_refuses(result, "encode_latent_hi_arithmetic");
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
        "ranked_no_op_sidecar_v1",
        "centered_delta_uint8_v1",
        "split_brotli_self_delim_v1",
        "merged_range_stream_v1",
        "latent_hi_arithmetic_v1",
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
