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
    pr81_quantizr::{encode_router_actions, pack_nibbles, FP4Codebook},
    pr84_adaptive_mask::{encode_adaptive_context_stream, AdaptiveContextSpec},
    pr91_hpac_grammar::{emit_qmqh_header, encode_categorical_stream, pack_hi_lo_split},
    pr92_joint_stream::{pack_rmc1_composite, pack_rsa1_side},
    pr93_pose_codec::{
        encode_delta_varint_pose, pack_qzmb1_block, serialize_lowpass_luma_residual,
        LowpassLumaResidual,
    },
    pr97_h3_grammar::{encode_length_prefixed_sections, encode_tile_band_streams},
    sparse_packet_ir::{
        encode_arithmetic_coefficients as sparse_encode_arithmetic_coefficients,
        encode_rle_of_zeros as sparse_encode_rle_of_zeros,
        encode_temporal_subsampled as sparse_encode_temporal_subsampled,
        serialize_arithmetic_coefficients as sparse_serialize_arithmetic_coefficients,
        serialize_rle_of_zeros as sparse_serialize_rle_of_zeros,
        serialize_temporal_subsampled as sparse_serialize_temporal_subsampled,
    },
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

// `assert_scaffold_refuses` was retired for the original 19 primitive
// promotion after those fixtures flipped from scaffold-only to
// `assert_sha256_parity` (2026-05-11). Later golden-vector families may still
// use explicit load-only/non-claim stubs until native implementations land;
// those stubs stay local to this test file rather than exported as crate API.

// ── PR101 parity tests ───────────────────────────────────────────────────────

#[test]
fn pr101_ranked_no_op_sidecar_parity() {
    let manifest = match try_load("ranked_no_op_sidecar_v1") {
        Some(m) => m,
        None => return,
    };
    let dims_bin = match try_load_bin("ranked_no_op_sidecar_v1_dims.bin") {
        Some(b) => b,
        None => return,
    };
    let delta_idx_bin = match try_load_bin("ranked_no_op_sidecar_v1_delta_indices.bin") {
        Some(b) => b,
        None => return,
    };
    assert_eq!(dims_bin.len() % 8, 0);
    assert_eq!(delta_idx_bin.len() % 8, 0);
    let mut dims: Vec<i64> = Vec::with_capacity(dims_bin.len() / 8);
    for chunk in dims_bin.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        dims.push(i64::from_le_bytes(buf));
    }
    let mut delta_indices: Vec<i64> = Vec::with_capacity(delta_idx_bin.len() / 8);
    for chunk in delta_idx_bin.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        delta_indices.push(i64::from_le_bytes(buf));
    }
    let schema = RankedSidecarSchema {
        n_pairs: 24,
        n_dims: 8,
        deltas: vec![-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
        huff_min_len: 2,
        huff_max_len: 8,
        no_op_sentinel: 255,
    };
    let payload = encode_ranked_no_op_sidecar(&dims, &delta_indices, &schema)
        .expect("encode_ranked_no_op_sidecar must succeed");
    assert_sha256_parity(&payload, &manifest)
        .expect("PR101 ranked-no-op sidecar payload must match Python oracle SHA-256");
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
        assert!(
            pos + 4 <= raw.len(),
            "fixture truncated reading length prefix"
        );
        let len = u32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]) as usize;
        pos += 4;
        assert!(
            pos + len <= raw.len(),
            "fixture truncated reading sub-stream body"
        );
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
    let manifest = match try_load("merged_range_stream_v1") {
        Some(m) => m,
        None => return,
    };
    // Sibling binary fixtures: flat int32 symbols (60 + 80 + 36 = 176) and
    // three per-tensor fp64 histograms (256 floats each).
    let flat_bytes = match try_load_bin("merged_range_stream_v1_flat.bin") {
        Some(b) => b,
        None => return,
    };
    let mut hists: Vec<Vec<f64>> = Vec::with_capacity(3);
    for i in 0..3 {
        let name = format!("merged_range_stream_v1_hist{i}.bin");
        let bin = match try_load_bin(&name) {
            Some(b) => b,
            None => return,
        };
        assert_eq!(bin.len() % 8, 0);
        let mut row: Vec<f64> = Vec::with_capacity(bin.len() / 8);
        for chunk in bin.chunks_exact(8) {
            row.push(f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
        hists.push(row);
    }
    assert_eq!(flat_bytes.len() % 4, 0);
    let mut flat: Vec<i32> = Vec::with_capacity(flat_bytes.len() / 4);
    for chunk in flat_bytes.chunks_exact(4) {
        flat.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let specs = vec![
        WeightTensorACSpec {
            name: "golden_t0".to_string(),
            shape: vec![60],
            histogram: hists[0].clone(),
            alphabet_size: 256,
        },
        WeightTensorACSpec {
            name: "golden_t1".to_string(),
            shape: vec![10, 8],
            histogram: hists[1].clone(),
            alphabet_size: 256,
        },
        WeightTensorACSpec {
            name: "golden_t2".to_string(),
            shape: vec![3, 3, 4],
            histogram: hists[2].clone(),
            alphabet_size: 256,
        },
    ];
    let stream =
        encode_merged_range_stream(&flat, &specs).expect("encode_merged_range_stream must succeed");
    assert_sha256_parity(&stream.payload, &manifest)
        .expect("merged-range-stream payload must match Python oracle SHA-256");
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

fn try_load_non_claim_stub(name: &'static str) {
    let manifest = match try_load(name) {
        Some(m) => m,
        None => return,
    };
    assert_eq!(
        manifest.extras.get("source_fixture_type").and_then(|v| v.as_str()),
        Some("synthetic_unit_vector"),
        "{name} is a load-only Rust stub; manifest must label the vector as a synthetic unit fixture"
    );
    assert_eq!(
        manifest.extras.get("score_claim").and_then(|v| v.as_bool()),
        Some(false),
        "{name} load-only stub must not imply a score claim"
    );
    assert_eq!(
        manifest
            .extras
            .get("runtime_consumption_proof")
            .and_then(|v| v.as_bool()),
        Some(false),
        "{name} load-only stub must not imply runtime-consumption proof"
    );
    assert_eq!(
        manifest
            .extras
            .get("promotion_eligible")
            .and_then(|v| v.as_bool()),
        Some(false),
        "{name} load-only stub must be promotion-ineligible until Rust byte parity lands"
    );
}

fn try_load_deterministic_compiler_stub(name: &'static str) {
    let path = vector_path(name);
    if !path.exists() {
        eprintln!(
            "deterministic compiler vector {} not present at {}; skipping",
            name,
            path.display()
        );
        return;
    }
    let text = std::fs::read_to_string(&path).expect("deterministic compiler vector must read");
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("deterministic compiler vector must parse as JSON");
    assert_eq!(
        value.get("schema_version").and_then(|v| v.as_str()),
        Some("deterministic_golden_vectors.v1"),
        "{name} must declare the deterministic compiler vector schema"
    );
    assert_eq!(
        value.get("score_claim").and_then(|v| v.as_bool()),
        Some(false),
        "{name} must not imply a score claim"
    );
    assert_eq!(
        value.get("promotion_eligible").and_then(|v| v.as_bool()),
        Some(false),
        "{name} must be promotion-ineligible until exact eval consumes emitted bytes"
    );
    assert_eq!(
        value
            .get("ready_for_exact_eval_dispatch")
            .and_then(|v| v.as_bool()),
        Some(false),
        "{name} must not be exact-eval-dispatch ready as a conformance vector"
    );
}

#[test]
fn pr81_fp4_codebook_parity() {
    let manifest = match try_load("pr81_fp4_codebook_v1") {
        Some(m) => m,
        None => return,
    };
    let values_bin = match try_load_bin("pr81_fp4_codebook_v1_values.bin") {
        Some(b) => b,
        None => return,
    };
    let scales_bin = match try_load_bin("pr81_fp4_codebook_v1_scales.bin") {
        Some(b) => b,
        None => return,
    };
    // Manifest pins block_size=32, n_blocks=2, n_values=64.
    let block_size = 32usize;
    let n_blocks = 2usize;
    let n_values = 64usize;
    assert_eq!(values_bin.len(), n_values * 4);
    let mut values: Vec<f32> = Vec::with_capacity(n_values);
    for chunk in values_bin.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(scales_bin.len(), n_blocks * 4);
    let mut scales: Vec<f32> = Vec::with_capacity(n_blocks);
    for chunk in scales_bin.chunks_exact(4) {
        scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let cb = FP4Codebook::default();
    let nibbles = cb
        .quantize(&values, &scales, block_size)
        .expect("FP4Codebook.quantize must succeed");
    let packed = pack_nibbles(&nibbles).expect("pack_nibbles must succeed");
    assert_sha256_parity(&packed, &manifest)
        .expect("PR81 FP4 codebook payload must match Python oracle SHA-256");
}

#[test]
fn pr81_router_action_parity() {
    let manifest = match try_load("pr81_router_action_v1") {
        Some(m) => m,
        None => return,
    };
    let actions_bin = match try_load_bin("pr81_router_action_v1_actions.bin") {
        Some(b) => b,
        None => return,
    };
    // Manifest pins bits=3, count=600, packed_len=225.
    assert_eq!(actions_bin.len(), 600);
    let payload =
        encode_router_actions(&actions_bin, 3).expect("encode_router_actions must succeed");
    assert_sha256_parity(&payload, &manifest)
        .expect("PR81 ROUTER_ACTION payload must match Python oracle SHA-256");
}

// ── PR84 parity (2026-05-11) ────────────────────────────────────────────────

#[test]
fn pr84_adaptive_mask_context_parity() {
    let manifest = match try_load("pr84_adaptive_mask_context_v1") {
        Some(m) => m,
        None => return,
    };
    let cdf_bin = match try_load_bin("pr84_adaptive_mask_context_v1_cdf.bin") {
        Some(b) => b,
        None => return,
    };
    let symbols_bin = match try_load_bin("pr84_adaptive_mask_context_v1_symbols.bin") {
        Some(b) => b,
        None => return,
    };
    let contexts_bin = match try_load_bin("pr84_adaptive_mask_context_v1_contexts.bin") {
        Some(b) => b,
        None => return,
    };
    // Manifest pins n_contexts=4, alphabet=5, n_symbols=256.
    let alphabet = 5u32;
    let n_contexts = 4u32;
    let cdf_len = (alphabet as usize) * (n_contexts as usize);
    assert_eq!(cdf_bin.len(), cdf_len * 8);
    let mut cdf: Vec<f64> = Vec::with_capacity(cdf_len);
    for chunk in cdf_bin.chunks_exact(8) {
        cdf.push(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    assert_eq!(symbols_bin.len() % 4, 0);
    let mut symbols: Vec<i32> = Vec::with_capacity(symbols_bin.len() / 4);
    for chunk in symbols_bin.chunks_exact(4) {
        symbols.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(contexts_bin.len() % 4, 0);
    let mut context_ids: Vec<i32> = Vec::with_capacity(contexts_bin.len() / 4);
    for chunk in contexts_bin.chunks_exact(4) {
        context_ids.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let spec = AdaptiveContextSpec::new(alphabet, n_contexts, cdf)
        .expect("AdaptiveContextSpec must construct");
    let payload = encode_adaptive_context_stream(&symbols, &context_ids, &spec)
        .expect("encode_adaptive_context_stream must succeed");
    assert_sha256_parity(&payload, &manifest)
        .expect("PR84 adaptive-context payload must match Python oracle SHA-256");
}

// ── PR91 parity (2026-05-11) ────────────────────────────────────────────────

#[test]
fn pr91_arithmetic_coder_constriction_parity() {
    let manifest = match try_load("pr91_arithmetic_coder_constriction_v1") {
        Some(m) => m,
        None => return,
    };
    let symbols_bin = match try_load_bin("pr91_arithmetic_coder_constriction_v1_symbols.bin") {
        Some(b) => b,
        None => return,
    };
    let probs_bin = match try_load_bin("pr91_arithmetic_coder_constriction_v1_probs.bin") {
        Some(b) => b,
        None => return,
    };
    // Manifest pins n_symbols=200, alphabet=8.
    let n_symbols = 200usize;
    let alphabet = 8usize;
    assert_eq!(symbols_bin.len(), n_symbols * 4);
    let mut symbols: Vec<i32> = Vec::with_capacity(n_symbols);
    for chunk in symbols_bin.chunks_exact(4) {
        symbols.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(probs_bin.len(), n_symbols * alphabet * 8);
    let mut probs: Vec<f64> = Vec::with_capacity(n_symbols * alphabet);
    for chunk in probs_bin.chunks_exact(8) {
        probs.push(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    let payload = encode_categorical_stream(&symbols, &probs, n_symbols, alphabet)
        .expect("encode_categorical_stream must succeed");
    assert_sha256_parity(&payload, &manifest)
        .expect("PR91 constriction AC payload must match Python oracle SHA-256");
}

#[test]
fn pr91_qmqh_grammar_parity() {
    let manifest = match try_load("pr91_qmqh_grammar_v1") {
        Some(m) => m,
        None => return,
    };
    // Python recipe: body = bytes(range(64)); split = pack_hi_lo_split(body);
    // full = emit_qmqh_header(hilo_split=True) + split.
    let body: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let split = pack_hi_lo_split(&body).expect("pack_hi_lo_split must succeed");
    let header = emit_qmqh_header(true);
    let mut full = Vec::with_capacity(header.len() + split.len());
    full.extend_from_slice(&header);
    full.extend_from_slice(&split);
    assert_sha256_parity(&full, &manifest)
        .expect("PR91 QH0 grammar payload must match Python oracle SHA-256");
}

// ── PR92 parity (2026-05-11) ────────────────────────────────────────────────

#[test]
fn pr92_rmc_joint_stream_parity() {
    let manifest = match try_load("pr92_rmc_joint_stream_v1") {
        Some(m) => m,
        None => return,
    };
    let seg_bin = match try_load_bin("pr92_rmc_joint_stream_v1_seg.bin") {
        Some(b) => b,
        None => return,
    };
    let actions_bin = match try_load_bin("pr92_rmc_joint_stream_v1_actions.bin") {
        Some(b) => b,
        None => return,
    };
    // Manifest pins seg_bytes_len=128, action_count=120, action_bits=3, table_id=2.
    assert_eq!(seg_bin.len(), 128);
    assert_eq!(actions_bin.len(), 120);
    // Step 1: pack actions via PR81 router_actions (bits=3).
    let body = encode_router_actions(&actions_bin, 3).expect("encode_router_actions must succeed");
    // Step 2: wrap in RSA1 (count=120, action_bits=3, table_id=2).
    let rsa = pack_rsa1_side(120, 3, 2, &body).expect("pack_rsa1_side must succeed");
    // Step 3: wrap in RMC1 (seg_bytes + rsa.payload).
    let composite =
        pack_rmc1_composite(&seg_bin, &rsa.payload).expect("pack_rmc1_composite must succeed");
    assert_sha256_parity(&composite.payload, &manifest)
        .expect("PR92 RMC1 joint stream payload must match Python oracle SHA-256");
}

// ── PR93 parity (2026-05-11) ────────────────────────────────────────────────

#[test]
fn pr93_delta_varint_pose_parity() {
    let manifest = match try_load("pr93_delta_varint_pose_v1") {
        Some(m) => m,
        None => return,
    };
    let poses_bin = match try_load_bin("pr93_delta_varint_pose_v1_poses.bin") {
        Some(b) => b,
        None => return,
    };
    let lo_bin = match try_load_bin("pr93_delta_varint_pose_v1_lo.bin") {
        Some(b) => b,
        None => return,
    };
    let scale_bin = match try_load_bin("pr93_delta_varint_pose_v1_scale.bin") {
        Some(b) => b,
        None => return,
    };
    // Manifest pins n_rows=16, n_dims=4, bits=8.
    let n_rows = 16usize;
    let n_dims = 4usize;
    let bits = 8u32;
    assert_eq!(poses_bin.len(), n_rows * n_dims * 4);
    let mut poses: Vec<f32> = Vec::with_capacity(n_rows * n_dims);
    for chunk in poses_bin.chunks_exact(4) {
        poses.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(lo_bin.len(), n_dims * 4);
    let mut lo: Vec<f32> = Vec::with_capacity(n_dims);
    for chunk in lo_bin.chunks_exact(4) {
        lo.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(scale_bin.len(), n_dims * 4);
    let mut scale: Vec<f32> = Vec::with_capacity(n_dims);
    for chunk in scale_bin.chunks_exact(4) {
        scale.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let stream = encode_delta_varint_pose(&poses, n_rows, n_dims, &lo, &scale, bits)
        .expect("encode_delta_varint_pose must succeed");
    assert_sha256_parity(&stream.payload, &manifest)
        .expect("PR93 delta-varint pose payload must match Python oracle SHA-256");
}

#[test]
fn pr93_qzmb1_parity() {
    let manifest = match try_load("pr93_qzmb1_v1") {
        Some(m) => m,
        None => return,
    };
    // Python recipe:
    //   arch_json = b'{"hidden": 64, "blocks": 3, "input_dim": 5}';
    //   body = bytes(range(64));
    //   block = pack_qzmb1_block(block_size=32, arch_config_json=arch_json, body=body)
    let arch_json: &[u8] = b"{\"hidden\": 64, \"blocks\": 3, \"input_dim\": 5}";
    let body: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let block = pack_qzmb1_block(32, arch_json, &body).expect("pack_qzmb1_block must succeed");
    assert_sha256_parity(&block.payload, &manifest)
        .expect("PR93 QZMB1 payload must match Python oracle SHA-256");
}

#[test]
fn pr93_lowpass_luma_parity() {
    let manifest = match try_load("pr93_lowpass_luma_v1") {
        Some(m) => m,
        None => return,
    };
    // Recipe from src/tac/tests/test_packet_compiler_pr93_lowpass_luma.py:
    //   coeffs = (1.0, 0.5, -0.25, 0.125, -0.0625, 0.03125)
    //   res = LowpassLumaResidual(coefficients=coeffs, height=384, width=512)
    //   blob = serialize_lowpass_luma_residual(res)
    let r = LowpassLumaResidual {
        coefficients: vec![1.0_f32, 0.5, -0.25, 0.125, -0.0625, 0.03125],
        height: 384,
        width: 512,
    };
    let blob =
        serialize_lowpass_luma_residual(&r).expect("serialize_lowpass_luma_residual must succeed");
    assert_sha256_parity(&blob, &manifest)
        .expect("PR93 lowpass-luma payload must match Python oracle SHA-256");
}

// ── PR97 H3 wire-format grammar parity (2026-05-11) ─────────────────────────

#[test]
fn pr97_h3_length_prefixed_sections_parity() {
    let manifest = match try_load("pr97_h3_length_prefixed_sections_v1") {
        Some(m) => m,
        None => return,
    };
    // Recipe from src/tac/tests/test_packet_compiler_pr97_h3_grammar.py:
    //   sections = [b"\x00"*16, b"\x01"*32, b"\x02"*8, b"\x03"*4]
    //   blob = encode_length_prefixed_sections(sections)
    let s0 = vec![0u8; 16];
    let s1 = vec![1u8; 32];
    let s2 = vec![2u8; 8];
    let s3 = vec![3u8; 4];
    let sections: Vec<&[u8]> = vec![&s0, &s1, &s2, &s3];
    let blob = encode_length_prefixed_sections(&sections)
        .expect("encode_length_prefixed_sections must succeed");
    assert_sha256_parity(&blob, &manifest)
        .expect("PR97 H3 length-prefixed sections payload must match Python oracle SHA-256");
}

#[test]
fn pr97_h3_tile_band_streams_parity() {
    let manifest = match try_load("pr97_h3_tile_band_streams_v1") {
        Some(m) => m,
        None => return,
    };
    // Recipe from src/tac/tests/test_packet_compiler_pr97_h3_grammar.py:
    //   streams = [bytes([i]) * (i + 1) for i in range(22)]
    //   blob = encode_tile_band_streams(streams)
    let bufs: Vec<Vec<u8>> = (0..22u8).map(|i| vec![i; (i + 1) as usize]).collect();
    let streams: Vec<&[u8]> = bufs.iter().map(|v| v.as_slice()).collect();
    let blob = encode_tile_band_streams(&streams).expect("encode_tile_band_streams must succeed");
    assert_sha256_parity(&blob, &manifest)
        .expect("PR97 H3 tile-band streams payload must match Python oracle SHA-256");
}

// ── Sparse PacketIR codec parity tests (2026-05-11) ─────────────────────────
//
// Closes O's L2 wire-format ceiling. Each test loads its sibling binary
// fixture (regenerated via tools/regenerate_packet_compiler_rust_parity_fixtures.py)
// and asserts byte-for-byte SHA parity against the committed manifest.

#[test]
fn sparse_rle_of_zeros_parity() {
    let manifest = match try_load("sparse_rle_of_zeros_v1") {
        Some(m) => m,
        None => return,
    };
    let dense_bin = match try_load_bin("sparse_rle_of_zeros_v1_dense.bin") {
        Some(b) => b,
        None => return,
    };
    // 1024 int8 dense bytes (mostly zero, 64 nonzero in [1, 32)).
    assert_eq!(dense_bin.len(), 1024);
    let dense: Vec<i8> = dense_bin.iter().map(|&b| b as i8).collect();
    let stream =
        sparse_encode_rle_of_zeros(&dense).expect("sparse_encode_rle_of_zeros must succeed");
    let blob =
        sparse_serialize_rle_of_zeros(&stream).expect("sparse_serialize_rle_of_zeros must succeed");
    assert_sha256_parity(&blob, &manifest)
        .expect("sparse RLE-of-zeros payload must match Python oracle SHA-256");
}

#[test]
fn sparse_arithmetic_coefficients_parity() {
    let manifest = match try_load("sparse_arithmetic_coefficients_v1") {
        Some(m) => m,
        None => return,
    };
    let values_bin = match try_load_bin("sparse_arithmetic_coefficients_v1_values.bin") {
        Some(b) => b,
        None => return,
    };
    // 500 int32 little-endian values.
    assert_eq!(values_bin.len(), 500 * 4);
    let mut values: Vec<i32> = Vec::with_capacity(500);
    for chunk in values_bin.chunks_exact(4) {
        values.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let stream = sparse_encode_arithmetic_coefficients(&values, None, None, None)
        .expect("sparse_encode_arithmetic_coefficients must succeed");
    let blob = sparse_serialize_arithmetic_coefficients(&stream)
        .expect("sparse_serialize_arithmetic_coefficients must succeed");
    assert_sha256_parity(&blob, &manifest)
        .expect("sparse arithmetic coefficients payload must match Python oracle SHA-256");
}

#[test]
fn sparse_temporal_subsampled_parity() {
    let manifest = match try_load("sparse_temporal_subsampled_v1") {
        Some(m) => m,
        None => return,
    };
    let signal_bin = match try_load_bin("sparse_temporal_subsampled_v1_signal_frames.bin") {
        Some(b) => b,
        None => return,
    };
    // K=10 signal frames * 20 bytes each = 200 bytes total.
    let n_frames = 50usize;
    let per_frame = 20usize;
    assert_eq!(signal_bin.len(), 10 * per_frame);
    // Reconstruct frame list: signal-carrying at i % 5 == 0, else None.
    let mut frame_storage: Vec<Vec<u8>> = Vec::with_capacity(10);
    for i in 0..10 {
        let off = i * per_frame;
        frame_storage.push(signal_bin[off..off + per_frame].to_vec());
    }
    let mut frames: Vec<Option<&[u8]>> = Vec::with_capacity(n_frames);
    let mut k_index = 0usize;
    for i in 0..n_frames {
        if i % 5 == 0 {
            frames.push(Some(&frame_storage[k_index]));
            k_index += 1;
        } else {
            frames.push(None);
        }
    }
    assert_eq!(k_index, 10);
    let stream = sparse_encode_temporal_subsampled(&frames)
        .expect("sparse_encode_temporal_subsampled must succeed");
    let blob = sparse_serialize_temporal_subsampled(&stream)
        .expect("sparse_serialize_temporal_subsampled must succeed");
    assert_sha256_parity(&blob, &manifest)
        .expect("sparse temporal-subsampled payload must match Python oracle SHA-256");
}

// ── PR63 / PR64 / PR65 / PR105 parity stubs (2026-05-12) ────────────────────
//
// The Rust source modules for these primitives are not yet scaffolded under
// `src/` (Python oracle landed first in `src/tac/packet_compiler/`). These
// stub tests register each new golden vector against the `try_load` index so
// the coverage gate at the bottom of this file recognises them as paired.
// The recipe for each vector is documented in the Python tests at
// `src/tac/tests/test_packet_compiler_pr{63,64,65,105}_*.py`.
//
// When the corresponding Rust module lands, flip the stub from `try_load_only`
// to `assert_sha256_parity(produced, &manifest)` once the impl lands.

#[test]
fn pr64_unified_brotli_pose_velocity_parity() {
    try_load_only("pr64_unified_brotli_pose_velocity_v1");
}

#[test]
fn pr63_qpose14_uint16_int16_parity() {
    try_load_only("pr63_qpose14_uint16_int16_v1");
}

#[test]
fn pr63_qpose14_single_zip_member_parity() {
    try_load_only("pr63_qpose14_single_zip_member_v1");
}

#[test]
fn pr65_pq12_pose_parity() {
    try_load_only("pr65_pq12_pose_v1");
}

#[test]
fn pr105_packed_state_schema_parity() {
    try_load_only("pr105_packed_state_schema_v1");
}

// ── PR101 GOLD primitive parity stubs (2026-05-12) ──────────────────────────
//
// Three NEW PR101 GOLD-medal primitives ported to `tac.packet_compiler` from
// the PR101 source (`submissions/hnerv_ft_microcodec/src/codec.py`):
//
// * `pr101_decoder_storage_order` — 28-tensor permutation + 7 split-brotli
//   stream boundaries.
// * `pr101_conv4_storage_perms` — per-tensor 4D-axis permutation table for
//   the 13 conv weight tensors.
// * `pr101_decoder_byte_maps` — per-tensor sign-encoding strategy selector
//   (negzig / zig / twos / off).
//
// Each is `try_load_only` until a Rust source module under
// `src/tac-packet-compiler/src/pr101_gold_*/` lands. The Python tests at
// `src/tac/tests/test_packet_compiler_pr101_gold_primitives.py` produce the
// pinned SHA-256 against deterministic fixtures so Rust ports can verify
// byte parity once they land.

#[test]
fn pr101_decoder_storage_order_parity() {
    try_load_non_claim_stub("pr101_decoder_storage_order_v1");
}

#[test]
fn pr101_conv4_storage_perms_parity() {
    try_load_non_claim_stub("pr101_conv4_storage_perms_v1");
}

#[test]
fn pr101_decoder_byte_maps_parity() {
    try_load_non_claim_stub("pr101_decoder_byte_maps_v1");
}

// ── Sign-encoding 5-strategy unified taxonomy parity stubs (2026-05-12) ────
//
// Five sign-encoding strategies unified from PR96 / PR101 / PR103 into one
// typed API in `src/tac/packet_compiler/sign_encoding.py`:
//
// * `negzig` — negated-zigzag (PR101 hnerv_ft_microcodec/src/codec.py:233-234)
// * `zig`    — zigzag (PR101 hnerv_ft_microcodec/src/codec.py:225-227)
// * `twos`   — two's-complement raw reinterpret (PR101 codec.py:237-238)
// * `off`    — signed-byte-offset x+128 (PR96 inflate.py:90; PR103 inflate.py:147)
// * `raw_uint8` — already-unsigned passthrough (PR103 lo/hi byte streams)
//
// Each is `try_load_only` until a Rust source module under
// `src/tac-packet-compiler/src/sign_encoding/` lands. The Python tests at
// `src/tac/tests/test_packet_compiler_sign_encoding.py` produce the pinned
// SHA-256 against deterministic fixtures so Rust ports can verify byte
// parity once they land.

#[test]
fn sign_encoding_zig_parity() {
    try_load_only("sign_encoding_zig_v1");
}

#[test]
fn sign_encoding_negzig_parity() {
    try_load_only("sign_encoding_negzig_v1");
}

#[test]
fn sign_encoding_twos_parity() {
    try_load_only("sign_encoding_twos_v1");
}

#[test]
fn sign_encoding_off_parity() {
    try_load_only("sign_encoding_off_v1");
}

#[test]
fn sign_encoding_raw_uint8_parity() {
    try_load_only("sign_encoding_raw_uint8_v1");
}

// ── Schema-elision V1 + V2 parity stubs (2026-05-12) ──────────────────────
//
// V1 — PR98 CD1 compact archive grammar (`b'CD1' + sb + n_t + per_tensor`).
// V2 — PR100 schema-driven decoder grammar (bodies blob + scales blob,
//      schema iteration order shared between encoder + decoder).
// V3 — PR105 size-sort (already landed as `pr105_packed_state_schema_v1`).
//
// V1 and V2 are MUTUALLY EXCLUSIVE (both elide the same per-tensor
// metadata region). V2 + V3 stack (different byte regions).
//
// Each is `try_load_only` until a Rust source module under
// `src/tac-packet-compiler/src/{pr98_cd1_compact, pr100_schema_driven}/`
// lands. The Python tests at
// `src/tac/tests/test_packet_compiler_{pr98_cd1_compact_format, pr100_schema_driven_decoder}.py`
// produce the pinned SHA-256 against deterministic fixtures so Rust
// ports can verify byte parity once they land.

#[test]
fn pr98_cd1_compact_format_parity() {
    try_load_only("pr98_cd1_compact_format_v1");
}

#[test]
fn pr100_schema_driven_decoder_parity() {
    try_load_only("pr100_schema_driven_decoder_v1");
}

// ── CompressAI reference neural-compression codec parity stubs (2026-05-12) ──
//
// Three CompressAI codec families wired as packet_compiler primitives:
//
// * `compressai_factorized_prior_v1` — `compressai.models.FactorizedPrior`
//   (Ballé 2018 baseline; single-substream EntropyBottleneck).
// * `compressai_balle_hyperprior_v1` — Ballé 2018 ScaleHyperprior /
//   MeanScaleHyperprior / JointAutoregressiveHierarchicalPriors
//   (two-substream y+z wire format, variant-tagged).
// * `compressai_cheng2020_v1` — Cheng-2020 anchor / attention
//   (two-substream y+z wire format, variant-tagged).
//
// Each is `try_load_only` until a Rust source module under
// `src/tac-packet-compiler/src/compressai_*/` lands. The Python tests at
// `src/tac/tests/test_packet_compiler_{factorized_prior,
// balle_hyperprior, cheng2020}_roundtrip.py` produce the pinned SHA-256
// against deterministic torch seeds (20260512). A native Rust port of the
// CompressAI inference path (analysis/synthesis nets + range coder) would
// be required to reproduce the SHA byte-for-byte — that work is out of
// scope for the Python oracle landing.

#[test]
fn compressai_factorized_prior_parity() {
    try_load_only("compressai_factorized_prior_v1");
}

#[test]
fn compressai_balle_hyperprior_parity() {
    try_load_only("compressai_balle_hyperprior_v1");
}

#[test]
fn compressai_cheng2020_parity() {
    try_load_only("compressai_cheng2020_v1");
}

// ── Deterministic compiler PacketIR parity stubs (2026-05-12) ────────────────
//
// These vectors pin the cross-product of compiler mode × target mode for the
// Python PacketIR compiler surface. They are manifest/conformance fixtures,
// not score claims; Rust ports must preserve the exact mode/target lowering
// semantics before replacing these stubs with byte-producing parity tests.

#[test]
fn deterministic_compiler_packetir_mode_target_parity() {
    for name in [
        "deterministic_compiler_canonicalize_contest_generalized_v1",
        "deterministic_compiler_canonicalize_contest_one_video_replay_v1",
        "deterministic_compiler_canonicalize_production_edge_adaptive_v1",
        "deterministic_compiler_canonicalize_production_generalized_v1",
        "deterministic_compiler_identity_contest_generalized_v1",
        "deterministic_compiler_identity_contest_one_video_replay_v1",
        "deterministic_compiler_identity_production_edge_adaptive_v1",
        "deterministic_compiler_identity_production_generalized_v1",
        "deterministic_compiler_optimize_contest_generalized_v1",
        "deterministic_compiler_optimize_contest_one_video_replay_v1",
        "deterministic_compiler_optimize_production_edge_adaptive_v1",
        "deterministic_compiler_optimize_production_generalized_v1",
    ] {
        try_load_deterministic_compiler_stub(name);
    }
}

// ── Magic codec auto-selector parity stub (2026-05-11) ──────────────────────
//
// The magic codec is a per-stream auto-selector over the existing
// packet_compiler primitives. The Rust scaffold is `try_load_only` — the
// canonical Python oracle in `src/tac/packet_compiler/magic_codec.py`
// produces the pinned SHA, and Rust ports must reproduce it byte-for-byte
// once the inner primitives (RLE / AC / centered-delta / delta-varint /
// categorical / lowpass-luma) land in the Rust crate.

#[test]
fn magic_codec_v1_parity() {
    try_load_only("magic_codec_v1");
}

// ── Magic codec dense streams parity stub (2026-05-12) ──────────────────────
//
// The dense-streams variant extends the magic_codec auto-selector from per-
// stream selection over a single dense array to a typed BUNDLE of dense
// streams (residual / latent / hyperprior / etc.), where each stream gets
// its OWN codec choice (one of: `brotli` | `lzma` | `magic_codec_classic`).
// Wire format declared in
// `src/tac/packet_compiler/magic_codec_dense_streams.py`. The Rust scaffold
// is `try_load_only` until the brotli / lzma / magic_codec_classic Rust
// implementations land — flip to `assert_sha256_parity(...)` then.

#[test]
fn magic_codec_dense_streams_v1_parity() {
    try_load_only("magic_codec_dense_streams_v1");
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
        // Magic codec auto-selector (2026-05-11)
        "magic_codec_v1",
        // Magic codec dense streams — multi-stream bundle (2026-05-12)
        "magic_codec_dense_streams_v1",
        // PR63 — qpose14 uint16-view int16 codec + packed-payload (2026-05-12)
        "pr63_qpose14_uint16_int16_v1",
        "pr63_qpose14_single_zip_member_v1",
        // PR64 — unified-brotli pose-velocity-only codec (2026-05-12)
        "pr64_unified_brotli_pose_velocity_v1",
        // PR65 — PQ12 12-bit / 3-byte / 2-value packed pose codec (2026-05-12)
        "pr65_pq12_pose_v1",
        // PR105 — kitchen_sink packed-state-schema size-sorted helper (2026-05-12)
        "pr105_packed_state_schema_v1",
        // PR101 GOLD — 3 newly-ported primitives (2026-05-12)
        "pr101_decoder_storage_order_v1",
        "pr101_conv4_storage_perms_v1",
        "pr101_decoder_byte_maps_v1",
        // Sign-encoding 5-strategy unified taxonomy (2026-05-12)
        "sign_encoding_zig_v1",
        "sign_encoding_negzig_v1",
        "sign_encoding_twos_v1",
        "sign_encoding_off_v1",
        "sign_encoding_raw_uint8_v1",
        // Schema-elision V1 (PR98 CD1) + V2 (PR100 schema-driven) (2026-05-12)
        "pr98_cd1_compact_format_v1",
        "pr100_schema_driven_decoder_v1",
        // CompressAI reference neural-compression codecs (2026-05-12)
        "compressai_factorized_prior_v1",
        "compressai_balle_hyperprior_v1",
        "compressai_cheng2020_v1",
        // Deterministic compiler PacketIR mode x target-mode conformance (2026-05-12)
        "deterministic_compiler_canonicalize_contest_generalized_v1",
        "deterministic_compiler_canonicalize_contest_one_video_replay_v1",
        "deterministic_compiler_canonicalize_production_edge_adaptive_v1",
        "deterministic_compiler_canonicalize_production_generalized_v1",
        "deterministic_compiler_identity_contest_generalized_v1",
        "deterministic_compiler_identity_contest_one_video_replay_v1",
        "deterministic_compiler_identity_production_edge_adaptive_v1",
        "deterministic_compiler_identity_production_generalized_v1",
        "deterministic_compiler_optimize_contest_generalized_v1",
        "deterministic_compiler_optimize_contest_one_video_replay_v1",
        "deterministic_compiler_optimize_production_edge_adaptive_v1",
        "deterministic_compiler_optimize_production_generalized_v1",
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
