//! Golden-vector loader + byte-for-byte parity helpers.
//!
//! The committed golden vectors under
//! `src/tac/packet_compiler/golden_vectors/` are the **single source of truth**
//! for byte-level parity between the Python oracle and any native port.
//! Each vector is a small JSON manifest pinning:
//!
//! - `schema` — versioned schema name (e.g. `"ranked_no_op_sidecar.v1"`).
//! - `sha256` — hex digest of the canonical encoded payload.
//! - per-schema metadata enough to reconstruct the encoder inputs.
//!
//! This module provides:
//!
//! - [`GoldenVectorManifest`] — the typed shape every JSON manifest matches.
//! - [`load_golden_vector`] — JSON parse + missing-file error mapping.
//! - [`assert_sha256_parity`] — compare a Rust-produced payload against the
//!   committed digest, surface a fail-loud error on mismatch.
//!
//! The parity test harness (`tests/golden_vector_parity.rs`) uses these
//! helpers to compare Rust output against the committed vectors.

use std::fs;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{PacketCompilerError, Result};

/// Common shape every golden-vector JSON manifest carries.
///
/// Per-schema fields (e.g. `n_pairs`, `tensor_shapes`) are not modeled here;
/// the parity harness deserialises into [`serde_json::Value`] for those and
/// extracts whatever it needs per vector.
#[derive(Debug, Clone, Deserialize)]
pub struct GoldenVectorManifest {
    /// Versioned schema label (e.g. `"ranked_no_op_sidecar.v1"`).
    pub schema: String,
    /// SHA-256 hex digest of the canonical encoded payload.
    pub sha256: String,
    /// All other manifest fields. Schema-specific reconstruction logic
    /// reaches into this map.
    #[serde(flatten)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

/// Load a golden-vector manifest from disk.
///
/// Returns [`PacketCompilerError::GoldenVectorIo`] on i/o or JSON parse
/// failure.
pub fn load_golden_vector(path: &Path) -> Result<GoldenVectorManifest> {
    let text = fs::read_to_string(path).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("read {}: {}", path.display(), e))
    })?;
    let manifest: GoldenVectorManifest = serde_json::from_str(&text).map_err(|e| {
        PacketCompilerError::GoldenVectorIo(format!("parse {}: {}", path.display(), e))
    })?;
    Ok(manifest)
}

/// Compute the SHA-256 hex digest of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Assert that `produced` matches the SHA-256 digest pinned in `manifest`.
///
/// Returns a structured [`PacketCompilerError::SidecarShaMismatch`] error on
/// mismatch so the parity harness can surface a clean diagnostic per vector
/// without per-test boilerplate.
pub fn assert_sha256_parity(produced: &[u8], manifest: &GoldenVectorManifest) -> Result<()> {
    let produced_hex = sha256_hex(produced);
    if produced_hex.eq_ignore_ascii_case(&manifest.sha256) {
        Ok(())
    } else {
        Err(PacketCompilerError::SidecarShaMismatch {
            schema: manifest.schema.clone(),
            produced: produced_hex,
            expected: manifest.sha256.clone(),
        })
    }
}

/// Resolve the repo-relative path to the committed golden-vector directory.
///
/// The crate lives at `runtime-rs/crates/tac-packet-compiler/`; the golden
/// vectors live at `src/tac/packet_compiler/golden_vectors/`. We resolve via
/// `CARGO_MANIFEST_DIR` so the path is stable across local checkouts and
/// CI.
pub fn golden_vectors_dir() -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    Path::new(&manifest_dir)
        .join("../../../src/tac/packet_compiler/golden_vectors")
        .canonicalize()
        .unwrap_or_else(|_| {
            Path::new(&manifest_dir).join("../../../src/tac/packet_compiler/golden_vectors")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_matches_known_value() {
        // empty input → standard SHA-256 of empty string
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(sha256_hex(b""), expected);
    }

    #[test]
    fn assert_parity_returns_structured_mismatch() {
        let manifest = GoldenVectorManifest {
            schema: "test.v1".to_string(),
            sha256: "deadbeef".to_string(),
            extras: serde_json::Map::new(),
        };
        let err = assert_sha256_parity(b"different", &manifest).expect_err("should mismatch");
        match err {
            PacketCompilerError::SidecarShaMismatch {
                schema, expected, ..
            } => {
                assert_eq!(schema, "test.v1");
                assert_eq!(expected, "deadbeef");
            }
            other => panic!("expected SidecarShaMismatch, got {other:?}"),
        }
    }

    #[test]
    fn golden_vectors_dir_resolves_against_committed_paths() {
        // We don't require the directory to exist at test time on every
        // shell, but the resolution must not panic and must be deterministic.
        let dir = golden_vectors_dir();
        let display = dir.display().to_string();
        assert!(display.contains("golden_vectors"));
    }
}
