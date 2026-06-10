// SPDX-License-Identifier: MIT OR Apache-2.0
//! Golden-vector loader + byte-for-byte parity helpers.
//!
//! Mirrors the canonical pattern in the sibling `tac-packet-compiler` crate. The
//! committed golden vectors under this crate's `golden_vectors/` are produced by
//! the Python ORACLE (`generate_golden_vectors.py`, which calls the REAL
//! `tac.boundary_math` functions). Each vector pins the SHA-256 of the *decoded
//! raw output* the Rust port must reproduce bit-for-bit, plus the input byte
//! fixtures the Rust port reads.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{BoundaryDecodeError, Result};

/// Common shape every golden-vector JSON manifest carries.
#[derive(Debug, Clone, Deserialize)]
pub struct GoldenVectorManifest {
    /// Versioned schema label (e.g. `"contour_decode.v1"`).
    pub schema: String,
    /// SHA-256 hex digest of the decoded raw output.
    pub sha256: String,
    /// All other manifest fields (height/width/n_classes/etc.).
    #[serde(flatten)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

impl GoldenVectorManifest {
    /// Read a usize extras field, failing loud if absent or non-integer.
    pub fn usize_field(&self, key: &str) -> Result<usize> {
        self.extras
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| {
                BoundaryDecodeError::GoldenVectorIo(format!(
                    "manifest {} missing usize field {key}",
                    self.schema
                ))
            })
    }
}

/// Load a golden-vector manifest from disk.
pub fn load_golden_vector(path: &Path) -> Result<GoldenVectorManifest> {
    let text = fs::read_to_string(path).map_err(|e| {
        BoundaryDecodeError::GoldenVectorIo(format!("read {}: {}", path.display(), e))
    })?;
    serde_json::from_str(&text).map_err(|e| {
        BoundaryDecodeError::GoldenVectorIo(format!("parse {}: {}", path.display(), e))
    })
}

/// Compute the SHA-256 hex digest of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Assert that `produced` matches the SHA-256 digest pinned in `manifest`.
pub fn assert_sha256_parity(produced: &[u8], manifest: &GoldenVectorManifest) -> Result<()> {
    let produced_hex = sha256_hex(produced);
    if produced_hex.eq_ignore_ascii_case(&manifest.sha256) {
        Ok(())
    } else {
        Err(BoundaryDecodeError::ShaMismatch {
            schema: manifest.schema.clone(),
            produced: produced_hex,
            expected: manifest.sha256.clone(),
        })
    }
}

/// Resolve the repo-relative path to this crate's committed golden-vector dir.
///
/// The crate lives at `runtime-rs/crates/tac-boundary-decode/`; the golden
/// vectors live in the crate's own `golden_vectors/` subdir. Resolved via
/// `CARGO_MANIFEST_DIR` so the path is stable across local checkouts + CI.
pub fn golden_vectors_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    Path::new(&manifest_dir).join("golden_vectors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_matches_known_value() {
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
            BoundaryDecodeError::ShaMismatch {
                schema, expected, ..
            } => {
                assert_eq!(schema, "test.v1");
                assert_eq!(expected, "deadbeef");
            }
            other => panic!("expected ShaMismatch, got {other:?}"),
        }
    }
}
