// SPDX-License-Identifier: MIT OR Apache-2.0
//! Golden-vector loader + byte-for-byte parity helpers.
//!
//! Mirrors the canonical pattern in the sibling `tac-boundary-decode` /
//! `tac-packet-compiler` crates. The committed golden vectors under this crate's
//! `golden_vectors/` are produced by the Python ORACLE (`generate_golden_vectors.py`,
//! which calls the REAL `tac.lossless.range_coder` / `tac.boundary_math.xi_pose_coder`
//! functions on REAL contest gt_poses ξ). Each vector pins the SHA-256 of the DECODED
//! output the Rust port must reproduce bit-for-bit, plus the input byte fixtures the
//! Rust port reads.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{LevelSetInflateError, Result};

/// Common shape every golden-vector JSON manifest carries. Per-schema fields
/// (e.g. `count`, `seed`, `lo`) live in `extras`.
#[derive(Debug, Clone, Deserialize)]
pub struct GoldenVectorManifest {
    /// Versioned schema label (e.g. `"levelset_xi_range_decode.v1"`).
    pub schema: String,
    /// SHA-256 hex digest of the canonical decoded output.
    pub sha256: String,
    /// All other manifest fields.
    #[serde(flatten)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

impl GoldenVectorManifest {
    /// Read a `usize`-valued manifest field, erroring loudly if absent/malformed.
    pub fn usize_field(&self, key: &str) -> Result<usize> {
        self.extras
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| {
                LevelSetInflateError::GoldenVectorIo(format!(
                    "manifest {} missing/invalid usize field '{}'",
                    self.schema, key
                ))
            })
    }

    /// Read an `i64`-valued manifest field, erroring loudly if absent/malformed.
    pub fn i64_field(&self, key: &str) -> Result<i64> {
        self.extras
            .get(key)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| {
                LevelSetInflateError::GoldenVectorIo(format!(
                    "manifest {} missing/invalid int field '{}'",
                    self.schema, key
                ))
            })
    }
}

/// Load a golden-vector manifest from disk.
pub fn load_golden_vector(path: &Path) -> Result<GoldenVectorManifest> {
    let text = fs::read_to_string(path).map_err(|e| {
        LevelSetInflateError::GoldenVectorIo(format!("read {}: {}", path.display(), e))
    })?;
    serde_json::from_str(&text).map_err(|e| {
        LevelSetInflateError::GoldenVectorIo(format!("parse {}: {}", path.display(), e))
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
        Err(LevelSetInflateError::ShaMismatch {
            schema: manifest.schema.clone(),
            produced: produced_hex,
            expected: manifest.sha256.clone(),
        })
    }
}

/// Serialize an `i64` slice as little-endian `<i8` C-order bytes — the canonical
/// decoded-output encoding the golden-vector manifests pin (matches numpy's
/// `np.asarray(x, "<i8").tobytes(order="C")`).
pub fn i64_slice_to_le_bytes(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Resolve the repo-relative path to this crate's committed golden-vector dir.
pub fn golden_vectors_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let base = Path::new(&manifest_dir).join("golden_vectors");
    base.canonicalize().unwrap_or(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_matches_known_empty_value() {
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(sha256_hex(b""), expected);
    }

    #[test]
    fn i64_le_bytes_roundtrip() {
        let v = [1i64, -899, 4001, 0];
        let b = i64_slice_to_le_bytes(&v);
        assert_eq!(b.len(), 32);
        assert_eq!(&b[0..8], &1i64.to_le_bytes());
        assert_eq!(&b[8..16], &(-899i64).to_le_bytes());
    }

    #[test]
    fn parity_returns_structured_mismatch() {
        let manifest = GoldenVectorManifest {
            schema: "test.v1".to_string(),
            sha256: "deadbeef".to_string(),
            extras: serde_json::Map::new(),
        };
        let err = assert_sha256_parity(b"x", &manifest).expect_err("mismatch");
        match err {
            LevelSetInflateError::ShaMismatch { schema, expected, .. } => {
                assert_eq!(schema, "test.v1");
                assert_eq!(expected, "deadbeef");
            }
            other => panic!("expected ShaMismatch, got {other:?}"),
        }
    }
}
