//! PR93 lowpass-luma residual codec — byte-grammar Rust port.
//!
//! Byte-for-byte parity contract: the Python oracle
//! (`src/tac/packet_compiler/pr93_lowpass_luma.py::serialize_lowpass_luma_residual`)
//! emits a 5-byte header (`u8 n_coeffs`, `u16 height LE`, `u16 width LE`)
//! followed by `n_coeffs` little-endian fp32 coefficients (12 or 24 bytes).
//!
//! The native port intentionally exposes only the SERIALIZE step (the
//! encoder's least-squares fit is research-grade numpy work and is not
//! reusable from a deterministic-bytes substrate). The golden vector pins
//! the SHA-256 of the serialised wire bytes; the encoder's coefficients are
//! supplied directly by the caller (the golden vector uses literal
//! `[1.0, 0.5, -0.25, 0.125, -0.0625, 0.03125]`).
//!
//! Wire format (matches the Python oracle exactly):
//!
//! ```text
//! n_coeffs  u8 LE          (3 = linear, 6 = Legendre quadratic)
//! height    u16 LE         (1..=65535)
//! width     u16 LE         (1..=65535)
//! coeffs    fp32 LE * n_coeffs
//! ```
//!
//! Sibling golden vector:
//! `src/tac/packet_compiler/golden_vectors/pr93_lowpass_luma_v1.json`.
//!
//! # Wire-format rationale
//!
//! 5-byte header is intentionally small: 384x512 (LE u16) fits in 4 bytes
//! plus the 1-byte coefficient count. No magic prefix is used; the consumer
//! locates the section via an outer container's length-prefix grammar.
//! Adding a magic would inflate every per-frame correction by 4 bytes on
//! the contest scoring path (24 bytes → 28 bytes = ~17% bloat for the
//! Legendre quadratic form), so the Python oracle deliberately omits it.

use crate::{PacketCompilerError, Result};

/// Result of serialising a lowpass-luma residual coefficient block.
///
/// Mirrors the wire-format contract of
/// `tac.packet_compiler.serialize_lowpass_luma_residual`.
#[derive(Debug, Clone)]
pub struct LowpassLumaResidual {
    /// Coefficient vector. Length MUST be exactly 3 (linear) or 6
    /// (Legendre quadratic) per the Python oracle's contract.
    pub coefficients: Vec<f32>,
    /// Eval canvas height. MUST be `1..=65535` (uint16 wire field).
    pub height: u16,
    /// Eval canvas width. MUST be `1..=65535` (uint16 wire field).
    pub width: u16,
}

/// Serialise a [`LowpassLumaResidual`] to little-endian wire bytes.
///
/// Mirrors `tac.packet_compiler.serialize_lowpass_luma_residual`.
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if `coefficients.len()` is not
///   3 or 6 (the only supported basis sizes per the Python oracle).
pub fn serialize_lowpass_luma_residual(residual: &LowpassLumaResidual) -> Result<Vec<u8>> {
    let n_coeffs = residual.coefficients.len();
    if n_coeffs != 3 && n_coeffs != 6 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lowpass-luma residual must have 3 or 6 coefficients; got {n_coeffs}"
        )));
    }
    if residual.height == 0 || residual.width == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lowpass-luma residual height/width must be >= 1; got height={} width={}",
            residual.height, residual.width
        )));
    }
    let mut out = Vec::with_capacity(5 + n_coeffs * 4);
    // Header: <BHH> — u8 n_coeffs, u16 height LE, u16 width LE.
    out.push(n_coeffs as u8);
    out.extend_from_slice(&residual.height.to_le_bytes());
    out.extend_from_slice(&residual.width.to_le_bytes());
    // Coefficients: fp32 little-endian.
    for &c in &residual.coefficients {
        out.extend_from_slice(&c.to_le_bytes());
    }
    Ok(out)
}

/// Deserialise wire bytes produced by [`serialize_lowpass_luma_residual`].
///
/// Mirrors `tac.packet_compiler.deserialize_lowpass_luma_residual`. The
/// caller must consume the exact byte slice produced by the encoder (a
/// trailing-bytes-included slice produces a hard error so silent
/// corruption is impossible).
///
/// # Errors
///
/// * [`PacketCompilerError::GoldenVectorIo`] if the header is truncated,
///   the coefficient count is unsupported, or the slice does not contain
///   exactly the expected number of bytes.
pub fn deserialize_lowpass_luma_residual(blob: &[u8]) -> Result<LowpassLumaResidual> {
    if blob.len() < 5 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lowpass-luma blob too short for header: need 5 bytes have {}",
            blob.len()
        )));
    }
    let n_coeffs = blob[0] as usize;
    if n_coeffs != 3 && n_coeffs != 6 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lowpass-luma unsupported n_coeffs {n_coeffs}; expected 3 or 6"
        )));
    }
    let height = u16::from_le_bytes([blob[1], blob[2]]);
    let width = u16::from_le_bytes([blob[3], blob[4]]);
    if height == 0 || width == 0 {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lowpass-luma blob height/width must be >= 1; got height={height} width={width}"
        )));
    }
    let expected = 5 + n_coeffs * 4;
    if blob.len() != expected {
        return Err(PacketCompilerError::GoldenVectorIo(format!(
            "lowpass-luma blob size mismatch: have {} bytes, expected {expected}",
            blob.len()
        )));
    }
    let mut coefficients = Vec::with_capacity(n_coeffs);
    for i in 0..n_coeffs {
        let off = 5 + i * 4;
        coefficients.push(f32::from_le_bytes([
            blob[off],
            blob[off + 1],
            blob[off + 2],
            blob[off + 3],
        ]));
    }
    Ok(LowpassLumaResidual {
        coefficients,
        height,
        width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_three_coeffs_roundtrip() {
        let r = LowpassLumaResidual {
            coefficients: vec![0.25_f32, -0.5_f32, 1.0_f32],
            height: 10,
            width: 20,
        };
        let blob = serialize_lowpass_luma_residual(&r).unwrap();
        assert_eq!(blob.len(), 5 + 3 * 4);
        assert_eq!(blob[0], 3);
        assert_eq!(u16::from_le_bytes([blob[1], blob[2]]), 10);
        assert_eq!(u16::from_le_bytes([blob[3], blob[4]]), 20);
        let back = deserialize_lowpass_luma_residual(&blob).unwrap();
        assert_eq!(back.coefficients, r.coefficients);
        assert_eq!(back.height, 10);
        assert_eq!(back.width, 20);
    }

    #[test]
    fn serialize_six_coeffs_roundtrip() {
        let r = LowpassLumaResidual {
            coefficients: vec![1.0_f32, 0.5, -0.25, 0.125, -0.0625, 0.03125],
            height: 384,
            width: 512,
        };
        let blob = serialize_lowpass_luma_residual(&r).unwrap();
        assert_eq!(blob.len(), 5 + 6 * 4);
        assert_eq!(blob[0], 6);
        let back = deserialize_lowpass_luma_residual(&blob).unwrap();
        assert_eq!(back.coefficients, r.coefficients);
    }

    #[test]
    fn rejects_invalid_n_coeffs() {
        let r = LowpassLumaResidual {
            coefficients: vec![1.0_f32, 0.5, 0.25, 0.125],
            height: 10,
            width: 10,
        };
        assert!(serialize_lowpass_luma_residual(&r).is_err());
    }

    #[test]
    fn rejects_zero_dimensions() {
        let r = LowpassLumaResidual {
            coefficients: vec![1.0_f32, 0.5, 0.25],
            height: 0,
            width: 10,
        };
        assert!(serialize_lowpass_luma_residual(&r).is_err());
    }

    #[test]
    fn deserialize_rejects_short_blob() {
        let bad = [3u8, 10, 0, 10, 0, 0, 0, 0, 0]; // claims 3 coeffs but only 9 bytes
        assert!(deserialize_lowpass_luma_residual(&bad).is_err());
    }

    #[test]
    fn deserialize_rejects_bad_n_coeffs() {
        // header claims 4 coefficients
        let bad = [
            4u8, 10, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert!(deserialize_lowpass_luma_residual(&bad).is_err());
    }

    #[test]
    fn deserialize_rejects_trailing_bytes() {
        let r = LowpassLumaResidual {
            coefficients: vec![1.0_f32, 0.5, 0.25],
            height: 1,
            width: 1,
        };
        let mut blob = serialize_lowpass_luma_residual(&r).unwrap();
        blob.push(0xFF);
        assert!(deserialize_lowpass_luma_residual(&blob).is_err());
    }

    #[test]
    fn known_layout_pinned_bytes() {
        // Pinned literal: n_coeffs=6, height=384, width=512, coeffs same as
        // golden vector. Verifies the wire layout is exactly what the Python
        // oracle emits (header byte order + fp32 LE).
        let r = LowpassLumaResidual {
            coefficients: vec![1.0_f32, 0.5, -0.25, 0.125, -0.0625, 0.03125],
            height: 384,
            width: 512,
        };
        let blob = serialize_lowpass_luma_residual(&r).unwrap();
        assert_eq!(blob.len(), 29); // matches golden vector payload_len=29
        assert_eq!(&blob[0..5], &[6, 0x80, 0x01, 0x00, 0x02]);
        // 1.0_f32 LE = 00 00 80 3F
        assert_eq!(&blob[5..9], &[0x00, 0x00, 0x80, 0x3F]);
        // 0.5_f32 LE = 00 00 00 3F
        assert_eq!(&blob[9..13], &[0x00, 0x00, 0x00, 0x3F]);
    }
}
