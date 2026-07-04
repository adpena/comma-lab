// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lane AA-SDF render-band COVERAGE rasterizer (task #283) — bit-exact + PORTABLE.
//!
//! Byte-for-byte mirror of the shipped inflate `_lane_coverage` (the `_INFLATE_PY`
//! raw string of `tools/levelset_byte_close_and_eval.py`), itself an op-for-op mirror
//! of `tac.boundary_math.analytic_lane_render_band.rasterize_lane_coverage_range_dependent`
//! (with `_line_row_params` and `_forward_of_rows`). It expands the per-pair openpilot-IPM
//! lane polynomial coeffs (the COUNTED payload) into the (H,W) sub-pixel coverage field for
//! FREE (rule 118 generic rasterizer; 0 archive bytes).
//!
//! # Why THIS piece is bit-exact AND portable (the ship property)
//!
//! Unlike the coord-INR neural forward (blocked by Apple Accelerate `dgemm` reduction
//! order + numpy fp64 `tanh`, MEASURED non-portable in #282) and unlike an fp32 raster
//! (max Δ22 uint8 flips argmax-boundary pixels cross-host, #281), `_lane_coverage` is
//! MEASURED matmul- and transcendental-FREE. Every op is a correctly-rounded IEEE-754
//! fp64 basic operation in a FIXED order:
//!
//! - `np.polyval` = Horner `y = y*x + c` (sequential mul then add, **no FMA**) — element
//!   -independent, so no cross-element reduction whose order a BLAS could permute.
//! - elementwise `/  *  -  abs  max  min  clip  mod`, comparisons — all correctly rounded.
//! - the final `f64 -> f32` cast is round-to-nearest-ties-to-even (IEEE default; Rust `as`
//!   and numpy `astype` agree).
//!
//! Correctly-rounded IEEE-754 basic ops are reproducible on any conformant host (x86-64
//! SSE2 scalar, ARM64) as long as FMA contraction is off (Rust does NOT auto-contract
//! `a*b+c`; we never call `mul_add`) and no extended precision is used (x86-64/ARM64 use
//! true 64-bit, not x87 80-bit). Hence byte-for-byte identical across the contest CPUs —
//! the property fp32 and the BLAS/`tanh` neural forward lack.

use crate::{LevelSetInflateError, Result};
use serde::Deserialize;

/// One lane line's dequantized fp64 coeffs (exactly what `_lane_coverage` consumes as
/// `(cc, hc, dp, dph, dd, fr0, fr1)`).
#[derive(Debug, Clone)]
pub struct LaneLineF64 {
    /// centerline coeffs: `lateral = polyval(cc, forward)` (highest-degree first).
    pub cc: Vec<f64>,
    /// half-width coeffs: `hw = max(polyval(hc, v_row), 0.5)`.
    pub hc: Vec<f64>,
    /// dash period (m); `> 0.0` enables the range-dependent dash gate.
    pub dp: f64,
    /// dash phase (m).
    pub dph: f64,
    /// dash duty (fraction "on" per period).
    pub dd: f64,
    /// forward-range validity `[fr0, fr1]` (m).
    pub fr0: f64,
    pub fr1: f64,
}

/// The scalar render-band geometry/config `_lane_coverage` reads from the LBND header.
#[derive(Debug, Clone)]
pub struct LaneBandGeom {
    pub cam_h: f64,
    pub fx: f64,
    pub fy: f64,
    pub v_h: f64,
    pub softness: f64,
    pub dash_gate: bool,
    pub dash_forward_max_m: f64,
    /// `None` -> use `rw / 2.0` (matches `hdr.get("cx") is None`).
    pub cx: Option<f64>,
}

// --- LBND1 wire-format header (serde; unknown fields — weight/lane_cls/u_mask/format/
// n_pairs/seg_h/seg_w/lane_rgb_mode — are ignored by default). ---------------------
#[derive(Deserialize)]
struct GeomJson {
    cam_h: f64,
    fx: f64,
    fy: f64,
}

#[derive(Deserialize)]
struct LineMetaJson {
    nc: u64,
    nh: u64,
    has_dash: bool,
}

#[derive(Deserialize)]
struct LbndHeaderJson {
    geom: GeomJson,
    v_h: f64,
    softness: f64,
    dash_gate: bool,
    dash_forward_max_m: f64,
    #[serde(default)]
    cx: Option<f64>,
    pairs: Vec<Vec<LineMetaJson>>,
}

const LANE_MAGIC: &[u8] = b"LBND1\x00";

/// numpy `np.mod(a, b)` (== `np.remainder`): C `fmod` then fold to the sign of the
/// divisor `b` (Python remainder convention). Bit-exact vs numpy for finite inputs
/// (`fmod` is exact; the fold is a single exact add). For the dash gate `b = dp > 0`.
#[inline]
fn npy_remainder(a: f64, b: f64) -> f64 {
    let mut m = a % b; // Rust `%` on f64 == C fmod (truncated quotient, sign of `a`)
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
        }
    } else {
        // copysign(0, b)
        m = if b.is_sign_negative() { -0.0 } else { 0.0 };
    }
    m
}

/// numpy `np.maximum` (NaN-propagating; ties -> first). No NaN arises here but we match
/// the semantics exactly so no `f64::max` -0.0/NaN corner can diverge from the oracle.
#[inline]
fn npy_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if a >= b {
        a
    } else {
        b
    }
}

/// numpy `np.minimum` (NaN-propagating; ties -> first).
#[inline]
fn npy_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if a <= b {
        a
    } else {
        b
    }
}

/// `np.clip(x, lo, hi)` == `minimum(maximum(x, lo), hi)`.
#[inline]
fn npy_clip(x: f64, lo: f64, hi: f64) -> f64 {
    npy_minimum(npy_maximum(x, lo), hi)
}

/// numpy `np.polyval(coeffs, x)` — Horner, highest-degree first: `y = 0; for c: y = y*x + c`.
/// Sequential mul-then-add, **no FMA** (matches numpy which never fuses). Element-independent.
#[inline]
fn polyval(coeffs: &[f64], x: f64) -> f64 {
    let mut y = 0.0f64;
    for &c in coeffs {
        y = y * x + c;
    }
    y
}

/// Rasterize the AA-SDF range-dependent lane-band coverage for ONE pair's `lines` into a
/// row-major `(rh, rw)` `f32` buffer, bit-for-bit identical to the shipped inflate
/// `_lane_coverage(lines, rh, rw, hdr)`.
pub fn lane_coverage(lines: &[LaneLineF64], rh: usize, rw: usize, geom: &LaneBandGeom) -> Vec<f32> {
    let mut cov = vec![0.0f32; rh * rw]; // np.zeros((rh,rw), float32) -- +0.0 == 0x00000000
    if lines.is_empty() {
        return cov;
    }
    let cam_h = geom.cam_h;
    let fx = geom.fx;
    let fy = geom.fy;
    let v_h = geom.v_h;
    let soft = npy_maximum(geom.softness, 1e-6); // max(softness, 1e-6)
    let dash_gate = geom.dash_gate;
    let dfm = geom.dash_forward_max_m;
    let cxx = match geom.cx {
        Some(c) => c,
        None => (rw as f64) / 2.0,
    };
    let numer = cam_h * fy; // (cam_h * fy) / max(vr - v_h, 1e-3)
    let threshold = v_h + 1.0; // rows > (v_h + 1.0)

    // Precompute per below-row forward distance (independent of line).
    // below rows are those with (r as f64) > v_h + 1.0.
    let mut below_rows: Vec<usize> = Vec::new();
    let mut vr: Vec<f64> = Vec::new();
    let mut forward: Vec<f64> = Vec::new();
    for r in 0..rh {
        let rf = r as f64;
        if rf > threshold {
            below_rows.push(r);
            vr.push(rf);
            forward.push(numer / npy_maximum(rf - v_h, 1e-3));
        }
    }
    if below_rows.is_empty() {
        return cov;
    }
    let hb = below_rows.len();

    // acc over (below-row, col), float64; unioned (max) across lines. Row-major (hb, rw).
    let mut acc = vec![0.0f64; hb * rw];

    for ln in lines {
        let fr_lo = ln.fr0 - 1.0;
        let fr_hi = ln.fr1 + 5.0;
        let dash_active = dash_gate && ln.dp > 0.0;
        for i in 0..hb {
            let fwd = forward[i];
            let lateral = polyval(&ln.cc, fwd);
            let u_c = cxx - (lateral * fx) / fwd; // cxx - (lateral*fx)/forward
            let hw = npy_maximum(polyval(&ln.hc, vr[i]), 0.5);
            let in_range = (fwd >= fr_lo) && (fwd <= fr_hi);
            let on = if dash_active {
                if fwd < dfm {
                    // near: gate the dash gaps
                    (npy_remainder(fwd - ln.dph, ln.dp) / ln.dp) < ln.dd
                } else {
                    true // far: dashes below SegNet Nyquist -> continuous
                }
            } else {
                true
            };
            let gate = if on && in_range { 1.0f64 } else { 0.0f64 };
            let row_off = i * rw;
            for c in 0..rw {
                let s = hw - (c as f64 - u_c).abs();
                let cov_l = npy_clip(s / soft + 0.5, 0.0, 1.0) * gate;
                let a = &mut acc[row_off + c];
                *a = npy_maximum(*a, cov_l);
            }
        }
    }

    // cov[below] = acc.astype(float32); other rows stay +0.0.
    for (i, &r) in below_rows.iter().enumerate() {
        let src = i * rw;
        let dst = r * rw;
        for c in 0..rw {
            cov[dst + c] = acc[src + c] as f32; // round-to-nearest-even (IEEE default)
        }
    }
    cov
}

/// Parse an LBND1 lane-band blob (`LANE_MAGIC | u32 header_len | header_json | f64 payload`)
/// into per-pair `LaneLineF64` lists + the scalar `LaneBandGeom`. Bit-exact inverse of the
/// shipped inflate `_lane_parse` (fp64 coeff bits preserved from the binary payload).
pub fn parse_lane_band_lbnd1(blob: &[u8]) -> Result<(Vec<Vec<LaneLineF64>>, LaneBandGeom)> {
    if blob.len() < LANE_MAGIC.len() + 4 || &blob[..LANE_MAGIC.len()] != LANE_MAGIC {
        return Err(LevelSetInflateError::BadLaneBand(
            "bad lane-band magic (expected LBND1)".to_string(),
        ));
    }
    let mut off = LANE_MAGIC.len();
    let hlen = u32::from_le_bytes([blob[off], blob[off + 1], blob[off + 2], blob[off + 3]]) as usize;
    off += 4;
    if off + hlen > blob.len() {
        return Err(LevelSetInflateError::BadLaneBand(
            "header length exceeds blob".to_string(),
        ));
    }
    let header: LbndHeaderJson = serde_json::from_slice(&blob[off..off + hlen])
        .map_err(|e| LevelSetInflateError::BadLaneBand(format!("header json: {e}")))?;
    off += hlen;

    // f64 payload, little-endian, C-order: per pair, per line -> cc(nc), hc(nh),
    // [dp,dph,dd iff has_dash], fr0, fr1.
    let payload = &blob[off..];
    if payload.len() % 8 != 0 {
        return Err(LevelSetInflateError::BadLaneBand(
            "float payload not 8-byte aligned".to_string(),
        ));
    }
    let mut vi = 0usize; // index into f64 payload
    let read_f64 = |vi: &mut usize| -> Result<f64> {
        let b = vi.checked_mul(8).ok_or_else(|| {
            LevelSetInflateError::BadLaneBand("payload index overflow".to_string())
        })?;
        if b + 8 > payload.len() {
            return Err(LevelSetInflateError::BadLaneBand(
                "float payload underrun vs header layout".to_string(),
            ));
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&payload[b..b + 8]);
        *vi += 1;
        Ok(f64::from_le_bytes(a))
    };

    let mut pairs: Vec<Vec<LaneLineF64>> = Vec::with_capacity(header.pairs.len());
    for line_metas in &header.pairs {
        let mut lines = Vec::with_capacity(line_metas.len());
        for meta in line_metas {
            let nc = meta.nc as usize;
            let nh = meta.nh as usize;
            let mut cc = Vec::with_capacity(nc);
            for _ in 0..nc {
                cc.push(read_f64(&mut vi)?);
            }
            let mut hc = Vec::with_capacity(nh);
            for _ in 0..nh {
                hc.push(read_f64(&mut vi)?);
            }
            let (dp, dph, dd) = if meta.has_dash {
                let dp = read_f64(&mut vi)?;
                let dph = read_f64(&mut vi)?;
                let dd = read_f64(&mut vi)?;
                (dp, dph, dd)
            } else {
                (0.0, 0.0, 0.5)
            };
            let fr0 = read_f64(&mut vi)?;
            let fr1 = read_f64(&mut vi)?;
            lines.push(LaneLineF64 { cc, hc, dp, dph, dd, fr0, fr1 });
        }
        pairs.push(lines);
    }

    let geom = LaneBandGeom {
        cam_h: header.geom.cam_h,
        fx: header.geom.fx,
        fy: header.geom.fy,
        v_h: header.v_h,
        softness: header.softness,
        dash_gate: header.dash_gate,
        dash_forward_max_m: header.dash_forward_max_m,
        cx: header.cx,
    };
    Ok((pairs, geom))
}

/// Serialize a stacked per-pair coverage set (`n_pairs` × `rh` × `rw` `f32`) to the
/// canonical little-endian `<f4` C-order bytes the golden-vector manifest pins.
pub fn coverage_stack_to_le_bytes(pairs_cov: &[Vec<f32>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pairs_cov.iter().map(|c| c.len() * 4).sum());
    for cov in pairs_cov {
        for &v in cov {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_lines_returns_zero_field() {
        let geom = LaneBandGeom {
            cam_h: 1.2,
            fx: 400.3,
            fy: 399.5,
            v_h: 174.0,
            softness: 1.0,
            dash_gate: true,
            dash_forward_max_m: 55.0,
            cx: None,
        };
        let cov = lane_coverage(&[], 384, 512, &geom);
        assert_eq!(cov.len(), 384 * 512);
        assert!(cov.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn npy_remainder_matches_numpy_sign_of_divisor() {
        // numpy: np.mod(-1.0, 3.0) == 2.0 ; np.mod(7.0, 3.0) == 1.0 ; np.mod(3.0, 3.0) == 0.0
        assert_eq!(npy_remainder(-1.0, 3.0), 2.0);
        assert_eq!(npy_remainder(7.0, 3.0), 1.0);
        assert_eq!(npy_remainder(3.0, 3.0), 0.0);
        assert!(npy_remainder(6.0, 3.0).is_sign_positive());
    }

    #[test]
    fn polyval_horner_no_fma_matches_manual() {
        // p(x) = 2 x^3 - 3 x^2 + 0 x + 7 at x = 1.5
        let cc = [2.0, -3.0, 0.0, 7.0];
        let x = 1.5f64;
        // manual Horner, same op order:
        let mut y = 0.0f64;
        for &c in &cc {
            y = y * x + c;
        }
        assert_eq!(polyval(&cc, x), y);
    }

    #[test]
    fn parse_rejects_bad_magic() {
        let err = parse_lane_band_lbnd1(b"NOPE00\x00\x00\x00\x00").unwrap_err();
        matches!(err, LevelSetInflateError::BadLaneBand(_));
    }

    #[test]
    fn coverage_in_unit_interval() {
        // A single synthetic straight line -> coverage is a valid alpha in [0,1].
        let geom = LaneBandGeom {
            cam_h: 1.2,
            fx: 400.3,
            fy: 399.5,
            v_h: 174.0,
            softness: 1.0,
            dash_gate: false,
            dash_forward_max_m: 55.0,
            cx: None,
        };
        let line = LaneLineF64 {
            cc: vec![0.0, 0.0, 0.0, 1.5],
            hc: vec![0.0, 8.0],
            dp: 0.0,
            dph: 0.0,
            dd: 0.5,
            fr0: 0.0,
            fr1: 200.0,
        };
        let cov = lane_coverage(&[line], 384, 512, &geom);
        assert!(cov.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert!(cov.iter().any(|&v| v > 0.0), "expected non-trivial coverage");
    }
}
