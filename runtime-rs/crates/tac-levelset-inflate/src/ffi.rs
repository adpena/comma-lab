// SPDX-License-Identifier: MIT OR Apache-2.0
//! C-ABI FFI for the lane AA-SDF coverage rasterizer — the ship-integration surface.
//!
//! Exposed via `crate-type = ["cdylib", "rlib"]` so the numpy inflate can swap the
//! per-pixel coverage hot path to the bit-exact Rust port through `ctypes` (no PyO3 /
//! maturin / Python-headers build dependency — a plain `cdylib` loaded with
//! `ctypes.CDLL`). The Python inflate keeps the (one-time, cheap) LBND1/LBND2 parse and
//! feeds the SAME per-pair coeffs; because the coverage output is byte-identical (proven
//! by the golden-vector SHA gate), swapping it in leaves the decoded frames — and thus
//! the archive-scored bytes — UNCHANGED. Opt-in only (`INFLATE_RUST_LANE=1`); default is
//! the numpy path.
//!
//! All entry points are panic-safe (`catch_unwind`) and validate pointers/lengths, per
//! the sibling crates' fail-loud convention (no silent-skip).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use crate::lane_coverage::{lane_coverage, parse_lane_band_lbnd1};

/// Error sentinels (negative). Success returns a non-negative count.
const ERR_NULL: i64 = -1;
const ERR_PARSE: i64 = -2;
const ERR_OUT_LEN: i64 = -3;
const ERR_PANIC: i64 = -4;

/// Number of pairs in an LBND1 lane-band blob. Returns `n_pairs >= 0` or a negative sentinel.
///
/// # Safety
/// `blob_ptr` must point to `blob_len` readable bytes (or be null with `blob_len == 0`).
#[no_mangle]
pub unsafe extern "C" fn tac_lane_band_n_pairs(blob_ptr: *const u8, blob_len: usize) -> i64 {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if blob_ptr.is_null() {
            return ERR_NULL;
        }
        let blob = slice::from_raw_parts(blob_ptr, blob_len);
        match parse_lane_band_lbnd1(blob) {
            Ok((pairs, _geom)) => pairs.len() as i64,
            Err(_) => ERR_PARSE,
        }
    }));
    res.unwrap_or(ERR_PANIC)
}

/// Rasterize the AA-SDF coverage for EVERY pair in an LBND1 blob at `(rh, rw)`, writing the
/// stacked `(n_pairs * rh * rw)` `f32` C-order field into `out_ptr`. Returns `n_pairs`
/// written, or a negative sentinel. `out_len` MUST equal `n_pairs * rh * rw`.
///
/// # Safety
/// `blob_ptr`/`blob_len` describe a readable byte range; `out_ptr`/`out_len` describe a
/// writable `f32` range. The caller allocates `out` (query the size with
/// `tac_lane_band_n_pairs` first).
#[no_mangle]
pub unsafe extern "C" fn tac_lane_coverage_all_lbnd1(
    blob_ptr: *const u8,
    blob_len: usize,
    rh: usize,
    rw: usize,
    out_ptr: *mut f32,
    out_len: usize,
) -> i64 {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if blob_ptr.is_null() || out_ptr.is_null() {
            return ERR_NULL;
        }
        let blob = slice::from_raw_parts(blob_ptr, blob_len);
        let (pairs, geom) = match parse_lane_band_lbnd1(blob) {
            Ok(v) => v,
            Err(_) => return ERR_PARSE,
        };
        let per = rh.saturating_mul(rw);
        let need = pairs.len().saturating_mul(per);
        if out_len != need {
            return ERR_OUT_LEN;
        }
        let out = slice::from_raw_parts_mut(out_ptr, out_len);
        for (pi, lines) in pairs.iter().enumerate() {
            let cov = lane_coverage(lines, rh, rw, &geom);
            out[pi * per..(pi + 1) * per].copy_from_slice(&cov);
        }
        pairs.len() as i64
    }));
    res.unwrap_or(ERR_PANIC)
}
