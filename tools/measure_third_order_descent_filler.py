#!/usr/bin/env python3
"""Measure the n600 third-order boundary and V9 phase-filler control surfaces.

This tool is deliberately analysis-only.  It writes one small JSON receipt and never
launches training, a provider job, or an evaluator.  The two measurements are:

* a real-n600 Road/Lane ground-frame quadratic-vs-cubic lane-code comparison, including
  the empirical third/fourth derivative distributions and the #275 margin-derived
  no-flip radius; and
* the existing PHAS1 pre-receiver candidate using the *actual V9 pose-carrier twist*,
  not ``gt_poses`` (PoseNet outputs are not se(3) twists).

Neither surface is an exact d_seg-through-R claim.  The JSON states that limit explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import brotli  # noqa: E402

from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    rasterize_lane_coverage_range_dependent,
    roundtrip_lines_through_rd_tracked,
    serialize_lane_band_rd_tracked,
)
from tac.boundary_math.curve_relative_offset_coder import encode_absolute_2d  # noqa: E402
from tac.boundary_math.lane_sdf_component import (  # noqa: E402
    _FX,
    cluster_lane_lines,
    fit_lane_line,
)
from tac.boundary_math.partition_anisotropy_map import _triple_junctions  # noqa: E402
from tac.boundary_math.phase_primitives import gt_tie_targets_numpy  # noqa: E402
from tac.boundary_math.phase_residual_carrier import (  # noqa: E402
    PhaseCarrierConfig,
    compute_tie_field_from_margins,
    encode_phase_carrier,
)
from tac.boundary_math.xi_spline_residual_coder import decode_residual_matrix  # noqa: E402

D36_BITS = 147_616
D36_BYTES = D36_BITS // 8
N_REQUIRED = 600
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Memory-map a ZIP_STORED NPY member without materializing the full n600 cache."""
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{npz_path}:{member} is compressed; memmap unavailable")
        local_header = int(info.header_offset)
    with npz_path.open("rb") as f:
        f.seek(local_header)
        fields = struct.unpack("<IHHHHHIIIHH", f.read(30))
        if fields[0] != 0x04034B50:
            raise ValueError(f"bad local ZIP header for {npz_path}:{member}")
        f.seek(local_header + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def qdict(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, np.float64)
    return {f"q{100.0 * q:g}": float(np.quantile(a, q)) for q in QUANTILES}


def line_length(coeffs: np.ndarray, lo: float, hi: float) -> float:
    """Numerical arclength of lateral(forward) over the observed line support."""
    if not np.isfinite(lo + hi) or hi <= lo:
        return 0.0
    s = np.linspace(lo, hi, 257, dtype=np.float64)
    deriv = np.polyval(np.polyder(np.asarray(coeffs, np.float64)), s)
    return float(np.trapz(np.sqrt(1.0 + deriv * deriv), s))


def boundary_derivative_measurement(
    lstars: np.memmap, margins: np.memmap
) -> tuple[dict[str, Any], list[list[Any]], list[list[Any]]]:
    """Measure flip radii and real Road/Lane quadratic/cubic/quartic fits."""
    cfg = LaneBandRenderConfig()
    lst_list = [np.asarray(lstars[p], np.int64) for p in range(N_REQUIRED)]
    cubic_pairs, cubic_stats = build_lane_band_pairs_from_lstars(lst_list, cfg, centerline_deg=3)
    quadratic_pairs, quadratic_stats = build_lane_band_pairs_from_lstars(lst_list, cfg, centerline_deg=2)

    flip_radii: list[np.ndarray] = []
    road_lane_flip_radii: list[np.ndarray] = []
    m3_cubic: list[float] = []
    m3_quartic_bound: list[float] = []
    m4_quartic: list[float] = []
    lengths: list[float] = []
    forward_lows: list[float] = []

    for p in range(N_REQUIRED):
        lst = np.asarray(lstars[p], np.int64)
        margin = np.asarray(margins[p], np.float32)
        tie, dirs, active = gt_tie_targets_numpy(lst, margin, band=1.0)
        t = np.asarray(tie[active], np.float64)
        flip_radii.append(np.minimum(t, 1.0 - t))
        # Road/Lane-only support: p class and its selected right/down partner are {0,1}.
        partner = np.empty_like(lst)
        partner[:, :-1] = lst[:, 1:]
        partner[:, -1] = lst[:, -1]
        down = dirs >= 0.5
        partner_down = np.empty_like(lst)
        partner_down[:-1, :] = lst[1:, :]
        partner_down[-1, :] = lst[-1, :]
        partner[down] = partner_down[down]
        rl = active & (((lst == 0) & (partner == 1)) | ((lst == 1) & (partner == 0)))
        trl = np.asarray(tie[rl], np.float64)
        if trl.size:
            road_lane_flip_radii.append(np.minimum(trl, 1.0 - trl))

        for line in cubic_pairs[p]:
            c = np.asarray(line.centerline_coeffs, np.float64)
            if c.size == 4:
                m3_cubic.append(abs(6.0 * float(c[0])))

        for cluster in cluster_lane_lines(lst, lane_cls=cfg.lane_cls, v_h=cfg.v_h):
            line4 = fit_lane_line(cluster, centerline_deg=4, fit_dash=False, v_h=cfg.v_h)
            if line4 is None:
                continue
            c = np.asarray(line4.centerline_coeffs, np.float64)
            if c.size != 5:
                continue
            lo, hi = map(float, line4.forward_range)
            # y'''(s)=24*a4*s+6*a3; its supremum on an interval occurs at an endpoint.
            d3 = np.polyder(c, 3)
            m3_quartic_bound.append(float(max(abs(np.polyval(d3, lo)), abs(np.polyval(d3, hi)))))
            m4_quartic.append(abs(24.0 * float(c[0])))
            lengths.append(line_length(c, lo, hi))
            forward_lows.append(lo)

    eps_all = np.concatenate(flip_radii)
    eps_rl = np.concatenate(road_lane_flip_radii)
    m3q = np.asarray(m3_quartic_bound, np.float64)
    m4q = np.asarray(m4_quartic, np.float64)
    lens = np.asarray(lengths, np.float64)
    fwd_lo = np.asarray(forward_lows, np.float64)

    # A strict all-state epsilon is the empirical minimum.  Also expose distributional
    # p0.1/p1 tolerances; those are not uniform guarantees and are labelled accordingly.
    predicted: dict[str, Any] = {}
    for name, eps_px in (("strict_min", float(eps_rl.min())),
                         ("distributional_p0.1", float(np.quantile(eps_rl, 0.001))),
                         ("distributional_p1", float(np.quantile(eps_rl, 0.01)))):
        if eps_px <= 0.0:
            predicted[name] = {"epsilon_px": eps_px, "N2_intervals": None, "N3_intervals": None,
                               "status": "UNBOUNDED_at_zero_uniform_tolerance"}
            continue
        # The fit lives in ground metres but #275 epsilon lives in render pixels.
        # u = cx - lateral*fx/forward, hence one pixel = forward/fx metres.
        # Use each arc's near endpoint for the smallest (conservative) ground tolerance.
        eps_m = eps_px * fwd_lo / float(_FX)
        n2 = np.ceil(lens * np.cbrt(m3q / (6.0 * eps_m))).astype(np.int64)
        n3 = np.ceil(lens * np.power(m4q / (24.0 * eps_m), 0.25)).astype(np.int64)
        predicted[name] = {
            "epsilon_px": eps_px,
            "epsilon_ground_m_near_endpoint": qdict(eps_m),
            "pixel_to_ground_conversion": "epsilon_m_i = epsilon_px * forward_min_i / fx",
            "N2_intervals_sum": int(n2.sum()),
            "N3_intervals_sum": int(n3.sum()),
            "N2_quadratic_terms_3_per_interval": int(3 * n2.sum()),
            "N3_cubic_terms_4_per_interval": int(4 * n3.sum()),
            "N3_over_N2_interval_ratio": float(n3.sum() / n2.sum()),
            "N3_over_N2_term_ratio": float((4 * n3.sum()) / (3 * n2.sum())),
            "status": "DERIVED_dimensionally_matched_Taylor_bound_on_quartic_ground_frame_fits",
        }

    return ({
        "scope": "real n600 Road/Lane ground-frame generator; not the full V9 separatrix",
        "n_pairs": N_REQUIRED,
        "n_scorer_states": int(N_REQUIRED * lstars.shape[1] * lstars.shape[2]),
        "flip_radius_all_boundaries_px": {"count": int(eps_all.size), "quantiles": qdict(eps_all)},
        "flip_radius_road_lane_px": {"count": int(eps_rl.size), "quantiles": qdict(eps_rl)},
        "epsilon_definition": "min(t,1-t) at #275 genuine-V straddles; linearized normal displacement before either sampled endpoint crosses the tie",
        "cubic_fit_M3_abs_lateral_m_per_forward_m3": {
            "count": len(m3_cubic), "quantiles": qdict(np.asarray(m3_cubic)),
            "empirical_bound_max": float(np.max(m3_cubic)),
        },
        "quartic_check_M3_sup_abs_lateral_m_per_forward_m3": {
            "count": int(m3q.size), "quantiles": qdict(m3q), "empirical_bound_max": float(m3q.max())
        },
        "quartic_check_M4_abs_lateral_m_per_forward_m4": {
            "count": int(m4q.size), "quantiles": qdict(m4q), "empirical_bound_max": float(m4q.max())
        },
        "line_arclength_m": {"count": int(lens.size), "quantiles": qdict(lens), "sum": float(lens.sum())},
        "predicted_piece_counts": predicted,
        "fit_stats": {"quadratic": quadratic_stats, "cubic": cubic_stats},
    }, quadratic_pairs, cubic_pairs)


def lane_code_measurement(
    lstars: np.memmap, quadratic_pairs: list[list[Any]], cubic_pairs: list[list[Any]]
) -> dict[str, Any]:
    """Compare the current cubic LBND2 code to quadratic plus exact label correction."""
    cfg = LaneBandRenderConfig()
    q2_dec, q2_raw, q2_meta = roundtrip_lines_through_rd_tracked(quadratic_pairs, cfg)
    q3_dec, q3_raw, q3_meta = roundtrip_lines_through_rd_tracked(cubic_pairs, cfg)
    q2_bytes = len(brotli.compress(q2_raw, quality=11))
    q3_bytes = len(brotli.compress(q3_raw, quality=11))
    # Assert the direct serializer agrees with the roundtrip payload custody.
    if q2_raw != serialize_lane_band_rd_tracked(quadratic_pairs, cfg)[0]:
        raise AssertionError("quadratic LBND2 roundtrip payload mismatch")
    if q3_raw != serialize_lane_band_rd_tracked(cubic_pairs, cfg)[0]:
        raise AssertionError("cubic LBND2 roundtrip payload mismatch")

    correction: list[np.ndarray] = []
    xor2_gt = xor3_gt = lane_gt = pred2 = pred3 = 0
    total = N_REQUIRED * int(lstars.shape[1]) * int(lstars.shape[2])
    for p in range(N_REQUIRED):
        gt = np.asarray(lstars[p]) == cfg.lane_cls
        m2 = rasterize_lane_coverage_range_dependent(
            q2_dec[p], h=gt.shape[0], w=gt.shape[1], softness=cfg.softness,
            dash_gate=cfg.dash_gate, dash_forward_max_m=cfg.dash_forward_max_m, v_h=cfg.v_h,
        ) >= 0.5
        m3 = rasterize_lane_coverage_range_dependent(
            q3_dec[p], h=gt.shape[0], w=gt.shape[1], softness=cfg.softness,
            dash_gate=cfg.dash_gate, dash_forward_max_m=cfg.dash_forward_max_m, v_h=cfg.v_h,
        ) >= 0.5
        xor2_gt += int(np.count_nonzero(m2 ^ gt))
        xor3_gt += int(np.count_nonzero(m3 ^ gt))
        lane_gt += int(np.count_nonzero(gt))
        pred2 += int(np.count_nonzero(m2))
        pred3 += int(np.count_nonzero(m3))
        correction.append(np.flatnonzero((m2 ^ m3).reshape(-1)))
    corr_blob = encode_absolute_2d(correction, int(lstars.shape[1]), int(lstars.shape[2]))
    corr_count = int(sum(a.size for a in correction))
    return {
        "scope": "pre-receiver Lane-vs-not-Lane scorer-grid mask; NOT d_seg-through-R",
        "current_coder_identity": "cubic LBND2 (the present horizon/lane family already stores c3)",
        "quadratic_LBND2_brotli_bytes": q2_bytes,
        "cubic_current_LBND2_brotli_bytes": q3_bytes,
        "quadratic_LBND2_meta": q2_meta,
        "cubic_current_LBND2_meta": q3_meta,
        "quadratic_xor_cubic_exact_correction_ABS2_bytes": len(corr_blob),
        "quadratic_plus_exact_cubic_mask_correction_bytes": q2_bytes + len(corr_blob),
        "matched_pre_receiver_target": "quadratic+ABS2 decodes bit-identically to the cubic Lane mask",
        "matched_pre_receiver_saving_bytes_vs_current_cubic": q3_bytes - (q2_bytes + len(corr_blob)),
        "quadratic_vs_gt_lane_xor_states": xor2_gt,
        "cubic_vs_gt_lane_xor_states": xor3_gt,
        "quadratic_vs_gt_lane_xor_fraction_all_states": xor2_gt / total,
        "cubic_vs_gt_lane_xor_fraction_all_states": xor3_gt / total,
        "quadratic_xor_cubic_support_states": corr_count,
        "quadratic_xor_cubic_support_fraction_all_states": corr_count / total,
        "gt_lane_states": lane_gt,
        "quadratic_predicted_lane_states": pred2,
        "cubic_predicted_lane_states": pred3,
        "receiver_blocker": "no V9 decoder consumes either jet stream into RGB before R+SegNet; matched d_seg-through-R bytes are therefore NOT MEASURED",
    }


def phase_candidate_measurement(
    lstars: np.memmap, margins: np.memmap, v9_ckpt: Path
) -> dict[str, Any]:
    """Measure PHAS1 using the V9 receiver's actual stored twist surface."""
    with np.load(v9_ckpt, allow_pickle=False) as z:
        for key in ("pose_carrier.xi_stored", "pose_carrier.dxi"):
            if key not in z.files:
                raise KeyError(f"{v9_ckpt} lacks {key}")
        xi = (np.asarray(z["pose_carrier.xi_stored"], np.float64)
              + np.asarray(z["pose_carrier.dxi"], np.float64))
    if xi.shape != (N_REQUIRED, 6):
        raise ValueError(f"V9 effective twist must be (600,6), got {xi.shape}")
    cfg = PhaseCarrierConfig()
    tie_fields: list[np.ndarray] = []
    active_masks: list[np.ndarray] = []
    class_maps: list[np.ndarray] = []
    for p in range(N_REQUIRED):
        tie, mask, cmap = compute_tie_field_from_margins(lstars[p], margins[p], cfg)
        tie_fields.append(tie)
        active_masks.append(mask)
        class_maps.append(cmap)
    # ``xi`` is already the V9 receiver's rho-first se(3) twist.  Calling the convenience
    # ``phase_carrier_report`` here would incorrectly apply pose-output calibration a second time.
    section, rep = encode_phase_carrier(tie_fields, active_masks, class_maps, xi, cfg)
    off = 6
    (hlen,) = struct.unpack_from("<I", section, off)
    off += 4
    hdr = json.loads(section[off:off + hlen].decode("utf-8"))
    off += hlen + N_REQUIRED * 6 * 2
    (rlen,) = struct.unpack_from("<I", section, off)
    off += 4
    residuals = decode_residual_matrix(
        section[off:off + rlen], int(hdr["residual_scheme_id"]), int(hdr["residual_count"]), 1
    ).reshape(-1)
    nonzero = int(np.count_nonzero(residuals))
    triple_total = triple_active = triple_nonzero = 0
    cursor = 0
    for p in range(N_REQUIRED):
        tj = _triple_junctions(np.asarray(lstars[p]))
        nz_map = np.zeros(tj.shape, dtype=bool)
        for cls, count in zip(cfg.classes, rep.per_frame_class_counts[p], strict=True):
            sel = active_masks[p] & (class_maps[p] == cls)
            idx = np.flatnonzero(sel.reshape(-1))
            if idx.size != count:
                raise AssertionError("phase residual routing count mismatch")
            nz_map.reshape(-1)[idx] = residuals[cursor:cursor + count] != 0
            cursor += count
        triple_total += int(np.count_nonzero(tj))
        triple_active += int(np.count_nonzero(tj & active_masks[p]))
        triple_nonzero += int(np.count_nonzero(tj & nz_map))
    if cursor != residuals.size:
        raise AssertionError("phase residual stream not fully routed")
    per_class = np.asarray(rep.per_frame_class_counts, np.int64).sum(axis=0)
    return {
        "scope": "V9-twist PHAS1 pre-receiver tie-residual candidate; not a finite/minimal descent filler",
        "xi_source": "pose_carrier.xi_stored + pose_carrier.dxi from V9 checkpoint",
        "n_pairs": rep.n_frames,
        "active_ground_straddle_states": rep.total_residual_count,
        "active_ground_straddle_fraction_all_states": rep.total_residual_count / (N_REQUIRED * 384 * 512),
        "nonzero_quantized_residual_states": nonzero,
        "nonzero_quantized_residual_fraction_active": nonzero / rep.total_residual_count,
        "nonzero_quantized_residual_fraction_all_states": nonzero / (N_REQUIRED * 384 * 512),
        "per_class_active_states_road_lane_undrivable": [int(v) for v in per_class],
        "triple_junction_states": triple_total,
        "triple_junction_active_ground_straddle_states": triple_active,
        "triple_junction_nonzero_quantized_residual_states": triple_nonzero,
        "section_bytes": len(section),
        "xi_amortized_residual_bytes": rep.xi_amortized_residual_bytes,
        "raw_tie_residual_bytes": rep.raw_tie_residual_bytes,
        "amortization_ratio": rep.amortization_ratio,
        "mean_abs_residual_q": rep.mean_abs_residual_q,
        "max_abs_residual_q": rep.max_abs_residual_q,
        "tie_recon_rmse_px": rep.tie_recon_rmse_px,
        "reconstruction_bit_identical": rep.reconstruction_bit_identical,
        "D36_bits": D36_BITS,
        "D36_bytes": D36_BYTES,
        "section_over_D36_ratio": len(section) / D36_BYTES,
        "section_minus_D36_bytes": len(section) - D36_BYTES,
        "receiver_blockers": [
            "d_cov+d_gauge is registered only as a scalar distortion taxonomy, not a pointwise state projection",
            "PHAS1 has no V9 inflate/render consumer and recovered_d_seg is explicitly OWED",
            "therefore section_bytes is a candidate upper bound, not the finite obstruction or a minimal filler",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--v9-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    lstars = stored_npy_memmap(args.gt_cache, "lstars")
    margins = stored_npy_memmap(args.gt_cache, "margins")
    if lstars.shape[0] < N_REQUIRED or margins.shape[0] < N_REQUIRED:
        raise ValueError(f"n600 is mandatory; got lstars={lstars.shape}, margins={margins.shape}")

    third, q2, q3 = boundary_derivative_measurement(lstars, margins)
    lane = lane_code_measurement(lstars, q2, q3)
    phase = phase_candidate_measurement(lstars, margins, args.v9_ckpt)
    receipt = {
        "schema": "third_order_descent_filler_measurement.v1",
        "labels": {
            "numeric_arrays": "MEASURED on real n600 cached SegNet states",
            "Taylor_piece_counts": "DERIVED from measured derivatives and epsilon",
            "score_authority": "NONE; no exact byte-closed d_seg-through-R row was run",
        },
        "provenance": {
            "git_sha": git_value("rev-parse", "HEAD"),
            "git_worktree_dirty": bool(git_value("status", "--porcelain", "--untracked-files=no")),
            "gt_cache": str(args.gt_cache),
            "gt_cache_sha256": sha256_file(args.gt_cache),
            "v9_ckpt": str(args.v9_ckpt),
            "v9_ckpt_sha256": sha256_file(args.v9_ckpt),
            "measurement_tool": str(Path(__file__).resolve().relative_to(REPO)),
            "measurement_tool_sha256": sha256_file(Path(__file__)),
            "numpy_version": np.__version__,
            "n_pairs": N_REQUIRED,
            "axis": "[macOS-CPU advisory] numpy-fp32/fp64 analysis; no score claim",
        },
        "arm1_third_order": third,
        "arm1_lane_code": lane,
        "arm2_phase_candidate": phase,
        "elapsed_seconds": time.time() - t0,
        "pointer_moved": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
