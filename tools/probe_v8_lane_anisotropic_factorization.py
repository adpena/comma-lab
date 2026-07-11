#!/usr/bin/env python
"""v8 LANE rate-crux probe: ground-frame + factorization + SPD-cone, MEASURED.

Measures the full factorized lane byte budget vs the two anchors from the v8
Laguerre feasibility memo (FEED-lag):
  * the 97 B/frame image-space baseline (poly runs model, recall 0.835 / IoU 0.601),
  * the SPEC's 1-2 KB/600-frames lane target.

Stage ladder (each stage REAL round-trip -> raster -> recall/precision/IoU):
  S0  image-space baseline reproduction (fit_lane_curves / lane_curve_bytes)
  S1  ground-frame per-frame independent (12-bit quantized knot vectors,
      per-obs periodic dash — documents the periodic-dash-model negative)
  S2  ground-frame + tracking + SPD-cone water-filled knot coding (theta sweep,
      solid band)
  S3  S2 (shape dims only) + LOSSLESS world-aligned occupancy gate (dashes +
      visibility; near-field bin-count sweep)
  S4  the SKELETON arm: smoothed knots + static world paint x visibility window
      (drops the per-frame argmax jitter; measures the smooth-geometry floor)

$0, memory-light: numpy load of the cached argmax label maps ONLY (no scorer
forward, no model inference). ``[macOS-MLX advisory]`` — a geometric rate
estimate, NOT byte-closed through R + the frozen SegNet, NOT a score.

Usage:
  .venv/bin/python tools/probe_v8_lane_anisotropic_factorization.py \
      --cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --out experiments/results/v8_lane_factorization_probe_20260710
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.lane_ground_factorization import (  # noqa: E402
    encode_shift_stream,
    encode_tracks_occupancy,
    encode_tracks_spd,
    estimate_global_shifts,
    evaluate_tracks_raster,
    fit_frame_ground_lanes,
    lane_band_metrics,
    lane_line_from_vec,
    per_frame_quantized_ground,
    pooled_hilbert_distance,
    robust_dim_scales,
    shifts_to_bins,
    track_ground_lanes,
    unit_dim_scales,
)
from tac.boundary_math.lane_sdf_component import rasterize_lane_band  # noqa: E402
from tac.boundary_math.partition_collapse import (  # noqa: E402
    LANE,
    fit_lane_curves,
    lane_curve_bytes,
    render_lane_band,
)


def _self_detect_lane(labels: np.ndarray) -> dict:
    """Verify the canonical class order by spatial/static signature (CLAUDE.md:
    never luma-sort, never trust the index blindly). Lane = thin (<3% area),
    MyCar = bottom static, Undrivable = top."""
    n_cls = 5
    h = labels.shape[1]
    areas = np.array([(labels == c).mean() for c in range(n_cls)])
    centroids = []
    for c in range(n_cls):
        rr = np.argwhere((labels == c).any(axis=2))
        centroids.append(float(rr[:, 1].mean()) if len(rr) else float("nan"))
    m4 = labels == 4
    union = (m4[:-1] | m4[1:]).sum()
    iou4 = float((m4[:-1] & m4[1:]).sum() / union) if union else 0.0
    ok = bool(
        areas[LANE] < 0.03
        and iou4 > 0.9
        and centroids[4] > 0.6 * h
        and centroids[2] < 0.5 * h
    )
    return {
        "lane_area_frac": float(areas[LANE]),
        "mycar_temporal_iou": iou4,
        "signature_ok": ok,
    }


def stage0_image_baseline(labels: np.ndarray) -> dict:
    rows = []
    for j in range(labels.shape[0]):
        gt = labels[j]
        curves, _uncov = fit_lane_curves(gt)
        band = render_lane_band(curves, gt.shape)
        m = lane_band_metrics(band, gt == LANE)
        m["bytes_runs"] = lane_curve_bytes(curves, model="runs")
        m["bytes_solid"] = lane_curve_bytes(curves, model="solid")
        rows.append(m)
    return {
        "bytes_per_frame_runs": float(np.mean([r["bytes_runs"] for r in rows])),
        "bytes_per_frame_solid": float(np.mean([r["bytes_solid"] for r in rows])),
        "recall": float(np.mean([r["recall"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "iou": float(np.mean([r["iou"] for r in rows])),
    }


def stage1_ground_independent(per_frame, labels, lane_cls: int, bits: int) -> dict:
    quant, bpf = per_frame_quantized_ground(per_frame, bits=bits)
    recalls, precisions, ious = [], [], []
    h, w = labels.shape[1], labels.shape[2]
    for j, obs_list in enumerate(quant):
        lines = [
            lane_line_from_vec(
                o.vec, dash=(o.dash_period_m, o.dash_phase_m, o.dash_duty)
            )
            for o in obs_list
        ]
        band = rasterize_lane_band(lines, h=h, w=w, dash_gate=True)
        m = lane_band_metrics(band, labels[j] == lane_cls)
        recalls.append(m["recall"])
        precisions.append(m["precision"])
        ious.append(m["iou"])
    return {
        "bytes_per_frame": float(bpf),
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "iou": float(np.mean(ious)),
        "bits": bits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache",
        default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    )
    ap.add_argument("--stride", type=int, default=1, help="frame stride for the fit")
    ap.add_argument(
        "--eval-stride",
        type=int,
        default=1,
        help="stride (within fitted frames) for raster metric evaluation",
    )
    ap.add_argument(
        "--water-levels",
        default="1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2",
        help="comma list of SPD water levels (normalized-units MSE targets)",
    )
    ap.add_argument(
        "--occ-code-bins",
        default="105,45",
        help="comma list of near-field occupancy bin counts to code (rest solid)",
    )
    ap.add_argument(
        "--unit-scales",
        action="store_true",
        help="code with physical unit scales instead of pooled-std scales",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    cache = Path(args.cache)
    with np.load(cache) as z:
        labels = z["lstars"].astype(np.uint8)
        gt_poses = z["gt_poses"].astype(np.float64) if "gt_poses" in z.files else None
    frames = np.arange(0, labels.shape[0], args.stride)
    labels = labels[frames]
    print(f"loaded {cache} -> {labels.shape} (stride {args.stride})", flush=True)

    sig = _self_detect_lane(labels)
    print(f"class signature: {sig}", flush=True)
    if not sig["signature_ok"]:
        print("FATAL: class signature mismatch — refusing to proceed", flush=True)
        return 2

    eval_idx = np.arange(0, labels.shape[0], args.eval_stride)

    # ---- S0: image-space baseline (the 97 B/frame anchor) ----
    print("S0: image-space baseline ...", flush=True)
    s0 = stage0_image_baseline(labels[eval_idx])
    print(f"  S0 {s0}", flush=True)

    # ---- fit ground-frame lanes on every fitted frame ----
    print("fitting ground-frame lanes ...", flush=True)
    per_frame = [
        fit_frame_ground_lanes(labels[j], j, lane_cls=LANE, fit_dash=True)
        for j in range(labels.shape[0])
    ]
    n_lines = float(np.mean([len(o) for o in per_frame]))
    print(f"  mean lines/frame {n_lines:.2f}", flush=True)

    # ---- S1: per-frame independent ground coding ----
    print("S1: ground per-frame independent ...", flush=True)
    labels_eval = labels[eval_idx]
    s1 = stage1_ground_independent(
        [per_frame[j] for j in eval_idx], labels_eval, LANE, bits=12
    )
    print(f"  S1 {s1}", flush=True)

    # ---- track + measure anisotropy ----
    tracks = track_ground_lanes(per_frame)
    scales = (
        unit_dim_scales() if args.unit_scales else robust_dim_scales(tracks)
    )  # std scales MEASURED better on n96 (unit variant kept for the A/B record)
    d_h = pooled_hilbert_distance(tracks, robust_dim_scales(tracks))
    n_obs_total = int(sum(t.n_obs for t in tracks))
    n_obs_fitted = int(sum(len(o) for o in per_frame))
    coverage = n_obs_total / max(1, n_obs_fitted)
    print(
        f"tracks: {len(tracks)} tracks, obs coverage {coverage:.3f}, d_H {d_h:.2f}",
        flush=True,
    )

    # ---- world-aligned occupancy factor (dashes + visibility) ----
    n_frames = labels.shape[0]
    shifts = estimate_global_shifts(tracks, n_frames)
    mean_speed = float(shifts[-1] / max(1, n_frames - 1)) if n_frames > 1 else 0.0
    sbins = shifts_to_bins(shifts)
    shift_bytes = encode_shift_stream(sbins)
    occ_bin_settings = [int(x) for x in args.occ_code_bins.split(",")]
    print(
        f"world occupancy: total travel {shifts[-1]:.1f} m ({mean_speed:.2f} m/frame); "
        f"S(t) {shift_bytes} B standalone / 0 B xi-amortized",
        flush=True,
    )

    # ---- S2 (solid) / S3 (occ-gated): SPD-cone x occ-bins sweep ----
    sweep = []
    for wl in [float(x) for x in args.water_levels.split(",")]:
        enc = encode_tracks_spd(tracks, water_level=wl, scales=scales)
        m_solid = evaluate_tracks_raster(
            tracks, labels, eval_idx, lane_cls=LANE, occ_gate=False
        )
        row = {
            "water_level": wl,
            "bytes_total_solid": enc["total_bytes"],
            "bytes_per_frame_solid": enc["total_bytes"] / n_frames,
            "breakdown": enc,
            "solid": m_solid,
            "occ_gated": [],
        }
        print(
            f"  theta={wl:g}: solid {enc['total_bytes']} B "
            f"({enc['total_bytes'] / n_frames:.1f} B/f) r={m_solid['recall']:.3f} "
            f"p={m_solid['precision']:.3f} iou={m_solid['iou']:.3f}",
            flush=True,
        )
        # occ-gated variant at its own optimum: shape dims only (occupancy owns
        # visibility, so the forward-range dims are not coded — n_dims=6)
        enc6 = encode_tracks_spd(tracks, water_level=wl, scales=scales, n_dims=6)
        for nb in occ_bin_settings:
            occ_enc = encode_tracks_occupancy(tracks, sbins, code_bins=nb)
            m_occ = evaluate_tracks_raster(
                tracks, labels, eval_idx, lane_cls=LANE, occ_gate=True
            )
            occ_total = enc6["total_bytes"] + occ_enc["occ_bytes"]
            row["occ_gated"].append(
                {
                    "code_bins": nb,
                    "coeff6_bytes": enc6["total_bytes"],
                    "occ_bytes": occ_enc["occ_bytes"],
                    "bytes_total_xi_amortized": occ_total,
                    "bytes_total_standalone": occ_total + shift_bytes,
                    "bytes_per_frame_xi_amortized": occ_total / n_frames,
                    **m_occ,
                }
            )
            print(
                f"    occ[{nb} bins]: coeff6 {enc6['total_bytes']} B + occ "
                f"{occ_enc['occ_bytes']} B -> total {occ_total} B "
                f"({occ_total / n_frames:.1f} B/f) r={m_occ['recall']:.3f} "
                f"p={m_occ['precision']:.3f} iou={m_occ['iou']:.3f}",
                flush=True,
            )
        # restore full-dim decode for the next theta's solid eval
        encode_tracks_spd(tracks, water_level=wl, scales=scales)
        sweep.append(row)

    # ---- S4: the SKELETON arm — smoothed knots + fully-static world occupancy.
    # Drops the per-frame argmax jitter (the INR/annulus budget per v8 SPEC) and
    # measures what the smooth geometry alone costs + what fidelity it retains.
    from tac.boundary_math.lane_ground_factorization import (
        LaneTrack,
        build_static_world_occupancy,
        smooth_track_coeffs,
    )

    s4 = []
    for wl in [1e-5, 1e-4]:
        smooth_tracks = [
            LaneTrack(
                frames=tr.frames,
                coeffs=smooth_track_coeffs(tr.coeffs, window=9),
                dash_obs=tr.dash_obs,
                lat_ref=tr.lat_ref,
                occ=tr.occ,
            )
            for tr in tracks
        ]
        enc_s = encode_tracks_spd(smooth_tracks, water_level=wl, scales=scales, n_dims=6)
        occ_s = build_static_world_occupancy(smooth_tracks, sbins)
        m_s = evaluate_tracks_raster(
            smooth_tracks, labels, eval_idx, lane_cls=LANE, occ_gate=True
        )
        total_s = enc_s["total_bytes"] + occ_s["occ_bytes"]
        s4.append(
            {
                "water_level": wl,
                "coeff6_bytes": enc_s["total_bytes"],
                "occ_static_bytes": occ_s["occ_bytes"],
                "bytes_total_xi_amortized": total_s,
                "bytes_total_standalone": total_s + shift_bytes,
                "bytes_per_frame_xi_amortized": total_s / n_frames,
                **m_s,
            }
        )
        print(
            f"S4 skeleton theta={wl:g}: coeff6 {enc_s['total_bytes']} B + static-occ "
            f"{occ_s['occ_bytes']} B -> total {total_s} B ({total_s / n_frames:.1f} B/f) "
            f"r={m_s['recall']:.3f} p={m_s['precision']:.3f} iou={m_s['iou']:.3f}",
            flush=True,
        )

    # ---- xi correlation: the shared temporal mode vs the cached pose ----
    xi_corr = None
    if gt_poses is not None and tracks:
        xi_corr = {}
        # (a) the estimated shared forward travel dS(t) vs the cached pose dims —
        # the dual-use proof: if dS IS the ego xi, it rides the stored pose free.
        ds = np.diff(shifts)
        pose_at = gt_poses[frames[1 : 1 + ds.size]]
        for d in range(gt_poses.shape[1]):
            if np.std(pose_at[:, d]) > 0 and np.std(ds) > 0:
                xi_corr[f"dS_vs_pose_dim_{d}"] = float(
                    np.corrcoef(ds, pose_at[:, d])[0, 1]
                )
        # (b) the longest track's lateral drift vs pose (yaw/lateral coupling)
        longest = max(tracks, key=lambda t: t.n_obs)
        if longest.n_obs > 16:
            fr = frames[longest.frames]
            dlat = np.diff(longest.lat_ref)
            pose_l = gt_poses[fr[1:]]
            for d in range(gt_poses.shape[1]):
                if np.std(pose_l[:, d]) > 0:
                    xi_corr[f"dlat_vs_pose_dim_{d}"] = float(
                        np.corrcoef(dlat, pose_l[:, d])[0, 1]
                    )
        print(f"xi correlation: {xi_corr}", flush=True)

    out = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache": str(cache),
        "n_frames_fitted": int(n_frames),
        "n_frames_eval": int(eval_idx.size),
        "stride": args.stride,
        "signature": sig,
        "mean_lines_per_frame": n_lines,
        "s0_image_baseline": s0,
        "s1_ground_independent": s1,
        "tracking": {
            "n_tracks": len(tracks),
            "obs_coverage": coverage,
            "d_H": d_h,
            "dim_scales": scales.tolist(),
        },
        "world_occupancy": {
            "total_travel_m": float(shifts[-1]),
            "mean_travel_m_per_frame": mean_speed,
            "shift_bytes_standalone": shift_bytes,
            "shift_bytes_xi_amortized": 0,
        },
        "s2_s3_spd_sweep": sweep,
        "s4_skeleton": s4,
        "xi_correlation": xi_corr,
        "authority": "[macOS-MLX advisory] cached-argmax geometric estimate; NOT byte-closed",
        "elapsed_s": time.time() - t0,
    }
    if args.out:
        od = Path(args.out)
        od.mkdir(parents=True, exist_ok=True)
        (od / "probe.json").write_text(json.dumps(out, indent=2))
        print(f"wrote {od / 'probe.json'}", flush=True)
    print(f"done in {out['elapsed_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
