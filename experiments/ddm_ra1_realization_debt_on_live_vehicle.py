"""ddm_ra1 — measure the camera-raster realization debt on the LIVE shipped vehicle.

$0, NO SCORER.  Scorer-free plane-space measurement of the crossing

    render r (384,512,3) float  --U(bicubic)-->  (874,1164,3) float
                                --clip(rint)-->  camera uint8
                                --D(bilinear, antialias=False)-->  (384,512,3)

``D`` is what the frozen scorers actually read (``SegNet.preprocess_input`` /
``PoseNet.preprocess_input`` both interpolate to (384,512) first).  The gap
``D(cam) - r`` is the REALIZATION debt: the part of our own description that the
uint8 camera lattice fails to deliver.  ``ddm_ll1.solve_camera_windows`` closes it
at zero counted bytes.

Everything here is read off the EXACT shipped archive bytes and the EXACT vendored
receiver in the submission directory.  Nothing is re-implemented.

Pairs are sampled STRIDED across all 600 (never a prefix): a contiguous prefix of a
temporally-correlated video is a scene block, not a sample (memory ``m88``).

NON-PROMOTABLE: this measures input-space perturbation, not d_seg.  Flip counts
require the frozen scorer and the held n600 slot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512


def _bilinear_taps(src: int, dst: int) -> tuple[np.ndarray, np.ndarray]:
    """torch ``F.interpolate(..., 'bilinear', align_corners=False)`` taps, downsampling."""
    centers = np.clip((np.arange(dst, dtype=np.float64) + 0.5) * (src / dst) - 0.5, 0.0, None)
    i0 = np.floor(centers).astype(np.int64)
    frac = centers - i0
    idx = np.stack([np.clip(i0, 0, src - 1), np.clip(i0 + 1, 0, src - 1)], axis=1)
    return idx, np.stack([1.0 - frac, frac], axis=1)


def down_to_scorer(camera: np.ndarray) -> np.ndarray:
    """D: the operator both frozen scorers read through. (874,1164,3) -> (384,512,3)."""
    ys, wy = _bilinear_taps(CAMERA_H, SEG_H)
    xs, wx = _bilinear_taps(CAMERA_W, SEG_W)
    src = np.asarray(camera, dtype=np.float64)
    out = np.zeros((SEG_H, SEG_W, 3), dtype=np.float64)
    for a in range(2):
        for b in range(2):
            w = wy[:, a][:, None] * wx[:, b][None, :]
            out += w[..., None] * src[np.ix_(ys[:, a], xs[:, b])]
    return out


def _stats(delta: np.ndarray) -> dict[str, float]:
    flat = np.abs(delta).ravel()
    return {
        "rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "mean_abs": float(flat.mean()),
        "p99": float(np.percentile(flat, 99.0)),
        "max_abs": float(flat.max()),
        "frac_over_half_lsb": float((flat > 0.5).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--pairs", type=int, default=16, help="how many STRIDED pairs to sample")
    ap.add_argument("--pose-coupling", action="store_true", help="also price the frame_0 warp delta")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sub = Path(args.submission_dir).resolve()
    sys.path.insert(0, str(sub))
    from ddm_tr1_runtime import (  # noqa: PLC0415
        bicubic_up_to_camera_float,
        render_frame1_float,
    )
    from inflate_runner import Decoder  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tac.optimization.ddm_ll1_window_solve import (  # noqa: PLC0415
        blind_mask,
        solve_camera_windows,
    )

    dec = Decoder(sub / "archive")
    n_pairs = int(dec.n_pairs)
    stride = max(1, n_pairs // args.pairs)
    sampled = list(range(0, n_pairs, stride))[: args.pairs]

    blind = blind_mask()
    rows: list[dict[str, object]] = []

    for i in sampled:
        r = np.asarray(render_frame1_float(dec.packet, i), dtype=np.float64)
        up = bicubic_up_to_camera_float(r)
        cam = np.ascontiguousarray(np.clip(np.rint(up), 0, 255).astype(np.uint8))
        cam_s = solve_camera_windows(cam, r)

        d_base = down_to_scorer(cam) - r
        d_solv = down_to_scorer(cam_s) - r

        moved = cam_s.astype(np.int16) - cam.astype(np.int16)
        moved_any = np.any(moved != 0, axis=2)
        blind_touched = int(np.count_nonzero(moved_any & blind))

        row: dict[str, object] = {
            "pair": i,
            "baseline": _stats(d_base),
            "solved": _stats(d_solv),
            "camera_px_moved": int(np.count_nonzero(moved_any)),
            "camera_px_moved_frac": float(np.count_nonzero(moved_any) / (CAMERA_H * CAMERA_W)),
            "blind_px_touched": blind_touched,
            "moved_max_lsb": int(np.abs(moved).max()),
            "render_grad_rms": float(
                np.sqrt(np.mean(np.diff(r, axis=0) ** 2) + np.mean(np.diff(r, axis=1) ** 2))
            ),
        }

        if args.pose_coupling:
            f0_base = dec.f0(i, cam)
            f0_solv = dec.f0(i, cam_s)
            df0 = f0_solv.astype(np.float64) - f0_base.astype(np.float64)
            row["frame0_warp_delta"] = _stats(df0)
            row["frame0_px_changed_frac"] = float(np.any(df0 != 0, axis=2).mean())

        rows.append(row)
        print(
            f"pair {i:3d}  base rms {row['baseline']['rms']:.5f} max {row['baseline']['max_abs']:.3f}"  # type: ignore[index]
            f"  ->  solved rms {row['solved']['rms']:.5f} max {row['solved']['max_abs']:.3f}"  # type: ignore[index]
            f"  moved {row['camera_px_moved_frac']:.3%} blind {blind_touched}",
            flush=True,
        )

    def agg(key: str, sub_key: str) -> float:
        return float(np.mean([r[key][sub_key] for r in rows]))  # type: ignore[index]

    summary = {
        "arm": "ddm_ra1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "submission_dir": str(sub),
        "n_pairs_total": n_pairs,
        "pairs_sampled": sampled,
        "sampling": "STRIDED across all pairs (not a prefix; memory m88)",
        "baseline_rms_mean": agg("baseline", "rms"),
        "solved_rms_mean": agg("solved", "rms"),
        "baseline_max_mean": agg("baseline", "max_abs"),
        "solved_max_mean": agg("solved", "max_abs"),
        "reduction_factor_rms": agg("baseline", "rms") / max(agg("solved", "rms"), 1e-12),
        "baseline_frac_over_half_lsb": agg("baseline", "frac_over_half_lsb"),
        "solved_frac_over_half_lsb": agg("solved", "frac_over_half_lsb"),
        "camera_px_moved_frac_mean": float(np.mean([r["camera_px_moved_frac"] for r in rows])),
        "blind_px_touched_total": int(sum(int(r["blind_px_touched"]) for r in rows)),  # type: ignore[arg-type]
        "rows": rows,
    }
    if args.pose_coupling:
        summary["frame0_warp_delta_rms_mean"] = agg("frame0_warp_delta", "rms")
        summary["frame0_warp_delta_max_mean"] = agg("frame0_warp_delta", "max_abs")
        summary["frame0_px_changed_frac_mean"] = float(
            np.mean([r["frame0_px_changed_frac"] for r in rows])
        )

    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
