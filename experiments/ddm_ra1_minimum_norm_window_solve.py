"""ddm_ra1 — SPREAD, don't CONCENTRATE: a minimum-norm variant of the ll1 window solve.

$0, NO SCORER.  Scorer-free.  NON-PROMOTABLE.

THE PROBLEM THIS ATTACKS (MEASURED by this arm, live ``v4d_dc1_fold`` bytes)
---------------------------------------------------------------------------
``ddm_ll1.solve_camera_windows`` closes the camera-raster realization debt almost
perfectly for reader #1 (``D``, the scorer downsample): plane residual rms
0.7994 -> 0.0298 LSB, 26.8x, and by construction it never writes a blind pixel.

But frame_1's camera raster has a SECOND reader on the v4d vehicle: the frame_0
homography warp (``inflate_runner.Decoder.f0``), which resamples ACROSS the private
2x2 windows.  The solve's intra-window redistribution is fully visible to it:

    frame_0 warp delta   rms 2.75 LSB   max 178 LSB   58.8% of pixels changed

So the solve is optimal for reader #1 and indifferent to reader #2.

THE STRUCTURE THAT MAKES A CURE POSSIBLE
----------------------------------------
Each private window carries FOUR integer degrees of freedom and ONE linear
constraint ``sum_k w_k c_k = r``.  That leaves a 3-dimensional per-window null
space that ll1 spends arbitrarily (weight-descending greedy, which DUMPS the whole
correction on the single highest-weight tap -- hence the 178 LSB tail).

CLAUDE.md's own Fridrich rule says the opposite is right:
    "Square root law: spread small errors (L-inf penalty), don't concentrate
     large ones."

So: allocate the SAME correction by minimum norm instead.  Continuous minimum-L2
solution of ``w . s = err`` is ``s = w * err / ||w||^2`` -- the correction spread
over all four taps in proportion to their weights.  Round, then close the integer
residual with a single weight-descending pass (which now has almost nothing left
to do).  Identical constraint, identical zero counted bytes, identical blind-set
invariance; only the null-space choice differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
_QUAD = ((0, 0), (0, 1), (1, 0), (1, 1))


def solve_camera_windows_min_norm(
    camera_uint8: np.ndarray,
    target_plane: np.ndarray,
    *,
    passes: int = 1,
) -> np.ndarray:
    """Minimum-norm variant: spread the correction over all four taps."""
    from tac.optimization.ddm_ll1_window_solve import window_geometry

    ys, xs, wy, wx = window_geometry()
    target = np.asarray(target_plane, dtype=np.float64)

    c = [camera_uint8[np.ix_(ys[:, a], xs[:, b])].astype(np.float64) for a, b in _QUAD]
    w = [(wy[:, a][:, None] * wx[:, b][None, :])[..., None] for a, b in _QUAD]
    wsq = sum(np.broadcast_to(w[k], target.shape) ** 2 for k in range(4))

    err = target - sum(w[k] * c[k] for k in range(4))

    # --- pass A: minimum-L2 spread over all four taps, rounded, clip-aware ---
    for k in range(4):
        wk = np.broadcast_to(w[k], err.shape)
        step = np.rint(err * wk / wsq)
        step = np.clip(c[k] + step, 0.0, 255.0) - c[k]
        c[k] = c[k] + step
        err = err - wk * step

    # --- pass B: close the integer residual, weight-descending (little left) ---
    order = np.argsort(
        -np.concatenate([np.broadcast_to(w[k], err.shape)[..., None] for k in range(4)], axis=-1),
        axis=-1,
    )
    for _ in range(max(1, passes)):
        for slot in range(4):
            chosen = order[..., slot]
            for k in range(4):
                mask = chosen == k
                if not mask.any():
                    continue
                wk = np.broadcast_to(w[k], err.shape)
                step = np.where(
                    mask,
                    np.rint(np.divide(err, wk, out=np.zeros_like(err), where=wk > 1e-12)),
                    0.0,
                )
                step = np.clip(c[k] + step, 0.0, 255.0) - c[k]
                c[k] = c[k] + step
                err = err - wk * step

    out = camera_uint8.astype(np.int16).copy()
    for k, (a, b) in enumerate(_QUAD):
        out[ys[:, a][:, None], xs[:, b][None, :], :] = np.clip(c[k], 0, 255).astype(np.int16)
    return np.ascontiguousarray(out.astype(np.uint8))


def _stats(delta: np.ndarray) -> dict[str, float]:
    flat = np.abs(delta).ravel()
    return {
        "rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "p99": float(np.percentile(flat, 99.0)),
        "max_abs": float(flat.max()),
        "frac_over_half_lsb": float((flat > 0.5).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sub = Path(args.submission_dir).resolve()
    sys.path.insert(0, str(sub))
    from ddm_tr1_runtime import bicubic_up_to_camera_float, render_frame1_float  # noqa: PLC0415
    from inflate_runner import Decoder  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tac.optimization.ddm_ll1_window_solve import (  # noqa: PLC0415
        blind_mask,
        solve_camera_windows,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ddm_ra1_realization_debt_on_live_vehicle import down_to_scorer  # noqa: PLC0415

    dec = Decoder(sub / "archive")
    n_pairs = int(dec.n_pairs)
    stride = max(1, n_pairs // args.pairs)
    sampled = list(range(0, n_pairs, stride))[: args.pairs]
    blind = blind_mask()

    rows = []
    for i in sampled:
        r = np.asarray(render_frame1_float(dec.packet, i), dtype=np.float64)
        cam = np.ascontiguousarray(
            np.clip(np.rint(bicubic_up_to_camera_float(r)), 0, 255).astype(np.uint8)
        )
        variants = {
            "greedy_ll1": solve_camera_windows(cam, r),
            "min_norm": solve_camera_windows_min_norm(cam, r),
        }
        f0_base = dec.f0(i, cam)
        row: dict[str, object] = {"pair": i, "baseline_plane": _stats(down_to_scorer(cam) - r)}
        for name, cam_v in variants.items():
            moved = cam_v.astype(np.int16) - cam.astype(np.int16)
            df0 = dec.f0(i, cam_v).astype(np.float64) - f0_base.astype(np.float64)
            row[name] = {
                "plane": _stats(down_to_scorer(cam_v) - r),
                "camera_move": _stats(moved.astype(np.float64)),
                "camera_px_moved_frac": float(np.any(moved != 0, axis=2).mean()),
                "blind_px_touched": int(np.count_nonzero(np.any(moved != 0, axis=2) & blind)),
                "frame0_warp_delta": _stats(df0),
                "frame0_px_changed_frac": float(np.any(df0 != 0, axis=2).mean()),
            }
        rows.append(row)
        g, m = row["greedy_ll1"], row["min_norm"]  # type: ignore[index]
        print(
            f"pair {i:3d} | plane rms  greedy {g['plane']['rms']:.5f}  min_norm {m['plane']['rms']:.5f}"
            f" | cam max  greedy {g['camera_move']['max_abs']:.0f}  min_norm {m['camera_move']['max_abs']:.0f}"
            f" | f0 rms  greedy {g['frame0_warp_delta']['rms']:.4f}  min_norm {m['frame0_warp_delta']['rms']:.4f}",
            flush=True,
        )

    def agg(variant: str, group: str, key: str) -> float:
        return float(np.mean([r[variant][group][key] for r in rows]))  # type: ignore[index]

    summary = {
        "arm": "ddm_ra1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pairs_sampled": sampled,
        "sampling": "STRIDED across all pairs (memory m88)",
        "baseline_plane_rms": float(np.mean([r["baseline_plane"]["rms"] for r in rows])),  # type: ignore[index]
        "variants": {
            v: {
                "plane_rms": agg(v, "plane", "rms"),
                "plane_max": agg(v, "plane", "max_abs"),
                "plane_frac_over_half_lsb": agg(v, "plane", "frac_over_half_lsb"),
                "camera_move_rms": agg(v, "camera_move", "rms"),
                "camera_move_max": agg(v, "camera_move", "max_abs"),
                "camera_move_p99": agg(v, "camera_move", "p99"),
                "frame0_rms": agg(v, "frame0_warp_delta", "rms"),
                "frame0_max": agg(v, "frame0_warp_delta", "max_abs"),
                "frame0_px_changed_frac": float(
                    np.mean([r[v]["frame0_px_changed_frac"] for r in rows])  # type: ignore[index]
                ),
                "blind_px_touched": int(sum(int(r[v]["blind_px_touched"]) for r in rows)),  # type: ignore[index]
            }
            for v in ("greedy_ll1", "min_norm")
        },
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
