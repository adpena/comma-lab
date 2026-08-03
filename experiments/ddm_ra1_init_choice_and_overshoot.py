"""ddm_ra1 — where the realization debt is BORN, and whether the init is a free lever.

$0, NO SCORER.  NON-PROMOTABLE.

TWO QUESTIONS, BOTH SCORER-FREE
-------------------------------
Q1. The live camera raster is ``clip(rint(U_bicubic(r)))``.  PyTorch bicubic uses
    A=-0.75, i.e. NEGATIVE side lobes, so ``U(r)`` OVERSHOOTS past [0,255] wherever
    ``r`` has a strong edge against a saturated region (our render is
    ``sigmoid*255``, so it saturates constantly).  ``clip`` then destroys exactly
    the overshoot that the reconstruction needed.  How much of the debt is this?

Q2. ``solve_camera_windows`` hits the target from ANY starting raster, so the init
    is a free per-window null-space choice with ZERO counted bytes.  ll1 tested one
    alternative init (r-broadcast) against the PLANE residual and rejected it.  But
    the plane residual is ~0.03 LSB for every init -- the solve makes it exact.
    The quantity that actually differs between inits is the one nobody measured:
    how far the camera raster ends up from a natural image, which is what the v4d
    frame_0 warp reads.  So: rank inits by FRAME_0 WARP DELTA, not by plane.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512


def _upsample_bilinear(r: np.ndarray) -> np.ndarray:
    """Non-overshooting upsample (convex weights): 384->874, align_corners=False."""

    def taps(src: int, dst: int) -> tuple[np.ndarray, np.ndarray]:
        centers = np.clip((np.arange(dst, dtype=np.float64) + 0.5) * (src / dst) - 0.5, 0.0, None)
        i0 = np.floor(centers).astype(np.int64)
        frac = centers - i0
        idx = np.stack([np.clip(i0, 0, src - 1), np.clip(i0 + 1, 0, src - 1)], axis=1)
        return idx, np.stack([1.0 - frac, frac], axis=1)

    ys, wy = taps(SEG_H, CAMERA_H)
    xs, wx = taps(SEG_W, CAMERA_W)
    src = np.asarray(r, dtype=np.float64)
    out = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.float64)
    for a in range(2):
        for b in range(2):
            w = wy[:, a][:, None] * wx[:, b][None, :]
            out += w[..., None] * src[np.ix_(ys[:, a], xs[:, b])]
    return out


def _upsample_nearest(r: np.ndarray) -> np.ndarray:
    ys = np.clip(((np.arange(CAMERA_H) + 0.5) * SEG_H / CAMERA_H).astype(np.int64), 0, SEG_H - 1)
    xs = np.clip(((np.arange(CAMERA_W) + 0.5) * SEG_W / CAMERA_W).astype(np.int64), 0, SEG_W - 1)
    return np.asarray(r, dtype=np.float64)[np.ix_(ys, xs)]


def _stats(delta: np.ndarray) -> dict[str, float]:
    flat = np.abs(delta).ravel()
    return {
        "rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "p99": float(np.percentile(flat, 99.0)),
        "max_abs": float(flat.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sub = Path(args.submission_dir).resolve()
    sys.path.insert(0, str(sub))
    from ddm_tr1_runtime import bicubic_up_to_camera_float, render_frame1_float  # noqa: PLC0415
    from inflate_runner import Decoder  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tac.optimization.ddm_ll1_window_solve import solve_camera_windows  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ddm_ra1_realization_debt_on_live_vehicle import down_to_scorer  # noqa: PLC0415

    dec = Decoder(sub / "archive")
    n_pairs = int(dec.n_pairs)
    stride = max(1, n_pairs // args.pairs)
    sampled = list(range(0, n_pairs, stride))[: args.pairs]

    rows = []
    for i in sampled:
        r = np.asarray(render_frame1_float(dec.packet, i), dtype=np.float64)
        up_bic = bicubic_up_to_camera_float(r).astype(np.float64)

        # --- Q1: how much of the raster is clipped, and how much debt does it own? ---
        over = (up_bic > 255.0) | (up_bic < 0.0)
        clipped_frac = float(over.mean())
        cam_bic = np.clip(np.rint(up_bic), 0, 255).astype(np.uint8)
        d_base = down_to_scorer(cam_bic) - r
        # debt attributable to clipping = debt with clip vs debt of pure rounding
        # (rounding-only reference is not realizable, it is a decomposition probe)
        d_round_only = down_to_scorer(np.rint(up_bic)) - r
        row: dict[str, object] = {
            "pair": i,
            "clipped_camera_frac": clipped_frac,
            "render_saturated_frac": float(((r > 254.5) | (r < 0.5)).mean()),
            "debt_with_clip": _stats(d_base),
            "debt_rounding_only_unrealizable": _stats(d_round_only),
        }

        # --- Q2: rank inits by FRAME_0 warp delta after an exact solve ---
        f0_ref = dec.f0(i, cam_bic)
        inits = {
            "bicubic_clip": cam_bic,
            "bilinear": np.clip(np.rint(_upsample_bilinear(r)), 0, 255).astype(np.uint8),
            "nearest": np.clip(np.rint(_upsample_nearest(r)), 0, 255).astype(np.uint8),
        }
        for name, cam0 in inits.items():
            cam_s = solve_camera_windows(cam0, r)
            move = cam_s.astype(np.int16) - cam_bic.astype(np.int16)
            df0 = dec.f0(i, cam_s).astype(np.float64) - f0_ref.astype(np.float64)
            row[f"init_{name}"] = {
                "plane_after_solve": _stats(down_to_scorer(cam_s) - r),
                "camera_move_vs_shipped": _stats(move.astype(np.float64)),
                "frame0_delta_vs_shipped": _stats(df0),
            }
        rows.append(row)
        print(
            f"pair {i:3d} clip {clipped_frac:.3%} sat {row['render_saturated_frac']:.3%}"
            f" | debt clip {row['debt_with_clip']['rms']:.4f} vs round-only "  # type: ignore[index]
            f"{row['debt_rounding_only_unrealizable']['rms']:.4f}"  # type: ignore[index]
            + "".join(
                f" | {n} f0 {row[f'init_{n}']['frame0_delta_vs_shipped']['rms']:.4f}"  # type: ignore[index]
                f"/{row[f'init_{n}']['frame0_delta_vs_shipped']['max_abs']:.0f}"  # type: ignore[index]
                for n in inits
            ),
            flush=True,
        )

    summary = {
        "arm": "ddm_ra1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "pairs_sampled": sampled,
        "clipped_camera_frac_mean": float(np.mean([r["clipped_camera_frac"] for r in rows])),
        "render_saturated_frac_mean": float(np.mean([r["render_saturated_frac"] for r in rows])),
        "debt_with_clip_rms": float(np.mean([r["debt_with_clip"]["rms"] for r in rows])),  # type: ignore[index]
        "debt_rounding_only_rms": float(
            np.mean([r["debt_rounding_only_unrealizable"]["rms"] for r in rows])  # type: ignore[index]
        ),
        "inits": {
            n: {
                "plane_after_solve_rms": float(
                    np.mean([r[f"init_{n}"]["plane_after_solve"]["rms"] for r in rows])  # type: ignore[index]
                ),
                "camera_move_rms": float(
                    np.mean([r[f"init_{n}"]["camera_move_vs_shipped"]["rms"] for r in rows])  # type: ignore[index]
                ),
                "frame0_delta_rms": float(
                    np.mean([r[f"init_{n}"]["frame0_delta_vs_shipped"]["rms"] for r in rows])  # type: ignore[index]
                ),
                "frame0_delta_max": float(
                    np.mean([r[f"init_{n}"]["frame0_delta_vs_shipped"]["max_abs"] for r in rows])  # type: ignore[index]
                ),
            }
            for n in ("bicubic_clip", "bilinear", "nearest")
        },
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
