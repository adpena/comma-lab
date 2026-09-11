# SPDX-License-Identifier: MIT
"""Band-local pricing of the ddm_rbf1 post-render boundary operators.

`experiments/ddm_rbf1_boundary_treatments.treat` evaluates each operator over the
whole camera frame and then keeps only the token-edge band, so the cost it
measures is the frame, not the band.  MEASURED on the shipped move-44 token field
the band is 2.21% of the camera frame, so a shipped receiver would pay far less
than the research render did.  This module evaluates the identical arithmetic at
the band only and proves byte-identity against the full-frame producer on real
retained frames, so the wall-clock a receiver change would actually pay can be
priced against the strict T4 slack of 27.581 s measured by ddm_mxo1
(233.809 ns per coded symbol at n600).

Nothing here is a score.  No scorer, no learned table, no fitted amplitude and no
video-selected constant: every literal is pixel geometry or a stencil size
inherited unchanged from the producer module.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]

from experiments.ddm_rbf1_boundary_treatments import edges, sample, treat

N, H, W, CH, CW = 600, 384, 512, 874, 1164
MODES = ("guided", "ssaa", "sdf", "composition")
T4_OVER_LOCAL = 0.978071  # ddm_mxo1 same-receiver projection factor, not a T4 run
STRICT_T4_SLACK_S = 27.581274745  # ddm_mxo1: 1260 s policy ceiling - 1232.418725255 s


def _clamped(index: np.ndarray, delta: int, size: int) -> np.ndarray:
    """Replicate-padding index arithmetic; identical to np.pad(mode='edge')."""
    return np.clip(index + delta, 0, size - 1)


def axis_map(camera: int, token: int) -> np.ndarray:
    """The receiver's align_corners=False nearest-token index for one axis."""
    coordinate = np.arange(camera, dtype=np.float32)
    projected = (coordinate + np.float32(.5)) * np.float32(token / camera) - np.float32(.5)
    return np.clip(np.floor(projected + np.float32(.5)).astype(np.int64), 0, token - 1)


def band_of(tokens: np.ndarray, shape: tuple[int, int]):
    """Camera-plane band indices, their token cell and their token coordinates."""
    height, width = shape
    token_height, token_width = tokens.shape
    row_map = axis_map(height, token_height)
    col_map = axis_map(width, token_width)
    mask = edges(tokens)[row_map][:, col_map]
    flat = np.flatnonzero(mask.ravel())
    by, bx = np.divmod(flat, width)
    guide = tokens[row_map[by], col_map[bx]]
    ty = (by.astype(np.float32) + np.float32(.5)) * np.float32(token_height / height) - np.float32(.5)
    tx = (bx.astype(np.float32) + np.float32(.5)) * np.float32(token_width / width) - np.float32(.5)
    return by, bx, guide, ty, tx


def guided_band(rgb: np.ndarray, tokens: np.ndarray, by, bx, guide) -> np.ndarray:
    """One-hot guided regression at band pixels; producer's fp32 operation order."""
    height, width = rgb.shape[:2]
    token_height, token_width = tokens.shape
    row_map = axis_map(height, token_height)
    col_map = axis_map(width, token_width)
    values = rgb.astype(np.float32)
    out = np.zeros((by.size, 3), dtype=np.float32)
    for label in np.unique(guide):
        selected = guide == label
        cy, cx = by[selected], bx[selected]
        outer = np.zeros((cy.size, 3), dtype=np.float32)
        for a in (-1, 0, 1):
            qy = _clamped(cy, a, height)
            for b in (-1, 0, 1):
                qx = _clamped(cx, b, width)
                numerator = np.zeros((cy.size, 3), dtype=np.float32)
                count = np.zeros(cy.size, dtype=np.float32)
                for c in (-1, 0, 1):
                    sy = _clamped(qy, c, height)
                    for d in (-1, 0, 1):
                        sx = _clamped(qx, d, width)
                        member = (tokens[row_map[sy], col_map[sx]] == label).astype(np.float32)
                        numerator += values[sy, sx] * member[:, None]
                        count += member
                numerator /= np.float32(9)
                count /= np.float32(9)
                numerator /= np.maximum(count, np.float32(1 / 9))[:, None]
                outer += numerator
        outer /= np.float32(9)
        out[selected] = outer
    return out


def ssaa_band(rgb: np.ndarray, by, bx) -> np.ndarray:
    """Four centred quarter-pixel samples of the rendered bilinear reconstruction."""
    yy = by.astype(np.float32)
    xx = bx.astype(np.float32)
    out = np.zeros((by.size, 3), dtype=np.float32)
    for oy in (-.25, .25):
        for ox in (-.25, .25):
            out += sample(rgb, yy + np.float32(oy), xx + np.float32(ox))
    out *= np.float32(.25)
    return out


def _phi_at(tokens: np.ndarray, label: int, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    """Producer's truncated signed Euclidean distance evaluated at listed cells."""
    height, width = tokens.shape
    centre = tokens[sy, sx] == label
    distance = np.full(sy.shape, 3, dtype=np.float32)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if not 0 < dy * dy + dx * dx < 9:
                continue
            other = tokens[_clamped(sy, dy, height), _clamped(sx, dx, width)] == label
            np.minimum(distance, np.where(other != centre, np.float32(np.sqrt(dy * dy + dx * dx)), np.float32(3)), out=distance)
    return np.where(centre, distance, -distance).astype(np.float32)


def displacement_at(tokens: np.ndarray, cy: np.ndarray, cx: np.ndarray):
    """Producer's curvature-compensated displacement evaluated at listed cells."""
    height, width = tokens.shape
    labels = tokens[cy, cx]
    dy_out = np.zeros(cy.shape, dtype=np.float32)
    dx_out = np.zeros(cy.shape, dtype=np.float32)
    for label in np.unique(labels):
        selected = labels == label
        y, x = cy[selected], cx[selected]
        here = _phi_at(tokens, label, y, x)
        above = _phi_at(tokens, label, _clamped(y, -1, height), x)
        below = _phi_at(tokens, label, _clamped(y, 1, height), x)
        left = _phi_at(tokens, label, y, _clamped(x, -1, width))
        right = _phi_at(tokens, label, y, _clamped(x, 1, width))
        gy = (below - above) * np.float32(.5)
        gx = (right - left) * np.float32(.5)
        laplacian = below + above + right + left - np.float32(4) * here
        amount = laplacian / (np.float32(24) * np.maximum(gx * gx + gy * gy, np.float32(1)))
        dy_out[selected] = np.clip(amount * gy, -.5, .5)
        dx_out[selected] = np.clip(amount * gx, -.5, .5)
    return dy_out, dx_out


def sdf_band(rgb: np.ndarray, tokens: np.ndarray, by, bx, ty, tx) -> np.ndarray:
    """Token-SDF displacement of the camera sampling position at band pixels."""
    token_height, token_width = tokens.shape
    cy = np.clip(ty, 0, token_height - 1).astype(np.float32)
    cx = np.clip(tx, 0, token_width - 1).astype(np.float32)
    y0 = np.floor(cy).astype(np.int32)
    x0 = np.floor(cx).astype(np.int32)
    y1 = np.minimum(y0 + 1, token_height - 1)
    x1 = np.minimum(x0 + 1, token_width - 1)
    fy = cy - y0.astype(np.float32)
    fx = cx - x0.astype(np.float32)
    corners = np.stack((y0 * token_width + x0, y0 * token_width + x1,
                        y1 * token_width + x0, y1 * token_width + x1))
    unique, inverse = np.unique(corners, return_inverse=True)
    plane_dy, plane_dx = displacement_at(tokens, unique // token_width, unique % token_width)
    inverse = inverse.reshape(4, -1)
    shifted = []
    for plane in (plane_dy, plane_dx):
        gathered = plane[inverse]
        top = gathered[0] * (np.float32(1) - fx)
        top += gathered[1] * fx
        bottom = gathered[2] * (np.float32(1) - fx)
        bottom += gathered[3] * fx
        shifted.append(top * (np.float32(1) - fy) + bottom * fy)
    scale_y = np.float32(rgb.shape[0] / token_height)
    scale_x = np.float32(rgb.shape[1] / token_width)
    return sample(rgb, by.astype(np.float32) + shifted[0] * scale_y,
                  bx.astype(np.float32) + shifted[1] * scale_x)


def treat_band(rgb: np.ndarray, tokens: np.ndarray, mode: str) -> np.ndarray:
    """Band-local equivalent of the producer's treat(); identical output bytes."""
    if mode == "composition":
        out = rgb
        for component in ("guided", "ssaa", "sdf"):
            out = treat_band(out, tokens, component)
        return out
    by, bx, guide, ty, tx = band_of(tokens, rgb.shape[:2])
    if mode == "guided":
        filtered = guided_band(rgb, tokens, by, bx, guide)
    elif mode == "ssaa":
        filtered = ssaa_band(rgb, by, bx)
    elif mode == "sdf":
        filtered = sdf_band(rgb, tokens, by, bx, ty, tx)
    else:
        raise ValueError(f"unknown boundary treatment: {mode}")
    out = rgb.copy()
    out[by, bx] = np.clip(np.rint(filtered), 0, 255).astype(np.uint8)
    return out


def _frames(chunks: Path, tokens, pairs: list[int]):
    for pair in pairs:
        start = (pair // 5) * 5
        directory = chunks / f"{start:03d}_{start + 5:03d}"
        camera = np.load(directory / "baseline_camera_u8.npy", mmap_mode="r", allow_pickle=False)
        yield pair, np.array(camera[pair - start, 1]), np.asarray(tokens[pair])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "price"))
    parser.add_argument("--root", default="/Volumes/VertigoDataTier/pact/ddm_rbf1/retained")
    parser.add_argument("--field", default="/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    tokens = np.memmap(args.field, mode="r", dtype=np.uint8, shape=(N, H, W))
    available = sorted(int(p.name.split("_")[0]) for p in (root / "chunks").glob("*_*") if (p / "baseline_camera_u8.npy").is_file())
    rng = np.random.default_rng(args.seed)
    pairs = sorted(int(rng.choice(available)) + int(rng.integers(0, 5)) for _ in range(args.pairs))
    rows = []
    for pair, rgb, token in _frames(root / "chunks", tokens, pairs):
        by, bx, _, _, _ = band_of(token, rgb.shape[:2])
        row = {"pair": pair, "band_pixels": int(by.size), "band_fraction": float(by.size / (CH * CW))}
        for mode in MODES:
            if args.action == "verify":
                reference = treat(rgb, token, mode)
                candidate = treat_band(rgb, token, mode)
                row[mode + "_identical"] = bool(np.array_equal(reference, candidate))
                row[mode + "_max_abs_diff"] = int(np.abs(reference.astype(np.int32) - candidate.astype(np.int32)).max())
                row[mode + "_changed_channels"] = int(np.count_nonzero(reference != rgb))
            else:
                tic = time.perf_counter()
                treat(rgb, token, mode)
                full = time.perf_counter() - tic
                tic = time.perf_counter()
                treat_band(rgb, token, mode)
                band = time.perf_counter() - tic
                row[mode + "_full_frame_s"] = full
                row[mode + "_band_local_s"] = band
                row[mode + "_speedup"] = full / band
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"schema": "ddm_rbf1_band_local.v1", "axis": "[macOS-CPU advisory]", "score_claim": False,
               "action": args.action, "pairs": pairs, "host": platform.platform(), "numpy": np.__version__,
               "threads_env": "inherited", "rows": rows,
               "strict_t4_slack_s": STRICT_T4_SLACK_S, "t4_over_local": T4_OVER_LOCAL}
    if args.action == "price":
        for mode in MODES:
            band = float(np.mean([r[mode + "_band_local_s"] for r in rows]))
            full = float(np.mean([r[mode + "_full_frame_s"] for r in rows]))
            summary[mode] = {"band_local_mean_s_per_frame": band, "full_frame_mean_s_per_frame": full,
                             "band_local_n600_s": band * N, "full_frame_n600_s": full * N,
                             "band_local_projected_t4_s": band * N * T4_OVER_LOCAL,
                             "fits_strict_t4_slack": bool(band * N * T4_OVER_LOCAL <= STRICT_T4_SLACK_S)}
    else:
        summary["all_identical"] = all(r[m + "_identical"] for r in rows for m in MODES)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
