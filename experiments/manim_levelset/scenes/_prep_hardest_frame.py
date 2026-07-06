"""Prep step for Scene 2 — find the HARDEST real contest frame and export
lightweight render assets from the cached SegNet outputs.

Heavy data (the ~2 GB gt_n600.npz) is touched ONCE here; the Manim scene then
renders from the small PNGs/NPYs this writes (memory-safe, reproducible).

Everything is REAL contest data (NO-FAKE):
  gt_f1   (600,874,1164,3) uint8 — the actual frame-1 of each scored pair
  lstars  (600,384,512)     int  — the frozen SegNet ARGMAX (0..4) on that frame
  margins (600,384,512)   float   — the top1−top2 logit gap (the flip-margin)

"Hardest" = the frame whose d_seg is intrinsically most at risk: the most
SMALL-MARGIN boundary pixels, weighted toward the fragile LANE class (class 1) —
which is exactly our measured erasure-prone long-tail (deepmath / lane-dash
findings). Fully documented, not a vibe.

Run:
    cd experiments/manim_levelset
    ../../.venv/bin/python scenes/_prep_hardest_frame.py
(uses the MAIN repo venv for numpy/scipy; writes to ./assets/)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ASSETS = _HERE.parent / "assets"
_CACHE = _HERE.parents[2] / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

# comma10k canonical class order + REAL mask colors (openpilot look; from
# src/tac/categorical_candidate_runtime_skeleton.py — NON-NEGOTIABLE):
# 0 Road #402020, 1 Lane #ff0000, 2 Undrivable #808060, 3 Movable #00ff66, 4 MyCar #cc00ff
CLASS_RGB = np.array(
    [
        [ 64,  32,  32],   # 0 road       — comma10k maroon
        [255,   0,   0],   # 1 lane       — comma10k red (fragile rare class)
        [128, 128,  96],   # 2 undrivable — comma10k olive
        [  0, 255, 102],   # 3 movable    — comma10k green
        [204,   0, 255],   # 4 my-car     — comma10k purple
    ],
    dtype=np.uint8,
)
_LANE = 1


def _boundary_mask(lab: np.ndarray) -> np.ndarray:
    """4-neighbour argmax-disagreement = the separatrix (codim-1)."""
    e = np.zeros_like(lab, dtype=bool)
    e[:-1, :] |= lab[:-1, :] != lab[1:, :]
    e[1:, :] |= lab[:-1, :] != lab[1:, :]
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    e[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return e


def main() -> None:
    _ASSETS.mkdir(exist_ok=True)
    if not _CACHE.exists():
        raise SystemExit(f"cache not found: {_CACHE}")

    z = np.load(_CACHE)
    lstars = np.asarray(z["lstars"])      # (600,384,512)
    margins = np.asarray(z["margins"]).astype(np.float32)
    n = lstars.shape[0]

    # small-margin threshold = global 12th percentile of boundary margins
    # (the flip-prone tail). Difficulty = small-margin boundary mass, lane-weighted.
    diff = np.zeros(n, dtype=np.float64)
    thr = np.percentile(margins, 12.0)
    for i in range(n):
        b = _boundary_mask(lstars[i])
        small = b & (margins[i] < thr)
        lane_small = small & (lstars[i] == _LANE)
        diff[i] = small.sum() + 3.0 * lane_small.sum()

    hardest = int(diff.argmax())
    order = np.argsort(-diff)
    print(f"hardest frame = {hardest}  (difficulty {diff[hardest]:.0f})")
    print("top-5 hardest:", [(int(k), int(diff[k])) for k in order[:5]])

    lab = lstars[hardest]                                  # (384,512)
    marg = margins[hardest]
    per_class = {int(c): int((lab == c).sum()) for c in range(5)}

    # ── colorized argmax partition ───────────────────────────────────────────
    argmax_rgb = CLASS_RGB[lab]                            # (384,512,3)

    # ── separatrix overlay on the argmax (bright cyan boundary) ──────────────
    edge = _boundary_mask(lab)
    sep_rgb = argmax_rgb.copy()
    sep_rgb[edge] = np.array([90, 240, 255], np.uint8)

    # ── margin field: dark interior → bright boundary annulus (Fisher metric) ─
    m = marg.copy()
    m = np.clip(m / (np.percentile(m, 99) + 1e-6), 0, 1)   # normalize
    inv = 1.0 - m                                          # small margin = bright
    # perceptual dark-navy → cyan → white ramp
    ramp = (np.stack([0.10 + 0.35 * inv, 0.12 + 0.80 * inv, 0.20 + 0.80 * inv], -1))
    margin_rgb = np.clip(ramp * 255, 0, 255).astype(np.uint8)

    # ── the real RGB frame (downsize to argmax aspect for clean overlay) ─────
    frame = np.asarray(z["gt_f1"][hardest])                # (874,1164,3) uint8
    # resize to 1024 wide keeping aspect (~1.333) via simple stride for speed
    fh, fw = frame.shape[:2]
    tw = 1024
    th = int(round(tw * fh / fw))
    ys = (np.linspace(0, fh - 1, th)).astype(int)
    xs = (np.linspace(0, fw - 1, tw)).astype(int)
    frame_small = frame[np.ix_(ys, xs)]

    # ── montage: 72 evenly-spaced frames, low-res, for the fast-forward scrub ─
    idxs = np.linspace(0, n - 1, 72).astype(int)
    mh, mw = 168, 224
    mont = np.empty((len(idxs), mh, mw, 3), np.uint8)
    gt = z["gt_f1"]                                        # loads full stack once
    yy = np.linspace(0, gt.shape[1] - 1, mh).astype(int)
    xx = np.linspace(0, gt.shape[2] - 1, mw).astype(int)
    for j, fi in enumerate(idxs):
        mont[j] = gt[fi][np.ix_(yy, xx)]

    # ── ego clip: a CONTIGUOUS run of real frames (the actual ego-motion the
    #    screw twists produce), + their twists ξ = the 6 PoseNet scalars. This
    #    renders "the screw-derived pose motion itself on the contest video".
    poses = np.asarray(z["gt_poses"])                     # (600,6) — the twists ξ
    K = 42
    start = int(np.clip(hardest - K // 2, 0, n - K))
    ci = np.arange(start, start + K)
    ch, cw = 336, 448
    cyy = np.linspace(0, gt.shape[1] - 1, ch).astype(int)
    cxx = np.linspace(0, gt.shape[2] - 1, cw).astype(int)
    ego = np.empty((K, ch, cw, 3), np.uint8)
    for j, fi in enumerate(ci):
        ego[j] = gt[fi][np.ix_(cyy, cxx)]
    np.save(_ASSETS / "ego_clip.npy", ego)
    np.save(_ASSETS / "ego_poses.npy", poses[ci].astype(np.float32))
    np.save(_ASSETS / "ego_idx.npy", ci.astype(int))

    # ── write assets (PNG via manim-independent tiny writer / npy) ───────────
    from PIL import Image
    Image.fromarray(frame_small).save(_ASSETS / "hardest_frame.png")
    Image.fromarray(argmax_rgb).save(_ASSETS / "hardest_argmax.png")
    Image.fromarray(sep_rgb).save(_ASSETS / "hardest_separatrix.png")
    Image.fromarray(margin_rgb).save(_ASSETS / "hardest_margin.png")
    np.save(_ASSETS / "montage.npy", mont)
    np.save(_ASSETS / "montage_idx.npy", idxs)

    meta = {
        "hardest_frame": hardest,
        "difficulty": float(diff[hardest]),
        "top5": [int(k) for k in order[:5]],
        "small_margin_thr": float(thr),
        "per_class_px": per_class,
        "lane_px": per_class[_LANE],
        "boundary_px": int(edge.sum()),
        "cache": str(_CACHE),
        "n_frames": int(n),
    }
    (_ASSETS / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote assets to", _ASSETS)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
