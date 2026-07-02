#!/usr/bin/env python
"""$0 n600 verify of the AA-SDF observation-map render (the MEASURED #1 representation lever).

[macOS-CPU advisory] NON-PROMOTABLE. Frozen CPU-torch SegNet authority, NEVER MPS. Realized
through the ACTUAL contest R (``_torch_R_to_camera_uint8`` = render-grid -> bicubic^ to camera
874 -> round/uint8, then ``SegNet.preprocess_input`` bilinear 874->384 -> argmax) -- the SAME
authority path as ``tools/levelset_gate_discriminators_n600.py`` (so numbers are directly
comparable to the committed gate), but exercising THIS campaign's
``tac.boundary_math.aa_sdf_observation_render.box_downsample_np`` footprint integrator across a
grid curve.

It answers the operator brief's $0 verify: n600 d_seg + LANE recall with AA ON vs OFF.

  * SIGNAL A = REAL FRAME (confound-free ACHIEVABLE-THROUGH-R upper bound; a witness render is a
    strict subset of "any RGB at the render grid", so this bounds what the witness can reach).
  * SIGNAL B = softmax-of-SDF PARTITION PROXY (per-class mean-RGB palette; the witness's actual
    output STRUCTURE). CONFOUNDED (a flat palette cannot fully represent the lane class -> low
    absolute recall) -- reported for the RELATIVE AA lift on the partition structure only.

For each render grid G in --grids, two renders through R:
  * POINT  = nearest (point) resample of the signal to G  (aliasing: thin lanes hit-or-miss).
  * AA     = supersample to (ss*G) then ``box_downsample_np`` (ss) to G  (footprint-integrated).

Resumable per-pair checkpoint (atomic tmp+rename). Reproduces the gate's +0.38 lane recall /
oracle-R floor toward 0.00091, and MAPS the --render-grid curve so the trainer knows where to
render. ~0 rate (decode-time deterministic op; witness bytes unchanged).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

# Reuse the committed gate's authority-path helpers (identical R + SegNet argmax + metrics).
from tools.levelset_gate_discriminators_n600 import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    _torch_R_to_camera_uint8,
    area_down,
    d_seg_of,
    lane_recall,
    mean_lut,
    nearest_down,
    nn_labels,
    seg_argmax_batch,
)
from tac.boundary_math.aa_sdf_observation_render import box_downsample_np  # noqa: E402
from tac.boundary_math.seg_core import load_real_segnet  # noqa: E402

CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
CKPT = REPO / "reports/aa_sdf_observation_render_verify_n600.ckpt.npz"
OUT = REPO / "reports/aa_sdf_observation_render_verify_n600_20260701.json"
NUM_PAIRS = int(os.environ.get("AA_NUM_PAIRS", "600"))
VBATCH = int(os.environ.get("AA_VBATCH", "12"))
SS = int(os.environ.get("AA_SS", "2"))
GRIDS = [int(g) for g in os.environ.get("AA_GRIDS", "192,256,384,512").split(",")]
SAVE_EVERY = 48


def _grid_hw(g: int) -> tuple[int, int]:
    """Render grid (h, w) at aspect 3:4 (matches the trainer's 384x512 / 192x256)."""
    return g, (g * 4) // 3


def _cond_names() -> list[str]:
    names = []
    for g in GRIDS:
        for sig in ("real", "part"):
            for mode in ("point", "aa"):
                names.append(f"{sig}_g{g}_{mode}")
    return names


def build_render(sig: str, mode: str, g: int, f1_cam: np.ndarray, lstar: np.ndarray,
                 lut: np.ndarray) -> np.ndarray:
    """Camera-res uint8 (874,1164,3) for signal/mode/grid, footprint-integrated via MY primitive."""
    h, w = _grid_hw(g)
    if sig == "real":
        if mode == "point":
            rg = nearest_down(f1_cam, h, w)
        else:  # supersample to (ss*g) [area from 874 detail] -> box_downsample(ss) -> g
            fine = area_down(f1_cam, h * SS, w * SS)
            rg = box_downsample_np(fine[None].astype(np.float64), SS)[0]
    elif sig == "part":
        if mode == "point":
            rg = lut[nn_labels(lstar, h, w)].astype(np.float64)
        else:  # partition palette rendered at (ss*g) -> box_downsample(ss) -> g
            fine = lut[nn_labels(lstar, h * SS, w * SS)].astype(np.float64)
            rg = box_downsample_np(fine[None], SS)[0]
    else:
        raise ValueError(sig)
    return _torch_R_to_camera_uint8(rg.clip(0, 255))


def main() -> None:
    t0 = time.time()
    seg = load_real_segnet("cpu")
    print(f"[{time.time() - t0:.1f}s] SegNet loaded. SS={SS} grids={GRIDS}", flush=True)
    z = np.load(CACHE, allow_pickle=False)
    gt_f1_all = z["gt_f1"]
    lstars_all = z["lstars"]
    P = min(NUM_PAIRS, int(z["n_pairs"]))
    conds = _cond_names()
    print(f"[{time.time() - t0:.1f}s] cache P={P}, {len(conds)} conditions", flush=True)

    dseg = {c: np.full(P, np.nan) for c in conds}
    rec = {c: np.full(P, np.nan) for c in conds}
    done = np.zeros(P, bool)
    if CKPT.exists():
        ck = np.load(CKPT, allow_pickle=False)
        for c in conds:
            if c in ck.files:
                dseg[c] = ck[c]
            if c + "__recall" in ck.files:
                rec[c] = ck[c + "__recall"]
        if "done" in ck.files:
            done = ck["done"].astype(bool)
        print(f"[resume] {int(done.sum())}/{P} done", flush=True)

    def save():
        payload = {c: dseg[c] for c in conds}
        payload.update({c + "__recall": rec[c] for c in conds})
        payload["done"] = done
        tmp = CKPT.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(CKPT)

    todo = [i for i in range(P) if not done[i]]
    last = int(done.sum())
    for s in range(0, len(todo), VBATCH):
        chunk = todo[s:s + VBATCH]
        f1s = [np.asarray(gt_f1_all[i], np.uint8) for i in chunk]
        lst = [np.asarray(lstars_all[i], np.int64) for i in chunk]
        luts = [mean_lut(f1s[j], lst[j]) for j in range(len(chunk))]
        for g in GRIDS:
            for sig in ("real", "part"):
                for mode in ("point", "aa"):
                    c = f"{sig}_g{g}_{mode}"
                    cams = [build_render(sig, mode, g, f1s[j], lst[j], luts[j])
                            for j in range(len(chunk))]
                    realized = seg_argmax_batch(seg, cams)
                    for j, i in enumerate(chunk):
                        dseg[c][i] = d_seg_of(realized[j], lst[j])
                        rec[c][i] = lane_recall(realized[j], lst[j])
        for i in chunk:
            done[i] = True
        nd = int(done.sum())
        if nd - last >= SAVE_EVERY or nd == P:
            save()
            last = nd
            g0 = GRIDS[0]
            print(f"[{time.time() - t0:.1f}s] {nd}/{P} | "
                  f"real_g{g0} pt/aa d_seg={np.nanmean(dseg[f'real_g{g0}_point']):.5f}/"
                  f"{np.nanmean(dseg[f'real_g{g0}_aa']):.5f} "
                  f"recall={np.nanmean(rec[f'real_g{g0}_point']):.3f}/"
                  f"{np.nanmean(rec[f'real_g{g0}_aa']):.3f}", flush=True)
    save()
    n = int(done.sum())

    def stat(a):
        a = np.asarray(a); a = a[~np.isnan(a)]
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "n": int(a.size)}

    curve = {}
    for g in GRIDS:
        curve[str(g)] = {
            "real": {
                "point": {"d_seg": stat(dseg[f"real_g{g}_point"]),
                          "lane_recall": stat(rec[f"real_g{g}_point"])},
                "aa": {"d_seg": stat(dseg[f"real_g{g}_aa"]),
                       "lane_recall": stat(rec[f"real_g{g}_aa"])},
                "aa_recall_lift": float(np.nanmean(rec[f"real_g{g}_aa"]) - np.nanmean(rec[f"real_g{g}_point"])),
                "aa_dseg_gain": float(np.nanmean(dseg[f"real_g{g}_point"]) - np.nanmean(dseg[f"real_g{g}_aa"])),
            },
            "partition_proxy_CONFOUNDED": {
                "point": {"d_seg": stat(dseg[f"part_g{g}_point"]),
                          "lane_recall": stat(rec[f"part_g{g}_point"])},
                "aa": {"d_seg": stat(dseg[f"part_g{g}_aa"]),
                       "lane_recall": stat(rec[f"part_g{g}_aa"])},
                "aa_recall_lift": float(np.nanmean(rec[f"part_g{g}_aa"]) - np.nanmean(rec[f"part_g{g}_point"])),
            },
        }

    g_ref = 192 if 192 in GRIDS else GRIDS[0]
    out = {
        "axis_tag": "[macOS-CPU advisory] NON-PROMOTABLE",
        "n_pairs": n,
        "supersample_ss": SS,
        "primitive": "tac.boundary_math.aa_sdf_observation_render.box_downsample_np",
        "R_path": "_torch_R_to_camera_uint8 (render->bicubic^874->uint8) + SegNet.preprocess_input "
                  "bilinear 874->384 -> argmax (frozen CPU-torch authority; identical to the "
                  "committed levelset_gate_discriminators_n600 gate)",
        "render_grid_curve": curve,
        "verdict": (
            f"AA (supersample->box, ss={SS}) vs POINT on the confound-free REAL-frame "
            f"achievable-through-R signal: at grid {g_ref} lane-recall lift "
            f"{np.nanmean(rec[f'real_g{g_ref}_aa']) - np.nanmean(rec[f'real_g{g_ref}_point']):+.4f}, "
            f"d_seg {np.nanmean(dseg[f'real_g{g_ref}_point']):.5f} (point) -> "
            f"{np.nanmean(dseg[f'real_g{g_ref}_aa']):.5f} (AA). The render-grid curve maps the "
            "achievable-through-R floor as G -> camera; AA recovers finest-scale lanes WITHIN a "
            "fixed grid. Both are ~0-rate decode-time levers (witness bytes unchanged)."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[{time.time() - t0:.1f}s] DONE n={n}. Wrote {OUT.name}", flush=True)
    for g in GRIDS:
        r = curve[str(g)]["real"]
        print(f"  REAL g{g:4d}: d_seg point={r['point']['d_seg']['mean']:.5f} aa={r['aa']['d_seg']['mean']:.5f} "
              f"| recall point={r['point']['lane_recall']['mean']:.3f} aa={r['aa']['lane_recall']['mean']:.3f} "
              f"(lift {r['aa_recall_lift']:+.3f})", flush=True)


if __name__ == "__main__":
    main()
