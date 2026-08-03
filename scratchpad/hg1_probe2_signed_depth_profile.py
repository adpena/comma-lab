# SPDX-License-Identifier: MIT
"""ddm_hg1 PROBE 2 -- the SIGNED DEPTH PROFILE of the margin field across each DIRECTED interface.

The operator's pin: "the asymmetry of the depth on either side of the separatrix".
This measures it directly, and it is VEHICLE-INDEPENDENT (frozen-scorer GT only, no witness,
no reconstruction, no theta) -- so unlike the 2026-07-08 witness-vehicle q1 artifact it
transfers to cx1/TR1 without the borrowed-number caveat.

CONSTRUCTION (per frame, n600, exact):
  * For every class c, EDT_c(p) = Euclidean distance from p to the nearest pixel of class c.
  * rival(p) = argmin over c != L(p) of EDT_c(p)      -- the geometrically nearest RIVAL class.
  * depth(p) = EDT_{rival(p)}(p)                       -- how deep p sits inside its own class,
                                                          measured toward the rival it would
                                                          flip to.
  * The DIRECTED population (i->j) = { p : L(p)=i, rival(p)=j }.  Its depth profile is
    margin(depth) binned at integer depth.  The (i,j) separatrix has TWO profiles, one per
    side, and the addendum's claim is that they differ.

WHY THIS IS THE RIGHT OBJECT.  The margin is the frozen head's potential; the separatrix is
its zero set; d(margin)/d(depth) on each side is the normal derivative of that potential.
A SYMMETRIC potential well would give matched profiles and a symmetric band/annulus loss
would be the correct shape.  Asymmetric profiles mean the cheapest direction for the
separatrix to move is the shallow side -- which is a statement about where the errors MUST
concentrate, independent of any particular witness.

FALSIFIABLE PREDICTION registered before looking at the answer:
  P1  the directed flip rate (i->j) measured on the witness surface should be RANK-predicted
      by the i-side profile shallowness (lower margin at small depth  =>  higher flip rate).
      REFUTED if Spearman over the 8 major directed sides is |rs| < 0.5.
  P2  Lane is the only class whose mean thickness is O(1) px, so the Lane sides should be the
      profile outliers.  REFUTED if Lane's profile sits inside the spread of the others.

$0, scorer-free, cached artifacts only.  [macOS-numpy advisory . NON-PROMOTABLE]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
NC = 5
MAXD = 12  # profile reported for integer depth 1..MAXD


def main() -> int:
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    z = np.load(REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz", mmap_mode="r")
    lstars, margins = z["lstars"], z["margins"]
    assert lstars.shape[0] >= n_frames, lstars.shape

    # running sums per directed pair per integer depth bin
    s_n = np.zeros((NC, NC, MAXD + 1), dtype=np.int64)
    s_m = np.zeros((NC, NC, MAXD + 1), dtype=np.float64)
    s_mm = np.zeros((NC, NC, MAXD + 1), dtype=np.float64)
    pair_n = np.zeros((NC, NC), dtype=np.int64)      # full directed population (all depths)
    pair_m = np.zeros((NC, NC), dtype=np.float64)
    cls_area = np.zeros(NC, dtype=np.int64)
    cls_shell = np.zeros(NC, dtype=np.int64)          # pixels at depth <= 1 (the erodible shell)

    for f in range(n_frames):
        L = np.asarray(lstars[f], dtype=np.int8)
        M = np.asarray(margins[f], dtype=np.float64)
        edt = np.empty((NC, *L.shape), dtype=np.float32)
        present = []
        for c in range(NC):
            mc = c == L
            if not mc.any():
                edt[c] = np.inf
                continue
            present.append(c)
            edt[c] = distance_transform_edt(~mc)
        # nearest rival: mask out own class then argmin
        e = edt.copy()
        for c in range(NC):
            e[c][c == L] = np.inf
        rival = np.argmin(e, axis=0).astype(np.int8)
        depth = np.take_along_axis(e, rival[None].astype(np.intp), axis=0)[0]

        for c in present:
            mc = c == L
            cls_area[c] += int(mc.sum())
            cls_shell[c] += int((mc & (depth <= 1.0)).sum())
            for r in present:
                if r == c:
                    continue
                pop = mc & (rival == r)
                if not pop.any():
                    continue
                d = np.clip(np.rint(depth[pop]), 0, MAXD).astype(np.int64)
                mv = M[pop]
                pair_n[c, r] += d.size
                pair_m[c, r] += float(mv.sum())
                np.add.at(s_n[c, r], d, 1)
                np.add.at(s_m[c, r], d, mv)
                np.add.at(s_mm[c, r], d, mv * mv)
        if (f + 1) % 100 == 0:
            print(f"  ...{f + 1}/{n_frames}", flush=True)

    out: dict = {
        "probe": "hg1_probe2_signed_depth_profile",
        "source": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz (lstars + margins)",
        "n_frames": n_frames,
        "vehicle_independent": True,
        "advisory": "[macOS-numpy advisory . NON-PROMOTABLE] pointer 0.1910828242 UNMOVED",
        "class_geometry": {},
        "directed_profiles": {},
    }
    for c in range(NC):
        if cls_area[c] == 0:
            continue
        out["class_geometry"][CLASS_NAMES[c]] = {
            "area_px": int(cls_area[c]),
            "shell_px_depth_le_1": int(cls_shell[c]),
            "shell_fraction": float(cls_shell[c] / cls_area[c]),
            "mean_depth": float((s_m[c].sum() * 0 + sum(
                k * s_n[c, :, k].sum() for k in range(MAXD + 1))) / max(1, s_n[c].sum())),
        }
    for i in range(NC):
        for j in range(NC):
            if pair_n[i, j] == 0:
                continue
            prof = []
            for k in range(1, MAXD + 1):
                n = int(s_n[i, j, k])
                if n == 0:
                    prof.append(None)
                    continue
                mean = s_m[i, j, k] / n
                var = max(0.0, s_mm[i, j, k] / n - mean * mean)
                prof.append({"depth": k, "n": n, "mean_margin": float(mean),
                             "sd_margin": float(np.sqrt(var))})
            out["directed_profiles"][f"{i}->{j}"] = {
                "from": CLASS_NAMES[i], "to": CLASS_NAMES[j],
                "n_pixels": int(pair_n[i, j]),
                "mean_margin_all_depths": float(pair_m[i, j] / pair_n[i, j]),
                "profile": prof,
            }

    dst = REPO / f".omx/research/ddm_hg1_signed_depth_profile_n{n_frames}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"[done] {dst}")

    print("\n=== class geometry (shell = px at depth<=1, i.e. one flip from the separatrix) ===")
    for k, v in out["class_geometry"].items():
        print(f"  {k:>12}: area={v['area_px']:>10}  shell={v['shell_px_depth_le_1']:>10} "
              f"({v['shell_fraction']*100:6.2f}% of the class)  mean_depth={v['mean_depth']:.3f}")
    print("\n=== directed margin depth profiles (mean margin at integer depth) ===")
    hdr = "  " + " ".join(f"d{k:<7}" for k in range(1, 7))
    print(f"  {'side':>20} {'n_px':>10}  " + hdr)
    for k, v in sorted(out["directed_profiles"].items(),
                       key=lambda kv: -kv[1]["n_pixels"]):
        if v["n_pixels"] < 20000:
            continue
        cells = []
        for e in v["profile"][:6]:
            cells.append(f"{e['mean_margin']:7.3f}" if e else "      -")
        print(f"  {k+' '+v['from'][:4]+'->'+v['to'][:4]:>20} {v['n_pixels']:>10}   " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
