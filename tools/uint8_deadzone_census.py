# SPDX-License-Identifier: MIT
"""uint8 deadzone census (per-class-pair, per-range) on cached fields -- the M1 bound's deciding measurement.

LAW + BAND PROVENANCE (T5 crucible, v5 draft S0.0a M1 + S0.0b precision note + S7c "P-DZ"; deriving
doc ct_deepresearch_2 S13 M1):
  uint8 quantization at camera resolution makes flips REQUIRING a through-R intensity change
  smaller than one quantum (1/255, i.e. 1.0 in 0-255 units) unreachable by any smooth witness
  without dither/phase tricks (#149 class). CT-2's derived per-pixel form:
      deadzone(x)  <=>  |H_R| * g_I(x) * dx_req(x) < 1.0    (0-255 units)
      dx_req(x) = m(x) / |grad m(x)|      (boundary displacement needed, px)
      g_I(x)    = |grad I(x)|             (luma edge contrast at seg-res, 0-255 per px)
      |H_R|     = 0.842 (measured through-R transfer lower bound; the CENSUS-MAXIMIZING =
                  conservative-for-DEFER choice; the |H_R| = 1.0 variant is also reported).
  Estimator form limitation (stated per requirement R / provenance discipline): dx_req uses the
  cached GT margin geometry (witness margins are not cached); this is the CT-2 formula's own
  form, not a witness-exact requirement.

  Pre-registered thresholds (v5 S0.0b, corrected units -- CENSUS, no kill):
    < 5.34e-6 d_seg-equivalent (0.3x crossing margin)  ==> M1 NOT binding, #149 stays DEFER;
    > 1.78e-5 d_seg-equivalent (1x crossing margin)    ==> #149 enters the duty queue.
  SCOPE (requirement H/L + requirement R): the claim is that the deadzone binds only on
  LOW-contrast boundaries (far-range lane, shadow edges) -- every "not binding" conclusion is
  scoped per-class-pair per-range (row band), never globally.

  Side artifact (SC-16 seed data): edge-contrast g_I histograms per class-pair over the GT
  boundary annulus, saved as npz.

ARTIFACT SCHEMA:
  JSON: { "schema": "uint8_deadzone_census.v1", "totals": { "flip_px", "total_px",
          "deadzone_flip_px_HR", "dseg_equivalent_HR", "deadzone_flip_px_H1",
          "dseg_equivalent_H1" }, "per_pair": { "<i>-><j>": { n_flips, n_deadzone,
          dseg_equivalent, frac_of_pair_flips, per_row_band: {...} } }, "thresholds": {...} }
  NPZ (SC-16): per major pair, g_I histogram (128 bins over [0,256]) on the boundary annulus.

USAGE
  .venv/bin/python tools/uint8_deadzone_census.py \
      --out experiments/results/t5_probe_waveB_20260708/pdz_deadzone_census.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEG_HW = (384, 512)
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
THRESH_DEFER = 5.34e-6   # d_seg-equivalent, 0.3x crossing margin (v5 S0.0b)
THRESH_BIND = 1.78e-5    # d_seg-equivalent, 1.0x crossing margin (v5 S0.0b)
HR_DEFAULT = 0.842
ROW_BANDS = [(0, 64), (64, 128), (128, 176), (176, 224), (224, 288), (288, 384)]


def _now_utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")):
        raise ValueError(f"{path!r} is a /tmp-class durable path; use the repo tier per CLAUDE.md.")


def l1_grad(field: np.ndarray) -> np.ndarray:
    dy = np.abs(np.diff(field, axis=0))
    dy = np.pad(dy, ((0, 1), (0, 0)))
    dx = np.abs(np.diff(field, axis=1))
    dx = np.pad(dx, ((0, 0), (0, 1)))
    return dy + dx


def downsample_luma(gt_f1_uint8: np.ndarray) -> np.ndarray:
    """gt_f1 (874,1164,3) uint8 -> luma (384,512) float64 [0,255], bilinear a_c=False (torch, CPU)."""
    import torch

    t = torch.from_numpy(np.asarray(gt_f1_uint8)).float().permute(2, 0, 1)[None]
    d = torch.nn.functional.interpolate(t, size=SEG_HW, mode="bilinear", align_corners=False)
    return d[0].mean(dim=0).numpy().astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--witness-dir", default="experiments/results/witness_per_stage_attribution")
    ap.add_argument("--witness-stage", default="MuonBest")
    ap.add_argument("--witness-to-gt-stride", type=int, default=2)
    ap.add_argument("--n-frames", type=int, default=0, help="0 = all witness frames")
    ap.add_argument("--hr", type=float, default=HR_DEFAULT, help="|H_R| through-R transfer bound")
    ap.add_argument("--quantum", type=float, default=1.0, help="uint8 quantum in 0-255 units")
    ap.add_argument("--annulus-radius", type=int, default=2, help="chebyshev radius for the g_I histogram annulus")
    ap.add_argument("--grad-eps", type=float, default=1e-6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = (REPO / args.out) if not os.path.isabs(args.out) else Path(args.out)
    _refuse_tmp(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from scipy.ndimage import binary_dilation

    z = np.load(REPO / args.gt_cache, mmap_mode="r")
    lstars, margins, gt_f1 = z["lstars"], z["margins"], z["gt_f1"]
    wdir = REPO / args.witness_dir
    wz = np.load(wdir / f"maps_{args.witness_stage}.npz")
    w_argmax = np.asarray(wz["argmax"])
    gt_arg_sub = np.load(wdir / "_gt_argmax_subset.npy", mmap_mode="r")
    n_w = w_argmax.shape[0] if args.n_frames <= 0 else min(args.n_frames, w_argmax.shape[0])

    struct = np.ones((2 * args.annulus_radius + 1,) * 2, dtype=bool)
    hist_bins = np.linspace(0.0, 256.0, 129)
    gi_hists: dict[tuple[int, int], np.ndarray] = {}
    per_pair: dict[tuple[int, int], dict] = {}
    tot_flip = 0
    tot_dead_hr = 0
    tot_dead_h1 = 0
    tot_px = 0

    for w in range(n_w):
        s_idx = args.witness_to_gt_stride * w
        gta = np.asarray(lstars[s_idx], dtype=np.int64)
        if not np.array_equal(gta.astype(np.int8), np.asarray(gt_arg_sub[w])):
            raise RuntimeError(f"frame alignment broken at witness frame {w} (gt idx {s_idx})")
        wa = w_argmax[w].astype(np.int64)
        m = np.asarray(margins[s_idx], dtype=np.float64)
        lum = downsample_luma(gt_f1[s_idx])
        g_i = l1_grad(lum)
        gm = l1_grad(m)
        dx_req = m / (gm + args.grad_eps)
        di_hr = args.hr * g_i * dx_req
        di_h1 = 1.0 * g_i * dx_req
        flip = wa != gta
        dead_hr = flip & (di_hr < args.quantum)
        dead_h1 = flip & (di_h1 < args.quantum)
        tot_flip += int(flip.sum())
        tot_dead_hr += int(dead_hr.sum())
        tot_dead_h1 += int(dead_h1.sum())
        tot_px += int(flip.size)

        rows_idx = np.arange(SEG_HW[0])[:, None] * np.ones((1, SEG_HW[1]), dtype=int)
        for ci, cj in {(int(a), int(b)) for a, b in zip(gta[flip].ravel(), wa[flip].ravel(), strict=True)}:
            sel = flip & (gta == ci) & (wa == cj)
            key = (ci, cj)
            if key not in per_pair:
                per_pair[key] = {"n_flips": 0, "n_deadzone_HR": 0, "n_deadzone_H1": 0,
                                 "row_bands": {f"{a}-{b}": {"n_flips": 0, "n_deadzone_HR": 0}
                                               for a, b in ROW_BANDS}}
            p = per_pair[key]
            p["n_flips"] += int(sel.sum())
            p["n_deadzone_HR"] += int((sel & dead_hr).sum())
            p["n_deadzone_H1"] += int((sel & dead_h1).sum())
            for a, b in ROW_BANDS:
                band = (rows_idx >= a) & (rows_idx < b)
                p["row_bands"][f"{a}-{b}"]["n_flips"] += int((sel & band).sum())
                p["row_bands"][f"{a}-{b}"]["n_deadzone_HR"] += int((sel & band & dead_hr).sum())
        # SC-16: g_I histograms per class-pair over the GT boundary annulus (both sides of the edge)
        for ci in np.unique(gta):
            m_i = gta == ci
            for cj in np.unique(gta):
                if cj <= ci:
                    continue
                ann = (m_i & binary_dilation(gta == cj, structure=struct)) | \
                      ((gta == cj) & binary_dilation(m_i, structure=struct))
                if not ann.any():
                    continue
                h, _ = np.histogram(g_i[ann], bins=hist_bins)
                key = (int(ci), int(cj))
                gi_hists[key] = gi_hists.get(key, np.zeros(128, np.int64)) + h.astype(np.int64)

    per_pair_out = {}
    for (ci, cj), p in sorted(per_pair.items()):
        dseg_eq = p["n_deadzone_HR"] / tot_px
        per_pair_out[f"{ci}->{cj}"] = {
            "class_from": CLASS_NAMES.get(ci, str(ci)),
            "class_to": CLASS_NAMES.get(cj, str(cj)),
            "n_flips": p["n_flips"],
            "n_deadzone_HR": p["n_deadzone_HR"],
            "n_deadzone_H1": p["n_deadzone_H1"],
            "dseg_equivalent_HR": dseg_eq,
            "frac_of_pair_flips_deadzone_HR": (p["n_deadzone_HR"] / p["n_flips"]) if p["n_flips"] else float("nan"),
            "row_bands": p["row_bands"],
        }

    npz_path = out.with_suffix(".gi_hists.npz")
    np.savez_compressed(
        npz_path,
        bins=hist_bins,
        **{f"gi_hist_{ci}_{cj}": h for (ci, cj), h in gi_hists.items()},
    )

    dseg_hr = tot_dead_hr / tot_px
    dseg_h1 = tot_dead_h1 / tot_px
    result = {
        "schema": "uint8_deadzone_census.v1",
        "generated_utc": _now_utc(),
        "argv": sys.argv[1:],
        "inputs": {"gt_cache": args.gt_cache, "witness_dir": args.witness_dir,
                   "witness_stage": args.witness_stage},
        "n_frames": n_w,
        "estimator": "deadzone <=> HR * g_I * m/|grad m| < quantum; GT-margin-geometry form "
                     "(CT-2 S13 M1); witness margins not cached -- stated form limitation",
        "hr": args.hr,
        "quantum_0_255": args.quantum,
        "totals": {
            "total_px": tot_px,
            "flip_px": tot_flip,
            "flip_dseg": tot_flip / tot_px,
            "deadzone_flip_px_HR": tot_dead_hr,
            "dseg_equivalent_HR": dseg_hr,
            "deadzone_flip_px_H1": tot_dead_h1,
            "dseg_equivalent_H1": dseg_h1,
            "deadzone_frac_of_flips_HR": tot_dead_hr / tot_flip if tot_flip else float("nan"),
        },
        "thresholds": {
            "defer_lt_dseg": THRESH_DEFER,
            "bind_gt_dseg": THRESH_BIND,
            "provenance": "v5 S0.0a M1 + S0.0b + S7c P-DZ",
            "disposition_HR": ("#149 stays DEFER" if dseg_hr < THRESH_DEFER
                               else "#149 enters duty queue" if dseg_hr > THRESH_BIND
                               else "BETWEEN bands (census; no kill)"),
        },
        "per_pair": per_pair_out,
        "sc16_gi_histograms_npz": str(npz_path.relative_to(REPO)),
        "scope_note": "per-class-pair per-range only; no global not-binding claim (req H/L/R)",
        "advisory": "[macOS-numpy advisory . NON-PROMOTABLE] pointer 0.19110 UNMOVED",
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, out)
    print(f"[done] {out}")
    t = result["totals"]
    print(f"  flips={t['flip_px']} deadzone_HR={t['deadzone_flip_px_HR']} "
          f"dseg_eq_HR={t['dseg_equivalent_HR']:.10e} dseg_eq_H1={t['dseg_equivalent_H1']:.10e} "
          f"-> {result['thresholds']['disposition_HR']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
