# SPDX-License-Identifier: MIT
"""Conley persistence certificate backtest + per-island ledger (B17-consumable schema).

LAW + BAND PROVENANCE (T5 crucible, v5 draft S3.4 / S7c "P-CON"; deriving doc
`.omx/research/t5_crucible/ct_deepresearch_2_pde_geometric_topological_control_20260707.md` S7):
  Certificate: island I survives stage k (and decode)  <==  pers(I) > tau_k * ln(5) + Delta_dec_logit.
  pers(I) is the H0 superlevel birth amplitude of the GT top1-top2 margin over the island;
  for a GT class component whose margin dies at the 0 boundary the birth-death amplitude is the
  component's PEAK GT margin (the same filtration as the 20260630 birth-death ledger,
  `tools/birth_death_persistence_dseg.py` / `.omx/research/birth_death_persistence_dseg_20260630T172510Z.md`).

  Pre-registered P-CON bands (BINDING):
    BAND : certified-survival P(survive | certified) >= 0.95 at BOTH known tau points.
    KILL : < 0.80  ==>  fit the safety factor s in  s * (tau * ln 5)  and report it.
           Kill SCOPE (requirement R): fitting s IS the reformulation -- the certificate FAMILY
           survives by construction; only s unboundedly large (no s in the fitted grid reaches
           the target survival) would be a FORMULATION kill.
  The two known tau points are the per-stage softmax temperatures stored in the witness maps
  (`softmax_temp` key): Tau stage tau ~= 0.050007 (ep599) and MuonBest tau ~= 0.215689 (ep900).

SURVIVAL DEFINITION (backtest form): island survives <=> witness argmax reproduces the GT class
on more than (1 - fliprate_max) of the island's pixels (default fliprate_max = 0.5: majority
survival). Honest boundary carried from v5 S3.4: the certificate is SUFFICIENT-not-necessary
(sub-threshold islands MAY survive) and pers is measured on the same smoothed field the flow
acts on, so the inequality's sides are not fully independent.

ARTIFACT SCHEMA (stable; the B17 in-run alarm reads the SAME per-island row format):
  JSON summary: { "schema": "conley_persistence_certifier.v1", "stages": { "<stage>": {
      "tau", "threshold_logit", "delta_dec_logit", "per_class": {...}, "overall": {
      "n_islands", "n_certified", "certified_survival", "uncertified_survival",
      "certified_survival_px_weighted" }, "safety_factor_fit": {...}|null } }, ... }
  NPZ per-island ledger (one row per island): stage_idx, frame, class, size_px, pers,
      flip_rate, certified (0/1), survived (0/1).

USAGE
  .venv/bin/python tools/conley_persistence_certifier.py \
      --stages Tau,MuonBest \
      --out experiments/results/t5_probe_waveB_20260708/pcon_conley_backtest.json
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
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
LN5 = float(np.log(5.0))
BAND_PASS = 0.95
BAND_KILL = 0.80


def _now_utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")):
        raise ValueError(f"{path!r} is a /tmp-class durable path; use the repo tier per CLAUDE.md.")


def islands_for_stage(lstars, margins, gt_arg_sub, w_argmax, stride: int, n_w: int,
                      classes: list[int], min_px: int):
    """Yield per-island rows (frame, class, size_px, pers, flip_rate) for one stage's witness maps."""
    from scipy.ndimage import label as nd_label
    from scipy.ndimage import maximum as nd_max
    from scipy.ndimage import sum as nd_sum

    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for w in range(n_w):
        s_idx = stride * w
        gta = np.asarray(lstars[s_idx], dtype=np.int64)
        if not np.array_equal(gta.astype(np.int8), np.asarray(gt_arg_sub[w])):
            raise RuntimeError(f"frame alignment broken at witness frame {w} (gt idx {s_idx})")
        gtm = np.asarray(margins[s_idx], dtype=np.float64)
        wa = w_argmax[w].astype(np.int64)
        flipf = (wa != gta).astype(np.float64)
        for c in classes:
            mask = gta == c
            if int(mask.sum()) < min_px:
                continue
            lab, n = nd_label(mask, structure=struct4)
            if n == 0:
                continue
            ids = np.arange(1, n + 1)
            sizes = np.asarray(nd_sum(np.ones_like(lab, dtype=np.float64), lab, ids), np.float64)
            peaks = np.asarray(nd_max(gtm, lab, ids), np.float64)
            flips = np.asarray(nd_sum(flipf, lab, ids), np.float64)
            keep = sizes >= min_px
            for sz, pk, fl in zip(sizes[keep], peaks[keep], flips[keep], strict=True):
                yield w, c, float(sz), float(pk), float(fl / sz)


def survival_given(pers: np.ndarray, survived: np.ndarray, sizes: np.ndarray, thr: float):
    cert = pers > thr
    n_c = int(cert.sum())
    out = {
        "n_islands": int(pers.size),
        "n_certified": n_c,
        "certified_survival": float(survived[cert].mean()) if n_c else float("nan"),
        "uncertified_survival": float(survived[~cert].mean()) if int((~cert).sum()) else float("nan"),
        "certified_survival_px_weighted": (
            float((survived[cert] * sizes[cert]).sum() / sizes[cert].sum()) if n_c else float("nan")
        ),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--witness-dir", default="experiments/results/witness_per_stage_attribution")
    ap.add_argument("--stages", default="Tau,MuonBest", help="comma list; each maps_<stage>.npz supplies its tau")
    ap.add_argument("--witness-to-gt-stride", type=int, default=2)
    ap.add_argument("--n-frames", type=int, default=0, help="0 = all witness frames")
    ap.add_argument("--classes", default="0,1,2,3,4")
    ap.add_argument("--min-island-px", type=int, default=1)
    ap.add_argument("--fliprate-max", type=float, default=0.5, help="survive <=> flip_rate < this")
    ap.add_argument("--delta-dec-logit", type=float, default=0.0,
                    help="decode-gap logit term (UNMEASURED at build; v5 S3.4 initializes 0)")
    ap.add_argument("--safety-target", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = (REPO / args.out) if not os.path.isabs(args.out) else Path(args.out)
    _refuse_tmp(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(REPO / args.gt_cache, mmap_mode="r")
    lstars, margins = z["lstars"], z["margins"]
    wdir = REPO / args.witness_dir
    gt_arg_sub = np.load(wdir / "_gt_argmax_subset.npy", mmap_mode="r")
    classes = [int(c) for c in args.classes.split(",")]

    stages_out: dict[str, dict] = {}
    ledger_rows: list[tuple] = []
    for st_i, stage in enumerate(args.stages.split(",")):
        wz = np.load(wdir / f"maps_{stage}.npz")
        w_argmax = np.asarray(wz["argmax"])
        tau = float(wz["softmax_temp"])
        n_w = w_argmax.shape[0] if args.n_frames <= 0 else min(args.n_frames, w_argmax.shape[0])
        rows = list(islands_for_stage(lstars, margins, gt_arg_sub, w_argmax,
                                      args.witness_to_gt_stride, n_w, classes, args.min_island_px))
        pers = np.array([r[3] for r in rows], np.float64)
        sizes = np.array([r[2] for r in rows], np.float64)
        flipr = np.array([r[4] for r in rows], np.float64)
        cls = np.array([r[1] for r in rows], np.int64)
        survived = (flipr < args.fliprate_max)
        thr = tau * LN5 + args.delta_dec_logit
        certified = pers > thr
        for (w, c, sz, pk, fl), ce, su in zip(rows, certified, survived, strict=True):
            ledger_rows.append((st_i, w, c, sz, pk, fl, int(ce), int(su)))

        overall = survival_given(pers, survived, sizes, thr)
        per_class = {}
        for c in classes:
            sel = cls == c
            if not sel.any():
                continue
            per_class[CLASS_NAMES.get(c, str(c))] = survival_given(pers[sel], survived[sel], sizes[sel], thr)

        cs = overall["certified_survival"]
        fit = None
        if np.isfinite(cs) and cs < BAND_KILL:
            # KILL branch: fit the safety factor s (the reformulation, requirement R).
            for s in np.arange(1.0, 40.01, 0.25):
                r = survival_given(pers, survived, sizes, s * tau * LN5 + args.delta_dec_logit)
                if r["n_certified"] >= 10 and np.isfinite(r["certified_survival"]) \
                        and r["certified_survival"] >= args.safety_target:
                    fit = {"s": float(s), "threshold_logit": float(s * tau * LN5 + args.delta_dec_logit),
                           **r}
                    break
            if fit is None:
                fit = {"s": None, "note": "no s <= 40 reaches target -- FORMULATION-level failure"}
        stages_out[stage] = {
            "tau": tau,
            "epoch": int(wz["epoch"]),
            "threshold_logit": float(thr),
            "delta_dec_logit": args.delta_dec_logit,
            "overall": overall,
            "per_class": per_class,
            "band": {"pass_gte": BAND_PASS, "kill_lt": BAND_KILL,
                     "status": ("PASS" if np.isfinite(cs) and cs >= BAND_PASS
                                else "KILL->fit_s" if np.isfinite(cs) and cs < BAND_KILL
                                else "BETWEEN")},
            "safety_factor_fit": fit,
        }

    led = np.array(ledger_rows, dtype=np.float64)
    npz_path = out.with_suffix(".ledger.npz")
    np.savez_compressed(
        npz_path,
        stage_idx=led[:, 0].astype(np.int32),
        frame=led[:, 1].astype(np.int32),
        cls=led[:, 2].astype(np.int32),
        size_px=led[:, 3].astype(np.float32),
        pers=led[:, 4].astype(np.float32),
        flip_rate=led[:, 5].astype(np.float32),
        certified=led[:, 6].astype(np.int8),
        survived=led[:, 7].astype(np.int8),
        stage_names=np.array(args.stages.split(",")),
    )

    result = {
        "schema": "conley_persistence_certifier.v1",
        "generated_utc": _now_utc(),
        "argv": sys.argv[1:],
        "inputs": {"gt_cache": args.gt_cache, "witness_dir": args.witness_dir},
        "survival_definition": f"flip_rate < {args.fliprate_max}",
        "persistence_definition": "H0 superlevel birth amplitude = component peak GT top1-top2 margin "
                                  "(same filtration as birth_death_persistence_dseg 20260630)",
        "bands": {"pass_gte": BAND_PASS, "kill_lt": BAND_KILL,
                  "provenance": "v5 S3.4/S7c P-CON; CT-2 S7"},
        "stages": stages_out,
        "per_island_ledger_npz": str(npz_path.relative_to(REPO)),
        "advisory": "[macOS-numpy advisory . NON-PROMOTABLE] pointer 0.19110 UNMOVED",
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, out)
    print(f"[done] {out}")
    for stage, s in stages_out.items():
        o = s["overall"]
        print(f"  {stage}: tau={s['tau']:.10f} thr={s['threshold_logit']:.10f} "
              f"islands={o['n_islands']} certified={o['n_certified']} "
              f"P(survive|cert)={o['certified_survival']:.10f} "
              f"P(survive|uncert)={o['uncertified_survival']:.10f} status={s['band']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
