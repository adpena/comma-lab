#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ANNULUS CONVERGENCE telemetry across a level-set witness run's preserved checkpoints.

Measures whether the codim-1 boundary ANNULUS (the low-GT-margin band that holds ~89% of
d_seg) TIGHTENS as training progresses -- and, crucially, whether it does so WITHOUT
leaving structural (interior) flips behind. This is the SENSE layer for the #247 costate
controller (task #333): observability, NOT a score. Pointer 0.19110 is UNMOVED.

AUTHORITY (NO-FAKE): renders each preserved checkpoint through the REAL render-through-R +
REAL frozen CPU-torch SegNet argmax path by REUSING ``tools/witness_per_stage_annulus_attribution.py``
(``process_ckpt`` / ``build_render_context`` / ``cfg_scalars`` / ``load_ckpt`` / ``gt_subset``
+ ``load_gt_from_cache``) -- the render path is NOT reimplemented here. The annulus is defined
on the FIXED GT SegNet top1-top2 margin (``gt.margins``), constant across checkpoints, so the
series is apples-to-apples. numpy-fp32 is the verdict authority; every row is tagged
``[macOS-numpy advisory . NON-PROMOTABLE]``. CPU/numpy only; READ-ONLY on checkpoints; does
NOT disturb the live training run.

CANONICAL comma10k class order (CLAUDE.md NON-NEGOTIABLE -- never luma-sort):
  0=Road  1=Lane  2=Undrivable(incl sky)  3=Movable(cars)  4=MyCar(ego hood)

Usage (auto-discover a run's stage checkpoints + BEST):
  .venv/bin/python tools/witness_annulus_convergence.py \\
      --run-dir experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z \\
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \\
      --num-pairs 600 --pairs 96 \\
      --out-dir experiments/results/witness_annulus_convergence

Or explicit checkpoints:
  .venv/bin/python tools/witness_annulus_convergence.py \\
      --ckpt CE=.../levelset_ckpt_stageCE_ep299.npz \\
      --ckpt BEST=.../levelset_witness_ema_BEST.npz \\
      --gt-cache .../gt_n600.npz --num-pairs 600 --pairs 16 --out-dir OUT
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# REUSE the canonical render/verdict machinery (do NOT reimplement it).
import witness_per_stage_annulus_attribution as attr  # noqa: E402

from tac.witness_annulus_metrics import (  # noqa: E402
    ADVISORY,
    CLASS_NAMES,
    N_CLASSES,
    checkpoint_metrics,
    convergence_series,
)

_EP_RE = re.compile(r"ep(\d+)")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _short_name(path: Path, epoch: int) -> str:
    stem = path.stem
    m = re.search(r"stage([A-Za-z0-9]+)", stem)
    if m:
        return f"{m.group(1)}_ep{epoch}"
    if "BEST" in stem:
        return f"BEST_ep{epoch}"
    mep = _EP_RE.search(stem)
    if mep:
        return f"{stem.split('_ep')[0].split('_')[-1]}_ep{epoch}"
    return f"{stem}_ep{epoch}"


def discover_checkpoints(run_dir: Path, log) -> list[tuple[str, Path, int]]:
    """Find stage checkpoints + BEST in a run dir; return (name, path, epoch) sorted by epoch."""
    patterns = ["levelset_ckpt_stage*.npz", "levelset_ema_stage*.npz", "levelset_witness_ema_BEST.npz"]
    seen: dict[Path, None] = {}
    for pat in patterns:
        for p in sorted(run_dir.glob(pat)):
            seen.setdefault(p.resolve(), None)
    found: list[tuple[str, Path, int]] = []
    for p in seen:
        try:
            _params, cfg = attr.load_ckpt(p)
            ep = int(cfg.get("__epoch", -1))
        except Exception as e:
            log(f"[discover] skip {p.name}: {e}")
            continue
        if ep < 0:
            mep = _EP_RE.search(p.stem)
            ep = int(mep.group(1)) if mep else 0
        found.append((_short_name(p, ep), p, ep))
    found.sort(key=lambda t: t[2])
    return found


def gt_margin_subset(gt, vpairs) -> np.ndarray:
    """FIXED GT SegNet top1-top2 margin for the verdict subset (V,h,w)."""
    return np.stack([np.asarray(gt.margins[pi], np.float32) for pi in vpairs], axis=0)


# ---------------------------------------------------------------------------
def render_ckpt_maps(name, path, gt, seg_cpu, vpairs, so_iters, seg_chunk, out_dir, log):
    """Render one checkpoint -> (witness_argmax (V,h,w) int8, witness_margin (V,h,w) f32).

    Reuses attr.process_ckpt (the REAL render-through-R + CPU-torch SegNet path). Caches
    to out_dir/maps_<name>.npz for resume (read-only on the checkpoint itself).
    """
    cache = out_dir / f"maps_{name}.npz"
    if cache.exists():
        log(f"[{name}] resume: cached maps {cache.name}")
        z = np.load(cache, allow_pickle=False)
        return z["argmax"], z["gt_margin"], int(z["epoch"]), float(z["softmax_temp"]), float(z["mean_iters"])
    params, cfg = attr.load_ckpt(path)
    sc = attr.cfg_scalars(cfg, params)
    log(f"[{name}] ep{sc['epoch']} softmax_temp={sc['softmax_temp']:.4f} "
        f"self_orient={sc['self_orient']} render={sc['render_h']}x{sc['render_w']}")
    coords, curv = attr.build_render_context(sc)
    dir_w_expect = 4 * sc["n_dir_freqs"] if sc["self_orient"] else 0
    if params["in_proj.weight"].shape[1] != curv.shape[1] + dir_w_expect:
        raise ValueError(
            f"[{name}] in_feat {params['in_proj.weight'].shape[1]} != curv {curv.shape[1]} + "
            f"dir_w {dir_w_expect}: bank cfg does not reproduce the trained input width (NO-FAKE).")
    res = attr.process_ckpt(name, path, sc, coords, curv, vpairs, gt, seg_cpu, so_iters, seg_chunk, log)
    mean_it = float(np.mean(res["iters"]))
    np.savez(out_dir / f".maps_{name}.tmp.npz", argmax=res["argmax"], gt_margin=res["gt_margin"],
             epoch=res["epoch"], softmax_temp=res["softmax_temp"], mean_iters=mean_it)
    os.replace(out_dir / f".maps_{name}.tmp.npz", cache)
    return res["argmax"], res["gt_margin"], int(res["epoch"]), float(res["softmax_temp"]), mean_it


# ---------------------------------------------------------------------------
def make_plot(out_png, series, cs_thr, cs_bk, run_label, n_vpairs, defn_used, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[png] matplotlib unavailable ({e}); skipping plot.")
        return None
    cs = cs_thr if defn_used == "threshold" else cs_bk
    epochs = cs["epochs"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    # panel 1: annulus vs interior flip frac (overall) + per-class annulus flip frac.
    ax[0].plot(epochs, cs["annulus_flip_frac_curve"], "o-", lw=2.4, color="#c0392b", label="annulus flip-frac (overall)")
    ax[0].plot(epochs, cs["interior_flip_frac_curve"], "s--", lw=2.0, color="#2c3e50", label="interior flip-frac (overall)")
    ax[0].plot(epochs, cs["overall_d_seg_curve"], "^:", lw=1.6, color="#7f8c8d", label="overall d_seg")
    class_colors = {0: "#34495e", 1: "#e67e22", 2: "#16a085", 3: "#8e44ad", 4: "#2980b9"}
    for c in range(N_CLASSES):
        ax[0].plot(epochs, cs["per_class_annulus_flip_frac_curve"][c], ".-", lw=1.1, alpha=0.75,
                   color=class_colors[c], label=f"annulus flip {CLASS_NAMES[c]}")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("flip fraction")
    ax[0].set_title(f"Annulus vs interior flip-frac  (defn={defn_used})")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=6.5, ncol=2, loc="best")
    # panel 2: witness margin p10/p50 within annulus (left) + annulus area frac (right).
    ax[1].plot(epochs, cs["margin_p10_curve"], "o-", color="#27ae60", label="witness margin p10 (annulus)")
    ax[1].plot(epochs, cs["margin_p50_curve"], "s-", color="#2980b9", label="witness margin p50 (annulus)")
    ax[1].axhline(0.0, color="k", lw=0.8, ls=":", alpha=0.6)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("realized witness margin-toward-GT")
    ax[1].set_title("Witness margin in annulus (>0 = argmax already correct) + area")
    ax[1].grid(alpha=0.3)
    axr = ax[1].twinx()
    axr.plot(epochs, cs["annulus_area_frac_curve"], "^--", color="#95a5a6", label="annulus area frac")
    axr.set_ylabel("annulus area frac")
    lines1, labs1 = ax[1].get_legend_handles_labels()
    lines2, labs2 = axr.get_legend_handles_labels()
    ax[1].legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="best")
    fig.suptitle(f"Witness annulus convergence  |  {run_label}  |  pairs={n_vpairs}  |  {ADVISORY}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=95)
    plt.close(fig)
    log(f"[png] wrote {out_png}")
    return str(out_png)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="auto-discover stage ckpts + BEST in this run dir.")
    src.add_argument("--ckpt", action="append", help="NAME=path.npz (repeat). explicit checkpoint set.")
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--pairs", type=int, default=96,
                    help="verdict pairs to render (strided ADVISORY SUBSET). default 96; use 600 for the full authority series.")
    ap.add_argument("--band", type=float, default=2.0, help="fixed-threshold annulus: |GT margin| < band. default 2.0.")
    ap.add_argument("--bottom-k", type=float, default=0.05, help="bottom-k-fraction annulus definition. default 0.05.")
    ap.add_argument("--defn-for-plot", choices=("threshold", "bottom_k_def"), default="threshold")
    ap.add_argument("--so-iters", type=int, default=4, help="self-orient deploy fixed-point iters (byte-close default 4).")
    ap.add_argument("--seg-chunk", type=int, default=16, help="SegNet batch chunk (memory bound).")
    ap.add_argument("--threads", type=int, default=4, help="OMP/torch threads (keep modest; do NOT starve the live run).")
    ap.add_argument("--out-dir", type=Path, default=REPO / "experiments/results/witness_annulus_convergence")
    args = ap.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    import torch
    torch.set_num_threads(args.threads)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{_utc()}] {msg}", flush=True)

    # resolve the checkpoint set.
    if args.run_dir:
        run_label = args.run_dir.name
        disc = discover_checkpoints(args.run_dir, log)
        if not disc:
            raise SystemExit(f"no checkpoints found under {args.run_dir}")
        ckpts = [(n, p) for n, p, _e in disc]
        log(f"discovered {len(ckpts)} checkpoints: {[n for n, _ in ckpts]}")
    else:
        run_label = "explicit"
        ckpts = []
        for spec in args.ckpt:
            if "=" not in spec:
                raise SystemExit(f"--ckpt must be NAME=path; got {spec!r}")
            name, p = spec.split("=", 1)
            ckpts.append((name, Path(p)))

    from train_witness_realized_through_R_mlx import load_gt_from_cache
    log(f"loading GT cache {args.gt_cache} (P={args.num_pairs}) ...")
    gt, seg_cpu, _posenet = load_gt_from_cache(args.gt_cache, args.num_pairs)
    P = gt.n_pairs
    stride = max(1, P // max(args.pairs, 1))
    vpairs = list(range(0, P, stride))[:args.pairs]
    log(f"verdict subset: {len(vpairs)} pairs (stride {stride}); annulus band={args.band} bottom_k={args.bottom_k}")

    g = attr.gt_subset(gt, vpairs)          # (V,h,w) int8 GT argmax
    gm = gt_margin_subset(gt, vpairs)       # (V,h,w) f32 FIXED GT margin (annulus definition)

    t_start = time.time()
    series = []
    for name, path in ckpts:
        wa, wm, epoch, softmax_temp, mean_it = render_ckpt_maps(
            name, path, gt, seg_cpu, vpairs, args.so_iters, args.seg_chunk, out_dir, log)
        metrics = checkpoint_metrics(wa, wm, g, gm, band=args.band, bottom_k=args.bottom_k)
        series.append({"name": name, "epoch": float(epoch), "softmax_temp": softmax_temp,
                       "mean_iters": mean_it, "metrics": metrics})
        thr = metrics["threshold"]
        log(f"[{name}] ep{epoch}: d_seg={metrics['overall_d_seg']:.6f} "
            f"annulus_flip={thr['annulus_flip_frac']:.5f} interior_flip={thr['interior_flip_frac']:.6f} "
            f"(annulus holds {100 * thr['annulus_flip_mass_share']:.1f}% of flips)")

    cs_thr = convergence_series(series, defn="threshold")
    cs_bk = convergence_series(series, defn="bottom_k_def")

    summary = {
        "advisory": ADVISORY,
        "generated": _utc(),
        "run_label": run_label,
        "gt_cache": str(args.gt_cache),
        "num_pairs": P,
        "verdict_pairs": len(vpairs),
        "stride": stride,
        "advisory_subset": len(vpairs) < P,
        "band": args.band,
        "bottom_k": args.bottom_k,
        "so_iters": args.so_iters,
        "class_names": CLASS_NAMES,
        "checkpoints": [{"name": s["name"], "epoch": s["epoch"], "softmax_temp": s["softmax_temp"],
                         "mean_iters": s["mean_iters"], "metrics": s["metrics"]} for s in series],
        "convergence": {"threshold": cs_thr, "bottom_k_def": cs_bk},
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    out_json = out_dir / "annulus_convergence.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log(f"wrote {out_json}")

    out_png = out_dir / "annulus_convergence.png"
    png = make_plot(out_png, series, cs_thr, cs_bk, run_label, len(vpairs), args.defn_for_plot, log)

    # echo the key numbers to stdout for the caller.
    print(json.dumps({
        "advisory": ADVISORY,
        "run_label": run_label,
        "verdict_pairs": len(vpairs),
        "advisory_subset": len(vpairs) < P,
        "per_ckpt": [{
            "name": s["name"], "epoch": s["epoch"],
            "d_seg": s["metrics"]["overall_d_seg"],
            "annulus_flip_frac": s["metrics"]["threshold"]["annulus_flip_frac"],
            "interior_flip_frac": s["metrics"]["threshold"]["interior_flip_frac"],
            "annulus_flip_mass_share": s["metrics"]["threshold"]["annulus_flip_mass_share"],
            "annulus_area_frac": s["metrics"]["threshold"]["annulus_area_frac"],
            "per_class_annulus_flip_frac": s["metrics"]["threshold"]["per_class_annulus_flip_frac"],
            "margin_p10": s["metrics"]["threshold"]["annulus_margin"]["p10"],
            "margin_p50": s["metrics"]["threshold"]["annulus_margin"]["p50"],
        } for s in series],
        "rates_threshold": {
            "overall_d_seg_rate": cs_thr["overall_d_seg_rate"],
            "annulus_flip_frac_rate": cs_thr["annulus_flip_frac_rate"],
            "interior_flip_frac_rate": cs_thr["interior_flip_frac_rate"],
            "per_class_annulus_flip_frac_rate": cs_thr["per_class_annulus_flip_frac_rate"],
        },
        "png": png,
        "json": str(out_json),
    }, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
