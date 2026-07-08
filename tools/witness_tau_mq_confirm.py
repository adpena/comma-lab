#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""tau-CONFIRM instrument: recompute m_q (flip-mass GT-margin quantiles) + tau*_end = m_q/ln5.

LAW + BAND PROVENANCE (T5 crucible): DRAFT_OPTIMAL_STACK v3 SS2.2d adopted
tau*_end = m_q/ln5 = 0.062 with m_q = 0.10 = the SUPPORT EDGE of the flip mass on the
GT-margin axis, anchored in ``birth_death_persistence_dseg_20260630T172510Z.md``
(flip-rate 0.7645 for GT-margin < 0.10, ~0.000 above -- measured on the theta*
per-stage-attribution l7 witness, 96 pairs, [::4,::4] pixel subsample). v3's own pre-GO
rule: recompute m_q from THIS vehicle's margin field; if the support edge differs
materially, re-derive. This tool is that recompute, runnable on ANY run/checkpoint.

Wave-A pre-registered interpretation (probe_waveA_ct_schedule_20260708.md SS4 follow-up):
m_q(90%) in 0.3-0.5 => tau*_end ~0.19-0.31; m_q(90%) ~ 0.10-0.15 => 0.062 stands.

AUTHORITY (NO-FAKE): checkpoint renders REUSE ``tools/witness_annulus_convergence.py``
``render_ckpt_maps`` -> ``witness_per_stage_annulus_attribution.process_ckpt`` (the REAL
render-through-R + frozen CPU-torch SegNet argmax path; NOT reimplemented). Metric math is
``tac.witness_annulus_metrics.flip_margin_quantiles`` (pure numpy, tested). Every row is
``[macOS-numpy advisory . NON-PROMOTABLE]``; advisory subset (pairs < 600); NO score claims.

MARGIN-FIELD TRAP (the confound this instrument's first run CAUGHT, 2026-07-08): the
``gt_margin`` key inside rendered maps npz files (``maps_*.npz`` from
``witness_per_stage_annulus_attribution.process_ckpt`` and every consumer that caches its
output) is the REALIZED WITNESS margin-toward-GT (signed; NEGATIVE at every flip by
definition), NOT the frozen GT-cache top1-top2 margin. Binning flips by that field makes
"all flip mass below any positive edge" a TAUTOLOGY -- which is exactly how the original
m_q = 0.10 anchor was produced (``tools/birth_death_persistence_dseg.py`` fed
``wz['gt_margin']`` as its per-pixel margin). This tool therefore ALWAYS takes the margin
axis from a GT cache's ``margins`` array (>= 0, witness-independent), never from a maps npz.

Usage (end-checkpoint confirm on a run):
  .venv/bin/python tools/witness_tau_mq_confirm.py \\
      --ckpt END=<run>/levelset_ckpt_stageTau_muon_ep1000.npz \\
      --ckpt BEST=<run>/levelset_witness_ema_BEST.npz \\
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \\
      --num-pairs 600 --pairs 16 --maps-dir OUT --out-json OUT/tau_mq_confirm.json

Precomputed maps (e.g. a live monitor's maps_*.npz, or another run's attribution maps with
its own aligned GT argmax .npy + its own GT cache/stride):
  --maps NAME=path/maps_X.npz[,gt=path/_gt_argmax_subset.npy][,cache=gt_strided_n200.npz][,npairs=200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.witness_annulus_metrics import ADVISORY, flip_margin_quantiles  # noqa: E402


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_quantiles(spec: str) -> tuple[float, ...]:
    qs = tuple(float(x) / 100.0 for x in spec.split(","))
    if not qs or any(not (0.0 < q <= 1.0) for q in qs):
        raise SystemExit(f"--quantiles must be percentages in (0,100]; got {spec!r}")
    return qs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="NAME=path.npz checkpoint to re-render (repeat). Requires --gt-cache.")
    ap.add_argument("--maps", action="append", default=[],
                    help="NAME=maps.npz[,gt=gt_argmax.npy][,cache=gtcache.npz][,npairs=N] precomputed "
                         "witness-argmax maps (repeat). GT argmax from gt=... or the cache subset; the "
                         "margin axis ALWAYS comes from the GT cache (see MARGIN-FIELD TRAP above).")
    ap.add_argument("--gt-cache", type=Path, default=None)
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--pairs", type=int, default=16,
                    help="verdict pairs (strided ADVISORY subset). default 16 (the tau-confirm spec).")
    ap.add_argument("--quantiles", default="50,80,90,95,99", help="flip-mass percentages. default 50,80,90,95,99")
    ap.add_argument("--edge", type=float, default=0.10, help="anchor support edge for the rate split. default 0.10")
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--seg-chunk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--maps-dir", type=Path, default=REPO / "experiments/results/witness_tau_mq_confirm",
                    help="cache dir for rendered maps (resume-safe).")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.ckpt and not args.maps:
        raise SystemExit("need at least one --ckpt or --maps")

    quantiles = _parse_quantiles(args.quantiles)

    def log(msg):
        print(f"[{_utc()}] {msg}", flush=True)

    if args.gt_cache is None:
        raise SystemExit("--gt-cache is required (the margin axis always comes from a GT cache)")

    def _cache_subset(cache_path: Path, n_pairs_avail: int, v: int):
        """(vpairs, lstars(V,h,w) int, margins(V,h,w) f32) from a GT cache, mmap'd, strided."""
        z = np.load(cache_path, mmap_mode="r")
        p_total = int(z["lstars"].shape[0])
        p_use = min(n_pairs_avail, p_total)
        stride = max(1, p_use // max(v, 1))
        vp = list(range(0, p_use, stride))[:v]
        ls = np.stack([np.asarray(z["lstars"][i]) for i in vp]).astype(np.int64)
        mg = np.stack([np.asarray(z["margins"][i], np.float32) for i in vp])
        return vp, ls, mg

    results = []
    t0 = time.time()

    # precomputed maps first (free). Only the witness ARGMAX is read from the maps npz;
    # the maps' own `gt_margin` key (witness margin-toward-GT) is deliberately IGNORED.
    for spec in args.maps:
        if "=" not in spec:
            raise SystemExit(f"--maps must be NAME=path[,gt=..][,cache=..][,npairs=N]; got {spec!r}")
        name, rest = spec.split("=", 1)
        parts = rest.split(",")
        maps_path, opts = Path(parts[0]), dict(kv.split("=", 1) for kv in parts[1:])
        z = np.load(maps_path, allow_pickle=False)
        wa = np.asarray(z["argmax"])
        epoch = int(z["epoch"]) if "epoch" in z.files else -1
        stemp = float(z["softmax_temp"]) if "softmax_temp" in z.files else float("nan")
        cache = Path(opts.get("cache", args.gt_cache))
        npairs = int(opts.get("npairs", args.num_pairs))
        vp, ls, gm = _cache_subset(cache, npairs, int(wa.shape[0]))
        ga = np.load(Path(opts["gt"])) if "gt" in opts else ls
        if "gt" in opts and not np.array_equal(np.asarray(ga, np.int64), ls):
            log(f"[{name}] WARNING: explicit gt argmax differs from {cache.name} lstars at this stride")
        if ga.shape != wa.shape:
            raise SystemExit(f"[{name}] gt argmax {ga.shape} != maps argmax {wa.shape} (stride mismatch?)")
        tbl = flip_margin_quantiles(wa, ga, gm, quantiles=quantiles, edge=args.edge)
        results.append({"name": name, "source": str(maps_path), "epoch": epoch, "softmax_temp": stemp,
                        "pairs": int(wa.shape[0]), "gt_cache": str(cache), "vpairs": vp, "table": tbl})
        log(f"[{name}] ep{epoch}: d_seg={tbl['d_seg']:.6f} n_flips={tbl['n_flips']} "
            f"mass<edge={tbl['flip_mass_share_below_edge']:.4f} m_q90={tbl['m_q'].get('q90')}")

    # checkpoint renders (REAL render path, cached to --maps-dir for resume).
    if args.ckpt:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
        import torch
        torch.set_num_threads(args.threads)
        import witness_per_stage_annulus_attribution as attr
        from witness_annulus_convergence import gt_margin_subset, render_ckpt_maps
        from train_witness_realized_through_R_mlx import load_gt_from_cache
        log(f"loading GT cache {args.gt_cache} (P={args.num_pairs}) ...")
        gt, seg_cpu, _posenet = load_gt_from_cache(args.gt_cache, args.num_pairs)
        stride = max(1, gt.n_pairs // max(args.pairs, 1))
        vpairs = list(range(0, gt.n_pairs, stride))[:args.pairs]
        log(f"verdict subset: {len(vpairs)} pairs (stride {stride})")
        args.maps_dir.mkdir(parents=True, exist_ok=True)
        ga = attr.gt_subset(gt, vpairs)
        gm = gt_margin_subset(gt, vpairs)  # FIXED GT top1-top2 margin (NOT the maps' witness margin)
        for spec in args.ckpt:
            if "=" not in spec:
                raise SystemExit(f"--ckpt must be NAME=path; got {spec!r}")
            name, p = spec.split("=", 1)
            wa, _wm, epoch, stemp, mean_it = render_ckpt_maps(
                name, Path(p), gt, seg_cpu, vpairs, args.so_iters, args.seg_chunk, args.maps_dir, log)
            tbl = flip_margin_quantiles(wa, ga, gm, quantiles=quantiles, edge=args.edge)
            results.append({"name": name, "source": p, "epoch": epoch, "softmax_temp": stemp,
                            "mean_iters": mean_it, "pairs": int(wa.shape[0]),
                            "gt_cache": str(args.gt_cache), "vpairs": vpairs, "table": tbl})
            log(f"[{name}] ep{epoch}: d_seg={tbl['d_seg']:.6f} n_flips={tbl['n_flips']} "
                f"mass<edge={tbl['flip_mass_share_below_edge']:.4f} m_q90={tbl['m_q'].get('q90')}")

    summary = {
        "advisory": ADVISORY,
        "generated": _utc(),
        "law": "tau*_end = m_q / ln5 (v3 SS2.2d; anchor m_q=0.10 from birth_death_persistence_dseg_20260630)",
        "quantiles_pct": [q * 100.0 for q in quantiles],
        "edge": args.edge,
        "pairs_requested": args.pairs,
        "advisory_subset": True,
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
    }
    out = json.dumps(summary, indent=2, default=float)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(out, encoding="utf-8")
        log(f"wrote {args.out_json}")
    print(out)


if __name__ == "__main__":
    main()
