# SPDX-License-Identifier: MIT
"""Max-plus (tropical) annulus fit of the frozen SegNet per-class logit fields (K-element prototype).

LAW + BAND PROVENANCE (T5 crucible, v5 draft S11 row 16 + S0.0a M5 + S7c "P-MP"; deriving doc
ct_deepresearch_2 max-plus band-residual decomposition):
  Fit each class's frozen-scorer logit field on the flip-support ANNULUS as a max-plus expansion
      logit_c(u,v)  ~=  max_{k<=K}  q_ck(u,v),   q_ck = quadratic (6 coeffs)
  (band/clamp/comb are the K=1 special cases; rule-118 makes the basis GENERATORS free at
  decode -- only the K coefficient sets are counted bytes).

  Pre-registered P-MP band (BINDING; the v5 S7c row leaves the accuracy bar to this instrument,
  registered HERE before any measurement):
    PASS : annulus argmax agreement (fitted 5-field argmax vs cached GT lstar) >= 0.95 at
           K <= 64 elements/class.  Secondary diagnostic: per-class p95 |residual| on the
           annulus vs the tau_end coupling 0.0998 logit (v5 row-16 coupling bound).
    KILL : K <= 64 fails the band-level annulus accuracy.
           Kill SCOPE (requirement R): kills the K<=64 element-count FORMULATION at the
           annulus only -- NOT max-plus methods as a family (untested reformulations: larger K,
           per-class K, log-sum-exp relaxation at finite tau, different element families,
           band-residual hybrid m_lane = max(band, m_INR)).
    PASS consequence: bytes of the K coefficient sets enter the lambda_bytes law at
           6.6586e-7 S/byte (requirement J currency).
  M5 control: the same fit on BULK patches (high-margin sample) is reported -- expected
  blow-up (the two-semiring split: bulk stays INR/curvelet).

METHOD: per frame per class, alternating piecewise-quadratic max-regression (Magnani-Boyd
style): assign each annulus pixel to its current argmax element, ridge-refit each element's
quadratic on its cell, reassign; init by k-means on (u,v). Logits computed with the frozen
CPU-torch SegNet on x = bilinear-downsample(gt_f1) (the gt-cache convention; per-frame argmax
agreement with cached lstars is asserted >= 0.99 -- the path self-validates).

ARTIFACT SCHEMA:
  { "schema": "maxplus_annulus_fit.v1", "frames": [...], "K_grid": [...],
    "per_K": { "<K>": { "annulus_argmax_agreement", "per_class_p95_residual",
                         "per_class_rms_residual", "bulk_argmax_agreement",
                         "bytes_per_frame_fp16", "bytes_n600_fp16", "S_cost_n600" } },
    "band": {...}, "verdict": {...} }

USAGE
  .venv/bin/python tools/maxplus_annulus_fit.py --frames 8 \
      --out experiments/results/t5_probe_waveB_20260708/pmp_maxplus_fit.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEG_HW = (384, 512)
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
PASS_AGREEMENT = 0.95
K_MAX = 64
TAU_END_COUPLING = 0.0998  # tau_end * ln5 (v5 row-16 coupling bound), logit units
LAMBDA_BYTES = 6.6586e-7   # S per byte (requirement J)
ANNULUS_BAND = 0.10        # flip-support edge (measured; v5 consistency row (a))


def _now_utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")):
        raise ValueError(f"{path!r} is a /tmp-class durable path; use the repo tier per CLAUDE.md.")


def quad_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(N,) coords in [0,1] -> (N,6) quadratic feature matrix [1,u,v,u^2,uv,v^2]."""
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)


def fit_maxplus(u, v, y, K: int, iters: int = 40, ridge: float = 1e-4, seed: int = 0,
                curv_cap: float = 2000.0):
    """Alternating max-of-quadratics regression (Magnani-Boyd style). Returns (coeffs (K,6), yhat (N,)).

    Elements whose cell drops below 7 pixels (or whose solve is non-finite) are set DEAD:
    a constant far below min(y), so they can never win the max (the zero-element pathology
    -- a 0-coeff element beating negative logits -- is structurally excluded).
    """
    rng = np.random.default_rng(seed)
    n = y.size
    feats = quad_features(u, v)
    K = min(K, max(1, n // 8))
    dead_c0 = float(y.min()) - 1.0e3
    centers = rng.choice(n, size=K, replace=False)
    d2 = (u[:, None] - u[centers][None, :]) ** 2 + (v[:, None] - v[centers][None, :]) ** 2
    assign = d2.argmin(axis=1)
    coeffs = np.zeros((K, 6), np.float64)
    coeffs[:, 0] = dead_c0
    eye = np.eye(6)
    for _ in range(iters):
        for k in range(K):
            sel = assign == k
            if int(sel.sum()) < 7:
                coeffs[k] = 0.0
                coeffs[k, 0] = dead_c0
                continue
            A, b = feats[sel], y[sel]
            sol = np.linalg.solve(A.T @ A + ridge * eye, A.T @ b)
            if not np.all(np.isfinite(sol)):
                coeffs[k] = 0.0
                coeffs[k, 0] = dead_c0
                continue
            # CONCAVITY PROJECTION (McEneaney max-plus basis: elements are CONCAVE quadratics;
            # unconstrained quadratics explode upward away from their cell and hijack the max).
            hess = np.array([[2.0 * sol[3], sol[4]], [sol[4], 2.0 * sol[5]]])
            evals, evecs = np.linalg.eigh(hess)
            if np.any(evals > 0.0) or np.any(evals < -curv_cap):
                # NSD + curvature cap: eigenvalues into [-curv_cap, 0]. Uncapped concave
                # quadratics on thin-band cells decay by 100s of logits between cells.
                hess_c = (evecs * np.clip(evals, -curv_cap, 0.0)) @ evecs.T
                sol[3], sol[4], sol[5] = 0.5 * hess_c[0, 0], hess_c[0, 1], 0.5 * hess_c[1, 1]
                # refit the affine part with the (now-fixed) constrained quadratic part
                resid = b - A[:, 3:] @ sol[3:]
                Al = A[:, :3]
                sol[:3] = np.linalg.solve(Al.T @ Al + ridge * np.eye(3), Al.T @ resid)
            coeffs[k] = sol
        vals = feats @ coeffs.T  # (N,K)
        new_assign = vals.argmax(axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
    vals = feats @ coeffs.T
    yhat = vals.max(axis=1)
    return coeffs, yhat


def fit_piecewise_capacity(u, v, y, K: int, iters: int = 40, ridge: float = 1e-4, seed: int = 0):
    """K-quadratic piecewise regression with MIN-RESIDUAL assignment (oracle selection).

    NOT a max-plus model: this is the capacity bound separating "K quadratics cannot track the
    field at all" from "the max-envelope cannot select the right element" (kill-scope
    sharpener, requirement R). Returns per-pixel |residual| under oracle selection.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    feats = quad_features(u, v)
    K = min(K, max(1, n // 8))
    centers = rng.choice(n, size=K, replace=False)
    d2 = (u[:, None] - u[centers][None, :]) ** 2 + (v[:, None] - v[centers][None, :]) ** 2
    assign = d2.argmin(axis=1)
    coeffs = np.zeros((K, 6), np.float64)
    eye = np.eye(6)
    for _ in range(iters):
        for k in range(K):
            sel = assign == k
            if int(sel.sum()) < 7:
                continue
            A, b = feats[sel], y[sel]
            sol = np.linalg.solve(A.T @ A + ridge * eye, A.T @ b)
            if np.all(np.isfinite(sol)):
                coeffs[k] = sol
        resid = np.abs(feats @ coeffs.T - y[:, None])  # (N,K)
        new_assign = resid.argmin(axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
    return np.abs((feats @ coeffs.T)[np.arange(n), assign] - y)


def segnet_logits(seg, gt_f1_uint8: np.ndarray) -> np.ndarray:
    """gt_f1 (874,1164,3) uint8 -> logits (5,384,512) float32 (frozen CPU SegNet, gt-cache path)."""
    import torch

    t = torch.from_numpy(np.asarray(gt_f1_uint8)).float().permute(2, 0, 1)[None]
    x = torch.nn.functional.interpolate(t, size=SEG_HW, mode="bilinear", align_corners=False)
    pair = torch.stack([x[0], x[0]], dim=0)[None]  # (1,2,3,384,512)
    with torch.no_grad():
        seg_in = seg.preprocess_input(pair)
        logits = seg(seg_in)
    return logits[0].numpy().astype(np.float64)  # (5,384,512)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--frames", type=int, default=8, help="number of frames (evenly spread over the cache)")
    ap.add_argument("--k-grid", default="4,8,16,32,64")
    ap.add_argument("--band", type=float, default=ANNULUS_BAND, help="EVAL annulus = GT margin < band (logit)")
    ap.add_argument("--fit-band", type=float, default=0.5,
                    help="FIT support = GT margin < fit-band (wider context stabilizes the fit; "
                         "eval stays on --band)")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--bulk-sample", type=int, default=8000, help="bulk control pixels per frame (margin >= 1.0)")
    ap.add_argument("--argmax-validate-min", type=float, default=0.99)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = (REPO / args.out) if not os.path.isabs(args.out) else Path(args.out)
    _refuse_tmp(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from tac.boundary_math.seg_core import load_real_segnet

    seg = load_real_segnet("cpu")
    seg.eval()

    z = np.load(REPO / args.gt_cache, mmap_mode="r")
    lstars, margins, gt_f1 = z["lstars"], z["margins"], z["gt_f1"]
    P = int(lstars.shape[0])
    frame_ids = sorted(set(np.linspace(0, P - 1, args.frames).astype(int).tolist()))
    k_grid = sorted(int(k) for k in args.k_grid.split(","))
    H, W = SEG_HW
    vv, uu = np.meshgrid(np.arange(W) / W, np.arange(H) / H)  # u=row/H, v=col/W in [0,1]

    per_K: dict[int, dict] = {k: {"ann_agree_num": 0, "ann_n": 0,
                                  "bulk_agree_num": 0, "bulk_n": 0,
                                  "res_by_class": {c: [] for c in range(5)},
                                  "own_by_class": {c: [] for c in range(5)}} for k in k_grid}
    t0 = time.time()
    rng = np.random.default_rng(0)
    for fi in frame_ids:
        gta = np.asarray(lstars[fi], np.int64)
        gtm = np.asarray(margins[fi], np.float64)
        logits = segnet_logits(seg, gt_f1[fi])
        agree = float((logits.argmax(axis=0) == gta).mean())
        if agree < args.argmax_validate_min:
            raise RuntimeError(f"logit path does not reproduce cached lstars (frame {fi}: {agree:.4f})")
        ann_eval = gtm < args.band
        ann_fit = gtm < args.fit_band
        n_ann = int(ann_eval.sum())
        if n_ann < 50:
            continue
        uf, vf = uu[ann_fit], vv[ann_fit]
        ue, ve = uu[ann_eval], vv[ann_eval]
        feats_eval = quad_features(ue, ve)
        gta_ann = gta[ann_eval]
        bulk = (gtm >= 1.0)
        bidx = rng.choice(np.flatnonzero(bulk), size=min(args.bulk_sample, int(bulk.sum())), replace=False)
        ub, vb = uu.ravel()[bidx], vv.ravel()[bidx]
        gtb = gta.ravel()[bidx]
        for K in k_grid:
            yhat_ann = np.zeros((5, n_ann))
            yhat_bulk = np.zeros((5, bidx.size))
            for c in range(5):
                y_fit = logits[c][ann_fit]
                coeffs, _ = fit_maxplus(uf, vf, y_fit, K, iters=args.iters, seed=c)
                yh = (feats_eval @ coeffs.T).max(axis=1)
                yhat_ann[c] = yh
                per_K[K]["res_by_class"][c].append(np.abs(yh - logits[c][ann_eval]))
                # oracle-selection capacity bound on the eval subset (eval mask subset of fit
                # mask; residuals follow the raveled fit-mask pixel order)
                own_res = fit_piecewise_capacity(uf, vf, y_fit, K, iters=args.iters, seed=c)
                eval_sel = ann_eval.ravel()[np.flatnonzero(ann_fit.ravel())]
                per_K[K]["own_by_class"][c].append(own_res[eval_sel])
                # bulk control: refit ON bulk to measure representability there (M5 blow-up check)
                _, yh_b = fit_maxplus(ub, vb, logits[c].ravel()[bidx], K, iters=args.iters, seed=100 + c)
                yhat_bulk[c] = yh_b
            per_K[K]["ann_agree_num"] += int((yhat_ann.argmax(axis=0) == gta_ann).sum())
            per_K[K]["ann_n"] += n_ann
            per_K[K]["bulk_agree_num"] += int((yhat_bulk.argmax(axis=0) == gtb).sum())
            per_K[K]["bulk_n"] += int(bidx.size)
        print(f"[frame {fi}] annulus={n_ann}px done ({time.time()-t0:.1f}s)")

    per_K_out = {}
    best_pass_K = None
    for K in k_grid:
        d = per_K[K]
        agree = d["ann_agree_num"] / d["ann_n"] if d["ann_n"] else float("nan")
        res = {c: np.concatenate(d["res_by_class"][c]) for c in range(5) if d["res_by_class"][c]}
        own = {c: np.concatenate(d["own_by_class"][c]) for c in range(5) if d["own_by_class"][c]}
        per_K_out[str(K)] = {
            "annulus_argmax_agreement": agree,
            "bulk_argmax_agreement": d["bulk_agree_num"] / d["bulk_n"] if d["bulk_n"] else float("nan"),
            "per_class_p95_residual": {CLASS_NAMES[c]: float(np.percentile(r, 95)) for c, r in res.items()},
            "per_class_rms_residual": {CLASS_NAMES[c]: float(np.sqrt((r ** 2).mean())) for c, r in res.items()},
            "per_class_rms_oracle_selection_bound": {CLASS_NAMES[c]: float(np.sqrt((r ** 2).mean())) for c, r in own.items()},
            "bytes_per_frame_fp16": 5 * K * 6 * 2,
            "bytes_n600_fp16": 600 * 5 * K * 6 * 2,
            "S_cost_n600": 600 * 5 * K * 6 * 2 * LAMBDA_BYTES,
        }
        if np.isfinite(agree) and agree >= PASS_AGREEMENT and K <= K_MAX and best_pass_K is None:
            best_pass_K = K

    result = {
        "schema": "maxplus_annulus_fit.v1",
        "generated_utc": _now_utc(),
        "argv": sys.argv[1:],
        "inputs": {"gt_cache": args.gt_cache},
        "frames": [int(f) for f in frame_ids],
        "annulus_band_logit": args.band,
        "element_form": "quadratic (6 coeffs), max over K, per class per frame",
        "band": {"pass_agreement_gte": PASS_AGREEMENT, "K_max": K_MAX,
                 "secondary_p95_vs_logit": TAU_END_COUPLING,
                 "provenance": "v5 S11 row 16 + S0.0a M5 + S7c P-MP; bar registered by this "
                               "instrument pre-measurement"},
        "per_K": per_K_out,
        "verdict": {
            "pass": best_pass_K is not None,
            "smallest_passing_K": best_pass_K,
            "scope_if_kill": "K<=64 element-count FORMULATION at the annulus only (requirement R); "
                             "reformulations: larger K, per-class K, log-sum-exp relaxation, "
                             "other element families, band-residual hybrid",
        },
        "advisory": "[macOS-numpy advisory subset-frames . NON-PROMOTABLE] pointer 0.19110 UNMOVED",
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, out)
    print(f"[done] {out}")
    for K in k_grid:
        r = per_K_out[str(K)]
        print(f"  K={K}: annulus_agree={r['annulus_argmax_agreement']:.6f} "
              f"bulk_agree={r['bulk_argmax_agreement']:.6f} bytes_n600={r['bytes_n600_fp16']} "
              f"S_cost={r['S_cost_n600']:.8f}")
    print(f"  VERDICT pass={result['verdict']['pass']} smallest_passing_K={best_pass_K}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
