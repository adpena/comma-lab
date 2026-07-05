#!/usr/bin/env python
"""$0 S_R-reachability calibration probe (#268 validation deliverable; focal-harness pattern
``experiments/probe_focal_gamma_calibration.py`` / d3c4771ac).

[macOS-CPU/MLX advisory] NON-PROMOTABLE calibration — NOT a score claim. Pointer 0.19110 UNMOVED.

On a live-run EMA checkpoint snapshot (read-only), real GT (gt_n24 with its cached ``sR``
reachability maps) and the real frozen MLX SegNet through the real R, MEASURE what the LEVER-4
margin-saliency term's weight flavor does to the training gradient geometry:

  flavors (exact trainer semantics, ``train_levelset_witness_realized_through_R_mlx.py`` LEVER-4):
    plain : sal = exp(-gt_margin/tau)                                (fragility only)
    tex   : sal = exp(-gt_margin/tau) / (1 + beta*tex(R(f)))          (UNIWARD proxy — MEASURED inert)
    reach : sal = exp(-gt_margin/tau) * sR_cached[pair]               (through-R reachability, #268)
  term  : msal = sum(relu(target - realized_margin) * sal) / (sum(sal)+1e-6)

  (a) per-region (island / bulk_boundary / bulk_interior) share of |dL/d(frame)| for
      L = base_CE + w*msal at w in {0.5, 1, 2} for EVERY flavor — composed EXACTLY from
      4 backward passes/pair via gradient linearity: grad(base + w*msal) == grad(base) +
      w*grad(msal) (identical mathematical object; composition done in float64 numpy);
  (b) the msal-term-only gradient's region shares per flavor (the pure steering direction);
  (c) the ANALYTIC sal weight-mass region distribution per flavor (no grad needed);
  (d) msal term values at the checkpoint (activation magnitude readback).

GRADIENT SURFACE: witness-alone (== deploy/verdict surface), same honest caveat as the focal
probe — the seed is not persisted in checkpoints, so island shares are an UPPER bound on the
live composed-surface shares.

Run (memory-light, MLX CPU, read-only on live artifacts):
  .venv/bin/python experiments/probe_sr_reachability_calibration.py \
      --ckpt experiments/results/bd_calib_20260705/snap/ema_BEST_ep100.npz \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
      --pairs 12 --out-json experiments/results/sr_calib_20260705/sr_calib_ep100.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "upstream")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx

# MLX CPU: never contend with a live MLX-GPU run; MLX-GPU is not bit-identical cross-process.
mx.set_default_device(mx.cpu)

from tac.boundary_math.lever_b_generator import (
    self_orientation_directional_feats,
)
from tac.boundary_math.lever_b_levelset_generator import (
    CurveletBankConfig,
    _boundary_band,
    build_coords,
    curvelet_directional_B,
    curvelet_feats,
    int8_dequant_params,
    levelset_rgb_forward_numpy,
)
from tac.local_acceleration.mlx_scorer_adapters import (
    load_mlx_distortion_scorer_adapter_from_upstream,
)
from tac.local_acceleration.pr95_hnerv_mlx_training import (
    apply_contest_faithful_roundtrip_nhwc,
)
from tac.optimization.md_decoupling import stiefel_project_columns

ISLAND_CLASSES = (1, 3)  # Lane, Movable — canonical comma10k order (CLAUDE.md NON-NEGOTIABLE)
WEIGHTS = (0.5, 1.0, 2.0)
FLAVORS = ("plain", "tex", "reach")
REGIONS = ("island", "bulk_boundary", "bulk_interior")


def _load_ckpt(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(path, allow_pickle=True)
    params = {k: np.asarray(z[k]) for k in z.files
              if not k.startswith("__") and not k.startswith("pose_carrier.")}
    cfg = {k: z[k][()] if z[k].shape == () else np.asarray(z[k]) for k in z.files if k.startswith("__")}
    return params, cfg


def _fixed_point_feats(deploy, cfg, coords, curv, pairs, iters=2):
    """Self-orient fixed-point reconstruction (validated disambiguator recipe; verbatim from the
    focal harness)."""
    h, w = int(cfg["__render_hw"][0]), int(cfg["__render_hw"][1])
    ndf = int(cfg["__cfg_n_dir_freqs"])
    dir_w = 4 * ndf
    fkw = {
        "n_hidden": int(cfg["__cfg_n_hidden"]), "hidden_dim": int(cfg["__cfg_hidden_dim"]),
        "n_classes": 5, "activation": str(cfg["__cfg_activation"]),
        "softmax_temp": float(cfg["__cfg_softmax_temp"]), "wire_w0": float(cfg["__cfg_wire_w0"]),
        "wire_s0": float(cfg["__cfg_wire_s0"]), "hosc_beta": float(cfg["__cfg_hosc_beta"]),
        "hosc_omega": float(cfg["__cfg_hosc_omega"]), "chroma": bool(cfg["__cfg_chroma"]),
    }
    feats = {pi: np.concatenate([curv, np.zeros((curv.shape[0], dir_w), np.float32)], axis=-1)
             for pi in pairs}
    prev_argmax: dict[int, np.ndarray] = {}
    flip_frac = float("nan")
    for _it in range(iters):
        flips, tot = 0, 0
        for pi in pairs:
            _rgb, phi = levelset_rgb_forward_numpy(deploy, feats[pi], deploy["code"][2 * pi + 1], **fkw)
            am = phi.argmax(-1).reshape(h, w).astype(np.int64)
            if pi in prev_argmax:
                flips += int((am != prev_argmax[pi]).sum())
                tot += am.size
            prev_argmax[pi] = am
            df = self_orientation_directional_feats(
                coords, am, n_freqs=ndf, freq_across=float(cfg["__cfg_freq_across"]),
                freq_along=float(cfg["__cfg_freq_along"])).astype(np.float32)
            feats[pi] = np.concatenate([curv, df], axis=-1).astype(np.float32)
        if tot:
            flip_frac = flips / tot
    return feats, fkw, flip_frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="EMA npz SNAPSHOT (read-only copy)")
    ap.add_argument("--gt-cache", type=Path, required=True, help="gt cache npz WITH an 'sR' key")
    ap.add_argument("--pairs", type=int, default=12, help="subset K (stride over the cache pairs)")
    ap.add_argument("--pair-list", type=str, default=None,
                    help="explicit comma-separated cache pair indices (resume-after-kill path)")
    ap.add_argument("--hinge-weight", type=float, default=4.0, help="live-config CE hinge weight")
    ap.add_argument("--msal-tau", type=float, default=0.5, help="trainer --margin-saliency-tau")
    ap.add_argument("--msal-target", type=float, default=0.5, help="trainer --margin-saliency-target")
    ap.add_argument("--uniward-beta", type=float, default=4.0,
                    help="trainer --margin-saliency-uniward-beta")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    params, cfg = _load_ckpt(args.ckpt)
    h, w = int(cfg["__render_hw"][0]), int(cfg["__render_hw"][1])
    epoch = int(cfg["__epoch"])

    if "film.weight" in params:
        params = dict(params)
        params["film.weight"] = np.asarray(
            stiefel_project_columns(mx.array(np.asarray(params["film.weight"], np.float32))), np.float32)
    deploy = int8_dequant_params(params)

    bank = CurveletBankConfig(
        n_scales=int(cfg["__bank_n_scales"]), n_orient0=int(cfg["__bank_n_orient0"]),
        f0=float(cfg["__bank_f0"]), base=float(cfg["__bank_base"]), n_iso=int(cfg["__bank_n_iso"]))
    max_bf = cfg.get("__cfg_max_bank_freq")
    B = curvelet_directional_B(bank, max_freq=(float(max_bf) if max_bf is not None else None))
    coords = build_coords(h, w)
    curv = curvelet_feats(coords, B).astype(np.float32)

    gt = np.load(args.gt_cache)
    if "sR" not in gt.files:
        raise ValueError(f"--gt-cache {args.gt_cache} has no 'sR' key; "
                         "build it with tools/precompute_sR_reachability.py first (fails closed).")
    n_cache = int(gt["n_pairs"])
    if args.pair_list:
        pairs = [int(x) for x in args.pair_list.split(",")]
    else:
        stride = max(1, n_cache // args.pairs)
        pairs = list(range(0, n_cache, stride))[: args.pairs]
    lstars = gt["lstars"]
    margins = gt["margins"]
    sR_all = gt["sR"]
    sr_sha = hashlib.sha256(np.ascontiguousarray(sR_all).tobytes()).hexdigest()

    feats, fkw, so_flip = _fixed_point_feats(deploy, cfg, coords, curv, pairs)
    print(json.dumps({"stage": "fixed_point", "pairs": len(pairs), "argmax_flip_frac_iter2": so_flip,
                      "epoch": epoch, "secs": round(time.time() - t0, 1)}), flush=True)

    adapter = load_mlx_distortion_scorer_adapter_from_upstream(_REPO / "upstream", device="cpu")

    variants_regions = {}  # variant -> region -> accumulated grad mass (float64)
    sal_share = {fl: dict.fromkeys(REGIONS, 0.0) for fl in FLAVORS}
    sal_tot = dict.fromkeys(FLAVORS, 0.0)
    term_vals = {fl: [] for fl in FLAVORS}
    base_vals = []

    def _acc(name, gmag, masks):
        d = variants_regions.setdefault(name, dict.fromkeys(REGIONS, 0.0))
        for rname, rmask in masks.items():
            d[rname] += float(gmag[rmask].sum())

    for pi in pairs:
        lstar = np.asarray(lstars[pi], np.int64)
        gt_margin = np.clip(np.asarray(margins[pi], np.float32), 0.0, None)
        sR_pi = np.asarray(sR_all[pi], np.float32)
        rgb, _phi = levelset_rgb_forward_numpy(deploy, feats[pi], deploy["code"][2 * pi + 1], **fkw)
        f_np = rgb.reshape(1, h, w, 3).astype(np.float32)

        isl = np.isin(lstar, ISLAND_CLASSES)
        bnd = _boundary_band(lstar, radius=2) & ~isl
        interior = ~(isl | bnd)
        masks = {"island": isl, "bulk_boundary": bnd, "bulk_interior": interior}

        oh = np.zeros((1, h, w, 5), np.float32)
        for k in range(5):
            oh[0, :, :, k] = (lstar == k)
        oh_mx = mx.array(oh)
        gt_margin_mx = mx.array(gt_margin[None])
        f0 = mx.array(f_np)
        hinge_w = 1.0 + args.hinge_weight * mx.exp(-mx.clip(gt_margin_mx, 0.0, 1e9))
        sal_base_mx = mx.exp(-gt_margin_mx / args.msal_tau)  # (1,H,W) GT fragility (stop-grad const)
        sR_mx = mx.array(sR_pi[None])

        # tex weight at the evaluation point (trainer: stop-grad from the realized R(f) frame).
        r0 = apply_contest_faithful_roundtrip_nhwc(f0, output_hw=(h, w), ste_round=True)
        mx.eval(r0)
        lum = mx.mean(mx.stop_gradient(r0), axis=-1)
        dy = mx.pad(mx.abs(lum[:, 1:, :] - lum[:, :-1, :]), [(0, 0), (0, 1), (0, 0)])
        dx = mx.pad(mx.abs(lum[:, :, 1:] - lum[:, :, :-1]), [(0, 0), (0, 0), (0, 1)])
        tex = dy + dx
        tex = tex / (mx.max(tex) + 1e-6)
        sal_flavor = {
            "plain": sal_base_mx,
            "tex": sal_base_mx / (1.0 + args.uniward_beta * tex),
            "reach": sal_base_mx * sR_mx,
        }
        for fl in FLAVORS:
            s_np = np.asarray(sal_flavor[fl])[0].astype(np.float64)
            sal_tot[fl] += float(s_np.sum())
            for rname, rmask in masks.items():
                sal_share[fl][rname] += float(s_np[rmask].sum())

        def base_loss(f, _oh=oh_mx, _hw=hinge_w):
            r = apply_contest_faithful_roundtrip_nhwc(f, output_hw=(h, w), ste_round=True)
            logits = adapter.segnet(r)
            ce = mx.logsumexp(logits, axis=-1) - mx.sum(logits * _oh, axis=-1)
            return mx.mean(ce * _hw)

        def msal_loss(f, _sal, _oh=oh_mx):
            r = apply_contest_faithful_roundtrip_nhwc(f, output_hw=(h, w), ste_round=True)
            logits = adapter.segnet(r)
            sig_gt = mx.sum(logits * _oh, axis=-1)
            sig_run = mx.max(logits + _oh * (-1e9), axis=-1)
            sgn = sig_gt - sig_run                                   # realized margin (trainer LEVER-4)
            hmap = mx.maximum(args.msal_target - sgn, 0.0) * _sal
            return mx.sum(hmap) / (mx.sum(_sal) + 1e-6)

        vb, gb = mx.value_and_grad(base_loss)(f0)
        mx.eval(vb, gb)
        g_base = np.asarray(gb)[0].astype(np.float64)               # (H,W,3)
        base_vals.append(float(vb))
        g_fl = {}
        for fl in FLAVORS:
            vt, gt_ = mx.value_and_grad(lambda f, _s=sal_flavor[fl]: msal_loss(f, _s))(f0)
            mx.eval(vt, gt_)
            g_fl[fl] = np.asarray(gt_)[0].astype(np.float64)
            term_vals[fl].append(float(vt))

        # Compose every variant EXACTLY via gradient linearity (float64 numpy).
        _acc("base_only", np.abs(g_base).sum(-1), masks)
        for fl in FLAVORS:
            _acc(f"msal_{fl}_term_only", np.abs(g_fl[fl]).sum(-1), masks)
            for wgt in WEIGHTS:
                _acc(f"{fl}_w{wgt:g}", np.abs(g_base + wgt * g_fl[fl]).sum(-1), masks)

        del f0, r0, gb
        mx.clear_cache()
        print(json.dumps({"stage": "pair_done", "pair": pi, "secs": round(time.time() - t0, 1)}),
              flush=True)
        # kill-durable incremental sidecar (focal-harness pattern)
        _inc = args.out_json.with_suffix(".pairs.jsonl")
        _inc.parent.mkdir(parents=True, exist_ok=True)
        with _inc.open("a") as fh:
            fh.write(json.dumps({"pair": pi, "epoch": epoch,
                                 "grad_mass_cum": {v: dict(d) for v, d in variants_regions.items()},
                                 "term_vals_last": {fl: term_vals[fl][-1] for fl in FLAVORS},
                                 "base_val_last": base_vals[-1]}) + "\n")

    out = {
        "axis": "[macOS-CPU/MLX advisory] NON-PROMOTABLE calibration",
        "ckpt": str(args.ckpt), "epoch": epoch, "pairs": pairs,
        "gt_cache": str(args.gt_cache), "sR_sha256": sr_sha,
        "gradient_surface": "witness-alone (deploy/verdict surface; UPPER bound on live composed "
                            "island share — same caveat as the focal probe)",
        "self_orient_fixed_point_flip_frac": so_flip,
        "composition": "grad(base + w*msal) composed EXACTLY as grad(base) + w*grad(msal) "
                       "(gradient linearity; float64)",
        "grad_share": {
            v: {r: d[r] / max(sum(d.values()), 1e-12) for r in REGIONS}
            for v, d in sorted(variants_regions.items())},
        "sal_weight_region_share_analytic": {
            fl: {r: sal_share[fl][r] / max(sal_tot[fl], 1e-12) for r in REGIONS} for fl in FLAVORS},
        "msal_term_mean": {fl: float(np.mean(term_vals[fl])) for fl in FLAVORS},
        "base_ce_mean": float(np.mean(base_vals)),
        "secs": round(time.time() - t0, 1),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({"stage": "done",
                      "grad_share_bulk_boundary": {v: round(out["grad_share"][v]["bulk_boundary"], 4)
                                                   for v in sorted(variants_regions)},
                      "out": str(args.out_json)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
