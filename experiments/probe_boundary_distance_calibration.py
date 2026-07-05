#!/usr/bin/env python
"""$0 boundary-distance-weight calibration probe (CE-window intervention pre-stage,
sister of ``probe_focal_gamma_calibration.py`` — council levelset-loss-geometry memo,
``--boundary-distance-weight`` build 535e142be).

[macOS-CPU/MLX advisory] NON-PROMOTABLE calibration — NOT a score claim. Pointer 0.19110 UNMOVED.

On a live-run EMA checkpoint (read-only snapshot), with real GT (gt_n24 subset) and the real
frozen MLX SegNet through the real R, MEASURE for w_bd in {0, 0.01, 0.05, 0.1, 0.5, 1.0}:

  (a) the island / bulk-boundary / bulk-interior shares of the TOTAL-loss gradient on the
      PHI (SDF-field) surface — the ONLY common surface: the bd term reads ``model.sdf``
      directly (zero gradient w.r.t. the rendered frame), so the frame-surface shares of the
      focal probe structurally CANNOT see it. The phi surface is the contour DOF the witness
      owns (Mallat move-the-contour) and both terms differentiate through it: CE via
      softmax->palette->sigmoid->render->R->SegNet, bd directly.
  (b) the raw magnitude of the bd term vs the live-config base CE seg term
      (w_seg*mean(ce*hinge_w), w_seg=100, hinge=4.0) => the ratio w*BD/(CE_term + w*BD),
      so the chosen weight neither vanishes nor dominates.
  (c) EMA-vs-LIVE seg-term diagnostic: the same witness-alone CE computed with the LIVE
      weights from the resume sidecar (``liveP__*``) — measures whether the ep92+ spike-guard
      deadlock regime corresponds to a seg-surface divergence of the live weights or not.

The bd band map + term are IMPORTED FROM THE TRAINER (``boundary_distance_band_map`` /
``boundary_distance_term_mlx``) — the build's exact semantics, never re-implemented.

Reconstruction harness: the validated disambiguator recipe via probe_focal_gamma_calibration
(_load_ckpt/_fixed_point_feats), plus an EXACT phi-leaf re-render: the numpy forward returns
(rgb, phi) with rgb = sigmoid(softmax(phi/T)@palette + tex)*255; tex is recovered exactly as
logit(rgb/255) - softmax(phi/T)@palette so the MLX phi-leaf graph reproduces the SAME frame
(validated per pair, reported as ``rgb_recon_max_abs``).

Run (memory-light, MLX CPU, chunked + kill-durable; never writes into the live run dir):
  .venv/bin/python experiments/probe_boundary_distance_calibration.py \
      --ckpt experiments/results/bd_calib_20260705/snap/ema_BEST_ep100.npz \
      --resume-npz experiments/results/bd_calib_20260705/snap/resume_state_ep100.npz \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
      --pair-list 0,2,4 --out-json experiments/results/bd_calib_20260705/bd_calib_ep100.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx

# MLX CPU: never contend with the live run's GPU stream; MLX-GPU is not bit-identical
# cross-process (memory mlx_gpu_not_bit_identical_crossprocess...).
mx.set_default_device(mx.cpu)

from probe_focal_gamma_calibration import _fixed_point_feats, _load_ckpt
from train_levelset_witness_realized_through_R_mlx import (
    boundary_distance_band_map,
    boundary_distance_term_mlx,
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
BD_WEIGHTS = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0)
W_SEG = 100.0  # live config --w-seg 100


def recover_tex_effective(rgb: np.ndarray, phi: np.ndarray, palette: np.ndarray,
                          softmax_temp: float) -> np.ndarray:
    """EXACT inverse of the deploy render's post-phi path: rgb = sigmoid(base+tex)*255 with
    base = softmax(phi/T) @ palette  =>  tex = logit(rgb/255) - base. float64; the returned
    tex_eff reproduces rgb bit-close when re-composed with the SAME phi (validated by caller)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # mirrors the numpy forward
        p = np.clip(np.asarray(rgb, np.float64) / 255.0, 1e-9, 1.0 - 1e-9)
        pre = np.log(p / (1.0 - p))
        z = np.asarray(phi, np.float64) / float(softmax_temp)
        z = z - z.max(axis=-1, keepdims=True)
        soft = np.exp(z)
        soft = soft / soft.sum(axis=-1, keepdims=True)
        base = soft @ np.asarray(palette, np.float64)
        tex = (pre - base).astype(np.float32)
    if not np.isfinite(tex).all():  # NO-FAKE: a nan-poisoned tex would corrupt every grad sum
        raise ValueError("recover_tex_effective produced non-finite tex (checkpoint phi non-finite?)")
    return tex


def bd_ratio(ce_term: float, bd_raw: float, w: float) -> float:
    """Share of the (CE + w*BD) total carried by the bd term at weight w."""
    tot = ce_term + w * bd_raw
    return (w * bd_raw / tot) if tot > 0 else 0.0


def _load_live_params(resume_npz: Path) -> dict[str, np.ndarray]:
    """LIVE (non-EMA) witness params from the resume sidecar: strip the ``liveP__`` prefix,
    drop optimizer/cfg/pose-carrier keys. NO-FAKE: raises if no liveP keys found."""
    z = np.load(resume_npz, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in z.files:
        if not k.startswith("liveP__"):
            continue
        name = k[len("liveP__"):]
        if name.startswith("pose_carrier."):
            continue
        out[name] = np.asarray(z[k])
    if "in_proj.weight" not in out:
        raise ValueError(f"{resume_npz} has no liveP__in_proj.weight — not a levelset resume sidecar?")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="EMA npz SNAPSHOT (read-only copy)")
    ap.add_argument("--resume-npz", type=Path, default=None,
                    help="resume-state npz SNAPSHOT for the LIVE-vs-EMA CE diagnostic (optional)")
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--pair-list", type=str, default=None,
                    help="explicit comma-separated cache pair indices (chunked/kill-durable path)")
    ap.add_argument("--hinge-weight", type=float, default=4.0, help="live-config CE hinge weight")
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated w_bd override (default: the standard grid "
                    + ",".join(f"{v:g}" for v in BD_WEIGHTS) + ")")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    bd_weights = (tuple(float(x) for x in args.weights.split(",")) if args.weights else BD_WEIGHTS)

    t0 = time.time()
    params, cfg = _load_ckpt(args.ckpt)
    h, w = int(cfg["__render_hw"][0]), int(cfg["__render_hw"][1])
    epoch = int(cfg["__epoch"])
    temp = float(cfg["__cfg_softmax_temp"])

    def _deploy_of(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(raw)
        if "film.weight" in d:  # mirror _project_shadow_film_np (live run has --film-stiefel)
            d["film.weight"] = np.asarray(
                stiefel_project_columns(mx.array(np.asarray(d["film.weight"], np.float32))), np.float32)
        return int8_dequant_params(d)

    deploy = _deploy_of(params)
    live_deploy = None
    if args.resume_npz is not None:
        live_deploy = _deploy_of(_load_live_params(args.resume_npz))

    bank = CurveletBankConfig(
        n_scales=int(cfg["__bank_n_scales"]), n_orient0=int(cfg["__bank_n_orient0"]),
        f0=float(cfg["__bank_f0"]), base=float(cfg["__bank_base"]), n_iso=int(cfg["__bank_n_iso"]))
    max_bf = cfg.get("__cfg_max_bank_freq")
    B = curvelet_directional_B(bank, max_freq=(float(max_bf) if max_bf is not None else None))
    coords = build_coords(h, w)
    curv = curvelet_feats(coords, B).astype(np.float32)

    gt = np.load(args.gt_cache)
    n_cache = int(gt["n_pairs"])
    if args.pair_list:
        pairs = [int(x) for x in args.pair_list.split(",")]
    else:
        stride = max(1, n_cache // args.pairs)
        pairs = list(range(0, n_cache, stride))[: args.pairs]
    lstars = gt["lstars"]
    margins = gt["margins"]

    feats, fkw, so_flip = _fixed_point_feats(deploy, cfg, coords, curv, n_cache, pairs)
    print(json.dumps({"stage": "fixed_point", "pairs": len(pairs), "epoch": epoch,
                      "argmax_flip_frac_iter2": so_flip, "secs": round(time.time() - t0, 1)}), flush=True)

    adapter = load_mlx_distortion_scorer_adapter_from_upstream(_REPO / "upstream", device="cpu")
    palette_mx = mx.array(np.asarray(deploy["palette"], np.float32))

    variants = [f"w{wv:g}" for wv in bd_weights]
    regions = ["island", "bulk_boundary", "bulk_interior"]
    grad_mass = {v: dict.fromkeys(regions, 0.0) for v in variants}
    ce_terms: list[float] = []
    bd_raws: list[float] = []
    ce_terms_live: list[float] = []
    rgb_recon_max_abs = 0.0

    for pi in pairs:
        lstar = np.asarray(lstars[pi], np.int64)
        margin = np.clip(np.asarray(margins[pi], np.float32), 0.0, None)
        rgb, phi = levelset_rgb_forward_numpy(deploy, feats[pi], deploy["code"][2 * pi + 1], **fkw)
        tex_eff = recover_tex_effective(rgb, phi, deploy["palette"], temp)

        isl = np.isin(lstar, ISLAND_CLASSES)
        bnd = _boundary_band(lstar, radius=2) & ~isl
        interior = ~(isl | bnd)
        region_masks = {"island": isl, "bulk_boundary": bnd, "bulk_interior": interior}

        oh = np.zeros((1, h, w, 5), np.float32)
        for k in range(5):
            oh[0, :, :, k] = (lstar == k)
        oh_mx = mx.array(oh)
        hinge_w = 1.0 + args.hinge_weight * mx.exp(-mx.clip(mx.array(margin[None]), 0.0, 1e9))
        band_mx = mx.array(boundary_distance_band_map(lstar))  # trainer's exact band (H,W)
        tex_mx = mx.array(tex_eff)
        phi0 = mx.array(np.asarray(phi, np.float32))

        def ce_term_of(phi_leaf, _tex=tex_mx, _oh=oh_mx, _hw=hinge_w):
            soft = mx.softmax(phi_leaf / temp, axis=-1)
            pre = soft @ palette_mx + _tex
            frame = mx.reshape(mx.sigmoid(pre) * 255.0, (1, h, w, 3))
            r = apply_contest_faithful_roundtrip_nhwc(frame, output_hw=(h, w), ste_round=True)
            logits = adapter.segnet(r)
            ce = mx.logsumexp(logits, axis=-1) - mx.sum(logits * _oh, axis=-1)
            return W_SEG * mx.mean(ce * _hw), frame

        # Validate the phi-leaf re-render against the numpy deploy frame (same codepath check).
        _ce0, frame0 = ce_term_of(phi0)
        mx.eval(frame0)
        rgb_recon_max_abs = max(rgb_recon_max_abs,
                                float(np.max(np.abs(np.asarray(frame0).reshape(-1, 3) - rgb))))
        ce_val = float(_ce0)
        bd_val = float(boundary_distance_term_mlx(phi0, oh_mx, band_mx, h, w))
        ce_terms.append(ce_val)
        bd_raws.append(bd_val)

        if live_deploy is not None:
            rgb_l, phi_l = levelset_rgb_forward_numpy(
                live_deploy, feats[pi], live_deploy["code"][2 * pi + 1], **fkw)
            tex_l = recover_tex_effective(rgb_l, phi_l, live_deploy["palette"], temp)
            _ce_l, _ = ce_term_of(mx.array(np.asarray(phi_l, np.float32)), _tex=mx.array(tex_l))
            ce_terms_live.append(float(_ce_l))

        for wv in bd_weights:
            def total_loss(phi_leaf, _wv=wv, _ce_of=ce_term_of, _oh=oh_mx, _band=band_mx):
                ce, _ = _ce_of(phi_leaf)
                if _wv > 0.0:
                    ce = ce + _wv * boundary_distance_term_mlx(phi_leaf, _oh, _band, h, w)
                return ce

            g = mx.grad(total_loss)(phi0)
            mx.eval(g)
            gmag = np.abs(np.asarray(g)).sum(-1).reshape(h, w)
            for rname, rmask in region_masks.items():
                grad_mass[f"w{wv:g}"][rname] += float(gmag[rmask].sum())
        mx.clear_cache()
        print(json.dumps({"stage": "pair_done", "pair": pi, "ce_term": round(ce_val, 4),
                          "bd_raw": round(bd_val, 4),
                          "ce_term_live": (round(ce_terms_live[-1], 4) if ce_terms_live else None),
                          "secs": round(time.time() - t0, 1)}), flush=True)
        # kill-durable per-pair sidecar (mergeable partial runs — no signal loss on a reap)
        _inc = args.out_json.with_suffix(".pairs.jsonl")
        _inc.parent.mkdir(parents=True, exist_ok=True)
        with _inc.open("a") as fh:
            fh.write(json.dumps({
                "pair": pi, "epoch": epoch, "ce_term": ce_val, "bd_raw": bd_val,
                "ce_term_live": (ce_terms_live[-1] if ce_terms_live else None),
                "grad_mass_pair_cum": {v: {r: float(grad_mass[v][r]) for r in regions}
                                       for v in variants}}) + "\n")

    ce_mean = float(np.mean(ce_terms))
    bd_mean = float(np.mean(bd_raws))
    out = {
        "axis": "[macOS-CPU/MLX advisory] NON-PROMOTABLE calibration",
        "ckpt": str(args.ckpt), "epoch": epoch, "pairs": pairs,
        "gradient_surface": "PHI (SDF-field) leaf — the only surface common to CE (via "
                            "softmax->palette->sigmoid->R->SegNet) and the bd term (direct); "
                            "witness-alone deploy recon (seed not persisted in ckpts)",
        "rgb_recon_max_abs": rgb_recon_max_abs,
        "self_orient_fixed_point_flip_frac": so_flip,
        "ce_term_mean": ce_mean, "bd_raw_mean": bd_mean,
        "ce_term_live_mean": (float(np.mean(ce_terms_live)) if ce_terms_live else None),
        "bd_ratio_of_total": {f"w{wv:g}": bd_ratio(ce_mean, bd_mean, wv) for wv in bd_weights},
        "grad_share": {v: {r: grad_mass[v][r] / max(sum(grad_mass[v].values()), 1e-12)
                           for r in regions} for v in variants},
        "secs": round(time.time() - t0, 1),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({"stage": "done", "ce_term_mean": round(ce_mean, 4),
                      "bd_raw_mean": round(bd_mean, 4),
                      "grad_share_bulk_boundary": {v: round(out["grad_share"][v]["bulk_boundary"], 4)
                                                   for v in variants},
                      "out": str(args.out_json)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
