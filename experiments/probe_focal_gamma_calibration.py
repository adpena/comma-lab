#!/usr/bin/env python
"""$0 focal-gamma calibration probe (council memo
``.omx/research/council_grand_symposium_levelset_loss_geometry_20260705.md`` deliverable 1).

[macOS-CPU/MLX advisory] NON-PROMOTABLE calibration — NOT a score claim. Pointer 0.19110 UNMOVED.

On a live-run EMA checkpoint (read-only snapshot), with real GT (gt_n24 subset) and the real
frozen MLX SegNet through the real R (``apply_contest_faithful_roundtrip_nhwc``), MEASURE:

  (a) the island-pixel (GT classes {1,3} = Lane, Movable) share of the BASE seg-loss gradient
      w.r.t. the rendered frame under the CURRENT loss (live config: seg_form=ce with the
      hinge*exp(-margin) GT-margin weight, hinge=4.0);
  (b) the same under focal reweighting ``(1-p_y)^gamma`` (stop-grad, mean-1 renormalized — the
      exact semantics of the ``--seg-focal-gamma`` build) for gamma in {0.5, 1, 2, 3};
  (c) the d_seg decomposition into islands (1+3) vs bulk on the same checkpoint (per-class flip
      table, D3-style), VALIDATED against the run's logged n600 witness-alone verdict.

GRADIENT SURFACE (traced, honest): the live base CE reads the SEED-COMPOSED frame, but the seed
module is NOT persisted in any checkpoint (verified: no seed keys in levelset_resume_state.npz),
so the composed surface is not reconstructable offline. The measured surface is the
WITNESS-ALONE render == the deploy/verdict surface == the #300 --witness-alone-island-loss
routing surface. Per the plateau-disambiguator mechanism (memo 2026-07-04) the composed-surface
island gradient is STRICTLY SMALLER (the seed satisfies the loss there), so the witness-alone
island share measured here is an UPPER bound on the live composed-surface share.

Reconstruction harness (the validated disambiguator recipe): EMA npz -> optional Stiefel
film.weight re-projection (mirrors ``_project_shadow_film_np``) -> ``int8_dequant_params`` ->
curvelet + self-orient FIXED-POINT feats (2 iterations from the checkpoint's own argmax) ->
``levelset_rgb_forward_numpy`` (THE ONE CODEPATH) -> mx leaf -> R -> frozen MLX SegNet.

Run (memory-light, MLX CPU, never touches the live run's dir except read-only):
  .venv/bin/python experiments/probe_focal_gamma_calibration.py \
      --ckpt <snapshot>/ema_ep50.npz \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
      --pairs 12 --out-json <snapshot>/focal_calib_ep50.json
"""

from __future__ import annotations

import argparse
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

# MLX CPU: the probe must not contend with the live run's GPU stream, and MLX-GPU is not
# bit-identical cross-process (memory ``mlx_gpu_not_bit_identical_crossprocess...``).
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
GAMMAS = (0.5, 1.0, 2.0, 3.0)


def _load_ckpt(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(path, allow_pickle=True)
    params = {k: np.asarray(z[k]) for k in z.files
              if not k.startswith("__") and not k.startswith("pose_carrier.")}
    cfg = {k: z[k][()] if z[k].shape == () else np.asarray(z[k]) for k in z.files if k.startswith("__")}
    return params, cfg


def _fixed_point_feats(deploy, cfg, coords, curv, P_pairs, pairs, iters=2):
    """Self-orient fixed-point reconstruction (the validated disambiguator recipe): start with
    zero dir feats -> forward -> argmax -> tangent dir feats -> repeat. Returns per-pair feats +
    the between-iteration argmax flip fraction (stability check)."""
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
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=12, help="subset K (stride over the cache pairs)")
    ap.add_argument("--pair-list", type=str, default=None,
                    help="explicit comma-separated cache pair indices (overrides --pairs stride; "
                    "resume-after-kill path)")
    ap.add_argument("--hinge-weight", type=float, default=4.0, help="live-config CE hinge weight")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    params, cfg = _load_ckpt(args.ckpt)
    h, w = int(cfg["__render_hw"][0]), int(cfg["__render_hw"][1])
    epoch = int(cfg["__epoch"])

    # Mirror _project_shadow_film_np (live run has --film-stiefel) then the int8 deploy roundtrip.
    if "film.weight" in params:
        params = dict(params)
        params["film.weight"] = np.asarray(
            stiefel_project_columns(mx.array(np.asarray(params["film.weight"], np.float32))), np.float32)
    deploy = int8_dequant_params(params)

    # Front end (from the checkpoint's own cfg — never guessed).
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
    lstars = gt["lstars"]     # (P, H, W) int
    margins = gt["margins"]   # (P, H, W) float

    feats, fkw, so_flip = _fixed_point_feats(deploy, cfg, coords, curv, n_cache, pairs)
    print(json.dumps({"stage": "fixed_point", "pairs": len(pairs), "argmax_flip_frac_iter2": so_flip,
                      "epoch": epoch, "secs": round(time.time() - t0, 1)}), flush=True)

    adapter = load_mlx_distortion_scorer_adapter_from_upstream(_REPO / "upstream", device="cpu")

    variants = ["current"] + [f"focal_g{g:g}" for g in GAMMAS]
    regions = ["island", "bulk_boundary", "bulk_interior"]
    grad_mass = {v: dict.fromkeys(regions, 0.0) for v in variants}
    weight_share_isl = {f"focal_g{g:g}": [] for g in GAMMAS}  # analytic share_isl(gamma) readback
    flip_counts = dict.fromkeys(regions, 0)
    px_counts = dict.fromkeys(regions, 0)
    per_class_flip = np.zeros(5, np.int64)
    per_class_area = np.zeros(5, np.int64)
    n_flip_tot, n_px_tot = 0, 0

    for pi in pairs:
        lstar = np.asarray(lstars[pi], np.int64)
        margin = np.clip(np.asarray(margins[pi], np.float32), 0.0, None)
        rgb, _phi = levelset_rgb_forward_numpy(deploy, feats[pi], deploy["code"][2 * pi + 1], **fkw)
        f_np = rgb.reshape(1, h, w, 3).astype(np.float32)

        isl = np.isin(lstar, ISLAND_CLASSES)
        bnd = _boundary_band(lstar, radius=2) & ~isl
        interior = ~(isl | bnd)
        region_masks = {"island": isl, "bulk_boundary": bnd, "bulk_interior": interior}

        oh = np.zeros((1, h, w, 5), np.float32)
        for k in range(5):
            oh[0, :, :, k] = (lstar == k)
        oh_mx = mx.array(oh)
        margin_mx = mx.array(margin[None])
        f0 = mx.array(f_np)

        # Base logits AT the evaluation point (for the focal weight + the d_seg decomposition).
        r0 = apply_contest_faithful_roundtrip_nhwc(f0, output_hw=(h, w), ste_round=True)
        logits0 = adapter.segnet(r0)
        mx.eval(logits0)
        am0 = np.asarray(mx.argmax(logits0, axis=-1))[0]
        flip = am0 != lstar
        n_flip_tot += int(flip.sum())
        n_px_tot += flip.size
        for rname, rmask in region_masks.items():
            flip_counts[rname] += int(flip[rmask].sum())
            px_counts[rname] += int(rmask.sum())
        for k in range(5):
            per_class_flip[k] += int(flip[lstar == k].sum())
            per_class_area[k] += int((lstar == k).sum())

        # Focal weights at the fixed point (stop-grad constants — the build's exact semantics).
        p_y = np.asarray(mx.exp(mx.sum(logits0 * oh_mx, axis=-1) - mx.logsumexp(logits0, axis=-1)))[0]
        focal_w = {"current": None}
        for g in GAMMAS:
            fw = np.power(np.clip(1.0 - p_y, 1e-12, 1.0), g).astype(np.float32)
            weight_share_isl[f"focal_g{g:g}"].append(float(fw[isl].sum() / fw.sum()))
            focal_w[f"focal_g{g:g}"] = mx.array((fw / (fw.mean() + 1e-8))[None])

        hinge_w = 1.0 + args.hinge_weight * mx.exp(-mx.clip(margin_mx, 0.0, 1e9))

        for v in variants:
            fw = focal_w[v]

            def seg_loss(f, _fw=fw, _oh=oh_mx, _hw=hinge_w):
                r = apply_contest_faithful_roundtrip_nhwc(f, output_hw=(h, w), ste_round=True)
                logits = adapter.segnet(r)
                ce = mx.logsumexp(logits, axis=-1) - mx.sum(logits * _oh, axis=-1)
                pw = ce * _hw
                if _fw is not None:
                    pw = pw * _fw
                return mx.mean(pw)

            g_f = mx.grad(seg_loss)(f0)
            mx.eval(g_f)
            gmag = np.abs(np.asarray(g_f))[0].sum(-1)  # (H, W) per-pixel |grad| mass
            for rname, rmask in region_masks.items():
                grad_mass[v][rname] += float(gmag[rmask].sum())
        del f0, logits0, r0
        mx.clear_cache()
        print(json.dumps({"stage": "pair_done", "pair": pi,
                          "secs": round(time.time() - t0, 1)}), flush=True)
        # INCREMENTAL per-pair sidecar (kill-durable; no signal loss on an external reap): append
        # this pair's raw region sums so a partial run is fully reconstructable/mergeable.
        _inc = args.out_json.with_suffix(".pairs.jsonl")
        _inc.parent.mkdir(parents=True, exist_ok=True)
        with _inc.open("a") as fh:
            fh.write(json.dumps({
                "pair": pi, "epoch": epoch,
                "grad_mass_pair": {v: {r: float(grad_mass[v][r]) for r in regions} for v in variants},
                "flip_counts_cum": dict(flip_counts), "px_counts_cum": dict(px_counts),
                "n_flip_tot": n_flip_tot, "n_px_tot": n_px_tot}) + "\n")

    d_seg_subset = n_flip_tot / max(n_px_tot, 1)
    out = {
        "axis": "[macOS-CPU/MLX advisory] NON-PROMOTABLE calibration",
        "ckpt": str(args.ckpt), "epoch": epoch, "pairs": pairs,
        "gradient_surface": "witness-alone (deploy/verdict surface; seed not persisted in ckpts "
                            "=> composed surface unreconstructable; wa share is an UPPER bound on "
                            "the live composed-surface share)",
        "self_orient_fixed_point_flip_frac": so_flip,
        "grad_share": {
            v: {r: grad_mass[v][r] / max(sum(grad_mass[v].values()), 1e-12) for r in regions}
            for v in variants},
        "focal_weight_island_share_analytic": {
            k: float(np.mean(vv)) for k, vv in weight_share_isl.items()},
        "d_seg_subset": d_seg_subset,
        "d_seg_regions": {r: {"flip_frac_within": flip_counts[r] / max(px_counts[r], 1),
                              "share_of_flips": flip_counts[r] / max(n_flip_tot, 1),
                              "px_share": px_counts[r] / max(n_px_tot, 1)} for r in regions},
        "per_class": {str(k): {"area": per_class_area[k] / max(n_px_tot, 1),
                               "within_flip": per_class_flip[k] / max(per_class_area[k], 1),
                               "share_of_d_seg": per_class_flip[k] / max(n_flip_tot, 1)}
                      for k in range(5)},
        "secs": round(time.time() - t0, 1),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({"stage": "done", "d_seg_subset": round(d_seg_subset, 6),
                      "grad_share_island": {v: round(out["grad_share"][v]["island"], 4)
                                            for v in variants},
                      "out": str(args.out_json)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
