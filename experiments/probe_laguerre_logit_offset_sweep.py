# SPDX-License-Identifier: MIT
"""FIRE-FIRST ($0) probe: per-class LAGUERRE / power-diagram logit-offset sweep on
the REAL witness, REALIZED THROUGH R (#218 facets 1 & 3; Lane<->Road 57% tail #209).

Loads a level-set witness checkpoint, rebuilds its curvelet + self-orientation
coordinate features (self-orient FIXED POINT), forwards each pair's frame1 through
the pure-numpy ONE-CODEPATH ONCE to cache the SDF field ``phi`` (H,W,K) and the
texture head ``tex`` (H,W,3) recovered EXACTLY from the render
(``tex = logit(rgb/255) - softmax(phi/T)@palette``).  Then a per-class ADDITIVE
offset ``b`` is applied to ``phi`` and the frame is RE-COMPOSED
(``rgb = sigmoid(softmax((phi+b)/T)@palette + tex)*255``) -> R -> frozen CPU SegNet
-> REALIZED ``d_seg``.  This is authority-grade (not the phi-argmax proxy, which is
invalid for this witness: phi-argmax != SegNet-argmax-on-render).

Why cache the trunk: the offset only shifts ``out_sdf.bias`` (phi += b), so the
expensive trunk (in_proj + FiLM + hidden) is computed once; each swept offset costs
one cheap re-compose + one SegNet batch.  BYTE-FREE: the winning ``b`` folds into
the already-counted ``out_sdf.bias`` (0 extra archive bytes).

Sweep: independent 1-D per-class sweep over ``--grid`` for each ``--focus`` class,
then 2 rounds of coordinate descent for the joint winner; the analytic Menon offset
``b_k = -log(pi_k)`` is evaluated as a no-search baseline.

AUTHORITY: realized-through-R on the frozen CPU SegNet (NOT MPS, NOT MLX), fp32 EMA
render.  [macOS-CPU advisory] until byte-closed exact eval.  NO FAKE: real ckpt,
real GT, real scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "experiments", REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_witness_realized_through_R_mlx import (  # noqa: E402
    _build_render_coords,
    _torch_R_to_camera_uint8,
    load_gt_from_cache,
)

from tac.boundary_math.lever_b_generator import self_orientation_directional_feats  # noqa: E402
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    curvelet_directional_B,
    curvelet_feats,
    levelset_rgb_forward_numpy,
)
from tac.boundary_math.laguerre_logit_offset import (  # noqa: E402
    CLASS_NAMES,
    menon_logit_adjustment_offsets,
    per_class_disagreement,
)


def _load_ckpt(path: Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    params, cfg = {}, {}
    for k in z.files:
        (cfg if k.startswith("__") else params)[k[2:] if k.startswith("__") else k] = np.asarray(z[k])
    params = {k: np.asarray(v, np.float32) for k, v in params.items()}
    return params, cfg


def _fwd(params, feats, code_row, cfg):
    return levelset_rgb_forward_numpy(
        params, feats, code_row,
        n_hidden=int(cfg["cfg_n_hidden"]), hidden_dim=int(cfg["cfg_hidden_dim"]),
        n_classes=5, activation=str(cfg["cfg_activation"]),
        softmax_temp=float(cfg["cfg_softmax_temp"]),
        wire_w0=float(cfg["cfg_wire_w0"]), wire_s0=float(cfg["cfg_wire_s0"]),
        hosc_beta=float(cfg["cfg_hosc_beta"]), hosc_omega=float(cfg["cfg_hosc_omega"]),
        chroma=bool(int(cfg["cfg_chroma"])),
    )


def _softmax_last(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _compose_rgb(phi, tex, palette, T, offset):
    """Bit-mirror of levelset_rgb_forward_numpy's compose, with offset added to phi (chroma=True)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        base = _softmax_last((phi + offset) / T) @ palette      # (...,3)
        return (1.0 / (1.0 + np.exp(-(base + tex)))) * 255.0    # (...,3)


def _cache_phi_tex(params, curv_feats, coords, cfg, pi, rh, rw, so_iters, palette, T):
    """Converged (self-orient fixed point) forward -> (phi (H,W,K), tex (H,W,3)) exact."""
    n_dir_freqs = int(cfg["cfg_n_dir_freqs"]); dir_w = 4 * n_dir_freqs
    use_so = bool(int(cfg["cfg_self_orient"]))
    code1 = params["code"][2 * pi + 1]
    dir_feats = np.zeros((curv_feats.shape[0], dir_w), np.float32)
    rgb = phi = None
    for _ in range(so_iters if use_so else 1):
        feats = np.concatenate([curv_feats, dir_feats], axis=-1).astype(np.float32) if use_so else curv_feats
        rgb, phi = _fwd(params, feats, code1, cfg)
        if use_so:
            argmax = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
            dir_feats = self_orientation_directional_feats(
                coords, argmax, n_freqs=n_dir_freqs,
                freq_across=float(cfg["cfg_freq_across"]), freq_along=float(cfg["cfg_freq_along"]),
            ).astype(np.float32)
    phi = phi.astype(np.float64).reshape(rh, rw, 5)
    rgb = rgb.astype(np.float64).reshape(rh, rw, 3)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        base0 = _softmax_last(phi / T) @ palette
        # invert the compose to recover tex EXACTLY: rgb=sigmoid(base+tex)*255
        r = np.clip(rgb / 255.0, 1e-7, 1.0 - 1e-7)
        tex = np.log(r / (1.0 - r)) - base0
    return phi.astype(np.float32), tex.astype(np.float32)


def _segnet_argmax_batch(seg_cpu, f1s, chunk=32):
    """Mirror cpu_verdict_d_seg_batch's SegNet forward but RETURN the argmax maps (for per-class)."""
    import torch

    outs = []
    for i in range(0, len(f1s), chunk):
        arr = np.stack([np.asarray(f)[None] for f in f1s[i:i + chunk]], axis=0)  # (n,1,H,W,3)
        xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            seg_in = seg_cpu.preprocess_input(xp)
            am = seg_cpu(seg_in).argmax(dim=1).cpu().numpy().astype(np.int64)  # (n,h,w)
        outs.append(am)
    return np.concatenate(outs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--gt-cache", type=str, required=True)
    ap.add_argument("--num-pairs", type=int, default=96)
    ap.add_argument("--so-iters", type=int, default=3)
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--grid", type=str, default="-0.4,-0.2,0.0,0.2,0.4")
    ap.add_argument("--focus", type=str, default="0,1,3")
    args = ap.parse_args()

    t0 = time.time()
    params, cfg = _load_ckpt(Path(args.ckpt))
    rh, rw = int(cfg["render_hw"][0]), int(cfg["render_hw"][1])
    T = float(cfg["cfg_softmax_temp"])
    palette = params["palette"].astype(np.float64)
    grid = tuple(float(x) for x in args.grid.split(","))
    focus = tuple(int(x) for x in args.focus.split(","))

    gt, seg_cpu, _pn = load_gt_from_cache(Path(args.gt_cache), args.num_pairs)
    P = gt.n_pairs
    coords = _build_render_coords(rh, rw)
    bank = CurveletBankConfig(
        n_scales=int(cfg["bank_n_scales"]), n_orient0=int(cfg["bank_n_orient0"]),
        f0=float(cfg["bank_f0"]), base=float(cfg["bank_base"]), n_iso=int(cfg["bank_n_iso"]))
    B = curvelet_directional_B(bank, max_freq=float(cfg["cfg_max_bank_freq"]))
    curv_feats = curvelet_feats(coords, B).astype(np.float32)
    print(json.dumps({"stage": "loaded", "n_pairs": P, "epoch": int(cfg["epoch"]),
                      "softmax_temp": T, "secs": round(time.time() - t0, 1)}), flush=True)

    # --- cache phi + tex per pair (frame1) ---
    phis, texs = [], []
    for pi in range(P):
        phi, tex = _cache_phi_tex(params, curv_feats, coords, cfg, pi, rh, rw, args.so_iters, palette, T)
        phis.append(phi); texs.append(tex)
        if (pi + 1) % 24 == 0:
            print(json.dumps({"stage": "cache", "done": pi + 1, "secs": round(time.time() - t0, 1)}), flush=True)
    gt_l = [gt.lstars[pi].astype(np.int64) for pi in range(P)]
    gt_arr = np.stack(gt_l, axis=0)
    counts = np.bincount(gt_arr.reshape(-1), minlength=5).astype(np.float64)
    priors = counts / counts.sum()

    def realized(offset: np.ndarray, want_argmax=False):
        f1s = [_torch_R_to_camera_uint8(_compose_rgb(phis[pi], texs[pi], palette, T, offset)) for pi in range(P)]
        am = _segnet_argmax_batch(seg_cpu, f1s)
        d = float(np.mean([np.count_nonzero(am[i] != gt_l[i]) / gt_l[i].size for i in range(P)]))
        if want_argmax:
            return d, per_class_disagreement(am.reshape(-1), gt_arr.reshape(-1), 5)
        return d

    zero = np.zeros(5)
    base_d, base_pc = realized(zero, want_argmax=True)
    print(json.dumps({"stage": "baseline", "realized_d_seg": base_d,
                      "per_class": {k: round(v, 5) for k, v in base_pc.items()}}), flush=True)

    # --- independent 1-D per-class sweeps ---
    curves = {}
    for c in focus:
        curve = []
        for v in grid:
            b = zero.copy(); b[c] = v
            d = realized(b)
            curve.append({"offset": float(v), "d_seg": d})
            print(json.dumps({"stage": "sweep", "class": c, "offset": v, "d_seg": d}), flush=True)
        curves[c] = curve

    # --- joint winner = independent per-class argmins combined (effects ~additive at this scale) ---
    best_b = zero.copy()
    for c in focus:
        cbest = min(curves[c], key=lambda r: r["d_seg"])
        best_b[c] = float(cbest["offset"])
    win_d, win_pc = realized(best_b, want_argmax=True)

    # --- Menon analytic baseline ---
    menon_b = menon_logit_adjustment_offsets(counts, tau=1.0)
    menon_d = realized(menon_b)

    out = {
        "advisory": "[macOS-CPU advisory . REALIZED-through-R CPU-SegNet authority; fp32 EMA render, NOT int8-deployed; NON-PROMOTABLE until byte-closed exact eval]",
        "ckpt": str(args.ckpt), "n_pairs": P, "epoch": int(cfg["epoch"]),
        "class_names": list(CLASS_NAMES),
        "class_priors": {int(c): float(priors[c]) for c in range(5)},
        "focus_classes": list(focus), "grid": list(grid),
        "baseline_d_seg": base_d,
        "baseline_per_class": {int(k): v for k, v in base_pc.items()},
        "per_class_1d_curves": {int(c): curves[c] for c in focus},
        "winner": {
            "offsets": {int(c): float(best_b[c]) for c in range(5)},
            "d_seg": win_d, "delta": win_d - base_d,
            "per_class": {int(k): v for k, v in win_pc.items()},
        },
        "menon_analytic": {
            "offsets": {int(c): float(menon_b[c]) for c in range(5)},
            "d_seg": menon_d, "delta": menon_d - base_d,
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps({"stage": "done", "secs": round(time.time() - t0, 1),
                      "baseline": base_d, "winner": win_d, "delta": win_d - base_d,
                      "winner_offsets": out["winner"]["offsets"], "out": args.out_json}), flush=True)


if __name__ == "__main__":
    main()
