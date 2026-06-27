#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""TOPOLOGY-NATIVE witness data-viz: level-set render + Morse-Smale critical-point complex +
Voronoi/level-set cells + dimensionality (code SVD eff-rank) + spacetime event stats, played
in realtime next to the actual contest video.  (operator vision 2026-06-27.)

MAX-OBSERVABILITY TOOL (CLAUDE.md "Max observability into behavior"): it makes the MEASURED
witness/partition structure VISIBLE. It is a TOOL = a MEANS; it moves NO score pointer (the exact
contest frontier is pointer-only). NO-FAKE: every panel shows a REAL measured field —
  * the contest frame is decoded via the canonical ``frame_utils.yuv420_to_rgb`` path (the gt cache
    ``gt_f1`` was built with it; NEVER PyAV rgb24 which manufactures ~100x phantom pose);
  * the SegNet L* argmax + per-pixel margin are the cached frozen-CPU-torch SegNet outputs;
  * the witness partition (panel) is the ACTUAL ``argmax_k phi_k`` of the level-set generator's
    forward on the trained EMA weights (op-for-op the shipped numpy/torch inflate). If the witness
    npz is unconverged it is shown HONESTLY (the disagreement number is reported, not hidden);
  * the Morse-Smale criticals / separatrices / Voronoi cells are computed from the REAL fields.

SCOPE: topology-FIRST. A sister viz owns the canonical comma 6-panel recon-vs-error (GT|recon|
pixel-error / GT-masks|our-masks|SegNet-disagreement) — this tool does NOT duplicate it.

Morse function: m(x) = phi_top1 - phi_top2  (the per-pixel margin; m>=0, ->0 on a class boundary).
  * index-0 MINIMA of m  = deepest boundary points (most precarious; where d_seg lives);
  * index-1 SADDLES of m = TRIPLE JUNCTIONS (3 classes meet: phi_top1~phi_top2~phi_top3); the
    Hessian eigenvectors are drawn (the unstable +lambda direction is the local separatrix tangent);
  * index-2 MAXIMA of m  = each class cell's confident "capital" (interior argmax-confidence peak).
The SEPARATRIX 1-skeleton (margin-zero curves connecting saddles<->extrema) = the partition
boundary graph = the Morse-Smale complex.  Two Morse functions are available and labelled:
  - the GT SegNet logit margin (cached ``margins``) — the authority precariousness field;
  - the GT signed-distance margin (top1-top2 of ``signed_distance_fields(L*)``) — smooth, so its
    critical-point taxonomy / Hessian classification is well-posed (used for the complex panel).

PALETTE: canonical comma10k (operator 2026-06-27) — 0 Road #402020, 1 Lane #ff0000,
2 Undrivable #808060, 3 Movable #00ff66, 4 MyCar #cc00ff. Class ORDER is the MEASURED comma10k
order (CLAUDE.md 2026-06-27); we deliberately do NOT use the repo's luma-sorted
``SEGNET_CLASS_NAMES`` (the wrong order that bit us 3x) nor luma-sort.

CPU-ONLY (numpy/scipy/matplotlib/torch-cpu); never touches the GPU. Deterministic. Two phases:
  precompute (witness forward + GT fields -> intermediate npz) and render (topology + GIF/MP4),
  so each step fits a short wall-clock budget and is resumable.

Usage::

    # one shot (precompute + render):
    python tools/render_witness_morse_smale_viz.py all \
        --ckpt experiments/results/levelset_n600_wpose1_.../levelset_witness_ema_mlx.npz \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n96.npz \
        --frames 24 --out-dir experiments/results/witness_viz_<utc>

    # or split (each phase short):
    python ... precompute --ckpt ... --gt-cache ... --frames 24 --out <intermediate.npz>
    python ... render --in <intermediate.npz> --out-dir ... --fps 6
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

# ---------------------------------------------------------------------------
# canonical comma10k palette + labels (operator 2026-06-27; MEASURED order, NOT luma-sort).
# ---------------------------------------------------------------------------
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_HEX = ("#402020", "#ff0000", "#808060", "#00ff66", "#cc00ff")
N_CLASSES = 5
CAMERA_H, CAMERA_W = 874, 1164
SCORER_H, SCORER_W = 384, 512


def _refuse_tmp(p: Path) -> None:
    if "/tmp/" in str(p) or str(p).startswith("/tmp"):
        raise ValueError(f"refusing /tmp evidence path {p} (CLAUDE.md durable-artifact rule)")


# ---------------------------------------------------------------------------
# witness level-set forward (op-for-op the shipped numpy/torch inflate; CPU torch).
# ---------------------------------------------------------------------------
def _load_witness(ckpt: Path):
    """Return (forward_fn, manifest, n_pairs) or (None, info, 0) if not a level-set SDF ckpt."""
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from tac.local_acceleration import torch_levelset_inflate as tli
    from tac.boundary_math.lever_b_levelset_generator import (
        CurveletBankConfig,
        curvelet_directional_B,
    )

    z = np.load(ckpt, allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    if "out_sdf.weight" not in params:
        return None, {"reason": "not a level-set SDF witness (no out_sdf head)", "keys": list(params)}, 0
    cfg = {k: (z[k].item() if z[k].size == 1 else z[k].tolist()) for k in z.files if k.startswith("__")}

    bank = CurveletBankConfig(
        n_scales=int(cfg["__bank_n_scales"]), n_orient0=int(cfg["__bank_n_orient0"]),
        f0=float(cfg["__bank_f0"]), base=float(cfg["__bank_base"]), n_iso=int(cfg["__bank_n_iso"]),
    )
    mbf = cfg.get("__cfg_max_bank_freq", None)
    mbf = None if (mbf is None or float(mbf) < 0) else float(mbf)
    curv_w = 2 * int(curvelet_directional_B(bank, max_freq=mbf).shape[1])
    in_feat = int(params["in_proj.weight"].shape[1])
    n_dir = (in_feat - curv_w) // 4
    rh, rw = [int(x) for x in cfg["__render_hw"]]
    man = dict(
        n_classes=int(params["out_sdf.weight"].shape[0]),
        hidden_dim=int(cfg["__cfg_hidden_dim"]), n_hidden=int(cfg["__cfg_n_hidden"]),
        activation=str(cfg["__cfg_activation"]), softmax_temp=float(cfg["__cfg_softmax_temp"]),
        chroma=bool(int(cfg["__cfg_chroma"])), wire_w0=float(cfg["__cfg_wire_w0"]),
        wire_s0=float(cfg["__cfg_wire_s0"]), hosc_beta=float(cfg["__cfg_hosc_beta"]),
        hosc_omega=float(cfg["__cfg_hosc_omega"]),
        bank_n_scales=bank.n_scales, bank_n_orient0=bank.n_orient0, bank_f0=bank.f0,
        bank_base=bank.base, bank_n_iso=bank.n_iso, max_bank_freq=mbf,
        render_h=rh, render_w=rw, camera_h=CAMERA_H, camera_w=CAMERA_W,
        self_orient=bool(int(cfg.get("__cfg_self_orient", 0))), n_dir_freqs=n_dir,
        so_freq_across=float(cfg.get("__cfg_freq_across", 32.0)),
        so_freq_along=float(cfg.get("__cfg_freq_along", 4.0)),
        so_tau=4.0, so_iters=4,  # trainer/byte-close decode defaults
    )
    man["epoch"] = int(cfg.get("__epoch", -1))

    coords = tli.coords_grid(rh, rw)
    B = tli.curvelet_B(man["bank_n_scales"], man["bank_n_orient0"], man["bank_f0"],
                       man["bank_base"], man["bank_n_iso"], man["max_bank_freq"])
    curv = tli.curvelet_feats(coords, B)
    P = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in params.items()}
    code = torch.as_tensor(params["code"], dtype=torch.float32)
    n_pairs = int(params["code"].shape[0] // 2)

    def forward_phi_final(pi: int) -> np.ndarray:
        """(H,W,K) phi for the pair's FINAL frame (the SegNet-scored frame), self-orient fixed pt."""
        if man["self_orient"]:
            dirf = np.zeros((curv.shape[0], 4 * man["n_dir_freqs"]), np.float32)
            prev = None
            for _ in range(man["so_iters"]):
                ft = torch.as_tensor(np.concatenate([curv, dirf], -1), dtype=torch.float32)
                phi, _ = tli.torch_outputs_from_h0(P, tli.torch_in_proj_h0(P, ft, man),
                                                   code[2 * pi + 1], man, False)
                am = phi.argmax(-1).reshape(rh, rw).cpu().numpy().astype(np.int64)
                if prev is not None and np.array_equal(am, prev):
                    break
                dirf = tli.dir_feats(coords, am, man["n_dir_freqs"], man["so_freq_along"],
                                     man["so_freq_across"], man["so_tau"])
                prev = am
            feats = np.concatenate([curv, dirf], -1)
        else:
            feats = curv
        ft = torch.as_tensor(feats, dtype=torch.float32)
        phi, _ = tli.torch_outputs_from_h0(P, tli.torch_in_proj_h0(P, ft, man),
                                           code[2 * pi + 1], man, False)
        return phi.reshape(rh, rw, man["n_classes"]).cpu().numpy().astype(np.float32)

    # code SVD effective rank (participation ratio) — the witness's per-pair code manifold dim.
    cm = np.asarray(params["code"], np.float64)
    cm = cm - cm.mean(0, keepdims=True)
    sv = np.linalg.svd(cm, compute_uv=False)
    eff_rank = float((sv.sum() ** 2) / max((sv ** 2).sum(), 1e-12))
    man["code_eff_rank"] = eff_rank
    man["code_dim"] = int(params["code"].shape[1])
    return forward_phi_final, man, n_pairs


# ---------------------------------------------------------------------------
# GT signed-distance margin (smooth Morse function) + topology helpers.
# ---------------------------------------------------------------------------
def _sdf_top_fields(lstar: np.ndarray):
    """From L* build per-class signed distance phi_k; return (argmax, m12=top1-top2, gap13=top1-top3)."""
    from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

    phi = signed_distance_fields(lstar.astype(np.int64), N_CLASSES)  # (H,W,K)
    srt = np.sort(phi, axis=-1)  # ascending
    top1, top2, top3 = srt[..., -1], srt[..., -2], srt[..., -3]
    am = phi.argmax(-1).astype(np.uint8)
    return am, (top1 - top2).astype(np.float32), (top1 - top3).astype(np.float32)


def _boundary_mask(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def _critical_points(m_smooth: np.ndarray, gap13: np.ndarray, bnd: np.ndarray):
    """Morse-Smale critical-point taxonomy on the smoothed margin m (>=0).

    minima (index-0): local minima of m on the boundary (m->0, deepest boundary pts).
    saddles (index-1): TRIPLE junctions (top3 SDF near-equal) on the boundary; Hessian eigvecs.
    maxima (index-2): local maxima of m (class-cell capitals).
    Returns dict of (row,col[,extra]) arrays, capped for legibility.
    """
    from scipy.ndimage import maximum_filter, minimum_filter, label

    H, W = m_smooth.shape
    # --- index-2 maxima: local maxima of m (confident interiors) ---
    locmax = (m_smooth == maximum_filter(m_smooth, size=15)) & (m_smooth > np.percentile(m_smooth, 90))
    mr, mc = np.where(locmax)
    if mr.size > 40:  # keep the strongest 40
        order = np.argsort(m_smooth[mr, mc])[::-1][:40]
        mr, mc = mr[order], mc[order]

    # --- index-0 minima: local minima of m on/near the boundary (precarious deepest pts) ---
    locmin = (m_smooth == minimum_filter(m_smooth, size=11)) & bnd
    nr, nc = np.where(locmin)
    if nr.size > 120:
        order = np.argsort(m_smooth[nr, nc])[:120]  # smallest margin first
        nr, nc = nr[order], nc[order]

    # --- index-1 saddles == triple junctions: small top1-top3 gap, on boundary; cluster ---
    tj = (gap13 < np.percentile(gap13[bnd], 8) if bnd.any() else np.zeros_like(bnd)) & bnd
    lab_tj, n_tj = label(tj)
    sr, sc = [], []
    for i in range(1, n_tj + 1):
        ys, xs = np.where(lab_tj == i)
        sr.append(int(ys.mean()))
        sc.append(int(xs.mean()))
    sr, sc = np.array(sr, int), np.array(sc, int)

    # Hessian eigenvectors at saddles (finite-diff grad^2 of m_smooth) -> separatrix tangents.
    eig_segs = []
    if sr.size:
        gy, gx = np.gradient(m_smooth)
        gyy, _ = np.gradient(gy)
        gxy, gxx = np.gradient(gx)
        for r, c in zip(sr, sc):
            r = int(np.clip(r, 1, H - 2)); c = int(np.clip(c, 1, W - 2))
            Hm = np.array([[gxx[r, c], gxy[r, c]], [gxy[r, c], gyy[r, c]]], float)
            w, v = np.linalg.eigh(Hm)  # ascending eigenvalues
            # unstable (separatrix tangent) = eigvec of the MORE-NEGATIVE / smaller eigenvalue
            vec = v[:, 0]
            eig_segs.append((c, r, float(vec[0]), float(vec[1])))
    return {
        "max_rc": np.stack([mr, mc], 1) if mr.size else np.zeros((0, 2), int),
        "min_rc": np.stack([nr, nc], 1) if nr.size else np.zeros((0, 2), int),
        "saddle_rc": np.stack([sr, sc], 1) if sr.size else np.zeros((0, 2), int),
        "saddle_eigvec": eig_segs,  # list of (x,y,dx,dy)
    }


def _component_count(lab: np.ndarray) -> int:
    from scipy.ndimage import label

    total = 0
    for k in range(N_CLASSES):
        _, n = label(lab == k)
        total += n
    return int(total)


def _flip_band_per_class(lstar: np.ndarray, margin: np.ndarray, frac: float = 0.15):
    """Among the small-margin (precarious) pixels (lowest `frac` margin), the per-class share.
    This is the MEASURED 'flip-band per-class breakdown' (where d_seg flips concentrate)."""
    thr = np.quantile(margin, frac)
    band = margin <= thr
    counts = np.bincount(lstar[band].ravel(), minlength=N_CLASSES).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


# ---------------------------------------------------------------------------
# PRECOMPUTE
# ---------------------------------------------------------------------------
def precompute(args):
    t_start = time.time()
    gt = np.load(args.gt_cache)
    n_gt = int(gt["lstars"].shape[0])
    fwd = man = None
    n_pairs_w = 0
    if args.ckpt and not args.no_witness:
        fwd, man, n_pairs_w = _load_witness(Path(args.ckpt))
        if fwd is None:
            print(f"[witness N/A] {man.get('reason')} -> GT-only panels (honest).", flush=True)
    n = min(args.frames, n_gt)
    if n_pairs_w:
        n = min(n, n_pairs_w)
    print(f"[precompute] frames={n} gt_pairs={n_gt} witness_pairs={n_pairs_w} "
          f"witness={'yes ep%d' % man['epoch'] if man and fwd else 'NO'}", flush=True)

    src_disp, lstar_a, seg_margin_a, wit_argmax_a = [], [], [], []
    sdf_margin_a, sdf_gap13_a, sdf_argmax_a = [], [], []
    poses, wit_disagree = [], []
    cc_counts = []
    flip_band_accum = np.zeros(N_CLASSES)
    for i in range(n):
        ls = gt["lstars"][i].astype(np.uint8)
        sm = gt["margins"][i].astype(np.float32)
        src = gt["gt_f1"][i][::2, ::2].copy()  # last frame (SegNet-scored), downsampled for display
        sdf_am, sdf_m12, sdf_g13 = _sdf_top_fields(ls)
        if fwd is not None:
            phi = fwd(i)
            wa = phi.argmax(-1).astype(np.uint8)
            wit_argmax_a.append(wa)
            wit_disagree.append(float((wa != ls).mean()))
        cc_counts.append(_component_count(ls))
        flip_band_accum += _flip_band_per_class(ls, sm)
        src_disp.append(src); lstar_a.append(ls); seg_margin_a.append(sm)
        sdf_margin_a.append(sdf_m12); sdf_gap13_a.append(sdf_g13); sdf_argmax_a.append(sdf_am)
        poses.append(gt["gt_poses"][i].astype(np.float64))
        if (i + 1) % 8 == 0:
            print(f"  frame {i+1}/{n}  ({time.time()-t_start:.1f}s)", flush=True)

    out = Path(args.out)
    _refuse_tmp(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = dict(
        src_disp=np.stack(src_disp), lstar=np.stack(lstar_a), seg_margin=np.stack(seg_margin_a),
        sdf_margin=np.stack(sdf_margin_a), sdf_gap13=np.stack(sdf_gap13_a),
        sdf_argmax=np.stack(sdf_argmax_a), poses=np.stack(poses),
        cc_counts=np.array(cc_counts, int),
        flip_band=(flip_band_accum / max(flip_band_accum.sum(), 1.0)).astype(np.float32),
    )
    meta = {"n_frames": n, "ckpt": str(args.ckpt) if (args.ckpt and fwd is not None) else None,
            "gt_cache": str(args.gt_cache), "has_witness": fwd is not None}
    if fwd is not None:
        save["wit_argmax"] = np.stack(wit_argmax_a)
        save["wit_disagree"] = np.array(wit_disagree, np.float32)
        meta.update({"witness_epoch": man["epoch"], "code_eff_rank": man["code_eff_rank"],
                     "code_dim": man["code_dim"], "render_hw": [man["render_h"], man["render_w"]],
                     "self_orient": man["self_orient"], "activation": man["activation"],
                     "chroma": man["chroma"]})
    save["__meta__"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(out, **save)
    print(f"[precompute] wrote {out} ({out.stat().st_size/1e6:.1f} MB) in {time.time()-t_start:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------------
def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import imageio.v2 as imageio

    z = np.load(args.infile, allow_pickle=False)
    meta = json.loads(bytes(z["__meta__"]).decode())
    n = int(meta["n_frames"])
    has_w = bool(meta["has_witness"])
    cmap = ListedColormap(list(CLASS_HEX))
    legend_handles = [Patch(facecolor=CLASS_HEX[k], edgecolor="k", label=f"{k} {CLASS_NAMES[k]}")
                      for k in range(N_CLASSES)]
    flip_band = z["flip_band"]
    cc = z["cc_counts"]
    cc_events = np.abs(np.diff(cc, prepend=cc[0]))  # |Delta connected-components| = births/deaths/merges
    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_frame_stats = []
    frames_png = []
    t0 = time.time()
    for i in range(n):
        src = z["src_disp"][i]
        ls = z["lstar"][i]
        seg_m = z["seg_margin"][i]
        sdf_m = z["sdf_margin"][i]
        sdf_g13 = z["sdf_gap13"][i]
        sdf_am = z["sdf_argmax"][i]
        bnd = _boundary_mask(ls)
        # smooth the SDF margin -> well-posed Morse function for the critical-point complex
        from scipy.ndimage import gaussian_filter
        m_smooth = gaussian_filter(sdf_m, sigma=2.0)
        crit = _critical_points(m_smooth, sdf_g13, bnd)
        n_max, n_min, n_sad = len(crit["max_rc"]), len(crit["min_rc"]), len(crit["saddle_rc"])

        fig = plt.figure(figsize=(16, 11), dpi=args.dpi)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.72], hspace=0.28, wspace=0.16)
        spd = float(np.linalg.norm(z["poses"][i][:3]))
        fig.suptitle(
            f"TOPOLOGY-NATIVE WITNESS VIZ  |  contest 0.mkv  pair {i+1}/{n}  |  "
            f"{'witness ep%d (level-set SDF)' % meta.get('witness_epoch', -1) if has_w else 'GT-only (no SDF witness ckpt)'}"
            f"   [MAX-OBSERVABILITY TOOL — pointer UNMOVED 0.19110; means != ends]",
            fontsize=12, y=0.985)

        # Panel 1: SOURCE contest frame (canonical decode)
        ax = fig.add_subplot(gs[0, 0]); ax.imshow(src); ax.set_title("1. contest frame (canonical YUV->RGB)", fontsize=10); ax.axis("off")

        # Panel 2: SegNet L* argmax
        ax = fig.add_subplot(gs[0, 1]); ax.imshow(ls, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title("2. SegNet L* argmax (frozen-CPU)", fontsize=10); ax.axis("off")
        ax.legend(handles=legend_handles, fontsize=6, loc="lower right", framealpha=0.85)

        # Panel 3: WITNESS level-set partition (argmax phi) OR honest N/A
        ax = fig.add_subplot(gs[0, 2])
        if has_w:
            wa = z["wit_argmax"][i]
            ax.imshow(wa, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            ax.set_title(f"3. witness level-set partition argmax_k phi_k\n(disagree vs L* = {float(z['wit_disagree'][i]):.3f})", fontsize=9)
        else:
            ax.imshow(np.zeros_like(ls), cmap="gray", vmin=0, vmax=1)
            ax.text(0.5, 0.5, "witness SDF ckpt N/A", color="w", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("3. witness level-set partition (N/A)", fontsize=10)
        ax.axis("off")

        # Panel 4: SegNet margin field (Morse function m=phi_top1-phi_top2; where d_seg lives)
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(seg_m, cmap="magma"); ax.set_title("4. SegNet margin m=phi1-phi2 (precariousness)", fontsize=10); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

        # Panel 5: MORSE-SMALE complex on the (smooth) SDF margin: criticals + Hessian eigvecs + separatrix
        ax = fig.add_subplot(gs[1, 1]); ax.imshow(m_smooth, cmap="bone"); ax.axis("off")
        by, bx = np.where(bnd)
        ax.scatter(bx, by, s=0.15, c="#00e5ff", alpha=0.35, marker=".", linewidths=0)  # separatrix 1-skeleton
        if n_max: ax.scatter(crit["max_rc"][:, 1], crit["max_rc"][:, 0], s=42, marker="^", facecolors="none", edgecolors="#39ff14", linewidths=1.3, label=f"max (idx2) cell capital ×{n_max}")
        if n_min: ax.scatter(crit["min_rc"][:, 1], crit["min_rc"][:, 0], s=14, marker="o", c="#ffd000", alpha=0.8, label=f"min (idx0) boundary ×{n_min}")
        if n_sad:
            ax.scatter(crit["saddle_rc"][:, 1], crit["saddle_rc"][:, 0], s=70, marker="X", c="#ff2bd6", edgecolors="k", linewidths=0.6, label=f"saddle (idx1) triple-junction ×{n_sad}")
            for (x, y, dx, dy) in crit["saddle_eigvec"]:
                L = 16.0
                ax.plot([x - L * dx, x + L * dx], [y - L * dy, y + L * dy], "-", c="#ff2bd6", lw=1.0, alpha=0.9)
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85)
        ax.set_title("5. Morse-Smale complex (criticals+separatrix+Hessian eigvec)", fontsize=9)

        # Panel 6: Voronoi / level-set cells (SDF argmax cells + iso-distance level sets + triple junctions)
        ax = fig.add_subplot(gs[1, 2]); ax.imshow(sdf_am, cmap=cmap, vmin=0, vmax=4, interpolation="nearest", alpha=0.82); ax.axis("off")
        # iso-distance level-set contours of the margin (the "level-set witness" rings)
        ax.contour(sdf_m, levels=np.linspace(2, sdf_m.max() * 0.9, 5), colors="k", linewidths=0.4, alpha=0.45)
        if n_sad:
            ax.scatter(crit["saddle_rc"][:, 1], crit["saddle_rc"][:, 0], s=55, marker="X", c="w", edgecolors="k", linewidths=0.7)
        ax.set_title("6. Voronoi cells (SDF argmax) + level sets + triple pts", fontsize=9)

        # ---- stats strip (row 2): 4 charts ----
        # (a) per-class pixel area (+ witness disagreement note)
        ax = fig.add_subplot(gs[2, 0])
        areas = np.bincount(ls.ravel(), minlength=N_CLASSES) / ls.size
        ax.bar(range(N_CLASSES), areas, color=list(CLASS_HEX), edgecolor="k", linewidth=0.4)
        ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, fontsize=6, rotation=30, ha="right")
        ax.set_title("L* per-class area frac", fontsize=8); ax.set_ylim(0, 1)

        # (b) flip-band per-class breakdown (where d_seg flips concentrate; measured)
        ax = fig.add_subplot(gs[2, 1])
        ax.bar(range(N_CLASSES), flip_band, color=list(CLASS_HEX), edgecolor="k", linewidth=0.4)
        ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, fontsize=6, rotation=30, ha="right")
        ax.set_title(f"flip-band per class (low-margin {int(0.15*100)}%ile, all frames)", fontsize=8); ax.set_ylim(0, 1)

        # (c) dimensionality / criticals counts
        ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
        lines = ["DIMENSIONALITY & TOPOLOGY"]
        if has_w:
            lines.append(f"code SVD eff-rank: {meta.get('code_eff_rank', float('nan')):.2f} / {meta.get('code_dim', '?')} dims")
            lines.append(f"render {meta.get('render_hw')}  self_orient={meta.get('self_orient')}  act={meta.get('activation')}")
        static_frac = float((z["lstar"][: i + 1] == z["lstar"][0]).mean()) if i >= 0 else 1.0
        lines.append(f"L* static frac (vs f0): {static_frac:.2f}")
        lines.append("")
        lines.append(f"Morse criticals (this frame):")
        lines.append(f"  index-2 maxima (capitals): {n_max}")
        lines.append(f"  index-1 saddles (triple):  {n_sad}")
        lines.append(f"  index-0 minima (boundary): {n_min}")
        lines.append(f"connected components: {int(cc[i])}")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace", transform=ax.transAxes)

        # (d) spacetime event timeline (CC births/deaths/merges) — embedded as an inset in panel (c) area
        axt = fig.add_axes([0.685, 0.045, 0.29, 0.11])
        axt.plot(range(n), cc_events, color="#888", lw=0.8)
        axt.fill_between(range(n), cc_events, color="#bbb", alpha=0.4)
        axt.axvline(i, color="#ff2bd6", lw=1.4)
        axt.set_title("spacetime events |Δ components|/frame", fontsize=7)
        axt.tick_params(labelsize=6)

        png = out_dir / f"frame_{i:03d}.png"
        fig.savefig(png, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frames_png.append(png)
        per_frame_stats.append({
            "frame": i, "pose_speed": spd, "n_max": n_max, "n_saddle_triple": n_sad,
            "n_min": n_min, "connected_components": int(cc[i]), "cc_event": int(cc_events[i]),
            "witness_disagree_vs_Lstar": (float(z["wit_disagree"][i]) if has_w else None),
        })
        if (i + 1) % 6 == 0:
            print(f"  rendered {i+1}/{n}  ({time.time()-t0:.1f}s)", flush=True)

    # encode GIF (+ MP4 if ffmpeg). bbox_inches="tight" -> per-frame sizes vary slightly;
    # white-pad every frame to a common (max H, max W) canvas (no resize distortion).
    gif = out_dir / "witness_morse_smale.gif"
    raw = [imageio.imread(p) for p in frames_png]
    Hm = max(im.shape[0] for im in raw); Wm = max(im.shape[1] for im in raw)

    def _pad(im):
        h, w = im.shape[:2]
        c = im.shape[2] if im.ndim == 3 else 1
        canvas = np.full((Hm, Wm, c), 255, np.uint8)
        a = im if im.ndim == 3 else im[..., None]
        canvas[:h, :w, : a.shape[2]] = a[..., : min(c, a.shape[2])]
        return canvas[..., :3] if canvas.shape[2] >= 3 else np.repeat(canvas, 3, axis=2)

    imgs = [_pad(im) for im in raw]
    imageio.mimsave(gif, imgs, duration=1.0 / max(args.fps, 1), loop=0)
    print(f"[render] GIF -> {gif} ({gif.stat().st_size/1e6:.1f} MB)", flush=True)
    mp4 = None
    import shutil
    import subprocess
    ffbin = shutil.which("ffmpeg")
    if ffbin:
        mp4 = out_dir / "witness_morse_smale.mp4"
        try:  # preferred: imageio FFMPEG plugin (if imageio-ffmpeg installed)
            evh, evw = Hm - (Hm % 2), Wm - (Wm % 2)
            with imageio.get_writer(mp4, format="FFMPEG", mode="I", fps=args.fps,
                                    codec="libx264", quality=8) as w:
                for im in imgs:
                    w.append_data(im[:evh, :evw])
            print(f"[render] MP4 -> {mp4} ({mp4.stat().st_size/1e6:.1f} MB)", flush=True)
        except Exception as e:  # fall back to the ffmpeg BINARY converting the (uniform) GIF
            print(f"[render] imageio-ffmpeg unavailable ({e}); using ffmpeg binary on the GIF.", flush=True)
            try:
                subprocess.run(
                    [ffbin, "-y", "-i", str(gif), "-movflags", "faststart", "-pix_fmt", "yuv420p",
                     "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", str(args.fps), str(mp4)],
                    check=True, capture_output=True)
                print(f"[render] MP4 -> {mp4} ({mp4.stat().st_size/1e6:.1f} MB)", flush=True)
            except Exception as e2:
                print(f"[render] MP4 skipped: {e2}", flush=True)
                if mp4.exists() and mp4.stat().st_size == 0:
                    mp4.unlink()
                mp4 = None

    stats_json = out_dir / "per_frame_stats.json"
    stats_json.write_text(json.dumps({
        "meta": meta,
        "flip_band_per_class": {CLASS_NAMES[k]: float(flip_band[k]) for k in range(N_CLASSES)},
        "code_eff_rank": meta.get("code_eff_rank"),
        "per_frame": per_frame_stats,
        "note": ("MAX-OBSERVABILITY TOOL; all fields REAL measured (canonical YUV decode, frozen "
                 "SegNet L*/margins, actual witness phi argmax). pointer UNMOVED 0.19110; means!=ends."),
    }, indent=2))
    print(f"[render] stats -> {stats_json}", flush=True)
    if not args.keep_pngs:
        for p in frames_png:
            p.unlink()
    return gif, mp4, stats_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("precompute", "all"):
        p = sub.add_parser(name)
        p.add_argument("--ckpt", default=None, help="level-set witness EMA npz (optional; GT-only if absent/unconverged-format)")
        p.add_argument("--gt-cache", required=True)
        p.add_argument("--frames", type=int, default=24)
        p.add_argument("--no-witness", action="store_true")
        if name == "precompute":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--out-dir", required=True)
            p.add_argument("--fps", type=int, default=6)
            p.add_argument("--dpi", type=int, default=96)
            p.add_argument("--keep-pngs", action="store_true")
    pr = sub.add_parser("render")
    pr.add_argument("--in", dest="infile", required=True)
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--fps", type=int, default=6)
    pr.add_argument("--dpi", type=int, default=96)
    pr.add_argument("--keep-pngs", action="store_true")
    args = ap.parse_args()

    if args.cmd == "precompute":
        precompute(args)
    elif args.cmd == "render":
        render(args)
    elif args.cmd == "all":
        inter = Path(args.out_dir) / "_intermediate.npz"
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        args.out = str(inter)
        precompute(args)
        args.infile = str(inter)
        render(args)


if __name__ == "__main__":
    main()
