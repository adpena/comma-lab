#!/usr/bin/env python3
"""POSE-DIMENSION viz -- the contest's PoseNet term given its own dimensional treatment.

Operator (2026-06-27): "pose is important too and probably interesting to viz dimensions of
in its own right." The contest README hero GIF dedicates a panel to "posenet errors"; SegNet
got the 6-panel + topology treatment -- this is the standalone POSE-dimension instrument.

PANELS / CHARTS (ALL REAL MEASURED -- NO-FAKE):
  1. POSENET-ERRORS WAVEFORM: per-pair d_pose = MSE over the 6 PoseNet scalars, for the
     witness-render pair vs GT, and (optionally) a JPEG baseline pair vs GT, animated.
  2. THE 6 POSE SCALARS over the drive: PoseNet(original consecutive-frame pairs)[:6] as 6
     z-scored time-series with an animated cursor (+ raw forward-speed inset).
  3. EGO-MOTION TRAJECTORY (headline): dead-reckon the 6-dim pose -> the driven bird's-eye
     path. GT-pose path vs witness-pose path overlaid (divergence = realized d_pose).
  4. POSE-MANIFOLD DIMENSIONALITY: SVD of the (Npairs x 6) GT-pose matrix -> spectrum +
     effective rank (raw AND per-dim-standardized) -> how few dims the real driving lives in.
  5. POSE-SENSITIVITY HEATMAP + FOE (telescopic-foveation test): per-pixel PoseNet input
     L1-Jacobian magnitude on GT pairs; locate the focus-of-expansion (FOE / road apex);
     report what fraction of pose-sensitivity falls in a foveal disk around the FOE vs
     uniform, plus the Movable-class (vehicle-light) regions. Concentration >> 1 => foveation
     is a justified theta* lever; diffuse => it is not.

DATA AUTHORITY (CLAUDE.md):
  * GT frames + GT SegNet argmax (L*) + GT poses come from the shared GT cache
    (build_shared_gt_cache_for_mlx_fleet.py -> precompute_gt -> FROZEN CPU-torch SegNet/PoseNet
    on upstream/videos/0.mkv decoded via frame_utils.yuv420_to_rgb -- NEVER PyAV rgb24).
    gt_poses[pi] = PoseNet(original_pair)[:6] = the d_pose authority target.
  * WITNESS realized pose = FROZEN CPU-torch PoseNet on the level-set witness rendered THROUGH
    the contest R operator (tac.local_acceleration.torch_levelset_inflate), float EMA shadow.
  * SENSITIVITY = autograd L1-Jacobian of PoseNet[:6] w.r.t. its 12-channel YUV6 input
    (upstream rgb_to_yuv6 is @no_grad/in-place => severs camera-RGB grad; sensitivity is in
    PoseNet INPUT space -- the YUV6-packed half-res grid (192x256), proportional to the camera
    view, so the FOE maps proportionally to camera space; labelled so).
  * NO MPS anywhere (CLAUDE.md). macOS-CPU = ADVISORY, NON-PROMOTABLE.

POSE-SCALAR SEMANTICS: PoseNet head out=12, distortion uses first 6. openpilot device-frame
convention => [trans_x(forward), trans_y, trans_z, rot_x, rot_y, rot_z]; the exact axis
assignment is INFERRED from the data (forward = largest |mean| dim; yaw = the rotation dim most
correlated with a lateral translation dim) and labelled honestly ("inferred").

MEANS, not an end: a viz moves NO pointer. The exact frontier is byte-closed contest-CPU/CUDA
(pointer UNMOVED 0.19110). This is a max-observability instrument (CLAUDE.md "Max observability").

Deterministic + re-runnable + chunkable (per-pair npz cache -> resumable). CPU-ONLY.

Example:
    .venv/bin/python tools/render_pose_dimension_viz.py \
        --witness-ckpt-dir experiments/results/levelset_amort_decoder_n200_20260627T143830Z \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz \
        --num-pairs 24 --sens-num-pairs 10 --baseline jpeg --phase all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "src", REPO, REPO / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the canonical witness-render + scorer + baseline helpers from the sister comma viz
# (single source of truth for the render/score path -> bit-identical to the inflate authority).
from tools.render_comma_baseline_vs_ours_viz import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    SEG_LABELS,
    SEG_PALETTE_HEX,
    _pose_raw,
    _refuse_tmp,
    baseline_jpeg,
    build_witness_render_context,
    colorize_seg,
    render_our_pairs,
)

POINTER_UNMOVED = 0.19110
SEG_MOVABLE_CLASS = 3  # comma10k canonical order: 0 Road,1 Lane,2 Undrivable,3 Movable,4 MyCar
POSE_DIM_LABELS = ["d0 trans_x (forward)", "d1 trans_y", "d2 trans_z",
                   "d3 rot_x", "d4 rot_y", "d5 rot_z"]


# ---------------------------------------------------------------------------
# PoseNet-only loader (sensitivity + realized pose; seg L* comes from the cache).
# ---------------------------------------------------------------------------
def _load_posenet():
    import torch

    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream

    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = dn.posenet
    for p in posenet.parameters():
        p.requires_grad = False
    return posenet


def _pose_input_sensitivity(posenet, f0_uint8: np.ndarray, f1_uint8: np.ndarray, mode: str):
    """L1-Jacobian magnitude of PoseNet[:6] w.r.t. its 12-channel YUV6 input.

    Returns (sens_combined HxW float32, sens_f0 HxW, sens_f1 HxW) at PoseNet input res
    (384x512). channels 0-5 = frame0 yuv6, 6-11 = frame1 yuv6.
      mode == 'per_output' : sum_{i<6} |d out_i / d input|   (6 backward passes; exact L1-Jac)
      mode == 'sos'        : |d (sum_{i<6} out_i^2) / d input| (1 backward; energy saliency)
    """
    import einops
    import torch

    pair = torch.from_numpy(np.stack([f0_uint8, f1_uint8], axis=0)[None]).float()
    x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    with torch.no_grad():
        pose_in = posenet.preprocess_input(x)  # (1,12,384,512) yuv6 -- grad severed up to here
    pose_in = pose_in.clone().requires_grad_(True)

    grad_abs = torch.zeros_like(pose_in)
    if mode == "sos":
        out = posenet(pose_in)
        pose = (out["pose"] if isinstance(out, dict) else out)[0, :6]
        loss = (pose ** 2).sum()
        (g,) = torch.autograd.grad(loss, pose_in)
        grad_abs = g.abs()
    else:  # per_output -- exact sum_i |d out_i / d input|
        out = posenet(pose_in)
        pose = (out["pose"] if isinstance(out, dict) else out)[0, :6]
        for i in range(6):
            (g,) = torch.autograd.grad(pose[i], pose_in, retain_graph=(i < 5))
            grad_abs = grad_abs + g.abs()

    ga = grad_abs[0].detach().cpu().numpy().astype(np.float64)  # (12,384,512)
    sens_f0 = ga[0:6].sum(0).astype(np.float32)
    sens_f1 = ga[6:12].sum(0).astype(np.float32)
    sens = (sens_f0 + sens_f1).astype(np.float32)
    return sens, sens_f0, sens_f1


# ---------------------------------------------------------------------------
# Pose dimensionality (SVD) + axis semantics + dead-reckoned trajectory.
# ---------------------------------------------------------------------------
def _svd_spectrum(gp: np.ndarray) -> dict:
    """SVD of (N,6) pose matrix, raw-centered AND per-dim standardized."""
    out = {}
    for name, scale in (("raw_centered", False), ("standardized", True)):
        X = gp - gp.mean(0)
        if scale:
            X = X / (gp.std(0) + 1e-12)
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        ev = (S ** 2) / max((S ** 2).sum(), 1e-30)
        pr = float((S ** 2).sum() ** 2 / max((S ** 4).sum(), 1e-30))  # participation ratio
        p = ev / max(ev.sum(), 1e-30)
        ent_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        out[name] = {
            "singular_values": [float(v) for v in S],
            "explained_var_ratio": [float(v) for v in ev],
            "cumulative": [float(v) for v in np.cumsum(ev)],
            "eff_rank_participation": pr,
            "eff_rank_entropy": ent_rank,
        }
    return out


def _detect_axes(gp: np.ndarray) -> dict:
    """Infer forward / lateral / yaw dims (openpilot device-frame convention; data-checked)."""
    means = np.abs(gp.mean(0))
    fwd = int(np.argmax(means))  # forward = largest |mean| (steady driving speed)
    C = np.corrcoef(gp.T)
    trans_cands = [d for d in (0, 1, 2) if d != fwd] or [1, 2]
    rot_cands = [3, 4, 5]
    best = (-1.0, rot_cands[0], trans_cands[0])
    for r in rot_cands:
        for t in trans_cands:
            c = abs(float(C[r, t]))
            if c > best[0]:
                best = (c, r, t)
    yaw, lateral = best[1], best[2]
    return {"forward_dim": fwd, "yaw_dim": int(yaw), "lateral_dim": int(lateral),
            "yaw_lateral_abscorr": float(best[0]),
            "corr_matrix": [[float(v) for v in row] for row in C],
            "note": "axis assignment INFERRED (openpilot device-frame); forward=max|mean|, "
                    "yaw=rotation dim most correlated with a lateral translation dim."}


def _dead_reckon(gp: np.ndarray, axes: dict) -> np.ndarray:
    """Integrate 6-dim pose -> (N+1, 2) bird's-eye path (pose-units). Heading from yaw."""
    fwd, yaw, lat = axes["forward_dim"], axes["yaw_dim"], axes["lateral_dim"]
    f = gp[:, fwd].astype(np.float64)
    y = gp[:, yaw].astype(np.float64)
    l = gp[:, lat].astype(np.float64)
    heading = np.cumsum(y)
    pos = np.zeros((len(f) + 1, 2), np.float64)
    for k in range(len(f)):
        h = heading[k]
        pos[k + 1, 0] = pos[k, 0] + f[k] * np.cos(h) - l[k] * np.sin(h)
        pos[k + 1, 1] = pos[k, 1] + f[k] * np.sin(h) + l[k] * np.cos(h)
    return pos


# ---------------------------------------------------------------------------
# FOE + foveal concentration.
# ---------------------------------------------------------------------------
def _foe_and_concentration(sens: np.ndarray, lstar: np.ndarray | None,
                           foveal_frac: float, horizon_frac: float, top_pct: float) -> dict:
    H, W = sens.shape
    total = float(sens.sum()) + 1e-30
    # data-driven FOE = centroid of the top-percentile sensitivity pixels (robust to bg drag)
    thr = np.percentile(sens, 100.0 - top_pct)
    mask = sens >= thr
    ys, xs = np.nonzero(mask)
    w = sens[mask]
    if w.sum() > 0:
        foe_row = float((ys * w).sum() / w.sum())
        foe_col = float((xs * w).sum() / w.sum())
    else:
        foe_row, foe_col = H * horizon_frac, W / 2.0
    # geometric prior FOE = image center-x, horizon row
    prior_row, prior_col = H * horizon_frac, W / 2.0

    r = foveal_frac * H  # foveal disk radius in input-space pixels
    yy, xx = np.mgrid[0:H, 0:W]

    def _disk_frac(cr, cc):
        disk = ((yy - cr) ** 2 + (xx - cc) ** 2) <= r * r
        area_frac = float(disk.mean())
        sens_frac = float(sens[disk].sum() / total)
        ratio = sens_frac / max(area_frac, 1e-30)
        return area_frac, sens_frac, ratio

    a_d, s_d, ratio_d = _disk_frac(foe_row, foe_col)
    a_p, s_p, ratio_p = _disk_frac(prior_row, prior_col)

    movable_frac = None
    if lstar is not None and lstar.shape == sens.shape:
        mv = (lstar == SEG_MOVABLE_CLASS)
        movable_frac = float(sens[mv].sum() / total) if mv.any() else 0.0

    return {
        "img_h": H, "img_w": W,
        "foe_data_driven_rowcol": [foe_row, foe_col],
        "foe_geometric_prior_rowcol": [prior_row, prior_col],
        "foveal_radius_px": float(r), "foveal_radius_frac_of_H": foveal_frac,
        "disk_area_frac": a_d,
        "sens_frac_in_foveal_disk": s_d,
        "concentration_ratio_data_driven": ratio_d,
        "sens_frac_in_prior_disk": s_p,
        "concentration_ratio_geometric_prior": ratio_p,
        "sens_frac_in_movable_class": movable_frac,
        "horizon_frac": horizon_frac, "top_pct_for_foe": top_pct,
    }


# ---------------------------------------------------------------------------
# Phase A -- precompute (resumable per-pair caches + global pose stats).
# ---------------------------------------------------------------------------
def phase_pose(args, out_dir: Path) -> None:
    gt = np.load(args.gt_cache)
    gt_f0, gt_f1 = gt["gt_f0"], gt["gt_f1"]
    lstars = gt["lstars"] if "lstars" in gt.files else None
    if "gt_poses" not in gt.files:
        raise ValueError(f"{args.gt_cache} has no gt_poses (need the PoseNet authority target)")
    gt_poses = np.asarray(gt["gt_poses"], np.float64)[:, :6]
    n_avail = int(gt_f1.shape[0])

    # --- GLOBAL (cheap, no rendering): SVD + axes + GT trajectory over ALL available pairs ---
    svd = _svd_spectrum(gt_poses)
    axes = _detect_axes(gt_poses)
    traj_gt = _dead_reckon(gt_poses, axes)
    np.savez_compressed(out_dir / "pose_global.npz",
                        gt_poses=gt_poses, traj_gt=traj_gt,
                        forward_dim=axes["forward_dim"], yaw_dim=axes["yaw_dim"],
                        lateral_dim=axes["lateral_dim"])
    (out_dir / "pose_global.json").write_text(json.dumps({
        "gt_cache": str(args.gt_cache), "n_pairs_available": n_avail,
        "pose_dim_labels": POSE_DIM_LABELS,
        "per_dim_mean": [float(v) for v in gt_poses.mean(0)],
        "per_dim_std": [float(v) for v in gt_poses.std(0)],
        "svd": svd, "axes": axes,
        "axis": "[macOS-CPU advisory, NON-PROMOTABLE]", "pointer_unmoved": POINTER_UNMOVED,
    }, indent=2))
    print(f"[global] raw eff-rank(PR)={svd['raw_centered']['eff_rank_participation']:.3f}  "
          f"standardized eff-rank(PR)={svd['standardized']['eff_rank_participation']:.3f}  "
          f"axes fwd={axes['forward_dim']} yaw={axes['yaw_dim']} lat={axes['lateral_dim']}", flush=True)

    # --- WITNESS realized pose + JPEG baseline, per-pair cached (resumable) ---
    start, end = args.pair_start, min(args.pair_start + args.num_pairs, n_avail)
    pair_dir = out_dir / "pairs_npz"
    pair_dir.mkdir(parents=True, exist_ok=True)

    witness_ok = False
    ctx = None
    try:
        ctx = build_witness_render_context(
            Path(args.witness_ckpt_dir), args.witness_npz,
            {"freq_across": args.so_freq_across, "freq_along": args.so_freq_along,
             "tau": args.so_tau, "iters": args.so_iters})
        if ctx[0]["n_pairs"] >= end:
            witness_ok = True
        else:
            print(f"[WARN] witness has {ctx[0]['n_pairs']} pairs < end {end}; witness curves -> N/A", flush=True)
    except Exception as e:
        print(f"[WARN] witness ckpt unavailable ({e}); witness curves -> N/A (not fabricated)", flush=True)

    posenet = _load_posenet()
    for pi in range(start, end):
        cache = pair_dir / f"pair_{pi:04d}.npz"
        if cache.exists() and not args.force:
            print(f"[skip] pair {pi} cached", flush=True)
            continue
        t0 = time.time()
        gp = gt_poses[pi]
        gtf0 = np.asarray(gt_f0[pi], np.uint8)
        gtf1 = np.asarray(gt_f1[pi], np.uint8)

        our_p = None
        our_dpose = None
        if witness_ok:
            (our_f0, our_f1), = render_our_pairs(ctx[0], ctx[1], ctx[2], pi, pi + 1)
            our_p = _pose_raw(posenet, our_f0, our_f1)[:6]
            our_dpose = float(((our_p - gp) ** 2).mean())

        base_p = base_dpose = base_bytes = None
        if args.baseline == "jpeg":
            base_f0, b0 = baseline_jpeg(gtf0, args.baseline_jpeg_quality)
            base_f1, b1 = baseline_jpeg(gtf1, args.baseline_jpeg_quality)
            base_bytes = int(b0 + b1)
            base_p = _pose_raw(posenet, base_f0, base_f1)[:6]
            base_dpose = float(((base_p - gp) ** 2).mean())

        np.savez_compressed(
            cache, pair_idx=pi, gt_pose=gp.astype(np.float64),
            our_pose=(our_p.astype(np.float64) if our_p is not None else np.zeros((1,), np.float64)),
            base_pose=(base_p.astype(np.float64) if base_p is not None else np.zeros((1,), np.float64)),
            has_witness=np.asarray(our_p is not None), has_baseline=np.asarray(base_p is not None),
            metrics=np.asarray(json.dumps({
                "pair_idx": pi, "our_dpose": our_dpose, "base_dpose": base_dpose,
                "base_bytes": base_bytes,
                "gt_pose": [float(v) for v in gp],
                "our_pose": ([float(v) for v in our_p] if our_p is not None else None),
                "base_pose": ([float(v) for v in base_p] if base_p is not None else None),
            })))
        msg = f"[pose] pair {pi}: "
        msg += (f"our_dpose={our_dpose:.3e} " if our_dpose is not None else "our_dpose=N/A ")
        msg += (f"base_dpose={base_dpose:.3e} " if base_dpose is not None else "")
        print(msg + f"({time.time()-t0:.1f}s)", flush=True)

    # --- SENSITIVITY subset (resumable per-pair) + FOE concentration ---
    if args.sens_num_pairs > 0:
        sens_dir = out_dir / "sens_npz"
        sens_dir.mkdir(parents=True, exist_ok=True)
        # spread the subset across [start,end) so the FOE motion over the drive is visible
        k = max(1, args.sens_num_pairs)
        idxs = np.unique(np.linspace(start, end - 1, k).round().astype(int)).tolist()
        for pi in idxs:
            cache = sens_dir / f"sens_{pi:04d}.npz"
            if cache.exists() and not args.force:
                print(f"[skip-sens] pair {pi} cached", flush=True)
                continue
            t0 = time.time()
            gtf0 = np.asarray(gt_f0[pi], np.uint8)
            gtf1 = np.asarray(gt_f1[pi], np.uint8)
            sens, sf0, sf1 = _pose_input_sensitivity(posenet, gtf0, gtf1, args.sens_mode)
            import cv2
            # sens is in PoseNet YUV6-packed input space (half of the SegNet 384x512 space).
            # Resize the cache L* (384x512) -> sens space (nearest, preserve class ids) so the
            # Movable mask + overlay align with the sensitivity map.
            lstar = None
            if lstars is not None:
                lraw = np.asarray(lstars[pi], np.int64)
                lstar = cv2.resize(lraw.astype(np.int32), (sens.shape[1], sens.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int64)
            conc = _foe_and_concentration(sens, lstar, args.foveal_radius_frac,
                                          args.horizon_frac, args.foe_top_pct)
            # downscale GT frame1 to PoseNet input space for overlay (cheap, display only)
            gt_small = cv2.resize(gtf1, (sens.shape[1], sens.shape[0]), interpolation=cv2.INTER_AREA)
            np.savez_compressed(cache, pair_idx=pi, sens=sens.astype(np.float32),
                                gt_small=gt_small.astype(np.uint8),
                                lstar=(lstar if lstar is not None else np.zeros((1,), np.int64)),
                                has_lstar=np.asarray(lstar is not None),
                                conc=np.asarray(json.dumps(conc)))
            print(f"[sens] pair {pi}: conc_ratio(FOE)={conc['concentration_ratio_data_driven']:.2f} "
                  f"sens_in_disk={conc['sens_frac_in_foveal_disk']:.3f} "
                  f"movable_frac={conc['sens_frac_in_movable_class']} ({time.time()-t0:.1f}s, mode={args.sens_mode})",
                  flush=True)


# ---------------------------------------------------------------------------
# Phase B -- static charts + animated per-pair frames.
# ---------------------------------------------------------------------------
def _load_pairs(out_dir: Path):
    rows = []
    for c in sorted((out_dir / "pairs_npz").glob("pair_*.npz")):
        rows.append(json.loads(str(np.load(c, allow_pickle=False)["metrics"])))
    return rows


def _load_sens(out_dir: Path):
    items = []
    for c in sorted((out_dir / "sens_npz").glob("sens_*.npz")):
        z = np.load(c, allow_pickle=False)
        items.append((int(z["pair_idx"]), z["sens"].astype(np.float32),
                      z["gt_small"], (z["lstar"] if bool(z["has_lstar"]) else None),
                      json.loads(str(z["conc"]))))
    return items


def phase_panels(args, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")

    glob = json.loads((out_dir / "pose_global.json").read_text())
    gnpz = np.load(out_dir / "pose_global.npz")
    gt_poses = gnpz["gt_poses"]
    traj_gt = gnpz["traj_gt"]
    axes = glob["axes"]
    fwd, yaw, lat = axes["forward_dim"], axes["yaw_dim"], axes["lateral_dim"]
    rows = _load_pairs(out_dir)
    sens_items = _load_sens(out_dir)

    png_dir = out_dir / "panels_png"
    png_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = out_dir / "charts_png"
    chart_dir.mkdir(parents=True, exist_ok=True)

    n = gt_poses.shape[0]
    x = np.arange(n)
    z = (gt_poses - gt_poses.mean(0)) / (gt_poses.std(0) + 1e-12)

    # witness path over the rendered subset (anchored at the matching GT trajectory point)
    rendered = sorted([r["pair_idx"] for r in rows if r.get("our_pose") is not None])
    witness_traj = None
    if rendered:
        seg = np.array([next(r["our_pose"] for r in rows if r["pair_idx"] == pi) for pi in rendered], np.float64)
        wt = _dead_reckon(seg, axes)
        anchor = traj_gt[rendered[0]]
        witness_traj = (rendered, wt - wt[0] + anchor)

    # ---------- STATIC CHART 1: SVD spectrum ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, key in enumerate(("raw_centered", "standardized")):
        s = glob["svd"][key]
        ax[j].bar(range(1, 7), s["explained_var_ratio"], color="#3aa0ff", alpha=0.85)
        ax[j].plot(range(1, 7), s["cumulative"], "-o", color="#ff8c42", label="cumulative")
        ax[j].set_title(f"{key}  eff-rank(PR)={s['eff_rank_participation']:.2f}  "
                        f"entropy={s['eff_rank_entropy']:.2f}", fontsize=10)
        ax[j].set_xlabel("pose singular component"); ax[j].set_ylim(0, 1.05)
        ax[j].legend(fontsize=8)
    fig.suptitle(f"POSE-MANIFOLD DIMENSIONALITY -- SVD of ({n}x6) GT-pose matrix  "
                 f"[macOS-CPU ADVISORY | pointer UNMOVED {POINTER_UNMOVED}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(chart_dir / "svd_spectrum.png", dpi=args.dpi); plt.close(fig)

    # ---------- STATIC CHART 2: trajectory ----------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(traj_gt[:, 0], traj_gt[:, 1], "-", color="#39ff88", lw=1.6, label="GT-pose path")
    ax.scatter([traj_gt[0, 0]], [traj_gt[0, 1]], c="#ffffff", s=40, zorder=5, label="start")
    if witness_traj is not None:
        ax.plot(witness_traj[1][:, 0], witness_traj[1][:, 1], "--", color="#ff5566", lw=1.4,
                label="witness-pose path (rendered subset)")
    ax.set_aspect("equal"); ax.set_title(
        f"EGO-MOTION TRAJECTORY (dead-reckoned, pose-units)\nfwd=d{fwd} yaw=d{yaw} lat=d{lat} "
        f"[axis assignment INFERRED]", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(chart_dir / "trajectory.png", dpi=args.dpi); plt.close(fig)

    # ---------- STATIC CHART 3: 6 scalars ----------
    fig, axs = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
    for d in range(6):
        axs[d].plot(x, gt_poses[:, d], color="#3aa0ff", lw=1.1)
        axs[d].set_ylabel(POSE_DIM_LABELS[d], fontsize=7)
        axs[d].grid(alpha=0.15)
    axs[0].set_title(f"THE 6 POSE SCALARS over the drive ({n} pairs, raw PoseNet[:6])", fontsize=11)
    axs[-1].set_xlabel("pair index"); fig.tight_layout()
    fig.savefig(chart_dir / "six_scalars.png", dpi=args.dpi); plt.close(fig)

    # ---------- STATIC CHART 4: FOE concentration + aggregate sensitivity ----------
    foe_summary = None
    if sens_items:
        agg = np.mean([s for (_, s, _, _, _) in sens_items], axis=0)
        agg_n = agg / (agg.max() + 1e-12)
        ratios = [c["concentration_ratio_data_driven"] for (_, _, _, _, c) in sens_items]
        sfracs = [c["sens_frac_in_foveal_disk"] for (_, _, _, _, c) in sens_items]
        mfracs = [c["sens_frac_in_movable_class"] for (_, _, _, _, c) in sens_items
                  if c["sens_frac_in_movable_class"] is not None]
        c0 = sens_items[len(sens_items) // 2][4]
        foe_summary = {
            "n_sens_pairs": len(sens_items),
            "mean_concentration_ratio_data_driven": float(np.mean(ratios)),
            "mean_sens_frac_in_foveal_disk": float(np.mean(sfracs)),
            "mean_sens_frac_in_movable_class": (float(np.mean(mfracs)) if mfracs else None),
            "disk_area_frac": c0["disk_area_frac"],
            "foveal_radius_frac_of_H": c0["foveal_radius_frac_of_H"],
            "uniform_baseline_ratio": 1.0,
            "sens_mode": args.sens_mode,
            "interpretation": ("concentration_ratio >> 1 => pose sensitivity is foveated at the "
                               "FOE => telescopic foveation is a justified theta* lever; ~1 => diffuse."),
        }
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(agg_n, cmap="inferno")
        cr = c0["foe_data_driven_rowcol"]
        ax[0].scatter([cr[1]], [cr[0]], c="#39ff88", marker="+", s=200, label="data-driven FOE")
        pr = c0["foe_geometric_prior_rowcol"]
        ax[0].scatter([pr[1]], [pr[0]], c="#00ffff", marker="x", s=120, label="geometric-prior FOE")
        th = np.linspace(0, 2 * np.pi, 100); rr = c0["foveal_radius_px"]
        ax[0].plot(cr[1] + rr * np.cos(th), cr[0] + rr * np.sin(th), color="#39ff88", lw=1.5)
        ax[0].set_title(f"aggregate pose-sensitivity (PoseNet YUV6 input space {agg.shape[1]}x{agg.shape[0]})",
                        fontsize=9)
        ax[0].legend(fontsize=7, loc="lower right"); ax[0].axis("off")
        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
        ax[1].bar(range(len(ratios)), ratios, color="#ff8c42")
        ax[1].axhline(1.0, color="#888", ls="--", label="uniform (ratio=1)")
        ax[1].set_title(f"foveal concentration ratio per pair\nmean={np.mean(ratios):.2f}x uniform "
                        f"(disk={c0['disk_area_frac']*100:.0f}% of frame)", fontsize=9)
        ax[1].set_xlabel("sensitivity-subset pair"); ax[1].legend(fontsize=7)
        fig.suptitle(f"TELESCOPIC-FOVEATION TEST -- pose-sensitivity FOE concentration "
                     f"[macOS-CPU ADVISORY | mode={args.sens_mode}]", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(chart_dir / "foe_concentration.png", dpi=args.dpi); plt.close(fig)
    (out_dir / "foe_summary.json").write_text(json.dumps(foe_summary or {"foe": "N/A (no sens pairs)"}, indent=2))

    # ---------- ANIMATED per-pair frames (10fps, dark) ----------
    dpose_our = {r["pair_idx"]: r["our_dpose"] for r in rows if r.get("our_dpose") is not None}
    dpose_base = {r["pair_idx"]: r["base_dpose"] for r in rows if r.get("base_dpose") is not None}
    anim_pairs = sorted([r["pair_idx"] for r in rows]) or list(range(min(n, args.num_pairs)))
    palette = ["#ff5050", "#50ff96", "#5096ff", "#ffcc44", "#cc66ff", "#44e0e0"]

    for pi in anim_pairs:
        fig, axx = plt.subplots(2, 2, figsize=(12.8, 7.6))
        fig.suptitle(f"POSE-DIMENSION viz -- pair {pi}/{n}  |  witness: {Path(args.witness_ckpt_dir).name}\n"
                     f"[macOS-CPU ADVISORY, NON-PROMOTABLE | PoseNet[:6] frozen CPU-torch | "
                     f"pointer UNMOVED {POINTER_UNMOVED} -- a viz is a MEANS]", fontsize=10)

        # TL: 6 z-scored scalars + cursor
        axa = axx[0, 0]
        for d in range(6):
            axa.plot(x, z[:, d], color=palette[d], lw=0.9, label=POSE_DIM_LABELS[d])
        axa.axvline(pi, color="#ffffff", lw=1.2, alpha=0.8)
        axa.set_title("6 pose scalars (z-scored) + cursor", fontsize=9)
        axa.legend(fontsize=5.5, ncol=2, loc="upper right"); axa.grid(alpha=0.15)
        axa.set_xlabel("pair idx")

        # TR: d_pose waveform (witness vs baseline) + cursor
        axb = axx[0, 1]
        if dpose_our:
            xs = sorted(dpose_our); axb.plot(xs, [dpose_our[i] for i in xs], "-o", ms=3,
                                             color="#ff5566", label="witness d_pose")
        if dpose_base:
            xs = sorted(dpose_base); axb.plot(xs, [dpose_base[i] for i in xs], "-s", ms=3,
                                              color="#ffaa33", label="JPEG baseline d_pose")
        if not dpose_our and not dpose_base:
            axb.text(0.5, 0.5, "d_pose curves N/A\n(no witness/baseline)", ha="center", va="center",
                     color="gray", transform=axb.transAxes)
        axb.axvline(pi, color="#ffffff", lw=1.2, alpha=0.8)
        axb.set_yscale("log"); axb.set_title("POSENET-ERRORS waveform: d_pose=MSE(pose[:6])", fontsize=9)
        axb.set_xlabel("pair idx"); axb.legend(fontsize=7); axb.grid(alpha=0.15)

        # BL: trajectory + moving dot
        axc = axx[1, 0]
        axc.plot(traj_gt[:, 0], traj_gt[:, 1], "-", color="#39ff88", lw=1.4, label="GT path")
        if witness_traj is not None:
            axc.plot(witness_traj[1][:, 0], witness_traj[1][:, 1], "--", color="#ff5566", lw=1.2,
                     label="witness path")
        axc.scatter([traj_gt[pi, 0]], [traj_gt[pi, 1]], c="#ffffff", s=45, zorder=6)
        axc.set_aspect("equal"); axc.set_title(
            f"EGO-MOTION trajectory (fwd=d{fwd},yaw=d{yaw},lat=d{lat}; INFERRED)", fontsize=9)
        axc.legend(fontsize=7); axc.grid(alpha=0.18)

        # BR: sensitivity heatmap (nearest sens pair) + FOE + foveal disk + movable overlay
        axd = axx[1, 1]
        if sens_items:
            nearest = min(sens_items, key=lambda it: abs(it[0] - pi))
            spi, sens, gt_small, lstar, conc = nearest
            sn = sens / (sens.max() + 1e-12)
            axd.imshow(gt_small)
            axd.imshow(sn, cmap="inferno", alpha=0.55)
            cr = conc["foe_data_driven_rowcol"]; rr = conc["foveal_radius_px"]
            th = np.linspace(0, 2 * np.pi, 100)
            axd.plot(cr[1] + rr * np.cos(th), cr[0] + rr * np.sin(th), color="#39ff88", lw=1.6)
            axd.scatter([cr[1]], [cr[0]], c="#39ff88", marker="+", s=160)
            if lstar is not None and lstar.shape == sens.shape:
                mv = (lstar == SEG_MOVABLE_CLASS)
                ov = np.zeros((*mv.shape, 4), np.float32); ov[mv] = (0.0, 1.0, 1.0, 0.35)
                axd.imshow(ov)
            axd.set_title(f"pose-sensitivity + FOE (sens pair {spi})  "
                          f"conc={conc['concentration_ratio_data_driven']:.1f}x uniform", fontsize=8)
            axd.axis("off")
        else:
            axd.text(0.5, 0.5, "sensitivity N/A", ha="center", va="center", color="gray",
                     transform=axd.transAxes); axd.axis("off")

        # stats line
        d_o = dpose_our.get(pi); d_b = dpose_base.get(pi)
        line = f"pair {pi}: GT pose[:6]=" + np.array2string(gt_poses[pi], precision=3, suppress_small=True)
        if d_o is not None:
            line += f"  | witness d_pose={d_o:.3e} -> sqrt(10*d_pose)={np.sqrt(10*d_o):.4f}"
        if d_b is not None:
            line += f"  | JPEG d_pose={d_b:.3e}"
        fig.text(0.01, 0.005, line, fontsize=7.5, family="monospace", va="bottom")
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        fig.savefig(png_dir / f"posepanel_{pi:04d}.png", dpi=args.dpi); plt.close(fig)
        print(f"[panel] posepanel_{pi:04d}.png", flush=True)


# ---------------------------------------------------------------------------
# Phase C -- assemble GIF/MP4 + merged stats JSON.
# ---------------------------------------------------------------------------
def phase_assemble(args, out_dir: Path) -> None:
    import imageio.v2 as imageio

    png_dir = out_dir / "panels_png"
    pngs = sorted(png_dir.glob("posepanel_*.png"))
    if not pngs:
        raise FileNotFoundError(f"no panels in {png_dir}; run --phase panels first")
    imgs = [imageio.imread(p) for p in pngs]
    gif = out_dir / "pose_dimension.gif"
    imageio.mimsave(gif, imgs, duration=1.0 / max(args.fps, 1), loop=0)
    print(f"[gif] {gif}  ({len(imgs)} frames @ {args.fps}fps)", flush=True)
    mp4 = out_dir / "pose_dimension.mp4"
    try:  # imageio plugin first
        imageio.mimsave(mp4, imgs, fps=args.fps)
        if mp4.stat().st_size == 0:
            raise RuntimeError("imageio wrote 0 bytes")
        print(f"[mp4] {mp4}", flush=True)
    except Exception as e:
        # fall back to system ffmpeg over the PNG sequence (even dims for yuv420p)
        import shutil as _sh
        import subprocess
        ff = _sh.which("ffmpeg")
        if ff:
            cmd = [ff, "-y", "-framerate", str(args.fps), "-pattern_type", "glob",
                   "-i", str(png_dir / "posepanel_*.png"),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(mp4)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and mp4.exists() and mp4.stat().st_size > 0:
                print(f"[mp4] {mp4} (system ffmpeg)", flush=True)
            else:
                print(f"[WARN] mp4 encode skipped (imageio:{e}; ffmpeg rc={r.returncode})", flush=True)
        else:
            print(f"[WARN] mp4 encode skipped ({e}; no system ffmpeg)", flush=True)

    glob = json.loads((out_dir / "pose_global.json").read_text())
    foe = json.loads((out_dir / "foe_summary.json").read_text())
    rows = _load_pairs(out_dir)
    def _mean(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else None
    summary = {
        "tool": "tools/render_pose_dimension_viz.py",
        "witness_ckpt_dir": str(args.witness_ckpt_dir), "gt_cache": str(args.gt_cache),
        "axis": "[macOS-CPU advisory, NON-PROMOTABLE]", "pointer_unmoved": POINTER_UNMOVED,
        "n_pairs_total": glob["n_pairs_available"],
        "pose_dimensionality": {
            "raw_eff_rank_PR": glob["svd"]["raw_centered"]["eff_rank_participation"],
            "raw_eff_rank_entropy": glob["svd"]["raw_centered"]["eff_rank_entropy"],
            "standardized_eff_rank_PR": glob["svd"]["standardized"]["eff_rank_participation"],
            "standardized_eff_rank_entropy": glob["svd"]["standardized"]["eff_rank_entropy"],
        },
        "axes_inferred": glob["axes"],
        "mean_witness_dpose": _mean("our_dpose"), "mean_baseline_dpose": _mean("base_dpose"),
        "foe_concentration": foe,
        "charts": [str(p.relative_to(out_dir)) for p in sorted((out_dir / "charts_png").glob("*.png"))],
        "gif": "pose_dimension.gif", "mp4": "pose_dimension.mp4",
        "per_pair": rows,
    }
    (out_dir / "pose_dimension_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"[stats] {out_dir/'pose_dimension_stats.json'}", flush=True)
    print(f"[FINDING] raw eff-rank={summary['pose_dimensionality']['raw_eff_rank_PR']:.3f}  "
          f"standardized eff-rank={summary['pose_dimensionality']['standardized_eff_rank_PR']:.3f}", flush=True)
    if isinstance(foe, dict) and foe.get("mean_concentration_ratio_data_driven") is not None:
        print(f"[FINDING] FOE pose-sensitivity concentration = "
              f"{foe['mean_concentration_ratio_data_driven']:.2f}x uniform "
              f"(disk={foe['disk_area_frac']*100:.0f}% of frame; movable_frac="
              f"{foe['mean_sens_frac_in_movable_class']})", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness-ckpt-dir", default="experiments/results/levelset_amort_decoder_n200_20260627T143830Z")
    ap.add_argument("--witness-npz", default=None)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--num-pairs", type=int, default=24, help="pairs to render witness/baseline pose for")
    ap.add_argument("--pair-start", type=int, default=0)
    ap.add_argument("--baseline", choices=["jpeg", "none"], default="jpeg")
    ap.add_argument("--baseline-jpeg-quality", type=int, default=10)
    ap.add_argument("--sens-num-pairs", type=int, default=10, help="pairs for pose-sensitivity/FOE (0=off)")
    ap.add_argument("--sens-mode", choices=["per_output", "sos"], default="per_output")
    ap.add_argument("--foveal-radius-frac", type=float, default=0.20, help="foveal disk radius / H")
    ap.add_argument("--horizon-frac", type=float, default=0.45, help="geometric-prior FOE row / H")
    ap.add_argument("--foe-top-pct", type=float, default=15.0, help="top%% sens pixels for data-driven FOE centroid")
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=4.0)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--phase", choices=["pose", "panels", "assemble", "all"], default="all")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.out_dir is None:
        utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.out_dir = f"experiments/results/witness_viz_pose_dimension_{utc}"
    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir, "--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pose-viz] out_dir={out_dir} phase={args.phase}", flush=True)

    if args.phase in ("pose", "all"):
        phase_pose(args, out_dir)
    if args.phase in ("panels", "all"):
        phase_panels(args, out_dir)
    if args.phase in ("assemble", "all"):
        phase_assemble(args, out_dir)
    print(f"[pose-viz] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
