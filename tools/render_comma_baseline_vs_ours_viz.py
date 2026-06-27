#!/usr/bin/env python3
"""Canonical comma.ai 6-panel multipane viz -- BASELINE-vs-OURS reconstruction error.

The operator's ask (2026-06-27): "existing comma.ai-style data viz showing BASELINE
errors compared to OURS." This renders the canonical comma multipane (per CLAUDE.md
"Multipane matplotlib data viz") over a representative frame range, as a baseline-vs-ours
RECONSTRUCTION-ERROR comparison:

  Row 1: GT Original (canonical decode) | OUR Reconstruction (witness render) | OUR Pixel Error (hot)
  Row 2: GT SegNet masks               | OUR SegNet masks (argmax on our render) | OUR SegNet DISAGREEMENT (red = realized d_seg map)
  Row 3: BASELINE Reconstruction       | BASELINE Pixel Error (hot)            | BASELINE SegNet DISAGREEMENT (red)

plus a per-frame STATS strip (d_seg / d_pose / pixel-MSE / implied-S distortion contribution,
for OURS and BASELINE, per-frame + cumulative).

NO-FAKE: EVERY panel is a REAL measured frame/mask/error.
  * GT frames + GT SegNet argmax (L*) + GT poses come from the shared GT cache
    (tools/build_shared_gt_cache_for_mlx_fleet.py -> precompute_gt -> the frozen CPU-torch
    SegNet/PoseNet on upstream/videos/0.mkv decoded via frame_utils -- NEVER PyAV rgb24).
  * OUR reconstruction = the level-set witness rendered THROUGH the contest R operator
    (tac.local_acceleration.torch_levelset_inflate.decode_levelset_torch), float EMA shadow.
  * OUR SegNet masks = the FROZEN CPU-torch SegNet argmax on OUR rendered frame1 (the realized
    d_seg path -- tac.boundary_math.seg_core.segnet_argmax_and_margin), the SAME authority L* was
    built with. NO MPS (CLAUDE.md). macOS-CPU = ADVISORY, NON-PROMOTABLE.
  * BASELINE = a generic lossy codec (JPEG at a chosen quality) decode of the SAME GT frames,
    scored the same way -- an HONEST "generic codec error vs our task-space witness." If no
    baseline source is available, the baseline row is honestly labelled N/A (never fabricated).

MEANS, not an end: a viz moves NO pointer. The exact frontier is byte-closed contest-CPU/CUDA
(pointer UNMOVED 0.19110). This tool is a max-observability instrument (CLAUDE.md "Max observability").

Deterministic + re-runnable + chunkable (per-frame npz cache -> resumable; assemble GIF/MP4 + stats JSON).

Example:
    .venv/bin/python tools/render_comma_baseline_vs_ours_viz.py \
        --witness-ckpt-dir experiments/results/levelset_amort_decoder_n200_20260627T143830Z \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz \
        --num-pairs 24 --baseline jpeg --baseline-jpeg-quality 10 --phase all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "src", REPO, REPO / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---- canonical openpilot / comma seg palette + labels (operator-specified 2026-06-27) ----
# class order = comma10k canonical (measured; do NOT luma-sort) per CLAUDE.md SegNet class table.
SEG_PALETTE_HEX = ["#402020", "#ff0000", "#808060", "#00ff66", "#cc00ff"]
SEG_LABELS = ["0 Road", "1 Lane", "2 Undrivable", "3 Movable", "4 MyCar"]
CAMERA_H, CAMERA_W = 874, 1164
SCORE_RATE_DENOM = 37_545_489  # evaluate.py rate-term denominator


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


SEG_PALETTE_RGB = np.asarray([_hex_to_rgb(h) for h in SEG_PALETTE_HEX], dtype=np.uint8)  # (5,3)


def colorize_seg(argmax_hw: np.ndarray) -> np.ndarray:
    """(H,W) int argmax in 0..4 -> (H,W,3) uint8 via the canonical palette."""
    a = np.clip(np.asarray(argmax_hw), 0, len(SEG_PALETTE_RGB) - 1).astype(np.int64)
    return SEG_PALETTE_RGB[a]


def _refuse_tmp(path: Path, field: str) -> None:
    if "/tmp/" in str(path) or str(path).startswith("/tmp"):
        raise ValueError(f"{field} points at /tmp ({path}); durable path required (CLAUDE.md).")


# ---------------------------------------------------------------------------
# OUR render: level-set witness through R (canonical inflate decode, float EMA shadow).
# ---------------------------------------------------------------------------
def build_witness_render_context(ckpt_dir: Path, npz_name: str | None, so_overrides: dict[str, Any]):
    """Load the witness EMA shadow + assemble the authoritative render manifest. Returns
    (manifest, params_np, code_np). Uses the canonical byte-close ckpt loader + self-orient
    detector so the float render matches the witness's trained forward (no divergence)."""
    from tools.levelset_byte_close_and_eval import _load_levelset_ckpt, detect_self_orient

    params, cfg = _load_levelset_ckpt(ckpt_dir, npz_name)
    so = detect_self_orient(cfg, so_overrides)
    code_np = np.asarray(params.pop("code"), dtype=np.float32)
    manifest = {
        "n_pairs": int(cfg["n_pairs"]),
        "n_hidden": int(cfg["n_hidden"]), "hidden_dim": int(cfg["hidden_dim"]),
        "softmax_temp": float(cfg["softmax_temp"]), "chroma": bool(cfg["chroma"]),
        "wire_w0": float(cfg["wire_w0"]), "wire_s0": float(cfg["wire_s0"]),
        "hosc_beta": float(cfg["hosc_beta"]), "hosc_omega": float(cfg["hosc_omega"]),
        "activation": str(cfg["activation"]),
        "bank_n_scales": int(cfg["bank_n_scales"]), "bank_n_orient0": int(cfg["bank_n_orient0"]),
        "bank_f0": float(cfg["bank_f0"]), "bank_base": float(cfg["bank_base"]),
        "bank_n_iso": int(cfg["bank_n_iso"]),
        "max_bank_freq": (None if cfg["max_bank_freq"] is None else float(cfg["max_bank_freq"])),
        "render_h": int(cfg["render_h"]), "render_w": int(cfg["render_w"]),
        "camera_h": CAMERA_H, "camera_w": CAMERA_W,
        "self_orient": bool(so["self_orient"]),
        "n_dir_freqs": int(so.get("n_dir_freqs", 0)),
        "so_freq_across": float(so.get("freq_across", 0.0)),
        "so_freq_along": float(so.get("freq_along", 0.0)),
        "so_tau": float(so.get("tau", 4.0)),
        "so_iters": int(so.get("iters", 0)),
    }
    return manifest, params, code_np


def render_our_pairs(manifest, params, code_np, pair_start: int, pair_end: int):
    """Render OUR (f0, f1) camera-res uint8 for pairs [pair_start, pair_end). Replicates the
    canonical decode_levelset_torch per-pair loop (so we can render a chunk without re-decoding
    from 0). Imports the same module-level torch helpers -> bit-identical to the inflate path."""
    import torch

    from tac.local_acceleration.torch_levelset_inflate import (
        coords_grid, curvelet_B, curvelet_feats, dir_feats,
        torch_in_proj_h0, torch_outputs_from_h0, torch_R,
    )

    m = manifest
    rh, rw = int(m["render_h"]), int(m["render_w"])
    ch, cw = int(m["camera_h"]), int(m["camera_w"])
    dev = torch.device("cpu")
    coords = coords_grid(rh, rw)
    B = curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"],
                   m["bank_n_iso"], m["max_bank_freq"])
    curv = curvelet_feats(coords, B)
    P = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev) for k, v in params.items()}
    code = torch.as_tensor(code_np, dtype=torch.float32, device=dev)

    out = []
    for pi in range(pair_start, pair_end):
        if m["self_orient"]:
            dirf = np.zeros((curv.shape[0], 4 * int(m["n_dir_freqs"])), np.float32)
            prev_am = None
            for _ in range(int(m["so_iters"])):
                feats_t = torch.as_tensor(np.concatenate([curv, dirf], axis=-1), dtype=torch.float32, device=dev)
                phi, _ = torch_outputs_from_h0(P, torch_in_proj_h0(P, feats_t, m), code[2 * pi + 1], m, False)
                am = phi.argmax(-1).reshape(rh, rw).cpu().numpy().astype(np.int64)
                if prev_am is not None and np.array_equal(am, prev_am):
                    break
                dirf = dir_feats(coords, am, m["n_dir_freqs"], m["so_freq_along"], m["so_freq_across"], m["so_tau"])
                prev_am = am
            feats = np.concatenate([curv, dirf], axis=-1)
        else:
            feats = curv
        h0 = torch_in_proj_h0(P, torch.as_tensor(feats, dtype=torch.float32, device=dev), m)
        _phi0, rgb0 = torch_outputs_from_h0(P, h0, code[2 * pi + 0], m, True)
        _phi1, rgb1 = torch_outputs_from_h0(P, h0, code[2 * pi + 1], m, True)
        f0 = torch_R(rgb0, rh, rw, ch, cw)
        f1 = torch_R(rgb1, rh, rw, ch, cw)
        out.append((np.asarray(f0, np.uint8), np.asarray(f1, np.uint8)))
    return out


# ---------------------------------------------------------------------------
# BASELINE: generic lossy codec (JPEG) recon of the GT frame (honest "generic codec" baseline).
# ---------------------------------------------------------------------------
def baseline_jpeg(frame_rgb_uint8: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    """JPEG encode->decode a camera-res RGB uint8 frame. Returns (decoded_rgb_uint8, byte_count)."""
    import cv2

    bgr = frame_rgb_uint8[..., ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode JPEG failed")
    dec_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(dec_bgr[..., ::-1]).astype(np.uint8), int(buf.size)


# ---------------------------------------------------------------------------
# Scorers (frozen CPU-torch authority -- NEVER MPS).
# ---------------------------------------------------------------------------
def _load_scorers(want_pose: bool):
    from tac.boundary_math.seg_core import load_real_segnet, segnet_argmax_and_margin

    seg = load_real_segnet("cpu")
    posenet = None
    if want_pose:
        try:
            import torch

            from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream

            dn = DistortionNet().eval()
            dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
            posenet = dn.posenet
            for p in posenet.parameters():
                p.requires_grad = False
        except Exception as e:  # pose is optional; degrade honestly
            print(f"[WARN] pose scorer unavailable ({e}); d_pose -> N/A", flush=True)
            posenet = None
    return seg, segnet_argmax_and_margin, posenet


def _pose_raw(posenet, f0_uint8: np.ndarray, f1_uint8: np.ndarray) -> np.ndarray:
    import einops
    import torch

    pair = torch.from_numpy(np.stack([f0_uint8, f1_uint8], axis=0)[None]).float()
    x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        pose_in = posenet.preprocess_input(x)
        out = posenet(pose_in)
        pose = out["pose"] if isinstance(out, dict) else out
        half = None
        for hh in posenet.hydra.heads:
            if hh.name == "pose":
                half = hh.out // 2
                break
        if half is None:
            half = pose.shape[-1] // 2
        return pose[0, :half].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Phase A -- render + score; cache per-frame arrays + metrics (resumable).
# ---------------------------------------------------------------------------
def phase_render(args, out_dir: Path) -> None:
    gt = np.load(args.gt_cache)
    gt_f0, gt_f1, lstars = gt["gt_f0"], gt["gt_f1"], gt["lstars"]
    gt_poses = gt["gt_poses"] if "gt_poses" in gt.files else None
    n_avail = int(gt_f1.shape[0])
    start, end = args.pair_start, min(args.pair_start + args.num_pairs, n_avail)

    ctx = build_witness_render_context(Path(args.witness_ckpt_dir), args.witness_npz,
                                       {"freq_across": args.so_freq_across, "freq_along": args.so_freq_along,
                                        "tau": args.so_tau, "iters": args.so_iters})
    manifest, params, code_np = ctx
    if manifest["n_pairs"] < end:
        raise ValueError(f"witness has {manifest['n_pairs']} pairs < requested end {end}")
    # SILENT-WRONGNESS GUARD (honest caveat; ckpt stores no cache identity to auto-check): OUR
    # d_seg/d_pose compare SegNet/PoseNet(witness pair i) vs GT[i] — valid ONLY if --gt-cache is
    # the witness's TRAINING cache. A strided/contiguous/different-n mismatch silently scores each
    # pair against the WRONG GT frame -> a FALSE 'authority' number. Pass the matching cache.
    print(f"[ALIGN WARN] OUR d_seg/d_pose validity REQUIRES --gt-cache ({args.gt_cache}) to be the "
          f"witness's TRAINING cache — NOT auto-checked.", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    seg, seg_argmax, posenet = _load_scorers(args.pose)
    want_pose = posenet is not None and gt_poses is not None

    frame_dir = out_dir / "frames_npz"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for pi in range(start, end):
        cache = frame_dir / f"frame_{pi:04d}.npz"
        if cache.exists() and not args.force:
            print(f"[skip] pair {pi} cached", flush=True)
            continue
        t0 = time.time()
        (our_f0, our_f1), = render_our_pairs(manifest, params, code_np, pi, pi + 1)
        gtf0 = np.asarray(gt_f0[pi], np.uint8)
        gtf1 = np.asarray(gt_f1[pi], np.uint8)
        gt_lstar = np.asarray(lstars[pi], np.int64)

        # OUR seg (realized argmax on our render) + disagreement vs GT L*
        our_lstar, _ = seg_argmax(seg, our_f1)
        our_lstar = np.asarray(our_lstar, np.int64)
        our_dis = (our_lstar != gt_lstar)
        our_dseg = float(our_dis.mean())
        our_mse = float(((gtf1.astype(np.float64) - our_f1.astype(np.float64)) ** 2).mean())

        # BASELINE
        base_kind = args.baseline
        if base_kind == "jpeg":
            base_f0, b0 = baseline_jpeg(gtf0, args.baseline_jpeg_quality)
            base_f1, b1 = baseline_jpeg(gtf1, args.baseline_jpeg_quality)
            base_bytes = int(b0 + b1)
            base_lstar, _ = seg_argmax(seg, base_f1)
            base_lstar = np.asarray(base_lstar, np.int64)
            base_dis = (base_lstar != gt_lstar)
            base_dseg = float(base_dis.mean())
            base_mse = float(((gtf1.astype(np.float64) - base_f1.astype(np.float64)) ** 2).mean())
        else:
            base_f0 = base_f1 = None
            base_lstar = base_dis = None
            base_bytes = base_dseg = base_mse = None

        # pose
        our_dpose = base_dpose = None
        if want_pose:
            gp = np.asarray(gt_poses[pi], np.float64)[:6]
            our_p = _pose_raw(posenet, our_f0, our_f1)[:6]
            our_dpose = float(((our_p - gp) ** 2).mean())
            if base_kind == "jpeg":
                base_p = _pose_raw(posenet, base_f0, base_f1)[:6]
                base_dpose = float(((base_p - gp) ** 2).mean())

        np.savez_compressed(
            cache,
            pair_idx=pi,
            gt_f1=gtf1, gt_lstar=gt_lstar,
            our_f1=our_f1, our_lstar=our_lstar, our_dis=our_dis,
            base_f1=(base_f1 if base_f1 is not None else np.zeros((1,), np.uint8)),
            base_lstar=(base_lstar if base_lstar is not None else np.zeros((1,), np.int64)),
            base_dis=(base_dis if base_dis is not None else np.zeros((1,), bool)),
            has_baseline=np.asarray(base_f1 is not None),
            metrics=np.asarray(json.dumps({
                "pair_idx": pi,
                "our_dseg": our_dseg, "our_mse": our_mse, "our_dpose": our_dpose,
                "base_dseg": base_dseg, "base_mse": base_mse, "base_dpose": base_dpose,
                "base_bytes": base_bytes, "baseline_kind": base_kind,
            })),
        )
        print(f"[done] pair {pi}: our_dseg={our_dseg:.5f}"
              + (f" base_dseg={base_dseg:.5f}" if base_dseg is not None else "")
              + f" ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Phase B -- panels (matplotlib) from cached frame npz (fast, no torch).
# ---------------------------------------------------------------------------
def _read_metrics(cache_npz) -> dict:
    return json.loads(str(cache_npz["metrics"]))


def phase_panels(args, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    frame_dir = out_dir / "frames_npz"
    png_dir = out_dir / "panels_png"
    png_dir.mkdir(parents=True, exist_ok=True)
    caches = sorted(frame_dir.glob("frame_*.npz"))
    if not caches:
        raise FileNotFoundError(f"no cached frames in {frame_dir}; run --phase render first")

    legend_handles = [Patch(facecolor=SEG_PALETTE_HEX[i], edgecolor="k", label=SEG_LABELS[i])
                      for i in range(len(SEG_LABELS))]
    cum = {"our_dseg": 0.0, "our_mse": 0.0, "base_dseg": 0.0, "base_mse": 0.0,
           "our_dpose": 0.0, "base_dpose": 0.0, "n": 0, "n_pose": 0}

    for cpath in caches:
        z = np.load(cpath, allow_pickle=False)
        mt = _read_metrics(z)
        has_base = bool(z["has_baseline"])
        gt_f1 = z["gt_f1"]; our_f1 = z["our_f1"]
        gt_seg = colorize_seg(z["gt_lstar"]); our_seg = colorize_seg(z["our_lstar"])
        our_err = np.abs(gt_f1.astype(np.float64) - our_f1.astype(np.float64)).mean(-1)
        our_dis = z["our_dis"]

        cum["n"] += 1
        cum["our_dseg"] += mt["our_dseg"]; cum["our_mse"] += mt["our_mse"]
        if has_base:
            cum["base_dseg"] += mt["base_dseg"]; cum["base_mse"] += mt["base_mse"]
        if mt.get("our_dpose") is not None:
            cum["n_pose"] += 1; cum["our_dpose"] += mt["our_dpose"]
            if has_base and mt.get("base_dpose") is not None:
                cum["base_dpose"] += mt["base_dpose"]

        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        pi = int(mt["pair_idx"])
        fig.suptitle(
            f"comma 6-panel  BASELINE-vs-OURS  -- pair {pi}  |  witness: {Path(args.witness_ckpt_dir).name}\n"
            f"[macOS-CPU ADVISORY, NON-PROMOTABLE | realized SegNet argmax through R | pointer UNMOVED 0.19110 -- a viz is a MEANS]",
            fontsize=11)

        def show(ax, img, title, cmap=None, vmax=None):
            ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=vmax)
            ax.set_title(title, fontsize=9); ax.axis("off")

        # Row 1: GT | OURS recon | OURS pixel err
        show(axes[0, 0], gt_f1, "GT Original (canonical decode)")
        show(axes[0, 1], our_f1, f"OUR Reconstruction (witness render)  MSE={mt['our_mse']:.1f}")
        im = axes[0, 2].imshow(our_err, cmap="hot", vmin=0, vmax=128)
        axes[0, 2].set_title("OUR Pixel Error |GT-ours|", fontsize=9); axes[0, 2].axis("off")
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Row 2: GT seg | OUR seg | OUR disagreement
        show(axes[1, 0], gt_seg, "GT SegNet argmax (L*)")
        show(axes[1, 1], our_seg, "OUR SegNet argmax (realized through R)")
        dis_rgb = np.zeros((*our_dis.shape, 3), np.uint8); dis_rgb[our_dis] = (255, 0, 0)
        show(axes[1, 2], dis_rgb, f"OUR SegNet DISAGREEMENT (red)  d_seg={mt['our_dseg']:.5f}")
        axes[1, 0].legend(handles=legend_handles, loc="upper right", fontsize=6, framealpha=0.8)

        # Row 3: baseline
        if has_base:
            base_f1 = z["base_f1"]
            base_err = np.abs(gt_f1.astype(np.float64) - base_f1.astype(np.float64)).mean(-1)
            base_dis = z["base_dis"]
            bd = dict(q=args.baseline_jpeg_quality, by=mt["base_bytes"])
            show(axes[2, 0], base_f1, f"BASELINE recon (JPEG q={bd['q']}, {bd['by']}B/pair)  MSE={mt['base_mse']:.1f}")
            imb = axes[2, 1].imshow(base_err, cmap="hot", vmin=0, vmax=128)
            axes[2, 1].set_title("BASELINE Pixel Error", fontsize=9); axes[2, 1].axis("off")
            fig.colorbar(imb, ax=axes[2, 1], fraction=0.046, pad=0.04)
            bdis_rgb = np.zeros((*base_dis.shape, 3), np.uint8); bdis_rgb[base_dis] = (255, 0, 0)
            show(axes[2, 2], bdis_rgb, f"BASELINE DISAGREEMENT (red)  d_seg={mt['base_dseg']:.5f}")
        else:
            for j in range(3):
                axes[2, j].text(0.5, 0.5, "baseline-recon source\nN/A (not fabricated)",
                                ha="center", va="center", fontsize=12, color="gray")
                axes[2, j].axis("off")

        # stats strip
        def s_contrib(dseg, dpose):
            v = 100.0 * dseg
            if dpose is not None:
                v += float(np.sqrt(10.0 * dpose))
            return v
        lines = [
            f"PER-FRAME (pair {pi}):  OUR d_seg={mt['our_dseg']:.5f}  MSE={mt['our_mse']:.1f}"
            + (f"  d_pose={mt['our_dpose']:.3e}" if mt.get('our_dpose') is not None else "  d_pose=N/A")
            + f"  ->100*d_seg+sqrt(10*d_pose)={s_contrib(mt['our_dseg'], mt.get('our_dpose')):.4f}",
        ]
        if has_base:
            lines.append(
                f"                       BASE d_seg={mt['base_dseg']:.5f}  MSE={mt['base_mse']:.1f}"
                + (f"  d_pose={mt['base_dpose']:.3e}" if mt.get('base_dpose') is not None else "  d_pose=N/A")
                + f"  ->{s_contrib(mt['base_dseg'], mt.get('base_dpose')):.4f}")
        n = cum["n"]
        lines.append(
            f"CUMULATIVE ({n} frm):  OUR mean d_seg={cum['our_dseg']/n:.5f}  mean MSE={cum['our_mse']/n:.1f}"
            + (f"  mean d_pose={cum['our_dpose']/cum['n_pose']:.3e}" if cum['n_pose'] else ""))
        if has_base:
            lines.append(
                f"                       BASE mean d_seg={cum['base_dseg']/n:.5f}  mean MSE={cum['base_mse']/n:.1f}"
                + (f"  mean d_pose={cum['base_dpose']/cum['n_pose']:.3e}" if cum['n_pose'] else ""))
            lines.append(
                f"RATE CAVEAT: JPEG q{args.baseline_jpeg_quality} is NOT byte-matched (spends ~{mt['base_bytes']}B/pair); the"
                " witness amortizes ~hundreds B/pair -> compare d_seg WITH this large rate gap in mind.")
        lines.append(
            "NOTE: OUR is a TASK-SPACE witness -> high RGB pixel-error is EXPECTED (it optimizes the SegNet"
            " argmax partition + pose, NOT full-RGB fidelity); read Row 2 (seg) as the scored signal.")
        fig.text(0.01, 0.005, "\n".join(lines), fontsize=8, family="monospace", va="bottom")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        out_png = png_dir / f"panel_{pi:04d}.png"
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)
        print(f"[panel] {out_png.name}", flush=True)


# ---------------------------------------------------------------------------
# Phase C -- assemble GIF/MP4 + merged stats JSON.
# ---------------------------------------------------------------------------
def phase_assemble(args, out_dir: Path) -> None:
    import imageio.v2 as imageio

    png_dir = out_dir / "panels_png"
    pngs = sorted(png_dir.glob("panel_*.png"))
    if not pngs:
        raise FileNotFoundError(f"no panels in {png_dir}; run --phase panels first")
    imgs = [imageio.imread(p) for p in pngs]
    gif_path = out_dir / "comma_baseline_vs_ours.gif"
    imageio.mimsave(gif_path, imgs, duration=1.0 / max(args.fps, 1), loop=0)
    print(f"[gif] {gif_path}  ({len(imgs)} frames)", flush=True)
    mp4_path = None
    try:
        mp4_path = out_dir / "comma_baseline_vs_ours.mp4"
        imageio.mimsave(mp4_path, imgs, fps=args.fps)
        print(f"[mp4] {mp4_path}", flush=True)
    except Exception as e:
        print(f"[WARN] mp4 encode skipped ({e})", flush=True)
        mp4_path = None

    # merged stats JSON
    frame_dir = out_dir / "frames_npz"
    rows = []
    for c in sorted(frame_dir.glob("frame_*.npz")):
        rows.append(_read_metrics(np.load(c, allow_pickle=False)))
    def _mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    summary = {
        "n_frames": len(rows),
        "witness_ckpt_dir": str(args.witness_ckpt_dir),
        "gt_cache": str(args.gt_cache),
        "baseline_kind": args.baseline,
        "baseline_jpeg_quality": args.baseline_jpeg_quality if args.baseline == "jpeg" else None,
        "axis": "[macOS-CPU advisory, NON-PROMOTABLE]",
        "pointer_unmoved": 0.19110,
        "mean_our_dseg": _mean("our_dseg"), "mean_our_mse": _mean("our_mse"),
        "mean_our_dpose": _mean("our_dpose"),
        "mean_base_dseg": _mean("base_dseg"), "mean_base_mse": _mean("base_mse"),
        "mean_base_dpose": _mean("base_dpose"),
        "per_frame": rows,
    }
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=2))
    print(f"[stats] {out_dir/'stats.json'}  mean OUR d_seg={summary['mean_our_dseg']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness-ckpt-dir", default="experiments/results/levelset_amort_decoder_n200_20260627T143830Z")
    ap.add_argument("--witness-npz", default=None, help="npz name in ckpt dir (default: ema then live)")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--pair-start", type=int, default=0)
    ap.add_argument("--baseline", choices=["jpeg", "none"], default="jpeg")
    ap.add_argument("--baseline-jpeg-quality", type=int, default=10)
    ap.add_argument("--pose", action="store_true", default=True, help="compute d_pose (default on)")
    ap.add_argument("--no-pose", dest="pose", action="store_false")
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=4.0)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--phase", choices=["render", "panels", "assemble", "all"], default="all")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="re-render cached frames")
    args = ap.parse_args(argv)

    if args.out_dir is None:
        utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.out_dir = f"experiments/results/witness_viz_comma_baseline_vs_ours_{utc}"
    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir, "--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] out_dir={out_dir} phase={args.phase}", flush=True)

    if args.phase in ("render", "all"):
        phase_render(args, out_dir)
    if args.phase in ("panels", "all"):
        phase_panels(args, out_dir)
    if args.phase in ("assemble", "all"):
        phase_assemble(args, out_dir)
    print(f"[viz] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
