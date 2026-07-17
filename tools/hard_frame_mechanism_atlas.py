# SPDX-License-Identifier: MIT
"""HARD-FRAME MECHANISM ATLAS — visual analysis of the witness-own residual (#273).

Operator standing method (2026-07-16, memory
`visual_hard_frame_analysis_names_mechanisms_statistics_only_rank_20260716.md`):
every numeric residual decomposition is INCOMPLETE until its top buckets are
traced VISUALLY on the actual hard frames/pairs, the physical dynamic at each
hard region is NAMED, and the name is mapped to its deep-math object + cheapest
treatment. Visual analysis NAMES mechanisms; bucket statistics only RANK them
(proof case: MyCar rim = specular reflections on the wet curved hood).

This tool renders the visual evidence; the NAMING is done by eye on the output.

Stages ($0 local, [macOS-CPU advisory], frozen CPU-torch fp32 SegNet, bit-exact
cached GT gt_n600.npz, cached mod32cap ep650 witness frames on the SSD tier):

  select   Rank hard pairs per residual bucket from the MEASURED witness-own
           decomposition rows (experiments/results/c2_witness_own_decomp_20260716/
           decomp_rows.jsonl) — top-K per bucket (Road-Lane / Movable border /
           Undrivable horizon / MyCar rim) + worst overall + easy controls,
           with min temporal separation. Writes selection.json.
  render   For each selected pair: extended canonical multipane montage —
           row 1: GT frame | witness render | argmax-diff overlay (red);
           row 2: signed witness margin field (top1-top2, negative=disagree,
           zero-set contoured) | sensitivity heat (VJP of the summed margin
           deficit at disagreeing px, |grad|^2 at camera res, luma-frac split
           per BT.601) | temporal flicker map (dis(i) vs dis(i+1): red=now-only,
           green=next-only, yellow=persist). Numbered callouts on the top
           disagreement clusters + native-res zoom crops (GT | witness |
           overlay) per callout for mechanism naming by eye.

Honesty: research_only; score_claim=false; promotable=false. The pointer
(0.19108) moves only via upstream/evaluate.py on exact archive bytes.

  ladder   RESOLUTION LADDER per hard pair — the frozen_scorer_exact_factorization
           memo made VISUAL, using the EXACT upstream ops (modules.py:109 bilinear
           interpolate = the shared resize A_seg==A_pose; frame_utils.rgb_to_yuv6):
           row 1 (camera res, S1): GT | witness | luma diff dY (BT.601) | chroma
           diff |(dU,dV)| (exact frame_utils rows);
           row 2 (S2 kernel/range split): camera diff = ker(A) part (invisible to
           BOTH scorers, blind B1) + range part (what reaches the scorers) — exact
           orthogonal decomposition via CG on (A A^T), adjoint by autograd of the
           REAL interpolate call; residual check printed on panel;
           row 3 (scorer res, S2->S3->S4): A.GT | A.witness | |A.diff| | signed
           margin | argmax-diff. Per-callout ladder metrics (cam_rms LSB,
           postA_rms, ker energy frac, survival ratio) classify each hard region
           sub-pixel-born / blind-dominant / faithful (MEASURED numbers, heuristic
           thresholds, stated on panel). --pose-frames adds a P2/P3 pose-path
           figure: chroma diff pre vs post the 2x2 BOX-AVERAGE (what PoseNet
           cannot see; luma 2x2 space-to-depth is lossless).

Usage:
  .venv/bin/python tools/hard_frame_mechanism_atlas.py --stage select
  .venv/bin/python tools/hard_frame_mechanism_atlas.py --stage render
  .venv/bin/python tools/hard_frame_mechanism_atlas.py --stage render --frames 172 388
  .venv/bin/python tools/hard_frame_mechanism_atlas.py --stage ladder --pose-frames 172
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "tools")

from c2_perclass_stratum_carrier_analysis import (
    CLASSES,
    H_C,
    H_S,
    W_C,
    W_S,
    _load_gt,
)
from c2_witness_own_decomp import (
    LUMA_HAT,
    MASKS_MM,
    ROWS,
    _frames_memmap,
    _rendered_set,
)
from necessity_dseg_calibration import _load_segnet

OUT_DIR = "experiments/results/hard_frame_mechanism_atlas_20260716"
AXIS = ("[macOS-CPU advisory] frozen CPU-torch fp32 SegNet; bit-exact cached GT "
        "(gt_n600.npz); FROZEN mod32cap EMA-best ep650 witness frames through exact R")
MIN_FREE_GIB = 12.0

# bucket -> pair_side keys that count toward it (from decomp_rows.jsonl schema)
BUCKETS = {
    "roadlane": ("Road-Lane|Road", "Road-Lane|Lane"),
    "movable": ("Road-Movable|Road", "Road-Movable|Movable",
                "Undrivable-Movable|Undrivable", "Undrivable-Movable|Movable",
                "Lane-Movable|Lane", "Lane-Movable|Movable"),
    "horizon": ("Road-Undrivable|Road", "Road-Undrivable|Undrivable",
                "Lane-Undrivable|Lane", "Lane-Undrivable|Undrivable"),
    "rim": ("Road-MyCar|Road", "Road-MyCar|MyCar",
            "Lane-MyCar|Lane", "Lane-MyCar|MyCar"),
}
CLASS_COLORS = np.array([  # display only
    [90, 90, 90],      # Road grey
    [255, 220, 0],     # Lane yellow
    [70, 130, 220],    # Undrivable blue
    [230, 60, 60],     # Movable red
    [160, 60, 200],    # MyCar purple
], np.uint8)


def _free_gib() -> float:
    try:
        try:
            from tools.mem_basis import conservative_free_gib
        except Exception:
            from mem_basis import conservative_free_gib  # type: ignore
        return conservative_free_gib(default=float("inf"))
    except Exception:
        return float("inf")


def _rows_by_frame() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(ROWS) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                rows[int(r["frame"])] = r  # last row wins (resume dedupe)
            except (json.JSONDecodeError, KeyError):
                pass
    return rows


def _pick_top(scored: list[tuple[float, int]], k: int, taken: set[int],
              min_sep: int = 12) -> list[int]:
    """Greedy top-k with temporal separation, skipping already-taken frames."""
    out: list[int] = []
    for _, f in sorted(scored, reverse=True):
        if f in taken:
            continue
        if any(abs(f - g) < min_sep for g in out):
            continue
        out.append(f)
        if len(out) >= k:
            break
    return out


def stage_select(per_bucket: int, n_worst: int, n_easy: int) -> None:
    rows = _rows_by_frame()
    sel: dict[str, list[dict]] = {}
    taken: set[int] = set()
    # worst overall first (they anchor the atlas)
    worst = _pick_top([(r["dseg"], f) for f, r in rows.items()], n_worst, taken)
    taken.update(worst)
    sel["worst_overall"] = [
        {"frame": f, "dseg": rows[f]["dseg"], "why": "top-N overall d_seg"}
        for f in worst]
    for name, keys in BUCKETS.items():
        scored = []
        for f, r in rows.items():
            s = sum(r.get("pair_side", {}).get(k, 0) for k in keys)
            if s:
                scored.append((float(s), f))
        picks = _pick_top(scored, per_bucket, taken)
        taken.update(picks)
        sel[name] = [{"frame": f,
                      "bucket_px": sum(rows[f].get("pair_side", {}).get(k, 0)
                                       for k in keys),
                      "dseg": rows[f]["dseg"],
                      "why": f"top {name} pair-side px"} for f in picks]
    easy = _pick_top([(-r["dseg"], f) for f, r in rows.items()], n_easy, taken)
    sel["easy_control"] = [
        {"frame": f, "dseg": rows[f]["dseg"], "why": "lowest d_seg (visual control)"}
        for f in easy]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"scope": f"selection from MEASURED decomp_rows.jsonl ({len(rows)} frames); "
                    f"{AXIS}", "selection": sel}
    with open(os.path.join(OUT_DIR, "selection.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# render helpers
# ---------------------------------------------------------------------------
def _witness_forward(model, cam_u8: np.ndarray):
    """Forward + margin-deficit VJP on ONE camera frame. Returns
    (pred(384x512), signed_margin(384x512), grad_cam(874x1164x3), luma_frac)."""
    import torch
    import torch.nn.functional as tfun

    x_cam = torch.from_numpy(cam_u8).permute(2, 0, 1).float().unsqueeze(0)
    x_cam.requires_grad_(True)
    x_s = tfun.interpolate(x_cam, size=(H_S, W_S), mode="bilinear")
    logits = model(x_s)[0]
    top2 = torch.topk(logits, 2, dim=0).values
    margin = (top2[0] - top2[1]).detach().numpy()
    pred = logits.argmax(dim=0).numpy()
    return x_cam, logits, pred, margin


def _sens_heat(x_cam, logits, pred: np.ndarray, lab: np.ndarray):
    """VJP of sum over disagreeing px of (logit_gt - logit_pred)."""
    import torch

    dis = pred != lab
    if not dis.any():
        return np.zeros((H_C, W_C)), float("nan")
    ys, xs = np.where(dis)
    yt = torch.from_numpy(ys)
    xt = torch.from_numpy(xs)
    gt_idx = torch.from_numpy(lab[dis])
    pr_idx = torch.from_numpy(pred[dis].astype(np.int64))
    cure = (logits[gt_idx, yt, xt] - logits[pr_idx, yt, xt]).sum()
    cure.backward()
    gr = x_cam.grad[0].permute(1, 2, 0).numpy()
    heat = (gr ** 2).sum(axis=2)
    gl = gr @ LUMA_HAT
    e_tot = float((gr ** 2).sum())
    luma_frac = float((gl ** 2).sum() / e_tot) if e_tot > 0 else float("nan")
    return heat, luma_frac


def _callouts(dis: np.ndarray, k: int = 5) -> list[dict]:
    """Top-k connected components of the disagreement mask by px count."""
    import cv2

    nlab, cc, stats, cent = cv2.connectedComponentsWithStats(
        dis.astype(np.uint8), connectivity=8)
    comps = []
    for j in range(1, nlab):
        comps.append({"px": int(stats[j, cv2.CC_STAT_AREA]),
                      "cy": float(cent[j][1]), "cx": float(cent[j][0]), "id": j})
    comps.sort(key=lambda d: -d["px"])
    out = []
    for c in comps[:k]:
        m = cc == c["id"]
        out.append({**c, "mask": m})
    return out


def _overlay(cam_u8: np.ndarray, dis: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    import cv2

    dis_up = cv2.resize(dis.astype(np.uint8), (W_C, H_C),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    out = cam_u8.copy()
    out[dis_up] = (np.array([255, 30, 30]) * alpha
                   + out[dis_up] * (1 - alpha)).astype(np.uint8)
    return out


def _render_frame(i: int, gt_f1: np.ndarray, lstars: np.ndarray, frames, masks,
                  model, meta: dict, crop_half: int = 130) -> dict:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = lstars.shape[0]
    gt = np.asarray(gt_f1[i])
    cam = np.asarray(frames[i])
    lab = lstars[i]
    dis = np.unpackbits(np.asarray(masks[i])).reshape(H_S, W_S).astype(bool)
    x_cam, logits, pred, margin = _witness_forward(model, cam)
    signed = margin * np.where(pred == lab, 1.0, -1.0)
    heat, luma_frac = _sens_heat(x_cam, logits, pred, lab)
    dis_next = (np.unpackbits(np.asarray(masks[i + 1])).reshape(H_S, W_S).astype(bool)
                if i + 1 < n else np.zeros_like(dis))
    # flicker composite over dimmed witness frame (seg res)
    cam_s = cv2.resize(cam, (W_S, H_S), interpolation=cv2.INTER_AREA)
    flick = (cam_s * 0.35).astype(np.uint8)
    flick[dis & ~dis_next] = [255, 60, 60]     # now only (dies)
    flick[~dis & dis_next] = [60, 220, 60]     # next only (born)
    flick[dis & dis_next] = [255, 235, 60]     # persists
    calls = _callouts(dis)
    fig, ax = plt.subplots(2, 3, figsize=(19.5, 10.2), dpi=110)
    ax[0, 0].imshow(gt)
    ax[0, 0].set_title(f"GT frame_1 (pair {i})")
    ax[0, 1].imshow(cam)
    ax[0, 1].set_title(f"witness render (mod32cap ep650)  d_seg={dis.mean():.5f}")
    ax[0, 2].imshow(_overlay(cam, dis))
    ax[0, 2].set_title("argmax-diff overlay (red = pred!=GT)")
    vmax = np.percentile(np.abs(signed), 99.0) or 1.0
    im = ax[1, 0].imshow(signed, cmap="RdBu", vmin=-vmax * 0.25, vmax=vmax * 0.25)
    ax[1, 0].contour(signed, levels=[0.0], colors="k", linewidths=0.4)
    ax[1, 0].set_title("signed margin top1-top2 (blue=correct, red=flipped)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.03)
    h = np.log10(heat + heat[heat > 0].min() * 1e-3) if (heat > 0).any() else heat
    ax[1, 1].imshow(h, cmap="inferno")
    ax[1, 1].set_title(f"sensitivity |dmargin/dpx|^2 (log10)  luma_frac={luma_frac:.2f}")
    ax[1, 2].imshow(flick)
    ax[1, 2].set_title("flicker: red=dies, green=born(next), yellow=persists")
    # numbered callouts (seg-res coords; scale for camera-res panels)
    for k, c in enumerate(calls, start=1):
        gtc = int(np.bincount(lab[c["mask"]], minlength=5).argmax())
        prc = int(np.bincount(pred[c["mask"]].astype(np.int64), minlength=5).argmax())
        c["label"] = f"{CLASSES[gtc]}->{CLASSES[prc]}"
        for a, sc in ((ax[0, 2], (W_C / W_S, H_C / H_S)), (ax[1, 2], (1, 1))):
            a.annotate(str(k), (c["cx"] * sc[0], c["cy"] * sc[1]),
                       color="w", fontsize=13, fontweight="bold",
                       bbox={"boxstyle": "circle", "fc": "black", "alpha": 0.6})
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"pair {i} — {meta.get('why', '')} [{meta.get('group', '')}]  "
                 f"(advisory; research_only)", fontsize=11)
    fig.tight_layout()
    mp = os.path.join(OUT_DIR, f"montage_f{i:03d}.png")
    fig.savefig(mp)
    plt.close(fig)
    # zoom crops per callout: GT | witness | overlay at native camera res
    if calls:
        fig2, ax2 = plt.subplots(len(calls), 3,
                                 figsize=(11.5, 3.6 * len(calls)), dpi=110,
                                 squeeze=False)
        ov = _overlay(cam, dis, alpha=0.55)
        for k, c in enumerate(calls):
            cy, cx = int(c["cy"] * H_C / H_S), int(c["cx"] * W_C / W_S)
            y0, y1 = max(cy - crop_half, 0), min(cy + crop_half, H_C)
            x0, x1 = max(cx - int(crop_half * 1.4), 0), min(cx + int(crop_half * 1.4), W_C)
            for j, (img, t) in enumerate(((gt, "GT"), (cam, "witness"),
                                          (ov, "overlay"))):
                ax2[k, j].imshow(img[y0:y1, x0:x1])
                ax2[k, j].set_title(f"#{k + 1} {c['label']} {c['px']}px — {t}",
                                    fontsize=9)
                ax2[k, j].set_xticks([])
                ax2[k, j].set_yticks([])
        fig2.suptitle(f"pair {i} callout crops (native res)", fontsize=11)
        fig2.tight_layout()
        cp = os.path.join(OUT_DIR, f"crops_f{i:03d}.png")
        fig2.savefig(cp)
        plt.close(fig2)
    else:
        cp = None
    return {"frame": i, "montage": mp, "crops": cp, "luma_frac": luma_frac,
            "callouts": [{k: v for k, v in c.items() if k != "mask"} for c in calls]}


def stage_render(only_frames: list[int] | None) -> None:
    free = _free_gib()
    if free < MIN_FREE_GIB:
        raise SystemExit(f"REFUSE: free RAM {free:.1f} GiB < {MIN_FREE_GIB}")
    with open(os.path.join(OUT_DIR, "selection.json")) as fh:
        sel = json.load(fh)["selection"]
    todo: list[tuple[int, dict]] = []
    for group, items in sel.items():
        for it in items:
            if only_frames and it["frame"] not in only_frames:
                continue
            todo.append((it["frame"], {**it, "group": group}))
    todo.sort()
    g = _load_gt(("lstars", "gt_f1"))
    lstars, gt_f1 = g["lstars"], g["gt_f1"]
    n = lstars.shape[0]
    frames = _frames_memmap(n, "r")
    rendered = _rendered_set()
    masks = np.memmap(MASKS_MM, dtype=np.uint8, mode="r", shape=(n, H_S * W_S // 8))
    model = _load_segnet()
    manifest = []
    for i, meta in todo:
        if i not in rendered:
            print(f"[atlas] skip {i}: witness frame not cached", flush=True)
            continue
        r = _render_frame(i, gt_f1, lstars, frames, masks, model, meta)
        manifest.append({**dict(meta.items()), **r})
        print(f"[atlas] rendered pair {i} ({meta['group']}) "
              f"luma_frac={r['luma_frac']:.2f}", flush=True)
    with open(os.path.join(OUT_DIR, "render_manifest.json"), "w") as fh:
        json.dump({"scope": AXIS, "rendered": manifest}, fh, indent=1)
    print(f"[atlas] {len(manifest)} montages -> {OUT_DIR}")


# ---------------------------------------------------------------------------
# Stage LADDER — frozen_scorer_exact_factorization made visual (S1/S2/S3/S4, P2/P3)
# ---------------------------------------------------------------------------
def _A_ops():
    """The EXACT shared scorer resize A (modules.py:109 == :73) + its adjoint.

    A = torch.nn.functional.interpolate(x, size=(384,512), mode='bilinear') on
    (1,3,874,1164). Adjoint via autograd of the REAL call (A is linear), never a
    reimplementation."""
    import torch
    import torch.nn.functional as tfun

    def A(x):
        return tfun.interpolate(x, size=(H_S, W_S), mode="bilinear")

    def At(y):
        x = torch.zeros(1, 3, H_C, W_C, requires_grad=True)
        out = tfun.interpolate(x, size=(H_S, W_S), mode="bilinear")
        out.backward(y)
        return x.grad.detach()

    return torch, A, At


def _range_ker_split(d_cam: np.ndarray, iters: int = 60):
    """Exact orthogonal split d = d_range + d_ker with A d_ker = 0.

    Solves (A A^T) z = A d by CG (A A^T is SPD; bilinear-downsample -> well
    conditioned), d_range = A^T z. Returns (d_range, d_ker, residual_ratio)."""
    torch, A, At = _A_ops()
    d = torch.from_numpy(d_cam.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    b = A(d)

    def AAt(z):
        return A(At(z))

    z = torch.zeros_like(b)
    r = b - AAt(z)
    p = r.clone()
    rs = float((r * r).sum())
    for _ in range(iters):
        Ap = AAt(p)
        alpha = rs / float((p * Ap).sum() + 1e-30)
        z = z + alpha * p
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        if rs_new < 1e-10 * float((b * b).sum() + 1e-30):
            break
        p = r + (rs_new / (rs + 1e-30)) * p
        rs = rs_new
    d_range = At(z)
    d_ker = d - d_range
    num = float((A(d_ker) ** 2).sum()) ** 0.5
    den = float((b ** 2).sum()) ** 0.5 + 1e-30
    to_np = lambda t: t[0].permute(1, 2, 0).numpy()  # noqa: E731
    return to_np(d_range), to_np(d_ker), num / den


def _yuv_diff(d_cam: np.ndarray):
    """Exact BT.601 diff channels per frame_utils.py:51-79 rows (pre-clamp affine):
    dY luma + |(dU,dV)| chroma-plane magnitude."""
    dR, dG, dB = d_cam[..., 0], d_cam[..., 1], d_cam[..., 2]
    dY = 0.299 * dR + 0.587 * dG + 0.114 * dB
    dU = (dB - dY) / 1.772
    dV = (dR - dY) / 1.402
    return dY, np.sqrt(dU ** 2 + dV ** 2)


def _box2(x: np.ndarray) -> np.ndarray:
    """The exact P3 chroma 2x2 box-average (frame_utils.py:65-72)."""
    return 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])


def _classify_callout(cam_rms: float, ratio: float, ker_frac: float) -> str:
    """Heuristic ladder-stage classification (thresholds stated; metrics MEASURED)."""
    if cam_rms < 4.0:
        return "sub-pixel-born"  # flip exists (callouts are flips) w/ near-invisible cam diff
    if ratio < 0.3 or ker_frac > 0.75:
        return "blind-dominant"
    if ratio > 0.7:
        return "faithful"
    return "mixed"


def stage_ladder(only_frames: list[int] | None, pose_frames: list[int]) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    free = _free_gib()
    if free < MIN_FREE_GIB:
        raise SystemExit(f"REFUSE: free RAM {free:.1f} GiB < {MIN_FREE_GIB}")
    with open(os.path.join(OUT_DIR, "selection.json")) as fh:
        sel = json.load(fh)["selection"]
    todo: list[tuple[int, str]] = []
    for group, items in sel.items():
        for it in items:
            if only_frames and it["frame"] not in only_frames:
                continue
            todo.append((it["frame"], group))
    todo.sort()
    g = _load_gt(("lstars", "gt_f1"))
    lstars, gt_f1 = g["lstars"], g["gt_f1"]
    n = lstars.shape[0]
    frames = _frames_memmap(n, "r")
    rendered = _rendered_set()
    masks = np.memmap(MASKS_MM, dtype=np.uint8, mode="r", shape=(n, H_S * W_S // 8))
    model = _load_segnet()
    torch, A, _ = _A_ops()
    rows = []
    for i, group in todo:
        if i not in rendered:
            continue
        gt = np.asarray(gt_f1[i]).astype(np.float32)
        cam = np.asarray(frames[i]).astype(np.float32)
        lab = lstars[i]
        dis = np.unpackbits(np.asarray(masks[i])).reshape(H_S, W_S).astype(bool)
        d = cam - gt
        dY, dC = _yuv_diff(d)
        d_range, d_ker, resid = _range_ker_split(d)
        e_tot = float((d ** 2).sum())
        ker_frac_glob = float((d_ker ** 2).sum() / (e_tot + 1e-30))
        with torch.no_grad():
            a_gt = A(torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0))
            a_cam = A(torch.from_numpy(cam).permute(2, 0, 1).unsqueeze(0))
            logits = model(a_cam)[0]
        a_gt_np = a_gt[0].permute(1, 2, 0).numpy()
        a_cam_np = a_cam[0].permute(1, 2, 0).numpy()
        a_d = a_cam_np - a_gt_np
        top2 = torch.topk(logits, 2, dim=0).values
        margin = (top2[0] - top2[1]).numpy()
        pred = logits.argmax(dim=0).numpy()
        signed = margin * np.where(pred == lab, 1.0, -1.0)
        fig, ax = plt.subplots(3, 5, figsize=(24, 12), dpi=100)
        v8 = max(np.percentile(np.abs(dY), 99.5), 1.0)
        ax[0, 0].imshow(gt.astype(np.uint8))
        ax[0, 0].set_title(f"S1 camera res — GT frame_1 (pair {i})")
        ax[0, 1].imshow(cam.astype(np.uint8))
        ax[0, 1].set_title("S1 camera res — witness")
        im = ax[0, 2].imshow(dY, cmap="RdBu", vmin=-v8, vmax=v8)
        ax[0, 2].set_title(f"dY luma diff (BT.601, LSB, ±{v8:.0f})")
        fig.colorbar(im, ax=ax[0, 2], fraction=0.03)
        im = ax[0, 3].imshow(dC, cmap="magma", vmin=0, vmax=v8)
        ax[0, 3].set_title("|(dU,dV)| chroma-plane diff (frame_utils rows)")
        fig.colorbar(im, ax=ax[0, 3], fraction=0.03)
        im = ax[0, 4].imshow(np.abs(d).mean(axis=2), cmap="magma", vmin=0, vmax=v8)
        ax[0, 4].set_title("|d| mean RGB (LSB)")
        fig.colorbar(im, ax=ax[0, 4], fraction=0.03)
        im = ax[1, 0].imshow(np.abs(d_range).mean(axis=2), cmap="magma", vmin=0, vmax=v8)
        ax[1, 0].set_title(f"S2 range(A^T) part — SEEN by both scorers "
                           f"({1 - ker_frac_glob:.0%} energy)")
        fig.colorbar(im, ax=ax[1, 0], fraction=0.03)
        im = ax[1, 1].imshow(np.abs(d_ker).mean(axis=2), cmap="magma", vmin=0, vmax=v8)
        ax[1, 1].set_title(f"S2 ker(A) part — BLIND B1 ({ker_frac_glob:.0%} energy; "
                           f"|A d_ker|/|A d|={resid:.1e})")
        fig.colorbar(im, ax=ax[1, 1], fraction=0.03)
        ax[1, 2].axis("off")
        ax[1, 2].text(0.02, 0.95,
                      f"pair {i} [{group}]\n"
                      f"camera diff energy: {e_tot:.3g}\n"
                      f"ker(A) fraction: {ker_frac_glob:.1%} (blind, costs 0 d_seg)\n"
                      f"range fraction: {1 - ker_frac_glob:.1%} (reaches S3)\n"
                      f"CG residual |A d_ker|/|A d| = {resid:.2e}\n"
                      f"stride-2 stem (S3): structure below ~2px @(384,512)\n"
                      f"  further attenuated before block 1\n"
                      f"[MEASURED; exact upstream A]",
                      va="top", fontsize=11, family="monospace")
        ax[1, 3].axis("off")
        ax[1, 4].axis("off")
        ax[2, 0].imshow(np.clip(a_gt_np, 0, 255).astype(np.uint8))
        ax[2, 0].set_title("S2 A.GT (384x512)")
        ax[2, 1].imshow(np.clip(a_cam_np, 0, 255).astype(np.uint8))
        ax[2, 1].set_title("S2 A.witness (384x512)")
        im = ax[2, 2].imshow(np.abs(a_d).mean(axis=2), cmap="magma", vmin=0, vmax=v8)
        ax[2, 2].set_title("|A d| — what S3 actually receives")
        fig.colorbar(im, ax=ax[2, 2], fraction=0.03)
        vm = np.percentile(np.abs(signed), 99.0) or 1.0
        im = ax[2, 3].imshow(signed, cmap="RdBu", vmin=-vm * 0.25, vmax=vm * 0.25)
        ax[2, 3].set_title("S3 signed margin (top1-top2)")
        fig.colorbar(im, ax=ax[2, 3], fraction=0.03)
        ov = (a_cam_np * 0.4).astype(np.uint8)
        ov[dis] = [255, 40, 40]
        ax[2, 4].imshow(ov)
        ax[2, 4].set_title("S4 argmax-diff (the flips)")
        # per-callout ladder metrics + classification
        calls = _callouts(dis)
        lines = []
        for k, c in enumerate(calls, start=1):
            m = c["mask"]
            ys, xs = np.where(m)
            m_up = cv2.resize(m.astype(np.uint8), (W_C, H_C),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            dm = d[m_up]
            cam_rms = float(np.sqrt((dm ** 2).mean())) if dm.size else 0.0
            pa = a_d[m]
            postA_rms = float(np.sqrt((pa ** 2).mean())) if pa.size else 0.0
            dk = d_ker[m_up]
            ker_frac = (float((dk ** 2).sum() / ((dm ** 2).sum() + 1e-30))
                        if dm.size else 0.0)
            ratio = postA_rms / (cam_rms + 1e-30)
            gtc = int(np.bincount(lab[m], minlength=5).argmax())
            prc = int(np.bincount(pred[m].astype(np.int64), minlength=5).argmax())
            cls = _classify_callout(cam_rms, ratio, ker_frac)
            lines.append(f"#{k} {CLASSES[gtc]}->{CLASSES[prc]} {c['px']}px "
                         f"cam_rms={cam_rms:.1f}LSB postA={postA_rms:.1f} "
                         f"r={ratio:.2f} ker={ker_frac:.0%} -> {cls}")
            rows.append({"frame": i, "group": group, "callout": k,
                         "gt": CLASSES[gtc], "pred": CLASSES[prc], "px": c["px"],
                         "cam_rms_lsb": cam_rms, "postA_rms_lsb": postA_rms,
                         "survival_ratio": ratio, "ker_energy_frac": ker_frac,
                         "born": cls})
            ax[2, 4].annotate(str(k), (c["cx"], c["cy"]), color="w", fontsize=12,
                              fontweight="bold",
                              bbox={"boxstyle": "circle", "fc": "black", "alpha": 0.6})
        ax[1, 3].axis("off")
        ax[1, 3].text(0.0, 0.95, "flip-born ladder classification\n"
                      "(MEASURED metrics; heuristic thresholds:\n"
                      " sub-pixel cam_rms<4LSB / blind r<0.3|ker>75% / faithful r>0.7)\n\n"
                      + "\n".join(lines), va="top", fontsize=9, family="monospace")
        for a in ax.ravel():
            if a.axison:
                a.set_xticks([])
                a.set_yticks([])
        fig.suptitle(f"RESOLUTION LADDER pair {i} [{group}] — "
                     f"frozen_scorer_exact_factorization S1/S2/S3/S4 made visual "
                     f"(exact upstream ops; advisory; research_only)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"ladder_f{i:03d}.png"))
        plt.close(fig)
        print(f"[atlas] ladder pair {i}: ker_frac={ker_frac_glob:.1%} "
              f"resid={resid:.1e}", flush=True)
        # optional P2/P3 pose-path figure
        if i in pose_frames:
            dYs, _ = _yuv_diff(a_d)
            dRs, dBs = a_d[..., 0], a_d[..., 2]
            dU_s = (dBs - dYs) / 1.772
            dV_s = (dRs - dYs) / 1.402
            dC_pre = np.sqrt(dU_s ** 2 + dV_s ** 2)
            dC_post = np.sqrt(_box2(dU_s) ** 2 + _box2(dV_s) ** 2)
            figp, axp = plt.subplots(1, 3, figsize=(16, 4.6), dpi=110)
            vv = max(np.percentile(dC_pre, 99.5), 0.5)
            im = axp[0].imshow(dC_pre, cmap="magma", vmin=0, vmax=vv)
            axp[0].set_title("P2: |(dU,dV)| at (384,512) pre box-avg")
            figp.colorbar(im, ax=axp[0], fraction=0.03)
            im = axp[1].imshow(dC_post, cmap="magma", vmin=0, vmax=vv)
            axp[1].set_title("P3: after exact 2x2 BOX-AVERAGE (192,256) — "
                             "what PoseNet sees")
            figp.colorbar(im, ax=axp[1], fraction=0.03)
            surv = float((dC_post ** 2).sum() / ((dC_pre ** 2).sum() / 4 + 1e-30))
            axp[2].axis("off")
            axp[2].text(0.02, 0.9,
                        f"pair {i} pose chroma path (P2/P3)\n"
                        f"chroma diff energy surviving box-avg: {surv:.1%}\n"
                        f"(luma path = 2x2 space-to-depth, LOSSLESS)\n"
                        f"sub-2px chroma structure -> INVISIBLE to PoseNet\n"
                        f"[MEASURED; exact frame_utils.rgb_to_yuv6 ops]",
                        va="top", fontsize=11, family="monospace")
            for a in axp[:2]:
                a.set_xticks([])
                a.set_yticks([])
            figp.suptitle(f"POSE PATH pair {i} — P2/P3 chroma box-average "
                          f"(pose-safety of fine chroma made visual)", fontsize=11)
            figp.tight_layout()
            figp.savefig(os.path.join(OUT_DIR, f"posepath_f{i:03d}.png"))
            plt.close(figp)
    with open(os.path.join(OUT_DIR, "ladder_manifest.json"), "w") as fh:
        json.dump({"scope": AXIS + "; exact upstream interpolate/rgb_to_yuv6 ops; "
                            "CG orthogonal ker/range split",
                   "classify_thresholds": {"sub_pixel_cam_rms_lsb": 4.0,
                                           "blind_ratio": 0.3, "blind_ker_frac": 0.75,
                                           "faithful_ratio": 0.7},
                   "callouts": rows}, fh, indent=1)
    print(f"[atlas] ladder: {len(rows)} callout rows -> ladder_manifest.json")


# ---------------------------------------------------------------------------
# Stage LANE — degraded-marking mechanism panels (operator scene physics x
# measured Lane head gain; memory degraded_lane_markings_x_lane_head_gain_...)
# ---------------------------------------------------------------------------
def _pair_boundary(lab: np.ndarray, a: int, b: int) -> np.ndarray:
    """Pixels of class a or b that 4-touch the other class (the a-b separatrix)."""
    ma, mb = lab == a, lab == b
    t = np.zeros_like(ma)
    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        t |= ma & np.roll(mb, sh, axis=ax)
        t |= mb & np.roll(ma, sh, axis=ax)
    # np.roll wraps: clear 1px image border so no spurious cross-border pairs
    t[0, :] = t[-1, :] = False
    t[:, 0] = t[:, -1] = False
    return t


def _dash_blob_stats(lab: np.ndarray) -> dict:
    """Lane GT connected components: ellipse aspect/tilt + centroid spacing CV
    (irregularity measure for the #287 dash-comb regularity assumption)."""
    import cv2

    m = (lab == 1).astype(np.uint8)
    nlab, cc, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    blobs = []
    for j in range(1, nlab):
        if stats[j, cv2.CC_STAT_AREA] < 8:
            continue
        pts = np.column_stack(np.where(cc == j)[::-1]).astype(np.float32)
        e = {"area": int(stats[j, cv2.CC_STAT_AREA]),
             "cx": float(cent[j][0]), "cy": float(cent[j][1])}
        if len(pts) >= 5:
            (_, _), (w, h), ang = cv2.fitEllipse(pts)
            major, minor = max(w, h), max(min(w, h), 1e-3)
            e["aspect"] = float(major / minor)
            e["tilt_from_vertical_deg"] = float(min(abs(ang % 180), abs(180 - ang % 180))
                                                if w < h else abs(90 - ang % 180))
        blobs.append(e)
    blobs.sort(key=lambda d: -d["area"])
    # spacing along the dominant (y-sorted) chain of blobs
    spacing = []
    import itertools

    bl = sorted(blobs, key=lambda d: d["cy"])
    for u, v in itertools.pairwise(bl):
        spacing.append(float(np.hypot(u["cx"] - v["cx"], u["cy"] - v["cy"])))
    cv_sp = (float(np.std(spacing) / np.mean(spacing))
             if len(spacing) >= 2 and np.mean(spacing) > 0 else float("nan"))
    return {"n_blobs": len(blobs), "blobs": blobs[:12],
            "spacing_px": spacing, "spacing_cv": cv_sp}


def stage_lane(only_frames: list[int] | None) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(OUT_DIR, "selection.json")) as fh:
        sel = json.load(fh)["selection"]
    todo = [it["frame"] for grp in ("roadlane", "worst_overall")
            for it in sel.get(grp, [])]
    if only_frames:
        todo = [f for f in todo if f in only_frames]
    g = _load_gt(("lstars", "gt_f0", "gt_f1", "margins"))
    lstars, gt_f0, gt_f1, gmargins = (g["lstars"], g["gt_f0"], g["gt_f1"],
                                      g["margins"])
    n = lstars.shape[0]
    frames = _frames_memmap(n, "r")
    masks = np.memmap(MASKS_MM, dtype=np.uint8, mode="r", shape=(n, H_S * W_S // 8))
    model = _load_segnet()
    from necessity_dseg_calibration import _segnet_argmax

    out_rows = []
    for i in sorted(set(todo)):
        lab = lstars[i]
        gt = np.asarray(gt_f1[i])
        cam = np.asarray(frames[i])
        dis = np.unpackbits(np.asarray(masks[i])).reshape(H_S, W_S).astype(bool)
        marg = np.asarray(gmargins[i])
        b_rl = _pair_boundary(lab, 0, 1)
        b_ru = _pair_boundary(lab, 0, 2)
        m_rl = marg[b_rl]
        m_ru = marg[b_ru]
        stats = _dash_blob_stats(lab)
        # GT self-flicker: SegNet argmax on GT frame_0 vs cached GT argmax(frame_1)
        pred0 = _segnet_argmax(model, np.asarray(gt_f0[i])[None])[0]
        selfflk = pred0 != lab
        lane_band = cv2.dilate(b_rl.astype(np.uint8),
                               np.ones((5, 5), np.uint8)).astype(bool)
        sf_lane = int((selfflk & lane_band).sum())
        wit_lane = dis & lane_band
        overlap = (float((wit_lane & selfflk).sum() / wit_lane.sum())
                   if wit_lane.any() else float("nan"))
        # figure
        fig, ax = plt.subplots(2, 3, figsize=(19, 10), dpi=110)
        # biggest lane blob region -> camera-res crop with GT lane contours
        ys, xs = np.where(lab == 1)
        if len(ys):
            cy, cx = int(np.median(ys)), int(np.median(xs))
        else:
            cy, cx = H_S // 2, W_S // 2
        cyc, cxc = int(cy * H_C / H_S), int(cx * W_C / W_S)
        y0, y1 = max(cyc - 220, 0), min(cyc + 220, H_C)
        x0, x1 = max(cxc - 320, 0), min(cxc + 320, W_C)
        lane_up = cv2.resize((lab == 1).astype(np.uint8), (W_C, H_C),
                             interpolation=cv2.INTER_NEAREST)
        crop_gt = gt[y0:y1, x0:x1].copy()
        cont, _ = cv2.findContours(lane_up[y0:y1, x0:x1], cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop_gt, cont, -1, (0, 255, 90), 1)
        ax[0, 0].imshow(crop_gt)
        ax[0, 0].set_title(f"GT lane markings + GT-argmax contour (pair {i})")
        ax[0, 1].imshow(gt[y0:y1, x0:x1])
        ax[0, 1].set_title("GT (clean view) — NAME the paint condition by eye")
        ax[0, 2].imshow(cam[y0:y1, x0:x1])
        ax[0, 2].set_title("witness render, same crop")
        # margin overlay along boundaries (seg-res, lower 2/3)
        mo = (cv2.resize(gt, (W_S, H_S)) * 0.3).astype(np.uint8)
        mv = np.clip(marg / 8.0, 0, 1)
        cmap = plt.get_cmap("plasma")
        col = (cmap(mv)[..., :3] * 255).astype(np.uint8)
        band = cv2.dilate((b_rl | b_ru).astype(np.uint8),
                          np.ones((3, 3), np.uint8)).astype(bool)
        mo[band] = col[band]
        ax[1, 0].imshow(mo)
        ax[1, 0].set_title("GT margin |m| along separatrices (plasma, 0..8 logits)")
        ax[1, 1].hist([m_rl, m_ru], bins=40, range=(0, 12), density=True,
                      label=[f"Road-Lane (med {np.median(m_rl):.2f})" if m_rl.size
                             else "Road-Lane (none)",
                             f"Road-Undriv (med {np.median(m_ru):.2f})" if m_ru.size
                             else "Road-Undriv (none)"],
                      color=["#d4a017", "#4477aa"], histtype="step", lw=2)
        ax[1, 1].legend(fontsize=9)
        ax[1, 1].set_title("GT boundary margin dist — faint paint = low |m|;\n"
                           "Lane head gain ||dw|| 3.75-4.01 (MEASURED) -> flip "
                           "d=|m|/||dw|| cheapest")
        sfo = (cv2.resize(gt, (W_S, H_S)) * 0.35).astype(np.uint8)
        sfo[selfflk & ~dis] = [60, 220, 60]
        sfo[dis & ~selfflk] = [255, 60, 60]
        sfo[selfflk & dis] = [255, 235, 60]
        ax[1, 2].imshow(sfo)
        ax[1, 2].set_title(f"GT SELF-FLICKER (SegNet on GT f0 vs f1 argmax): green=GT-self, "
                           f"red=witness-only, yellow=both\nlane-band self-flips {sf_lane}px; "
                           f"witness-lane overlap {overlap:.0%} (GT-floor bound)")
        for a in ax.ravel():
            if a is not ax[1, 1]:
                a.set_xticks([])
                a.set_yticks([])
        tilts = [b.get("tilt_from_vertical_deg") for b in stats["blobs"]
                 if "tilt_from_vertical_deg" in b]
        aspects = [b.get("aspect") for b in stats["blobs"] if "aspect" in b]
        fig.suptitle(
            f"LANE DEGRADATION pair {i} — blobs {stats['n_blobs']}, spacing CV "
            f"{stats['spacing_cv']:.2f} (regular comb ~0), "
            f"tilt med {np.median(tilts):.0f}deg, aspect med {np.median(aspects):.1f} "
            f"[MEASURED from GT argmax] (advisory; research_only)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"lane_f{i:03d}.png"))
        plt.close(fig)
        out_rows.append({
            "frame": i, "median_margin_roadlane": float(np.median(m_rl)) if m_rl.size
            else None,
            "median_margin_roadundriv": float(np.median(m_ru)) if m_ru.size else None,
            "gt_selfflicker_laneband_px": sf_lane,
            "witness_lane_overlap_with_selfflicker": overlap,
            "dash_spacing_cv": stats["spacing_cv"],
            "blob_tilt_med_deg": float(np.median(tilts)) if tilts else None,
            "blob_aspect_med": float(np.median(aspects)) if aspects else None,
            "n_lane_blobs": stats["n_blobs"]})
        print(f"[atlas] lane pair {i}: med|m|RL="
              f"{out_rows[-1]['median_margin_roadlane']:.2f} spacingCV="
              f"{stats['spacing_cv']:.2f} selfflk={sf_lane}px overlap={overlap:.0%}",
              flush=True)
    with open(os.path.join(OUT_DIR, "lane_manifest.json"), "w") as fh:
        json.dump({"scope": AXIS + "; GT margins from gt_n600 cache; GT self-flicker "
                            "= SegNet(gt_f0) vs cached argmax(gt_f1) — upper-bounds "
                            "label noise + 1-frame advection",
                   "rows": out_rows}, fh, indent=1)
    print(f"[atlas] lane: {len(out_rows)} rows -> lane_manifest.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=("select", "render", "ladder", "lane"))
    ap.add_argument("--per-bucket", type=int, default=4)
    ap.add_argument("--n-worst", type=int, default=3)
    ap.add_argument("--n-easy", type=int, default=3)
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--pose-frames", type=int, nargs="*", default=[])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage == "select":
        stage_select(args.per_bucket, args.n_worst, args.n_easy)
    elif args.stage == "render":
        stage_render(args.frames)
    elif args.stage == "lane":
        stage_lane(args.frames)
    else:
        stage_ladder(args.frames, args.pose_frames)


if __name__ == "__main__":
    main()
