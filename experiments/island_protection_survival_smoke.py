# SPDX-License-Identifier: MIT
"""$0 island-SURVIVAL smoke for the EARLY-SEED + CONTAINMENT stack (NO paid GPU).

Measures, through the REAL FROZEN CPU-torch SegNet argmax (the authority, NEVER
MLX/MPS) on the REAL n600 GT cache, whether:

  1. EARLY-SEED births the finest-scale islands: on an ERASED (low-pass) base that
     drops the finest scale (a faithful surrogate for the witness's measured
     spectral-bias erasure of lane dashes / small movables), does compositing the
     SPARSE seed (GT island appearance on the self-detected island band ONLY) RECOVER
     the island-class argmax recall through the frozen scorer?  A NON-tautological
     test: the seed is a LOCAL sparse paste on a degraded context, so the scorer's
     stride-2 receptive field must still BIRTH the class from the local evidence.

  2. CONTAINMENT protects the seeded islands from the bulk wash: after seeding, a
     bulk-class smoothing pass (the bulk-CE-dominated optimizer pulling every pixel
     toward the locally-dominant bulk appearance) WASHES the islands. WITHOUT
     containment the wash hits the island pixels -> recall collapses; WITH containment
     (freeze the protected seed pixels) the islands survive.

Reference axis: the seg RECALL is [contest-CPU advisory] (frozen CPU-torch SegNet
argmax; the SAME authority as evaluate.py's SegNet, on a controlled erasure model —
the erasure is a surrogate, the BIRTH verdict is the real scorer). Deterministic,
seeded, disk-hygienic (JSON -> experiments/results/, never /tmp). promotion_eligible
=False; this is training-time infrastructure validation, not a score row.

Run (n600 real; batched SegNet):
    .venv/bin/python experiments/island_protection_survival_smoke.py --n 600
    .venv/bin/python experiments/island_protection_survival_smoke.py --n 48   # fast dev-loop
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.boundary_math import island_protection as ip  # noqa: E402

SEG_H, SEG_W = 384, 512


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nn_resize(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resize (H,W[,C]) — deterministic, dependency-light."""
    ys = np.linspace(0, a.shape[0] - 1, out_h).round().astype(np.int64)
    xs = np.linspace(0, a.shape[1] - 1, out_w).round().astype(np.int64)
    return a[np.ix_(ys, xs)] if a.ndim == 2 else a[np.ix_(ys, xs)]


def _lowpass(frame: np.ndarray, factor: int) -> np.ndarray:
    """Erase the finest scale: down-by-`factor` (block-mean) then NN up. uint8 in/out.

    A faithful surrogate for the witness's finest-scale spectral-bias erasure: the
    high-frequency lane dashes / small movables (period < factor px) are removed."""
    h, w, c = frame.shape
    hc, wc = h // factor, w // factor
    crop = frame[: hc * factor, : wc * factor, :].astype(np.float32)
    blocks = crop.reshape(hc, factor, wc, factor, c).mean(axis=(1, 3))  # (hc,wc,c) block-mean
    up = np.repeat(np.repeat(blocks, factor, axis=0), factor, axis=1)   # (hc*factor, wc*factor, c)
    out = frame.astype(np.float32).copy()
    out[: hc * factor, : wc * factor, :] = up
    return np.clip(out, 0, 255).astype(np.uint8)


def _bulk_wash(frame: np.ndarray, k: int = 5) -> np.ndarray:
    """Bulk smoothing pass (box blur) — the bulk-CE wash that erases islands. uint8."""
    f = frame.astype(np.float32)
    pad = k // 2
    fp = np.pad(f, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(f)
    for dy in range(k):
        for dx in range(k):
            acc += fp[dy : dy + f.shape[0], dx : dx + f.shape[1], :]
    return np.clip(acc / (k * k), 0, 255).astype(np.uint8)


def _segnet_argmax_batch(segnet, frames_seg_uint8: list[np.ndarray], batch: int) -> list[np.ndarray]:
    """Batched frozen CPU-torch SegNet argmax on SEG-grid frames. Mirrors
    cpu_verdict_d_seg's preprocess/forward EXACTLY (the authority)."""
    import torch

    out: list[np.ndarray] = []
    for s in range(0, len(frames_seg_uint8), batch):
        chunk = frames_seg_uint8[s : s + batch]
        arr = np.stack(chunk, axis=0)                         # (B,H,W,3)
        pair = torch.from_numpy(np.stack([arr, arr], axis=1)).float()  # (B,2,H,W,3)
        xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()          # (B,2,3,H,W)
        with torch.inference_mode():
            seg_in = segnet.preprocess_input(xp)
            logits = segnet(seg_in)                           # (B,C,H,W)
            am = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        out.extend([am[i] for i in range(am.shape[0])])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=48, help="number of pairs (600 = full n600 evidence)")
    ap.add_argument("--gt-cache", type=str,
                    default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--erase-factor", type=int, default=4, help="low-pass block size (finest-scale erasure)")
    ap.add_argument("--dilate-px", type=int, default=1, help="island annulus dilation")
    ap.add_argument("--wash-k", type=int, default=5, help="bulk-wash box-blur kernel")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)

    cache = REPO / args.gt_cache
    if not cache.exists():
        # fall back to a smaller cache if the requested one is absent
        for alt in ("gt_n600.npz", "gt_n96.npz", "gt_n24.npz"):
            p = REPO / "experiments/results/mlx_fleet_gt_cache" / alt
            if p.exists():
                cache = p
                break
    z = np.load(cache, allow_pickle=False)
    cached = int(z["n_pairs"])
    N = min(int(args.n), cached)
    print(json.dumps({"stage": "load", "cache": str(cache), "cached_pairs": cached, "using": N,
                      "utc": _utc()}), flush=True)

    gt_f1_cam = z["gt_f1"]  # (cached,874,1164,3) uint8
    lstars_cached = z["lstars"]  # (cached,384,512)

    from tac.boundary_math.seg_core import load_real_segnet

    seg = load_real_segnet("cpu")

    # SEG-grid clean frames (downsample camera -> 384x512) — the smoke's frame grid.
    t0 = time.time()
    gt_seg = [_nn_resize(np.asarray(gt_f1_cam[i]), SEG_H, SEG_W).astype(np.uint8) for i in range(N)]

    # BASELINE: SegNet on the clean seg-grid frame -> the reference argmax (recall=1 here).
    ref = _segnet_argmax_batch(seg, gt_seg, args.batch)
    ref_stack = np.stack(ref, axis=0)
    print(json.dumps({"stage": "baseline_forward_done", "sec": round(time.time() - t0, 1)}), flush=True)

    # SELF-DETECT islands on the reference argmax stack (never hardcode).
    det = ip.identify_island_classes(ref_stack)
    print(json.dumps({"stage": "island_detect", "lane_cls": det.lane_cls, "movable_cls": det.movable_cls,
                      "island_classes": list(det.island_classes),
                      "evidence": [{"cls": e.cls, "area": round(e.area_frac, 4), "iou": round(e.static_iou, 3),
                                    "thick": round(e.mean_thickness_px, 2), "island": e.is_island,
                                    "kind": e.island_kind} for e in det.evidence]}), flush=True)

    # Build per-frame masks + seeds + wash frames.
    erased, seeded, wash_uncontained, wash_contained = [], [], [], []
    masks_list: list[ip.IslandMasks] = []
    for i in range(N):
        m = ip.build_island_masks(ref[i], det.lane_cls, det.movable_cls, dilate_px=args.dilate_px)
        masks_list.append(m)
        er = _lowpass(gt_seg[i], args.erase_factor)
        erased.append(er)
        # EARLY-SEED: composite GT island appearance onto the erased base (sparse).
        seed = ip.build_island_seed(gt_seg[i].astype(np.float32), m,
                                    base_render_segres=er.astype(np.float32))
        seed_frame = np.clip(ip.compose_seed(er.astype(np.float32), seed.residual), 0, 255).astype(np.uint8)
        seeded.append(seed_frame)
        # BULK WASH: blur the seeded frame. UNCONTAINED = wash everywhere.
        washed = _bulk_wash(seed_frame, args.wash_k)
        wash_uncontained.append(washed)
        # CONTAINED = freeze the protected island pixels back to the seed value.
        cont = washed.copy()
        cont[m.any_mask] = seed_frame[m.any_mask]
        wash_contained.append(cont)

    # Forwards for each condition.
    conds = {}
    for name, frames in (("erased", erased), ("seeded", seeded),
                         ("wash_uncontained", wash_uncontained), ("wash_contained", wash_contained)):
        tt = time.time()
        preds = _segnet_argmax_batch(seg, frames, args.batch)
        conds[name] = np.stack(preds, axis=0)
        print(json.dumps({"stage": "forward", "cond": name, "sec": round(time.time() - tt, 1)}), flush=True)

    # Per-condition island RECALL (mean over frames, per island class).
    def recall_over(cond_stack: np.ndarray, cls: int) -> float:
        rs = [ip.island_recall(cond_stack[i], ref[i], cls) for i in range(N)]
        rs = [r for r in rs if not np.isnan(r)]
        return float(np.mean(rs)) if rs else float("nan")

    summary = {"n": N, "erase_factor": args.erase_factor, "wash_k": args.wash_k, "dilate_px": args.dilate_px}
    for kind, cls in (("lane", det.lane_cls), ("movable", det.movable_cls)):
        if cls is None:
            continue
        summary[kind] = {
            "baseline": round(recall_over(ref_stack, cls), 4),   # == 1.0 by construction
            "erased": round(recall_over(conds["erased"], cls), 4),
            "seeded": round(recall_over(conds["seeded"], cls), 4),
            "wash_uncontained": round(recall_over(conds["wash_uncontained"], cls), 4),
            "wash_contained": round(recall_over(conds["wash_contained"], cls), 4),
        }
        s = summary[kind]
        s["seed_birth_gain"] = round(s["seeded"] - s["erased"], 4)          # EARLY-SEED effect
        s["containment_gain"] = round(s["wash_contained"] - s["wash_uncontained"], 4)  # CONTAINMENT effect

    out = {"schema": "island_protection_survival_smoke.v1", "utc": _utc(),
           "cache": str(cache), "authority": "frozen_cpu_torch_segnet_argmax",
           "evidence_axis": "[contest-CPU advisory] (surrogate erasure; real scorer birth verdict)",
           "promotion_eligible": False, "island_detect": {"lane_cls": det.lane_cls, "movable_cls": det.movable_cls},
           "summary": summary}
    print(json.dumps({"stage": "RESULT", **summary}), flush=True)

    out_path = Path(args.out) if args.out else (
        REPO / "experiments/results" / f"island_protection_survival_n{N}_{_utc().replace(':', '').replace('-', '')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"stage": "wrote", "path": str(out_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
