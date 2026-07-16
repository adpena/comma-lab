# SPDX-License-Identifier: MIT
"""Necessity §9 calibration — DP-ε seed → d_seg through the REAL scorer chain (n600).

Converts the necessity solver's K-ladder geometric seed (DP-simplified per-class region
contours; ``tools/necessity_inverse_factorization_solver.py`` stage KLADDER) into a REAL
measured d_seg by actually REALIZING the witness from the (generator, seed) design and
pushing it through the real scorer preprocess + frozen CPU-torch SegNet:

  seed        DP-simplified per-class contours at eps px @ (384,512)  [COUNTED bytes]
  generator   closed-polygon rasterizer (fillPoly even-odd) + paint-order overwrite
              (decreasing class area) + nearest-assigned gap fill + per-class palette
              fill + INTER_NEAREST upsample to camera res (874,1164)  [FREE, rule 118]
  chain       uint8 camera frame -> float -> bilinear (384,512) -> frozen SegNet ->
              argmax  vs  cached GT argmax (gt_n600.npz lstars, bit-exact)
              == the exact upstream/modules.py SegnetWrapper preprocess (raw 0-255,
              no normalization; verified: stage_vjp parity 0.0 vs cached margins).

Outputs per eps in {0 (exact partition), 0.5, 1.0, 2.0}:
  d_seg_real   mean over n600 frames of per-pixel argmax disagreement (THE d_seg term)
  d_seg_geo    reconstructed-partition vs GT-partition disagreement (DP geometry only)
  per-stratum attribution of disagreeing pixels (saddle flank > 1px edge (by class
  pair) > near-edge <=3px > far interior (by GT class))
  seed bytes   SAME coder as kladder (first vertex u16 + int16 deltas, brotli -q11),
              raw single-stream AND /2 shared-edge-adjusted
  predicted S  100*d_seg_real + sqrt(10*d_pose_banked) + 25*(seed+palette+dxi)/N_rate

Axes / honesty: ``[macOS-CPU advisory]`` — frozen CPU-torch fp32 scorer on the exact
cached GT argmax; the score-authority follow-on is ``upstream/evaluate.py`` on the
byte-closed archive (operator-GO'd). research_only; score_claim=false; promotable=false.
d_pose is the BANKED witness R1 dxi row (0.001610, 7.2KB), not re-measured here: the
palette render carries no pose-legible photometric signal by construction; composing
with the joint-descent dxi vehicle is the named integration step, labeled in the memo.

Resumable: per-eps JSONL rows (one per frame); rerun skips completed frames. Stages:
  palette   choose per-class palette (mean vs median, stride subset smoke pick)
  measure   chunked n600 SegNet sweep for one eps (--eps, --limit for chunking)
  finalize  seed-byte coding (fast, no SegNet) + aggregation + S table -> summary.json

Usage:
  .venv/bin/python tools/necessity_dseg_calibration.py --stage palette
  .venv/bin/python tools/necessity_dseg_calibration.py --stage measure --eps 1.0
  .venv/bin/python tools/necessity_dseg_calibration.py --stage finalize
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

GT_N600 = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
OUT_DIR = "experiments/results/necessity_dseg_calibration_20260715"
CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
# paint order: decreasing measured interior area (strat.json) so small classes overwrite
PAINT_ORDER = (2, 4, 0, 3, 1)  # Undrivable, MyCar, Road, Movable, Lane
H_S, W_S = 384, 512
H_C, W_C = 874, 1164
N_RATE = 37_545_489
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; bit-exact cached GT (gt_n600.npz)"
D_POSE_BANKED = 0.001610  # R1 dxi through byte-close (L68); contribution sqrt(10*d_pose)
DXI_BYTES_BANKED = 7200   # R1 dxi section (7.2KB)
PALETTE_BYTES = 15        # 5 classes x 3 bytes
EPS_LIST = (0.0, 0.5, 1.0, 2.0)


def _pair_code(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return lo * 5 + hi


def _pair_name(code: int) -> str:
    return f"{CLASSES[code // 5]}-{CLASSES[code % 5]}"


def _class_polys(lab: np.ndarray, c: int, eps: float) -> list[np.ndarray]:
    """DP-simplified contours for class c — SAME conventions as stage_kladder."""
    import cv2

    mask = (lab == c).astype(np.uint8)
    if not mask.any():
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cont in contours:
        if len(cont) < 3:
            continue
        if eps > 0.0:
            ap = cv2.approxPolyDP(cont, eps, True)[:, 0, :]
        else:
            ap = cont[:, 0, :]
        polys.append(ap.astype(np.int32))
    return polys


def _reconstruct(lab: np.ndarray, eps: float) -> np.ndarray:
    """Rasterize the DP seed back to a full partition at (384,512)."""
    import cv2
    from scipy.ndimage import distance_transform_edt

    if eps <= 0.0:
        return lab.astype(np.uint8)  # exact-partition baseline (pure value realization)
    recon = np.full((H_S, W_S), 255, np.uint8)
    for c in PAINT_ORDER:
        polys = _class_polys(lab, c, eps)
        if not polys:
            continue
        cmask = np.zeros((H_S, W_S), np.uint8)
        cv2.fillPoly(cmask, polys, 1)  # even-odd fill handles hole contours
        recon[cmask.astype(bool)] = c
    if (recon == 255).any():
        inds = distance_transform_edt(recon == 255, return_distances=False,
                                      return_indices=True)
        recon = recon[inds[0], inds[1]]
    return recon


def _strata_masks(lab: np.ndarray):
    """GT stratum masks: saddle-flank, 1px edge (+pair id), near-edge<=3px."""
    import cv2

    dx = lab[:, :-1] != lab[:, 1:]
    dy = lab[:-1, :] != lab[1:, :]
    bmask = np.zeros_like(lab, dtype=bool)
    bmask[:, :-1] |= dx
    bmask[:, 1:] |= dx
    bmask[:-1, :] |= dy
    bmask[1:, :] |= dy
    pair_id = np.full((H_S, W_S), 255, np.uint8)
    for m_, (sl_a, sl_b) in ((dx, (np.s_[:, :-1], np.s_[:, 1:])),
                            (dy, (np.s_[:-1, :], np.s_[1:, :]))):
        code = _pair_code(lab[sl_a][m_], lab[sl_b][m_]).astype(np.uint8)
        for sl in (sl_a, sl_b):
            sub = pair_id[sl]
            sub[m_] = code
            pair_id[sl] = sub
    blk = np.stack([lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]])
    bs = np.sort(blk, axis=0)
    distinct = 1 + (bs[1] != bs[0]).astype(np.int8) + (bs[2] != bs[1]) + (bs[3] != bs[2])
    smask = np.zeros_like(lab, dtype=bool)
    ys, xs = np.where(distinct >= 3)
    for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1)):
        smask[ys + oy, xs + ox] = True
    near = cv2.dilate(bmask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return smask, bmask, pair_id, near


def _attribute(dis: np.ndarray, lab: np.ndarray, smask, bmask, pair_id, near) -> dict:
    out: dict = {}
    out["saddle"] = int((dis & smask).sum())
    edge = dis & bmask & ~smask
    out["edge_by_pair"] = {}
    if edge.any():
        codes, counts = np.unique(pair_id[edge], return_counts=True)
        for c, n in zip(codes, counts, strict=True):
            if int(c) != 255:
                out["edge_by_pair"][_pair_name(int(c))] = int(n)
    out["near_edge_3px"] = int((dis & near & ~bmask & ~smask).sum())
    far = dis & ~near & ~smask
    out["far_interior_by_class"] = {}
    if far.any():
        cls, counts = np.unique(lab[far], return_counts=True)
        for c, n in zip(cls, counts, strict=True):
            out["far_interior_by_class"][CLASSES[int(c)]] = int(n)
    return out


def _load_segnet():
    import sys

    import torch

    torch.set_num_threads(4)  # live trainer coexistence; inference-only advisory sweep
    sys.path.insert(0, "upstream")
    import segmentation_models_pytorch as smp
    from safetensors.torch import load_file

    model = smp.Unet("tu-efficientnet_b2", classes=5, activation=None, encoder_weights=None)
    model.load_state_dict(load_file("upstream/models/segnet.safetensors", device="cpu"),
                          strict=True)
    model.eval()
    return model


def _segnet_argmax(model, frames_u8: np.ndarray) -> np.ndarray:
    """Exact upstream SegnetWrapper preprocess: raw 0-255 float -> bilinear (384,512)."""
    import torch
    import torch.nn.functional as tfun

    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float()
    x = tfun.interpolate(x, size=(H_S, W_S), mode="bilinear")
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(dim=1).numpy()


def _palette_from_gt(lstars: np.ndarray, gt_f1: np.ndarray, stride: int = 25):
    """Per-class mean + median RGB from GT frames masked by the GT argmax."""
    import cv2

    sums = np.zeros((5, 3), np.float64)
    cnts = np.zeros(5, np.int64)
    samples: list[list[np.ndarray]] = [[] for _ in range(5)]
    for i in range(0, lstars.shape[0], stride):
        small = cv2.resize(gt_f1[i], (W_S, H_S), interpolation=cv2.INTER_LINEAR)
        lab = lstars[i]
        for c in range(5):
            m = lab == c
            if m.any():
                px = small[m]
                sums[c] += px.sum(axis=0)
                cnts[c] += len(px)
                if len(samples[c]) < 40:
                    samples[c].append(px[:: max(1, len(px) // 512)])
    mean_pal = np.zeros((5, 3), np.uint8)
    med_pal = np.zeros((5, 3), np.uint8)
    for c in range(5):
        if cnts[c]:
            mean_pal[c] = np.clip(np.round(sums[c] / cnts[c]), 0, 255)
            allpx = np.concatenate(samples[c])
            med_pal[c] = np.clip(np.round(np.median(allpx, axis=0)), 0, 255)
    return mean_pal, med_pal


def _realize_frame(lab: np.ndarray, eps: float, palette: np.ndarray,
                   hood: tuple[np.ndarray, np.ndarray] | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    recon = _reconstruct(lab, eps)
    cam_lab = cv2.resize(recon, (W_C, H_C), interpolation=cv2.INTER_NEAREST)
    frame = palette[cam_lab]
    if hood is not None:
        hood0, texd = hood
        m = (cam_lab == 4) & hood0
        frame[m] = texd[m]
    return recon, frame


HOOD_DS = 16  # measured knee: ds16 beats ds8 AND full-res on d_seg at 1.8KB seed


def _hood_seed(lstars: np.ndarray, gt_f1: np.ndarray):
    """Static one-time MyCar (hood) texture seed: frame-0 GT, 16x downsampled.

    The necessity frame's 'Road-MyCar -> static one-time seed' element extended to the
    MyCar INTERIOR values (measured: 74% of the eps=0 palette-wall disagreement is
    far-interior MyCar; ds16 texture cures it at ~1.8KB brotli).
    """
    import brotli
    import cv2

    lab0_cam = cv2.resize(lstars[0].astype(np.uint8), (W_C, H_C),
                          interpolation=cv2.INTER_NEAREST)
    hood0 = lab0_cam == 4
    small = cv2.resize(gt_f1[0], (W_C // HOOD_DS, H_C // HOOD_DS),
                       interpolation=cv2.INTER_AREA)
    texd = cv2.resize(small, (W_C, H_C), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(hood0)
    crop = small[ys.min() // HOOD_DS:ys.max() // HOOD_DS + 1,
                 xs.min() // HOOD_DS:xs.max() // HOOD_DS + 1]
    seed_bytes = len(brotli.compress(crop.tobytes(), quality=11))
    return (hood0, texd), seed_bytes


def stage_palette(out_dir: str, stride: int, smoke_n: int) -> None:
    """Pick mean vs median palette on a stride smoke (choice only, not evidence)."""
    z = np.load(GT_N600)
    lstars = z["lstars"]
    gt_f1 = z["gt_f1"]
    mean_pal, med_pal = _palette_from_gt(lstars, gt_f1, stride=stride)
    model = _load_segnet()
    frames = list(range(0, lstars.shape[0], max(1, lstars.shape[0] // smoke_n)))[:smoke_n]
    res = {}
    for name, pal in (("mean", mean_pal), ("median", med_pal)):
        ds = []
        for i in frames:
            _, cam = _realize_frame(lstars[i], 0.0, pal)
            pred = _segnet_argmax(model, cam[None])[0]
            ds.append(float((pred != lstars[i]).mean()))
        res[name] = {"dseg_smoke_mean": float(np.mean(ds)), "n_frames": len(frames)}
    winner = min(res, key=lambda k: res[k]["dseg_smoke_mean"])
    out = {
        "scope": f"palette CHOICE smoke ({len(frames)} frames, stride subset); {AXIS}",
        "palette_mean_rgb": mean_pal.tolist(),
        "palette_median_rgb": med_pal.tolist(),
        "smoke": res,
        "winner": winner,
    }
    with open(os.path.join(out_dir, "palette.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


def stage_measure(out_dir: str, eps: float, limit: int, batch: int,
                  hood_tex: bool = False) -> None:
    """Chunked resumable n600 sweep for one eps; JSONL rows, one per frame."""
    pal_path = os.path.join(out_dir, "palette.json")
    with open(pal_path) as fh:
        pj = json.load(fh)
    palette = np.array(pj[f"palette_{pj['winner']}_rgb"], np.uint8)
    suffix = "_hoodtex" if hood_tex else ""
    rows_path = os.path.join(out_dir, f"rows_eps{eps}{suffix}.jsonl")
    done: set[int] = set()
    if os.path.exists(rows_path):
        with open(rows_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["frame"])
                except (json.JSONDecodeError, KeyError):
                    pass  # truncated trailing line from a killed run; frame is redone
    z = np.load(GT_N600)
    lstars = z["lstars"]
    hood = _hood_seed(lstars, z["gt_f1"])[0] if hood_tex else None
    n = lstars.shape[0]
    todo = [i for i in range(n) if i not in done][:limit]
    if not todo:
        print(f"[cal] eps={eps}: all {n} frames done")
        return
    model = _load_segnet()
    t0 = time.time()
    with open(rows_path, "a") as fh:
        for k in range(0, len(todo), batch):
            idx = todo[k:k + batch]
            recons, cams = [], []
            for i in idx:
                r, cam = _realize_frame(lstars[i], eps, palette, hood=hood)
                recons.append(r)
                cams.append(cam)
            preds = _segnet_argmax(model, np.stack(cams))
            for j, i in enumerate(idx):
                lab = lstars[i]
                dis = preds[j] != lab
                smask, bmask, pair_id, near = _strata_masks(lab)
                row = {
                    "frame": int(i),
                    "dseg_real": float(dis.mean()),
                    "dseg_geo": float((recons[j] != lab).mean()),
                    "attr": _attribute(dis, lab, smask, bmask, pair_id, near),
                }
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            el = time.time() - t0
            done_n = k + len(idx)
            print(f"[cal] eps={eps}: {done_n}/{len(todo)} ({el:.0f}s, "
                  f"{el / done_n:.2f}s/frame)", flush=True)


def _seed_bytes(lstars: np.ndarray, eps: float) -> dict:
    """SAME coder as stage_kladder on the polygons the generator actually rasterizes."""
    import brotli

    payload = bytearray()
    vtx = 0
    for i in range(lstars.shape[0]):
        lab = lstars[i].astype(np.uint8)
        for c in range(5):
            for ap in _class_polys(lab, c, eps):
                vtx += len(ap)
                first = ap[0].astype(np.uint16)
                deltas = np.diff(ap, axis=0).astype(np.int16)
                payload += first.tobytes() + np.int16(len(deltas)).tobytes() + deltas.tobytes()
    coded = brotli.compress(bytes(payload), quality=11)
    return {
        "vertices_total_n600": vtx,
        "raw_payload_bytes": len(payload),
        "brotli_q11_bytes": len(coded),
        "brotli_q11_bytes_shared_edge_adjusted": len(coded) / 2,
    }


def stage_finalize(out_dir: str) -> None:
    z = np.load(GT_N600)
    lstars = z["lstars"]
    n = lstars.shape[0]
    with open(os.path.join(out_dir, "palette.json")) as fh:
        pj = json.load(fh)
    summary: dict = {
        "scope": f"n600 (ALL scored frames); {AXIS}",
        "model": (
            "seed: DP per-class contours (kladder coder: u16 first vertex + i16 deltas + "
            "brotli -q11); generator (FREE rule 118): fillPoly even-odd + area paint-order + "
            "nearest gap fill + palette fill + INTER_NEAREST to camera res; chain: uint8 -> "
            "float -> bilinear (384,512) -> frozen SegNet argmax vs cached lstars"
        ),
        "palette_winner": pj["winner"],
        "d_pose_banked": D_POSE_BANKED,
        "pose_contribution": float(np.sqrt(10 * D_POSE_BANKED)),
        "eps": {},
    }
    pose_c = float(np.sqrt(10 * D_POSE_BANKED))
    hood_seed_bytes = _hood_seed(lstars, z["gt_f1"])[1]
    summary["hood_seed_bytes_ds16"] = hood_seed_bytes
    seed_cache: dict[float, dict] = {}
    for eps, suffix in [(e, s) for e in EPS_LIST for s in ("", "_hoodtex")]:
        rows_path = os.path.join(out_dir, f"rows_eps{eps}{suffix}.jsonl")
        if not os.path.exists(rows_path):
            continue
        rows = []
        seen: set[int] = set()
        for li in open(rows_path):
            try:
                r = json.loads(li)
            except json.JSONDecodeError:
                continue  # truncated trailing line from a killed run
            if r["frame"] not in seen:  # dedupe (a redone frame appends a second row)
                seen.add(r["frame"])
                rows.append(r)
        if len(rows) < n:
            print(f"[cal] WARNING eps={eps}: only {len(rows)}/{n} frames — skipping")
            continue
        dseg = float(np.mean([r["dseg_real"] for r in rows]))
        dgeo = float(np.mean([r["dseg_geo"] for r in rows]))
        # aggregate attribution
        agg: dict = {"saddle": 0, "near_edge_3px": 0, "edge_by_pair": {},
                     "far_interior_by_class": {}}
        for r in rows:
            a = r["attr"]
            agg["saddle"] += a["saddle"]
            agg["near_edge_3px"] += a["near_edge_3px"]
            for k, v in a["edge_by_pair"].items():
                agg["edge_by_pair"][k] = agg["edge_by_pair"].get(k, 0) + v
            for k, v in a["far_interior_by_class"].items():
                agg["far_interior_by_class"][k] = agg["far_interior_by_class"].get(k, 0) + v
        agg["edge_by_pair"] = dict(sorted(agg["edge_by_pair"].items(), key=lambda kv: -kv[1]))
        agg["far_interior_by_class"] = dict(
            sorted(agg["far_interior_by_class"].items(), key=lambda kv: -kv[1]))
        if eps not in seed_cache:
            seed_cache[eps] = _seed_bytes(lstars, eps)
        sb = dict(seed_cache[eps])
        extra = hood_seed_bytes if suffix else 0
        for key, tag in (("brotli_q11_bytes", "raw"),
                         ("brotli_q11_bytes_shared_edge_adjusted", "adjusted")):
            total = sb[key] + PALETTE_BYTES + DXI_BYTES_BANKED + extra
            sb[f"rate_term_{tag}"] = 25.0 * total / N_RATE
            sb[f"predicted_S_{tag}"] = 100.0 * dseg + pose_c + 25.0 * total / N_RATE
        summary["eps"][f"{eps}{suffix}"] = {
            "dseg_real": dseg,
            "dseg_geo": dgeo,
            "dseg_real_contribution": 100.0 * dseg,
            "attribution_px": agg,
            "seed": sb,
        }
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))
    print(f"[cal] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=("palette", "measure", "finalize"))
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--hood-tex", action="store_true",
                    help="add the static ds16 hood texture seed to the realization")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--palette-stride", type=int, default=25)
    ap.add_argument("--palette-smoke-n", type=int, default=24)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.stage == "palette":
        stage_palette(args.out_dir, args.palette_stride, args.palette_smoke_n)
    elif args.stage == "measure":
        stage_measure(args.out_dir, args.eps, args.limit, args.batch,
                      hood_tex=args.hood_tex)
    else:
        stage_finalize(args.out_dir)


if __name__ == "__main__":
    main()
