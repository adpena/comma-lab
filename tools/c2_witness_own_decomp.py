# SPDX-License-Identifier: MIT
"""c2 WITNESS-OWN per-class x per-stratum residual decomposition + law-transfer checks.

Closes the #1 caveat of the c2 taxonomy (`.omx/research/c2_perclass_stratum_carrier_
taxonomy_20260716.md` §6): that arm decomposed the PALETTE (necessity) vehicle's
residual, which DILATES the fine classes; the TRAINED WITNESS ERODES them (L65).
This arm runs the SAME decomposition on the best available trained witness — the
FROZEN mod32cap EMA-best ep650 checkpoint (d_seg 0.003146 n600 through R, reproduced
by tools/dash_comb_probe_n600.py) — so c2 targets the witness's OWN bucket weights.

Stages (all $0 local, frozen CPU-torch fp32 SegNet, bit-exact cached GT gt_n600.npz):

  decomp    Render the frozen witness pair-by-pair (op-for-op the canonical torch
            inflate primitives via tools/dash_comb_probe_n600.Renderer), through the
            EXACT contest R (bicubic^ 874x1164 -> round/clamp/uint8), cache the uint8
            camera frame (SSD tier memmap; deterministic-rebuildable from the frozen
            checkpoint), SegNet argmax -> per-frame disagreement mask (packbits) +
            per-stratum confusion + pair-side rows (JSONL; schema identical to
            tools/c2_perclass_stratum_carrier_analysis.py stage decomp). Resumable.
  temporal  persist_next + occupancy per (class x stratum) bucket from the stored
            masks (numpy only; same model as the taxonomy arm).
  sens      Margin-deficit VJP at DISAGREEING pixels of the WITNESS render through
            the REAL frozen SegNet (incl. exact bilinear resize): luma-vs-chroma
            energy, locality, side split, flat coherence — law-transfer check for
            the taxonomy's luma/non-local/flat-orthogonal cure structure.
  smoke     Carrier-variant feasibility ON THE CACHED WITNESS FRAMES (stride subset,
            LABELLED n<600): one-sided vs symmetric vs deep-side boundary-contrast
            band (0 counted B), blur — law-transfer check for asymmetric one-sided
            dominance + region-from-boundary + crisp>blurry on the witness residual.

Axes / honesty: [macOS-CPU advisory]; research_only; score_claim=false;
promotable=false. The pointer (0.19108) moves only via upstream/evaluate.py on
exact archive bytes. Chunked-foreground resumable (exit cleanly on --chunk-seconds).

Usage:
  .venv/bin/python tools/c2_witness_own_decomp.py --stage decomp --chunk-seconds 240
  .venv/bin/python tools/c2_witness_own_decomp.py --stage temporal
  .venv/bin/python tools/c2_witness_own_decomp.py --stage sens
  .venv/bin/python tools/c2_witness_own_decomp.py --stage smoke --variant oneside_movable --beta 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "tools")

from c2_perclass_stratum_carrier_analysis import (  # noqa: E402
    AXIS,
    CLASSES,
    H_C,
    H_S,
    STRATA,
    W_C,
    W_S,
    _apply_band_contrast,
    _load_gt,
    _palette,
    _stratum_field,
)
from necessity_dseg_calibration import (  # noqa: E402
    _load_segnet,
    _pair_name,
    _segnet_argmax,
)

OUT_DIR = "experiments/results/c2_witness_own_decomp_20260716"
SSD_DIR = "/Volumes/VertigoDataTier/pact/c2_witness_own_decomp_20260716"
FRAMES_MM = os.path.join(SSD_DIR, "witness_frames_cam_u8.dat")  # (n,H_C,W_C,3) u8
MASKS_MM = os.path.join(OUT_DIR, "dis_masks_packbits.u8")
ROWS = os.path.join(OUT_DIR, "decomp_rows.jsonl")
AXIS_W = AXIS + "; FROZEN mod32cap EMA-best ep650 witness rendered through exact R"
MIN_FREE_GIB = 12.0

LUMA = np.array([0.299, 0.587, 0.114])
LUMA_HAT = LUMA / np.linalg.norm(LUMA)


def _free_gib() -> float:
    # CLASS-1 fix: reclaimable-aware basis (raw psutil .available over-trusts dirty inactive anon).
    try:
        try:
            from tools.mem_basis import conservative_free_gib
        except Exception:
            from mem_basis import conservative_free_gib  # type: ignore
        return conservative_free_gib(default=float("inf"))
    except Exception:
        return float("inf")


def _witness_renderer():
    """Frozen mod32cap ep650 renderer + R (reuses the dash-probe snapshot, read-only)."""
    import dash_comb_probe_n600 as dcp

    if not dcp.CKPT_NPZ.exists():
        raise SystemExit(f"frozen ckpt snapshot missing: {dcp.CKPT_NPZ}")
    params, code, m, cfg = dcp.load_ckpt()
    from experiments.train_witness_realized_through_R_mlx import _torch_R_to_camera_uint8

    return dcp.Renderer(params, code, m), _torch_R_to_camera_uint8, cfg


def _frames_memmap(n: int, mode: str) -> np.ndarray:
    os.makedirs(SSD_DIR, exist_ok=True)
    exists = os.path.exists(FRAMES_MM)
    return np.memmap(FRAMES_MM, dtype=np.uint8,
                     mode=mode if exists or mode == "r" else "w+",
                     shape=(n, H_C, W_C, 3))


def _rendered_set() -> set[int]:
    p = os.path.join(OUT_DIR, "rendered_frames.json")
    if os.path.exists(p):
        return set(json.load(open(p)))
    return set()


def _save_rendered(done: set[int]) -> None:
    p = os.path.join(OUT_DIR, "rendered_frames.json")
    tmp = p + ".tmp"
    json.dump(sorted(done), open(tmp, "w"))
    os.replace(tmp, p)


def _get_frame(i: int, frames: np.ndarray, rendered: set[int], R, to_cam) -> np.ndarray:
    if i not in rendered:
        bulk, _ = R.render_pair(i)
        frames[i] = np.asarray(to_cam(bulk.astype(np.float64)))
        rendered.add(i)
    return np.asarray(frames[i])


# ---------------------------------------------------------------------------
# Stage DECOMP — render witness n600 through R -> SegNet argmax -> rows + masks
# ---------------------------------------------------------------------------
def stage_decomp(limit: int, batch: int, chunk_seconds: float) -> None:
    free = _free_gib()
    if free < MIN_FREE_GIB:
        raise SystemExit(f"REFUSE: free RAM {free:.1f} GiB < {MIN_FREE_GIB}")
    done: set[int] = set()
    if os.path.exists(ROWS):
        for line in open(ROWS):
            try:
                done.add(json.loads(line)["frame"])
            except (json.JSONDecodeError, KeyError):
                pass
    g = _load_gt(("lstars",))
    lstars = g["lstars"]
    n = lstars.shape[0]
    masks = np.memmap(MASKS_MM, dtype=np.uint8,
                      mode="r+" if os.path.exists(MASKS_MM) else "w+",
                      shape=(n, H_S * W_S // 8))
    frames = _frames_memmap(n, "r+")
    rendered = _rendered_set()
    todo = [i for i in range(n) if i not in done][:limit]
    if not todo:
        print(f"[c2w] decomp: all {n} frames done")
        return
    R, to_cam, cfg = _witness_renderer()
    model = _load_segnet()
    t0 = time.time()
    from scipy.ndimage import distance_transform_edt
    with open(ROWS, "a") as fh:
        for k in range(0, len(todo), batch):
            idx = todo[k:k + batch]
            cams = [_get_frame(i, frames, rendered, R, to_cam) for i in idx]
            preds = _segnet_argmax(model, np.stack(cams))
            for j, i in enumerate(idx):
                lab = lstars[i]
                pred = preds[j]
                dis = pred != lab
                masks[i] = np.packbits(dis)
                strat, pair_id = _stratum_field(lab)
                conf: dict[str, int] = {}
                ss, gg, pp = strat[dis], lab[dis], pred[dis]
                key = ss.astype(np.int64) * 25 + gg * 5 + pp
                for kk, cc in zip(*np.unique(key, return_counts=True), strict=True):
                    s_, g_, p_ = int(kk) // 25, (int(kk) % 25) // 5, int(kk) % 5
                    conf[f"{STRATA[s_]}|{CLASSES[g_]}|{CLASSES[p_]}"] = int(cc)
                side: dict[str, int] = {}
                en = dis & (strat <= 2) & (strat >= 1)
                if en.any():
                    bm = pair_id != 255
                    inds = distance_transform_edt(~bm, return_distances=False,
                                                  return_indices=True)
                    near_pair = pair_id[inds[0], inds[1]]
                    codes = near_pair[en].astype(np.int64) * 5 + lab[en]
                    for kk, cc in zip(*np.unique(codes, return_counts=True), strict=True):
                        pc, sc = int(kk) // 5, int(kk) % 5
                        if pc != 255:
                            side[f"{_pair_name(pc)}|{CLASSES[sc]}"] = int(cc)
                fh.write(json.dumps({"frame": int(i), "dseg": float(dis.mean()),
                                     "conf": conf, "pair_side": side}) + "\n")
            fh.flush()
            masks.flush()
            frames.flush()
            _save_rendered(rendered)
            el = time.time() - t0
            print(f"[c2w] decomp {k + len(idx)}/{len(todo)} ({el:.0f}s)", flush=True)
            if el > chunk_seconds:
                print(f"[chunk-exit] resumable; {len(done) + k + len(idx)}/{n} rows done",
                      flush=True)
                return


# ---------------------------------------------------------------------------
# Stage TEMPORAL — persist/occupancy per bucket (same model as the taxonomy arm)
# ---------------------------------------------------------------------------
def stage_temporal() -> None:
    g = _load_gt(("lstars",))
    lstars = g["lstars"]
    n = lstars.shape[0]
    masks = np.memmap(MASKS_MM, dtype=np.uint8, mode="r", shape=(n, H_S * W_S // 8))
    occ = np.zeros((H_S, W_S), np.float64)
    dis_all = []
    for i in range(n):
        dis = np.unpackbits(masks[i]).reshape(H_S, W_S).astype(bool)
        dis_all.append(dis)
        occ += dis
    occ /= n
    acc = np.zeros((5, 4, 3), np.float64)
    for i in range(n):
        dis = dis_all[i]
        lab = lstars[i]
        strat, _ = _stratum_field(lab)
        nxt = dis_all[i + 1] if i + 1 < n else None
        for c in range(5):
            for s in range(4):
                m = dis & (lab == c) & (strat == s)
                cnt = int(m.sum())
                if not cnt:
                    continue
                acc[c, s, 0] += cnt
                if nxt is not None:
                    acc[c, s, 1] += int((m & nxt).sum())
                acc[c, s, 2] += float(occ[m].sum())
        if i % 100 == 0:
            print(f"[c2w] temporal {i}/{n}", flush=True)
    out = {"scope": f"n600; {AXIS_W}",
           "model": ("persist_next = P(same-pixel disagrees at i+1 | disagrees at i); "
                     "occ = per-pixel disagreement frequency over n600; buckets keyed "
                     "by GT class x stratum at frame i — WITNESS residual"),
           "buckets": {}}
    for c in range(5):
        for s in range(4):
            nd = acc[c, s, 0]
            if nd < 100:
                continue
            out["buckets"][f"{CLASSES[c]}|{STRATA[s]}"] = {
                "dis_px_total": int(nd),
                "frac_of_all_dis": float(nd / acc[:, :, 0].sum()),
                "dseg_contribution": float(nd / (n * H_S * W_S)),
                "persist_next_frac": float(acc[c, s, 1] / nd),
                "mean_occupancy": float(acc[c, s, 2] / nd),
            }
    with open(os.path.join(OUT_DIR, "temporal.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# Stage SENS — margin-deficit VJP at disagreeing px of the WITNESS render
# ---------------------------------------------------------------------------
def stage_sens(frame_stride: int, per_frame: int, seed: int) -> None:
    import torch
    import torch.nn.functional as tfun

    g = _load_gt(("lstars",))
    lstars = g["lstars"]
    n = lstars.shape[0]
    model = _load_segnet()
    rng = np.random.default_rng(seed)
    masks = np.memmap(MASKS_MM, dtype=np.uint8, mode="r", shape=(n, H_S * W_S // 8))
    frames = _frames_memmap(n, "r")
    rendered = _rendered_set()
    rows = []
    for i in range(0, n, frame_stride):
        if i not in rendered:
            print(f"[c2w] sens skip frame {i}: not rendered yet", flush=True)
            continue
        lab = lstars[i]
        dis = np.unpackbits(masks[i]).reshape(H_S, W_S).astype(bool)
        if not dis.any():
            continue
        strat, pair_id = _stratum_field(lab)
        cam = np.asarray(frames[i])
        x_cam = torch.from_numpy(cam).permute(2, 0, 1).float().unsqueeze(0)
        x_cam.requires_grad_(True)
        x_s = tfun.interpolate(x_cam, size=(H_S, W_S), mode="bilinear")
        logits = model(x_s)[0]
        pred = logits.argmax(dim=0).numpy()
        picks = []
        order = [(c, s) for s in range(4) for c in range(5)]
        rng.shuffle(order)
        for c, s in order:
            m = dis & (lab == c) & (strat == s) & (pred != lab)
            ys, xs = np.where(m)
            if len(ys):
                j = rng.integers(len(ys))
                picks.append((int(ys[j]), int(xs[j]), c, s))
            if len(picks) >= per_frame:
                break
        cam_lab_up = None
        for py, px, c, s in picks:
            p_ = int(pred[py, px])
            cure = logits[c, py, px] - logits[p_, py, px]
            if x_cam.grad is not None:
                x_cam.grad = None
            cure.backward(retain_graph=True)
            gr = x_cam.grad[0].permute(1, 2, 0).detach().numpy()
            e_tot = float((gr ** 2).sum())
            if e_tot <= 0:
                continue
            gl = gr @ LUMA_HAT
            e_luma = float((gl ** 2).sum())
            cy, cx = int(py * H_C / H_S), int(px * W_C / W_S)
            loc = {}
            for r in (4, 12, 36):
                y0, y1 = max(cy - r, 0), min(cy + r + 1, H_C)
                x0, x1 = max(cx - r, 0), min(cx + r + 1, W_C)
                loc[f"e_frac_r{r}"] = float((gr[y0:y1, x0:x1] ** 2).sum() / e_tot)
            if cam_lab_up is None:
                import cv2
                cam_lab_up = cv2.resize(lab.astype(np.uint8), (W_C, H_C),
                                        interpolation=cv2.INTER_NEAREST)
            m_gt = cam_lab_up == c
            m_pr = cam_lab_up == p_
            e_gt = float((gr[m_gt] ** 2).sum() / e_tot)
            e_pr = float((gr[m_pr] ** 2).sum() / e_tot)
            gl_gt, gl_pr = gl[m_gt], gl[m_pr]
            coh_gt = float(abs(gl_gt.sum()) / (np.abs(gl_gt).sum() + 1e-12))
            coh_pr = float(abs(gl_pr.sum()) / (np.abs(gl_pr).sum() + 1e-12))
            rows.append({
                "frame": i, "y": py, "x": px, "gt": CLASSES[c], "pred": CLASSES[p_],
                "stratum": STRATA[s], "deficit_margin": float(-cure.item()),
                "luma_energy_frac": e_luma / e_tot,
                "chroma_energy_frac": 1.0 - e_luma / e_tot,
                **loc,
                "e_frac_gt_side": e_gt, "e_frac_pred_side": e_pr,
                "flat_coherence_gt_side": coh_gt, "flat_coherence_pred_side": coh_pr,
                "gt_side_shift_sign": float(np.sign(gl_gt.sum())),
                "pred_side_shift_sign": float(np.sign(gl_pr.sum())),
            })
        del logits, x_s, x_cam
        print(f"[c2w] sens frame {i}: {len(picks)} samples (total {len(rows)})", flush=True)
    out = {"scope": f"frame stride {frame_stride}, {len(rows)} samples; VJP of "
                    f"logit_gt - logit_pred at DISAGREEING px of the WITNESS render, "
                    f"through the REAL frozen SegNet incl. exact resize; {AXIS_W}",
           "rows": rows}
    with open(os.path.join(OUT_DIR, "sens.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[c2w] sens: {len(rows)} rows written")


# ---------------------------------------------------------------------------
# Stage SMOKE — carrier variants ON the cached witness frames (stride subset)
# ---------------------------------------------------------------------------
def stage_smoke(variant: str, stride: int, beta: float) -> None:
    import cv2

    g = _load_gt(("lstars",))
    lstars = g["lstars"]
    n = lstars.shape[0]
    palette = _palette()
    model = _load_segnet()
    frames = _frames_memmap(n, "r")
    rendered = _rendered_set()
    frames_idx = [i for i in range(0, n, stride) if i in rendered]
    rows = []
    t0 = time.time()
    for fi, i in enumerate(frames_idx):
        lab = lstars[i]
        cam = np.asarray(frames[i]).copy()
        if variant.startswith("blur"):
            sigma = float(variant[4:])
            cam = cv2.GaussianBlur(cam, (0, 0), sigma)
        elif variant in ("oneside_shallow", "oneside_deep", "symmetric",
                         "oneside_lane", "oneside_movable", "oneside_lane_movable",
                         "baseline"):
            side = {"oneside_shallow": {1, 3}, "oneside_deep": {0, 2, 4},
                    "symmetric": None, "oneside_lane": {1},
                    "oneside_movable": {3}, "oneside_lane_movable": {1, 3},
                    "baseline": set()}[variant]
            if variant != "baseline":
                cam = _apply_band_contrast(cam, lab, palette, beta, side)
        else:
            raise SystemExit(f"unknown variant {variant}")
        pred = _segnet_argmax(model, cam[None])[0]
        dis = pred != lab
        strat, _ = _stratum_field(lab)
        bucket = {}
        for c in range(5):
            for s in range(4):
                cnt = int((dis & (lab == c) & (strat == s)).sum())
                if cnt:
                    bucket[f"{CLASSES[c]}|{STRATA[s]}"] = cnt
        rows.append({"frame": int(i), "dseg": float(dis.mean()), "bucket": bucket})
        if fi % 20 == 0:
            print(f"[c2w] smoke {variant} {fi}/{len(frames_idx)} ({time.time() - t0:.0f}s)",
                  flush=True)
    dseg = float(np.mean([r["dseg"] for r in rows]))
    agg: dict[str, int] = {}
    for r in rows:
        for k, v in r["bucket"].items():
            agg[k] = agg.get(k, 0) + v
    out = {"scope": f"stride-{stride} subset ({len(frames_idx)} frames of n600) — "
                    f"LABELLED SUBSET, ranking only; winner owed an n600 re-measure; "
                    f"carriers applied to the CACHED WITNESS camera frames; {AXIS_W}",
           "variant": variant, "beta": beta, "dseg_subset": dseg,
           "counted_bytes": 0,
           "bucket_px": dict(sorted(agg.items(), key=lambda kv: -kv[1]))}
    with open(os.path.join(OUT_DIR, f"smoke_{variant}_b{beta}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("variant", "beta", "dseg_subset")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=("decomp", "temporal", "sens", "smoke"))
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--chunk-seconds", type=float, default=240.0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--frame-stride", type=int, default=50)
    ap.add_argument("--per-frame", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="oneside_movable")
    ap.add_argument("--beta", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage == "decomp":
        stage_decomp(args.limit, args.batch, args.chunk_seconds)
    elif args.stage == "temporal":
        stage_temporal()
    elif args.stage == "sens":
        stage_sens(args.frame_stride, args.per_frame, args.seed)
    else:
        stage_smoke(args.variant, args.stride, args.beta)


if __name__ == "__main__":
    main()
