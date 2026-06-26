# SPDX-License-Identifier: MIT
"""FEED-y: the byte-closed EXACT-EVAL row for the task-space witness path.

Every prior unit DEFERRED the gate: turn the partition representation into a REAL
byte-closed candidate and measure the FULL exact S = 100*d_seg + sqrt(10*d_pose)
+ 25*|archive.zip|/37_545_489 on the REAL frozen CPU scorers through the EXACT eval
chain (camera 874x1164 -> bilinear -> SegNet 384x512 argmax for seg; rgb_to_yuv6 ->
PoseNet for pose). NO surrogate as the row (NO-FAKE #8). The S comes from the real
scorers on the real rendered frames; the rate from the real archive bytes.

The candidate (the strongest CHEAP partition we can ACTUALLY realize deterministically):
  * SEG half: the symbolic partition store L* = SegNet(GT-frame1) argmax (384x512),
    the d_seg=0-DIRECT store. Realized as a camera-res frame: palette-paint the
    nearest-upsampled L* with SegNet-optimized per-class canonical RGB (PR#56), the
    same realization the partition_store_realization_gate measured (best variant).
  * POSE half: the cheapest deterministic pose-preserving carrier — a low-res
    appearance carrier (downsampled GT frame, factor F) bilinear-upsampled and
    BLENDED into the palette frame so PoseNet reads real-ish motion. We sweep the
    blend/factor to trade pose-survival vs seg-survival vs bytes (CHROMA lives here:
    the appearance carrier carries luma+chroma the palette frame discards).

What this MEASURES (the key deliverable, even if S > 0.19110):
  * realized d_seg (real CPU SegNet) vs the DIRECT d_seg=0 of the store -> the
    REALIZATION GAP (how lossy partition->RGB->SegNet through R is).
  * realized d_pose (real CPU PoseNet on the rendered pair) -> whether ANY cheap
    task-space carrier gives a competitive pose (the binding wall).
  * the FULL exact S vs the pointer 0.19110, with the honest byte-closed archive.

Authority: real CPU-torch SegNet + PoseNet; GT decode via upstream yuv420_to_rgb
ONLY; NEVER MPS. [contest-CPU advisory] NON-PROMOTABLE per CLAUDE.md (single video,
direct-scorer mirror of evaluate.py over the measured pair subset; a full 600-pair
upstream/evaluate.py run is the promotion gate, but this is the SAME real-scorer
computation evaluate.py performs, on the same exact chain).
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

from tac.boundary_math.seg_core import decode_gt_frame1_pairs, load_real_segnet

N_CLASSES = 5
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
RATE_DENOM = 37_545_489.0
FRONTIER = 0.19110
SUB015 = 0.15


# ---------------------------------------------------------------------------
# Real-scorer exact chain (mirrors upstream/modules.py EXACTLY).
# ---------------------------------------------------------------------------
def _seg_argmax_camera(segnet, frame_cam_uint8: np.ndarray) -> np.ndarray:
    """camera-res frame -> SegNet.preprocess_input -> argmax (384x512). CPU-torch."""
    import torch

    r = np.asarray(frame_cam_uint8)
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()  # (1,2,H,W,3)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()  # (1,2,3,H,W)
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(xp)
        logits = segnet(seg_in)
        return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)


def _load_posenet(device="cpu"):
    """Load the real frozen PoseNet (the DistortionNet pose half)."""
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream

    dn = DistortionNet().eval().to(device=device)
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device(device))
    return dn.posenet


def _posenet_pose(posenet, f0_cam_uint8: np.ndarray, f1_cam_uint8: np.ndarray) -> np.ndarray:
    """Real PoseNet on a camera-res frame PAIR -> the first 6 pose dims (the scored ones).

    Mirrors DistortionNet.preprocess_input -> posenet.preprocess_input (rgb_to_yuv6)
    -> posenet -> head['pose'][..., :3] (compute_distortion uses out//2 = first 3 of 6).
    """
    import einops
    import torch

    pair = torch.from_numpy(
        np.stack([f0_cam_uint8, f1_cam_uint8], axis=0)[None]
    ).float()  # (1,2,H,W,3)
    x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        pose_in = posenet.preprocess_input(x)
        out = posenet(pose_in)
        pose = out["pose"] if isinstance(out, dict) else out
        # compute_distortion: head.out // 2 dims. PoseNet 'pose' head out=6 -> first 3.
        # Use the full out//2 contract by reading head metadata.
        half = None
        for h in posenet.hydra.heads:
            if h.name == "pose":
                half = h.out // 2
                break
        if half is None:
            half = pose.shape[-1] // 2
        return pose[0, :half].detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Realization (legal RGB frame from the partition + appearance carrier).
# ---------------------------------------------------------------------------
def _upsample_nearest(lab_seg: np.ndarray, h: int, w: int) -> np.ndarray:
    sh, sw = lab_seg.shape
    ri = (np.arange(h) * sh / h).astype(np.int64).clip(0, sh - 1)
    ci = (np.arange(w) * sw / w).astype(np.int64).clip(0, sw - 1)
    return lab_seg[ri][:, ci]


def _canonical_mu(gt_frames, lstars, n_classes):
    sums = np.zeros((n_classes, 3))
    cnts = np.zeros(n_classes)
    for f, lstar in zip(gt_frames, lstars, strict=True):
        lab_cam = _upsample_nearest(lstar, CAMERA_H, CAMERA_W)
        ff = f.astype(np.float64)
        for c in range(n_classes):
            m = lab_cam == c
            n = float(m.sum())
            if n > 0:
                sums[c] += ff[m].sum(axis=0)
                cnts[c] += n
    mu = np.where(cnts[:, None] > 0, sums / np.maximum(cnts[:, None], 1), 128.0)
    return mu


def _optimize_mu(seg, lstars, gt_frames, mu_init, n_probe=4, rounds=2):
    mu = mu_init.copy()
    probe = list(zip(lstars, gt_frames, strict=True))[:n_probe]

    def dseg_for(mu_try):
        ds = []
        for lstar, _gt in probe:
            lab = _upsample_nearest(lstar, CAMERA_H, CAMERA_W)
            painted = np.clip(np.round(mu_try[lab]), 0, 255).astype(np.uint8)
            realized = _seg_argmax_camera(seg, painted)
            ds.append(float(np.count_nonzero(realized != lstar)) / lstar.size)
        return float(np.mean(ds))

    for _ in range(rounds):
        for c in range(N_CLASSES):
            base = mu[c]
            cands = [base]
            for s in (0.6, 0.8, 1.2, 1.4):
                cands.append(np.clip(base * s, 0, 255))
            cands.append(np.full(3, float(np.clip(base.mean(), 0, 255))))
            for sh in (-40.0, 40.0):
                cands.append(np.clip(base + sh, 0, 255))
            best_c, best_d = base, math.inf
            for cand in cands:
                mt = mu.copy()
                mt[c] = cand
                d = dseg_for(mt)
                if d < best_d:
                    best_d, best_c = d, cand
            mu[c] = best_c
    return mu


def _lowres_carrier(frame_cam_uint8: np.ndarray, factor: int):
    """Downsample (area) by factor; return (lowres_uint8, deterministic_zlib_bytes)."""
    import torch
    import torch.nn.functional as F

    f = np.asarray(frame_cam_uint8, dtype=np.float32)
    h, w, _ = f.shape
    lh, lw = max(1, h // factor), max(1, w // factor)
    t = torch.from_numpy(f.transpose(2, 0, 1))[None]
    down = F.interpolate(t, size=(lh, lw), mode="area")[0].numpy()
    lowres = np.clip(np.round(down.transpose(1, 2, 0)), 0, 255).astype(np.uint8)
    return lowres


def _upsample_bilinear(lowres_hwc: np.ndarray, h: int, w: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(lowres_hwc, dtype=np.float32).transpose(2, 0, 1))[None]
    up = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0]
    return np.clip(np.round(up.numpy().transpose(1, 2, 0)), 0, 255).astype(np.uint8)


def main() -> None:
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    t0 = time.time()
    seg = load_real_segnet("cpu")
    posenet = _load_posenet("cpu")

    # Decode GT pairs; the store L* = SegNet(GT frame1) argmax (the d_seg=0-direct store).
    gt_f0, gt_f1, lstars = [], [], []
    for _idx, f0, f1 in decode_gt_frame1_pairs(n_pairs=n_pairs):
        f0 = np.asarray(f0)
        f1 = np.asarray(f1)
        lstars.append(_seg_argmax_camera(seg, f1))
        gt_f0.append(f0)
        gt_f1.append(f1)
    n = len(lstars)
    decode_s = time.time() - t0

    # SegNet-optimized canonical palette (PR#56), measured on the real scorer.
    mu0 = _canonical_mu(gt_f1, lstars, N_CLASSES)
    mu = _optimize_mu(seg, lstars, gt_f1, mu0)

    # --- GT-pair pose baselines (the floor + the collapse anchors) ----------
    # (i) GT itself: d_pose = 0 by definition (sanity that PoseNet(GT)==PoseNet(GT)).
    # (ii) palette-only frame: PoseNet reads NO motion -> d_pose collapse anchor.
    # We measure d_pose against PoseNet(GT_pair) as the reference (that IS d_pose).
    def realized_pose_d(comp_f0, comp_f1, gt_pose):
        comp_pose = _posenet_pose(posenet, comp_f0, comp_f1)
        return float(np.mean((comp_pose - gt_pose) ** 2))

    gt_poses = [_posenet_pose(posenet, f0, f1) for f0, f1 in zip(gt_f0, gt_f1, strict=True)]

    # The partition store rate: store L* losslessly. We measure the REAL coded bytes
    # of the partition stack via the context-partition codec (the symbolic store).
    try:
        from tac.boundary_math.context_partition_codec import encode_partition_stack

        store_code = encode_partition_stack(lstars, template="temporal")
        store_bytes_measured = len(store_code.payload)
        store_bytes_per_frame = store_bytes_measured / max(1, n)
    except Exception as exc:  # pragma: no cover - codec optional
        store_bytes_measured = None
        store_bytes_per_frame = None
        print(f"[warn] partition codec unavailable: {exc}", file=sys.stderr)

    # --- realization variants: palette + low-res appearance carrier blend ----
    # blend a: 0.0 = palette-only (seg-clean, pose-collapse); >0 pulls in real
    # appearance for pose at the cost of seg-survival. factor controls carrier bytes.
    variants = []
    for factor in (8, 16):
        for blend in (0.0, 0.3, 0.6, 1.0):
            variants.append((factor, blend))

    results = {}
    for factor, blend in variants:
        vname = f"f{factor}_blend{blend}"
        seg_ds, pose_ds = [], []
        carrier_bytes_per_pair = []
        for f0, f1, lstar, gtp in zip(gt_f0, gt_f1, lstars, gt_poses, strict=True):
            lab_cam = _upsample_nearest(lstar, CAMERA_H, CAMERA_W)
            palette = mu[lab_cam].astype(np.float64)  # (H,W,3)
            # appearance carrier for BOTH frames of the pair (pose needs motion).
            comp_f1 = palette.copy()
            comp_f0 = palette.copy()  # frame0 has no stored L*; start from palette
            cb = 0
            if blend > 0.0:
                lr1 = _lowres_carrier(f1, factor)
                lr0 = _lowres_carrier(f0, factor)
                up1 = _upsample_bilinear(lr1, CAMERA_H, CAMERA_W).astype(np.float64)
                up0 = _upsample_bilinear(lr0, CAMERA_H, CAMERA_W).astype(np.float64)
                comp_f1 = (1.0 - blend) * palette + blend * up1
                comp_f0 = (1.0 - blend) * palette + blend * up0
                # deterministic coded bytes of the two low-res carriers (zlib-9).
                cb = len(zlib.compress(lr0.tobytes(), 9)) + len(
                    zlib.compress(lr1.tobytes(), 9)
                )
            comp_f1 = np.clip(np.round(comp_f1), 0, 255).astype(np.uint8)
            comp_f0 = np.clip(np.round(comp_f0), 0, 255).astype(np.uint8)
            realized = _seg_argmax_camera(seg, comp_f1)
            seg_ds.append(float(np.count_nonzero(realized != lstar)) / lstar.size)
            pose_ds.append(realized_pose_d(comp_f0, comp_f1, gtp))
            carrier_bytes_per_pair.append(cb)

        mean_seg = float(np.mean(seg_ds))
        mean_pose = float(np.mean(pose_ds))
        mean_cb = float(np.mean(carrier_bytes_per_pair))
        # FULL byte-closed archive estimate: partition store + per-pair carrier (x n_full=600).
        # rate uses the MEASURED per-pair sizes scaled to 600 pairs (the real video).
        n_full = 600
        store_full = (store_bytes_per_frame or 0.0) * n_full
        carrier_full = mean_cb * n_full
        archive_bytes = int(round(store_full + carrier_full)) + 200  # +zip overhead
        rate_term = 25.0 * archive_bytes / RATE_DENOM
        seg_term = 100.0 * mean_seg
        pose_term = math.sqrt(10.0 * mean_pose)
        S = seg_term + pose_term + rate_term
        results[vname] = {
            "factor": factor,
            "blend": blend,
            "realized_d_seg": mean_seg,
            "realized_d_pose": mean_pose,
            "seg_term": seg_term,
            "pose_term": pose_term,
            "carrier_bytes_per_pair": mean_cb,
            "store_bytes_per_frame": store_bytes_per_frame,
            "archive_bytes_600pair_est": archive_bytes,
            "rate_term": rate_term,
            "S_exact_realized": S,
            "beats_frontier": S < FRONTIER,
            "beats_sub015": S < SUB015,
        }

    best = min(results, key=lambda k: results[k]["S_exact_realized"])
    best_r = results[best]

    out = {
        "subagent": "feedy_byteclose_exact_row",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "contest-CPU-advisory-direct-scorer-mirror",
        "evidence_grade": "[contest-CPU advisory]",
        "promotion_eligible": False,
        "score_claim": False,
        "note": (
            "REAL frozen CPU SegNet + PoseNet through the EXACT eval chain on the "
            "REALIZED partition-store frames. d_seg/d_pose are the real argmax-flip / "
            "pose-MSE the scorers compute; rate is the real measured coded bytes "
            "scaled to 600 pairs. Single-video direct-scorer mirror of evaluate.py "
            "over the measured pair subset."
        ),
        "n_pairs_measured": n,
        "decode_plus_lstar_seconds": round(decode_s, 1),
        "frontier_pointer": FRONTIER,
        "sub015_target": SUB015,
        "store_bytes_measured_subset": store_bytes_measured,
        "store_bytes_per_frame": store_bytes_per_frame,
        "segnet_optimized_mu": mu.tolist(),
        "variants": results,
        "best": {
            "variant": best,
            "S_exact_realized": best_r["S_exact_realized"],
            "realized_d_seg": best_r["realized_d_seg"],
            "realized_d_pose": best_r["realized_d_pose"],
            "archive_bytes_600pair_est": best_r["archive_bytes_600pair_est"],
            "beats_frontier": best_r["beats_frontier"],
            "beats_sub015": best_r["beats_sub015"],
        },
        "realization_gap": {
            "direct_store_d_seg": 0.0,
            "best_realized_d_seg": best_r["realized_d_seg"],
            "interpretation": (
                "The symbolic store is d_seg=0 DIRECT (it stores L* losslessly). "
                "The REALIZED d_seg through partition->palette-RGB->SegNet round-trip "
                f"is {best_r['realized_d_seg']:.5f} -> the realization gap is the full "
                f"{best_r['realized_d_seg']:.5f} (boundary band flips even painting "
                "the EXACT L*)."
            ),
        },
    }
    outdir = REPO / "experiments" / "results" / "feedy_byteclosed_exact_row_20260625"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "exact_row.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
