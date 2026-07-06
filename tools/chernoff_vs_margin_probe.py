#!/usr/bin/env python3
"""$0 ADVISORY geometry probe: multi-class Chernoff information vs raw margin (top1-top2).

NO score claim. NO pointer move. NO training. Advisory geometry ONLY. CPU-torch is the
authority; the frozen upstream SegNet forward is run exactly as ``upstream/modules.py``
does (last-frame, bilinear interpolate to 512x384, raw 0-255 float input, 5-class logits).

Deep-math question (Frank Nielsen info-geometry): the PRINCIPLED bits-to-flip quantity is
the multi-class Chernoff information from the FULL 5-class softmax, which can RE-RANK
boundary pixels vs raw margin (top1-top2) ONLY where 3+ classes compete (lane / junction).
This probe measures (1) Spearman(raw_margin, chernoff) over the codim-1 separatrix, (2)
whether the top-decile re-ranking DISAGREEMENT pixels concentrate on class-1 (lane) /
junction pixels, and (3) whether chernoff predicts witness-render flips better than margin.

Chernoff model (documented, honest first-order surrogate):
  Under an equal-variance Gaussian-logit-noise model, the pairwise Chernoff exponent
  between the winning class and runner-up j is C_j = (z_top1 - z_j)^2 / 8. The aggregate
  multi-class flip-resistance is  chernoff = -log( sum_{j != top1} exp(-C_j) ). This
  aggregates ALL runner-ups, whereas raw_margin = z_top1 - z_top2 sees only the single
  nearest competitor. LIMITATION: the equal-variance Gaussian-logit-noise assumption is a
  surrogate for the true (unknown) SegNet logit perturbation law; it is the defensible
  first-order form, not a measured noise model.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

JUNCTION_PROB_THRESH = 0.15  # a class "competes" if softmax prob >= this
JUNCTION_MIN_CLASSES = 3     # junction pixel = >=3 classes competing


def rss_gib() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; linux reports kib
    return r / (1024**3) if sys.platform == "darwin" else r / (1024**2)


def load_segnet(device: str = "cpu"):
    import torch
    from safetensors.torch import load_file
    from modules import SegNet  # type: ignore

    seg = SegNet().eval()
    sd_path = ROOT / "upstream" / "models" / "segnet.safetensors"
    seg.load_state_dict(load_file(str(sd_path), device="cpu"))
    seg = seg.to(device)
    for p in seg.parameters():
        p.requires_grad_(False)
    import hashlib

    sha = hashlib.sha256(sd_path.read_bytes()).hexdigest()[:16]
    return seg, sha


def segnet_logits(seg, frames_hwc_u8: np.ndarray, device: str = "cpu") -> np.ndarray:
    """frames_hwc_u8: (B, 874, 1164, 3) uint8 -> logits (B,5,384,512) float32.

    Faithful to upstream SegNet.preprocess_input: last-frame (here each frame IS the
    scored last frame gt_f1), rearrange to (B,3,H,W).float(), bilinear interpolate to
    (segnet_model_input_size[1], segnet_model_input_size[0]) = (384, 512).
    """
    import torch
    from frame_utils import segnet_model_input_size  # type: ignore

    with torch.inference_mode():
        x = torch.from_numpy(frames_hwc_u8).to(device).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(
            x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode="bilinear"
        )
        out = seg(x)  # (B,5,384,512)
        return out.float().cpu().numpy()


def boundary_mask_from_labels(lab: np.ndarray) -> np.ndarray:
    """lab: (H,W) int -> bool (H,W) True where a 4-neighbour has a DIFFERENT label
    (codim-1 separatrix)."""
    m = np.zeros_like(lab, dtype=bool)
    m[:-1, :] |= lab[:-1, :] != lab[1:, :]
    m[1:, :] |= lab[1:, :] != lab[:-1, :]
    m[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    m[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    return m


def per_frame_scalars(logits: np.ndarray):
    """logits: (5,H,W) -> (margin, chernoff, top1_class, junction) each (H,W).

    chernoff = -log( sum_{j!=top1} exp(-(z_top1 - z_j)^2 / 8) ).
    """
    z = logits.astype(np.float64)  # (5,H,W)
    top1_class = np.argmax(z, axis=0)  # (H,W)
    zmax = np.max(z, axis=0)  # (H,W) = z_top1
    d = zmax[None, :, :] - z  # (5,H,W) >=0, winner entry == 0
    # margin = z_top1 - z_top2 = smallest POSITIVE d over non-winners.
    d_pos = np.where(d > 0, d, np.inf)
    margin = np.min(d_pos, axis=0)  # (H,W)
    margin = np.where(np.isfinite(margin), margin, 0.0)
    # chernoff aggregate: sum over all 5 of exp(-d^2/8) minus the winner's exp(0)=1.
    e = np.exp(-(d * d) / 8.0)  # (5,H,W); winner entry exp(0)=1
    sum_runnerup = np.sum(e, axis=0) - 1.0  # subtract winner contribution
    sum_runnerup = np.clip(sum_runnerup, 1e-300, None)
    chernoff = -np.log(sum_runnerup)  # (H,W), larger = more flip-resistant
    # junction: softmax prob >= thresh on >= JUNCTION_MIN_CLASSES classes.
    zc = z - zmax[None, :, :]
    p = np.exp(zc)
    p /= np.sum(p, axis=0, keepdims=True)
    junction = np.sum(p >= JUNCTION_PROB_THRESH, axis=0) >= JUNCTION_MIN_CLASSES
    return (
        margin.astype(np.float32),
        chernoff.astype(np.float32),
        top1_class.astype(np.int8),
        junction,
    )


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def auc(labels_bool: np.ndarray, scores: np.ndarray) -> float:
    """AUC that P(score_pos > score_neg). Here the outcome is FLIP; a lower resistance
    should predict a higher flip prob, so we test the NEGATED resistance as the flip score.
    Returns AUC of (-score) predicting flip (so >0.5 means low-resistance -> flip)."""
    from scipy.stats import rankdata

    s = -scores  # low resistance -> high flip score
    pos = labels_bool
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    auc_v = (r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc_v)


def run(cache: Path, chunk: int, device: str, max_frames: int | None):
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    seg, sha = load_segnet(device)
    d = np.load(cache)
    gt_f1 = d["gt_f1"]  # (N,874,1164,3) u8
    lstars_cache = d["lstars"].astype(np.int64)  # (N,384,512)
    margins_cache = d["margins"].astype(np.float32)
    N = gt_f1.shape[0]
    if max_frames is not None:
        N = min(N, max_frames)

    margins_probe, chern_probe = [], []
    cls_probe, junc_probe = [], []
    frame_idx_probe = []
    argmax_match = 0
    margin_absdiff_sum = 0.0
    margin_absdiff_cnt = 0
    t0 = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        logits = segnet_logits(seg, np.ascontiguousarray(gt_f1[s:e]), device)  # (B,5,384,512)
        for i in range(e - s):
            fi = s + i
            mg, ch, cl, jn = per_frame_scalars(logits[i])
            lab = np.argmax(logits[i], axis=0)
            # authority-faithfulness check vs cache
            argmax_match += int((lab == lstars_cache[fi]).mean() == 1.0)
            bnd = boundary_mask_from_labels(lstars_cache[fi])
            # verify margin reproduction on boundary pixels
            mc = margins_cache[fi][bnd]
            margin_absdiff_sum += float(np.abs(mg[bnd] - mc).sum())
            margin_absdiff_cnt += int(bnd.sum())
            margins_probe.append(mg[bnd])
            chern_probe.append(ch[bnd])
            cls_probe.append(cl[bnd])
            junc_probe.append(jn[bnd])
            frame_idx_probe.append(np.full(int(bnd.sum()), fi, dtype=np.int32))
        del logits
        if (s // chunk) % 4 == 0:
            print(
                f"  chunk {s}-{e}/{N}  RSS={rss_gib():.1f}GiB  elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
            if rss_gib() > 100.0:
                print("  ABORT: RSS approaching 100GiB", flush=True)
                sys.exit(3)

    margin = np.concatenate(margins_probe)
    chern = np.concatenate(chern_probe)
    cls = np.concatenate(cls_probe)
    junc = np.concatenate(junc_probe)
    fidx = np.concatenate(frame_idx_probe)
    del margins_probe, chern_probe, cls_probe, junc_probe, frame_idx_probe

    n_bnd = margin.size
    sp = spearman(margin, chern)

    # re-ranking disagreement: global ranks
    from scipy.stats import rankdata

    rm = rankdata(margin) / n_bnd
    rc = rankdata(chern) / n_bnd
    disagree = np.abs(rc - rm)
    thr = np.quantile(disagree, 0.90)
    top_dec = disagree >= thr

    base_c1 = float((cls == 1).mean())
    base_jn = float(junc.mean())
    dis_c1 = float((cls[top_dec] == 1).mean())
    dis_jn = float(junc[top_dec].mean())

    result = {
        "n_frames": int(N),
        "n_boundary_pixels": int(n_bnd),
        "spearman_margin_chernoff": sp,
        "argmax_faithful_frames": int(argmax_match),
        "margin_mean_absdiff_boundary": (
            margin_absdiff_sum / margin_absdiff_cnt if margin_absdiff_cnt else None
        ),
        "baseline_class1_frac": base_c1,
        "baseline_junction_frac": base_jn,
        "disagree_decile_class1_frac": dis_c1,
        "disagree_decile_junction_frac": dis_jn,
        "class1_concentration_ratio": (dis_c1 / base_c1 if base_c1 > 0 else None),
        "junction_concentration_ratio": (dis_jn / base_jn if base_jn > 0 else None),
        "junction_prob_thresh": JUNCTION_PROB_THRESH,
        "junction_min_classes": JUNCTION_MIN_CLASSES,
    }
    return result, (margin, chern, cls, junc, fidx, disagree, top_dec)


def survival_test(cache: Path, flips_npz: Path, chunk: int, device: str, max_frames: int | None):
    """Secondary [advisory]: does chernoff predict witness-render FLIP better than margin?"""
    import torch

    seg, _ = load_segnet(device)
    d = np.load(cache)
    gt_f1 = d["gt_f1"]
    lstars_cache = d["lstars"].astype(np.int64)
    bf = np.load(flips_npz)
    flips = bf["flips"]  # (N,384,512) bool
    bf_lstars = bf["lstars"].astype(np.int64)
    N = gt_f1.shape[0]
    if max_frames is not None:
        N = min(N, max_frames)
    N = min(N, flips.shape[0])
    # alignment guard
    align = float((bf_lstars[:N] == lstars_cache[:N]).mean())
    if align < 0.999:
        return {"status": "SKIP_misaligned", "align_frac": align}

    mg_all, ch_all, fl_all = [], [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        logits = segnet_logits(seg, np.ascontiguousarray(gt_f1[s:e]), device)
        for i in range(e - s):
            fi = s + i
            mg, ch, _cl, _jn = per_frame_scalars(logits[i])
            bnd = boundary_mask_from_labels(lstars_cache[fi])
            mg_all.append(mg[bnd])
            ch_all.append(ch[bnd])
            fl_all.append(flips[fi][bnd])
        del logits
    margin = np.concatenate(mg_all)
    chern = np.concatenate(ch_all)
    fl = np.concatenate(fl_all).astype(bool)
    auc_margin = auc(fl, margin)
    auc_chern = auc(fl, chern)
    return {
        "status": "ok",
        "n_boundary_pixels": int(margin.size),
        "n_flips_on_boundary": int(fl.sum()),
        "boundary_flip_rate": float(fl.mean()),
        "auc_margin_predicts_flip": auc_margin,
        "auc_chernoff_predicts_flip": auc_chern,
        "auc_delta_chernoff_minus_margin": (
            auc_chern - auc_margin if auc_margin == auc_margin else None
        ),
        "align_frac": align,
        "note": "witness-render flip label (n200 strided); flip=argmax disagrees GT vs render",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--flips-cache", default="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz")
    ap.add_argument("--flips-npz", default="experiments/results/wave0_residual_id_20260628/baseline_flips.npz")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / f"experiments/results/chernoff_probe_{utc}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git = "unknown"

    cache = ROOT / args.cache
    print(f"[chernoff-probe] cache={cache} chunk={args.chunk} device={args.device}", flush=True)
    result, arrays = run(cache, args.chunk, args.device, args.max_frames)
    margin, chern, cls, junc, fidx, disagree, top_dec = arrays

    # survival secondary
    surv = survival_test(
        ROOT / args.flips_cache, ROOT / args.flips_npz, args.chunk, args.device, args.max_frames
    )

    meta = {
        "schema": "chernoff_vs_margin_probe.v1",
        "utc": utc,
        "git": git,
        "advisory_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "authority": "cpu-torch frozen upstream SegNet (segnet.safetensors)",
        "seed": 0,
        "cache": str(cache),
        "primary": result,
        "survival": surv,
    }
    (out_dir / "result.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    # plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # subsample for scatter
        n = margin.size
        idx = np.random.RandomState(0).choice(n, size=min(n, 40000), replace=False)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        sc = ax[0].scatter(margin[idx], chern[idx], s=2, c=disagree[idx], cmap="viridis", alpha=0.5)
        ax[0].set_xlabel("raw margin (z_top1 - z_top2)")
        ax[0].set_ylabel("multi-class Chernoff resistance")
        ax[0].set_title(f"boundary pixels  Spearman={result['spearman_margin_chernoff']:.4f}")
        fig.colorbar(sc, ax=ax[0], label="|rank(chern)-rank(margin)|")
        cats = ["baseline\nall boundary", "top-decile\ndisagreement"]
        c1 = [result["baseline_class1_frac"], result["disagree_decile_class1_frac"]]
        jn = [result["baseline_junction_frac"], result["disagree_decile_junction_frac"]]
        x = np.arange(2)
        ax[1].bar(x - 0.2, c1, 0.4, label="class-1 (lane) frac", color="#d1495b")
        ax[1].bar(x + 0.2, jn, 0.4, label="junction frac", color="#30638e")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(cats)
        ax[1].set_ylabel("fraction")
        ax[1].set_title(
            f"concentration  c1x{result['class1_concentration_ratio']:.2f} "
            f"junctx{result['junction_concentration_ratio']:.2f}"
        )
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "chernoff_vs_margin.png", dpi=110)
        print(f"[chernoff-probe] plot -> {out_dir/'chernoff_vs_margin.png'}", flush=True)
    except Exception as exc:
        print(f"[chernoff-probe] plot skipped: {exc}", flush=True)

    print(f"[chernoff-probe] out_dir={out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    main()
