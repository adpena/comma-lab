#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""WITNESS RESIDUAL LOCALIZATION probe — WHERE do the witness's d_seg flips concentrate?

The cheap d_seg levers saturate ~0.0024 (need ~0.0011 pointer-tie / ~0.0007 sub-0.15). To find a
STRUCTURALLY NEW lever we must first MEASURE the residual: of the pixels the trained witness still
flips (pred argmax != frozen-SegNet GT argmax), what distinguishes them?

  Q1  MARGIN DEPTH — are the residual flips at SHALLOW margin (the binding band, training-fixable
      via more oomph / KD on the soft logits) or at DEEP margin (a represent-ability failure: the
      witness's basis cannot place the boundary correctly)?
  Q2  DISTANCE-TO-BOUNDARY — do flips hug the GT inter-class edges (codim-1 boundary band) or do
      they appear in the interior (a global representation error, NOT an edge-placement error)?
  Q3  CLASS-PAIR — which (gt_class -> pred_class) confusions dominate? (lane c1 vs road c2 vs the
      c0/c4 sky/structure split). Tells us which class the basis is starving.
  Q4  TEACHER-AGREEMENT — at the residual flips, is the frozen-SegNet teacher itself UNCERTAIN
      (soft logits near-tie, the flip is "almost right") or CONFIDENT (the witness is plainly
      wrong)? Distinguishes a KD-fixable margin-shallowness from a hard capacity miss.
  Q5  PER-PAIR — is the residual spread across all pairs or concentrated in a few hard frames?

METHOD (NO-FAKE): train the EXACT in-tree witness (ImprovedSegGenerator, all-class directional
basis + capacity, the current best config) for a bounded budget on n_pairs, then for every pixel
of every eval pair record (flip?, gt_margin, dist_to_GT_boundary, gt_class, pred_class,
teacher_top2_gap). Aggregate. The d_seg authority is the frozen CPU-torch SegNet argmax already
cached as gt_segnet_argmax.u8 (NEVER MPS as authority). The witness FORWARD is MLX (gradient
device only); its argmax is compared against the cached frozen authority. Evidence
[macOS-MLX research-signal], promotion_eligible=false, NO score claim.

This is a DIAGNOSTIC, not a score lever — its output RANKS the next lever (a represent-ability
miss argues a basis/activation lever; a margin-shallowness miss argues a KD/oomph lever; a
single-class starvation argues a class-conditional capacity lever).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "upstream", REPO_ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from witness_capstone_deepmath_smoke import (  # noqa: E402
    ImprovedSegGenerator,
    _build_coords_np,
    deterministic_fourier_B,
    load_targets,
    load_teacher_logits,
    precompute_features,
)
import mlx.nn as nn  # noqa: E402

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dist_to_boundary(argmax_hw: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) from each pixel to the nearest GT inter-class boundary."""
    from scipy import ndimage

    a = argmax_hw
    b = np.zeros_like(a, dtype=bool)
    b[:-1, :] |= a[:-1, :] != a[1:, :]
    b[1:, :] |= a[:-1, :] != a[1:, :]
    b[:, :-1] |= a[:, :-1] != a[:, 1:]
    b[:, 1:] |= a[:, :-1] != a[:, 1:]
    if not b.any():
        return np.full(a.shape, float(np.hypot(*a.shape)), np.float32)
    return ndimage.distance_transform_edt(~b).astype(np.float32)


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    if any(str(out_dir).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError("out_dir is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    argmax, margin, H, W = load_targets(Path(args.targets_dir), args.num_pairs)
    P = argmax.shape[0]
    n_px = H * W
    coords = _build_coords_np(H, W)
    fourier_B = deterministic_fourier_B(args.n_fourier, args.fourier_sigma)

    feats = precompute_features(
        coords, argmax, H, W, fourier_B,
        use_prox=False, use_dir=True, n_dir_freqs=args.n_dir_freqs,
        freq_across=args.freq_across, freq_along=args.freq_along,
        prox_tau=args.prox_tau, lane_class=args.lane_class, all_class=True,
    )
    feats_mx = mx.array(feats)
    labels_flat = argmax.reshape(P, n_px).astype(np.int32)
    margin_flat = margin.reshape(P, n_px).astype(np.float32)

    teacher_mm = None
    kd_active = float(args.kd_weight) > 0.0
    if kd_active:
        teacher_mm = load_teacher_logits(Path(args.teacher_logits_dir), args.num_pairs, H, W)

    model = ImprovedSegGenerator(
        num_pairs=P, n_fourier=args.n_fourier, hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden, mod_dim=args.mod_dim, fourier_sigma=args.fourier_sigma,
        use_prox=False, use_dir=True, n_dir_freqs=args.n_dir_freqs, activation=args.activation,
    )
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    tau = 1.0
    kd_w, kd_t, kd_ce = float(args.kd_weight), float(args.kd_temp), float(args.kd_ce_blend)

    from witness_capstone_deepmath_smoke import kd_kl_logits

    def loss_fn(model, pair_idx, fbatch, labels, wpx, teacher):
        logits = model(fbatch, pair_idx)
        ce = nn.losses.cross_entropy(logits, labels, reduction="none")
        ce_term = (ce * wpx).mean()
        if teacher is None:
            return ce_term
        kd = kd_kl_logits(logits, teacher, kd_t)
        return kd_ce * ce_term + kd_w * (kd * wpx).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(P)
        for pi_np in order:
            pi = int(pi_np)
            idx = rng.integers(0, n_px, size=args.px_per_step)
            fb = feats_mx[pi][mx.array(idx)]
            lab = mx.array(labels_flat[pi][idx])
            mg = margin_flat[pi][idx]
            wnp = 1.0 + args.hinge_weight * np.exp(-np.maximum(mg, 0.0) / tau)
            wpx = mx.array(wnp.astype(np.float32))
            teacher_b = None
            if teacher_mm is not None:
                tlog = np.asarray(teacher_mm[pi]).reshape(5, n_px).astype(np.float32)
                teacher_b = mx.array(tlog[:, idx].T)
            loss, grads = loss_and_grad(model, pi, fb, lab, wpx, teacher_b)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
        if ep % args.eval_every == 0 or ep == args.epochs:
            d = _quick_d_seg(model, feats_mx, labels_flat, P)
            print(json.dumps({"epoch": ep, "d_seg": round(d, 6), "wall_s": round(time.time() - t0, 1)}), flush=True)

    # ---- localization pass: accumulate per-pixel residual descriptors ----
    margin_bins = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]
    dist_bins = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 1e9]
    n_mb, n_db = len(margin_bins) - 1, len(dist_bins) - 1
    flip_by_margin = np.zeros(n_mb)
    tot_by_margin = np.zeros(n_mb)
    flip_by_dist = np.zeros(n_db)
    tot_by_dist = np.zeros(n_db)
    confusion = np.zeros((5, 5), dtype=np.int64)  # gt_class -> pred_class (flips only)
    teacher_gap_flip = []  # teacher top2 logit gap at flip pixels
    teacher_gap_correct = []
    per_pair_flip = np.zeros(P)
    total_flips = 0
    total_px = 0

    for pi in range(P):
        logits = model(feats_mx[pi], pi)
        pred = np.array(mx.argmax(logits, axis=-1)).astype(np.int32)
        gt = labels_flat[pi]
        flip = pred != gt
        per_pair_flip[pi] = flip.mean()
        total_flips += int(flip.sum())
        total_px += flip.size
        mg = margin_flat[pi]
        dist = _dist_to_boundary(argmax[pi]).reshape(-1)
        mb = np.clip(np.digitize(mg, margin_bins) - 1, 0, n_mb - 1)
        db = np.clip(np.digitize(dist, dist_bins) - 1, 0, n_db - 1)
        for k in range(n_mb):
            sel = mb == k
            tot_by_margin[k] += int(sel.sum())
            flip_by_margin[k] += int(flip[sel].sum())
        for k in range(n_db):
            sel = db == k
            tot_by_dist[k] += int(sel.sum())
            flip_by_dist[k] += int(flip[sel].sum())
        gtf = gt[flip]
        prf = pred[flip]
        np.add.at(confusion, (gtf, prf), 1)
        if teacher_mm is not None:
            tlog = np.asarray(teacher_mm[pi]).reshape(5, n_px).astype(np.float32).T  # (n_px,5)
            srt = np.sort(tlog, axis=-1)
            gap = (srt[:, -1] - srt[:, -2])  # top1-top2 logit gap (teacher confidence)
            if flip.any():
                teacher_gap_flip.append(gap[flip])
            corr = ~flip
            # subsample correct (huge) to keep memory bounded
            ci = np.where(corr)[0]
            if ci.size:
                pick = ci[rng.integers(0, ci.size, size=min(ci.size, 4000))]
                teacher_gap_correct.append(gap[pick])

    d_seg = total_flips / total_px
    rate_by_margin = (flip_by_margin / np.maximum(tot_by_margin, 1)).round(6)
    share_by_margin = (flip_by_margin / max(total_flips, 1)).round(4)
    rate_by_dist = (flip_by_dist / np.maximum(tot_by_dist, 1)).round(6)
    share_by_dist = (flip_by_dist / max(total_flips, 1)).round(4)

    # confusion: row-normalize within flips originating from each gt class
    conf_share = (confusion / max(confusion.sum(), 1)).round(4)
    gt_flip_share = (confusion.sum(axis=1) / max(confusion.sum(), 1)).round(4)
    pred_flip_share = (confusion.sum(axis=0) / max(confusion.sum(), 1)).round(4)

    tg_flip = np.concatenate(teacher_gap_flip) if teacher_gap_flip else np.array([])
    tg_corr = np.concatenate(teacher_gap_correct) if teacher_gap_correct else np.array([])
    teacher_diag = {}
    if tg_flip.size:
        teacher_diag = {
            "teacher_top2_gap_at_flips_median": round(float(np.median(tg_flip)), 4),
            "teacher_top2_gap_at_flips_mean": round(float(tg_flip.mean()), 4),
            "teacher_top2_gap_at_correct_median": round(float(np.median(tg_corr)), 4) if tg_corr.size else None,
            "frac_flips_where_teacher_uncertain_gap_lt1": round(float((tg_flip < 1.0).mean()), 4),
            "frac_flips_where_teacher_confident_gap_gt3": round(float((tg_flip > 3.0).mean()), 4),
        }

    pp = per_pair_flip
    result = {
        "subagent": "probe_witness_residual_localization_20260625",
        "utc": _utc(),
        "evidence_grade": "[macOS-MLX research-signal]",
        "promotion_eligible": False, "score_claim": False,
        "config": {
            "num_pairs": P, "epochs": args.epochs, "hidden_dim": args.hidden_dim,
            "n_hidden": args.n_hidden, "mod_dim": args.mod_dim, "n_fourier": args.n_fourier,
            "kd_weight": kd_w, "kd_temp": kd_t, "kd_ce_blend": kd_ce, "kd_active": kd_active,
            "activation": args.activation, "lr": args.lr, "seed": args.seed,
        },
        "d_seg": round(d_seg, 6),
        "margin_bins": margin_bins,
        "Q1_flip_rate_by_margin_bin": rate_by_margin.tolist(),
        "Q1_flip_share_by_margin_bin": share_by_margin.tolist(),
        "dist_bins": dist_bins,
        "Q2_flip_rate_by_dist_to_boundary": rate_by_dist.tolist(),
        "Q2_flip_share_by_dist_to_boundary": share_by_dist.tolist(),
        "Q3_confusion_gt_to_pred_share": conf_share.tolist(),
        "Q3_gt_class_flip_share": gt_flip_share.tolist(),
        "Q3_pred_class_flip_share": pred_flip_share.tolist(),
        "Q4_teacher": teacher_diag,
        "Q5_per_pair_flip_frac_mean": round(float(pp.mean()), 6),
        "Q5_per_pair_flip_frac_p50": round(float(np.percentile(pp, 50)), 6),
        "Q5_per_pair_flip_frac_p90": round(float(np.percentile(pp, 90)), 6),
        "Q5_per_pair_flip_frac_max": round(float(pp.max()), 6),
        "Q5_top10pct_pairs_share_of_flips": round(
            float(np.sort(pp)[int(0.9 * P):].sum() / max(pp.sum(), 1)), 4),
    }
    (out_dir / "residual_localization.json").write_text(json.dumps(result, indent=2))
    return result


def _quick_d_seg(model, feats_mx, labels_flat, P) -> float:
    dis = tot = 0
    for pi in range(P):
        pred = np.array(mx.argmax(model(feats_mx[pi], pi), axis=-1)).astype(np.int32)
        dis += int((pred != labels_flat[pi]).sum())
        tot += labels_flat[pi].size
    return dis / tot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Witness residual localization probe")
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--teacher-logits-dir", type=Path, default=Path(base) / "teacher_logits_n600")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--n-fourier", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-hidden", type=int, default=5)
    ap.add_argument("--mod-dim", type=int, default=48)
    ap.add_argument("--fourier-sigma", type=float, default=8.0)
    ap.add_argument("--n-dir-freqs", type=int, default=6)
    ap.add_argument("--freq-across", type=float, default=32.0)
    ap.add_argument("--freq-along", type=float, default=4.0)
    ap.add_argument("--prox-tau", type=float, default=4.0)
    ap.add_argument("--lane-class", type=int, default=1)
    ap.add_argument("--activation", choices=["relu", "gelu", "gauss"], default="relu")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    ap.add_argument("--kd-weight", type=float, default=0.3)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--kd-ce-blend", type=float, default=1.0)
    ap.add_argument("--px-per-step", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    result = run(args)
    print("\n=== RESIDUAL LOCALIZATION ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
