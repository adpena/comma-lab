# SPDX-License-Identifier: MIT
"""CONTENT EDGE, STRONG PREDICTOR: does a non-overflowing model extract materially more
flip-predictability from CONTENT than FEED-kd's weak logistic (MI=0.000166 bits)?

THE QUESTION (coordinator scope-widen of FEED-kd). FEED-kd
(``tools/measure_jitter_predictability.py``) measured I(flip; content | free+margin) =
0.000166 bits with a self-contained numpy LOGISTIC (IRLS, threw overflow/divide warnings)
-> "content adds ~nothing -> a trained content generator is demoted." But a logistic is a
LINEAR model and a FLOOR on predictability, not a ceiling. The contest gives UNLIMITED
compress-time compute, so the encoder may train a strong model. This tool re-measures the
content edge with a torch MLP (CPU, non-overflowing, trained to convergence with early stop)
on the SAME contiguous held-out split, and reports whether the MLP beats the logistic's MI.

It reuses the jitter probe's machinery (vote prediction, flip set, feature buckets) and the
SAVED per-frame SegNet argmax cache (``--seg-cache``, int8) so it runs CPU-only with NO SegNet
forwards (no contention with a concurrent through-R run). If no cache is given it computes one.

Feature buckets (decode-realizability flagged, same as FEED-kd):
  FREE     row/col position, is_hood_prior, vote_conf_{ground,rotonly,identity}, sdf_prior
  MARGIN   SegNet argmax margin (decode-available via the warped-stored-canonical margin per
           the compliant (b) mechanism; an UPPER bound here uses the GT per-pair margin)
  CONTENT  texture_grad, rgb_{r,g,b}, sdf_gt (GT-boundary SDF)  <- the "trained generator" edge

THE CRUX reported: I(flip; content | free+margin) for LOGISTIC vs MLP (held-out CE difference),
overall AND within the low-margin (ambiguous) band. AUC too. A materially-larger MLP content MI
re-opens the content-generator door; ~equal confirms FEED-kd's demotion (with a stronger model).

AUTHORITY / HONESTY FIREWALL (CLAUDE.md): ``[macOS advisory / CPU-torch research-signal]``
ONLY. NOT a contest score. Pointer 0.19110 UNMOVED. score_claim/promotable=False. MEANS. flip =
REAL argmax-disagreement vs frozen CPU-torch SegNet argmax (lstars); cross-validated on a
CONTIGUOUS held-out block (no temporal-adjacency leakage). torch MLP is a PREDICTABILITY probe,
NOT a shipped artifact; it bounds how compressible the jitter is from content, not a score.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.measure_pose_warp_dseg import (  # noqa: E402
    CLASS_NAMES, NATIVE_H, NATIVE_W, intrinsics_at, _target_grid,
)
from tools.measure_screw_warp_through_R import fit_calibration_within_pair  # noqa: E402
from tools.measure_clean_canonical_warp_through_R import rgb_at  # noqa: E402
from tools.measure_budget_gate_overturn import build_exact_step_poses  # noqa: E402
from tools.measure_jitter_predictability import (  # noqa: E402
    vote_predict, bulk_flip_map, FNAMES, BUCKET, CONT, design_matrix,
    fit_logistic_irls, _sigmoid, cross_entropy_bits, binary_entropy_bits, auc_score, _boundary,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
BULK_IDX = [CLASS_NAMES.index(c) for c in ("Road", "Undriv", "MyCar")]


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def train_mlp(Xtr, ytr, Xte, yte, hidden=(64, 64, 32), epochs=300, lr=2e-3, wd=1e-5,
              batch=65536, patience=25, pos_weight_cap=50.0, seed=0):
    """Small CPU torch MLP, BCEWithLogits, Adam, early-stop on held-out CE. Returns held-out
    probabilities. Unlimited-compress-time budget => train to convergence with early stopping."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)); ytr_t = torch.from_numpy(ytr.astype(np.float32))
    Xte_t = torch.from_numpy(Xte.astype(np.float32)); yte_t = torch.from_numpy(yte.astype(np.float32))
    layers = []; d = Xtr.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        d = h
    layers += [torch.nn.Linear(d, 1)]
    net = torch.nn.Sequential(*layers).to(dev)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pw = torch.tensor([min(neg / max(pos, 1.0), pos_weight_cap)], dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n = Xtr.shape[0]
    best_ce = float("inf"); best_p = None; bad = 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = net(Xtr_t[idx]).squeeze(1)
            loss = lossf(out, ytr_t[idx])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pte = torch.sigmoid(net(Xte_t).squeeze(1)).numpy()
        eps = 1e-12; pc = np.clip(pte, eps, 1 - eps)
        ce = float((-(yte * np.log(pc) + (1 - yte) * np.log(1 - pc))).mean() / math.log(2.0))
        if ce < best_ce - 1e-6:
            best_ce = ce; best_p = pte; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_p, best_ce


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--comma-pose", default="experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz")
    ap.add_argument("--seg-cache", default="experiments/results/waterfill_through_R_n96_r2/seg_cache_n96.npz",
                    help="saved per-frame SegNet argmax (int8); if missing, computed with SegNet.")
    ap.add_argument("--n-pairs", type=int, default=0)
    ap.add_argument("--window-radius", type=int, default=2)
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--train-sample", type=int, default=2_000_000)
    ap.add_argument("--eval-sample", type=int, default=4_000_000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"]); gt_f1 = np.asarray(z["gt_f1"])
    lstars = np.asarray(z["lstars"], dtype=np.int64)
    margins = np.asarray(z["margins"], dtype=np.float32)
    posenet_poses = np.asarray(z["gt_poses"], dtype=np.float64)
    P_cache = lstars.shape[0]
    P = P_cache if not args.n_pairs else min(args.n_pairs, P_cache)
    gt_f0, gt_f1, lstars, margins, posenet_poses = (gt_f0[:P], gt_f1[:P], lstars[:P],
                                                    margins[:P], posenet_poses[:P])
    SEG_H, SEG_W = lstars.shape[1], lstars.shape[2]
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]
    n_frames = 2 * P

    cp = np.load((REPO / args.comma_pose) if not Path(args.comma_pose).is_absolute() else Path(args.comma_pose),
                 allow_pickle=True)
    pos = np.asarray(cp["frame_positions"], dtype=np.float64)
    ori = np.asarray(cp["frame_orientations"], dtype=np.float64)

    # ---- seg_cache: load saved (no SegNet) or compute ----
    scp = (REPO / args.seg_cache) if (args.seg_cache and not Path(args.seg_cache).is_absolute()) else (
        Path(args.seg_cache) if args.seg_cache else None)
    seg_source = "computed"
    selfcheck = {"PASS": None}
    if scp is not None and scp.exists():
        sc = np.load(scp)["seg_cache"].astype(np.int64)
        if sc.shape[0] >= n_frames:
            seg_cache = sc[:n_frames]
            seg_source = f"loaded {scp.name}"
            selfcheck = {"PASS": bool(all(np.array_equal(seg_cache[2 * p + 1], lstars[p]) for p in range(min(4, P)))),
                         "note": "seg_cache[2p+1]==lstars selfcheck on first 4 pairs"}
        else:
            scp = None
    if seg_source == "computed":
        from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax
        from tac.boundary_math.seg_core import load_real_segnet
        seg = load_real_segnet("cpu")
        def seg_argmax(fr):
            am, _ = measure_segnet_argmax(seg, np.asarray(fr, dtype=np.float64)); return am
        for p in range(4):
            if int(np.count_nonzero(seg_argmax(gt_f1[p]) != lstars[p])) != 0:
                raise SystemExit("NO-FAKE self-check FAILED: SegNet(gt_f1)!=lstars.")
        seg_cache = np.zeros((n_frames, SEG_H, SEG_W), dtype=np.int64)
        for g in range(n_frames):
            seg_cache[g] = seg_argmax(rgb_at(gt_f0, gt_f1, g))
        selfcheck = {"PASS": True, "note": "computed + selfchecked"}
    if selfcheck.get("PASS") is False:
        raise SystemExit("NO-FAKE self-check FAILED: loaded seg_cache[2p+1] != lstars.")

    exact_step_poses = build_exact_step_poses(pos, ori, n_frames)
    K_seg = intrinsics_at(SEG_W, SEG_H); Kinv_seg = np.linalg.inv(K_seg); grid_seg = _target_grid(SEG_H, SEG_W)
    lstar0 = seg_cache[0::2]
    fit = fit_calibration_within_pair(lstar0, lstars, np.stack([exact_step_poses[2 * p] for p in range(P)], 0),
                                      K_seg, Kinv_seg, grid_seg)
    fit_params = (fit["s_t"], fit["s_r"], fit["pitch"])

    # ---- per-pair features + flip set (replicates the jitter probe feature loop) ----
    print(f"[content-edge] building features over {P} pairs ({seg_source})...", flush=True)
    feat = {c: [] for c in FNAMES}; flip_list = []; pair_list = []
    rows_norm = (np.arange(SEG_H)[:, None] / (SEG_H - 1)).repeat(SEG_W, 1)
    cols_norm = (np.arange(SEG_W)[None, :] / (SEG_W - 1)).repeat(SEG_H, 0)
    zy, zx = SEG_H / NAT_H, SEG_W / NAT_W
    for p in range(P):
        t = 2 * p + 1
        tgt = lstars[p]
        vote_am, vote_conf = vote_predict(seg_cache, t, args.window_radius, exact_step_poses,
                                          K_seg, Kinv_seg, grid_seg, fit_params, n_frames)
        flip = bulk_flip_map(vote_am, tgt)
        bulk_mask = np.isin(tgt, BULK_IDX)
        idx = np.where(bulk_mask)
        rgb_nat = gt_f1[p].astype(np.float64); gray_nat = rgb_nat.mean(2)
        rgb_seg = np.stack([ndimage.zoom(rgb_nat[:, :, k], (zy, zx), order=1) for k in range(3)], 2)
        gray_seg = ndimage.zoom(gray_nat, (zy, zx), order=1)
        tex = np.hypot(ndimage.sobel(gray_seg, axis=1), ndimage.sobel(gray_seg, axis=0))
        prior = vote_am["identity"]
        sdf_gt = np.minimum(np.nan_to_num(ndimage.distance_transform_edt(~_boundary(tgt)), posinf=64.0), 64.0)
        sdf_prior = np.minimum(np.nan_to_num(ndimage.distance_transform_edt(~_boundary(prior)), posinf=64.0), 64.0)
        is_hood = (prior == 4).astype(np.float64)
        feat["row_norm"].append(rows_norm[idx]); feat["col_norm"].append(cols_norm[idx])
        feat["is_hood_prior"].append(is_hood[idx])
        feat["vote_conf_ground"].append(vote_conf["ground"][idx]); feat["vote_conf_rotonly"].append(vote_conf["rotonly"][idx])
        feat["vote_conf_identity"].append(vote_conf["identity"][idx]); feat["sdf_prior"].append(sdf_prior[idx])
        feat["margin"].append(margins[p][idx].astype(np.float64))
        feat["texture_grad"].append(tex[idx]); feat["rgb_r"].append(rgb_seg[:, :, 0][idx])
        feat["rgb_g"].append(rgb_seg[:, :, 1][idx]); feat["rgb_b"].append(rgb_seg[:, :, 2][idx])
        feat["sdf_gt"].append(sdf_gt[idx])
        flip_list.append(flip[idx]); pair_list.append(np.full(idx[0].shape[0], p, dtype=np.int32))
        if (p + 1) % 24 == 0 or p == P - 1:
            print(f"  ...{p + 1}/{P}", flush=True)

    F = {c: np.concatenate(feat[c]).astype(np.float32) for c in FNAMES}
    y = np.concatenate(flip_list).astype(np.float64); pair_all = np.concatenate(pair_list)

    # ---- contiguous held-out split ----
    n_tr_pairs = max(1, int(round(P * args.train_frac)))
    tr_mask = pair_all < n_tr_pairs; te_mask = ~tr_mask
    def sub(mask, cap):
        idx = np.where(mask)[0]
        return rng.choice(idx, size=cap, replace=False) if len(idx) > cap else idx
    tr_idx = sub(tr_mask, args.train_sample); te_idx = sub(te_mask, args.eval_sample)
    Ftr = {c: F[c][tr_idx] for c in FNAMES}; ytr = y[tr_idx]
    Fte = {c: F[c][te_idx] for c in FNAMES}; yte = y[te_idx]
    Hflip = binary_entropy_bits(float(yte.mean()))

    free = [c for c in FNAMES if BUCKET[c] == "free"]
    fm = free + [c for c in FNAMES if BUCKET[c] == "margin"]
    full = fm + [c for c in FNAMES if BUCKET[c] == "content"]

    def cols_design(cols):
        mean = {c: float(Ftr[c].mean()) for c in cols if CONT[c]}
        std = {c: float(Ftr[c].std()) for c in cols if CONT[c]}
        return design_matrix(Ftr, cols, mean, std), design_matrix(Fte, cols, mean, std)

    results = {}
    for name, cols in [("free", free), ("free_margin", fm), ("full_content", full)]:
        Xtr, Xte = cols_design(cols)
        # logistic
        w = fit_logistic_irls(Xtr, ytr, l2=2.0)
        plog = _sigmoid(Xte @ w)
        ce_log = cross_entropy_bits(plog, yte); auc_log = auc_score(plog, yte)
        # MLP (drop the bias column for the net; use standardized features)
        pmlp, ce_mlp = train_mlp(Xtr[:, 1:], ytr, Xte[:, 1:], yte)
        auc_mlp = auc_score(pmlp, yte)
        results[name] = {"logistic": {"ce_bits": ce_log, "auc": auc_log},
                         "mlp": {"ce_bits": ce_mlp, "auc": auc_mlp}}
        print(f"  [{name:13s}] logistic CE={ce_log:.5f} AUC={auc_log:.4f} | MLP CE={ce_mlp:.5f} AUC={auc_mlp:.4f}", flush=True)

    mi_content_log = results["free_margin"]["logistic"]["ce_bits"] - results["full_content"]["logistic"]["ce_bits"]
    mi_content_mlp = results["free_margin"]["mlp"]["ce_bits"] - results["full_content"]["mlp"]["ce_bits"]

    # ---- low-margin crux (within margin<0.5): free vs free+content, MLP ----
    low_te = Fte["margin"] < 0.5
    low_tr = Ftr["margin"] < 0.5
    low_crux = None
    if int(low_te.sum()) > 2000 and yte[low_te].sum() > 20 and ytr[low_tr].sum() > 20:
        content_cols = free + [c for c in FNAMES if BUCKET[c] == "content"]
        def low_design(cols):
            mean = {c: float(Ftr[c][low_tr].mean()) for c in cols if CONT[c]}
            std = {c: float(Ftr[c][low_tr].std()) for c in cols if CONT[c]}
            Xtr_l = design_matrix({k: Ftr[k][low_tr] for k in FNAMES}, cols, mean, std)
            Xte_l = design_matrix({k: Fte[k][low_te] for k in FNAMES}, cols, mean, std)
            return Xtr_l, Xte_l
        ytr_l = ytr[low_tr]; yte_l = yte[low_te]
        Xtr_f, Xte_f = low_design(free); Xtr_c, Xte_c = low_design(content_cols)
        pf, ce_f = train_mlp(Xtr_f[:, 1:], ytr_l, Xte_f[:, 1:], yte_l)
        pc, ce_c = train_mlp(Xtr_c[:, 1:], ytr_l, Xte_c[:, 1:], yte_l)
        # logistic low-margin for comparison
        wf = fit_logistic_irls(Xtr_f, ytr_l, l2=2.0); wc = fit_logistic_irls(Xtr_c, ytr_l, l2=2.0)
        cef_log = cross_entropy_bits(_sigmoid(Xte_f @ wf), yte_l)
        cec_log = cross_entropy_bits(_sigmoid(Xte_c @ wc), yte_l)
        low_crux = {
            "low_margin_flip_rate_heldout": float(yte_l.mean()),
            "Hflip_lowmargin_bits": binary_entropy_bits(float(yte_l.mean())),
            "mlp_free_ce": ce_f, "mlp_content_ce": ce_c, "mlp_free_auc": auc_score(pf, yte_l),
            "mlp_content_auc": auc_score(pc, yte_l), "mlp_content_MI_bits": ce_f - ce_c,
            "logistic_free_ce": cef_log, "logistic_content_ce": cec_log,
            "logistic_content_MI_bits": cef_log - cec_log,
            "FEEDkd_logistic_content_MI_within_lowmargin": 0.010529291421311315,
        }
        print(f"  [low-margin crux] MLP content MI={ce_f - ce_c:.5f}b (logistic {cef_log - cec_log:.5f}b; "
              f"FEED-kd logistic 0.01053b)", flush=True)

    elapsed = round(time.time() - t0, 1)
    out = {
        "tool": "tools/measure_content_edge_strong.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS advisory / CPU-torch research-signal]",
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory; not a contest score)",
        "n_pairs": P, "seg_cache_source": seg_source, "no_fake_selfcheck": selfcheck,
        "Hflip_marginal_bits": Hflip,
        "probes": results,
        "content_MI_overall_bits": {
            "logistic": mi_content_log, "mlp": mi_content_mlp,
            "FEEDkd_logistic_reference": 0.00016595846934743148,
            "mlp_beats_logistic_factor": (mi_content_mlp / mi_content_log) if mi_content_log not in (0, None) else None,
        },
        "low_margin_crux": low_crux,
        "VERDICT": {
            "note": ("does a strong MLP extract materially MORE content MI than FEED-kd's logistic "
                     "(0.000166b overall / 0.01053b within low-margin)? MLP content MI >> logistic => the "
                     "content-generator door RE-OPENS (a trained model predicts flips from content). "
                     "MLP ~= logistic => FEED-kd's demotion holds even with a stronger model. CALIBRATED: "
                     "this is PREDICTABILITY of WHICH pixel flips, a SEPARATE question from whether a render "
                     "CORRECTS the flip through R (that is measure_waterfill_through_R.py's Measurement B)."),
            "mlp_overall_content_MI_bits": mi_content_mlp,
            "logistic_overall_content_MI_bits": mi_content_log,
        },
        "assumptions": {
            "PROVEN": "flip = real argmax-disagreement vs frozen CPU-torch SegNet (lstars); contiguous "
                      "held-out split; MLP trained to convergence (early stop on held-out CE).",
            "CONTENT_is_upper_bound": "content features (texture/rgb/sdf_gt) need representing content at "
                                      "decode; this bounds predictability, it is not the free-decoder number.",
            "CAVEAT": "predictability != correctability-through-R; bulk-only; entropy on a finite held-out set.",
        },
        "elapsed_secs": elapsed,
    }
    out_dir = (Path(args.out_dir) if args.out_dir else (REPO / f"experiments/results/content_edge_strong_n{P}"))
    _refuse_tmp(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[content-edge] overall content MI: logistic={mi_content_log:.6f}b  MLP={mi_content_mlp:.6f}b "
          f"(FEED-kd 0.000166b)")
    print(f"[written] {out_dir/'results.json'} (elapsed {elapsed}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
