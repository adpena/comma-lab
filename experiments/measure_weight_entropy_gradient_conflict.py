# SPDX-License-Identifier: MIT
"""P0 MEASUREMENT (operator 2026-07-15): does the weight_entropy rate penalty (lambda=15)
gradient COUNTER the d_seg / d_pose convergence gradient?

Standalone, read-only measurement (NO training, NO paid dispatch, does NOT touch the live
trainer or its launcher). Loads a CACHED epoch-25 levelset witness EMA checkpoint and, at that
FIXED theta, computes SEPARATE per-term weight-space gradients:

  * g_dseg  = grad_theta[ w_seg * seg_l ]     (the realized through-R SegNet-margin descent)
  * g_pose  = grad_theta[ w_pose * pose_term ] (the realized through-R PoseNet descent)
  * g_we    = grad_theta[ lambda * rate_term ] (the Balle weight-entropy penalty; pair-independent)

and reports their alignment TWO ways:

  1. Euclidean cosine (BASELINE only, per the operator metric-correction 2026-07-15).
  2. FISHER / reachable-decision-geometry cosine (AUTHORITY): each weight-gradient direction is
     pulled through to its induced LOGIT change via a forward-mode JVP of the scorer-logit map
     l(theta)=SegNet(R(witness(theta),f1)); the inner product is taken per-pixel in the
     categorical Fisher metric g_i = diag(p_i) - p_i p_i^T (tac.information_geometry.optimal_metric).
     The margin field IS the Fisher surrogate (Pearson 0.978), so this is the geometry d_seg lives in.

SCOPE (labeled honestly): the realized render here is the WITNESS-ALONE render_through_R_mlx (no
seed-island / lane-band compose). At the checkpoint epoch (25) the trainer's lane-band
(start-epoch 500) and chroma-boundary (start-epoch 450) compose/terms are OFF, so the only omitted
surface is the seed-island bulk composite. The witness-alone render is exactly the witness's OWN
d_seg-descent surface -- the surface g_we (a pure function of the same witness weights) acts against
-- so it is the faithful surface for the gradient-DIRECTION question. The seed-compose shift is a
scope caveat noted in the memo.

Run:  .venv/bin/python experiments/measure_weight_entropy_gradient_conflict.py \
          --ckpt .omx/tmp/we_conflict_measure/ckpt_ema_BEST.npz \
          --n-euclid 600 --n-fisher 96 --out .omx/tmp/we_conflict_measure/result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src"), str(REPO / "upstream"), str(REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _flatten_grad_tree(tree) -> tuple[list[str], np.ndarray]:
    """tree_flatten a param/grad tree -> (sorted names, concatenated float64 vector)."""
    from mlx.utils import tree_flatten

    items = sorted(tree_flatten(tree), key=lambda kv: kv[0])
    names = [n for n, _ in items]
    vec = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for _, a in items])
    return names, vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=".omx/tmp/we_conflict_measure/ckpt_ema_BEST.npz")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n-euclid", type=int, default=600, help="pairs for the weight-space (Euclidean) accumulation")
    ap.add_argument("--n-fisher", type=int, default=96, help="pairs for the JVP Fisher-metric measurement (evenly spaced subset)")
    ap.add_argument("--we-lambda", type=float, default=15.0)
    ap.add_argument("--we-sigma", type=float, default=0.2)
    ap.add_argument("--w-seg", type=float, default=100.0)
    ap.add_argument("--w-pose", type=float, default=1.0)
    ap.add_argument("--hinge", type=float, default=4.0)
    ap.add_argument("--margin-target", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=1.0, help="unify_tau tau at epoch 25 (== softmax_temp_start)")
    ap.add_argument("--pose-eps", type=float, default=1e-2)
    ap.add_argument("--out", default=".omx/tmp/we_conflict_measure/result.json")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from train_levelset_witness_realized_through_R_mlx import build_levelset_rgb_witness
    from train_witness_realized_through_R_mlx import (
        _build_render_coords,
        load_gt_from_cache,
        make_loss_fn,
        render_through_R_mlx,
    )

    from tac.boundary_math.lever_b_levelset_generator import (  # type: ignore
        CurveletBankConfig,
        curvelet_directional_B,
        curvelet_feats,
    )
    from tac.boundary_math.weight_entropy_penalty_mlx import weight_entropy_rate_term_mlx
    from tac.information_geometry.optimal_metric import log_partition_hessian
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    ck = Path(args.ckpt)
    z = np.load(ck, allow_pickle=False)
    cfg = {k[6:]: z[k] for k in z.files if k.startswith("__cfg_")}
    render_h, render_w = (int(z["__render_hw"][0]), int(z["__render_hw"][1]))
    in_feat = int(cfg["in_feat"])
    hidden_dim = int(cfg["hidden_dim"])
    n_hidden = int(cfg["n_hidden"])
    mod_dim = int(z["film.weight"].shape[1])
    epoch = int(z["__epoch"])
    print(json.dumps({"stage": "ckpt", "epoch": epoch, "in_feat": in_feat, "hidden_dim": hidden_dim,
                      "n_hidden": n_hidden, "mod_dim": mod_dim, "render_hw": [render_h, render_w],
                      "activation": str(cfg["activation"]), "self_orient": int(cfg["self_orient"]),
                      "softmax_temp": float(cfg["softmax_temp"])}), flush=True)

    # ---- coord features (self_orient=0 => one shared curvelet bank tensor) ----
    coords_np = _build_render_coords(render_h, render_w)
    bank = CurveletBankConfig(
        n_scales=int(z["__bank_n_scales"]), n_orient0=int(z["__bank_n_orient0"]),
        f0=float(z["__bank_f0"]), base=float(z["__bank_base"]), n_iso=int(z["__bank_n_iso"]),
    )
    B = curvelet_directional_B(bank, max_freq=float(cfg["max_bank_freq"]))
    curv = curvelet_feats(coords_np, B).astype(np.float32)
    assert curv.shape[1] == in_feat, f"coord feat width {curv.shape[1]} != in_feat {in_feat}"
    coord_feats_mx = mx.array(curv)

    # ---- build module + load checkpoint params ----
    model = build_levelset_rgb_witness(
        num_pairs=int(z["code"].shape[0] // 2), in_feat=in_feat, hidden_dim=hidden_dim,
        n_hidden=n_hidden, mod_dim=mod_dim, n_classes=int(z["out_sdf.weight"].shape[0]),
        activation=str(cfg["activation"]), softmax_temp=float(cfg["softmax_temp"]),
        wire_w0=float(cfg.get("wire_w0", 20.0)), wire_s0=float(cfg.get("wire_s0", 10.0)),
        hosc_beta=float(cfg["hosc_beta"]), hosc_omega=float(cfg["hosc_omega"]),
        chroma=bool(int(cfg["chroma"])), render_h=render_h, render_w=render_w,
    )
    mx.eval(model.parameters())
    # load params: every non-'__' npz key that matches a module param name
    param_names = {n for n, _ in tree_flatten(model.parameters())}
    updates = []
    for k in z.files:
        if k.startswith("__"):
            continue
        if k in param_names:
            updates.append((k, mx.array(z[k])))
    loaded = {n for n, _ in updates}
    missing = param_names - loaded
    model.update(tree_unflatten(updates))
    mx.eval(model.parameters())
    print(json.dumps({"stage": "loaded_params", "n_loaded": len(loaded),
                      "n_missing": len(missing), "missing": sorted(missing)[:8]}), flush=True)

    adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
    n_all = int(z["code"].shape[0] // 2)
    n_eu = min(args.n_euclid, n_all)
    gt, _seg_cpu, _pose_cpu = load_gt_from_cache(Path(args.gt_cache), n_all)

    base_loss = make_loss_fn(
        adapter, render_h, render_w, score_domain=True, pose_eps=args.pose_eps,
        seg_loss="ce", tau_softplus_tau=0.3, l7_mult=4.0, l7_threshold=1.0, render_fn=None,
    )

    def _oh(pi: int):
        ls = np.asarray(gt.lstars[pi], dtype=np.int64)  # (H,W)
        oh = np.eye(int(z["out_sdf.weight"].shape[0]), dtype=np.float32)[ls]  # (H,W,K)
        return mx.array(oh[None])  # (1,H,W,K)

    def _margin(pi: int):
        return mx.array(np.asarray(gt.margins[pi], np.float32)[None])  # (1,H,W)

    def _pose(pi: int):
        return mx.array(np.asarray(gt.gt_poses[pi], np.float32))  # (6,)

    # ---- per-term weight-space gradients (accumulated over pairs) ----
    def seg_term(m, pi):
        t: dict = {}
        base_loss(m, coord_feats_mx, 2 * pi, 2 * pi + 1, _oh(pi), _margin(pi), _pose(pi),
                  args.w_seg, args.w_pose, args.hinge, args.margin_target,
                  seg_form="unify_tau", tau_override=args.tau, terms_out=t, compute_pose=False)
        return t["seg"]

    def pose_term(m, pi):
        t: dict = {}
        base_loss(m, coord_feats_mx, 2 * pi, 2 * pi + 1, _oh(pi), _margin(pi), _pose(pi),
                  args.w_seg, args.w_pose, args.hinge, args.margin_target,
                  seg_form="unify_tau", tau_override=args.tau, terms_out=t, compute_pose=True)
        return t["pose"]

    def we_term(m):
        _bits, rate = weight_entropy_rate_term_mlx(m, sigma=args.we_sigma)
        return args.we_lambda * rate

    seg_vg = nn.value_and_grad(model, lambda m, pi: seg_term(m, pi))
    pose_vg = nn.value_and_grad(model, lambda m, pi: pose_term(m, pi))
    we_vg = nn.value_and_grad(model, we_term)

    # weight entropy gradient (pair-independent, once)
    we_val, g_we_tree = we_vg(model)
    mx.eval(we_val, g_we_tree)
    names_we, v_we = _flatten_grad_tree(g_we_tree)  # already includes lambda

    # accumulate seg + pose grads (mean over pairs). Keep the seg direction as a TREE (for the
    # Fisher JVP) and as a flat vector (for the Euclidean cosine).
    g_seg_sum = None
    g_pose_sum = None
    seg_tree_sum: dict = {}
    seg_loss_acc = 0.0
    pose_loss_acc = 0.0
    for pi in range(n_eu):
        sval, gseg = seg_vg(model, pi)
        pval, gpose = pose_vg(model, pi)
        mx.eval(sval, gseg, pval, gpose)
        seg_loss_acc += float(sval)
        pose_loss_acc += float(pval)
        _, vs = _flatten_grad_tree(gseg)
        _, vp = _flatten_grad_tree(gpose)
        g_seg_sum = vs if g_seg_sum is None else g_seg_sum + vs
        g_pose_sum = vp if g_pose_sum is None else g_pose_sum + vp
        for n, a in tree_flatten(gseg):
            seg_tree_sum[n] = a if n not in seg_tree_sum else seg_tree_sum[n] + a
        if (pi + 1) % 50 == 0:
            print(json.dumps({"stage": "accum", "pair": pi + 1, "n": n_eu}), flush=True)
    v_seg = g_seg_sum / n_eu  # mean-over-pairs, the actual step direction for the seg term
    v_pose = g_pose_sum / n_eu
    seg_dir_tree = {n: seg_tree_sum[n] / float(n_eu) for n in seg_tree_sum}

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")

    eu_cos_seg = cos(v_we, v_seg)
    eu_cos_pose = cos(v_we, v_pose)
    rel_norm_seg = float(np.linalg.norm(v_we) / (np.linalg.norm(v_seg) + 1e-30))
    rel_norm_pose = float(np.linalg.norm(v_we) / (np.linalg.norm(v_pose) + 1e-30))
    # PCGrad negative-projection conflict (Euclidean): how much of v_seg is opposed by v_we.
    dot_we_seg = float(v_we @ v_seg)
    eu_conflict_seg = max(0.0, -dot_we_seg) / (np.linalg.norm(v_seg) + 1e-30)

    print(json.dumps({"stage": "euclid_done", "n": n_eu,
                      "seg_loss_mean": seg_loss_acc / n_eu, "pose_loss_mean": pose_loss_acc / n_eu,
                      "eu_cos_seg": eu_cos_seg, "eu_cos_pose": eu_cos_pose,
                      "rel_norm_seg": rel_norm_seg, "rel_norm_pose": rel_norm_pose,
                      "eu_conflict_seg": eu_conflict_seg}), flush=True)

    # ---- FISHER metric (authority): pull each weight direction to LOGIT space, per-pixel Fisher inner.
    # mx.jvp is not implemented for Convolution (SegNet), but the uint8-STE ROUND lives in the RENDER
    # (R), NOT in the SegNet. So: (1) JVP the render-only map theta->R-frame (no conv; the STE round's
    # jvp == identity, so this is the training-faithful Dframe); (2) linearize the SMOOTH SegNet
    # (conv/relu/bn, NO round) by a central finite-difference along the UNIT frame direction, rescaled
    # by |Dframe| -> Dlogit = J_segnet . Dframe (first-order exact for a smooth map).
    p_items = tree_flatten(model.trainable_parameters())
    p_names = [n for n, _ in p_items]
    primals = [a for _, a in p_items]
    _dwe = dict(tree_flatten(g_we_tree))
    seg_tan = [seg_dir_tree[n] for n in p_names]      # seg descent direction (mean over n_euclid pairs)
    we_tan = [mx.array(_dwe[n]) for n in p_names]      # lambda * weight-entropy gradient

    n_fi = min(args.n_fisher, n_all)
    fisher_idx = np.linspace(0, n_all - 1, n_fi).astype(int).tolist()
    _fd_h = 0.5  # SegNet FD step in pixel units (frames ~0..255; smooth net -> linear regime)

    def _seg_np(frame_mx):
        return np.asarray(adapter.segnet(frame_mx)[0], np.float64)  # (H,W,K)

    def _dlogit_from_dframe(frame0_mx, dframe_mx):
        # directional derivative of SegNet at frame0 along dframe (unit-normalize + rescale).
        dn = float(mx.sqrt(mx.sum(dframe_mx * dframe_mx)))
        if dn <= 0.0:
            return np.zeros_like(_seg_np(frame0_mx))
        u = dframe_mx / dn
        lp = _seg_np(frame0_mx + _fd_h * u)
        lm = _seg_np(frame0_mx - _fd_h * u)
        return (lp - lm) / (2.0 * _fd_h) * dn  # J_segnet . dframe

    fi_num = 0.0   # <Ja, Jb>_g  (a=seg, b=we)
    fi_aa = 0.0    # <Ja, Ja>_g
    fi_bb = 0.0    # <Jb, Jb>_g
    _selfcheck_done = False
    for j, pi in enumerate(fisher_idx):
        code1 = 2 * pi + 1

        def render_only(*arrs, _c1=code1):
            model.update(tree_unflatten(list(zip(p_names, arrs, strict=True))))
            return render_through_R_mlx(model, coord_feats_mx, _c1, render_h, render_w)  # (1,H,W,3)

        frs, dfr_seg = mx.jvp(render_only, primals, seg_tan)
        _, dfr_we = mx.jvp(render_only, primals, we_tan)
        frame0 = frs[0]
        mx.eval(frame0, dfr_seg[0], dfr_we[0])
        model.update(tree_unflatten(list(zip(p_names, primals, strict=True))))  # restore theta
        logits = adapter.segnet(frame0)[0]  # (H,W,K)
        mx.eval(logits)
        A = _dlogit_from_dframe(frame0, dfr_seg[0]).reshape(-1, logits.shape[-1])
        Bx = _dlogit_from_dframe(frame0, dfr_we[0]).reshape(-1, logits.shape[-1])
        L = np.asarray(logits, np.float64).reshape(-1, logits.shape[-1])  # (Npx, K)
        if not _selfcheck_done:
            # verify the vectorized Fisher inner (below) == the explicit u^T g u with the canonical
            # metric g = diag(p) - p p^T from tac.information_geometry.optimal_metric (NO-FAKE: the
            # vectorized form IS that identity). Checked on one representative pixel.
            _px = int(np.argmax(L.max(axis=1) - np.partition(L, -2, axis=1)[:, -2]))  # a small-margin px
            _p1 = np.exp(L[_px] - L[_px].max())
            _p1 = _p1 / _p1.sum()
            _g = log_partition_hessian(_p1)
            _explicit = float(A[_px] @ _g @ Bx[_px])
            _vec = float((_p1 * A[_px] * Bx[_px]).sum() - (_p1 * A[_px]).sum() * (_p1 * Bx[_px]).sum())
            assert abs(_explicit - _vec) <= 1e-6 * (abs(_explicit) + 1.0), (_explicit, _vec)
            _selfcheck_done = True
        # per-pixel softmax + Fisher metric g = diag(p) - p p^T; inner = a^T g b = (a.p b.p sums)
        # vectorized: a^T g b = sum(p*a*b) - (sum(p*a))*(sum(p*b))
        m = L.max(axis=1, keepdims=True)
        e = np.exp(L - m)
        P = e / e.sum(axis=1, keepdims=True)  # (Npx,K)
        pa = (P * A).sum(axis=1)
        pb = (P * Bx).sum(axis=1)
        ab = (P * A * Bx).sum(axis=1) - pa * pb
        aa = (P * A * A).sum(axis=1) - pa * pa
        bb = (P * Bx * Bx).sum(axis=1) - pb * pb
        fi_num += float(ab.sum())
        fi_aa += float(aa.sum())
        fi_bb += float(bb.sum())
        if (j + 1) % 16 == 0:
            print(json.dumps({"stage": "fisher", "pair": j + 1, "n": n_fi}), flush=True)

    fisher_cos_seg = fi_num / (np.sqrt(fi_aa) * np.sqrt(fi_bb) + 1e-300) if fi_aa > 0 and fi_bb > 0 else float("nan")
    # Fisher PCGrad conflict: how much of the seg descent is opposed by we, in Fisher units.
    fisher_conflict_seg = max(0.0, -fi_num) / (np.sqrt(fi_aa) + 1e-300)
    fisher_relnorm_seg = float(np.sqrt(fi_bb) / (np.sqrt(fi_aa) + 1e-300))

    result = {
        "checkpoint": str(ck), "epoch": epoch, "n_euclid": n_eu, "n_fisher": n_fi,
        "we_lambda": args.we_lambda, "w_seg": args.w_seg, "w_pose": args.w_pose,
        "tau": args.tau, "seg_form": "unify_tau", "hinge": args.hinge,
        "scope": "witness-alone realized render_through_R_mlx (no seed-island compose); "
                 "lane-band(start500)/chroma-boundary(start450) OFF at epoch 25",
        "seg_loss_mean": seg_loss_acc / n_eu, "pose_loss_mean": pose_loss_acc / n_eu,
        "euclid_baseline": {
            "cos_we_seg": eu_cos_seg, "cos_we_pose": eu_cos_pose,
            "rel_norm_lam_we_over_seg": rel_norm_seg, "rel_norm_lam_we_over_pose": rel_norm_pose,
            "pcgrad_conflict_seg": eu_conflict_seg, "dot_we_seg": dot_we_seg,
        },
        "fisher_authority": {
            "cos_we_seg": float(fisher_cos_seg),
            "rel_norm_we_over_seg": fisher_relnorm_seg,
            "pcgrad_conflict_seg": float(fisher_conflict_seg),
            "inner_we_seg": fi_num, "norm2_seg": fi_aa, "norm2_we": fi_bb,
        },
    }

    def _verdict(c):
        if not np.isfinite(c):
            return "undetermined"
        if c < -0.05:
            return "ANTAGONISTIC"
        if c > 0.05:
            return "synergistic"
        return "orthogonal"

    result["verdict_fisher_seg"] = _verdict(fisher_cos_seg)
    result["verdict_euclid_seg"] = _verdict(eu_cos_seg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": "RESULT", **result}), flush=True)


if __name__ == "__main__":
    main()
