# SPDX-License-Identifier: MIT
"""Dual-metric gradient read-back harness (SPEC_v10 §13.4 surface 3, build-wave arm A).

Generalizes ``experiments/measure_weight_entropy_gradient_conflict.py`` into a TOOL: for ANY
supported loss-term pair at a cached levelset-witness checkpoint, report the §13.4(3) TRIPLE —

    * Euclidean cosine  (weight-space, BASELINE only)
    * Fisher cosine     (decision-geometry AUTHORITY: per-pixel categorical Fisher inner of the
                         induced logit changes, g = diag(p) − p pᵀ)
    * rel-norm          (the decisive magnitude ratio, in BOTH metrics)

per the operator 2026-07-17 dual-metric discipline ("both are informative stop forgetting that";
measured sign-flip anchor: −0.00105 Euclid vs +0.0435 Fisher on the SAME pair, FEED-we-conflict).

Supported terms: ``seg`` (the armed base seg form) · ``pose`` · ``weight_entropy`` ·
``phase_advect`` (T1 cross-pair phase-advection, θ-independent providers reconstructed with the
trainer's own shared phase primitives).

ADVISORY, NON-PROMOTABLE: [macOS-MLX research-signal] / [macOS advisory]; score_claim=false;
n<600 subsets are labeled; nothing here is a score row and the pointer is UNMOVED by construction
(read-only measurement — no training state is touched).

SELF-ORIENT RECONSTRUCTION (labeled approximation): a self_orient=1 checkpoint's per-pair
directional feats depend on the LAST training-time reorientation (not persisted in the EMA ckpt).
The harness reconstructs them by the DECODE-style self-orientation fixed point (zero-dir iso pass
-> own-sdf argmax -> tangent feats, iterated --so-fixed-point-iters times) — the same procedure
the byte-close decode uses; the residual difference vs the training-time orientation (<=
--reorient-every epochs stale) is a documented approximation, not a bit-identity claim.

Run (the arm-A owed row: phase_advect vs armed seg base at the live c2 BEST ckpt):
  .venv/bin/python tools/dual_metric_readback.py \
      --ckpt experiments/results/levelset_n600_witness_20260717T113932Z/levelset_witness_ema_BEST.npz \
      --term-a phase_advect --term-b seg --seg-form tau_softplus --tau 0.3 \
      --pairs 96 --fisher-pairs 96 --out .omx/tmp/dual_metric_readback/pa_vs_seg.json
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


def _resolve_m_safe() -> float:
    """m_safe = headroom * delta_R through the canonical law (n600 artifact, ddm_dr1
    2026-09-04); the exact n600 fallback keeps the tool runnable without ``tac``."""
    try:
        from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
            resolve_margin_band_threshold,
        )

        return float(resolve_margin_band_threshold().m_safe)
    except Exception:  # tool must stay runnable; the value is the same law's own fallback
        return 0.04376363754272461


_M_SAFE = _resolve_m_safe()

SUPPORTED_TERMS = ("seg", "pose", "weight_entropy", "phase_advect", "margin_satisfice", "subpix")


def _flatten_grad_tree(tree):
    from mlx.utils import tree_flatten

    items = sorted(tree_flatten(tree), key=lambda kv: kv[0])
    names = [n for n, _ in items]
    vec = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for _, a in items])
    return names, vec


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _verdict(c: float) -> str:
    if not np.isfinite(c):
        return "undetermined"
    if c < -0.05:
        return "ANTAGONISTIC"
    if c > 0.05:
        return "synergistic"
    return "orthogonal"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--upstream", default=str(REPO / "upstream"),
                    help="pinned upstream snapshot dir (worktrees without a local copy pass the "
                    "primary checkout's absolute path)")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--term-a", required=True, choices=SUPPORTED_TERMS)
    ap.add_argument("--term-b", required=True, choices=SUPPORTED_TERMS)
    ap.add_argument("--pairs", type=int, default=96,
                    help="pairs for the weight-space (Euclidean) accumulation (evenly spaced; "
                    "<600 => labeled advisory subset)")
    ap.add_argument("--fisher-pairs", type=int, default=96,
                    help="pairs for the JVP+FD Fisher-metric leg (evenly spaced subset)")
    # armed seg-base params (match the run's launch flags; defaults = the c2 live run at ep725).
    ap.add_argument("--seg-form", default="tau_softplus",
                    choices=("ce", "tau_softplus", "unify_tau", "l7_softplus", "margin_hinge"))
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--w-seg", type=float, default=100.0)
    ap.add_argument("--w-pose", type=float, default=1.0)
    ap.add_argument("--hinge", type=float, default=4.0)
    ap.add_argument("--margin-target", type=float, default=0.5)
    ap.add_argument("--pose-eps", type=float, default=1e-2)
    # phase-advect params (defaults = the c2 launch flags + trainer gfc defaults).
    ap.add_argument("--pa-weight", type=float, default=0.4)
    ap.add_argument("--pa-band", type=float, default=2.0)
    ap.add_argument("--pa-classes", default="0,1,2")
    ap.add_argument("--pa-eps", type=float, default=1e-6)
    ap.add_argument("--gfc-pitch", type=float, default=-0.01)
    ap.add_argument("--gfc-s-t", type=float, default=-0.003224707899359239)
    ap.add_argument("--gfc-s-r", type=float, default=0.0)
    # weight-entropy params.
    ap.add_argument("--we-lambda", type=float, default=15.0)
    ap.add_argument("--we-sigma", type=float, default=0.2)
    # margin-satisfice params (defaults = the c2 launch flags; Force-2 #360).
    ap.add_argument("--ms-weight", type=float, default=0.2)
    ap.add_argument("--ms-msafe", type=float, default=_M_SAFE)
    ap.add_argument("--ms-band", type=float, default=2.0)
    # subpix boundary-placement params (defaults = the c2 launch flags; Force-3 #360).
    ap.add_argument("--subpix-weight", type=float, default=0.3)
    ap.add_argument("--subpix-band", type=float, default=1.0)
    ap.add_argument("--subpix-eps", type=float, default=1e-6)
    ap.add_argument("--subpix-ew-path", default="reports/pa_edge_weights.json",
                    help="P0 FORCE-3 W_e flip-mass edge-weight artifact (pa_flipmass; the c2 "
                    "launch's --seg-subpix-edge-weight-path). Missing file FAILS LOUD (no silent "
                    "uniform downgrade in a measurement tool).")
    ap.add_argument("--state-label", default="",
                    help="free-text state annotation for the output row (e.g. 'EMA-BEST-ep725', "
                    "'warm-start-seed-ep650-live-init')")
    # self-orient reconstruction.
    ap.add_argument("--so-fixed-point-iters", type=int, default=2)
    ap.add_argument("--fd-h", type=float, default=0.5,
                    help="SegNet central-FD step (pixel units) for the Fisher leg")
    ap.add_argument("--out", default=".omx/tmp/dual_metric_readback/result.json")
    args = ap.parse_args()

    if args.term_a == args.term_b:
        raise SystemExit("term-a and term-b must differ")

    upstream_dir = Path(args.upstream).resolve()
    if not (upstream_dir / "modules.py").exists():
        raise SystemExit(f"--upstream {upstream_dir} has no modules.py (pinned snapshot required)")
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))

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

    from tac.boundary_math.lever_b_generator import self_orientation_directional_feats
    from tac.boundary_math.lever_b_levelset_generator import (
        CurveletBankConfig,
        curvelet_directional_B,
        curvelet_feats,
    )
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    ck = Path(args.ckpt)
    z = np.load(ck, allow_pickle=False)
    cfg = {k[6:]: z[k] for k in z.files if k.startswith("__cfg_")}
    render_h, render_w = (int(z["__render_hw"][0]), int(z["__render_hw"][1]))
    in_feat = int(cfg["in_feat"])
    n_classes = int(z["out_sdf.weight"].shape[0])
    mod_dim = int(z["film.weight"].shape[1])
    epoch = int(z["__epoch"])
    self_orient = bool(int(cfg.get("self_orient", 0)))
    n_dir_freqs = int(cfg.get("n_dir_freqs", 0))
    dir_w = 4 * n_dir_freqs if self_orient else 0
    print(json.dumps({"stage": "ckpt", "epoch": epoch, "in_feat": in_feat,
                      "self_orient": self_orient, "dir_w": dir_w,
                      "render_hw": [render_h, render_w],
                      "activation": str(cfg["activation"]),
                      "hosc_beta_live": float(cfg["hosc_beta"]),
                      "softmax_temp_live": float(cfg["softmax_temp"])}), flush=True)

    # ---- shared curvelet base feats ----
    coords_np = _build_render_coords(render_h, render_w)
    bank = CurveletBankConfig(
        n_scales=int(z["__bank_n_scales"]), n_orient0=int(z["__bank_n_orient0"]),
        f0=float(z["__bank_f0"]), base=float(z["__bank_base"]), n_iso=int(z["__bank_n_iso"]),
    )
    B = curvelet_directional_B(bank, max_freq=float(cfg["max_bank_freq"]))
    curv = curvelet_feats(coords_np, B).astype(np.float32)
    assert curv.shape[1] + dir_w == in_feat, (
        f"feat width mismatch: curvelet {curv.shape[1]} + dir {dir_w} != in_feat {in_feat}")

    # ---- model build + load ----
    model = build_levelset_rgb_witness(
        num_pairs=int(z["code"].shape[0] // 2), in_feat=in_feat,
        hidden_dim=int(cfg["hidden_dim"]), n_hidden=int(cfg["n_hidden"]), mod_dim=mod_dim,
        n_classes=n_classes, activation=str(cfg["activation"]),
        softmax_temp=float(cfg["softmax_temp"]),
        wire_w0=float(cfg.get("wire_w0", 20.0)), wire_s0=float(cfg.get("wire_s0", 10.0)),
        hosc_beta=float(cfg["hosc_beta"]), hosc_omega=float(cfg["hosc_omega"]),
        chroma=bool(int(cfg.get("chroma", 0))), render_h=render_h, render_w=render_w,
    )
    mx.eval(model.parameters())
    param_names_all = {n for n, _ in tree_flatten(model.parameters())}
    updates = [(k, mx.array(z[k])) for k in z.files
               if not k.startswith("__") and k in param_names_all]
    model.update(tree_unflatten(updates))
    mx.eval(model.parameters())
    missing = sorted(param_names_all - {n for n, _ in updates})
    print(json.dumps({"stage": "loaded_params", "n_loaded": len(updates),
                      "missing": missing[:6]}), flush=True)

    adapter = load_mlx_distortion_scorer_adapter_from_upstream(upstream_dir, device="cpu")
    n_all = int(z["code"].shape[0] // 2)
    gt, _sc, _pc = load_gt_from_cache(Path(args.gt_cache), n_all)

    # ---- per-pair coord feats (self-orient decode-style fixed point; cached per pair) ----
    _feats_cache: dict[int, np.ndarray] = {}

    def feats_np_for_pair(pi: int) -> np.ndarray:
        if not self_orient:
            return curv
        if pi in _feats_cache:
            return _feats_cache[pi]
        dirf = np.zeros((curv.shape[0], dir_w), np.float32)
        for _ in range(max(int(args.so_fixed_point_iters), 1)):
            feats = np.concatenate([curv, dirf], axis=-1).astype(np.float32)
            phi = model.sdf(mx.array(feats), 2 * pi + 1)
            am = mx.argmax(phi, axis=-1)
            mx.eval(am)
            argmax = np.asarray(am).reshape(render_h, render_w).astype(np.int64)
            dirf = self_orientation_directional_feats(
                coords_np, argmax, n_freqs=n_dir_freqs,
                freq_across=float(cfg["freq_across"]),
                freq_along=float(cfg["freq_along"])).astype(np.float32)
            del phi, am
        out = np.concatenate([curv, dirf], axis=-1).astype(np.float32)
        _feats_cache.clear()  # keep exactly one pair resident (RSS-bounded)
        _feats_cache[pi] = out
        return out

    # ---- base loss (the armed seg form / pose) ----
    base_loss = make_loss_fn(
        adapter, render_h, render_w, score_domain=True, pose_eps=args.pose_eps,
        seg_loss="ce", tau_softplus_tau=args.tau, render_fn=None,
    )

    def _oh(pi: int):
        ls = np.asarray(gt.lstars[pi], dtype=np.int64)
        return mx.array(np.eye(n_classes, dtype=np.float32)[ls][None])

    def _mg(pi: int):
        return mx.array(np.asarray(gt.margins[pi], np.float32)[None])

    def _pose_t(pi: int):
        return mx.array(np.asarray(gt.gt_poses[pi], np.float32))

    # ---- phase-advect θ-independent providers (the trainer's own precompute, subset-aware) ----
    _pa_state: dict = {}

    def _pa_providers(pi: int):
        if pi in _pa_state:
            return _pa_state[pi]
        from tac.boundary_math.phase_primitives import (
            advect_tie_field_numpy,
            cross_scored_frame_xi_interp,
            gt_tie_targets_numpy,
        )
        from tac.boundary_math.warp_real_luma_frame0 import (
            GroundHomographyGeom,
            xi_from_pose_calibration,
        )
        H, W = np.asarray(gt.lstars[0]).shape
        geom = GroundHomographyGeom.eon(native_hw=(H, W), pitch=args.gfc_pitch)
        sel = {int(c) for c in args.pa_classes.split(",") if c.strip() != ""}
        t_p, d_p, a_p = gt_tie_targets_numpy(
            np.asarray(gt.lstars[pi]), np.asarray(gt.margins[pi]),
            band=args.pa_band, eps=args.pa_eps)
        ann_p = (np.asarray(gt.margins[pi], np.float32) < args.pa_band)
        ground_p = np.isin(np.asarray(gt.lstars[pi]), list(sel))
        if pi == 0:
            ref = np.full((H, W), -1.0, np.float32)
            wm = np.zeros((H, W), np.float32)
        else:
            xi_prev = np.asarray(xi_from_pose_calibration(
                np.asarray(gt.gt_poses[pi - 1]), args.gfc_s_t, args.gfc_s_r, args.gfc_pitch),
                np.float64)
            xi_cur = np.asarray(xi_from_pose_calibration(
                np.asarray(gt.gt_poses[pi]), args.gfc_s_t, args.gfc_s_r, args.gfc_pitch),
                np.float64)
            xi_cross = cross_scored_frame_xi_interp(xi_prev, xi_cur)
            t_prev, _d_prev, a_prev = gt_tie_targets_numpy(
                np.asarray(gt.lstars[pi - 1]), np.asarray(gt.margins[pi - 1]),
                band=args.pa_band, eps=args.pa_eps)
            val_prev = np.where(t_prev >= 0.0, t_prev, 0.0).astype(np.float32)
            ref_w = advect_tie_field_numpy(val_prev, xi_cross, geom)
            act_w = advect_tie_field_numpy(a_prev.astype(np.float32), xi_cross, geom)
            ref_active = act_w >= 0.5
            ref = np.where(ref_active, ref_w, -1.0).astype(np.float32)
            wm = (ann_p & ground_p & ref_active).astype(np.float32)
        prov = (mx.array(ref[None]), mx.array(d_p[None]), mx.array(wm[None]))
        _pa_state.clear()
        _pa_state[pi] = prov
        return prov

    # ---- term functions: (model, pi) -> scalar mx term (weighted as trained) ----
    def seg_term(m, pi):
        t: dict = {}
        base_loss(m, mx.array(feats_np_for_pair(pi)), 2 * pi, 2 * pi + 1, _oh(pi), _mg(pi),
                  _pose_t(pi), args.w_seg, args.w_pose, args.hinge, args.margin_target,
                  seg_form=args.seg_form, tau_override=args.tau, terms_out=t,
                  compute_pose=False)
        return t["seg"]

    def pose_term(m, pi):
        t: dict = {}
        base_loss(m, mx.array(feats_np_for_pair(pi)), 2 * pi, 2 * pi + 1, _oh(pi), _mg(pi),
                  _pose_t(pi), args.w_seg, args.w_pose, args.hinge, args.margin_target,
                  seg_form=args.seg_form, tau_override=args.tau, terms_out=t,
                  compute_pose=True)
        return t["pose"]

    def phase_term(m, pi):
        from tac.boundary_math.phase_primitives import (
            phase_advection_weighted_mse_mlx,
            witness_tie_coordinate_mlx,
        )
        ref, dirm, wm = _pa_providers(pi)
        f1 = render_through_R_mlx(m, mx.array(feats_np_for_pair(pi)), 2 * pi + 1,
                                  render_h, render_w)
        seg_logits = adapter.segnet(f1)
        oh = _oh(pi)
        gt_logit = mx.sum(seg_logits * oh, axis=-1)
        runner_up = mx.max(seg_logits + oh * (-1e9), axis=-1)
        signed = gt_logit - runner_up
        t_wit = witness_tie_coordinate_mlx(signed, dirm, args.pa_eps)
        return args.pa_weight * phase_advection_weighted_mse_mlx(t_wit, ref, wm, args.pa_eps)

    def we_term(m, _pi=None):
        from tac.boundary_math.weight_entropy_penalty_mlx import weight_entropy_rate_term_mlx
        _bits, rate = weight_entropy_rate_term_mlx(m, sigma=args.we_sigma)
        return args.we_lambda * rate

    def _signed_for(m, pi):
        """Realized live signed margin (1,H,W) — the trainer's shared ``_signed``."""
        f1 = render_through_R_mlx(m, mx.array(feats_np_for_pair(pi)), 2 * pi + 1,
                                  render_h, render_w)
        seg_logits = adapter.segnet(f1)
        oh = _oh(pi)
        gt_logit = mx.sum(seg_logits * oh, axis=-1)
        runner_up = mx.max(seg_logits + oh * (-1e9), axis=-1)
        return gt_logit - runner_up

    def ms_term(m, pi):
        # P0 FORCE 2 (trainer block @ ms_w>0): one-sided relu(m_safe - m_wit) on the
        # theta-independent GT-margin annulus (|GT margin| < band), mean over annulus px.
        ann = mx.array((np.asarray(gt.margins[pi], np.float32) < args.ms_band)
                       .astype(np.float32)[None])                       # (1,H,W)
        signed = _signed_for(m, pi)
        hinge = mx.maximum(args.ms_msafe - signed, 0.0) * ann
        return args.ms_weight * (mx.sum(hinge) / (mx.sum(ann) + 1e-6))

    _sx_cache: dict = {}

    def _subpix_providers(pi: int):
        """The trainer's theta-independent genuine-V straddle precompute (t, dir, W_e map),
        replicated verbatim from the subpix_w>0 block (dominant = shallower partner margin,
        ties -> right; W_e from the pa_flipmass artifact — FAIL LOUD if missing)."""
        if pi in _sx_cache:
            return _sx_cache[pi]
        _we_p = Path(args.subpix_ew_path)
        if not _we_p.is_absolute():
            _we_p = REPO / args.subpix_ew_path
        if not _we_p.is_file():
            raise SystemExit(f"subpix W_e artifact missing: {_we_p} (fail loud; pass "
                             "--subpix-ew-path to the pa_flipmass artifact)")
        we_mat = np.asarray(json.loads(_we_p.read_text())["W_e"], dtype=np.float32)
        assert we_mat.shape == (5, 5), f"W_e shape {we_mat.shape} != (5,5)"
        lst = np.asarray(gt.lstars[pi], np.int64)
        mg = np.asarray(gt.margins[pi], np.float32)
        H, W = lst.shape
        eps = args.subpix_eps
        band = args.subpix_band
        dh = lst[:, :-1] != lst[:, 1:]
        mph, mqh = mg[:, :-1], mg[:, 1:]
        th = mph / (mph + mqh + eps)
        vh = dh & (mph < band) & (mqh < band)
        dv = lst[:-1, :] != lst[1:, :]
        mpv, mqv = mg[:-1, :], mg[1:, :]
        tv = mpv / (mpv + mqv + eps)
        vv = dv & (mpv < band) & (mqv < band)
        has_r = np.zeros((H, W), bool); has_r[:, :W - 1] = vh
        qr = np.full((H, W), np.inf, np.float32); qr[:, :W - 1] = mqh
        tr = np.zeros((H, W), np.float32); tr[:, :W - 1] = th
        has_d = np.zeros((H, W), bool); has_d[:H - 1, :] = vv
        qd = np.full((H, W), np.inf, np.float32); qd[:H - 1, :] = mqv
        td = np.zeros((H, W), np.float32); td[:H - 1, :] = tv
        pick_r = has_r & (~has_d | (qr <= qd))
        pick_d = has_d & (~has_r | (qd < qr))
        t_full = np.full((H, W), -1.0, np.float32)
        dir_full = np.zeros((H, W), np.float32)
        t_full[pick_r] = tr[pick_r]; dir_full[pick_r] = 0.0
        t_full[pick_d] = td[pick_d]; dir_full[pick_d] = 1.0
        act = pick_r | pick_d
        cb_r = np.zeros_like(lst); cb_r[:, :-1] = lst[:, 1:]
        cb_d = np.zeros_like(lst); cb_d[:-1, :] = lst[1:, :]
        c_b = np.where(dir_full < 0.5, cb_r, cb_d)
        wmap = np.zeros((H, W), np.float32)
        if act.any():
            ai, aj = np.nonzero(act)
            wmap[ai, aj] = we_mat[lst[ai, aj], c_b[ai, aj]]
        prov = (mx.array(t_full[None]), mx.array(dir_full[None]), mx.array(wmap[None]))
        _sx_cache.clear()
        _sx_cache[pi] = prov
        return prov

    def subpix_term(m, pi):
        # LEVER-4b / P0 FORCE 3 (trainer block @ subpix_w>0): supervise the witness realized
        # margin ratio t_wit = Mw[p]/(Mw[p]+Mw[q]) toward the GT t on genuine-V straddles,
        # W_e-weighted mean (pa_flipmass armed in c2).
        t_tgt, dir_m, ew = _subpix_providers(pi)
        signed = _signed_for(m, pi)
        active = (t_tgt >= 0.0).astype(signed.dtype)
        mw = mx.maximum(signed, 0.0)
        m_right = mx.pad(mw[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
        m_down = mx.pad(mw[:, 1:, :], [(0, 0), (0, 1), (0, 0)])
        mq = mx.where(dir_m < 0.5, m_right, m_down)
        t_wit = mw / (mw + mq + args.subpix_eps)
        t_ref = mx.maximum(t_tgt, 0.0)
        sq = mx.square(t_wit - t_ref) * active
        return args.subpix_weight * (mx.sum(sq * ew) / (mx.sum(active * ew) + 1e-6))

    term_fns = {"seg": seg_term, "pose": pose_term, "phase_advect": phase_term,
                "weight_entropy": we_term, "margin_satisfice": ms_term, "subpix": subpix_term}
    pair_dependent = {"seg": True, "pose": True, "phase_advect": True, "weight_entropy": False,
                      "margin_satisfice": True, "subpix": True}

    n_eu = min(args.pairs, n_all)
    eu_idx = np.linspace(0, n_all - 1, n_eu).astype(int).tolist()

    def accumulate(term_name: str):
        fn = term_fns[term_name]
        vg = nn.value_and_grad(model, fn)
        if not pair_dependent[term_name]:
            val, g = vg(model, None)
            mx.eval(val, g)
            _, vec = _flatten_grad_tree(g)
            return float(val), vec, dict(tree_flatten(g))
        vec_sum = None
        tree_sum: dict = {}
        val_sum = 0.0
        for j, pi in enumerate(eu_idx):
            val, g = vg(model, pi)
            mx.eval(val, g)
            val_sum += float(val)
            _, v = _flatten_grad_tree(g)
            vec_sum = v if vec_sum is None else vec_sum + v
            for n, a in tree_flatten(g):
                tree_sum[n] = a if n not in tree_sum else tree_sum[n] + a
            if (j + 1) % 16 == 0:
                print(json.dumps({"stage": f"accum_{term_name}", "pair": j + 1, "n": n_eu}),
                      flush=True)
        vec = vec_sum / n_eu
        tree = {n: tree_sum[n] / float(n_eu) for n in tree_sum}
        return val_sum / n_eu, vec, tree

    print(json.dumps({"stage": "euclid_start", "term_a": args.term_a, "term_b": args.term_b,
                      "n_pairs": n_eu}), flush=True)
    val_a, vec_a, tree_a = accumulate(args.term_a)
    val_b, vec_b, tree_b = accumulate(args.term_b)

    for nm, v in (("a", vec_a), ("b", vec_b)):
        n_bad = int((~np.isfinite(v)).sum())
        if n_bad:
            raise SystemExit(f"term-{nm} gradient has {n_bad} non-finite entries (fail closed; "
                             "no verdict from a corrupted measurement)")

    eu_cos = _cos(vec_a, vec_b)
    eu_rel = float(np.linalg.norm(vec_a) / (np.linalg.norm(vec_b) + 1e-30))
    dot_ab = float(vec_a @ vec_b)
    eu_conflict_b = max(0.0, -dot_ab) / (np.linalg.norm(vec_b) + 1e-30)
    print(json.dumps({"stage": "euclid_done", "cos": eu_cos, "rel_norm_a_over_b": eu_rel,
                      "val_a": val_a, "val_b": val_b}), flush=True)

    # ---- FISHER leg: pull both directions to logit space per pair; categorical Fisher inner ----
    p_items = tree_flatten(model.trainable_parameters())
    p_names = [n for n, _ in p_items]
    primals = [a for _, a in p_items]

    def _tan(tree: dict):
        return [mx.array(tree[n]) if n in tree else mx.zeros_like(dict(p_items)[n])
                for n in p_names]

    tan_a = _tan(tree_a)
    tan_b = _tan(tree_b)
    n_fi = min(args.fisher_pairs, n_all)
    fi_idx = np.linspace(0, n_all - 1, n_fi).astype(int).tolist()
    fd_h = float(args.fd_h)

    def _seg_np(frame_mx):
        return np.asarray(adapter.segnet(frame_mx)[0], np.float64)

    def _dlogit(frame0_mx, dframe_mx):
        dn = float(mx.sqrt(mx.sum(dframe_mx * dframe_mx)))
        if dn <= 0.0:
            return np.zeros_like(_seg_np(frame0_mx))
        u = dframe_mx / dn
        return (_seg_np(frame0_mx + fd_h * u) - _seg_np(frame0_mx - fd_h * u)) / (2.0 * fd_h) * dn

    fi_num = fi_aa = fi_bb = 0.0
    for j, pi in enumerate(fi_idx):
        feats_mx = mx.array(feats_np_for_pair(pi))
        code1 = 2 * pi + 1

        def render_only(*arrs, _c1=code1, _f=feats_mx):
            model.update(tree_unflatten(list(zip(p_names, arrs, strict=True))))
            return render_through_R_mlx(model, _f, _c1, render_h, render_w)

        frs, dfr_a = mx.jvp(render_only, primals, tan_a)
        _, dfr_b = mx.jvp(render_only, primals, tan_b)
        frame0 = frs[0]
        mx.eval(frame0, dfr_a[0], dfr_b[0])
        model.update(tree_unflatten(list(zip(p_names, primals, strict=True))))
        L = _seg_np(frame0).reshape(-1, n_classes)
        A = _dlogit(frame0, dfr_a[0]).reshape(-1, n_classes)
        Bm = _dlogit(frame0, dfr_b[0]).reshape(-1, n_classes)
        m0 = L.max(axis=1, keepdims=True)
        e = np.exp(L - m0)
        P = e / e.sum(axis=1, keepdims=True)
        pa = (P * A).sum(axis=1)
        pb = (P * Bm).sum(axis=1)
        fi_num += float(((P * A * Bm).sum(axis=1) - pa * pb).sum())
        fi_aa += float(((P * A * A).sum(axis=1) - pa * pa).sum())
        fi_bb += float(((P * Bm * Bm).sum(axis=1) - pb * pb).sum())
        del feats_mx, frs, dfr_a, dfr_b, frame0
        mx.clear_cache()
        if (j + 1) % 16 == 0:
            print(json.dumps({"stage": "fisher", "pair": j + 1, "n": n_fi}), flush=True)

    fisher_cos = (fi_num / (np.sqrt(fi_aa) * np.sqrt(fi_bb) + 1e-300)
                  if fi_aa > 0 and fi_bb > 0 else float("nan"))
    fisher_rel = float(np.sqrt(fi_aa) / (np.sqrt(fi_bb) + 1e-300))
    fisher_conflict_b = max(0.0, -fi_num) / (np.sqrt(fi_bb) + 1e-300)

    result = {
        "checkpoint": str(ck), "epoch": epoch, "state_label": args.state_label,
        "term_a": args.term_a, "term_b": args.term_b,
        "n_pairs_euclid": n_eu, "n_pairs_fisher": n_fi, "n_all": n_all,
        "seg_form": args.seg_form, "tau": args.tau, "w_seg": args.w_seg,
        "pa_weight": args.pa_weight, "pa_band": args.pa_band,
        "self_orient_reconstruction": ("decode_style_fixed_point_"
                                       f"{int(args.so_fixed_point_iters)}iters"
                                       if self_orient else "n/a"),
        "term_values": {"a": val_a, "b": val_b},
        "euclid_baseline": {
            "cos_a_b": eu_cos, "rel_norm_a_over_b": eu_rel,
            "pcgrad_conflict_of_b": eu_conflict_b, "dot_a_b": dot_ab,
            "verdict": _verdict(eu_cos),
        },
        "fisher_authority": {
            "cos_a_b": float(fisher_cos), "rel_norm_a_over_b": fisher_rel,
            "pcgrad_conflict_of_b": float(fisher_conflict_b),
            "inner_a_b": fi_num, "norm2_a": fi_aa, "norm2_b": fi_bb,
            "verdict": _verdict(fisher_cos),
        },
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False, "promotable": False, "promotion_eligible": False,
        "note": "SPEC_v10 §13.4(3) dual-metric read-back; advisory n<600 subset; pointer UNMOVED",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": "RESULT", **result}), flush=True)


if __name__ == "__main__":
    main()
