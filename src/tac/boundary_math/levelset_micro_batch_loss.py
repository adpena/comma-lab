"""Micro-batch (B-pair) realized loss for the LEVELSET witness trainer (--micro-batch-pairs).

This module holds the BATCHED twin of the trainer's nested ``total_loss_fn`` so it is
IMPORTABLE + UNIT-TESTABLE (the nested closure in
``experiments/train_levelset_witness_realized_through_R_mlx.py`` cannot be reached from a
test). The trainer's ``total_loss_fn_batch`` is a thin wrapper that packs its ~30 closure
levers into a :class:`LeverConfig` and delegates here.

Design contract (see the trainer's ``total_loss_fn_batch`` docstring for the full rationale):

* The ONLY batched operations are the EXPENSIVE realized render + FROZEN-SCORER forwards
  (one SegNet over the B ``f1`` frames, one PoseNet over the B pairs). EfficientNet-B2 /
  FastViT are batch-independent in EVAL mode (running-stat BN, per-pixel conv, per-frame
  head) so ``segnet(f1_batch)[k] == segnet(f1_batch[k:k+1])[0]`` up to fp reduction.
* EVERY per-pair loss reduction — base seg-form, the score-domain pose ``sqrt(10*d_pose)``
  (NONLINEAR: ``sqrt(mean) != mean(sqrt)``), and every weighted-mean lever
  ``sum(x*w)/sum(w)`` — is computed PER PAIR on the batched scorer outputs and MEAN-ed over
  B. Hence ``batched_realized_loss(B pairs) == mean_b single_realized_loss(pair_b)`` within
  fp tolerance (the mean-over-B is the only reduction re-order). This EXACT per-pair-mean
  identity is what the numerical-equivalence unit test pins.
* The realized ``segnet(f1)`` forward is computed ONCE and shared by the base seg-form AND
  the lever ``_signed`` (bit-identical to the trainer's two deterministic-render forwards:
  ``(f'+g')·dS == f'·dS + g'·dS``).
* Per-MODEL code penalties (rankfloor / code-spec / code-nuc) are added ONCE (matching the
  serial mean-over-chunk of an identical-per-pair term).

The math here MIRRORS the trainer's ``total_loss_fn`` op-for-op. ``single_realized_loss`` is
the per-pair reference (renders + scores ONE pair, no batching) used by the test as the
"mean of per-pair grads" baseline; the trainer never calls it (it uses its own untouched
``total_loss_fn`` for the B=1 path).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class LeverConfig:
    """All per-run loss config the batched/per-pair loss needs (mirrors the trainer closures).

    Scalar seg-form + score-domain config, the lever weights/gates/tensors, and the two
    trainer-local helper callables that are NOT importable without a cycle
    (``eikonal_length`` = ``_eikonal_length_mlx``; ``nuclear_norm_smooth`` =
    ``_nuclear_norm_smooth_mlx``). Everything else is imported directly.
    """

    # ── base seg-form + score-domain (from make_loss_fn's config) ──
    seg_loss_default: str = "ce"
    tau_use: float = 0.3
    l7_thr_use: float = 0.42
    l7_mult: float = 4.0
    score_domain: bool = True
    pose_eps: float = 1e-2
    # ── (--seg-focal-gamma) focal per-pixel seg reweight (1-p_y)^gamma, mean-1 stop-grad ──
    # Routed leg (was fail-closed pre-#313). Folds into every seg-form's per-pixel map BEFORE the
    # mean, EXACTLY as make_loss_fn does (``seg_pixel_w``); passed as a callable to stay bit-identical
    # to the canonical ``focal_pixel_weight_mlx`` (one math, one backend). Default 0.0 => never built.
    focal_gamma: float = 0.0
    focal_pixel_weight: Callable | None = None   # focal_pixel_weight_mlx(seg_logits, oh, gamma)
    # ── (--boundary-distance-weight) SDF-native boundary-placement term on frame1 ──
    # Routed leg (was fail-closed pre-#313). Per-pair; one extra sdf(cf,c1) forward when ON.
    bd_w: float = 0.0
    bd_band_prov: Any = None                     # dict[pair -> (1,H,W) GT-boundary band map]
    boundary_distance_term: Callable | None = None  # boundary_distance_term_mlx(phi_flat, oh, band, rh, rw)
    # ── eikonal/length junction relax (default OFF = bit-identical) ──
    eik_jrelax: float = 0.0
    eik_jtau: float = 0.5
    eikonal_length: Callable | None = None       # _eikonal_length_mlx(phi_pk, rh, rw, ...)
    nuclear_norm_smooth: Callable | None = None  # _nuclear_norm_smooth_mlx(code, ...)
    # ── (EIK-STAB) ViscoReg viscosity residual (REPLACES eik while eps>0) + StEik damping (ADD) ──
    # Routed leg (was fail-closed pre-#313). ``eik_stab`` is the trainer's ``_eik_stab`` dict passed
    # BY REFERENCE (mutated in-place by the per-epoch viscosity anneal, so the batched twin sees the
    # live eps exactly like total_loss_fn re-reads it). Both helpers passed as callables (module-level
    # trainer fns; imported here would be a cycle). Default (None / 0.0) => legacy residual unchanged.
    eik_stab: dict | None = None                 # {"visco_eps": float, "steik_w": float} BY REF
    eikonal_visco: Callable | None = None        # _eikonal_visco_mlx(phi, rh, rw, visco_eps)
    eikonal_steik: Callable | None = None        # _eikonal_steik_mlx(phi, rh, rw)
    # ── (--witness-alone-island-loss #300a) island levers read the seed-EXCLUDED margin ──
    # Routed leg (was fail-closed pre-#313; THE binding blocker for the live config). When True AND a
    # ``render_fn_wa`` is supplied to the batched/single loss AND >=1 island lever (amplify/persist) is
    # engaged, the island levers read a SECOND (seed-excluded) SegNet forward's margin/logits; base +
    # non-island levers keep the seed-composed forward. Default False => island levers alias the
    # composed forward => byte-identical (no 2nd forward).
    wa_island: bool = False
    # ── LEVER-3 lane-edge ──
    lane_w: float = 0.0
    lane_gate: dict = field(default_factory=lambda: {"on": True})
    lane_cls: int = 1
    lane_tgt: float = 0.5
    # ── LEVER-4 margin-saliency (+UNIWARD) ──
    msal_w: float = 0.0
    msal_gate: dict = field(default_factory=lambda: {"on": True})
    msal_tau: float = 0.3
    msal_tgt: float = 0.5
    msal_uni: bool = False
    msal_uni_beta: float = 1.0
    # ── #224(5) island amplify ──
    amplify_w: float = 0.0
    island_weight_mx: Any = None    # dict[pair -> (H,W) mx weight]
    amplify_mtgt: float = 1.0
    amplify_form: str = "hinge"
    # ── #224(4) persistence/topology ──
    persist_gate: dict = field(default_factory=lambda: {"w": 0.0})
    persist_classes: tuple = ()
    persist_cldice_iters: int = 5
    persist_recall_w: float = 1.0
    # ── LEVER-B thin-lane ──
    lane_thin_w: float = 0.0
    lane_thin_gate: dict = field(default_factory=lambda: {"on": True})
    thin_maps_mx: Any = None        # dict[pair -> (1,H,W) mx weight]
    lane_thin_tgt: float = 0.5
    # ── #218 margin-field head ──
    mfh_w: float = 0.0
    mfh_target_mx: Any = None       # (1,1,1,5) mx per-class margin target
    # ── LEVER-A rankfloor (per-MODEL, once) ──
    rankfloor_w: float = 0.0
    rankfloor_idx: Any = None
    rankfloor_tgt: float = 4.0
    # ── DM1b code spectral-entropy (per-MODEL, once) ──
    code_spec_w: float = 0.0
    code_spec_idx: Any = None
    # ── code nuclear-norm (per-MODEL, once) ──
    code_nuc_w: float = 0.0
    code_nuc_eps: float = 1e-3
    code_nuc_iters: int = 25


def _pair_loss_from_scored(model, sl, pose_row, cf, c0: int, c1: int, oh, mg, pose_tgt,
                           f1_frame, render_h: int, render_w: int,
                           w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w,
                           lc: LeverConfig, sl_wa=None):
    """ONE pair's loss (base seg-form + score-domain pose + eikonal/length + all seg levers)
    from a PRE-SCORED SegNet logit slice ``sl`` (1,H,W,5) and PoseNet output ``pose_row``
    (half,). ``f1_frame`` (1,H,W,3) is the realized frame1 (only used by the UNIWARD texture
    map). ``sl_wa`` (1,H,W,5) is the WITNESS-ALONE (seed-EXCLUDED) SegNet logit slice the
    island-formation levers (amplify/persistence) read under --witness-alone-island-loss; None
    => aliases ``sl`` (byte-identical when wa routing is off). NO once-per-step per-MODEL
    penalties (added once by the caller). Mirrors the trainer's ``total_loss_fn`` per-pair body
    op-for-op.
    """
    import mlx.core as mx

    from tac.boundary_math.island_protection import island_birth_from_signed_mx
    from tac.boundary_math.persistence_topology_loss import persistence_topology_loss_mlx

    # SHARED realized decision margin (base tau/l7/margin_hinge + surgical seg levers) on the
    # seed-COMPOSED forward.
    _sig_gt = mx.sum(sl * oh, axis=-1)
    _sig_run = mx.max(sl + oh * (-1e9), axis=-1)
    _signed = _sig_gt - _sig_run                                  # (1,H,W)
    # WITNESS-ALONE margin (#300a): island levers read the seed-EXCLUDED forward when routed. When
    # sl_wa is None (wa off) it aliases sl => _signed_wa IS _signed's math on the SAME logits =>
    # byte-identical. When routed, sl_wa is a distinct (seed-excluded) forward.
    if sl_wa is None:
        sl_wa = sl
        _signed_wa = _signed
    else:
        _signed_wa = mx.sum(sl_wa * oh, axis=-1) - mx.max(sl_wa + oh * (-1e9), axis=-1)
    # (--seg-focal-gamma) focal per-pixel reweight, folded into EVERY seg form BEFORE the mean,
    # exactly as make_loss_fn does (multiplicative, mean-1 stop-grad). Default 0.0 => None => the
    # per-pixel maps are untouched (byte-identical). Uses the SAME callable as the canonical loss.
    _seg_px_w = None
    if float(lc.focal_gamma) > 0.0 and lc.focal_pixel_weight is not None:
        _seg_px_w = lc.focal_pixel_weight(sl, oh, float(lc.focal_gamma))
    form = seg_form if seg_form is not None else lc.seg_loss_default
    tau_use = float(lc.tau_use)
    # ----- base seg-form (mirror make_loss_fn; margin-weight OFF, static tau/l7) -----
    if form == "tau_softplus":
        z = -_signed / tau_use
        _pp = tau_use * mx.logaddexp(mx.zeros_like(z), z)
        seg_l = mx.mean(_pp if _seg_px_w is None else _pp * _seg_px_w)
    elif form == "l7_softplus":
        z = -_signed / tau_use
        per_pixel = tau_use * mx.logaddexp(mx.zeros_like(z), z)
        w = 1.0 + float(lc.l7_mult) * (_signed < float(lc.l7_thr_use)).astype(sl.dtype)
        w = mx.stop_gradient(w / (mx.mean(w) + 1e-8))
        _pp = per_pixel * w
        seg_l = mx.mean(_pp if _seg_px_w is None else _pp * _seg_px_w)
    elif form == "margin_hinge":
        _pp = mx.maximum(mtgt - _signed, 0.0)
        seg_l = mx.mean(_pp if _seg_px_w is None else _pp * _seg_px_w)
    else:  # ce
        logsum = mx.logsumexp(sl, axis=-1)
        tgt = mx.sum(sl * oh, axis=-1)
        ce = logsum - tgt
        w = 1.0 + hinge * mx.exp(-mx.clip(mg, 0.0, 1e9))
        _pp = ce * w[None]
        seg_l = mx.mean(_pp if _seg_px_w is None else _pp * _seg_px_w)
    # ----- base pose: per-pair sqrt matches the serial mean-over-chunk (sqrt(mean)!=mean(sqrt)) -----
    pose_l = mx.mean(mx.square(pose_row - pose_tgt))
    pose_term = mx.sqrt(10.0 * pose_l + float(lc.pose_eps)) if lc.score_domain else pose_l
    Lk = w_seg * seg_l + w_pose * pose_term
    # ----- eikonal + length (per-pair; on model.sdf(cf, c0)) -----
    phi0 = model.sdf(cf, c0)
    eik, length, _ = lc.eikonal_length(phi0, render_h, render_w,
                                       junction_relax=float(lc.eik_jrelax), junction_tau=float(lc.eik_jtau))
    # (EIK-STAB) ViscoReg viscous residual REPLACES the eikonal residual while eps>0 (same
    # constraint, viscous form — adding both double-counts). eps read LIVE from the by-ref dict
    # (per-epoch anneal). Default (None / eps<=0) => the legacy residual is used (byte-identical).
    if lc.eik_stab is not None and float(lc.eik_stab.get("visco_eps", 0.0)) > 0.0 and lc.eikonal_visco is not None:
        eik = lc.eikonal_visco(phi0, render_h, render_w, float(lc.eik_stab["visco_eps"]))
    Lk = Lk + eik_w * eik + len_w * length
    # (EIK-STAB) StEik directional-divergence damping (ADDITIVE). Default (None / w<=0) => skipped.
    if lc.eik_stab is not None and float(lc.eik_stab.get("steik_w", 0.0)) > 0.0 and lc.eikonal_steik is not None:
        Lk = Lk + float(lc.eik_stab["steik_w"]) * lc.eikonal_steik(phi0, render_h, render_w)
    # (--boundary-distance-weight) SDF-native boundary placement on frame1 (one extra sdf forward).
    # Default (bd_w<=0 / no provider) => skipped => byte-identical.
    if float(lc.bd_w) > 0.0 and lc.bd_band_prov is not None and lc.boundary_distance_term is not None:
        Lk = Lk + float(lc.bd_w) * lc.boundary_distance_term(
            model.sdf(cf, c1), oh, lc.bd_band_prov[c1 // 2], render_h, render_w)
    # ----- seg levers (mirror total_loss_fn per-pair; all default-off) -----
    if lc.lane_w > 0.0 and lc.lane_gate["on"]:
        lane_mask = oh[..., lc.lane_cls]
        hinge_map = mx.maximum(lc.lane_tgt - _signed, 0.0) * lane_mask
        Lk = Lk + lc.lane_w * (mx.sum(hinge_map) / (mx.sum(lane_mask) + 1e-6))
    if lc.msal_w > 0.0 and lc.msal_gate["on"]:
        sal = mx.exp(-mg / lc.msal_tau)
        if lc.msal_uni:
            lum = mx.mean(mx.stop_gradient(f1_frame), axis=-1)
            dy = mx.pad(mx.abs(lum[:, 1:, :] - lum[:, :-1, :]), [(0, 0), (0, 1), (0, 0)])
            dx = mx.pad(mx.abs(lum[:, :, 1:] - lum[:, :, :-1]), [(0, 0), (0, 0), (0, 1)])
            tex = dy + dx
            tex = tex / (mx.max(tex) + 1e-6)
            sal = sal / (1.0 + lc.msal_uni_beta * tex)
        hmap = mx.maximum(lc.msal_tgt - _signed, 0.0) * sal
        Lk = Lk + lc.msal_w * (mx.sum(hmap) / (mx.sum(sal) + 1e-6))
    # #224(5) island amplify + #224(4) persistence/topology read the WITNESS-ALONE margin/logits
    # (_signed_wa / sl_wa) under --witness-alone-island-loss (#300a); they alias _signed / sl when
    # wa routing is off => byte-identical.
    if lc.amplify_w > 0.0 and lc.island_weight_mx is not None:
        Lk = Lk + lc.amplify_w * island_birth_from_signed_mx(
            _signed_wa, lc.island_weight_mx[c1 // 2], lc.amplify_mtgt, form=lc.amplify_form)
    if lc.persist_gate["w"] > 0.0 and lc.persist_classes:
        Lk = Lk + lc.persist_gate["w"] * persistence_topology_loss_mlx(
            sl_wa, oh, lc.persist_classes, cldice_iters=lc.persist_cldice_iters,
            w_cldice=1.0, w_recall=lc.persist_recall_w)
    if lc.lane_thin_w > 0.0 and lc.lane_thin_gate["on"] and lc.thin_maps_mx is not None:
        tw = lc.thin_maps_mx[c0 // 2]
        hmap_t = mx.maximum(lc.lane_thin_tgt - _signed, 0.0) * tw
        Lk = Lk + lc.lane_thin_w * (mx.sum(hmap_t) / (mx.sum(tw) + 1e-6))
    if lc.mfh_w > 0.0 and lc.mfh_target_mx is not None:
        per_pix_tgt = mx.sum(oh * lc.mfh_target_mx, axis=-1)
        Lk = Lk + lc.mfh_w * mx.mean(mx.maximum(per_pix_tgt - _signed, 0.0))
    return Lk


def _once_terms(model, lc: LeverConfig):
    """Per-MODEL code-matrix penalties (rankfloor / code-spec / code-nuc). Returns a scalar mx
    array (0.0 when all off). Added ONCE per step (matches the serial mean-over-chunk)."""
    import mlx.core as mx

    L = mx.zeros(())
    if lc.rankfloor_w > 0.0 and lc.rankfloor_idx is not None:
        M = model.film(model.code[lc.rankfloor_idx])
        Mc = M - mx.mean(M, axis=0, keepdims=True)
        tr = mx.sum(Mc * Mc)
        G = Mc @ Mc.T
        fro2 = mx.sum(G * G)
        pr = (tr * tr) / (fro2 + 1e-12)
        L = L + lc.rankfloor_w * mx.maximum(lc.rankfloor_tgt - pr, 0.0)
    if lc.code_spec_w > 0.0 and lc.code_spec_idx is not None:
        Cm = model.code[lc.code_spec_idx]
        Cc = Cm - mx.mean(Cm, axis=0, keepdims=True)
        Cov = Cc.T @ Cc
        ctr = mx.sum(Cc * Cc)
        cfro2 = mx.sum(Cov * Cov)
        cpr = (ctr * ctr) / (cfro2 + 1e-12)
        L = L - lc.code_spec_w * mx.log(cpr + 1e-12)
    if lc.code_nuc_w > 0.0:
        L = L + lc.code_nuc_w * lc.nuclear_norm_smooth(
            model.code, rel_eps=float(lc.code_nuc_eps), ns_iters=int(lc.code_nuc_iters))
    return L


def _island_levers_on(lc: LeverConfig) -> bool:
    """The #224 island-formation levers (amplify OR persistence) engaged this step. Mirrors the
    trainer's ``_island_levers_on`` (gate dicts read live)."""
    return bool((lc.amplify_w > 0.0 and lc.island_weight_mx is not None) or
                (lc.persist_gate["w"] > 0.0 and bool(lc.persist_classes)))


def _wa_route_active(lc: LeverConfig, render_fn_wa) -> bool:
    """--witness-alone-island-loss routing active for THIS step: the flag is set, a seed-excluded
    render is supplied, and >=1 island lever is engaged. Mirrors the trainer's ``_wa_route``."""
    return bool(lc.wa_island) and (render_fn_wa is not None) and _island_levers_on(lc)


def batched_realized_loss(model, adapter, render_fn, render_h: int, render_w: int,
                          cf_list, c0_list, c1_list, oh_list, mg_list, pose_tgt_list,
                          w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc: LeverConfig,
                          render_fn_wa=None):
    """BATCHED (K-pair) realized loss = mean_k(per-pair loss) + once-terms. THE speed win:
    ONE segnet forward over the K f1 frames + ONE posenet forward over the K pairs.

    ``render_fn(model, coord_feats, code_idx, render_h, render_w) -> (1,H,W,3)`` is the
    effective per-frame render (witness / residual-compose / AA / pose-carrier), rendered
    per-frame (preserves per-pair self-orient feats + all compose hooks) then stacked.

    ``render_fn_wa`` (--witness-alone-island-loss #300a): the SEED-EXCLUDED per-frame render. When
    given AND ``lc.wa_island`` AND >=1 island lever is engaged, a SECOND batched SegNet forward over
    the K witness-alone f1 frames feeds the island levers' margin/logits; base + non-island levers
    keep the seed-composed forward. None (default) => island levers alias the composed forward =>
    byte-identical (no 2nd forward). Batch-independence of the frozen SegNet (eval-mode BN/conv)
    makes the batched wa forward equal to the K per-pair wa forwards, so the mean-of-per-pair
    equivalence contract holds for this leg exactly as for the composed forward.
    """
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    K = len(c1_list)
    f1_b = mx.concatenate([render_fn(model, cf_list[k], int(c1_list[k]), render_h, render_w)
                           for k in range(K)], axis=0)                 # (K,H,W,3)
    f0_b = mx.concatenate([render_fn(model, cf_list[k], int(c0_list[k]), render_h, render_w)
                           for k in range(K)], axis=0)                 # (K,H,W,3)
    seg_logits_b = adapter.segnet(f1_b)                               # (K,H,W,5) shared base+levers
    # (--witness-alone-island-loss #300a) SECOND (seed-excluded) SegNet forward for the island
    # levers, ONLY when routed. Only frame1 is needed (island levers read the SegNet-scored margin).
    seg_logits_wa_b = None
    if _wa_route_active(lc, render_fn_wa):
        f1_wa_b = mx.concatenate([render_fn_wa(model, cf_list[k], int(c1_list[k]), render_h, render_w)
                                  for k in range(K)], axis=0)          # (K,H,W,3) seed-EXCLUDED
        seg_logits_wa_b = adapter.segnet(f1_wa_b)                      # (K,H,W,5)
    pair = mx.stack([f0_b, f1_b], axis=1)                             # (K,2,H,W,3)
    yuv = rgb_to_yuv6_mlx(pair)                                       # (K,2,h2,w2,6)
    _k, _t, _h2, _w2, _c6 = yuv.shape
    yuv_nhwc = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (_k, _h2, _w2, _t * _c6))
    _half = pose_tgt_list[0].shape[-1]
    pose_b = adapter.posenet(yuv_nhwc)["pose"][..., :_half]           # (K, half)

    def _sl_wa(k):
        return None if seg_logits_wa_b is None else seg_logits_wa_b[k:k + 1]

    L = _pair_loss_from_scored(
        model, seg_logits_b[0:1], pose_b[0], cf_list[0], int(c0_list[0]), int(c1_list[0]),
        oh_list[0], mg_list[0], pose_tgt_list[0], f1_b[0:1], render_h, render_w,
        w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, sl_wa=_sl_wa(0))
    for k in range(1, K):
        L = L + _pair_loss_from_scored(
            model, seg_logits_b[k:k + 1], pose_b[k], cf_list[k], int(c0_list[k]), int(c1_list[k]),
            oh_list[k], mg_list[k], pose_tgt_list[k], f1_b[k:k + 1], render_h, render_w,
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, sl_wa=_sl_wa(k))
    L = L / float(K)
    L = L + _once_terms(model, lc)
    return L


def single_realized_loss(model, adapter, render_fn, render_h: int, render_w: int,
                         cf, c0: int, c1: int, oh, mg, pose_tgt,
                         w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc: LeverConfig,
                         render_fn_wa=None):
    """PER-PAIR reference (renders + scores ONE pair with a single scorer forward). Used by
    the numerical-equivalence test as the "mean of per-pair grads" baseline. Includes the
    once-terms (so mean_k single == batched: the once-term is identical in each pair -> its
    mean is applied once, exactly as batched adds it once). ``render_fn_wa`` routes the island
    levers through a second (seed-excluded) SegNet forward, symmetric to ``batched_realized_loss``.
    """
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    f1 = render_fn(model, cf, int(c1), render_h, render_w)            # (1,H,W,3)
    f0 = render_fn(model, cf, int(c0), render_h, render_w)            # (1,H,W,3)
    sl = adapter.segnet(f1)                                           # (1,H,W,5)
    sl_wa = None
    if _wa_route_active(lc, render_fn_wa):
        f1_wa = render_fn_wa(model, cf, int(c1), render_h, render_w)  # (1,H,W,3) seed-EXCLUDED
        sl_wa = adapter.segnet(f1_wa)                                 # (1,H,W,5)
    pair = mx.stack([f0[0], f1[0]], axis=0)[None]                    # (1,2,H,W,3)
    yuv = rgb_to_yuv6_mlx(pair)
    _b, _t, _h2, _w2, _c6 = yuv.shape
    yuv_nhwc = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (_b, _h2, _w2, _t * _c6))
    half = pose_tgt.shape[-1]
    pose_row = adapter.posenet(yuv_nhwc)["pose"][0, :half]           # (half,)
    L = _pair_loss_from_scored(model, sl, pose_row, cf, int(c0), int(c1), oh, mg, pose_tgt,
                               f1, render_h, render_w,
                               w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, sl_wa=sl_wa)
    L = L + _once_terms(model, lc)
    return L
