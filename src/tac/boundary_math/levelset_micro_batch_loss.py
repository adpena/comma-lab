"""Micro-batch (B-pair) realized loss for the LEVELSET witness trainer (--micro-batch-pairs).

This module holds the BATCHED twin of the trainer's nested ``total_loss_fn`` so it is
IMPORTABLE + UNIT-TESTABLE (the nested closure in
``experiments/train_levelset_witness_realized_through_R_mlx.py`` cannot be reached from a
test). The trainer's ``total_loss_fn_batch`` is a thin wrapper that packs its ~30 closure
levers into a :class:`LeverConfig` and delegates here.

Design contract (see the trainer's ``total_loss_fn_batch`` docstring for the full rationale):

* The ONLY batched operations are the EXPENSIVE realized render + FROZEN-SCORER forwards
  (one SegNet over the B ``f1`` frames, one PoseNet over the B pairs). Frozen does NOT imply
  numerically batch-independent: the measured scorer-batch-dependence law permits training-time
  drift under the 2026-07-12 throughput override. Admission is functional loss/gradient parity
  within its measured tolerance, never bit identity and never score authority.
* EVERY per-pair loss reduction — base seg-form, the score-domain pose ``sqrt(10*d_pose)``
  (NONLINEAR: ``sqrt(mean) != mean(sqrt)``), and every weighted-mean lever
  ``sum(x*w)/sum(w)`` — is computed PER PAIR on the batched scorer outputs and MEAN-ed over
  B. This pins the same mathematical reduction while measured scorer/fusion drift is admitted
  only within the functional-parity tolerance.
* The realized ``segnet(f1)`` forward is computed ONCE and shared by the base seg-form AND
  the lever ``_signed``; this is a graph/throughput identity, not a bit-identity claim.
* Per-MODEL penalties (rankfloor / code-spec / code-nuc / weight-entropy rate) are added ONCE
  (matching the serial mean-over-chunk of an identical-per-pair term).
* (#D15) --logit-adjust-loss-tau routes as a per-class constant offset added to the BASE seg-form
  logits ONLY (``sl_base = seg_logits + offset``); the surgical seg levers + the witness-alone
  forward keep the RAW logits — EXACTLY the serial split (base_loss reads the wrapped
  ``_LogitAdjustSegAdapter``, ``total_loss_fn``'s levers read the raw adapter). The add is
  row-/pixel-local and introduces no further cross-pair reduction, though its input logits can
  already carry scorer-batch drift. --seg-form-unify-tau routes as the
  ``unify_tau`` branch (the ONE continuous ``L_τ = τ·logsumexp(φ/τ) − φ_y``) reading the LIVE
  render-coupled τ from the by-ref ``unify_tau_state``; the class-axis logsumexp is per-pixel
  row-local, preserving the functional per-pair reduction contract.

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
    # ── (--logit-adjust-loss-tau, #D15 routing) Menon per-class logit-adjustment offset ──
    # (5,) mx.array = tau*log(prior_c), or None (OFF => byte-identical). Added ADDITIVELY to the BASE
    # seg-form logits ONLY (sl_base = seg_logits + offset), NEVER to the surgical seg levers' raw
    # logits (_signed) NOR the witness-alone forward — EXACTLY mirroring the serial split (base_loss
    # reads the WRAPPED _LogitAdjustSegAdapter; total_loss_fn's levers + wa forward read the RAW
    # adapter). The add is a per-class constant broadcast over (K,H,W,5) and introduces no new
    # cross-pair reduction; the surrounding scorer can still drift with batch shape. Default None =>
    # sl_base aliases sl => byte-identical (mirrors ``tau==0 => _loss_adapter is adapter``).
    logit_adjust_offset: Any = None
    # Live birth-completion cell. When present, ``state["offset"]`` is re-read on every loss call
    # and takes precedence over the construction-time compatibility value above.
    logit_adjust_state: dict | None = None
    # ── (--seg-form-unify-tau, #D15 routing) the ONE continuous L_τ seg form ──
    # unify_tau_state = the trainer's by-ref {"tau": float|None} refreshed per-epoch (the SAME
    # closure-cell idiom as eik_stab): the batched twin reads the LIVE render-coupled τ when
    # seg_form == "unify_tau" (else falls back to lc.tau_use, mirroring make_loss_fn's
    # ``tau_use = tau_softplus_tau if tau_override is None``). seg_unify_tau_perpixel = the trainer's
    # ``_seg_unify_tau_perpixel`` callable (passed to avoid a tac<-trainer import cycle; one math,
    # one backend — bit-identical to make_loss_fn's unify branch). Both None => unify never routed
    # (the twin refuses seg_form == "unify_tau" without the callable rather than silently CE-ing).
    unify_tau_state: dict | None = None
    seg_unify_tau_perpixel: Callable | None = None
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
    # v7.5 birth-completion live state + disjoint per-class masks. State keys mirror the trainer:
    # amp_active(bool), amp_lane(float), amp_mov(float), persist_scale(sequence|None).
    amplify_ramp_state: dict | None = None
    amplify_lane_masks: Any = None
    amplify_movable_masks: Any = None
    # ── #224(4) persistence/topology ──
    persist_gate: dict = field(default_factory=lambda: {"w": 0.0})
    persist_classes: tuple = ()
    persist_cldice_iters: int = 5
    persist_recall_w: float = 1.0
    persistence_sg_cache: Any = None
    # ── canonical V9 realized levers ──
    # Providers are list-like or dict-like by pair index pi=c1//2 and retain their serial shapes.
    chroma_w: float = 0.0
    chroma_gate: dict = field(default_factory=lambda: {"on": True})
    chroma_gt_prov: Any = None       # pi -> (1,H,W,3) target chroma
    chroma_ann_prov: Any = None      # pi -> (1,H,W) GT annulus
    phase_w: float = 0.0
    phase_gate: dict = field(default_factory=lambda: {"on": True})
    phase_ref_prov: Any = None       # pi -> (1,H,W)
    phase_dir_prov: Any = None       # pi -> (1,H,W), 0 right / 1 down
    phase_weight_prov: Any = None    # pi -> (1,H,W)
    phase_eps: float = 1e-6
    temporal_w: float = 0.0
    temporal_gate: dict = field(default_factory=lambda: {"on": True})
    temporal_ann_prov: Any = None    # pi -> (1,H,W)
    temporal_xi_prov: Any = None     # pi -> (6,), required for ground_gt
    temporal_xi_source: str = "ground_gt"  # ground_gt | carrier_live
    temporal_geom_mlx: Any = None
    temporal_class_mask: Any = None  # (3,)
    # Serial Force-1 scores the witness's OWN raw frame0, not the outer pose-carrier render dispatch.
    # Signature matches render_fn(model, cf, code_idx, H, W). None aliases the general f0 render for
    # backward compatibility, but V9+pose-carrier must wire the trainer's raw `_render_R` surface.
    temporal_render_f0_fn: Callable | None = None
    temporal_rot_mask: Any = None    # optional (6,), zero translation
    temporal_sky_rowmask: Any = None # optional (H,W,1)
    area_lambda: dict | None = None  # {class_index: derived lambda_c}
    use_metal_v9_levers: bool = True
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
    # ── weight-entropy rate penalty (per-MODEL, once; the Ballé rate-in-the-loss MLX port) ──
    # λ·rate_term of the COUNTED witness weights under the deterministic soft-histogram
    # surrogate (tac.boundary_math.weight_entropy_penalty_mlx). Default 0.0 => branch never
    # taken => byte-identical (mirrors the serial total_loss_fn guard exactly).
    we_lambda: float = 0.0
    we_sigma: float = 0.2


def _provider_at(provider, pair_index: int, name: str):
    """Fetch one required provider row with a lever-specific fail-closed error."""
    if provider is None:
        raise ValueError(f"micro-batch V9 lever active but {name} is not wired")
    try:
        return provider[int(pair_index)]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"micro-batch V9 lever active but {name} has no pair {int(pair_index)}"
        ) from exc


def _gate_on(gate: dict | None) -> bool:
    return gate is None or bool(gate.get("on", False))


def _live_logit_offset(lc: LeverConfig):
    if lc.logit_adjust_state is not None:
        if "offset" not in lc.logit_adjust_state:
            raise ValueError("LeverConfig.logit_adjust_state must contain live key 'offset'")
        return lc.logit_adjust_state["offset"]
    return lc.logit_adjust_offset


def _pair_loss_from_scored(model, sl, pose_row, cf, c0: int, c1: int, oh, mg, pose_tgt,
                           f1_frame, render_h: int, render_w: int,
                           w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w,
                           lc: LeverConfig, sl_wa=None, sl_base=None):
    """ONE pair's loss (base seg-form + score-domain pose + eikonal/length + all seg levers)
    from a PRE-SCORED SegNet logit slice ``sl`` (1,H,W,5) and PoseNet output ``pose_row``
    (half,). ``f1_frame`` (1,H,W,3) is the realized frame1 (only used by the UNIWARD texture
    map). ``sl_wa`` (1,H,W,5) is the WITNESS-ALONE (seed-EXCLUDED) SegNet logit slice the
    island-formation levers (amplify/persistence) read under --witness-alone-island-loss; None
    => aliases ``sl`` (byte-identical when wa routing is off). ``sl_base`` (1,H,W,5) is the
    class-prior-ADJUSTED logit slice the BASE seg-form reads under --logit-adjust-loss-tau (#D15);
    None => aliases ``sl`` (byte-identical when logit-adjust is off) — EXACTLY the serial split
    (base_loss reads the wrapped _LogitAdjustSegAdapter; the surgical levers + wa forward read the
    RAW adapter). NO once-per-step per-MODEL penalties (added once by the caller). Mirrors the
    trainer's ``total_loss_fn`` per-pair body op-for-op.
    """
    import mlx.core as mx

    from tac.boundary_math.island_protection import (
        island_birth_from_signed_mx,
        island_birth_perclass_from_signed_mx,
    )
    from tac.boundary_math.persistence_topology_loss import persistence_topology_loss_mlx

    # SHARED realized decision margin (surgical seg levers) on the seed-COMPOSED forward — the RAW
    # (un-adjusted) logits, matching the serial total_loss_fn's ``_slog = adapter.segnet`` levers.
    _sig_gt = mx.sum(sl * oh, axis=-1)
    _sig_run = mx.max(sl + oh * (-1e9), axis=-1)
    _signed = _sig_gt - _sig_run                                  # (1,H,W)
    # (#D15 logit-adjust routing) the BASE seg-form (+ its focal reweight) reads the ADJUSTED logits
    # sl_base = sl + offset while the surgical levers keep the RAW sl/_signed — EXACTLY the serial
    # split (base_loss uses the wrapped _LogitAdjustSegAdapter; total_loss_fn's levers use the raw
    # adapter). sl_base is None (offset OFF) => aliases sl => _signed_base IS _signed (SAME object)
    # => aliases the raw scored row. The offset add is row-/pixel-local and introduces no additional
    # cross-pair reduction; its input can already carry scorer-batch drift, so admission is functional
    # parity within tolerance only and confers no score authority.
    if sl_base is None:
        sl_base = sl
        _signed_base = _signed
    else:
        _signed_base = mx.sum(sl_base * oh, axis=-1) - mx.max(sl_base + oh * (-1e9), axis=-1)
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
        # focal reads the SAME (ADJUSTED) logits the base form does — the serial make_loss_fn's
        # focal reads ``adapter``, which is the wrapped _LogitAdjustSegAdapter when logit-adjust is
        # on (the standard logit-adjusted-focal composition). sl_base aliases sl when offset OFF.
        _seg_px_w = lc.focal_pixel_weight(sl_base, oh, float(lc.focal_gamma))
    form = seg_form if seg_form is not None else lc.seg_loss_default
    tau_use = float(lc.tau_use)
    # ----- base seg-form (mirror make_loss_fn; margin-weight OFF, static tau/l7). Reads the ADJUSTED
    # logits (sl_base / _signed_base) so --logit-adjust-loss-tau routes into the batched twin (#D15). -
    if form == "tau_softplus":
        z = -_signed_base / tau_use
        _pp = tau_use * mx.logaddexp(mx.zeros_like(z), z)
        seg_l = mx.mean(_pp if _seg_px_w is None else _pp * _seg_px_w)
    elif form == "l7_softplus":
        z = -_signed_base / tau_use
        per_pixel = tau_use * mx.logaddexp(mx.zeros_like(z), z)
        w = 1.0 + float(lc.l7_mult) * (_signed_base < float(lc.l7_thr_use)).astype(sl_base.dtype)
        w = mx.stop_gradient(w / (mx.mean(w) + 1e-8))
        _pp = per_pixel * w
        seg_l = mx.mean(_pp if _seg_px_w is None else _pp * _seg_px_w)
    elif form == "margin_hinge":
        _pp = mx.maximum(mtgt - _signed_base, 0.0)
        seg_l = mx.mean(_pp if _seg_px_w is None else _pp * _seg_px_w)
    elif form == "unify_tau":
        # (--seg-form-unify-tau, #D15 routing) the ONE continuous L_τ = τ·logsumexp(φ/τ) − φ_y with
        # the SAME hinge/margin/pixel weight the ce branch uses (τ=1 ≡ ce), mirroring make_loss_fn's
        # unify_tau branch op-for-op. τ is the LIVE render-coupled temp from the by-ref
        # unify_tau_state (else lc.tau_use). The class-axis logsumexp is PER-PIXEL/row-local, so it
        # preserves the functional per-pair reduction. NO-FAKE: refuse (not silently
        # CE) when the callable was not wired.
        if lc.seg_unify_tau_perpixel is None:
            raise ValueError(
                "micro-batch twin: seg_form == 'unify_tau' but LeverConfig.seg_unify_tau_perpixel is "
                "None (the trainer must pass _seg_unify_tau_perpixel); refusing to silently fall back "
                "to CE.")
        _uni_tau = tau_use
        if lc.unify_tau_state is not None and lc.unify_tau_state.get("tau") is not None:
            _uni_tau = float(lc.unify_tau_state["tau"])
        lt = lc.seg_unify_tau_perpixel(sl_base, oh, _uni_tau)
        w = 1.0 + hinge * mx.exp(-mx.clip(mg, 0.0, 1e9))
        _pp = lt * w[None]
        seg_l = mx.mean(_pp if _seg_px_w is None else _pp * _seg_px_w)
    else:  # ce
        logsum = mx.logsumexp(sl_base, axis=-1)
        tgt = mx.sum(sl_base * oh, axis=-1)
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
    eikonal_length = lc.eikonal_length
    if eikonal_length is None:
        raise ValueError("micro-batch twin requires LeverConfig.eikonal_length")
    eik, length, _ = eikonal_length(
        phi0,
        render_h,
        render_w,
        junction_relax=float(lc.eik_jrelax),
        junction_tau=float(lc.eik_jtau),
    )
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
        pi = int(c1) // 2
        island_weight = _provider_at(lc.island_weight_mx, pi, "island_weight_mx")
        ramp = lc.amplify_ramp_state
        if ramp is not None and bool(ramp.get("amp_active", False)):
            lane_mask = _provider_at(lc.amplify_lane_masks, pi, "amplify_lane_masks")
            movable_mask = _provider_at(lc.amplify_movable_masks, pi, "amplify_movable_masks")
            amp_term = island_birth_perclass_from_signed_mx(
                _signed_wa, island_weight, lane_mask, movable_mask, lc.amplify_mtgt,
                float(ramp.get("amp_lane", 1.0)), float(ramp.get("amp_mov", 1.0)),
                form=lc.amplify_form,
            )
        else:
            amp_term = island_birth_from_signed_mx(
                _signed_wa, island_weight, lc.amplify_mtgt, form=lc.amplify_form)
        Lk = Lk + lc.amplify_w * amp_term
    if lc.persist_gate["w"] > 0.0 and lc.persist_classes:
        sg_precomputed = None
        if lc.persistence_sg_cache is not None:
            sg_precomputed = _provider_at(
                lc.persistence_sg_cache, int(c0) // 2, "persistence_sg_cache")
        recall_scale = None
        if lc.amplify_ramp_state is not None:
            recall_scale = lc.amplify_ramp_state.get("persist_scale")
        Lk = Lk + lc.persist_gate["w"] * persistence_topology_loss_mlx(
            sl_wa, oh, lc.persist_classes, cldice_iters=lc.persist_cldice_iters,
            w_cldice=1.0, w_recall=lc.persist_recall_w, sg_precomputed=sg_precomputed,
            recall_class_scale=recall_scale)
    if lc.lane_thin_w > 0.0 and lc.lane_thin_gate["on"] and lc.thin_maps_mx is not None:
        tw = lc.thin_maps_mx[c0 // 2]
        hmap_t = mx.maximum(lc.lane_thin_tgt - _signed, 0.0) * tw
        Lk = Lk + lc.lane_thin_w * (mx.sum(hmap_t) / (mx.sum(tw) + 1e-6))
    if lc.mfh_w > 0.0 and lc.mfh_target_mx is not None:
        per_pix_tgt = mx.sum(oh * lc.mfh_target_mx, axis=-1)
        Lk = Lk + lc.mfh_w * mx.mean(mx.maximum(per_pix_tgt - _signed, 0.0))
    return Lk


def _once_terms(model, lc: LeverConfig):
    """Per-MODEL penalties (rankfloor / code-spec / code-nuc / weight-entropy rate). Returns a
    scalar mx array (0.0 when all off). Added ONCE per step (matches the serial mean-over-chunk)."""
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
        nuclear_norm_smooth = lc.nuclear_norm_smooth
        if nuclear_norm_smooth is None:
            raise ValueError(
                "micro-batch twin requires LeverConfig.nuclear_norm_smooth when code_nuc_w > 0"
            )
        L = L + lc.code_nuc_w * nuclear_norm_smooth(
            model.code,
            rel_eps=float(lc.code_nuc_eps),
            ns_iters=int(lc.code_nuc_iters),
        )
    # Weight-entropy rate penalty (the Ballé rate-in-the-loss MLX port): identical per pair =>
    # added ONCE per step, exactly like the code penalties above. Default 0.0 => never built.
    if lc.we_lambda > 0.0:
        from tac.boundary_math.weight_entropy_penalty_mlx import weight_entropy_rate_term_mlx
        _we_bits, _we_rate = weight_entropy_rate_term_mlx(model, sigma=float(lc.we_sigma))
        L = L + lc.we_lambda * _we_rate
    return L


def _island_levers_on(lc: LeverConfig) -> bool:
    """The #224 island-formation levers (amplify OR persistence) engaged this step. Mirrors the
    trainer's ``_island_levers_on`` (gate dicts read live)."""
    return bool((lc.amplify_w > 0.0 and lc.island_weight_mx is not None) or
                (lc.persist_gate["w"] > 0.0 and bool(lc.persist_classes)) or
                (lc.area_lambda is not None))


def _wa_route_active(lc: LeverConfig, render_fn_wa) -> bool:
    """--witness-alone-island-loss routing active for THIS step: the flag is set, a seed-excluded
    render is supplied, and >=1 island lever is engaged. Mirrors the trainer's ``_wa_route``."""
    return bool(lc.wa_island) and (render_fn_wa is not None) and _island_levers_on(lc)


def _stack_pair_providers(provider, pair_indices, name: str, *, concatenate: bool = True):
    """Stack theta-independent rows without performing lever math in Python."""
    import mlx.core as mx

    rows = [_provider_at(provider, pi, name) for pi in pair_indices]
    return mx.concatenate(rows, axis=0) if concatenate else mx.stack(rows, axis=0)


def _single_area_constraint(logits, oh, lc: LeverConfig):
    """Independent B=1 serial-form reference for the vectorized area implementation."""
    import mlx.core as mx

    if lc.area_lambda is None:
        return mx.zeros(())
    soft = mx.softmax(logits, axis=-1)
    total = mx.zeros(())
    for area_cls, area_lam in lc.area_lambda.items():
        cls = int(area_cls)
        over = mx.maximum(mx.mean(soft[..., cls]) - mx.mean(oh[..., cls]), 0.0)
        total = total + 0.5 * float(area_lam) * over * over
    return total


def _batched_v9_map_terms(model, seg_logits_b, island_logits_b, f1_b,
                          temporal_f0_logits_b, oh_b, pair_indices, lc: LeverConfig,
                          *, include_area: bool = True):
    """Fully vectorized V9 pixel-map levers; returns mean of pair-local normalized terms."""
    import mlx.core as mx

    from tac.local_acceleration.metal_micro_batch_v9_levers import (
        chroma_squared_map,
        phase_squared_map,
        temporal_squared_map,
    )

    axes = (1, 2)
    total = mx.zeros(())
    signed = None

    def raw_signed():
        nonlocal signed
        if signed is None:
            gt = mx.sum(seg_logits_b * oh_b, axis=-1)
            runner = mx.max(seg_logits_b + oh_b * (-1e9), axis=-1)
            signed = gt - runner
        return signed

    # Chan-Vese area is vectorized across B: each row first obtains its own spatial mean, class
    # penalties are summed per row, then rows are averaged. There is no Python pair loop and no
    # accidental global-B area average before the one-sided hinge.
    if include_area and lc.area_lambda is not None:
        if island_logits_b is None:
            raise ValueError("micro-batch area constraint requires realized witness logits")
        soft = mx.softmax(island_logits_b, axis=-1)
        per_pair = mx.zeros((int(seg_logits_b.shape[0]),), dtype=seg_logits_b.dtype)
        for area_cls, area_lam in lc.area_lambda.items():
            cls = int(area_cls)
            realized = mx.mean(soft[..., cls], axis=axes)
            target = mx.mean(oh_b[..., cls], axis=axes)
            over = mx.maximum(realized - target, 0.0)
            per_pair = per_pair + 0.5 * float(area_lam) * mx.square(over)
        total = total + mx.mean(per_pair)

    if float(lc.chroma_w) > 0.0 and _gate_on(lc.chroma_gate):
        target = _stack_pair_providers(lc.chroma_gt_prov, pair_indices, "chroma_gt_prov")
        ann = _stack_pair_providers(lc.chroma_ann_prov, pair_indices, "chroma_ann_prov")
        sq = chroma_squared_map(f1_b, target, use_metal=bool(lc.use_metal_v9_levers))
        numer = mx.sum(sq * ann, axis=axes)
        denom = mx.sum(ann, axis=axes) + 1e-6
        total = total + float(lc.chroma_w) * mx.mean(numer / denom)

    if float(lc.phase_w) > 0.0 and _gate_on(lc.phase_gate):
        reference = _stack_pair_providers(lc.phase_ref_prov, pair_indices, "phase_ref_prov")
        direction = _stack_pair_providers(lc.phase_dir_prov, pair_indices, "phase_dir_prov")
        weight = _stack_pair_providers(lc.phase_weight_prov, pair_indices, "phase_weight_prov")
        sq = phase_squared_map(
            raw_signed(), direction, reference, float(lc.phase_eps),
            use_metal=bool(lc.use_metal_v9_levers),
        )
        numer = mx.sum(sq * weight, axis=axes)
        denom = mx.sum(weight, axis=axes) + 1e-6
        total = total + float(lc.phase_w) * mx.mean(numer / denom)

    if float(lc.temporal_w) > 0.0 and _gate_on(lc.temporal_gate):
        if temporal_f0_logits_b is None:
            raise ValueError("micro-batch temporal screw active but frame-0 logits were not scored")
        if lc.temporal_geom_mlx is None or lc.temporal_class_mask is None:
            raise ValueError(
                "micro-batch temporal screw active but temporal_geom_mlx/class_mask is not wired")
        source = str(lc.temporal_xi_source)
        if source == "ground_gt":
            xi = _stack_pair_providers(
                lc.temporal_xi_prov, pair_indices, "temporal_xi_prov", concatenate=False)
        elif source == "carrier_live":
            carrier = getattr(model, "pose_carrier", None)
            if carrier is None or not hasattr(carrier, "xi_effective"):
                raise ValueError("temporal_xi_source=carrier_live requires model.pose_carrier")
            # FiLM residual carriers require the frame-0 code vector; table carriers ignore it.
            # The carrier call itself is pair-specific controller state, while the downstream
            # warp/residual pixel path remains one batch-native operation and one fused dispatch.
            xi = mx.stack([
                carrier.xi_effective(int(pi), model.code[2 * int(pi)])
                for pi in pair_indices
            ], axis=0)
        else:
            raise ValueError(f"unknown temporal_xi_source {source!r}")
        ann = _stack_pair_providers(lc.temporal_ann_prov, pair_indices, "temporal_ann_prov")
        phi1 = mx.softmax(seg_logits_b, axis=-1)
        phi0 = mx.softmax(temporal_f0_logits_b, axis=-1)
        # The first-three-of-five view is not row-contiguous, while both the homography gather and
        # custom temporal kernel require a packed RGB-like field. Pack both fields into ONE explicit
        # contiguous allocation so Metal does not silently insert one copy at each consumer.
        ground_pair = mx.contiguous(mx.stack([phi1[..., 0:3], phi0[..., 0:3]], axis=0))
        g1 = ground_pair[0]
        g0 = ground_pair[1]
        from tac.boundary_math.warp_real_luma_frame0 import compiled_batch_native_warp

        # The existing compiled warp is batch-native and gradient-preserving. Reusing one compiled
        # callable for full and optional rotation-only twists keeps the B axis out of Python and
        # removes the eager homography/gather dispatch train from this hot micro-batch path.
        warp_batch = compiled_batch_native_warp(lc.temporal_geom_mlx)
        full = warp_batch(g0, xi)
        if (lc.temporal_rot_mask is None) != (lc.temporal_sky_rowmask is None):
            raise ValueError(
                "temporal sky rotation requires both temporal_rot_mask and temporal_sky_rowmask")
        if lc.temporal_rot_mask is not None:
            rot = warp_batch(g0, xi * lc.temporal_rot_mask)
            g0w = lc.temporal_sky_rowmask * rot + (1.0 - lc.temporal_sky_rowmask) * full
        else:
            g0w = full
        sq = temporal_squared_map(
            g1, g0w, lc.temporal_class_mask, use_metal=bool(lc.use_metal_v9_levers))
        numer = mx.sum(sq * ann, axis=axes)
        denom = mx.sum(ann, axis=axes) + 1e-6
        total = total + float(lc.temporal_w) * mx.mean(numer / denom)
    return total


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
    byte-identical (no 2nd forward). The frozen SegNet can exhibit measured batch-dependent fp
    drift on this witness-alone forward too. Its contract is functional loss/gradient equivalence
    within the registered tolerance, not exact row equality.
    """
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    K = len(c1_list)
    f1_b = mx.concatenate([render_fn(model, cf_list[k], int(c1_list[k]), render_h, render_w)
                           for k in range(K)], axis=0)                 # (K,H,W,3)
    f0_b = mx.concatenate([render_fn(model, cf_list[k], int(c0_list[k]), render_h, render_w)
                           for k in range(K)], axis=0)                 # (K,H,W,3)
    seg_logits_b = adapter.segnet(f1_b)                               # (K,H,W,5) RAW: levers + wa base
    # (#D15 logit-adjust routing) the BASE seg-form reads the class-prior-ADJUSTED logits (offset a
    # (5,) per-class constant broadcast over (K,H,W,5). The add is row-local, but its input logits can
    # carry scorer batch drift; the surgical levers keep the RAW seg_logits_b. offset None
    # => _sl_base returns None => _pair_loss_from_scored aliases sl => byte-identical.
    live_offset = _live_logit_offset(lc)
    seg_logits_base_b = seg_logits_b if live_offset is None else seg_logits_b + live_offset
    temporal_active = float(lc.temporal_w) > 0.0 and _gate_on(lc.temporal_gate)
    # Force-1 reads the witness's own frame-0 partition. Exactly one additional batched SegNet
    # forward is issued only while the live temporal gate is on.
    temporal_f0_logits_b = None
    if temporal_active:
        if lc.temporal_render_f0_fn is None:
            temporal_f0_b = f0_b
        else:
            # Rendering remains per frame because coordinate/self-orient features are pair-local;
            # scoring is still exactly ONE batched temporal frame0 SegNet call (never a scorer loop).
            temporal_f0_b = mx.concatenate([
                lc.temporal_render_f0_fn(
                    model, cf_list[k], int(c0_list[k]), render_h, render_w)
                for k in range(K)
            ], axis=0)
        temporal_f0_logits_b = adapter.segnet(temporal_f0_b)
    # (--witness-alone-island-loss #300a) SECOND (seed-excluded) SegNet forward for the island
    # levers, ONLY when routed. Only frame1 is needed (island levers read the SegNet-scored margin).
    seg_logits_wa_b = None
    if _wa_route_active(lc, render_fn_wa):
        if render_fn_wa is None:
            raise ValueError("witness-alone route is active without render_fn_wa")
        f1_wa_b = mx.concatenate([
            render_fn_wa(model, cf_list[k], int(c1_list[k]), render_h, render_w)
            for k in range(K)
        ], axis=0)          # (K,H,W,3) seed-EXCLUDED
        seg_logits_wa_b = adapter.segnet(f1_wa_b)                      # (K,H,W,5)
    pair = mx.stack([f0_b, f1_b], axis=1)                             # (K,2,H,W,3)
    yuv = rgb_to_yuv6_mlx(pair)                                       # (K,2,h2,w2,6)
    _k, _t, _h2, _w2, _c6 = yuv.shape
    yuv_nhwc = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (_k, _h2, _w2, _t * _c6))
    _half = pose_tgt_list[0].shape[-1]
    pose_b = adapter.posenet(yuv_nhwc)["pose"][..., :_half]           # (K, half)
    oh_b = mx.concatenate(oh_list, axis=0)

    def _sl_wa(k):
        return None if seg_logits_wa_b is None else seg_logits_wa_b[k:k + 1]

    def _sl_base(k):
        return None if live_offset is None else seg_logits_base_b[k:k + 1]

    L = _pair_loss_from_scored(
        model, seg_logits_b[0:1], pose_b[0], cf_list[0], int(c0_list[0]), int(c1_list[0]),
        oh_list[0], mg_list[0], pose_tgt_list[0], f1_b[0:1], render_h, render_w,
        w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, sl_wa=_sl_wa(0), sl_base=_sl_base(0))
    for k in range(1, K):
        L = L + _pair_loss_from_scored(
            model, seg_logits_b[k:k + 1], pose_b[k], cf_list[k], int(c0_list[k]), int(c1_list[k]),
            oh_list[k], mg_list[k], pose_tgt_list[k], f1_b[k:k + 1], render_h, render_w,
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, sl_wa=_sl_wa(k), sl_base=_sl_base(k))
    L = L / float(K)
    island_logits_b = seg_logits_b if seg_logits_wa_b is None else seg_logits_wa_b
    L = L + _batched_v9_map_terms(
        model, seg_logits_b, island_logits_b, f1_b, temporal_f0_logits_b, oh_b,
        [int(c1) // 2 for c1 in c1_list], lc)
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
    sl = adapter.segnet(f1)                                           # (1,H,W,5) RAW: levers
    # (#D15 logit-adjust routing) BASE seg-form reads the ADJUSTED logits; None => aliases sl.
    live_offset = _live_logit_offset(lc)
    sl_base = None if live_offset is None else sl + live_offset
    temporal_active = float(lc.temporal_w) > 0.0 and _gate_on(lc.temporal_gate)
    temporal_f0_logits = None
    if temporal_active:
        temporal_f0 = (f0 if lc.temporal_render_f0_fn is None else
                       lc.temporal_render_f0_fn(model, cf, int(c0), render_h, render_w))
        temporal_f0_logits = adapter.segnet(temporal_f0)
    sl_wa = None
    if _wa_route_active(lc, render_fn_wa):
        if render_fn_wa is None:
            raise ValueError("witness-alone route is active without render_fn_wa")
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
                               w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc,
                               sl_wa=sl_wa, sl_base=sl_base)
    island_logits = sl if sl_wa is None else sl_wa
    L = L + _single_area_constraint(island_logits, oh, lc)
    # The B=1 reference intentionally uses an independent area formula above. The remaining map
    # levers share only their vectorized primitive implementations, with B fixed to one.
    L = L + _batched_v9_map_terms(
        model, sl, island_logits, f1, temporal_f0_logits, oh, [int(c1) // 2], lc,
        include_area=False)
    L = L + _once_terms(model, lc)
    return L
