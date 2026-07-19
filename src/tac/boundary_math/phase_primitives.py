"""SHARED phase primitives for the level-set witness — ONE home for the two objects
the cross-pair phase-advection lever (T1) and the #360 within-pair forces both need.

The two primitives (the coordinator's dedup deliverable — #360's Force-1 / Force-3 and
this file's T1 consumer import the SAME code, they do not duplicate it inline):

  1. **the sub-pixel TIE COORDINATE** ``t_wit`` — the straddle-pixel phase of the φ=0
     boundary. At an inter-class straddle pixel ``p`` with dominant cross-edge partner
     ``q`` the witness's realized GT-class margin ratio ``t = Mw[p] / (Mw[p] + Mw[q])``
     (``Mw = relu(signed)``) is a differentiable sub-pixel boundary POSITION in [0,1]
     between ``p`` and ``q`` (#275 vector-field margin-saliency sub-pixel ``t``). This is
     the EXACT quantity the incumbent ``subpix`` term (Force-3) computes inline in
     ``experiments/train_levelset_witness_realized_through_R_mlx.py`` (~L5171-5177) — the
     extraction here is op-for-op identical so Force-3 and T1 read ONE tie coordinate.

  2. **the se(3) ξ-transport** ``A_ξ`` of a boundary field — the ego-screw GROUND-plane
     homography warp ``H(ξ) = K(R − t·nᵀ/d)K⁻¹`` (twist ξ → tac.lie ``exp_se3`` →
     ``tac.boundary_math.warp_real_luma_frame0.warp_frame0_native_mlx``, the SAME
     bit-checked warp Force-1 uses; #193). :func:`advect_tie_field_mlx` is a thin
     channel-agnostic wrapper (tile the scalar tie field to 3 channels → the exact shared
     warp → slice channel 0) so the tie-field advection is LITERALLY the same A_ξ as the
     GROUND-prob-field warp Force-1 applies.

Plus the cross-scored-frame screw interpolation (:func:`cross_scored_frame_xi_interp`) and
the T1 residual reducer (:func:`phase_advection_weighted_mse_mlx`).

**MLX fast path + numpy-fp32 REFERENCE (the deterministic authority) + parity.** Every MLX
function has a numpy-fp32 twin; the twin is the verdict authority (per the
deterministic-reproducibility spine). MPS is NEVER an authority. Nothing here is a score —
these are training-time priors; the pointer moves only through a byte-closed
``upstream/evaluate.py`` exact row.

Consumers:
  * T1 CROSS-PAIR (this lever): witness tie coordinate at scored pair ``p`` vs the
    ξ-advected GT tie of the PREVIOUS scored pair ``p-1`` (a θ-independent precomputed
    target — the cross-pair coupling never enters the per-pair value_and_grad, so it fits
    the incumbent random-permutation batching with ZERO change).
  * #360 Force-3 (tie-locus displacement): witness tie coordinate at ``p`` vs the
    per-pair GT tie of ``p`` (the raw target). T1 composed with Force-3 = the optimal
    shrinkage of the noisy per-pair phase targets toward the ξ-advected trajectory.
  * #360 Force-1 (temporal-screw): the SAME A_ξ (:func:`advect_tie_field_mlx` shares the
    warp with the GROUND-prob field path).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "advect_tie_field_mlx",
    "advect_tie_field_numpy",
    "cross_scored_frame_xi_interp",
    "gt_tie_targets_numpy",
    "phase_advection_weighted_mse_mlx",
    "phase_advection_weighted_mse_numpy",
    "witness_tie_coordinate_mlx",
    "witness_tie_coordinate_numpy",
]


# --------------------------------------------------------------------------- #
# 0. GT-side tie-target selection (dominant genuine-V straddle)                #
# --------------------------------------------------------------------------- #
def gt_tie_targets_numpy(
    lstars: np.ndarray, margins: np.ndarray, band: float = 1.0, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT sub-pixel tie target ``t`` + dominant direction + active mask for one pair.

    Op-for-op the incumbent ``subpix`` provider's per-pair straddle selection (trainer
    ~L5816-5843): at each pixel ``p`` consider the RIGHT ``(i,j+1)`` and DOWN ``(i+1,j)``
    inter-class straddles; keep only GENUINE-V straddles (``lstar`` differs AND BOTH GT
    margins ``< band``, i.e. inside the flip band); assign ``p`` its DOMINANT straddle = the
    one with the SHALLOWER partner margin (ties → right). Returns:

      * ``t_full`` (H,W) f32 = ``M_GT[p]/(M_GT[p]+M_GT[q])`` in [0,1] where active, ``-1.0``
        sentinel elsewhere (the sentinel encodes the active mask, matching the trainer);
      * ``dir_full`` (H,W) f32 in {0,1} = dominant direction (0=right, 1=down);
      * ``active`` (H,W) bool.

    Factored here so the T1 cross-pair lever and the #360 Force-3 within-pair lever select
    the GT tie targets from ONE implementation (the coordinator's dedup mandate — GT side).
    """
    lst = np.asarray(lstars, np.int64)
    mg = np.asarray(margins, np.float32)
    H, W = lst.shape
    _dh = lst[:, :-1] != lst[:, 1:]
    _mph, _mqh = mg[:, :-1], mg[:, 1:]
    _th = _mph / (_mph + _mqh + np.float32(eps))
    _vh = _dh & (_mph < band) & (_mqh < band)
    _dv = lst[:-1, :] != lst[1:, :]
    _mpv, _mqv = mg[:-1, :], mg[1:, :]
    _tv = _mpv / (_mpv + _mqv + np.float32(eps))
    _vv = _dv & (_mpv < band) & (_mqv < band)
    _has_r = np.zeros((H, W), bool)
    _has_r[:, : W - 1] = _vh
    _qr = np.full((H, W), np.inf, np.float32)
    _qr[:, : W - 1] = _mqh
    _tr = np.zeros((H, W), np.float32)
    _tr[:, : W - 1] = _th
    _has_d = np.zeros((H, W), bool)
    _has_d[: H - 1, :] = _vv
    _qd = np.full((H, W), np.inf, np.float32)
    _qd[: H - 1, :] = _mqv
    _td = np.zeros((H, W), np.float32)
    _td[: H - 1, :] = _tv
    _pick_r = _has_r & (~_has_d | (_qr <= _qd))
    _pick_d = _has_d & (~_has_r | (_qd < _qr))
    t_full = np.full((H, W), -1.0, np.float32)
    dir_full = np.zeros((H, W), np.float32)
    t_full[_pick_r] = _tr[_pick_r]
    dir_full[_pick_r] = 0.0
    t_full[_pick_d] = _td[_pick_d]
    dir_full[_pick_d] = 1.0
    active = _pick_r | _pick_d
    return t_full, dir_full, active


# --------------------------------------------------------------------------- #
# 1. the sub-pixel tie coordinate  t_wit = Mw[p] / (Mw[p] + Mw[q])            #
# --------------------------------------------------------------------------- #
def _as_hw(a: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (arr as (H,W), had_leading_1) so we can restore (1,H,W)."""
    arr = np.asarray(a)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"expected (H,W) or (1,H,W), got shape {arr.shape}")


def witness_tie_coordinate_numpy(
    signed: np.ndarray, dir_map: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Numpy-fp32 REFERENCE for the witness sub-pixel tie coordinate.

    ``signed`` (H,W) or (1,H,W) = the realized through-R witness GT-class SIGNED margin
    (``_signed``; Mw = relu(signed)). ``dir_map`` in {0,1}: 0 = the cross-edge partner is
    RIGHT ``(i, j+1)``, 1 = DOWN ``(i+1, j)`` (the dominant straddle direction — the same
    convention as the incumbent ``_subpix_dir_prov``). Returns ``t = Mw / (Mw + Mq + eps)``
    with the same shape as ``signed``. Op-for-op the trainer's inline subpix computation.

    Edge column/row of the shift is pad-with-0 (inert: those pixels can never be an active
    straddle in that direction, so the padded partner is masked out downstream).
    """
    sig, had1 = _as_hw(signed)
    d, _ = _as_hw(dir_map)
    sig = sig.astype(np.float32)
    Mw = np.maximum(sig, 0.0)
    # partner margins via pure shifts (Mw[i,j+1] and Mw[i+1,j]); pad the far edge with 0.
    M_right = np.zeros_like(Mw)
    M_right[:, :-1] = Mw[:, 1:]
    M_down = np.zeros_like(Mw)
    M_down[:-1, :] = Mw[1:, :]
    Mq = np.where(d < 0.5, M_right, M_down).astype(np.float32)
    t = Mw / (Mw + Mq + np.float32(eps))
    return t[None] if had1 else t


def witness_tie_coordinate_mlx(signed: Any, dir_map: Any, eps: float = 1e-6) -> Any:
    """Differentiable MLX twin of :func:`witness_tie_coordinate_numpy`.

    ``signed`` / ``dir_map`` are (1,H,W) MLX arrays (the trainer's shape). Returns (1,H,W)
    ``t_wit`` differentiable in ``signed`` (the gradient seats both Mw[p] and Mw[q]). This
    is EXACTLY the trainer's inline subpix ratio (L5171-5177) so Force-3 and T1 share it.
    """
    import mlx.core as mx

    Mw = mx.maximum(signed, 0.0)
    # Mw[i,j+1] (right) and Mw[i+1,j] (down) via pure pad-shifts (same as the trainer).
    M_right = mx.pad(Mw[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
    M_down = mx.pad(Mw[:, 1:, :], [(0, 0), (0, 1), (0, 0)])
    Mq = mx.where(dir_map < 0.5, M_right, M_down)
    return Mw / (Mw + Mq + eps)


# --------------------------------------------------------------------------- #
# 2. A_ξ : ego-screw ground-homography advection of a scalar boundary field    #
# --------------------------------------------------------------------------- #
def advect_tie_field_numpy(
    t_field: np.ndarray, xi: np.ndarray, geom: Any, *, compute_dtype: type = np.float32
) -> np.ndarray:
    """Numpy REFERENCE: warp a scalar tie/boundary field by twist ``xi`` (A_ξ).

    ``t_field`` (H,W) or (1,H,W) in [0,1]. ``geom`` = a
    ``warp_real_luma_frame0.GroundHomographyGeom`` at the field resolution. Reuses the
    SHARED ``warp_frame0_native_numpy`` (tile scalar → 3 channels → warp → channel 0) so
    the geometry is byte-identical to the GROUND-prob field warp Force-1 uses. Off-frame /
    behind-camera pixels take the PERSIST fallback (source value), the same non-gameable
    accounting the warp's numpy oracle uses.
    """
    from tac.boundary_math.warp_real_luma_frame0 import warp_frame0_native_numpy

    fld, had1 = _as_hw(t_field)
    src3 = np.repeat(fld.astype(compute_dtype)[:, :, None], 3, axis=2)  # (H,W,3)
    warped = warp_frame0_native_numpy(src3, np.asarray(xi), geom, compute_dtype=compute_dtype)
    out = warped[:, :, 0].astype(np.float32)
    return out[None] if had1 else out


def advect_tie_field_mlx(t_field: Any, xi: Any, geom_mlx: Any) -> Any:
    """MLX twin: warp a scalar tie/boundary field by twist ``xi`` (A_ξ).

    ``t_field`` (1,H,W) MLX. ``geom_mlx`` = ``GroundHomographyGeom.mlx()`` /
    ``_GeomMLX``. Reuses the EXACT ``warp_frame0_native_mlx`` (tile scalar → 3 channels →
    warp → channel 0) — the SAME A_ξ Force-1's GROUND-prob warp uses. Typically called at
    PROVIDER-BUILD on the θ-independent GT tie target (so it never enters value_and_grad);
    it is nonetheless differentiable in the field for the owed witness-self-consistency mode.
    """
    import mlx.core as mx

    from tac.boundary_math.warp_real_luma_frame0 import warp_frame0_native_mlx

    fld = t_field[0] if (t_field.ndim == 3 and t_field.shape[0] == 1) else t_field  # (H,W)
    src3 = mx.stack([fld, fld, fld], axis=-1)  # (H,W,3)
    warped = warp_frame0_native_mlx(src3, xi, geom_mlx)  # (H,W,3)
    return warped[:, :, 0][None]  # (1,H,W)


def cross_scored_frame_xi_interp(xi_pi: np.ndarray, xi_pi1: np.ndarray) -> np.ndarray:
    """Cross-SCORED-frame twist f1(pi) → f1(pi+1), se(3)-composed (``gap_xi=interp``).

    Pairs are NON-OVERLAPPING consecutive frames: pair ``pi`` = (2pi, 2pi+1),
    pair ``pi+1`` = (2pi+2, 2pi+3). The scored frames are f1(pi)=2pi+1 and f1(pi+1)=2pi+3,
    two frame-intervals apart: the GAP (2pi+1→2pi+2, odd→even, NOT in ``gt_poses``) then
    pair pi+1's within-pair screw (2pi+2→2pi+3). ``interp`` estimates the gap screw as the
    average of the two adjacent within-pair screws ``ξ_gap ≈ ½(ξ_pi + ξ_pi+1)`` (memo §4
    T1), then composes ``ξ_cross = log_se3( exp_se3(ξ_pi+1) · exp_se3(ξ_gap) )`` via the
    tac.lie numpy se(3) oracle (proper group composition — NOT a linear twist sum). A
    train-time regularizer target; the gap approximation error is tolerated (memo).

    ``xi_pi`` / ``xi_pi1`` are (6,) rho-first twists (as returned by
    ``warp_real_luma_frame0.xi_from_pose_calibration``). Returns (6,) fp64.
    """
    from tac.lie import _se3_numpy as _np_se3

    a = np.asarray(xi_pi, dtype=np.float64).reshape(6)
    b = np.asarray(xi_pi1, dtype=np.float64).reshape(6)
    xi_gap = 0.5 * (a + b)
    T_cross = _np_se3.exp_se3(b) @ _np_se3.exp_se3(xi_gap)  # apply gap first, then pair pi+1
    return np.asarray(_np_se3.log_se3(T_cross), dtype=np.float64).reshape(6)


# --------------------------------------------------------------------------- #
# 3. the T1 residual reducer  mean_{weight}( (t_wit − t_ref)² )                #
# --------------------------------------------------------------------------- #
def phase_advection_weighted_mse_numpy(
    t_wit: np.ndarray, t_ref: np.ndarray, weight: np.ndarray, eps: float = 1e-6
) -> float:
    """Numpy REFERENCE: weighted-mean squared phase residual on the active annulus.

    ``weight`` in {0,1} (annulus ∩ active straddle) selects where the reference is
    defined; the term is the mean of ``(t_wit − t_ref)²`` over those pixels. Zero weight
    everywhere ⇒ 0.0 (no-op, byte-safe).
    """
    tw, _ = _as_hw(t_wit)
    tr, _ = _as_hw(t_ref)
    w, _ = _as_hw(weight)
    sq = (tw.astype(np.float64) - tr.astype(np.float64)) ** 2 * w.astype(np.float64)
    return float(sq.sum() / (w.astype(np.float64).sum() + eps))


def phase_advection_weighted_mse_mlx(t_wit: Any, t_ref: Any, weight: Any, eps: float = 1e-6) -> Any:
    """MLX twin: weighted-mean squared phase residual (differentiable in ``t_wit``)."""
    import mlx.core as mx

    sq = mx.square(t_wit - t_ref) * weight
    return mx.sum(sq) / (mx.sum(weight) + eps)


# --------------------------------------------------------------------------- #
# 4. the EVENT-FALLBACK target composer (FEED-lane-gain §4b; SPEC_v10 §13.1)   #
# --------------------------------------------------------------------------- #
def event_fallback_ref_and_weight_numpy(
    t_own: np.ndarray, own_active: np.ndarray,
    ref_advected: np.ndarray, ref_active: np.ndarray,
    annulus: np.ndarray, ground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """The event-fallback phase target/weight — "advect-where-persistent, target-where-born".

    The MEASURED gap this closes (FEED-lane-gain §4b, n600 T1 audit): T1's incumbent weight
    ``ann ∧ ground ∧ ref_active`` is birth-SILENT — 26.3% of candidate straddle pixels
    (601/frame; 354 lane-adjacent) receive NO T1 supervision because a birth/fast-moved site
    has no advected reference. This composer implements the memo's exact amendment::

        t_ref  := where(ref_active, advected_prev_tie, own_gt_tie)
        weight := ann ∧ ground ∧ (ref_active ∨ own_active)

    as a FALLBACK, NOT as gating advection off: wherever a valid ξ-advected reference
    exists it is kept verbatim (the transport channel is untouched); ONLY the uncovered
    own-tie-active sites (births + fast-moved structure) gain supervision, toward pair p's
    OWN GT tie. STATELESS by construction — there is NO per-island persistence hold and no
    cross-epoch memory (the memo's anti-scope: a persistence hold would fight GT's genuine
    deaths; the fallback target is θ-independent and per-pair-local exactly like T1's).

    Args (all (H,W)): ``t_own`` f32 own GT tie (−1 sentinel), ``own_active`` bool own-tie
    active mask, ``ref_advected`` f32 warped previous-pair tie VALUE (defined where
    ``ref_active``), ``ref_active`` bool valid-warped-reference mask, ``annulus`` bool,
    ``ground`` bool. For pair 0 pass ``ref_active`` all-False (no previous scored frame):
    the fallback then supervises pair 0's own-tie sites (the incumbent mode leaves pair 0
    a no-op).

    Returns ``(t_ref, weight, n_advected, n_fallback)``: ``t_ref`` (H,W) f32 with −1
    sentinel outside the weight support; ``weight`` (H,W) f32 in {0,1};
    ``n_advected``/``n_fallback`` the supervised-pixel counts per channel (telemetry —
    the fallback share is the memo's 26.3% coverage gap, now covered).
    """
    to, _ = _as_hw(t_own)
    ao, _ = _as_hw(own_active)
    rv, _ = _as_hw(ref_advected)
    ra, _ = _as_hw(ref_active)
    ann, _ = _as_hw(annulus)
    gnd, _ = _as_hw(ground)
    ao = ao.astype(bool)
    ra = ra.astype(bool)
    base = ann.astype(bool) & gnd.astype(bool)
    adv = base & ra                       # transport channel (incumbent, verbatim)
    fb = base & ao & ~ra                  # event-fallback channel (births/fast-moved only)
    t_ref = np.full(to.shape, -1.0, np.float32)
    t_ref[adv] = rv.astype(np.float32)[adv]
    t_ref[fb] = to.astype(np.float32)[fb]
    weight = (adv | fb).astype(np.float32)
    return t_ref, weight, int(adv.sum()), int(fb.sum())
