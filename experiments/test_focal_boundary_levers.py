"""FOCAL-GAMMA + BOUNDARY-DISTANCE seg-loss levers — unit + $0 real-scorer byte-identity tests.

Council mandate (memo ``.omx/research/council_grand_symposium_levelset_loss_geometry_20260705.md``
deliverable 2): build ``--seg-focal-gamma`` and ``--boundary-distance-weight`` as DEFAULT-OFF
flags, tested + byte-identity-proven, READY but NOT deployed (the pre-registered fire criterion
is the parent's decision). Pointer 0.19110 UNMOVED — these are $0 build gates, not score claims.

Tests:
  1. FOCAL MATH (pure, exact): the 5^gamma weight-ratio law (Rudin's readback: under STOP-GRAD
     mean-1-renormalized reweighting the gradient-weight ratio between a p=0.5 and a p=0.9 pixel
     is EXACTLY ((1-0.5)/(1-0.9))^gamma = 5^gamma), mean-1 renorm, stop-grad (zero gradient
     through the weight), and monotone island-share concentration in gamma (Shannon's law
     share_isl(gamma) = sum_isl (1-p)^gamma / sum_all (1-p)^gamma).
  2. BOUNDARY BAND MAP (pure, exact): distance-transform geometry (1.0 on the straddle pixels,
     0.5 at 1 px, 0.0 at >=2 px with the default band_px=2), degenerate single-class -> zeros.
  3. BOUNDARY TERM (mx, synthetic known-offset): the SDF-native band-weighted |phi_GT-phi_runner|
     is ~minimal when the witness tie-line sits ON the GT boundary and grows MONOTONICALLY with
     the boundary offset (the move-the-contour DOF).
  4. RESUME DRIFT GUARD (pure): both new levers are registered in _resume_lever_divergences
     (silently dropping/changing them on --resume-from fails closed).
  5. ARGPARSE (source-verified, never-invent): the exact flags exist with default 0.0.
  6. BYTE-IDENTITY (slow, real gt_n6 + real frozen MLX SegNet + real make_loss_fn): with
     focal_gamma=0.0 the loss AND grads are byte-for-byte the no-kwarg (current-trainer) path AND
     byte-for-byte a reference re-implementation of the pre-change CE expression; gamma>0 changes
     the loss; seg_pixel_w=ones composes bitwise-identically with the focal fold-in.

Run: .venv/bin/pytest experiments/test_focal_boundary_levers.py -v -m "not slow"   (fast)
     .venv/bin/pytest experiments/test_focal_boundary_levers.py -v                 (incl. real proof)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
# MLX CPU is the equivalence AUTHORITY device (GPU is not bit-identical cross-process per
# [[mlx_gpu_not_bit_identical_crossprocess...]]; same convention as test_seed_absorption_fix.py).
mx.set_default_device(mx.cpu)
import mlx.nn as nn
from mlx.utils import tree_flatten

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_GT_N6 = _REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n6.npz"
_UPSTREAM = _REPO / "upstream"
_LVL_SRC = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _base():
    import train_witness_realized_through_R_mlx as base
    return base


def _lvl():
    import train_levelset_witness_realized_through_R_mlx as lvl
    return lvl


def _logit_for_p(p: float, n_classes: int = 5) -> float:
    """GT-class logit a (others 0) s.t. softmax GT prob == p: e^a/(e^a+(K-1)) = p."""
    return float(np.log(p / (1.0 - p) * (n_classes - 1)))


# ===========================================================================
# TEST 1 — FOCAL MATH (pure, exact)
# ===========================================================================
def test_focal_weight_ratio_law_5_gamma():
    """Rudin's readback: weight(p=0.5)/weight(p=0.9) == ((1-0.5)/(1-0.9))^gamma == 5^gamma
    EXACTLY (the mean-1 renorm cancels in the ratio) — the registered weight-ratio law."""
    base = _base()
    logits = np.zeros((1, 1, 2, 5), np.float32)
    logits[0, 0, 0, 0] = _logit_for_p(0.5)
    logits[0, 0, 1, 0] = _logit_for_p(0.9)
    oh = np.zeros((1, 1, 2, 5), np.float32)
    oh[..., 0] = 1.0
    for gamma in (0.5, 1.0, 2.0, 3.0):
        fw = np.asarray(base.focal_pixel_weight_mlx(mx.array(logits), mx.array(oh), gamma))
        ratio = fw[0, 0, 0] / fw[0, 0, 1]
        assert ratio == pytest.approx(5.0 ** gamma, rel=2e-4), (gamma, ratio)


def test_focal_weight_mean_one_and_stop_grad():
    """The renorm makes the weight mean ~1 (total gradient BUDGET conserved) and the weight is
    fully STOP-GRAD (zero gradient through it — the reweight semantics, not full focal loss)."""
    base = _base()
    rng = np.random.default_rng(0)
    logits = mx.array(rng.standard_normal((1, 4, 6, 5)).astype(np.float32))
    lab = rng.integers(0, 5, (1, 4, 6))
    oh = mx.array(np.eye(5, dtype=np.float32)[lab])
    fw = base.focal_pixel_weight_mlx(logits, oh, 2.0)
    assert float(mx.mean(fw).item()) == pytest.approx(1.0, rel=1e-4)
    g = mx.grad(lambda lg: mx.sum(base.focal_pixel_weight_mlx(lg, oh, 2.0)))(logits)
    mx.eval(g)
    assert float(mx.max(mx.abs(g)).item()) == 0.0, "focal weight must be stop-grad (zero grad path)"


def test_focal_island_share_monotone_in_gamma():
    """Shannon's concentration law: share_isl(gamma) = sum_isl (1-p)^gamma / sum_all (1-p)^gamma
    is MONOTONE INCREASING in gamma when island pixels are the low-p (hard) pixels."""
    base = _base()
    logits = np.zeros((1, 1, 10, 5), np.float32)
    # 2 "island" pixels at p=0.3 (hard), 8 "bulk" pixels at p=0.95 (easy).
    for i in range(10):
        logits[0, 0, i, 0] = _logit_for_p(0.3 if i < 2 else 0.95)
    oh = np.zeros((1, 1, 10, 5), np.float32)
    oh[..., 0] = 1.0
    shares = []
    for gamma in (0.0001, 0.5, 1.0, 2.0, 3.0):
        fw = np.asarray(base.focal_pixel_weight_mlx(mx.array(logits), mx.array(oh), gamma))
        shares.append(float(fw[0, 0, :2].sum() / fw.sum()))
    assert all(b > a for a, b in itertools.pairwise(shares)), shares
    assert shares[0] == pytest.approx(0.2, abs=0.01)  # gamma->0 => uniform => pixel share


# ===========================================================================
# TEST 2 — BOUNDARY BAND MAP (pure, exact)
# ===========================================================================
def test_boundary_band_map_vertical_split_geometry():
    lvl = _lvl()
    ls = np.zeros((8, 16), np.int64)
    ls[:, 8:] = 1                       # boundary between cols 7|8 -> straddles at cols 7 and 8
    m = lvl.boundary_distance_band_map(ls, band_px=2.0)
    assert m.shape == (8, 16) and m.dtype == np.float32
    assert np.all(m[:, 7] == 1.0) and np.all(m[:, 8] == 1.0)      # D=0 on the straddle pixels
    assert np.all(m[:, 6] == 0.5) and np.all(m[:, 9] == 0.5)      # D=1 -> 1 - 1/2
    assert np.all(m[:, :6] == 0.0) and np.all(m[:, 10:] == 0.0)   # D>=2 -> clipped to 0


def test_boundary_band_map_single_class_is_zero():
    lvl = _lvl()
    m = lvl.boundary_distance_band_map(np.full((6, 6), 3, np.int64))
    assert m.shape == (6, 6) and not m.any()


# ===========================================================================
# TEST 3 — BOUNDARY TERM (mx, synthetic known-offset)
# ===========================================================================
def test_boundary_distance_term_monotone_in_offset():
    """K=2 vertical-split partition; witness phi = per-class SDFs with the tie-line at col b.
    The band-weighted |phi_GT - phi_runner| must be ~minimal at b == GT boundary and grow
    monotonically as the witness boundary is offset."""
    lvl = _lvl()
    H, W, GT_B = 8, 32, 16
    ls = np.zeros((H, W), np.int64)
    ls[:, GT_B:] = 1
    band = mx.array(lvl.boundary_distance_band_map(ls)[None])                  # (1,H,W)
    oh = mx.array(np.eye(5, dtype=np.float32)[ls][None])                       # (1,H,W,5)
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)

    def term_at(b: float) -> float:
        phi = np.full((H, W, 5), -1e3, np.float32)  # absent classes: hugely negative fields
        phi[..., 0] = (b - 0.5) - xs                # class-0 SDF: + left of the witness boundary
        phi[..., 1] = xs - (b - 0.5)                # class-1 SDF: + right of it
        t = lvl.boundary_distance_term_mlx(
            mx.array(phi.reshape(-1, 5)), oh, band, H, W)
        mx.eval(t)
        return float(t.item())

    t0, t1, t2, t4 = term_at(GT_B), term_at(GT_B + 1), term_at(GT_B + 2), term_at(GT_B + 4)
    assert t0 < t1 < t2 < t4, (t0, t1, t2, t4)
    # aligned tie-line: mean gap on the 2px band is ~1px-scale; offset by 4 is ~2*4px-scale.
    assert t4 > t0 + 4.0, (t0, t4)


def test_boundary_distance_term_differentiable_wrt_phi():
    lvl = _lvl()
    H, W = 4, 8
    ls = np.zeros((H, W), np.int64)
    ls[:, 4:] = 1
    band = mx.array(lvl.boundary_distance_band_map(ls)[None])
    oh = mx.array(np.eye(5, dtype=np.float32)[ls][None])
    phi = mx.array(np.random.default_rng(1).standard_normal((H * W, 5)).astype(np.float32))
    g = mx.grad(lambda p: lvl.boundary_distance_term_mlx(p, oh, band, H, W))(phi)
    mx.eval(g)
    gn = float(mx.sum(mx.abs(g)).item())
    assert np.isfinite(gn) and gn > 0.0, "boundary term must carry gradient to the SDF head"


# ===========================================================================
# TEST 4 — RESUME LEVER-DRIFT GUARD (pure)
# ===========================================================================
def test_resume_drift_flags_focal_and_boundary():
    lvl = _lvl()
    ckpt_cfg = {"__cfg_seg_focal_gamma": np.asarray(0.0),
                "__cfg_boundary_distance_weight": np.asarray(0.0)}
    same = SimpleNamespace(seg_focal_gamma=0.0, boundary_distance_weight=0.0)
    assert lvl._resume_lever_divergences(ckpt_cfg, same) == []
    drift = SimpleNamespace(seg_focal_gamma=2.0, boundary_distance_weight=0.0)
    div = lvl._resume_lever_divergences(ckpt_cfg, drift)
    assert any("seg_focal_gamma" in d for d in div), div
    drift2 = SimpleNamespace(seg_focal_gamma=0.0, boundary_distance_weight=0.5)
    div2 = lvl._resume_lever_divergences(ckpt_cfg, drift2)
    assert any("boundary_distance_weight" in d for d in div2), div2
    # pre-feature sidecar (keys absent) => NO spurious divergence.
    assert lvl._resume_lever_divergences({}, drift) == []


# ===========================================================================
# TEST 5 — ARGPARSE (source-verified; never-invent discipline)
# ===========================================================================
def test_argparse_flags_exist_default_off():
    src = _LVL_SRC.read_text()
    assert 'ap.add_argument("--seg-focal-gamma", type=float, default=0.0' in src
    assert 'ap.add_argument("--boundary-distance-weight", type=float, default=0.0' in src


# ===========================================================================
# TEST 6 — BYTE-IDENTITY on a real tiny batch (slow, real gt_n6 + frozen MLX SegNet)
# ===========================================================================
def _grad_leaves(tree) -> list[tuple[str, np.ndarray]]:
    return [(k, np.asarray(g)) for k, g in tree_flatten(tree)]


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(not _GT_N6.exists(), reason="gt_n6.npz cache missing")
@pytest.mark.skipif(not (_UPSTREAM / "modules.py").exists(), reason="upstream snapshot missing")
def test_focal_zero_is_byte_identical_and_gamma_changes_loss():
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    base = _base()
    lvl = _lvl()
    RH, RW = 384, 512
    z = np.load(_GT_N6, allow_pickle=False)
    P = int(z["n_pairs"])
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(_UPSTREAM, device="cpu")

    model = lvl.build_levelset_rgb_witness(
        num_pairs=P, in_feat=12, hidden_dim=8, n_hidden=1, mod_dim=4, n_classes=5,
        activation="relu", softmax_temp=1.0, wire_w0=8.0, wire_s0=4.0,
        hosc_beta=4.0, hosc_omega=1.0, chroma=True)
    rng = np.random.default_rng(0)
    model.code = mx.array(rng.standard_normal((P * 2, 4)).astype(np.float32) * 0.3)
    mx.eval(model.parameters())

    ys, xs = np.meshgrid(np.linspace(-1, 1, RH, dtype=np.float32),
                         np.linspace(-1, 1, RW, dtype=np.float32), indexing="ij")
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        proj = coords @ (rng.standard_normal((6, 2)).astype(np.float32) * 2.0).T
    assert np.isfinite(proj).all()
    cf = mx.array(np.concatenate([np.sin(2 * np.pi * proj), np.cos(2 * np.pi * proj)], axis=1))
    mx.eval(cf)

    pi = 0
    lst = z["lstars"][pi].astype(np.int64)
    oh = mx.array(np.eye(5, dtype=np.float32)[lst.ravel()].reshape(1, RH, RW, 5))
    margin = mx.array(np.asarray(z["margins"][pi], np.float32)[None])
    pose_tgt = mx.array(np.asarray(z["gt_poses"][pi], np.float32))
    c0, c1 = 2 * pi, 2 * pi + 1
    W_SEG, W_POSE, HINGE, MTGT = 100.0, 1.0, 4.0, 0.5

    def loss_and_grads(loss_fn, **kw):
        def f(m):
            return loss_fn(m, cf, c0, c1, oh, margin, pose_tgt, W_SEG, W_POSE, HINGE, MTGT, **kw)
        L, g = nn.value_and_grad(model, f)(model)
        mx.eval(L, g)
        return float(L.item()), _grad_leaves(g)

    # (a) the CURRENT-trainer path (no focal kwarg at all) vs focal_gamma=0.0: BYTE-IDENTICAL.
    fn_current = base.make_loss_fn(adapter, RH, RW, seg_loss="ce")
    fn_zero = base.make_loss_fn(adapter, RH, RW, seg_loss="ce", focal_gamma=0.0)
    L_cur, g_cur = loss_and_grads(fn_current)
    L_zero, g_zero = loss_and_grads(fn_zero)
    assert L_cur == L_zero, (L_cur, L_zero)
    for (ka, ga), (kb, gb) in zip(g_cur, g_zero, strict=True):
        assert ka == kb and np.array_equal(ga, gb), f"grad leaf {ka} diverged at focal_gamma=0"

    # (b) REFERENCE re-implementation of the pre-change CE expression (the documented base form):
    # byte-identical to make_loss_fn at focal_gamma=0 (golden equivalence to the current trainer).
    def ref_loss(m):
        f0 = base.render_through_R_mlx(m, cf, c0, RH, RW)
        f1 = base.render_through_R_mlx(m, cf, c1, RH, RW)
        seg_logits = adapter.segnet(f1)
        ce = mx.logsumexp(seg_logits, axis=-1) - mx.sum(seg_logits * oh, axis=-1)
        w = 1.0 + HINGE * mx.exp(-mx.clip(margin, 0.0, 1e9))
        seg_l = mx.mean(ce * w[None])
        pair = mx.stack([f0[0], f1[0]], axis=0)[None]
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx
        yuv = rgb_to_yuv6_mlx(pair)
        b, t, h2, w2, c6 = yuv.shape
        yuv_nhwc = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (b, h2, w2, t * c6))
        pose = adapter.posenet(yuv_nhwc)["pose"][..., : pose_tgt.shape[-1]]
        pose_l = mx.mean(mx.square(pose[0] - pose_tgt))
        return W_SEG * seg_l + W_POSE * mx.sqrt(10.0 * pose_l + 1e-8)

    L_ref, g_ref = (lambda r: (float(r[0].item()), _grad_leaves(r[1])))(
        (lambda f: nn.value_and_grad(model, f)(model))(ref_loss))
    assert L_ref == L_zero, (L_ref, L_zero)
    for (ka, ga), (kb, gb) in zip(g_ref, g_zero, strict=True):
        assert ka == kb and np.array_equal(ga, gb), f"grad leaf {ka} != pre-change reference"

    # (c) gamma>0 CHANGES the loss (the lever is live when engaged)...
    fn_g2 = base.make_loss_fn(adapter, RH, RW, seg_loss="ce", focal_gamma=2.0)
    L_g2, g_g2 = loss_and_grads(fn_g2)
    assert L_g2 != L_zero
    assert any(not np.array_equal(ga, gb) for (_, ga), (_, gb) in zip(g_g2, g_zero, strict=True))

    # (d) ...and composes bitwise with the seg_pixel_w hook (ones * fw == fw elementwise in fp).
    ones_w = mx.ones((1, RH, RW))
    L_g2_ones, g_g2_ones = loss_and_grads(fn_g2, seg_pixel_w=ones_w)
    assert L_g2_ones == L_g2
    for (ka, ga), (kb, gb) in zip(g_g2_ones, g_g2, strict=True):
        assert ka == kb and np.array_equal(ga, gb), f"grad leaf {ka} diverged under ones seg_pixel_w"
