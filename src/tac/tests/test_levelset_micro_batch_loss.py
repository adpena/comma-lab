"""Numerical-equivalence + correctness tests for the --micro-batch-pairs batched loss.

The batched loss (``tac.boundary_math.levelset_micro_batch_loss``) is the OPT-IN speed lever
for the LEVELSET witness trainer (the pointer-mover). The #1 correctness gate: the batched
grad over K pairs MUST equal the mean of the K per-pair grads within fp tolerance — a wrong
gradient corrupts training (NO-FAKE). These tests pin that with a real tiny witness + a
batch-independent mock frozen scorer (EfficientNet-B2/FastViT are batch-independent in eval
mode, which is the property the batching relies on), plus a check that the extracted per-pair
base math matches the canonical importable ``make_loss_fn`` op-for-op.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
# Run on MLX CPU: matmul/reductions are bit-identical batch-vs-single there, so the batched
# loss and the mean-of-per-pair reference agree to fp32 machine precision -> this isolates the
# MATH equivalence (exact). The GPU batched-fp-reduction noise (~1e-3, from GPU matmul kernel
# tiling on the pose path) is the acknowledged non-bit-identity of the --micro-batch-pairs opt-in
# and is validated END-TO-END on GPU by the trajectory A/B smoke, not here.
mx.set_default_device(mx.cpu)
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from tac.boundary_math.levelset_micro_batch_loss import (  # noqa: E402
    LeverConfig,
    batched_realized_loss,
    single_realized_loss,
)

# ---- import the trainer-local eikonal/nuclear helpers (module-level, cheap import ~0.2s) ----
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_TRAINER_PATH = os.path.join(_REPO, "experiments", "train_levelset_witness_realized_through_R_mlx.py")
_spec = importlib.util.spec_from_file_location("_lvl_trainer_for_test", _TRAINER_PATH)
_lvl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lvl)
_EIKONAL = _lvl._eikonal_length_mlx
_NUCLEAR = _lvl._nuclear_norm_smooth_mlx


# ===========================================================================
# Tiny witness + batch-independent mock scorer
# ===========================================================================
class _TinyWitness(nn.Module):
    def __init__(self, feat_dim, mod_dim, hidden, n_frames, n_classes=5, seed=0):
        super().__init__()
        self.n_hidden = 1
        self.hidden_dim = hidden
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.film = nn.Linear(mod_dim, self.n_hidden * 2 * hidden)
        self.hidden = [nn.Linear(hidden, hidden)]
        self.out_sdf = nn.Linear(hidden, n_classes)
        self.out_tex = nn.Linear(hidden, 3)
        rng = np.random.default_rng(seed)
        self.code = mx.array(rng.standard_normal((n_frames, mod_dim)).astype(np.float32) * 0.5)

    def _trunk(self, cf, code_idx):
        h = nn.relu(self.in_proj(cf))
        film = mx.reshape(self.film(self.code[code_idx]), (self.n_hidden, 2, self.hidden_dim))
        for li, layer in enumerate(self.hidden):
            pre = layer(h) * (1.0 + film[li, 0]) + film[li, 1]
            h = nn.relu(pre)
        return h

    def sdf(self, cf, code_idx):
        return self.out_sdf(self._trunk(cf, code_idx))  # (n_px, n_classes)

    def __call__(self, cf, code_idx):
        h = self._trunk(cf, code_idx)
        return mx.sigmoid(self.out_tex(h)) * 255.0       # (n_px, 3)


class _MockAdapter:
    """Batch-independent mock SegNet/PoseNet (linear per-pixel / per-frame) -> the batching
    invariance we rely on (segnet(batch)[k] == segnet(batch[k:k+1])[0]) holds EXACTLY."""

    def __init__(self, pose_dims=6, seed=1):
        rng = np.random.default_rng(seed)
        self.seg_w = mx.array(rng.standard_normal((3, 5)).astype(np.float32))
        self.seg_b = mx.array(rng.standard_normal((5,)).astype(np.float32))
        self.pose_w = mx.array(rng.standard_normal((12, pose_dims)).astype(np.float32) * 0.01)

    def segnet(self, f):                 # f (B,H,W,3) -> (B,H,W,5)
        return f @ self.seg_w + self.seg_b

    def posenet(self, yuv):              # yuv (B,h2,w2,12) -> {"pose": (B, pose_dims)}
        return {"pose": mx.mean(yuv, axis=(1, 2)) @ self.pose_w}


def _render_fn(model, cf, code_idx, rh, rw):
    # deterministic render (no R): witness (n_px,3) -> (1,H,W,3). The equivalence we test is the
    # batched-scorer + per-pair-reduction identity, which is R-independent.
    return mx.reshape(model(cf, int(code_idx)), (1, rh, rw, 3))


def _build(K, seed=0):
    rh, rw = 8, 12
    n_px = rh * rw
    feat_dim, mod_dim, hidden = 7, 4, 6
    n_frames = 2 * K
    rng = np.random.default_rng(seed + 7)
    model = _TinyWitness(feat_dim, mod_dim, hidden, n_frames, seed=seed)
    adapter = _MockAdapter(pose_dims=6, seed=seed + 1)
    cf = mx.array(rng.standard_normal((n_px, feat_dim)).astype(np.float32))
    cf_list = [cf for _ in range(K)]                     # shared feats (self-orient off case)
    c0_list = [2 * p + 0 for p in range(K)]
    c1_list = [2 * p + 1 for p in range(K)]
    oh_list, mg_list, pt_list = [], [], []
    for k in range(K):
        arg = rng.integers(0, 5, size=(rh, rw))
        oh = np.eye(5, dtype=np.float32)[arg].reshape(1, rh, rw, 5)
        mg = (rng.random((1, rh, rw)).astype(np.float32) * 0.5)
        oh_list.append(mx.array(oh))
        mg_list.append(mx.array(mg))
        pt_list.append(mx.array(rng.standard_normal(6).astype(np.float32) * 0.01))
    mx.eval(model.parameters(), cf, *oh_list, *mg_list, *pt_list)
    return dict(model=model, adapter=adapter, rh=rh, rw=rw, cf_list=cf_list,
                c0_list=c0_list, c1_list=c1_list, oh_list=oh_list, mg_list=mg_list, pt_list=pt_list)


def _base_lc(**over):
    lc = LeverConfig(seg_loss_default="ce", tau_use=0.3, l7_thr_use=0.42, l7_mult=4.0,
                     score_domain=True, pose_eps=1e-2,
                     eikonal_length=_EIKONAL, nuclear_norm_smooth=_NUCLEAR)
    for k, v in over.items():
        setattr(lc, k, v)
    return lc


def _max_rel_grad_err(g_batched, g_mean):
    """Global relative L2 error between two grad pytrees — the standard gradient-check metric
    (``||g_b - g_m||_2 / ||g_m||_2``). This is the honest fp32 measure: a per-leaf ratio with a
    tiny denom floor blows up to ~1e-3 on near-zero grad leaves (catastrophic cancellation) even
    when the two gradients agree to machine precision globally; the global L2 rel err is ~1e-7
    here (batched-grad == mean-of-per-pair-grad to fp32 precision)."""
    fb = dict(tree_flatten(g_batched))
    fm = dict(tree_flatten(g_mean))
    assert set(fb.keys()) == set(fm.keys()), (set(fb) ^ set(fm))
    diff = np.concatenate([(np.asarray(fb[k], np.float64) - np.asarray(fm[k], np.float64)).ravel()
                           for k in fb])
    ref = np.concatenate([np.asarray(fm[k], np.float64).ravel() for k in fm])
    return float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12))


def _batched_and_meanpairs(env, lc, *, w_seg=100.0, w_pose=1.0, hinge=4.0, mtgt=0.5,
                           seg_form=None, eik_w=0.0, len_w=0.0, render_fn_wa=None):
    """Return (loss_batched, grad_batched, mean_loss_pairs, mean_grad_pairs). ``render_fn_wa``
    routes the island levers (amplify/persistence) through a distinct (seed-excluded) forward in
    BOTH the batched and the mean-of-per-pair paths — the equivalence gate for the wa leg."""
    model = env["model"]

    def _bfn(m):
        return batched_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"], env["c0_list"], env["c1_list"],
            env["oh_list"], env["mg_list"], env["pt_list"],
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, render_fn_wa=render_fn_wa)

    lb, gb = nn.value_and_grad(model, _bfn)(model)
    mx.eval(lb, gb)

    K = len(env["c1_list"])
    from mlx.utils import tree_map
    accum = None
    lsum = 0.0

    def _sfn(m, k):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][k], env["c0_list"][k], env["c1_list"][k],
            env["oh_list"][k], env["mg_list"][k], env["pt_list"][k],
            w_seg, w_pose, hinge, mtgt, seg_form, eik_w, len_w, lc, render_fn_wa=render_fn_wa)

    for k in range(K):
        ls, gs = nn.value_and_grad(model, _sfn)(model, k)
        mx.eval(ls, gs)
        lsum += float(ls)
        accum = gs if accum is None else tree_map(lambda a, b: a + b, accum, gs)
        mx.eval(accum)
    mean_grad = tree_map(lambda g, c=float(K): g / c, accum)
    mx.eval(mean_grad)
    return float(lb), gb, lsum / K, mean_grad


# ===========================================================================
# TEST 1 — batched grad == mean of per-pair grads (the #1 gate), several regimes
# ===========================================================================
@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "l7_softplus", "margin_hinge"])
@pytest.mark.parametrize("K", [2, 4])
def test_batched_grad_equals_mean_of_per_pair_grads_base(seg_form, K):
    env = _build(K, seed=K)
    lc = _base_lc(seg_loss_default=seg_form)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form=seg_form, eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (seg_form, K, lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"{seg_form} K={K} max rel grad err {err:.2e} >= 1e-4"


# ===========================================================================
# TEST 2 — with weighted-mean levers (msal +UNIWARD, lane-edge, mfh) still matches.
#          (global sum(x*w)/sum(w) over K would BREAK this; per-pair mean must be used.)
# ===========================================================================
@pytest.mark.parametrize("K", [3, 4])
def test_batched_grad_matches_with_weighted_mean_levers(K):
    env = _build(K, seed=100 + K)
    lc = _base_lc(
        lane_w=0.7, lane_cls=1, lane_tgt=0.5, lane_gate={"on": True},
        msal_w=0.9, msal_tau=0.3, msal_tgt=0.5, msal_uni=True, msal_uni_beta=1.0,
        msal_gate={"on": True},
        mfh_w=0.4, mfh_target_mx=mx.array(np.full((1, 1, 1, 5), 0.3, np.float32)),
    )
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="tau_softplus", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"weighted-mean levers K={K} max rel grad err {err:.2e}"


# ===========================================================================
# TEST 3 — with a per-MODEL once-term (code-nuclear) still matches (added once in both paths).
# ===========================================================================
@pytest.mark.parametrize("K", [2, 4])
def test_batched_grad_matches_with_once_term(K):
    env = _build(K, seed=200 + K)
    lc = _base_lc(code_nuc_w=0.05, code_nuc_eps=1e-3, code_nuc_iters=15)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"once-term K={K} max rel grad err {err:.2e}"


# ===========================================================================
# TEST 4 — the score-domain pose sqrt is per-pair (NOT sqrt-of-mean): the batched loss must
#          MATCH mean-of-per-pair (this is why we do NOT reuse make_loss_fn_batch verbatim).
# ===========================================================================
def test_pose_sqrt_is_per_pair_not_sqrt_of_mean():
    K = 4
    env = _build(K, seed=321)
    lc = _base_lc(score_domain=True)
    # pose-heavy so the sqrt nonlinearity dominates.
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, w_seg=0.0, w_pose=10.0, seg_form="ce",
                                            eik_w=0.0, len_w=0.0)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"pose sqrt per-pair max rel grad err {err:.2e}"


# ===========================================================================
# TEST 5 — the extracted per-pair BASE math (levers/eik off) matches the canonical importable
#          make_loss_fn op-for-op (validates the seg-form + pose replication vs the real loss).
# ===========================================================================
@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "l7_softplus", "margin_hinge"])
@pytest.mark.parametrize("score_domain", [True, False])
def test_single_base_matches_canonical_make_loss_fn(seg_form, score_domain):
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn

    env = _build(1, seed=55)
    lc = _base_lc(seg_loss_default=seg_form, score_domain=score_domain)
    w_seg, w_pose, hinge, mtgt = 100.0, 1.0, 4.0, 0.5

    def _sfn(m):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            w_seg, w_pose, hinge, mtgt, seg_form, 0.0, 0.0, lc)

    ls, gs = nn.value_and_grad(env["model"], _sfn)(env["model"])
    mx.eval(ls, gs)

    canonical = make_loss_fn(env["adapter"], env["rh"], env["rw"], score_domain=score_domain,
                             pose_eps=lc.pose_eps, seg_loss=seg_form, tau_softplus_tau=lc.tau_use,
                             l7_mult=lc.l7_mult, l7_threshold=lc.l7_thr_use, render_fn=_render_fn)

    def _cfn(m):
        return canonical(m, env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
                         env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
                         w_seg, w_pose, hinge, mtgt)

    lc_val, gc = nn.value_and_grad(env["model"], _cfn)(env["model"])
    mx.eval(lc_val, gc)
    assert abs(float(ls) - float(lc_val)) / (abs(float(lc_val)) + 1e-6) < 1e-4, (seg_form, score_domain, float(ls), float(lc_val))
    err = _max_rel_grad_err(gs, gc)
    assert err < 1e-4, f"{seg_form} score_domain={score_domain} base-vs-canonical grad err {err:.2e}"


# ===========================================================================
# MB-TWIN #313 — newly-ROUTED legs (were fail-closed): focal / boundary-distance /
# eik-stab (ViscoReg + StEik) / witness-alone-island routing. Each pins the SAME #1 gate:
# batched grad == mean-of-per-pair grad within fp tol (MLX-CPU ~1e-7). The fail-closed
# remainder (msal-reachability, spike-reweight) is a trainer-level gate, tested separately below.
# ===========================================================================
_FOCAL = _lvl.focal_pixel_weight_mlx
_BDTERM = _lvl.boundary_distance_term_mlx
_BDBAND = _lvl.boundary_distance_band_map
_VISCO = _lvl._eikonal_visco_mlx
_STEIK = _lvl._eikonal_steik_mlx


def _island_weight_prov(env, val=1.0):
    """dict[pair -> (1,H,W) island weight] keyed by pi == c1//2 (same key the levers use)."""
    prov = {}
    for k in range(len(env["c1_list"])):
        pi = int(env["c1_list"][k]) // 2
        prov[pi] = mx.array(np.full((1, env["rh"], env["rw"]), val, np.float32))
    return prov


def _bd_band_prov(env, band_px=2.0):
    """dict[pair -> (1,H,W) GT-boundary band map] built from each pair's GT argmax."""
    prov = {}
    for k in range(len(env["c1_list"])):
        pi = int(env["c1_list"][k]) // 2
        arg = np.asarray(env["oh_list"][k])[0].argmax(-1).astype(np.int64)  # (H,W)
        prov[pi] = mx.array(_BDBAND(arg, band_px=band_px)[None].astype(np.float32))
    return prov


# ---- LEG 1: FOCAL ---------------------------------------------------------
@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "l7_softplus", "margin_hinge"])
@pytest.mark.parametrize("K", [2, 3])
def test_leg_focal_batched_grad_equals_mean_of_pairs(seg_form, K):
    env = _build(K, seed=400 + K)
    lc = _base_lc(seg_loss_default=seg_form, focal_gamma=2.0, focal_pixel_weight=_FOCAL)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form=seg_form, eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (seg_form, K, lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"focal {seg_form} K={K} grad err {err:.2e}"


def test_leg_focal_single_matches_canonical_make_loss_fn_with_gamma():
    """The batched twin's focal must be BIT-consistent with the canonical make_loss_fn focal path
    (same focal_pixel_weight callable, same multiplicative fold) — the one-math-one-backend gate."""
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn

    env = _build(1, seed=411)
    gamma = 2.5
    lc = _base_lc(seg_loss_default="ce", focal_gamma=gamma, focal_pixel_weight=_FOCAL)
    w_seg, w_pose, hinge, mtgt = 100.0, 1.0, 4.0, 0.5

    def _sfn(m):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
            env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
            w_seg, w_pose, hinge, mtgt, "ce", 0.0, 0.0, lc)

    ls, gs = nn.value_and_grad(env["model"], _sfn)(env["model"])
    mx.eval(ls, gs)
    canonical = make_loss_fn(env["adapter"], env["rh"], env["rw"], score_domain=True,
                             pose_eps=lc.pose_eps, seg_loss="ce", tau_softplus_tau=lc.tau_use,
                             l7_mult=lc.l7_mult, l7_threshold=lc.l7_thr_use, render_fn=_render_fn,
                             focal_gamma=gamma)

    def _cfn(m):
        return canonical(m, env["cf_list"][0], env["c0_list"][0], env["c1_list"][0],
                         env["oh_list"][0], env["mg_list"][0], env["pt_list"][0],
                         w_seg, w_pose, hinge, mtgt)

    lcv, gc = nn.value_and_grad(env["model"], _cfn)(env["model"])
    mx.eval(lcv, gc)
    assert abs(float(ls) - float(lcv)) / (abs(float(lcv)) + 1e-6) < 1e-4, (float(ls), float(lcv))
    assert _max_rel_grad_err(gs, gc) < 1e-4


def test_leg_focal_actually_changes_the_loss():
    """NO-FAKE: focal>0 must MOVE the loss vs focal=0 (else it's a silent no-op)."""
    env = _build(2, seed=412)
    l_off = _batched_and_meanpairs(env, _base_lc(seg_loss_default="ce"), seg_form="ce")[0]
    l_on = _batched_and_meanpairs(env, _base_lc(seg_loss_default="ce", focal_gamma=2.0,
                                                focal_pixel_weight=_FOCAL), seg_form="ce")[0]
    assert abs(l_on - l_off) > 1e-6, (l_on, l_off)


# ---- LEG 2: BOUNDARY-DISTANCE --------------------------------------------
@pytest.mark.parametrize("K", [2, 4])
def test_leg_boundary_distance_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=500 + K)
    lc = _base_lc(bd_w=0.3, bd_band_prov=_bd_band_prov(env), boundary_distance_term=_BDTERM)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


def test_leg_boundary_distance_actually_changes_the_loss():
    env = _build(2, seed=511)
    l_off = _batched_and_meanpairs(env, _base_lc(), seg_form="ce")[0]
    l_on = _batched_and_meanpairs(env, _base_lc(bd_w=0.3, bd_band_prov=_bd_band_prov(env),
                                                boundary_distance_term=_BDTERM), seg_form="ce")[0]
    assert abs(l_on - l_off) > 1e-6, (l_on, l_off)


# ---- LEG 3: EIK-STAB (ViscoReg + StEik) ----------------------------------
@pytest.mark.parametrize("K", [2, 3])
def test_leg_eik_viscosity_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=600 + K)
    lc = _base_lc(eik_stab={"visco_eps": 0.05, "steik_w": 0.0}, eikonal_visco=_VISCO,
                  eikonal_steik=_STEIK)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


@pytest.mark.parametrize("K", [2, 3])
def test_leg_eik_steik_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=650 + K)
    lc = _base_lc(eik_stab={"visco_eps": 0.0, "steik_w": 0.02}, eikonal_visco=_VISCO,
                  eikonal_steik=_STEIK)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


def test_leg_eik_viscosity_replaces_residual_not_double_counts():
    """ViscoReg REPLACES the eikonal residual (not additive): visco-on differs from visco-off. The
    eikonal term is ISOLATED (w_seg=w_pose=len_w=0, eik_w=1) so its magnitude is not swamped by the
    ~1e4 seg/pose loss (fp32 resolution)."""
    env = _build(2, seed=611)
    l_legacy = _batched_and_meanpairs(env, _base_lc(), seg_form="ce",
                                      w_seg=0.0, w_pose=0.0, eik_w=1.0, len_w=0.0)[0]
    l_visco = _batched_and_meanpairs(
        env, _base_lc(eik_stab={"visco_eps": 0.05, "steik_w": 0.0}, eikonal_visco=_VISCO,
                      eikonal_steik=_STEIK), seg_form="ce",
        w_seg=0.0, w_pose=0.0, eik_w=1.0, len_w=0.0)[0]
    assert abs(l_visco - l_legacy) > 1e-6, (l_visco, l_legacy)
    # eik_w=0 kills BOTH the legacy residual and the visco replacement -> visco branch rides eik_w
    # (replaces the SAME term) -> with only the length term left, visco-on == visco-off.
    l_visco_noeik = _batched_and_meanpairs(
        env, _base_lc(eik_stab={"visco_eps": 0.05, "steik_w": 0.0}, eikonal_visco=_VISCO,
                      eikonal_steik=_STEIK), seg_form="ce",
        w_seg=0.0, w_pose=0.0, eik_w=0.0, len_w=1.0)[0]
    l_legacy_noeik = _batched_and_meanpairs(env, _base_lc(), seg_form="ce",
                                            w_seg=0.0, w_pose=0.0, eik_w=0.0, len_w=1.0)[0]
    assert abs(l_visco_noeik - l_legacy_noeik) / (abs(l_legacy_noeik) + 1e-9) < 1e-5


# ---- LEG 4: WITNESS-ALONE ISLAND ROUTING ---------------------------------
def _render_fn_wa(model, cf, code_idx, rh, rw):
    # DISTINCT (seed-excluded analog) render: a genuine perturbation so sl_wa != sl and the
    # routing is OBSERVABLE. Both batched + single use this SAME fn -> equivalence must hold.
    return mx.reshape(model(cf, int(code_idx)) * 0.85 + 6.0, (1, rh, rw, 3))


@pytest.mark.parametrize("K", [2, 3])
def test_leg_wa_island_amplify_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=700 + K)
    lc = _base_lc(wa_island=True, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
                  island_weight_mx=_island_weight_prov(env))
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3,
                                            render_fn_wa=_render_fn_wa)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


@pytest.mark.parametrize("K", [2, 3])
def test_leg_wa_island_persist_batched_grad_equals_mean_of_pairs(K):
    env = _build(K, seed=750 + K)
    lc = _base_lc(wa_island=True, persist_gate={"w": 0.4}, persist_classes=(1, 3),
                  persist_cldice_iters=3, persist_recall_w=1.0)
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3,
                                            render_fn_wa=_render_fn_wa)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (K, lb, lm)
    assert _max_rel_grad_err(gb, gm) < 1e-4


def test_leg_wa_island_routing_is_real_not_aliased():
    """NO-FAKE: with wa_island ON and a DISTINCT render_fn_wa, the amplify term must read the
    seed-excluded forward -> loss differs from the aliased (render_fn_wa=None) case. If the two
    were equal, the wa routing would be a silent no-op."""
    env = _build(2, seed=711)
    lc = _base_lc(wa_island=True, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
                  island_weight_mx=_island_weight_prov(env))
    l_wa = _batched_and_meanpairs(env, lc, seg_form="ce", render_fn_wa=_render_fn_wa)[0]
    l_aliased = _batched_and_meanpairs(env, lc, seg_form="ce", render_fn_wa=None)[0]
    assert abs(l_wa - l_aliased) > 1e-6, (l_wa, l_aliased)


def test_leg_wa_off_is_byte_identical_regardless_of_render_fn_wa():
    """When wa_island is False, passing render_fn_wa must NOT change anything (no 2nd forward):
    the island levers alias the composed forward -> byte-identical to render_fn_wa=None."""
    env = _build(2, seed=712)
    lc = _base_lc(wa_island=False, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
                  island_weight_mx=_island_weight_prov(env))
    l_passed = _batched_and_meanpairs(env, lc, seg_form="ce", render_fn_wa=_render_fn_wa)[0]
    l_none = _batched_and_meanpairs(env, lc, seg_form="ce", render_fn_wa=None)[0]
    assert abs(l_passed - l_none) < 1e-9, (l_passed, l_none)


def test_leg_wa_no_island_lever_no_second_forward():
    """wa_island True but NO island lever engaged -> _wa_route_active False -> render_fn_wa ignored
    (base + surgical levers never read a wa forward) -> byte-identical to wa off."""
    from tac.boundary_math.levelset_micro_batch_loss import _wa_route_active
    lc = _base_lc(wa_island=True)  # no amplify / persist
    assert _wa_route_active(lc, _render_fn_wa) is False


# ---- COMBINED: all four legs stacked still equivalent ---------------------
def test_all_four_new_legs_stacked_batched_grad_equals_mean_of_pairs():
    K = 3
    env = _build(K, seed=800)
    lc = _base_lc(
        seg_loss_default="ce", focal_gamma=2.0, focal_pixel_weight=_FOCAL,
        bd_w=0.3, bd_band_prov=_bd_band_prov(env), boundary_distance_term=_BDTERM,
        eik_stab={"visco_eps": 0.05, "steik_w": 0.02}, eikonal_visco=_VISCO, eikonal_steik=_STEIK,
        wa_island=True, amplify_w=0.6, amplify_mtgt=1.0, amplify_form="hinge",
        island_weight_mx=_island_weight_prov(env),
        persist_gate={"w": 0.3}, persist_classes=(1, 3), persist_cldice_iters=3,
    )
    lb, gb, lm, gm = _batched_and_meanpairs(env, lc, seg_form="ce", eik_w=1e-2, len_w=1e-3,
                                            render_fn_wa=_render_fn_wa)
    assert abs(lb - lm) / (abs(lm) + 1e-6) < 1e-4, (lb, lm)
    err = _max_rel_grad_err(gb, gm)
    assert err < 1e-4, f"all-four-legs grad err {err:.2e}"
