# SPDX-License-Identifier: MIT
"""FISHER-ACTUATION levers (build-wave arm A, SPEC_v10 §13.1 row 2 + §13.4 surfaces 1-2) —
unit + $0 byte-identity tests.

Covers:
  * --fisher-density-weight / --fisher-density-source (exact-law sech² Fisher-trace per-pixel
    seg reweight; base helper ``fisher_density_pixel_weight_mlx``)
  * --head-natural-grad / --head-natural-grad-eps (forward-identity / backward-g⁺ logit
    natural-gradient preconditioner; base helper ``make_seg_logits_natural_grad_mlx``)
  * make_loss_fn OFF-path byte-identity (loss AND grads bitwise equal to the no-kwarg path)
  * resume-drift registration + argparse existence (never-invent-flags)
  * DSL Lever factories (flag names match the trainer argparse; invalid params fail closed)

Pointer UNMOVED — these are $0 build gates, not score claims. [macOS-MLX research-signal]
apparatus tests; no score/promotion semantics.

Run: .venv/bin/pytest experiments/test_fisher_actuation_levers.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src"), str(REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402


def _base():
    import train_witness_realized_through_R_mlx as base
    return base


def _lvl():
    import train_levelset_witness_realized_through_R_mlx as lvl
    return lvl


def _rand_logits(seed=0, h=6, w=8, k=5, scale=2.0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.normal(0, scale, (1, h, w, k)).astype(np.float32))


def _rand_oh(seed=1, h=6, w=8, k=5):
    rng = np.random.default_rng(seed)
    ls = rng.integers(0, k, (h, w))
    return mx.array(np.eye(k, dtype=np.float32)[ls][None])


# ---------------------------------------------------------------------------
# 1. Fisher-density weight math
# ---------------------------------------------------------------------------
def test_fisher_weight_mean_one_at_full_blend():
    base = _base()
    logits, oh = _rand_logits(), _rand_oh()
    margin = mx.zeros(logits.shape[:3])
    w = base.fisher_density_pixel_weight_mlx(logits, oh, margin, 1.0, "model")
    mx.eval(w)
    assert float(mx.mean(w)) == pytest.approx(1.0, abs=1e-5)


def test_fisher_weight_blend_formula():
    base = _base()
    logits, oh = _rand_logits(), _rand_oh()
    margin = mx.zeros(logits.shape[:3])
    w_full = base.fisher_density_pixel_weight_mlx(logits, oh, margin, 1.0, "model")
    w_half = base.fisher_density_pixel_weight_mlx(logits, oh, margin, 0.5, "model")
    mx.eval(w_full, w_half)
    np.testing.assert_allclose(
        np.asarray(w_half), 0.5 + 0.5 * np.asarray(w_full), rtol=1e-6, atol=1e-6)


def test_fisher_weight_model_source_peaks_at_zero_live_margin():
    base = _base()
    # Construct logits with known live margins: pixel A margin 0 (tie), pixel B margin 6.
    z = np.zeros((1, 1, 2, 5), np.float32)
    z[0, 0, 0, 0] = 3.0
    z[0, 0, 0, 1] = 3.0      # tie -> live margin 0 -> tr g max
    z[0, 0, 1, 0] = 6.0      # GT logit 6, others 0 -> margin 6 -> tr g small
    oh = np.zeros((1, 1, 2, 5), np.float32)
    oh[..., 0] = 1.0         # GT class 0 both pixels
    w = base.fisher_density_pixel_weight_mlx(mx.array(z), mx.array(oh),
                                             mx.zeros((1, 1, 2)), 1.0, "model")
    mx.eval(w)
    wn = np.asarray(w)[0, 0]
    assert wn[0] > wn[1]     # boundary pixel out-weighs the confident pixel


def test_fisher_weight_model_source_symmetric_in_margin_sign():
    base = _base()
    # margin +2 (correct) and margin -2 (confidently wrong) must weigh EQUALLY (metric is even).
    z = np.zeros((1, 1, 2, 5), np.float32)
    z[0, 0, 0, 0] = 2.0      # GT ahead by 2
    z[0, 0, 1, 1] = 2.0      # rival ahead by 2 (GT margin -2)
    oh = np.zeros((1, 1, 2, 5), np.float32)
    oh[..., 0] = 1.0
    w = base.fisher_density_pixel_weight_mlx(mx.array(z), mx.array(oh),
                                             mx.zeros((1, 1, 2)), 1.0, "model")
    mx.eval(w)
    wn = np.asarray(w)[0, 0]
    assert wn[0] == pytest.approx(wn[1], rel=1e-6)


def test_fisher_weight_gt_source_reads_margin_not_logits():
    base = _base()
    logits, oh = _rand_logits(2), _rand_oh(3)
    m = np.zeros((1, 6, 8), np.float32)
    m[0, 0, 0] = 0.0
    m[0, :, :] += 5.0
    m[0, 0, 0] = 0.0
    w = base.fisher_density_pixel_weight_mlx(logits, oh, mx.array(m), 1.0, "gt")
    w2 = base.fisher_density_pixel_weight_mlx(_rand_logits(9), oh, mx.array(m), 1.0, "gt")
    mx.eval(w, w2)
    # gt source: logits-independent...
    np.testing.assert_array_equal(np.asarray(w), np.asarray(w2))
    # ...and the m=0 pixel carries the max weight.
    wn = np.asarray(w)[0]
    assert wn[0, 0] == wn.max()


def test_fisher_weight_matches_numpy_twin_law():
    base = _base()
    from tac.witness_control.fisher_annulus import fisher_trace_from_margin
    m_np = np.linspace(-4, 4, 48).reshape(1, 6, 8).astype(np.float32)
    logits, oh = _rand_logits(), _rand_oh()
    w = base.fisher_density_pixel_weight_mlx(logits, oh, mx.array(m_np), 1.0, "gt")
    mx.eval(w)
    tr = fisher_trace_from_margin(m_np)
    ref = tr / (tr.mean() + 1e-8)
    np.testing.assert_allclose(np.asarray(w), ref, rtol=1e-4, atol=1e-5)


def test_fisher_weight_stop_grad():
    base = _base()
    oh = _rand_oh()
    m = mx.zeros((1, 6, 8))

    def f(z):
        w = base.fisher_density_pixel_weight_mlx(z, oh, m, 1.0, "model")
        return mx.sum(w)

    g = mx.grad(f)(_rand_logits())
    mx.eval(g)
    assert float(mx.max(mx.abs(g))) == 0.0


def test_fisher_weight_bad_source_raises():
    base = _base()
    with pytest.raises(ValueError, match="model.*gt|'model' or 'gt'"):
        base.fisher_density_pixel_weight_mlx(
            _rand_logits(), _rand_oh(), mx.zeros((1, 6, 8)), 1.0, "banana")


# ---------------------------------------------------------------------------
# 2. Natural-gradient transform math
# ---------------------------------------------------------------------------
def test_ng_forward_is_identity():
    base = _base()
    ng = base.make_seg_logits_natural_grad_mlx(1e-3)
    z = _rand_logits(4)
    out = ng(z)
    mx.eval(out)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(z))


def test_ng_pseudo_inverse_closed_form_inverts_fisher():
    # For tiny eps: g (g+ v) == v for any per-pixel cotangent with sum 0.
    base = _base()
    eps = 1e-12
    ng = base.make_seg_logits_natural_grad_mlx(eps)
    rng = np.random.default_rng(7)
    z = rng.normal(0, 1.0, (1, 1, 1, 5)).astype(np.float32)
    v = rng.normal(0, 1.0, 5).astype(np.float64)
    v -= v.mean()  # gauge-projected cotangent

    def f(zz):
        return mx.sum(ng(zz) * mx.array(v[None, None, None, :].astype(np.float32)))

    u = np.asarray(mx.grad(f)(mx.array(z)), np.float64)[0, 0, 0]  # u = g+ v
    p = np.exp(z[0, 0, 0].astype(np.float64))
    p /= p.sum()
    g = np.diag(p) - np.outer(p, p)
    np.testing.assert_allclose(g @ u, v, rtol=2e-4, atol=2e-4)
    # min-norm branch: u itself sums to ~0
    assert abs(u.sum()) < 1e-3


def test_ng_ce_gradient_matches_closed_form():
    # d/dz CE(ng(z), y) == g+ (p - y), the natural gradient of CE.
    base = _base()
    eps = 1e-12
    ng = base.make_seg_logits_natural_grad_mlx(eps)
    rng = np.random.default_rng(11)
    z = rng.normal(0, 1.5, (1, 2, 2, 5)).astype(np.float32)
    y = np.eye(5, dtype=np.float32)[rng.integers(0, 5, (2, 2))][None]

    def ce(zz):
        zt = ng(zz)
        return mx.sum(mx.logsumexp(zt, axis=-1) - mx.sum(zt * mx.array(y), axis=-1))

    got = np.asarray(mx.grad(ce)(mx.array(z)), np.float64)
    zf = z.astype(np.float64)
    p = np.exp(zf - zf.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    v = p - y.astype(np.float64)  # sums to 0 per pixel already
    u = v / (p + eps)
    u -= u.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, u, rtol=5e-3, atol=5e-3)


def test_ng_damping_bounds_low_probability_blowup():
    base = _base()
    ng = base.make_seg_logits_natural_grad_mlx(1e-2)
    z = np.zeros((1, 1, 1, 5), np.float32)
    z[..., 0] = 30.0  # p ~= [1, ~0, ~0, ...]: undamped 1/p would be ~e^30

    def f(zz):
        zt = ng(zz)
        return zt[0, 0, 0, 1] - zt[0, 0, 0, 2]  # cotangent [0,1,-1,0,0], sum 0

    g = np.asarray(mx.grad(f)(mx.array(z)), np.float64)
    assert np.all(np.isfinite(g))
    assert np.max(np.abs(g)) < 2.0 / 1e-2  # bounded by the eps floor


# ---------------------------------------------------------------------------
# 3. make_loss_fn byte-identity + engagement
# ---------------------------------------------------------------------------
class _TinySeg:
    """Deterministic stand-in scorer for LOSS-GRAPH identity tests (exercises the real
    make_loss_fn code path; NOT a SegNet claim — the lever's score effect is RUN-GATED)."""

    def __init__(self):
        rng = np.random.default_rng(42)
        self._w = mx.array(rng.normal(0, 1.0, (3, 5)).astype(np.float32))

    def segnet(self, f1):
        return f1 @ self._w  # (1,H,W,3) @ (3,5) -> (1,H,W,5)


class _TinyRenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = mx.zeros((3,))


def _mini_loss(base, **kw):
    h, w = 6, 8
    rng = np.random.default_rng(5)
    frame = mx.array(rng.uniform(0, 255, (1, h, w, 3)).astype(np.float32))

    def render_fn(model, coord_feats, code_idx, rh, rw):
        return frame + model.bias

    adapter = _TinySeg()
    loss_fn = base.make_loss_fn(adapter, h, w, score_domain=True, seg_loss="ce",
                                render_fn=render_fn, **kw)
    model = _TinyRenderModel()
    mx.eval(model.parameters())
    oh = _rand_oh(21, h, w)
    margin = mx.array(rng.uniform(0, 4, (1, h, w)).astype(np.float32))
    pose_tgt = mx.zeros((6,))

    def f(m):
        return loss_fn(m, None, 0, 1, oh, margin, pose_tgt, 100.0, 1.0, 4.0, 0.5,
                       compute_pose=False)

    val, grads = nn.value_and_grad(model, f)(model)
    mx.eval(val, grads)
    from mlx.utils import tree_flatten
    return float(val), {k: np.asarray(v).copy() for k, v in tree_flatten(grads)}


def test_off_path_is_byte_identical_to_no_kwarg_path():
    base = _base()
    v0, g0 = _mini_loss(base)
    v1, g1 = _mini_loss(base, fisher_density_weight=0.0, fisher_density_source="model",
                        head_natural_grad=False, head_natural_grad_eps=1e-3)
    assert v0 == v1
    for k in g0:
        np.testing.assert_array_equal(g0[k], g1[k])


def test_fisher_weight_on_changes_grads_not_budget_sign():
    base = _base()
    v0, g0 = _mini_loss(base)
    v1, g1 = _mini_loss(base, fisher_density_weight=1.0, fisher_density_source="model")
    changed = any(not np.array_equal(g0[k], g1[k]) for k in g0)
    assert changed
    assert np.isfinite(v1)


def test_fisher_weight_gt_source_engages():
    base = _base()
    v0, g0 = _mini_loss(base)
    v1, g1 = _mini_loss(base, fisher_density_weight=1.0, fisher_density_source="gt")
    assert any(not np.array_equal(g0[k], g1[k]) for k in g0)


def test_head_ng_preserves_loss_value_changes_grads():
    base = _base()
    v0, g0 = _mini_loss(base)
    v1, g1 = _mini_loss(base, head_natural_grad=True, head_natural_grad_eps=1e-3)
    assert v0 == pytest.approx(v1, rel=1e-7)  # forward identity
    assert any(not np.array_equal(g0[k], g1[k]) for k in g0)  # direction changed


def test_make_loss_fn_validation_fail_closed():
    base = _base()
    with pytest.raises(ValueError, match="fisher_density_weight"):
        _mini_loss(base, fisher_density_weight=1.5)
    with pytest.raises(ValueError, match="fisher_density_source"):
        _mini_loss(base, fisher_density_weight=0.5, fisher_density_source="nope")
    with pytest.raises(ValueError, match="head_natural_grad_eps"):
        _mini_loss(base, head_natural_grad=True, head_natural_grad_eps=0.0)


# ---------------------------------------------------------------------------
# 4. Trainer wiring: argparse + resume drift + micro-batch fail-close
# ---------------------------------------------------------------------------
def test_argparse_flags_exist_default_off():
    # never-invent-flags: the exact add_argument literals exist with default-OFF values.
    lvl = _lvl()
    src = Path(lvl.__file__).read_text()
    assert '"--fisher-density-weight", type=float, default=0.0' in src.replace("\n", "").replace(
        "    ", "").replace('",type', '", type') or '"--fisher-density-weight"' in src
    for flag in ('"--fisher-density-weight"', '"--fisher-density-source"',
                 '"--head-natural-grad"', '"--head-natural-grad-eps"'):
        assert flag in src, f"missing argparse flag literal {flag}"
    # defaults: the weight default is 0.0 and NG default False (default-OFF contract).
    i = src.index('"--fisher-density-weight"')
    assert "default=0.0" in src[i:i + 200]
    j = src.index('"--head-natural-grad"')
    assert "default=False" in src[j:j + 200]


def test_resume_drift_rows_registered():
    lvl = _lvl()
    sidecar = {
        "__cfg_fisher_density_weight": np.asarray(1.0),
        "__cfg_fisher_density_source_gt": np.asarray(1),
        "__cfg_head_natural_grad": np.asarray(1),
        "__cfg_head_natural_grad_eps": np.asarray(1e-3),
    }
    args = SimpleNamespace(fisher_density_weight=0.0, fisher_density_source="model",
                           head_natural_grad=False, head_natural_grad_eps=1e-3)
    div = lvl._resume_lever_divergences(sidecar, args)
    joined = " ".join(div)
    assert "fisher_density_weight" in joined
    assert "fisher_density_source_gt" in joined
    assert "head_natural_grad" in joined


def test_resume_drift_silent_when_matching():
    lvl = _lvl()
    sidecar = {
        "__cfg_fisher_density_weight": np.asarray(0.7),
        "__cfg_fisher_density_source_gt": np.asarray(0),
        "__cfg_head_natural_grad": np.asarray(0),
        "__cfg_head_natural_grad_eps": np.asarray(1e-3),
    }
    args = SimpleNamespace(fisher_density_weight=0.7, fisher_density_source="model",
                           head_natural_grad=False, head_natural_grad_eps=1e-3)
    div = lvl._resume_lever_divergences(sidecar, args)
    assert not any("fisher" in d or "natural_grad" in d for d in div)


def test_micro_batch_fail_closed_guard_present():
    lvl = _lvl()
    src = Path(lvl.__file__).read_text()
    assert "--fisher-density-weight is not supported with --micro-batch-pairs>1" in src
    assert "--head-natural-grad is not supported with --micro-batch-pairs>1" in src


# ---------------------------------------------------------------------------
# 5. DSL Lever factories
# ---------------------------------------------------------------------------
def test_dsl_fisher_density_lever_flags_match_trainer():
    from tac.witness_dsl.curriculum_dsl import FisherDensityWeight
    lv = FisherDensityWeight(blend=0.8, source="gt")
    assert lv.overrides["--fisher-density-weight"] == 0.8
    assert lv.overrides["--fisher-density-source"] == "gt"
    lvl = _lvl()
    src = Path(lvl.__file__).read_text()
    for flag in lv.overrides:
        assert f'"{flag}"' in src  # never-invent-flags


def test_dsl_head_ng_lever_flags_match_trainer():
    from tac.witness_dsl.curriculum_dsl import HeadNaturalGradient
    lv = HeadNaturalGradient(eps=1e-2)
    assert lv.overrides["--head-natural-grad"] is True
    assert lv.overrides["--head-natural-grad-eps"] == 1e-2
    lvl = _lvl()
    src = Path(lvl.__file__).read_text()
    for flag in lv.overrides:
        assert f'"{flag}"' in src


def test_dsl_factories_fail_closed_on_bad_params():
    from tac.witness_dsl.curriculum_dsl import FisherDensityWeight, HeadNaturalGradient
    with pytest.raises(ValueError):
        FisherDensityWeight(blend=0.0)
    with pytest.raises(ValueError):
        FisherDensityWeight(blend=1.2)
    with pytest.raises(ValueError):
        FisherDensityWeight(source="live")
    with pytest.raises(ValueError):
        HeadNaturalGradient(eps=0.0)
